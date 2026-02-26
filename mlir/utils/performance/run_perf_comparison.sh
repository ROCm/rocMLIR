#!/bin/bash
# End-to-end A/B performance comparison between two rocMLIR branches.
#
# Orchestrates: build -> tune -> perf -> ISA extraction -> Excel report
#
# Usage:
#   ./run_perf_comparison.sh                          # auto-detect everything
#   GPU_OVERRIDE="4 5 6 7" ./run_perf_comparison.sh   # specific GPUs
#   FEATURE_BRANCH=myBranch ./run_perf_comparison.sh   # different branches
#
# Environment variable overrides:
#   FEATURE_BRANCH  - feature branch name (default: swapOperands2)
#   BASE_BRANCH     - base branch name (default: develop)
#   OUTPUT_DIR      - output directory (default: <repo>/perf_comparison_results)
#   TUNING_SPACE    - tuning space (default: greedy)
#   GPU_OVERRIDE    - space-separated GPU IDs (default: auto-detect all)
#   CPU_OVERRIDE    - number of CPUs (default: auto-detect)
#   SKIP_ISA        - set to 1 to skip ISA extraction

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

FEATURE_BRANCH="${FEATURE_BRANCH:-swapOperands2}"
BASE_BRANCH="${BASE_BRANCH:-develop}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/perf_comparison_results}"
TUNING_SPACE="${TUNING_SPACE:-greedy}"
SKIP_ISA="${SKIP_ISA:-0}"

CMAKE_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DLLVM_CCACHE_BUILD=ON
    -DLLD_BUILD_TOOLS=ON
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DMLIR_ENABLE_EXECUTION_ENGINE=ON
    -DMLIR_ENABLE_ROCM_RUNNER=ON
    -DROCK_E2E_TEST_ENABLED=1
    -DROCMLIR_DRIVER_PR_E2E_TEST_ENABLED=ON
    -DROCMLIR_DRIVER_E2E_TEST_ENABLED=1
    -DROCMLIR_DRIVER_TEST_GPU_VALIDATION=1
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
    "-DLLVM_LIT_ARGS=-j 8"
)

NINJA_TARGETS=(
    rocmlir-driver rocmlir-gen mlir-runner ci-performance-scripts conv-validation-wrappers rocmlir-tuning-driver
    mlir_runner_utils mlir_rocm_runtime mlir_c_runner_utils mlir_async_runtime
)

GEMM_CONFIGS="$REPO_DIR/mlir/utils/performance/configs/tier1-gemm-configs"
CONV_CONFIGS="$REPO_DIR/mlir/utils/performance/configs/tier1-conv-configs"

log() { echo "=== $1 ==="; }
die() { echo "ERROR: $1" >&2; exit 1; }

# --- Install Python dependencies ---
python3 -c "import openpyxl" 2>/dev/null || {
    log "Installing required Python package: openpyxl"
    pip install openpyxl --quiet
}

# --- Stage 0: Hardware Detection ---
log "Stage 0: Detecting hardware"

if command -v rocm-smi &>/dev/null; then
    NUM_GPUS=$(rocm-smi --showproductname --json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
gpus = [k for k in data if k.startswith('card')]
print(len(gpus))
" 2>/dev/null || echo "1")
else
    NUM_GPUS=1
fi

# Extract gfx chip name (e.g. gfx950) from rocminfo
GFX_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-z]+' | head -1 || echo "unknown")

ALL_GPU_IDS=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ' ' | sed 's/ $//')
NUM_CPUS=$(nproc 2>/dev/null || echo "1")

GPU_ARGS="${GPU_OVERRIDE:-$ALL_GPU_IDS}"
CPU_ARGS="${CPU_OVERRIDE:-$NUM_CPUS}"

echo "  GPUs: $NUM_GPUS (IDs: $GPU_ARGS) Arch: $GFX_ARCH"
echo "  CPUs: $CPU_ARGS"
echo "  Feature branch: $FEATURE_BRANCH"
echo "  Base branch: $BASE_BRANCH"
echo "  Tuning space: $TUNING_SPACE"
echo "  Output dir: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Save original branch to restore on exit
ORIGINAL_BRANCH=$(git -C "$REPO_DIR" branch --show-current 2>/dev/null || echo "")
STASH_CREATED=0

cleanup() {
    if [ -n "$ORIGINAL_BRANCH" ]; then
        echo ""
        log "Restoring original branch: $ORIGINAL_BRANCH"
        git -C "$REPO_DIR" checkout HEAD -- \
            mlir/utils/performance/run_perf_comparison.sh \
            mlir/utils/performance/perfRunner.py \
            mlir/utils/performance/tuningRunner.py 2>/dev/null || true
        git -C "$REPO_DIR" checkout "$ORIGINAL_BRANCH" 2>/dev/null || true
        if [ "$STASH_CREATED" = "1" ]; then
            git -C "$REPO_DIR" stash pop 2>/dev/null || true
            STASH_CREATED=0
        fi
    fi
}
trap cleanup EXIT

# --- Helper: build a branch ---
build_branch() {
    local branch="$1"
    local build_dir="$2"

    if [ ! -f "$build_dir/bin/rocmlir-gen" ]; then
        echo "  Configuring cmake in $build_dir..."
        mkdir -p "$build_dir"
        cmake -S "$REPO_DIR" -B "$build_dir" "${CMAKE_FLAGS[@]}"
    fi

    echo "  Building targets..."
    ninja -C "$build_dir" "${NINJA_TARGETS[@]}"
}

# --- Helper: run tuning for one operation ---
run_tuning() {
    local build_dir="$1"
    local operation="$2"
    local configs_file="$3"
    local output_tsv="$4"

    if [ -f "$output_tsv" ]; then
        echo "  Tuning output already exists: $output_tsv (skipping)"
        return 0
    fi

    # Remove stale state files from previous (possibly killed) runs
    rm -f "${output_tsv}.state" "${output_tsv}.state.tmp"

    echo "  Running $operation tuning -> $output_tsv"
    python3 "$build_dir/bin/tuningRunner.py" \
        --operation "$operation" \
        -o "$output_tsv" \
        -c "$configs_file" \
        --tuning-space "$TUNING_SPACE" \
        --gpus $GPU_ARGS \
        --num-cpus "$CPU_ARGS" \
        --mlir-build-dir "$build_dir"
}

# --- Helper: run perf for one operation ---
run_perf() {
    local build_dir="$1"
    local operation="$2"
    local configs_file="$3"
    local tuning_tsv="$4"
    local output_csv="$5"

    if [ -f "$output_csv" ]; then
        echo "  Perf output already exists: $output_csv (skipping)"
        return 0
    fi

    echo "  Running $operation perf -> $output_csv"
    python3 "$build_dir/bin/perfRunner.py" \
        --operation "$operation" \
        -t "$tuning_tsv" \
        -o "$output_csv" \
        -c "$configs_file" \
        --batch_mlir \
        --mlir-build-dir "$build_dir"
}

HOSTNAME_STR=$(hostname 2>/dev/null || echo "unknown")

# --- Stage 1: Feature branch ---
log "Stage 1: Checking out feature branch ($FEATURE_BRANCH)"
cd "$REPO_DIR"
CURRENT=$(git branch --show-current 2>/dev/null || echo "")
if [ "$CURRENT" != "$FEATURE_BRANCH" ]; then
    git checkout "$FEATURE_BRANCH"
fi
FEATURE_COMMIT=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "  Feature commit: $FEATURE_COMMIT"

BUILD_FEATURE="$REPO_DIR/build"
log "Stage 2: Building feature branch ($FEATURE_BRANCH)"
build_branch "$FEATURE_BRANCH" "$BUILD_FEATURE"

# --- Stage 3: Tune feature branch ---
log "Stage 3: Tuning on $FEATURE_BRANCH"
run_tuning "$BUILD_FEATURE" "gemm" "$GEMM_CONFIGS" "$OUTPUT_DIR/${FEATURE_BRANCH}_gemm.tsv"
run_tuning "$BUILD_FEATURE" "conv" "$CONV_CONFIGS" "$OUTPUT_DIR/${FEATURE_BRANCH}_conv.tsv"

# --- Stage 4: Perf feature branch ---
log "Stage 4: Performance measurement on $FEATURE_BRANCH"
run_perf "$BUILD_FEATURE" "gemm" "$GEMM_CONFIGS" \
    "$OUTPUT_DIR/${FEATURE_BRANCH}_gemm.tsv" "$OUTPUT_DIR/${FEATURE_BRANCH}_gemm_perf.csv"
run_perf "$BUILD_FEATURE" "conv" "$CONV_CONFIGS" \
    "$OUTPUT_DIR/${FEATURE_BRANCH}_conv.tsv" "$OUTPUT_DIR/${FEATURE_BRANCH}_conv_perf.csv"

# --- Stage 5: Switch to base branch and build ---
log "Stage 5: Checking out base branch ($BASE_BRANCH)"
cd "$REPO_DIR"
# Save working-tree scripts (which may have uncommitted fixes) before stash
cp mlir/utils/performance/perfRunner.py /tmp/_perfRunner_save.py
cp mlir/utils/performance/tuningRunner.py /tmp/_tuningRunner_save.py
git stash push -m "run_perf_comparison auto-stash"
STASH_CREATED=1
git checkout "$BASE_BRANCH"
# Restore saved scripts so bash can continue and perfRunner.py has the fixes
cp /tmp/_perfRunner_save.py mlir/utils/performance/perfRunner.py
cp /tmp/_tuningRunner_save.py mlir/utils/performance/tuningRunner.py
BASE_COMMIT=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "  Base commit: $BASE_COMMIT"

BUILD_BASE="$REPO_DIR/build"
log "Stage 6: Building base branch ($BASE_BRANCH)"
build_branch "$BASE_BRANCH" "$BUILD_BASE"
# Copy the saved scripts into the build dir after build, since ninja
# overwrites them with the base branch's versions during build_branch.
cp /tmp/_perfRunner_save.py "$REPO_DIR/build/bin/perfRunner.py"
cp /tmp/_tuningRunner_save.py "$REPO_DIR/build/bin/tuningRunner.py"

# --- Stage 7: Tune base branch ---
log "Stage 7: Tuning on $BASE_BRANCH"
run_tuning "$BUILD_BASE" "gemm" "$GEMM_CONFIGS" "$OUTPUT_DIR/${BASE_BRANCH}_gemm.tsv"
run_tuning "$BUILD_BASE" "conv" "$CONV_CONFIGS" "$OUTPUT_DIR/${BASE_BRANCH}_conv.tsv"

# --- Stage 8: Perf base branch ---
log "Stage 8: Performance measurement on $BASE_BRANCH"
run_perf "$BUILD_BASE" "gemm" "$GEMM_CONFIGS" \
    "$OUTPUT_DIR/${BASE_BRANCH}_gemm.tsv" "$OUTPUT_DIR/${BASE_BRANCH}_gemm_perf.csv"
run_perf "$BUILD_BASE" "conv" "$CONV_CONFIGS" \
    "$OUTPUT_DIR/${BASE_BRANCH}_conv.tsv" "$OUTPUT_DIR/${BASE_BRANCH}_conv_perf.csv"

# --- Stage 9: Return to feature branch ---
log "Stage 9: Returning to feature branch ($FEATURE_BRANCH)"
cd "$REPO_DIR"
git checkout HEAD -- mlir/utils/performance/run_perf_comparison.sh \
                     mlir/utils/performance/perfRunner.py \
                     mlir/utils/performance/tuningRunner.py 2>/dev/null || true
git checkout "$FEATURE_BRANCH"
if [ "$STASH_CREATED" = "1" ]; then
    git stash pop
    STASH_CREATED=0
fi
# Clear the trap since we manually restored
ORIGINAL_BRANCH=""

# --- Stage 10: Generate Excel report ---
log "Stage 10: Generating comparison Excel"

ISA_FLAG=""
if [ "$SKIP_ISA" = "1" ]; then
    ISA_FLAG="--skip-isa"
fi

python3 "$SCRIPT_DIR/swap_operands_perf_compare.py" \
    --feature-branch "$FEATURE_BRANCH" \
    --base-branch "$BASE_BRANCH" \
    --feature-build-dir "$BUILD_FEATURE" \
    --base-build-dir "$BUILD_BASE" \
    --feature-gemm-tsv "$OUTPUT_DIR/${FEATURE_BRANCH}_gemm.tsv" \
    --feature-conv-tsv "$OUTPUT_DIR/${FEATURE_BRANCH}_conv.tsv" \
    --feature-gemm-csv "$OUTPUT_DIR/${FEATURE_BRANCH}_gemm_perf.csv" \
    --feature-conv-csv "$OUTPUT_DIR/${FEATURE_BRANCH}_conv_perf.csv" \
    --base-gemm-tsv "$OUTPUT_DIR/${BASE_BRANCH}_gemm.tsv" \
    --base-conv-tsv "$OUTPUT_DIR/${BASE_BRANCH}_conv.tsv" \
    --base-gemm-csv "$OUTPUT_DIR/${BASE_BRANCH}_gemm_perf.csv" \
    --base-conv-csv "$OUTPUT_DIR/${BASE_BRANCH}_conv_perf.csv" \
    --gemm-configs "$GEMM_CONFIGS" \
    --conv-configs "$CONV_CONFIGS" \
    --output "$OUTPUT_DIR/perf_comparison_${GFX_ARCH}.xlsx" \
    --base-commit "$BASE_COMMIT" \
    --feature-commit "$FEATURE_COMMIT" \
    --hostname "$HOSTNAME_STR" \
    --gpu-arch "$GFX_ARCH" \
    --validate \
    $ISA_FLAG

# --- Stage 11: Independent validation against source files ---
log "Stage 11: Independent validation of Excel against source CSVs/TSVs"
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python3 -c "
from swap_operands_perf_compare import (
    parse_tuning_tsv, parse_perf_csv, validate_excel_against_sources
)
import sys

ok = validate_excel_against_sources(
    excel_path='$OUTPUT_DIR/perf_comparison_${GFX_ARCH}.xlsx',
    base_gemm_tsv='$OUTPUT_DIR/${BASE_BRANCH}_gemm.tsv',
    base_gemm_csv='$OUTPUT_DIR/${BASE_BRANCH}_gemm_perf.csv',
    feat_gemm_tsv='$OUTPUT_DIR/${FEATURE_BRANCH}_gemm.tsv',
    feat_gemm_csv='$OUTPUT_DIR/${FEATURE_BRANCH}_gemm_perf.csv',
    base_conv_tsv='$OUTPUT_DIR/${BASE_BRANCH}_conv.tsv',
    base_conv_csv='$OUTPUT_DIR/${BASE_BRANCH}_conv_perf.csv',
    feat_conv_tsv='$OUTPUT_DIR/${FEATURE_BRANCH}_conv.tsv',
    feat_conv_csv='$OUTPUT_DIR/${FEATURE_BRANCH}_conv_perf.csv',
    base_branch='$BASE_BRANCH',
    feature_branch='$FEATURE_BRANCH',
    num_checks=10,
)
sys.exit(0 if ok else 1)
"

echo ""
log "All done!"
echo "Results directory: $OUTPUT_DIR"
echo "Excel report:     $OUTPUT_DIR/perf_comparison_${GFX_ARCH}.xlsx"
echo ""
echo "Artifacts:"
ls -lh "$OUTPUT_DIR"/*.tsv "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.xlsx 2>/dev/null || true
