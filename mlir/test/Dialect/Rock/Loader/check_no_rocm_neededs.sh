#!/usr/bin/env bash
# Helper for `no_rocm_neededs.test`. Scans build artefacts and asserts
# that none of them transitively `NEED` a ROCm runtime library.
#
# Usage: check_no_rocm_neededs.sh <shlib_dir> <tools_dir>
#
# Version-agnostic by design: the artefact list uses globs against
# library *base names*, never hardcoded `.so.<MAJOR>.<MINOR>` suffixes,
# so a future LLVM version bump (e.g. to libMLIRRocmRuntimeLoader.so.24)
# or a rocMLIR version bump (libMLIRRockOps.so.3.0) does not silently
# turn this test into a no-op.

set -u
shlib_dir="${1:?shlib dir}"
tools_dir="${2:?tools dir}"

# SONAMEs that must never appear in `NEEDED`. We anchor on the SONAME
# prefix (without `.so.<ver>`) so the assertion holds across ROCm major
# versions. `libLLVM.so` (no `.<MAJOR>.<MINOR>git` suffix) is ROCm's
# monolithic build; the in-tree LLVM uses libLLVMSupport.so.<MAJOR>git
# etc., which we deliberately do NOT match.
forbidden=(
  "libamdhip64"
  "libhiprtc"
  "libamd_comgr"
  "libLLVM.so"   # ROCm's monolithic libLLVM
)

# Globs (resolved at run time) for the artefacts we want to audit.
# Tools have no extension so we list them directly. Shared libraries
# are matched by base-name + a wildcard suffix; whichever
# version-decorated file the build emits will match.
tool_names=(
  "rocmlir-driver"
  "rocmlir-opt"
  "rocmlir-gen"
  "rocmlir-tuning-driver"
  "xmir-runner"
  "mlir-runner"
)

shlib_globs=(
  "libMLIRRockOps.so*"
  "libMLIRRocmRuntimeLoader.so*"
  "libmlir_rocm_runtime.so*"
  "libMLIRRocmExecutionEngineUtils.so*"
)

# Build the concrete artefact list at run time.
artefacts=()
for t in "${tool_names[@]}"; do
  artefacts+=("${tools_dir}/${t}")
done
for g in "${shlib_globs[@]}"; do
  # `nullglob` lets the loop simply iterate zero times when no file
  # matches (e.g. mlir_rocm_runtime is only built when
  # MLIR_ENABLE_ROCM_RUNNER=ON), without leaving the literal pattern
  # in the array.
  shopt -s nullglob
  for f in "${shlib_dir}/"${g}; do
    # Skip plain dev symlinks (e.g. `libfoo.so`) and only audit the
    # actual file or the SONAME-versioned symlink. This avoids
    # double-reporting the same NEEDED set.
    if [ -L "${f}" ] && [ "$(basename "${f}")" = "$(basename "${f}" .so).so" ]; then
      continue
    fi
    artefacts+=("${f}")
  done
  shopt -u nullglob
done

if [ "${#artefacts[@]}" -eq 0 ]; then
  echo "no_rocm_neededs: no artefacts found under ${shlib_dir} / ${tools_dir}" >&2
  exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
  echo "no_rocm_neededs: skipping; readelf is not available" >&2
  exit 0
fi

failed=0
checked=0
for art in "${artefacts[@]}"; do
  if [ ! -e "${art}" ]; then
    # Tools may legitimately not be built in every config (e.g.
    # mlir-runner needs MLIR_ENABLE_ROCM_RUNNER=ON).
    continue
  fi
  checked=$((checked + 1))
  needed="$(readelf -d "${art}" 2>/dev/null | awk '/\(NEEDED\)/{print $5}' | tr -d '[]')"
  for forb in "${forbidden[@]}"; do
    while IFS= read -r soname; do
      [ -z "${soname}" ] && continue
      case "${soname}" in
        ${forb}*)
          echo "FAIL: ${art} declares NEEDED ${soname} (forbidden: ${forb})" >&2
          failed=1
          ;;
      esac
    done <<<"${needed}"
  done
done

if [ "${failed}" -ne 0 ]; then
  echo "no_rocm_neededs: at least one forbidden NEEDED entry was found." >&2
  exit 1
fi

echo "no_rocm_neededs: ${checked} artefact(s) checked, all clean."
exit 0
