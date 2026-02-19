#!/usr/bin/env python3
"""
Compare TFLops from two branches (develop vs schedGroup) by:
1. Building each branch
2. Running tier1-gemm-configs with the branch's tuned perfconfigs
3. Merging results and writing an Excel file with Develop as reference and % difference.

Usage:
  From repo root:
    python3 scripts/compare_tflops_branches.py [--no-build] [--output FILE.xlsx]

  Options:
    --no-build     Skip building; only run perf and compare (use if both builds already exist)
    --output       Output Excel path (default: build/tflops_comparison_develop_vs_schedGroup.xlsx)
    --build-dir    Build directory (default: build)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("pandas and numpy are required. Install with: pip install pandas numpy", file=sys.stderr)
    sys.exit(1)
try:
    import openpyxl  # for Excel writing
except ImportError:
    print("openpyxl is required for Excel output. Install with: pip install openpyxl",
          file=sys.stderr)
    sys.exit(1)

# Default paths (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
TIER1_GEMM_CONFIGS = REPO_ROOT / "mlir/utils/performance/configs/tier1-gemm-configs"
DEVELOP_TSV = "develop_greedy.tsv"
SCHEDGROUP_TSV = "schedGroup_200_greedy.tsv"
DEVELOP_BRANCH = "develop"
SCHEDGROUP_BRANCH = "schedGroup"

# Columns that identify a unique config (problem dimensions); exclude PerfConfig and TFlops
GEMM_MERGE_KEY = [
    "DataType",
    "OutDataType",
    "Chip",
    "numCU",
    "numChiplets",
    "TransA",
    "TransB",
    "G",
    "M",
    "K",
    "N",
    "ScaledGemm",
    "ScaleADtype",
    "ScaleBDtype",
    "TransScaleA",
    "TransScaleB",
]


def run_cmd(cmd, cwd=None, env=None, check=True):
    """Run a command; raise on failure if check=True."""
    cwd = cwd or os.getcwd()
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env or os.environ)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result.returncode


def get_current_branch(repo_root):
    """Return current git branch name."""
    out = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root,
        text=True,
    )
    return out.strip()


def build_repo(build_dir):
    """Configure and build in build_dir (from repo root)."""
    build_dir = Path(build_dir)
    if not build_dir.exists():
        build_dir.mkdir(parents=True)
    # Configure if needed
    if not (build_dir / "build.ninja").exists() and not (build_dir / "Makefile").exists():
        run_cmd(["cmake", "-G", "Ninja", "-B", str(build_dir), str(REPO_ROOT)], cwd=REPO_ROOT)
    run_cmd(["ninja", "-C", str(build_dir)], cwd=REPO_ROOT)


def run_perf_runner(build_dir, tuning_tsv, configs_file, output_csv):
    """Run perfRunner.py from build_dir with given tuning DB and configs; write CSV to output_csv."""
    build_dir = Path(build_dir)
    configs_file = Path(configs_file)
    # Config path relative to build dir
    configs_rel = os.path.relpath(configs_file, build_dir)
    perf_runner = build_dir / "bin" / "perfRunner.py"
    if not perf_runner.exists():
        raise FileNotFoundError(f"perfRunner not found: {perf_runner}")
    tuning_path = build_dir / tuning_tsv
    if not tuning_path.exists():
        raise FileNotFoundError(f"Tuning DB not found: {tuning_path}")
    cmd = [
        sys.executable,
        str(perf_runner),
        "-t",
        str(tuning_path),
        "-c",
        configs_rel,
        "--batch_mlir",
        "--op",
        "gemm",
        "-o",
        str(output_csv),
    ]
    run_cmd(cmd, cwd=str(build_dir))


def load_and_normalize_csv(path):
    """Load CSV and ensure merge key columns exist."""
    df = pd.read_csv(path)
    for col in GEMM_MERGE_KEY:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")
    return df


def compare_and_export_excel(develop_csv, schedgroup_csv, output_excel):
    """
    Merge develop and schedGroup results on problem config, add Develop TFlops (reference),
    schedGroup TFlops, and % difference. Write to Excel.
    """
    df_dev = load_and_normalize_csv(develop_csv)
    df_sched = load_and_normalize_csv(schedgroup_csv)

    # Keep one set of key columns and TFlops from each
    df_dev_merge = df_dev[GEMM_MERGE_KEY + ["TFlops"]].copy()
    df_dev_merge = df_dev_merge.rename(columns={"TFlops": "Develop_TFlops"})

    df_sched_merge = df_sched[GEMM_MERGE_KEY + ["TFlops"]].copy()
    df_sched_merge = df_sched_merge.rename(columns={"TFlops": "schedGroup_TFlops"})

    # Merge on key (outer to keep all configs from either run)
    merged = df_dev_merge.merge(df_sched_merge, on=GEMM_MERGE_KEY, how="outer")

    # Reference = Develop
    ref = merged["Develop_TFlops"]
    other = merged["schedGroup_TFlops"]
    # % difference: (schedGroup - develop) / develop * 100. Positive = schedGroup faster.
    merged["Pct_Diff_vs_Develop"] = np.where(
        pd.notna(ref) & (ref != 0),
        (other - ref) / ref * 100.0,
        np.nan,
    )
    # Reorder columns: key cols, then Develop TFlops (reference), then schedGroup, then %
    key_cols = [c for c in GEMM_MERGE_KEY if c in merged.columns]
    rest = [
        c for c in ["Develop_TFlops", "schedGroup_TFlops", "Pct_Diff_vs_Develop"]
        if c in merged.columns
    ]
    merged = merged[key_cols + rest]

    merged.to_excel(output_excel, index=False)
    print(f"Wrote: {output_excel}")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=
        "Compare TFLops between develop and schedGroup branches (build, run tier1-gemm, export Excel)."
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building; only run perf and compare (both branch builds must already exist)",
    )
    parser.add_argument(
        "--only-merge",
        action="store_true",
        help=
        "Only merge existing develop and schedGroup CSVs and write Excel (no build, no perf run)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output Excel path (default: build/tflops_comparison_develop_vs_schedGroup.xlsx)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="Build directory (default: build)",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if args.output is None:
        args.output = build_dir / "tflops_comparison_develop_vs_schedGroup.xlsx"
    else:
        args.output = Path(args.output).resolve()

    develop_csv = build_dir / "develop_tier1_gemm_results.csv"
    schedgroup_csv = build_dir / "schedGroup_tier1_gemm_results.csv"

    if args.only_merge:
        if not develop_csv.exists() or not schedgroup_csv.exists():
            print(
                f"For --only-merge, both CSVs must exist: {develop_csv}, {schedgroup_csv}",
                file=sys.stderr,
            )
            sys.exit(1)
        compare_and_export_excel(develop_csv, schedgroup_csv, args.output)
        print("Done.")
        return

    if not TIER1_GEMM_CONFIGS.exists():
        print(f"Config file not found: {TIER1_GEMM_CONFIGS}", file=sys.stderr)
        sys.exit(1)

    # --- Develop branch ---
    print("--- Develop branch ---")
    current = get_current_branch(REPO_ROOT)
    if current != DEVELOP_BRANCH and not args.no_build:
        print(f"Checking out {DEVELOP_BRANCH} (current: {current})")
        run_cmd(["git", "checkout", DEVELOP_BRANCH], cwd=REPO_ROOT)
    if not args.no_build:
        build_repo(build_dir)
    if (build_dir / DEVELOP_TSV).exists():
        print(f"Running perfRunner with {DEVELOP_TSV} -> {develop_csv.name}")
        run_perf_runner(build_dir, DEVELOP_TSV, TIER1_GEMM_CONFIGS, develop_csv)
    else:
        print(
            f"Warning: {build_dir / DEVELOP_TSV} not found; skipping develop run. Copy it to build/ or run with existing CSV."
        )
        if not develop_csv.exists():
            print("No develop CSV found; cannot compare.", file=sys.stderr)
            sys.exit(1)

    # --- schedGroup branch ---
    print("--- schedGroup branch ---")
    current = get_current_branch(REPO_ROOT)
    if current != SCHEDGROUP_BRANCH and not args.no_build:
        print(f"Checking out {SCHEDGROUP_BRANCH} (current: {current})")
        run_cmd(["git", "checkout", SCHEDGROUP_BRANCH], cwd=REPO_ROOT)
    if not args.no_build:
        build_repo(build_dir)
    if (build_dir / SCHEDGROUP_TSV).exists():
        print(f"Running perfRunner with {SCHEDGROUP_TSV} -> {schedgroup_csv.name}")
        run_perf_runner(build_dir, SCHEDGROUP_TSV, TIER1_GEMM_CONFIGS, schedgroup_csv)
    else:
        print(f"Warning: {build_dir / SCHEDGROUP_TSV} not found; skipping schedGroup run.")
        if not schedgroup_csv.exists():
            print("No schedGroup CSV found; cannot compare.", file=sys.stderr)
            sys.exit(1)

    # --- Compare and export Excel ---
    if not develop_csv.exists() or not schedgroup_csv.exists():
        print("Missing one or both result CSVs.", file=sys.stderr)
        sys.exit(1)
    compare_and_export_excel(develop_csv, schedgroup_csv, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
