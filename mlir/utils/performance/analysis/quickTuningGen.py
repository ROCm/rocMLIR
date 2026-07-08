#!/usr/bin/env python3
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Quick Tuning Generator

Generates per-key C++ .inc files for the quick-tuning database from tuning data produced by
tuningRunner.py. Each table key (arch_op_dtype) gets its own .inc file in the output directory.

Architecture and operation type are auto-detected from file headers. The script groups input
files by (arch, op) and processes each group independently.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
import xxhash

# =============================================================================
# Constants
# =============================================================================

DB_DIR_NAME = "QuickTuningDb"

# Default output directory, relative to the repo root.
DEFAULT_OUTPUT_REL_PATH = Path("mlir") / "lib" / "Dialect" / "Rock" / "Tuning" / DB_DIR_NAME

# Column definitions for grouping problems
GEMM_COLUMNS = ['TransA', 'TransB', 'G', 'M', 'K', 'N']
CONV_COLUMNS = [
    'Direction', 'FilterLayout', 'InputLayout', 'OutputLayout', 'N', 'C', 'H', 'W', 'K', 'Y', 'X',
    'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW'
]
ATTENTION_COLUMNS = [
    'TransQ', 'TransK', 'TransV', 'TransO', 'Causal', 'ReturnLSE', 'SplitKV', 'WithAttnScale',
    'WithAttnBias', 'G', 'SeqLenQ', 'SeqLenK', 'NumHeadsQ', 'NumHeadsKV', 'HeadDimQK', 'HeadDimV'
]

# =============================================================================
# Helper Functions
# =============================================================================


def get_target_columns(op):
    """Get the columns used to identify unique problems for an operation."""
    if op == "gemm":
        return GEMM_COLUMNS
    elif op == "conv":
        return CONV_COLUMNS
    elif op == "attention":
        return ATTENTION_COLUMNS
    else:
        raise ValueError(f"Unknown operation: {op}")


def detect_op(columns):
    """Detect operation type from column names."""
    col_set = set(columns)
    if 'TransQ' in col_set:
        return "attention"
    if 'Direction' in col_set:
        return "conv"
    if 'TransA' in col_set:
        return "gemm"
    raise ValueError(f"Cannot detect operation from columns: {columns}")


def make_table_key(arch, op, dtype):
    """Build the table key string from architecture, operation, and data type."""
    return f"{arch}_{op}_{dtype}"


def make_problem_key(prob_tuple):
    """Build the problem key string from a groupby key tuple."""
    return "_".join(str(v) for v in prob_tuple)


def hash_problem_key(key):
    """Hash a problem key string to uint64 via xxh3_64."""
    return xxhash.xxh3_64_intdigest(key.encode())


def parse_perfconfig(perfconfig):
    """Parse a perfconfig string into format, version, and params."""
    parts = perfconfig.split(":")
    if len(parts) == 3:
        # format:vN:params
        return parts[0], int(parts[1][1:]), parts[2].split(",")
    elif len(parts) == 2:
        if parts[0].startswith("v"):
            # vN:params
            return None, int(parts[0][1:]), parts[1].split(",")
        else:
            # format:params
            return parts[0], 1, parts[1].split(",")
    else:
        # params only
        return None, 1, perfconfig.split(",")


def get_splitk_value(perfconfig):
    """Extract the Split-K value from a perfconfig string."""
    fmt, version, params = parse_perfconfig(perfconfig)

    idx = None
    if fmt == "attn":
        if version >= 3:
            idx = 8
        elif version >= 2:
            idx = 7
    else:
        if version >= 4:
            idx = 7
        elif version >= 2:
            idx = 6

    if idx is not None and idx < len(params):
        return int(params[idx])

    return None


# =============================================================================
# File Scanning & Loading
# =============================================================================


def validate_files(files):
    """Validate that all files exist and are .debug files."""
    errors = []
    for f in files:
        if not f.endswith('.debug'):
            errors.append(f"{f} is not a .debug file")
        elif not os.path.isfile(f):
            errors.append(f"{f} not found")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def group_files(files):
    """Scan file headers to detect (arch, op) and group files.

    Reads only the header + first data row of each file to determine the architecture (Chip column)
    and operation type (from column names).

    Returns a dict: (arch, op) -> list of file paths.
    """
    groups = defaultdict(list)

    for f in files:
        header_df = pd.read_csv(f, sep='\t', nrows=1)
        op = detect_op(header_df.columns)
        arch = str(header_df['Chip'].iloc[0])
        groups[(arch, op)].append(f)

    return dict(groups)


def load_files(files, op, no_splitk):
    """Load tuning data from files for a single (arch, op) group.

    Returns a dict: dtype -> DataFrame with columns [*target_cols, PerfConfig, TFlops].
    """
    target_cols = get_target_columns(op)
    usecols = ['DataType'] + target_cols + ['PerfConfig', 'TFlops']

    dfs = [pd.read_csv(f, sep='\t', usecols=usecols) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Cast bool columns to int so groupby keys carry plain Python ints. Keeps
    # serialization simple.
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    # Filter out configs where Split-K != 1
    if no_splitk and not df.empty:
        before = len(df)
        df = df[df['PerfConfig'].apply(lambda x: get_splitk_value(x) in (None, 1))]
        if len(df) < before:
            print(f"Filtered out {before - len(df)} of {before} Split-K configs")

    # Aggregate by keeping only the best TFlops per (problem, config), grouped by dtype
    group_cols = target_cols + ['PerfConfig']
    return {
        dtype: g.groupby(group_cols, as_index=False)['TFlops'].max()
        for dtype, g in df.groupby('DataType')
    }


# =============================================================================
# Set Cover Solver
# =============================================================================


def solve_set_cover(df_agg, op, threshold):
    """Find minimal covering set of perfconfigs using set cover optimization.

    Accepts a pre-aggregated DataFrame for a single (arch, dtype) with columns [*target_cols,
    PerfConfig, TFlops] where each (problem, config) pair has a single best TFlops value.

    The ILP formulation:
        minimize    sum(x[j] for all configs j)
        subject to  sum(coverage[i,j] * x[j]) >= 1  for each problem i
                    x[j] in {0, 1}

    where coverage[i,j] = 1 if config j is among the top performers for problem i.

    Returns (set_cover_list, problem_map):
      - set_cover_list: list of perfconfig strings, sorted by coverage count descending
      - problem_map: dict mapping problem_key_hash (uint64) -> list of indices into set_cover_list
    """
    target_cols = get_target_columns(op)

    # Build coverage: for each problem, which configs are "good enough"?
    coverage = {}
    for name, group in df_agg.groupby(target_cols):
        max_tflops = group['TFlops'].max()
        top = group[group['TFlops'] >= max_tflops * threshold]['PerfConfig'].tolist()
        coverage[name] = top

    problems = sorted(coverage.keys())
    configs = sorted({c for cs in coverage.values() for c in cs})
    config_idx = {c: i for i, c in enumerate(configs)}

    # Build coverage matrix: matrix[i,j] = 1 if config j covers problem i
    n_problems, n_configs = len(problems), len(configs)
    matrix = np.zeros((n_problems, n_configs), dtype=int)
    for i, prob in enumerate(problems):
        for cfg in coverage[prob]:
            matrix[i, config_idx[cfg]] = 1

    # Solve set cover with ILP
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n_configs), cat='Binary')

    # Objective: minimize number of selected configs
    prob += pulp.lpSum(x[j] for j in range(n_configs))

    # Constraints: each problem must be covered by at least one config
    for i in range(n_problems):
        prob += pulp.lpSum(matrix[i, j] * x[j] for j in range(n_configs)) >= 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if status != pulp.LpStatusOptimal:
        status_name = pulp.LpStatus.get(status, "Unknown")
        raise RuntimeError(
            f"Set cover failed: {status_name}. This likely indicates corrupted input data or a bug."
        )

    # Extract selected configs, sorted by how many problems they cover
    selected = [configs[j] for j in range(n_configs) if x[j].varValue == 1]
    counts = {c: sum(matrix[i, config_idx[c]] for i in range(n_problems)) for c in selected}
    set_cover = sorted(selected, key=lambda c: counts[c], reverse=True)
    set_cover_idx = {c: i for i, c in enumerate(set_cover)}

    problem_map = {}
    for i, prob_key_tuple in enumerate(problems):
        key_str = make_problem_key(prob_key_tuple)
        h = hash_problem_key(key_str)
        indices = [set_cover_idx[c] for c in selected if matrix[i, config_idx[c]] == 1]
        if h in problem_map:
            print(f"WARNING: Hash collision for problem key '{key_str}' (hash={h})")
            problem_map[h] = sorted(set(problem_map[h] + indices))
        else:
            problem_map[h] = sorted(indices)

    return set_cover, problem_map


# =============================================================================
# Inc File Output
# =============================================================================


def get_output_dir():
    """Get the default output directory for per-key .inc files."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    return repo_root / DEFAULT_OUTPUT_REL_PATH


def to_pascal(key):
    """Convert a snake_case key like 'gfx942_conv_f16' to PascalCase 'Gfx942ConvF16'."""
    return "".join(part.capitalize() for part in key.split("_"))


def format_inc(key, set_cover, problem_map):
    """Format a table entry as a C++ .inc file with two-phase inclusion.

    The generated file uses #ifdef guards so it can be included twice:
      QUICK_TUNING_DB_ARRAYS  - emits file-scope static const array declarations
      QUICK_TUNING_DB_ENTRIES - emits a single initializer entry
    """
    pascal = to_pascal(key)
    lines = [f"// {pascal}.inc -- auto-generated by quickTuningGen.py"]

    # Phase 1: array declarations
    lines.append("#ifdef QUICK_TUNING_DB_ARRAYS")

    lines.append(f"static const StringRef kSetCover{pascal}[] = {{")
    for cfg in set_cover:
        lines.append(f'  "{cfg}",')
    lines.append("};")

    # Flatten indices into a single array, recording (offset, count) per problem
    if problem_map:
        flat_indices = []
        problem_refs = []
        for h in sorted(problem_map.keys()):
            indices = problem_map[h]
            problem_refs.append((h, len(flat_indices), len(indices)))
            flat_indices.extend(indices)

        lines.append(f"static const unsigned kIndices{pascal}[] = {{")
        for i in range(0, len(flat_indices), 16):
            chunk = ", ".join(str(v) for v in flat_indices[i:i + 16])
            lines.append(f"  {chunk},")
        lines.append("};")

        lines.append(f"static const ProblemRef kProblemMap{pascal}[] = {{")
        for h, offset, count in problem_refs:
            lines.append(f"  {{0x{h:016x}ULL, {offset}, {count}}},")
        lines.append("};")

    lines.append("#endif")

    # Phase 2: initializer entries
    lines.append("#ifdef QUICK_TUNING_DB_ENTRIES")

    sc_size = len(set_cover)
    if problem_map:
        idx_size = sum(len(v) for v in problem_map.values())
        pm_size = len(problem_map)
        lines.append(f'{{"{key}", kSetCover{pascal}, {sc_size},'
                     f' kIndices{pascal}, {idx_size},'
                     f' kProblemMap{pascal}, {pm_size}}},')
    else:
        lines.append(f'{{"{key}", kSetCover{pascal}, {sc_size},'
                     f' nullptr, 0, nullptr, 0}},')

    lines.append("#endif")

    return "\n".join(lines) + "\n"


def save_entry(key, set_cover, problem_map, output_dir):
    """Save a single table entry as a per-key .inc file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pascal = to_pascal(key)
    path = output_dir / f"{pascal}.inc"
    path.write_text(format_inc(key, set_cover, problem_map))
    return path


# =============================================================================
# Entry Point
# =============================================================================


def process_files(files, arch, op, threshold, no_splitk, output_dir):
    """Process a group of files for a single (arch, op) combination.

    Loads the files, solves set cover per dtype, and writes per-key .inc files.
    Returns a list of written file paths.
    """
    print(f"Processing ({arch}, {op}): {len(files)} file(s)...")

    dtype_data = load_files(files, op, no_splitk)

    if not dtype_data:
        print("No data after loading/filtering.")
        return []

    written = []
    for dtype in sorted(dtype_data):
        df = dtype_data[dtype]
        key = make_table_key(arch, op, dtype)

        set_cover, problem_map = solve_set_cover(df, op, threshold)

        path = save_entry(key, set_cover, problem_map, output_dir)
        written.append(path)

        print(f"Wrote {path.name}: {len(set_cover)} configs, {len(problem_map)} problems")

    return written


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='quickTuningGen.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate per-key quick-tuning database .inc files from tuning data.',
        epilog='''
Examples:
    %(prog)s tuningData/*.debug
    %(prog)s tuningData/gfx942/*.debug -o /tmp/quick-tuning-db
    find . -name "*.debug" | xargs %(prog)s
''')

    parser.add_argument('files',
                        nargs='+',
                        metavar='FILE',
                        help='.debug files produced by tuningRunner.py')
    parser.add_argument('--th',
                        type=float,
                        default=0.93,
                        metavar='THRESHOLD',
                        help='Coverage threshold (default: 0.93)')
    parser.add_argument('--no-splitk', action='store_true', help='Exclude Split-K configurations')
    parser.add_argument('-o',
                        '--output',
                        type=Path,
                        default=None,
                        metavar='DIR',
                        help='Output directory for per-key .inc files '
                        f'(default: <repo>/{DEFAULT_OUTPUT_REL_PATH.as_posix()})')

    pargs = parser.parse_args(args)

    validate_files(pargs.files)

    print("Input files:")
    for f in pargs.files:
        print(f"    {f}")

    print("Scanning file headers...")
    groups = group_files(pargs.files)

    if not groups:
        print("No matching data to process.", file=sys.stderr)
        return 1

    print("Grouped files by (arch, op):")
    for (arch, op), file_list in groups.items():
        print(f"    ({arch}, {op}): {len(file_list)} file(s)")

    output_dir = pargs.output or get_output_dir()

    all_written = []
    for (arch, op), file_list in groups.items():
        written = process_files(file_list, arch, op, pargs.th, pargs.no_splitk, output_dir)
        all_written.extend(written)

    print(f"Wrote {len(all_written)} file(s) to {output_dir}")
    for p in all_written:
        print(f"    {p.name}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
