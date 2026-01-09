#!/usr/bin/env python3
"""
Quick Tuning Generator - Generates QuickTuningPerfconfigs.inc from tuning data.

Usage:
    python3 quickTuningGen.py --input-dir tunedData --op conv --arch gfx90a --update --no-splitk
    python3 quickTuningGen.py --input-dir tunedData --op attention --arch gfx942 --update
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

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


def get_instruction_type(arch, dtype, op):
    """Determine instruction type based on architecture, data type, and operation."""
    if op == "attention":
        return "GemmGemm"
    if arch.startswith("gfx9"):
        return "XDL"
    elif arch.startswith("gfx1") and dtype != "f32":
        return "Wmma"
    return "NonAccel"


def is_accel(arch, dtype, op):
    """Check if this combination uses accelerated instructions."""
    return get_instruction_type(arch, dtype, op) != "NonAccel"


def get_class_name(arch, dtype, op):
    """Get the PopulateParams class name."""
    instr = get_instruction_type(arch, dtype, op)
    return f"PopulateParams{instr}" if instr != "NonAccel" else "PopulateParams"


def get_param_names(arch, dtype, op):
    """Generate array and count variable names."""
    base = f"initParameters{dtype.capitalize()}{op.capitalize()}{arch.capitalize()}"
    return base, f"n{base[0].upper()}{base[1:]}"


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
    """Extract the split-K value from a perfconfig string."""
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
        return params[idx]
    return None


# =============================================================================
# Data Loading & Processing
# =============================================================================


def load_data(input_dir, no_splitk, pattern=None):
    """Load and combine all .debug tuning files."""
    files = glob.glob(os.path.join(input_dir, "*.debug"))
    if not files:
        print(f"No .debug files found in '{input_dir}'", file=sys.stderr)
        return None

    if pattern:
        regex = re.compile(pattern)
        files = [f for f in files if regex.search(os.path.basename(f))]
        if not files:
            print(f"No .debug files matched the pattern '{pattern}' in '{input_dir}'",
                  file=sys.stderr)
            return None

    print(f"Found {len(files)} .debug file(s) in '{input_dir}':")
    for f in files:
        print(f"    - {os.path.basename(f)}")
    print()

    dfs = [pd.read_csv(f, sep='\t', index_col=None) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    if no_splitk:
        # Filter out configs where splitK != 1
        mask = df['PerfConfig'].apply(lambda x: get_splitk_value(x) in (None, '1'))
        df = df[mask]

    return df


def find_perfconfigs(df, op, threshold=0.93):
    """
    Find minimal covering set of perfconfigs using set cover optimization.

    For each problem (unique combination of problem dimensions), we identify
    configs that achieve >= threshold * best_tflops. We then solve a set cover
    problem to find the minimum number of configs that cover all problems.

    The ILP formulation:
        minimize    sum(x[j] for all configs j)
        subject to  sum(coverage[i,j] * x[j]) >= 1  for each problem i
                    x[j] in {0, 1}

    where coverage[i,j] = 1 if config j is among the top performers for problem i.
    """
    target_cols = get_target_columns(op)
    results = {}

    for dtype in df['DataType'].unique():
        df_typed = df[df['DataType'] == dtype]

        # Build coverage: for each problem, which configs are "good enough"?
        coverage = {}
        for name, group in df_typed.groupby(target_cols):
            max_tflops = group['TFlops'].max()
            top = group[group['TFlops'] >= max_tflops * threshold]['PerfConfig'].tolist()
            coverage[name] = top

        problems = list(coverage.keys())
        configs = list({c for cs in coverage.values() for c in cs})
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
            raise RuntimeError(f"Set cover failed for {dtype}: {status_name}. "
                               f"This likely indicates corrupted input data or a bug.")

        # Extract selected configs, sorted by how many problems they cover
        selected = [configs[j] for j in range(n_configs) if x[j].varValue == 1]
        counts = {c: sum(matrix[i, config_idx[c]] for i in range(n_problems)) for c in selected}
        results[dtype] = sorted(selected, key=lambda c: counts[c], reverse=True)

    return results


# =============================================================================
# File Generation
# =============================================================================


def get_output_path():
    """Get the output .inc file path relative to this script."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent.parent / "include/mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"


def get_generator_path():
    """Get this script's path relative to the repo root for the header comment."""
    script_path = Path(__file__).resolve()
    # Find repo root (contains .git or mlir directory)
    for parent in script_path.parents:
        if (parent / ".git").exists() or (parent / "mlir").is_dir():
            try:
                return script_path.relative_to(parent)
            except ValueError:
                pass
    return script_path.name


def init_inc_file(path):
    """Create empty .inc file with required structure."""
    sections = ["NonAccel", "XDL", "Wmma", "GemmGemm"]
    lookup_table_sections = ["NonAccel", "Accel", "GemmGemm"]
    lines = [f"// Generated by: {get_generator_path()}", "", "// clang-format off", ""]
    for s in sections:
        lines += [f"#ifdef {s}_DEFINITIONS_GEN", "", f"#endif // {s}_DEFINITIONS_GEN", ""]
        lines += [f"#ifdef {s}_DECLARATIONS_GEN", "", f"#endif // {s}_DECLARATIONS_GEN", ""]
    for s in lookup_table_sections:
        lines += [f"#ifdef {s}_LOOKUP_TABLE_GEN", "", f"#endif // {s}_LOOKUP_TABLE_GEN", ""]
    path.write_text("\n".join(lines))


def replace_section(content, ifdef_guard, begin_marker, end_marker, new_content):
    """Replace content between markers, creating section if needed."""
    pattern = re.compile(f'{re.escape(begin_marker)}.*?{re.escape(end_marker)}', re.DOTALL)

    if pattern.search(content):
        return pattern.sub(f'{begin_marker}\n{new_content}\n{end_marker}', content)

    # Section doesn't exist - find the #endif for this guard and insert before it
    endif_pattern = re.compile(
        rf'{re.escape(ifdef_guard)}.*?(#endif(?:\s*//\s*{re.escape(ifdef_guard[7:])})?)', re.DOTALL)
    match = endif_pattern.search(content)
    if not match:
        raise ValueError(f"Cannot find {ifdef_guard}")

    insert_pos = match.start(1)
    section = f'{begin_marker}\n{new_content}\n{end_marker}\n\n'
    return content[:insert_pos] + section + content[insert_pos:]


def add_lookup_entry(content, ifdef_guard, entry):
    """Add lookup table entry if not present."""
    if entry in content:
        return content

    # Find the #endif for this guard
    endif_pattern = re.compile(
        rf'{re.escape(ifdef_guard)}.*?(#endif(?:\s*//\s*{re.escape(ifdef_guard[7:])})?)', re.DOTALL)
    match = endif_pattern.search(content)
    if not match:
        raise ValueError(f"Cannot find {ifdef_guard}")

    insert_pos = match.start(1)
    return content[:insert_pos] + f'{entry}\n\n' + content[insert_pos:]


def get_lookup_guard(arch, dtype, op):
    """Get the appropriate lookup table ifdef guard."""
    if op == "attention":
        return "#ifdef GemmGemm_LOOKUP_TABLE_GEN"
    elif is_accel(arch, dtype, op):
        return "#ifdef Accel_LOOKUP_TABLE_GEN"
    else:
        return "#ifdef NonAccel_LOOKUP_TABLE_GEN"


def update_inc_file(results, arch, op):
    """Update the .inc file with results."""
    path = get_output_path()
    if not path.exists():
        init_inc_file(path)

    content = path.read_text()

    for dtype, configs in results.items():
        instr = get_instruction_type(arch, dtype, op)
        class_name = get_class_name(arch, dtype, op)
        param_name, count_name = get_param_names(arch, dtype, op)

        # Generate definition
        def_lines = [f"const StringRef {class_name}::{param_name}[] = {{"]
        for i, cfg in enumerate(configs):
            comma = "," if i < len(configs) - 1 else ""
            def_lines.append(f'    "{cfg}"{comma}')
        def_lines.append("};")

        content = replace_section(content, f"#ifdef {instr}_DEFINITIONS_GEN",
                                  f"// BEGIN_{op.upper()}_{instr}_{dtype}_{arch}_DEFS",
                                  f"// END_{op.upper()}_{instr}_{dtype}_{arch}_DEFS",
                                  "\n".join(def_lines))

        # Generate declaration
        dec_lines = [
            f"static constexpr size_t {count_name} = {len(configs)};",
            f"static const StringRef {param_name}[{count_name}];"
        ]

        content = replace_section(content, f"#ifdef {instr}_DECLARATIONS_GEN",
                                  f"// BEGIN_{op.upper()}_{instr}_{dtype}_{arch}_DECS",
                                  f"// END_{op.upper()}_{instr}_{dtype}_{arch}_DECS",
                                  "\n".join(dec_lines))

        # Add lookup entry
        lookup_guard = get_lookup_guard(arch, dtype, op)
        entry = f'{{"{arch}_{op}_{dtype}", {{{class_name}::{param_name}, {class_name}::{count_name}}}}},'
        content = add_lookup_entry(content, lookup_guard, entry)

    path.write_text(content)


# =============================================================================
# Main
# =============================================================================


def print_results(results):
    """Print selected perfconfigs."""
    for dtype, configs in results.items():
        print(f"Datatype: {dtype} ({len(configs)} configs)")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i:3d}: {cfg}")
        print()


def main(args=None):
    parser = argparse.ArgumentParser(prog='quickTuningGen.py')
    parser.add_argument('--input-dir', required=True, help='Directory with .debug files')
    parser.add_argument('--op', required=True, choices=['gemm', 'conv', 'attention'])
    parser.add_argument('--arch', required=True, help='Target arch (e.g., gfx90a)')
    parser.add_argument('--th', type=float, default=0.93, help='Threshold (default: 0.93)')
    parser.add_argument('--update', action='store_true', help='Update .inc file')
    parser.add_argument('--no-splitk', action='store_true', help='Exclude Split-K configs')
    parser.add_argument('--pattern', help='Regex pattern to filter .debug filenames')

    pargs = parser.parse_args(args)

    df = load_data(pargs.input_dir, pargs.no_splitk, pargs.pattern)
    if df is None or df.empty:
        return 1

    results = find_perfconfigs(df, pargs.op, pargs.th)
    print_results(results)

    if pargs.update:
        print(f"Updating: {get_output_path()}")
        update_inc_file(results, pargs.arch, pargs.op)
        print("Done!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
