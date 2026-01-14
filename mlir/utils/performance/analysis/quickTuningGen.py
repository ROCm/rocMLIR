#!/usr/bin/env python3
"""Quick Tuning Generator

Generates QuickTuningPerfconfigs.inc from tuning data produced by tuningRunner.py.
"""

import argparse
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

# =============================================================================
# Helper Functions
# =============================================================================


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
        return params[idx]
    return None


# =============================================================================
# Data Loading & Processing
# =============================================================================


def validate_files(files):
    """Validate that all files exist and are .debug files."""
    errors = []
    for f in files:
        if not f.endswith('.debug'):
            errors.append(f"{f} is not a .debug file")
        elif not os.path.isfile(f):
            errors.append(f"{f} not found")

    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if errors:
        sys.exit(1)


def load_data(files, no_splitk):
    """Load tuning data from files or stdin."""
    if files:
        validate_files(files)

        print(f"Processing {len(files)} file(s):")
        for f in files:
            print(f"    {f}")

        dfs = [pd.read_csv(f, sep='\t', index_col=None) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        # Read TSV content from stdin
        print("Reading from stdin...")
        df = pd.read_csv(sys.stdin, sep='\t', index_col=None)

    if no_splitk:
        # Filter out configs where Split-K != 1
        before = len(df)
        mask = df['PerfConfig'].apply(lambda x: get_splitk_value(x) in (None, '1'))
        df = df[mask]
        if len(df) < before:
            print(f"Filtered out {before - len(df)} out of {before} Split-K configs")

    return df


def find_perfconfigs(df, op, threshold):
    """Find minimal covering set of perfconfigs using set cover optimization.

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


def print_results(results, arch):
    """Print selected perfconfigs for an architecture."""
    print(f"=== {arch} ===")
    for dtype, configs in results.items():
        print(f"{dtype}: {len(configs)} configs")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i:3d}: {cfg}")
    print()


def process_arch(df, arch, op, threshold, update):
    """Process data for a single architecture."""
    df_arch = df[df['Chip'] == arch]

    results = find_perfconfigs(df_arch, op, threshold)
    print_results(results, arch)

    if update:
        update_inc_file(results, arch, op)


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='quickTuningGen.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate QuickTuningPerfconfigs.inc from tuning data.',
        epilog='''
Examples:
    %(prog)s tuningData/*.debug --op conv --update
    %(prog)s gfx90a/*.debug gfx942/*.debug --op gemm --update
    cat data.debug | %(prog)s --op attention --update
    find . -name "*.debug" | xargs %(prog)s --op gemm --update
''')

    parser.add_argument(
        'files',
        nargs='*',
        metavar='FILE',
        help='.debug files produced by tuningRunner.py (reads TSV from stdin if none provided)')
    parser.add_argument('--op',
                        required=True,
                        choices=['gemm', 'conv', 'attention'],
                        help='Operation')
    parser.add_argument('--th',
                        type=float,
                        default=0.93,
                        metavar='THRESHOLD',
                        help='Coverage threshold (default: 0.93)')
    parser.add_argument('--update', action='store_true', help='Update QuickTuningPerfconfigs.inc')
    parser.add_argument('--no-splitk', action='store_true', help='Exclude Split-K configurations')

    pargs = parser.parse_args(args)

    df = load_data(pargs.files, pargs.no_splitk)
    if df.empty:
        print("ERROR: No data loaded", file=sys.stderr)
        return 1

    # Process each architecture found in the data
    archs = sorted(df['Chip'].unique())
    print(f"Processing {len(archs)} architecture(s): {', '.join(archs)}")
    print()

    for arch in archs:
        process_arch(df, arch, pargs.op, pargs.th, pargs.update)

    if pargs.update:
        print(f"Updated: {get_output_path()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
