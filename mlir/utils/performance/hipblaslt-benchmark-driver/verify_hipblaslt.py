#!/usr/bin/env python3
"""
Verification helper for hipblaslt-benchmark-driver.

This script compares the output of hipblaslt-benchmark-driver against
rocmlir-gen by running both with matching GEMM parameters and comparing
the printed results.

Usage:
    python verify_hipblaslt.py -m 32 -n 32 -k 32 -t f32 -arch gfx942 \\
        --hipblaslt-path <path> \\
        --rocmlir-gen-path <path> \\
        --rocmlir-driver-path <path> \\
        --runner-path <path> \\
        --libs <shared_libs>
"""

import argparse
import re
import subprocess
import sys


def parse_tensor_output(output):
    """Parse the printed tensor values from output.

    Extracts numeric values from the 'data = ' section of printMemrefF32 output.
    Returns a list of floats.
    """
    # Find the data section
    match = re.search(r'data\s*=\s*\n?(.*)', output, re.DOTALL)
    if not match:
        return []

    data_str = match.group(1)
    # Extract all numbers (including negative and scientific notation)
    numbers = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', data_str)
    return [float(n) for n in numbers]


def run_hipblaslt(hipblaslt_path, m, n, k, g, dtype, trans_a, trans_b):
    """Run hipblaslt-benchmark-driver with --print-results."""
    cmd = [
        hipblaslt_path,
        '-m',
        str(m),
        '-n',
        str(n),
        '-k',
        str(k),
        '-g',
        str(g),
        '-t',
        dtype,
        f'-transA={trans_a}',
        f'-transB={trans_b}',
        '--kernel-repeats',
        '1',
        '--warmup-runs',
        '1',
        '--print-results',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None, f"hipblaslt failed: {result.stderr}"
        return result.stdout, None
    except subprocess.TimeoutExpired:
        return None, "hipblaslt timed out"
    except Exception as e:
        return None, str(e)


def run_rocmlir_gen(rocmlir_gen_path, rocmlir_driver_path, runner_path, libs, m, n, k, g, dtype,
                    trans_a, trans_b, arch):
    """Run rocmlir-gen with -pr and execute to get reference output."""
    # Map dtype to rocmlir-gen format
    dtype_map = {'f32': 'f32', 'f16': 'f16', 'bf16': 'bf16', 'i8': 'i8', 'fp8': 'fp8'}
    rocmlir_dtype = dtype_map.get(dtype, dtype)

    # Convert trans_a/trans_b to lowercase for rocmlir-gen
    trans_a = trans_a.lower() if isinstance(trans_a, str) else str(trans_a).lower()
    trans_b = trans_b.lower() if isinstance(trans_b, str) else str(trans_b).lower()

    # Build rocmlir-gen command
    gen_cmd = [
        rocmlir_gen_path,
        '--arch',
        arch,
        '--operation',
        'gemm',
        '-t',
        rocmlir_dtype,
        '-m',
        str(m),
        '-n',
        str(n),
        '-k',
        str(k),
        '-g',
        str(g if g > 0 else 1),
        f'-transA={trans_a}',
        f'-transB={trans_b}',
        '-pr',  # print results
        '-ph',  # generate host harness
    ]

    try:
        # Generate MLIR
        gen_result = subprocess.run(gen_cmd, capture_output=True, text=True, timeout=60)
        if gen_result.returncode != 0:
            return None, f"rocmlir-gen failed: {gen_result.stderr}"

        # Compile with rocmlir-driver
        driver_cmd = [rocmlir_driver_path, '-c']
        driver_result = subprocess.run(driver_cmd,
                                       input=gen_result.stdout,
                                       capture_output=True,
                                       text=True,
                                       timeout=120)
        if driver_result.returncode != 0:
            return None, f"rocmlir-driver failed: {driver_result.stderr}"

        # Run with mlir-runner
        runner_cmd = [
            runner_path,
            '-O2',
            f'--shared-libs={libs}',
            '--entry-point-result=void',
        ]
        runner_result = subprocess.run(runner_cmd,
                                       input=driver_result.stdout,
                                       capture_output=True,
                                       text=True,
                                       timeout=120)
        if runner_result.returncode != 0:
            return None, f"mlir-runner failed: {runner_result.stderr}"

        return runner_result.stdout, None

    except subprocess.TimeoutExpired:
        return None, "rocmlir pipeline timed out"
    except Exception as e:
        return None, str(e)


def compare_results(hipblaslt_values, rocmlir_values, tolerance=0.01):
    """Compare two lists of float values with tolerance."""
    if len(hipblaslt_values) != len(rocmlir_values):
        return False, f"Size mismatch: {len(hipblaslt_values)} vs {len(rocmlir_values)}"

    if len(hipblaslt_values) == 0:
        return False, "No values to compare"

    max_rel_diff = 0.0
    max_abs_diff = 0.0
    num_mismatches = 0

    for i, (h, r) in enumerate(zip(hipblaslt_values, rocmlir_values)):
        abs_diff = abs(h - r)
        max_abs_diff = max(max_abs_diff, abs_diff)

        # Relative difference
        denom = max(abs(r), 1e-8)
        rel_diff = abs_diff / denom
        max_rel_diff = max(max_rel_diff, rel_diff)

        if rel_diff > tolerance:
            num_mismatches += 1

    passed = num_mismatches == 0 or (num_mismatches / len(hipblaslt_values) < 0.001)

    return passed, {
        'max_abs_diff': max_abs_diff,
        'max_rel_diff': max_rel_diff,
        'num_mismatches': num_mismatches,
        'total': len(hipblaslt_values),
    }


def main():
    parser = argparse.ArgumentParser(description='Verify hipblaslt vs rocmlir-gen')
    parser.add_argument('--hipblaslt-path',
                        required=True,
                        help='Path to hipblaslt-benchmark-driver')
    parser.add_argument('--rocmlir-gen-path', required=True, help='Path to rocmlir-gen')
    parser.add_argument('--rocmlir-driver-path', required=True, help='Path to rocmlir-driver')
    parser.add_argument('--runner-path', required=True, help='Path to mlir-runner')
    parser.add_argument('--libs', required=True, help='Comma-separated shared libraries')
    parser.add_argument('-arch',
                        '--arch',
                        required=True,
                        help='GPU architecture (e.g., gfx90a, gfx942)')
    parser.add_argument('--tolerance',
                        type=float,
                        default=0.01,
                        help='Relative tolerance (default: 0.01)')

    # GEMM parameters
    parser.add_argument('-m', type=int, default=32, help='M dimension')
    parser.add_argument('-n', type=int, default=32, help='N dimension')
    parser.add_argument('-k', type=int, default=32, help='K dimension')
    parser.add_argument('-g', type=int, default=0, help='Batch count (0 for non-batched)')
    parser.add_argument('-t', '--dtype', default='f32', help='Data type')
    parser.add_argument('--transA', default='False', help='Transpose A')
    parser.add_argument('--transB', default='False', help='Transpose B')

    args = parser.parse_args()

    # Run hipblaslt
    hipblaslt_out, hipblaslt_err = run_hipblaslt(args.hipblaslt_path, args.m, args.n, args.k,
                                                 args.g, args.dtype, args.transA, args.transB)

    if hipblaslt_err:
        print(f"HIPBLASLT ERROR: {hipblaslt_err}")
        sys.exit(1)

    # Run rocmlir-gen
    rocmlir_out, rocmlir_err = run_rocmlir_gen(args.rocmlir_gen_path, args.rocmlir_driver_path,
                                               args.runner_path, args.libs, args.m, args.n, args.k,
                                               args.g, args.dtype, args.transA, args.transB,
                                               args.arch)

    if rocmlir_err:
        print(f"ROCMLIR ERROR: {rocmlir_err}")
        sys.exit(1)

    # Parse outputs
    hipblaslt_values = parse_tensor_output(hipblaslt_out)
    rocmlir_values = parse_tensor_output(rocmlir_out)

    # Compare
    passed, result = compare_results(hipblaslt_values, rocmlir_values, args.tolerance)

    if passed:
        print(f"PASSED: m={args.m} n={args.n} k={args.k} g={args.g} dtype={args.dtype}")
        if isinstance(result, dict):
            print(f"  max_rel_diff={result['max_rel_diff']:.2e}")
        sys.exit(0)
    else:
        print(f"FAILED: m={args.m} n={args.n} k={args.k} g={args.g} dtype={args.dtype}")
        if isinstance(result, dict):
            print(f"  max_abs_diff: {result['max_abs_diff']:.6e}")
            print(f"  max_rel_diff: {result['max_rel_diff']:.6e}")
            print(f"  mismatches: {result['num_mismatches']}/{result['total']}")
        else:
            print(f"  Error: {result}")
        sys.exit(1)


if __name__ == '__main__':
    main()
