#!/usr/bin/env python3
"""Sweeps the parameters of the rocmlir driver for bugs for attention and gemm+gemm kernel configurations.

Usage:
    python3 attentionSweeps.py --mlir-build-dir <path-to-mlir-build-dir> [options]

Options:
    --op                Which accelerated op to sweep: attention (default) or gemm_gemm
    --mlir-build-dir    Path to the MLIR build directory (default: auto-detected)
    --samples           Number of random configuration samples to the test (default: 1000)
    --jobs              Number of concurrent tests to run in parallel (default: os.cpu_count())
    --debug             Enable debug output
    --quiet             Disable per-test result output
    --log-failures      Save failing configurations to csv file
"""

from __future__ import annotations

import argparse
import itertools
import asyncio
from typing import Iterable, List, TypeVar
from dataclasses import replace
from datetime import datetime, timezone
import sys
import csv
import random
import os

from perfRunner import AttentionConfiguration, GemmGemmConfiguration
from perfRunner import get_arch, get_num_cu, get_num_chiplets, initialize_dtypes_attn
from perfRunner import create_paths
from perfRunner import find_mlir_build_dir
from perfRunner import GFX_CHIP_RE
from perfRunner import DATA_TYPES_GEMM_GEMM
from parameterSweeps import (
    Options,
    sweep_parameters,
    multiline_repr,
    get_codegen_flags_for_codepath,
)
from amd_arch_db import GemmFeatures, has_feature, lookup_arch_info

# GLOBAL VARIABLES
DATA_TYPES_ATTENTION = initialize_dtypes_attn()
BOOLS = [True, False]
MAX_TOKENS = 64 * 64  # temporarily hardcoded
SPLIT_KV_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]
# TODO: Keep these sweep bounds and perf options in sync with attention tuning
# search space in mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp
# (createGemmGemmTuningRangeBF).
MAX_SEQ_LEN = 16384
MAX_HEAD_DIM = 1024

# gemm+gemm sweep bounds. Matmuls are (g,m,k)x(g,k,n)->(g,m,n) then
# (g,m,n)x(g,n,o)->(g,m,o).
GEMM_GEMM_GROUP_CHOICES = [1, 2, 4, 8, 12, 16, 20, 32, 64, 128, 256]
GEMM_GEMM_DIM_CHOICES = [1, 2, 16, 32, 64, 77, 128, 216, 256, 384, 512, 768, 1024, 1280, 3072, 4096]
# Per-tensor element budget to bound device memory and runtime.
MAX_GEMM_GEMM_TENSOR_ELEMS = 64 * 1024 * 1024

MFMA_PERF_CONFIG_OPTIONS = {
    'm_per_block_g0': [16, 32, 64, 128, 256],
    'm_per_block_g1': [16, 32, 64, 128, 256],
    'n_per_block_g0': [16, 32, 64, 128, 256],
    'kpack_per_block': [8, 16, 32, 64],
    'm_per_wave': [16, 32, 64, 128, 256],
    'n_per_wave': [16, 32, 64, 128, 256],
    'mn_per_xdl': [4, 16, 32],
    'kpack': [4, 8, 16],
    'split_k_factor': [1],
    'output_swizzle': [0, 1, 2],
    'waves_per_eu': [0, 1, 2, 4, 8],
    'force_unroll': [0, 1],
}

WMMA_PERF_CONFIG_OPTIONS = {
    'm_per_block_g0': [16, 32, 64, 128],
    'm_per_block_g1': [16, 32, 64, 128],
    'n_per_block_g0': [16, 32, 64, 128, 256],
    'kpack_per_block': [8, 16, 32, 64],
    'm_per_wave': [16, 32, 64],
    'n_per_wave': [16, 32, 64],
    'mn_per_xdl': [16],
    'kpack': [4, 8, 16],
    'split_k_factor': [1],
    'output_swizzle': [0, 1, 2],
    'waves_per_eu': [0, 1, 2, 4, 8, 16],
    'force_unroll': [0, 1],
}

SCHEDULE_OPTIONS_BASE = [1, 2]
SCHEDULE_OPTIONS_DIRECT_TO_LDS = [3, 4]

# Week number is used as seed to make sure weekly CI is reproducible
seed = datetime.now(timezone.utc).isocalendar()[1]
random.seed(seed)


def to_attn_config(params, options: Options) -> AttentionConfiguration:
    """Converts a sampled parameter tuple into a AttentionConfiguration instance."""
    shape, perf = params
    *shape_params, current_seqlen = shape
    dtype, g, slq, slk, nhq, nhkv, hdqk, hdv, scale, bias, tq, tk, tv, to, causal, rlse, split_kv = shape_params
    perf_str = f"attn:v3:{','.join(str(x) for x in perf)}"
    attn_config = AttentionConfiguration(dtype=dtype,
                                         g=g,
                                         seq_len_q=slq,
                                         seq_len_k=slk,
                                         num_heads_q=nhq,
                                         num_heads_kv=nhkv,
                                         head_dim_qk=hdqk,
                                         head_dim_v=hdv,
                                         with_attn_scale=scale,
                                         with_attn_bias=bias,
                                         trans_q=tq,
                                         trans_k=tk,
                                         trans_v=tv,
                                         trans_o=to,
                                         causal=causal,
                                         return_lse=rlse,
                                         split_kv=split_kv,
                                         arch=options.arch,
                                         num_cu=options.num_cu,
                                         num_chiplets=options.num_chiplets,
                                         perf_config=perf_str)
    attn_config.current_seqlen = current_seqlen
    return attn_config


def to_gemm_gemm_config(params, options: Options) -> GemmGemmConfiguration:
    """Converts a sampled parameter tuple into a GemmGemmConfiguration instance."""
    shape, perf = params
    dtype, g, m, k, n, o, trans_a, trans_b, trans_c, trans_o = shape
    perf_str = f"attn:v3:{','.join(str(x) for x in perf)}"
    gemm_gemm_config = GemmGemmConfiguration(dtype=dtype,
                                             g=g,
                                             m=m,
                                             k=k,
                                             n=n,
                                             o=o,
                                             trans_a=trans_a,
                                             trans_b=trans_b,
                                             trans_c=trans_c,
                                             trans_o=trans_o,
                                             arch=options.arch,
                                             num_cu=options.num_cu,
                                             num_chiplets=options.num_chiplets,
                                             perf_config=perf_str)
    # gemm+gemm has no kvcache current_seqlen.
    gemm_gemm_config.current_seqlen = None
    return gemm_gemm_config


IterType = TypeVar('IterType')


def grouper(iterable: Iterable[IterType], n: int):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def gen_current_seqlens(g: int, max_seqlen: int) -> list[int]:
    return [random.randint(0, max_seqlen - 1) for _ in range(g)]


def _within_limit(g: int, slq: int, slk: int) -> bool:
    # Checks that the total token count stays within MAX_TOKENS.
    # Used both to filter generated samples and to derive per-group seq len bounds.
    return max(slq, slk) * g <= MAX_TOKENS


def sample_attn_shape():
    g = random.randint(1, 256)  # GROUPS
    # Keep generated shapes under the same budget checked by _within_limit:
    #   max(seq_len_q, seq_len_k) * g <= MAX_TOKENS
    # Therefore per-group sequence length is capped at floor(MAX_TOKENS / g),
    # then clamped by MAX_SEQ_LEN to respect the model upper bound.
    per_group_token_budget = MAX_TOKENS // g
    max_valid_seqlen = max(1, min(MAX_SEQ_LEN, per_group_token_budget))

    use_kvcache = random.choice(BOOLS)
    seqlen_k = random.randint(1, max_valid_seqlen)  # SEQ_LEN_K
    seqlen_q = 1 if use_kvcache else random.randint(1, max_valid_seqlen)  # SEQ_LEN_Q

    current_seqlen = gen_current_seqlens(g, seqlen_k) if use_kvcache else None

    num_heads_q = 1
    num_heads_kv = 1
    '''By default num_heads_q and num_heads_kv are both 1. If num_heads_q
    and num_heads_kv are equal GQA is disabled. Both values are powers
    of 2 typically. And num_heads_q is divisible by num_heads_kv
    Here we decide randomly if we will use num_heads_q and num_heads_kv
    different from the default values.

    Requirements:
        - num_heads_q >= num_heads_kv
        - num_heads_q % num_heads_kv == 0'''
    gen_num_heads = random.choice(BOOLS)
    if gen_num_heads:
        while True:
            num_heads_q = 2**random.randint(1, 6)
            num_heads_kv = 2**random.randint(1, 6)

            if num_heads_q > num_heads_kv and num_heads_q % num_heads_kv == 0:  # found valid case
                break

    split_kv = 1
    return_lse = random.choice(BOOLS)
    if return_lse:
        split_kv = random.choice(SPLIT_KV_OPTIONS)

    return (
        random.choice(DATA_TYPES_ATTENTION),
        g,  # GROUPS
        seqlen_q,  # SEQ_LEN_Q
        seqlen_k,  # SEQ_LEN_K
        num_heads_q,  # NUM_HEADS_Q
        num_heads_kv,  # NUM_HEADS_KV
        random.randint(1, MAX_HEAD_DIM),  # HEAD_DIM_QK
        random.randint(1, MAX_HEAD_DIM),  # HEAD_DIM_V
        random.choice(BOOLS),  # with_attn_scale
        random.choice(BOOLS),  # with_attn_bias
        random.choice(BOOLS),  # trans_q
        random.choice(BOOLS),  # trans_k
        random.choice(BOOLS),  # trans_v
        random.choice(BOOLS),  # trans_o
        random.choice(BOOLS),  # causal
        return_lse,
        split_kv,
        current_seqlen)


def _gemm_gemm_within_limit(g: int, m: int, k: int, n: int, o: int) -> bool:
    # Bound every tensor of both matmuls by MAX_GEMM_GEMM_TENSOR_ELEMS.
    tensors = (g * m * k, g * k * n, g * m * n, g * n * o, g * m * o)
    return max(tensors) <= MAX_GEMM_GEMM_TENSOR_ELEMS


def sample_gemm_gemm_shape(dtypes: list[str]):
    return (
        random.choice(dtypes),
        random.choice(GEMM_GEMM_GROUP_CHOICES),  # GROUPS
        random.choice(GEMM_GEMM_DIM_CHOICES),  # M
        random.choice(GEMM_GEMM_DIM_CHOICES),  # K
        random.choice(GEMM_GEMM_DIM_CHOICES),  # N
        random.choice(GEMM_GEMM_DIM_CHOICES),  # O
        random.choice(BOOLS),  # trans_a
        random.choice(BOOLS),  # trans_b
        random.choice(BOOLS),  # trans_c
        random.choice(BOOLS),  # trans_o
    )


def sample_gemm_gemm_case(dtypes: list[str], instruction_set: str, flags: list[str]):
    return (sample_gemm_gemm_shape(dtypes), sample_perf_config(instruction_set, flags))


def sample_gemm_gemm_batch(batch_size: int, dtypes: list[str], instruction_set: str,
                           flags: list[str]):
    filtered_samples = []
    filtered_out = 0
    for _ in range(batch_size):
        sample = sample_gemm_gemm_case(dtypes, instruction_set, flags)
        # sample[0] is (dtype, g, m, k, n, o, ...)
        _, g, m, k, n, o = sample[0][:6]
        if _gemm_gemm_within_limit(g, m, k, n, o):
            filtered_samples.append(sample)
        else:
            filtered_out += 1
    return filtered_samples, filtered_out


def _infer_instruction_set(arch: str, requested: str) -> str:
    if requested in ('mfma', 'wmma'):
        return requested

    features = lookup_arch_info(arch).default_features
    if has_feature(features, GemmFeatures.MFMA):
        return 'mfma'
    if has_feature(features, GemmFeatures.WMMA):
        return 'wmma'
    raise RuntimeError(f"Unsupported arch for attention sweep: {arch}. "
                       "Attention sweep requires MFMA or WMMA.")


def _resolve_codegen_flags(arch: str, instruction_set: str) -> list[str]:
    return get_codegen_flags_for_codepath(arch, instruction_set)


def _compute_schedule_options(flags: list[str]) -> list[int]:
    options = list(SCHEDULE_OPTIONS_BASE)
    if '-direct_to_lds_32b=on' in flags or '-direct_to_lds_128b=on' in flags:
        options.extend(SCHEDULE_OPTIONS_DIRECT_TO_LDS)
    return options


def sample_perf_config(instruction_set: str, flags: list[str]) -> tuple[int, ...]:
    options = MFMA_PERF_CONFIG_OPTIONS if instruction_set == 'mfma' else WMMA_PERF_CONFIG_OPTIONS
    schedule_options = _compute_schedule_options(flags)

    return (
        random.choice(options['m_per_block_g0']),
        random.choice(options['m_per_block_g1']),
        random.choice(options['n_per_block_g0']),
        random.choice(options['kpack_per_block']),
        random.choice(options['m_per_wave']),
        random.choice(options['n_per_wave']),
        random.choice(options['mn_per_xdl']),
        random.choice(options['kpack']),
        random.choice(options['split_k_factor']),
        random.choice(schedule_options),
        random.choice(options['output_swizzle']),
        random.choice(options['waves_per_eu']),
        random.choice(options['force_unroll']),
    )


def sample_attention_case(instruction_set: str, flags: list[str]):
    return (sample_attn_shape(), sample_perf_config(instruction_set, flags))


def sample_attention_batch(batch_size: int, instruction_set: str, flags: list[str]):
    filtered_samples = []
    filtered_out = 0
    for _ in range(batch_size):
        sample = sample_attention_case(instruction_set, flags)
        if _within_limit(sample[0][1], sample[0][2], sample[0][3]):  # g, slq, slk
            filtered_samples.append(sample)
        else:
            filtered_out += 1
    return filtered_samples, filtered_out


def log_failing_configs(configs: List[AttentionConfiguration], filename: str):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CommandLine'])
        for config in configs:
            writer.writerow([config.generate_mlir_driver_commandline('', kernel_repeats=None)])


def run_attention_sweep(args, options, paths, chip):
    try:
        instruction_set = _infer_instruction_set(options.arch, args.codepath)
    except RuntimeError as e:
        print(f"Skipping attention sweep: {e}")
        return 0

    rocmlir_gen_flags = _resolve_codegen_flags(options.arch, instruction_set)
    sweep_options = replace(options, flags=rocmlir_gen_flags)

    if not args.quiet:
        print(f"Attention codepath: {instruction_set.upper()} on {chip}")
        print(
            f"rocmlir-gen flags: {' '.join(rocmlir_gen_flags) if rocmlir_gen_flags else '(none)'}")

    samples, filtered_out = sample_attention_batch(args.samples, instruction_set, rocmlir_gen_flags)

    if not args.quiet:
        print(f"Filtered out {filtered_out} samples exceeding MAX_TOKENS={MAX_TOKENS}.")
        print(f"Proceeding with {len(samples)} initial samples.\n")

    passed, invalid, failing = asyncio.run(
        sweep_parameters(samples, to_attn_config, sweep_options, paths))

    total_passed = passed
    total_invalid = invalid
    total_failing = list(failing)

    while (total_passed + len(total_failing)) < args.samples:
        remaining_valid = args.samples - (total_passed + len(total_failing))
        batch_target = max(remaining_valid * 2, args.jobs if args.jobs else 1)
        batch, _ = sample_attention_batch(batch_target, instruction_set, rocmlir_gen_flags)
        if not batch:
            continue

        p, i, f = asyncio.run(sweep_parameters(batch, to_attn_config, sweep_options, paths))
        total_passed += p
        total_invalid += i
        total_failing.extend(f)

    if total_failing:
        print("\n" + "-" * 80)
        print(f"{'Failing Configurations':^80}\n")
        for fail in total_failing:
            print(multiline_repr(fail))

    print(f"\nPassed: {total_passed}, Invalid: {total_invalid}, Failed: {len(total_failing)}")

    return 1 if total_failing else 0


def run_gemm_gemm_sweep(args, options, paths, chip):
    try:
        instruction_set = _infer_instruction_set(options.arch, args.codepath)
    except RuntimeError as e:
        print(f"Skipping gemm+gemm sweep: {e}")
        return 0

    rocmlir_gen_flags = _resolve_codegen_flags(options.arch, instruction_set)
    sweep_options = replace(options, flags=rocmlir_gen_flags)

    # RDNA has no f32 WMMA.
    dtypes = list(DATA_TYPES_GEMM_GEMM)
    if instruction_set == 'wmma':
        dtypes = [dt for dt in dtypes if dt != 'f32']

    if not args.quiet:
        print(f"Gemm+gemm codepath: {instruction_set.upper()} on {chip}")
        print(f"Data types: {', '.join(dtypes)}")
        print(
            f"rocmlir-gen flags: {' '.join(rocmlir_gen_flags) if rocmlir_gen_flags else '(none)'}")

    samples, filtered_out = sample_gemm_gemm_batch(args.samples, dtypes, instruction_set,
                                                   rocmlir_gen_flags)

    if not args.quiet:
        print(f"Filtered out {filtered_out} samples exceeding "
              f"MAX_GEMM_GEMM_TENSOR_ELEMS={MAX_GEMM_GEMM_TENSOR_ELEMS}.")
        print(f"Proceeding with {len(samples)} initial samples.\n")

    passed, invalid, failing = asyncio.run(
        sweep_parameters(samples, to_gemm_gemm_config, sweep_options, paths))

    total_passed = passed
    total_invalid = invalid
    total_failing = list(failing)

    while (total_passed + len(total_failing)) < args.samples:
        remaining_valid = args.samples - (total_passed + len(total_failing))
        batch_target = max(remaining_valid * 2, args.jobs if args.jobs else 1)
        batch, _ = sample_gemm_gemm_batch(batch_target, dtypes, instruction_set, rocmlir_gen_flags)
        if not batch:
            continue

        p, i, f = asyncio.run(sweep_parameters(batch, to_gemm_gemm_config, sweep_options, paths))
        total_passed += p
        total_invalid += i
        total_failing.extend(f)

    if total_failing:
        print("\n" + "-" * 80)
        print(f"{'Failing Configurations':^80}\n")
        for fail in total_failing:
            print(multiline_repr(fail))

    print(f"\nPassed: {total_passed}, Invalid: {total_invalid}, Failed: {len(total_failing)}")

    return 1 if total_failing else 0


def main():
    parser = argparse.ArgumentParser(
        description='Sweep parameter values for attention or gemm+gemm to detect bugs')
    parser.add_argument('--op',
                        type=str,
                        default='attention',
                        choices=['attention', 'gemm_gemm'],
                        help='Which accelerated op to sweep')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug-fails', action='store_true')
    parser.add_argument('-j', '--jobs', type=int, default=(os.cpu_count() or 1))
    parser.add_argument('--mlir-build-dir', type=str, default=find_mlir_build_dir())
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--codepath',
                        type=str,
                        default='auto',
                        choices=['auto', 'mfma', 'wmma'],
                        help='Override the codepath (MFMA/WMMA) selection')
    parser.add_argument('--test-timeout-sec',
                        type=int,
                        default=600,
                        help='Per-config timeout in seconds (0 disables timeout)')
    parser.add_argument('--log-failures', action='store_true')

    args = parser.parse_args()

    # Set default mlir-build-dir if not provided
    if args.mlir_build_dir is None:
        args.mlir_build_dir = find_mlir_build_dir()

    arch = get_arch()
    chip_match = GFX_CHIP_RE.search(arch)
    if chip_match is None:
        raise RuntimeError(f"Could not find GFX chip in arch string: {arch}")
    chip = chip_match.group(0)
    num_cu = get_num_cu()
    paths = create_paths(None, args.mlir_build_dir)
    options = Options(debug_fails=args.debug_fails,
                      debug=args.debug,
                      quiet=args.quiet,
                      arch=arch,
                      flags=[],
                      concurrent_tests=args.jobs,
                      num_cu=num_cu,
                      num_chiplets=get_num_chiplets(),
                      log_failures=args.log_failures,
                      test_timeout_sec=args.test_timeout_sec)

    if not args.quiet:
        print(f"Sampling {args.samples} configurations from {args.op} space...")

    if args.op == 'gemm_gemm':
        return run_gemm_gemm_sweep(args, options, paths, chip)
    return run_attention_sweep(args, options, paths, chip)


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
