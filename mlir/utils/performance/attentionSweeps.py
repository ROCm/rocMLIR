#!/usr/bin/env python3
"""Sweeps the parameters of the rocmlir driver for bugs for attention-based kernel configurations.

Usage:
    python3 attentionSweeps.py --mlir-build-dir <path-to-mlir-build-dir> [options]

Options:
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
from typing import Iterable, List, Optional, TypeVar
from dataclasses import replace
from datetime import datetime, timezone
import sys
import csv
import random
import os

from perfRunner import AttentionConfiguration
from perfRunner import get_arch, get_num_cu, get_num_chiplets, initialize_dtypes_attn
from perfRunner import create_paths
from perfRunner import find_mlir_build_dir
from perfRunner import GFX_CHIP_RE
from perfRunner import hip, hip_check
from parameterSweeps import (
    Options,
    sweep_parameters,
    multiline_repr,
    infer_codegen_flags_from_arch,
    get_codegen_flags_for_codepath,
)

# GLOBAL VARIABLES
DATA_TYPES_ATTENTION = initialize_dtypes_attn()
BOOLS = [True, False]
MAX_TOKENS = 64 * 64  # temporarily hardcoded
SPLIT_KV_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]
# When splitKV > 1, attention materializes extra partial output/LSE buffers.
# In the sweep generator we filter out samples with estimated splitKV extra storage
# above a conservative budget to avoid generating configs that are likely to
# OOM/time out late in the gen->driver->run pipeline.
#
# Default policy: use ~12.5% of VRAM (deviceMem/8) as a conservative budget, then
# clamp to keep behavior stable across very small/very large devices. If device
# memory cannot be queried, fall back to a mid-range default.
DEFAULT_SWEEP_LIMIT_BYTES = 1536 * 1024 * 1024
MIN_DYNAMIC_SWEEP_LIMIT_BYTES = 1024 * 1024 * 1024
MAX_DYNAMIC_SWEEP_LIMIT_BYTES = 8192 * 1024 * 1024
DYNAMIC_SWEEP_LIMIT_DIVISOR = 8
DTYPE_BYTES = {
    'f16': 2,
    'bf16': 2,
    'f32': 4,
    'i8': 1,
}
# LSE is stored as f32 in the attention API/contracts, so 4 bytes per element.
LSE_ELEM_BYTES = 4
# TODO: Keep these sweep bounds and perf options in sync with attention tuning
# search space in mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp
# (createGemmGemmTuningRangeBF).
MAX_SEQ_LEN = 16384
MAX_HEAD_DIM = 1024

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


def _parse_nonnegative_int64(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    # Bound to int64 to catch accidental huge values.
    if parsed > (1 << 63) - 1:
        raise argparse.ArgumentTypeError("value exceeds int64 max")
    return parsed


def _query_visible_device_memory_bytes() -> Optional[int]:
    try:
        device_count = hip_check(hip.hipGetDeviceCount())
    except RuntimeError:
        return None

    if device_count <= 0:
        return None

    try:
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))
    except RuntimeError:
        return None

    device_memory = getattr(props, 'totalGlobalMem', None)
    if device_memory is None:
        return None

    try:
        device_memory = int(device_memory)
    except (TypeError, ValueError):
        return None

    if device_memory <= 0:
        return None
    return device_memory


def _get_default_sweep_limit_bytes() -> int:
    device_memory = _query_visible_device_memory_bytes()
    if device_memory is None:
        return DEFAULT_SWEEP_LIMIT_BYTES

    dynamic_limit = device_memory // DYNAMIC_SWEEP_LIMIT_DIVISOR
    return max(MIN_DYNAMIC_SWEEP_LIMIT_BYTES,
               min(MAX_DYNAMIC_SWEEP_LIMIT_BYTES, dynamic_limit))


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


def _infer_instruction_set(arch: str, requested: str) -> str:
    if requested in ('mfma', 'wmma'):
        return requested

    codepath, _ = infer_codegen_flags_from_arch(arch)
    if codepath == 'unknown':
        raise RuntimeError(f"Unknown arch for attention sweep: {arch}")
    if codepath == 'vanilla':
        raise RuntimeError(f"Unsupported attention codepath '{codepath}' for arch {arch}. "
                           "Attention sweep requires MFMA or WMMA.")
    return codepath


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


def _estimate_splitkv_extra_bytes(shape_sample: tuple) -> Optional[int]:
    dtype, g, seq_len_q, _seq_len_k, num_heads_q, _num_heads_kv, _head_dim_qk, head_dim_v, _scale, _bias, _tq, _tk, _tv, _to, _causal, return_lse, split_kv, _current_seqlen = shape_sample
    if split_kv <= 1:
        return 0

    # DTYPE_BYTES maps the attention datatype to bytes per element so we can
    # estimate extra storage in bytes.
    out_elem_bytes = DTYPE_BYTES.get(dtype)
    if out_elem_bytes is None:
        return None

    extra_factor = split_kv - 1
    batch_heads = g * num_heads_q
    base = batch_heads * seq_len_q
    extra_out = base * head_dim_v * out_elem_bytes * extra_factor
    extra_lse = base * LSE_ELEM_BYTES * extra_factor if return_lse else 0
    return extra_out + extra_lse


def sample_attention_batch(batch_size: int, instruction_set: str, flags: list[str],
                           splitkv_limit_bytes: int):
    filtered_samples = []
    filtered_counts = {'max_tokens': 0, 'splitkv_extra': 0}
    for _ in range(batch_size):
        sample = sample_attention_case(instruction_set, flags)
        if _within_limit(sample[0][1], sample[0][2], sample[0][3]):  # g, slq, slk
            maybe_extra_bytes = _estimate_splitkv_extra_bytes(sample[0])
            if maybe_extra_bytes is None or maybe_extra_bytes <= splitkv_limit_bytes:
                filtered_samples.append(sample)
            else:
                filtered_counts['splitkv_extra'] += 1
        else:
            filtered_counts['max_tokens'] += 1
    return filtered_samples, filtered_counts


def log_failing_configs(configs: List[AttentionConfiguration], filename: str):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CommandLine'])
        for config in configs:
            writer.writerow([config.generate_mlir_driver_commandline('', kernel_repeats=None)])


def run_attention_sweep(args, options, paths, chip):
    # TODO: use AmdArchDb python version when available
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
    splitkv_limit_bytes = (args.splitkv_extra_bytes_limit
                           if args.splitkv_extra_bytes_limit is not None else
                           _get_default_sweep_limit_bytes())
    if not args.quiet:
        print(f"SplitKV sweep extra-storage limit: {splitkv_limit_bytes} bytes")

    samples, filtered_counts = sample_attention_batch(args.samples, instruction_set, rocmlir_gen_flags,
                                                      splitkv_limit_bytes)
    cumulative_filtered_counts = dict(filtered_counts)

    if not args.quiet:
        print(f"Filtered out {filtered_counts['max_tokens']} samples exceeding MAX_TOKENS={MAX_TOKENS}.")
        print(
            f"Filtered out {filtered_counts['splitkv_extra']} samples exceeding splitKV extra-storage limit.")
        print(f"Proceeding with {len(samples)} initial samples.\n")

    passed, invalid, failing = asyncio.run(
        sweep_parameters(samples, to_attn_config, sweep_options, paths))

    total_passed = passed
    total_invalid = invalid
    total_failing = list(failing)

    while (total_passed + len(total_failing)) < args.samples:
        remaining_valid = args.samples - (total_passed + len(total_failing))
        batch_target = max(remaining_valid * 2, args.jobs if args.jobs else 1)
        batch, refill_filtered_counts = sample_attention_batch(batch_target, instruction_set,
                                                               rocmlir_gen_flags, splitkv_limit_bytes)
        cumulative_filtered_counts['max_tokens'] += refill_filtered_counts['max_tokens']
        cumulative_filtered_counts['splitkv_extra'] += refill_filtered_counts['splitkv_extra']
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

    if not args.quiet:
        print("Filtered samples summary:")
        print(f"  - MAX_TOKENS filter: {cumulative_filtered_counts['max_tokens']}")
        print(f"  - splitKV extra-storage filter: {cumulative_filtered_counts['splitkv_extra']}")

    print(f"\nPassed: {total_passed}, Invalid: {total_invalid}, Failed: {len(total_failing)}")

    return 1 if total_failing else 0


def main():
    parser = argparse.ArgumentParser(
        description='Sweep parameter values for attention to detect bugs')
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
                        help='Override attention codepath selection')
    parser.add_argument('--test-timeout-sec',
                        type=int,
                        default=600,
                        help='Per-config timeout in seconds (0 disables timeout)')
    parser.add_argument(
        '--splitkv-extra-bytes-limit',
        type=_parse_nonnegative_int64,
        default=None,
        help=("Max allowed estimated splitKV extra temporary storage (bytes). "
              "If unset, a device-based default is used (deviceMem/8 clamped to "
              "[1 GiB, 8 GiB], fallback 1.5 GiB)."))
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
    num_cu = get_num_cu(chip)
    paths = create_paths(None, args.mlir_build_dir)
    options = Options(debug_fails=args.debug_fails,
                      debug=args.debug,
                      quiet=args.quiet,
                      arch=arch,
                      flags=[],
                      concurrent_tests=args.jobs,
                      num_cu=num_cu,
                      num_chiplets=get_num_chiplets(chip, num_cu),
                      log_failures=args.log_failures,
                      test_timeout_sec=args.test_timeout_sec)

    if not args.quiet:
        print(f"Sampling {args.samples} configurations from attention space...")

    return run_attention_sweep(args, options, paths, chip)


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
