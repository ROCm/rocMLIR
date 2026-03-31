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
import argparse
import itertools
import asyncio
from typing import Iterable, List, TypeVar
from dataclasses import replace
from datetime import datetime, UTC
import sys
import csv
import random
import os

from perfRunner import AttentionConfiguration
from perfRunner import get_arch, get_num_cu, get_num_chiplets, initialize_dtypes_attn
from perfRunner import create_paths
from perfRunner import find_mlir_build_dir
from perfRunner import GFX_CHIP_RE
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
MAX_TOKENS = 16 * 1024  # temporarily hardcoded
SPLIT_KV_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]
MAX_SPLIT_KV = 16
MAX_VALIDATION_ATTEMPTS_MULTIPLIER = 20
GROUP_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEQ_LEN_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
NUM_HEAD_OPTIONS = [1, 2, 4, 8, 16, 32, 64]
HEAD_DIM_OPTIONS = [16, 32, 64, 128]

# Keep in sync with createGemmGemmTuningRangeBF in RockTuningImpl.cpp.
D_PER_BLOCK = [16, 32, 64, 128, 256]
KPACK_PER_BLOCK_OPTIONS = [2, 4, 8, 16, 32, 64]
KPACK_OPTIONS = [4, 8, 16]
MN_PER_XDL_OPTIONS = {
    'mfma': [4, 16, 32],
    'wmma': [16],
}
SCHEDULE_OPTIONS_BASE = [1, 2]
SCHEDULE_OPTIONS_DIRECT_TO_LDS = [3, 4]

# Week number is used as seed to make sure weekly CI is reproducible
seed = datetime.now(UTC).isocalendar()[1]
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
    return max(slq, slk) * g <= MAX_TOKENS


def _split_kv_cap_for_seq_len(seq_len_k: int) -> int:
    # Keep larger-sequence split-KV cases bounded to avoid long-running kernels.
    if seq_len_k >= 512:
        return min(MAX_SPLIT_KV, 2)
    if seq_len_k >= 256:
        return min(MAX_SPLIT_KV, 4)
    return MAX_SPLIT_KV


def sample_attn_shape():
    g = random.choice(GROUP_OPTIONS)  # GROUPS
    max_valid_seqlen = max(1, min(16384, MAX_TOKENS // g))
    valid_seq_len_options = [s for s in SEQ_LEN_OPTIONS if s <= max_valid_seqlen]
    if not valid_seq_len_options:
        valid_seq_len_options = [max_valid_seqlen]

    use_kvcache = random.choice(BOOLS)
    if use_kvcache:
        seqlen_k = random.choice(valid_seq_len_options)  # SEQ_LEN_K
        seqlen_q = 1
    else:
        non_kvcache_seq_options = [s for s in valid_seq_len_options if s >= 4]
        if not non_kvcache_seq_options:
            non_kvcache_seq_options = valid_seq_len_options
        seqlen_k = random.choice(non_kvcache_seq_options)  # SEQ_LEN_K
        seqlen_q = random.choice(non_kvcache_seq_options)  # SEQ_LEN_Q

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
            num_heads_q = random.choice(NUM_HEAD_OPTIONS)
            num_heads_kv = random.choice(NUM_HEAD_OPTIONS)

            if num_heads_q > num_heads_kv and num_heads_q % num_heads_kv == 0:  # found valid case
                break

    split_kv = 1
    return_lse = random.choice(BOOLS)
    if return_lse:
        split_kv_cap = _split_kv_cap_for_seq_len(seqlen_k)
        valid_split_kv_options = [
            v for v in SPLIT_KV_OPTIONS if v <= seqlen_k and v <= split_kv_cap
        ]
        split_kv = random.choice(valid_split_kv_options) if valid_split_kv_options else 1

    # Avoid currently unsupported combinations for split-KV causal masking in non-kv-cache mode.
    causal = random.choice(BOOLS)
    if return_lse and split_kv > 1 and not use_kvcache:
        causal = False

    return (
        random.choice(DATA_TYPES_ATTENTION),
        g,  # GROUPS
        seqlen_q,  # SEQ_LEN_Q
        seqlen_k,  # SEQ_LEN_K
        num_heads_q,  # NUM_HEADS_Q
        num_heads_kv,  # NUM_HEADS_KV
        random.choice(HEAD_DIM_OPTIONS),  # HEAD_DIM_QK
        random.choice(HEAD_DIM_OPTIONS),  # HEAD_DIM_V
        random.choice(BOOLS),  # with_attn_scale
        random.choice(BOOLS),  # with_attn_bias
        random.choice(BOOLS),  # trans_q
        random.choice(BOOLS),  # trans_k
        random.choice(BOOLS),  # trans_v
        random.choice(BOOLS),  # trans_o
        causal,
        return_lse,
        split_kv,
        current_seqlen)


def _infer_instruction_set(chip: str, arch: str, requested: str) -> str:
    if requested in ('mfma', 'wmma'):
        return requested

    codepath, _ = infer_codegen_flags_from_arch(arch, enable_gfx10_wmma=False)
    if codepath == 'unknown':
        raise RuntimeError(f"Unknown arch for attention sweep: {arch}")
    if codepath == 'vanilla':
        raise RuntimeError(
            f"Unsupported attention codepath '{codepath}' for arch {arch}. "
            "Attention sweep requires MFMA or WMMA.")
    return codepath


def _resolve_codegen_flags(arch: str, chip: str, instruction_set: str) -> list[str]:
    if instruction_set == 'wmma' and chip.startswith('gfx10'):
        # Navi 2x uses the WMMA path in attention sweeps.
        return ['-mfma=off', '-dot=on', '-atomic_add=on', '-wmma=infer']
    return get_codegen_flags_for_codepath(arch, instruction_set)


def _compute_m_per_block_g1_options(gemm0_m_per_block: int) -> list[int]:
    options = []
    m_per_block = gemm0_m_per_block
    while m_per_block <= D_PER_BLOCK[-1]:
        options.append(m_per_block)
        m_per_block *= 2
    return options


def _compute_d_per_wave_options(d_per_block: int) -> list[int]:
    options = []
    for factor in [1, 2, 4, 8, 16]:
        if d_per_block % factor != 0:
            continue
        d_per_wave = d_per_block // factor
        if d_per_wave >= 16 and d_per_wave <= 128:
            options.append(d_per_wave)
    return options


def _compute_schedule_options(flags: list[str]) -> list[int]:
    options = list(SCHEDULE_OPTIONS_BASE)
    if '-direct_to_lds_32b=on' in flags or '-direct_to_lds_128b=on' in flags:
        options.extend(SCHEDULE_OPTIONS_DIRECT_TO_LDS)
    return options


def sample_perf_config(instruction_set: str, flags: list[str]) -> tuple[int, ...]:
    schedule_options = _compute_schedule_options(flags)

    for _ in range(25):
        m_per_block_g0 = random.choice(D_PER_BLOCK)
        m_per_block_g1 = random.choice(_compute_m_per_block_g1_options(m_per_block_g0))
        n_per_block_g0 = random.choice(D_PER_BLOCK)
        m_per_wave = random.choice(_compute_d_per_wave_options(m_per_block_g0))
        n_per_wave = random.choice(_compute_d_per_wave_options(n_per_block_g0))

        if instruction_set == 'wmma':
            rdna_waves = (m_per_block_g0 // m_per_wave) * (n_per_block_g0 // n_per_wave)
            if rdna_waves < 4:
                continue

        return (
            m_per_block_g0,
            m_per_block_g1,
            n_per_block_g0,
            random.choice(KPACK_PER_BLOCK_OPTIONS),
            m_per_wave,
            n_per_wave,
            random.choice(MN_PER_XDL_OPTIONS[instruction_set]),
            random.choice(KPACK_OPTIONS),
            1,  # splitKFactor
            random.choice(schedule_options),
            2,  # outputSwizzle
            0,  # wavesPerEU
            1,  # forceUnroll
        )

    # Conservative fallback if constrained random retries are exhausted.
    return (64, 64, 64, 8, 32, 32, 16, 4, 1, 1, 2, 0, 1)


def sample_attention_case(instruction_set: str, flags: list[str]):
    return (sample_attn_shape(), sample_perf_config(instruction_set, flags))


def sample_attention_batch(batch_size: int, instruction_set: str, flags: list[str]):
    raw_samples = [sample_attention_case(instruction_set, flags) for _ in range(batch_size)]
    filtered_samples = [
        s for s in raw_samples if _within_limit(s[0][1], s[0][2], s[0][3])  # g, slq, slk
    ]
    return raw_samples, filtered_samples


def log_failing_configs(configs: List[AttentionConfiguration], filename: str):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CommandLine'])
        for config in configs:
            writer.writerow([config.generate_mlir_driver_commandline('', kernel_repeats=None)])


def run_attention_sweep(args, options, paths, chip):
    try:
        instruction_set = _infer_instruction_set(chip, options.arch, args.codepath)
    except RuntimeError as e:
        print(f"Skipping attention sweep: {e}")
        return 0

    rocmlir_gen_flags = _resolve_codegen_flags(options.arch, chip, instruction_set)
    sweep_options = replace(options, flags=rocmlir_gen_flags)

    if not args.quiet:
        print(f"Attention codepath: {instruction_set.upper()} on {chip}")
        print(
            f"rocmlir-gen flags: {' '.join(rocmlir_gen_flags) if rocmlir_gen_flags else '(none)'}"
        )

    raw_samples, samples = sample_attention_batch(args.samples, instruction_set, rocmlir_gen_flags)

    if not args.quiet:
        print(
            f"Filtered out {len(raw_samples) - len(samples)} samples exceeding MAX_TOKENS={MAX_TOKENS}."
        )
        print(f"Proceeding with {len(samples)} initial samples.\n")

    passed, invalid, failing = asyncio.run(
        sweep_parameters(samples, to_attn_config, sweep_options, paths))

    target_valid = args.samples
    total_passed = passed
    total_invalid = invalid
    total_failing = list(failing)
    drawn_configs = len(raw_samples)
    tested_configs = len(samples)
    max_attempts = args.samples * args.max_attempt_multiplier

    while (total_passed + len(total_failing)) < target_valid and drawn_configs < max_attempts:
        remaining_valid = target_valid - (total_passed + len(total_failing))
        batch_target = max(remaining_valid * 2, args.jobs if args.jobs else 1)
        raw_batch, batch = sample_attention_batch(batch_target, instruction_set, rocmlir_gen_flags)
        drawn_configs += len(raw_batch)
        if not batch:
            continue

        tested_configs += len(batch)
        p, i, f = asyncio.run(sweep_parameters(batch, to_attn_config, sweep_options, paths))
        total_passed += p
        total_invalid += i
        total_failing.extend(f)

    achieved_valid = total_passed + len(total_failing)
    if achieved_valid < target_valid:
        print(
            f"WARNING: Reached max attempts ({max_attempts}) with only "
            f"{achieved_valid}/{target_valid} valid samples.")

    if total_failing:
        print("\n" + "-" * 80)
        print(f"{'Failing Configurations':^80}\n")
        for fail in total_failing:
            print(multiline_repr(fail))

    print(
        f"\nPassed: {total_passed}, Invalid: {total_invalid}, Failed: {len(total_failing)}, "
        f"ValidSamples: {achieved_valid}/{target_valid}, Tested: {tested_configs}, Drawn: {drawn_configs}")

    return 1 if total_failing or achieved_valid < target_valid else 0


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
    parser.add_argument('--max-attempt-multiplier',
                        type=int,
                        default=MAX_VALIDATION_ATTEMPTS_MULTIPLIER,
                        help='Limit retries when too many sampled configs are invalid')
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
