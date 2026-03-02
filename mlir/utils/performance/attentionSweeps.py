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
from datetime import datetime
import sys
import csv
import random
import os

from perfRunner import AttentionConfiguration
from perfRunner import get_arch, get_num_cu, get_num_chiplets, initialize_dtypes_attn
from perfRunner import create_paths
from perfRunner import find_mlir_build_dir
from perfRunner import GFX_CHIP_RE
from parameterSweeps import Options, sweep_parameters, multiline_repr

# GLOBAL VARIABLES
DATA_TYPES_ATTENTION = initialize_dtypes_attn()
BOOLS = [True, False]
MAX_TOKENS = 64 * 64  # temporarily hardcoded
SPLIT_KV_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]

# Week number is used as seed to make sure weekly CI is reproducible
seed = datetime.utcnow().isocalendar()[1]
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


def sample_attn_shape():
    g = random.randint(1, 256)  # GROUPS
    seqlen_k = random.randint(1, 16384)  # SEQ_LEN_K

    use_kvcache = random.choice(BOOLS)
    current_seqlen = gen_current_seqlens(g, seqlen_k) if use_kvcache else None
    seqlen_q = 1 if use_kvcache else random.randint(1, 16384)  # SEQ_LEN_Q

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
        random.randint(1, 1024),  # HEAD_DIM_QK
        random.randint(1, 1024),  # HEAD_DIM_V
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


# Keep in sync with RockTuningImpl.cpp
perfconfig_space_mfma = list(
    itertools.product(  # MFMA perfConfig space
        [16, 32, 64, 128, 256],  # M/block G0
        [16, 32, 64, 128, 256],  # M/block G1
        [16, 32, 64, 128, 256],  # N/block G0
        [8, 16, 32, 64],  # Kpack/Block
        [16, 32, 64, 128, 256],  # M/Wave
        [16, 32, 64, 128, 256],  # N/Wave
        [4, 16, 32],  # MN/Xdl
        [4, 8, 16],  # kPack
        [1],  # splitKFactor
        [1, 2, 3, 4],  # scheduleVersion
        [0, 1, 2],  # outputSwizzle
        [0, 1, 2, 4, 8],  # wavesPerEU
        [0, 1]  # forceUnroll
    ))

perfconfig_space_wmma = list(
    itertools.product(  # WMMA perfConfig space
        [16, 32, 64, 128],  # M/block G0
        [16, 32, 64, 128],  # M/block G1
        [16, 32, 64, 128, 256],  # N/block G0
        [8, 16, 32, 64],  # Kpack/Block
        [16, 32, 64],  # M/Wave
        [16, 32, 64],  # N/Wave
        [0],  # MN/Xdl
        [4, 8, 16],  # kPack
        [1],  # splitKFactor
        [1, 2, 3, 4],  # scheduleVersion
        [0, 1, 2],  # outputSwizzle
        [0, 1, 2, 4, 8, 16],  # wavesPerEU
        [0, 1]  # forceUnroll
    ))


def log_failing_configs(configs: List[AttentionConfiguration], filename: str):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CommandLine'])
        for config in configs:
            writer.writerow([config.generate_mlir_driver_commandline('', kernel_repeats=None)])


def run_attention_sweep(args, options, paths, chip):
    # TODO: use AmdArchDb python version when available

    if chip.startswith('gfx9'):
        perfconfig_space = perfconfig_space_mfma
    else:
        perfconfig_space = perfconfig_space_wmma

    samples = [(sample_attn_shape(), random.choice(perfconfig_space)) for _ in range(args.samples)]

    # Filter out samples that exceed MAX_TOKENS
    filtered_samples = [
        s for s in samples if _within_limit(s[0][1], s[0][2], s[0][3])  # g, slq, slk
    ]

    if not args.quiet:
        print(
            f"Filtered out {len(samples) - len(filtered_samples)} samples exceeding MAX_TOKENS={MAX_TOKENS}."
        )
        print(f"Proceeding with {len(filtered_samples)} samples.\n")

    samples = filtered_samples

    passed, invalid, failing = asyncio.run(sweep_parameters(samples, to_attn_config, options,
                                                            paths))

    target_valid = args.samples
    total_passed = 0
    total_invalid = 0
    total_failing = []

    while (total_passed + len(total_failing)) < target_valid:
        batch = [(sample_attn_shape(), random.choice(perfconfig_space))
                 for _ in range(args.samples - total_passed - len(total_failing))]
        batch = [
            s for s in batch if _within_limit(s[0][1], s[0][2], s[0][3])  # g, slq, slk
        ]

        p, i, f = asyncio.run(sweep_parameters(batch, to_attn_config, options, paths))
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
        description='Sweep parameter values for attention to detect bugs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug-fails', action='store_true')
    parser.add_argument('-j', '--jobs', type=int, default=os.cpu_count())
    parser.add_argument('--mlir-build-dir', type=str, default=find_mlir_build_dir())
    parser.add_argument('--samples', type=int, default=1000)
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
                      log_failures=args.log_failures)

    if not args.quiet:
        print(f"Sampling {args.samples} configurations from attention space...")

    return run_attention_sweep(args, options, paths, chip)


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
