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
from perfRunner import getArch, getNumCU, initializeDataTypesAttention
from perfRunner import create_paths as createPaths
from perfRunner import find_mlir_build_dir as findMlirBuildDir
from perfRunner import DATA_TYPES_ATTENTION, GFX_CHIP_RE
from parameterSweeps import Options, sweepParameters, multilineRepr

# GLOBAL VARIABLES
DATA_TYPES_ATTENTION = initializeDataTypesAttention()
BOOLS = [True, False]

# Week number is used as seed to make sure weekly CI is reproducible
seed = datetime.utcnow().isocalendar()[1]
random.seed(seed)

def toAttentionConfig(params, options: Options) -> AttentionConfiguration:
    """Converts a sampled parameter tuple into a AttentionConfiguration instance."""
    shape, perf = params
    *shapeParams, currentSeqLen = shape
    dtype, g, slq, slk, nhq, nhkv, hdqk, hdv, scale, bias, tq, tk, tv, to, causal, rlse = shapeParams
    perfString = f"attn:v1:{','.join(str(x) for x in perf)}"
    attnConfig = AttentionConfiguration(
        dtype=dtype,
        g=g,
        seq_len_q=slq,
        seq_len_k=slk,
        num_heads_q=nhq,
        num_heads_kv=nhkv,
        head_dim_qk=hdqk,
        head_dim_v=hdv,
        with_attn_scale=scale,
        with_attn_bias=bias,
        transQ=tq,
        transK=tk,
        transV=tv,
        transO=to,
        causal=causal,
        return_lse=rlse,
        arch=options.arch,
        numCU=options.numCu,
        perf_config=perfString
    )
    attnConfig.currentSeqLen = currentSeqLen
    return attnConfig

IterType = TypeVar('IterType')
def grouper(iterable: Iterable[IterType], n: int):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def genCurrentSeqLens(g: int, maxSeqLen: int) -> list[int]:
    return [random.randint(0, maxSeqLen-1) for _ in range(g)]

def sampleAttentionShape():
    g = random.randint(1, 256) # GROUPS
    seqLenK = random.randint(1, 16384) # SEQ_LEN_K 

    useKVCache = random.choice(BOOLS)
    currentSeqLen = genCurrentSeqLens(g, seqLenK) if useKVCache else None
    seqLenQ = 1 if useKVCache else random.randint(1, 16384) # SEQ_LEN_Q

    numHeadsQ = 1
    numHeadsKV = 1
    '''By default numHeadsQ and numHeadsKV are both 1. If numHeadsQ
    and numHeadsKV are equal GQA is disabled. Both values are powers
    of 2 typically. And numHeadsQ is divisible by numHeadsKV
    Here we decide randomly if we will use numHeadsQ and numHeadsKV
    different from the default values.
    
    Requirements:
        - numHeadsQ >= numHeadsKV
        - numHeadsQ % numHeadsKV == 0'''
    genNumHeads = random.choice(BOOLS)
    if genNumHeads:
        while True:
            numHeadsQ = 2**random.randint(1, 6)
            numHeadsKV = 2**random.randint(1, 6)

            if numHeadsQ > numHeadsKV and numHeadsQ%numHeadsKV == 0: # found valid case
                break

    return (
        random.choice(DATA_TYPES_ATTENTION),
        g, # GROUPS
        seqLenQ, # SEQ_LEN_Q
        seqLenK, # SEQ_LEN_K
        numHeadsQ, # NUM_HEADS_Q
        numHeadsKV, # NUM_HEADS_KV
        random.randint(1, 1024), # HEAD_DIM_QK
        random.randint(1, 1024), # HEAD_DIM_V
        random.choice(BOOLS),   # with_attn_scale
        random.choice(BOOLS),   # with_attn_bias
        random.choice(BOOLS),   # transQ
        random.choice(BOOLS),   # transK
        random.choice(BOOLS),   # transV
        random.choice(BOOLS),   # transO
        random.choice(BOOLS),   # causal
        random.choice(BOOLS),   # return_lse
        currentSeqLen
    )

# Keep in sync with RockTuningImpl.cpp
perfConfigSpaceMFMA = list(itertools.product( # MFMA perfConfig space
        [32, 64, 128, 256], # M/block G0
        [32, 64, 128, 256], # M/block G1
        [32, 64, 128, 256], # N/block G0
        [8, 16, 32, 64], # Kpack/Block
        [32, 64, 128, 256], # M/Wave
        [4, 16, 32], # MN/Xdl
        [4, 8, 16], # kPack
        [0, 1] # forceUnroll
    ))

perfConfigSpaceWMMA = list(itertools.product( # WMMA perfConfig space
        [32, 64, 128],         # M/block G0
        [32, 64, 128],         # M/block G1
        [32, 64, 128, 256],    # N/block G0
        [8, 16, 32, 64],       # Kpack/Block
        [32, 64],              # M/Wave
        [32, 64],              # N/Wave
        [4, 8, 16],            # kPack
        [0, 1]                 # forceUnroll
    ))

def logFailingConfigs(configs: List[AttentionConfiguration], filename: str):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CommandLine'])
        for config in configs:
            writer.writerow([config.generateMlirDriverCommandLine('', kernel_repeats=None)])
            
def main():
    parser = argparse.ArgumentParser(
            description='Sweep parameter values for attention to detect bugs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--jobs', type=int, default=os.cpu_count())
    parser.add_argument('--mlir-build-dir', type=str, default=findMlirBuildDir()),
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--log-failures', action='store_true')

    args = parser.parse_args()
    arch = getArch()
    chip_match = GFX_CHIP_RE.search(arch)
    if chip_match is None:
        raise RuntimeError(f"Could not find GFX chip in arch string: {arch}")
    chip = chip_match.group(0)
    paths = createPaths(None, args.mlir_build_dir)
    options = Options(
        debug=args.debug,
        quiet=args.quiet,
        arch=arch,
        flags=[],
        concurrent_tests=args.jobs,
        numCu=getNumCU(chip),
        logFailures=args.log_failures
    )
   

    if not args.quiet:
        print(f"Sampling {args.samples} configurations from attention space...")

    # TODO: use AmdArchDb python version when available
    
    if chip.startswith('gfx9'):
        perfConfigSpace = perfConfigSpaceMFMA
    else:
        perfConfigSpace = perfConfigSpaceWMMA

    samples = [
        (sampleAttentionShape(), random.choice(perfConfigSpace))
        for _ in range(args.samples)
    ]

    passed, invalid, failing = asyncio.run(sweepParameters(samples, toAttentionConfig, options, paths))
    if failing:
        print("\n" + "-" * 80)
        print(f"{'Failing Configurations':^80}\n")
        for fail in failing:
            print(multilineRepr(fail))
    
    print(f"\nPassed: {passed}, Invalid: {invalid}, Failed: {len(failing)}")
    
    return 0

if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
