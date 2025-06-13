#!/usr/bin/env python3
"""Sweeps the parameters of the rocmlir driver for bugs for attention-based kernel configurations.

Usage:
    python3 attentionSweeps.py --mlir-build-dir <path-to-mlir-build-dir> [options]

Options:
    --mlir-build-dir    Path to the MLIR build directory (required)
    --samples           Number of random configuration samples to the test (default: 1000)
    --jobs              Number of concurrent tests to run in parallel (default: 4)
    --debug             Enable debug output
    --quiet             Disable per-test result output
    --log-failures      Save failing configurations to csv file
"""
import argparse
import itertools
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union, Iterable, Callable, Tuple, List, TypeVar
from enum import Enum
from datetime import datetime
import sys
import csv
import random
import subprocess
import os
import re

from perfRunner import AttentionConfiguration
from perfRunner import Paths
from perfRunner import getArch
from perfRunner import getNumCU
from perfRunner import create_paths as createPaths
from perfRunner import MLIR_N_REPEATS

# GLOBAL VARIABLES
DATA_TYPES_ATTENTION = ['i8', 'f32', 'f16', 'bf16']
BOOLS = [True, False]
LOGFILE = 'failing_configs.csv'
GFX_CHIP_RE = re.compile(r"gfx[0-9a-z]+")
CURRENT_SEQ_LEN = None

# Week number is used as seed to make sure weekly CI is reproducible
seed = datetime.utcnow().isocalendar()[1]
random.seed(seed)

class TestResult(Enum):
    PASS = 1
    INVALID = 2
    FAIL = 3
    
@dataclass(frozen=True)
class Options:
    """Class for keeping option state for the sweep."""
    debug: bool
    quiet: bool
    arch: str
    flags: list
    concurrentTests: int
    numCu: int


def generateMlirDriverArgs(self, rocmlir_gen_flags: Optional[List[str]] = None) -> List[str]:
    result = [
        '-operation', 'attention',
        '-t', self.dataType,
        '--arch', self.arch,
        '--num_cu', str(self.numCU),
        '-g', str(self.g),
        '-seq_len_q', str(self.seq_len_q),
        '-seq_len_k', str(self.seq_len_k),
        '-num_heads_q', str(self.num_heads_q),
        '-num_heads_kv', str(self.num_heads_kv),
        '-head_dim_qk', str(self.head_dim_qk),
        '-head_dim_v', str(self.head_dim_v),
        f"-with-attn-scale={self.with_attn_scale}",
        f"-with-attn-bias={self.with_attn_bias}",
        f"-transQ={self.transQ}",
        f"-transK={self.transK}",
        f"-transV={self.transV}",
        f"-transO={self.transO}",
        f"-causal={self.causal}",
        f"-return_lse={self.return_lse}",
        '--kernel-repeats', str(MLIR_N_REPEATS),
        f"--perf_config={self.perfConfig}"
    ]
    
    result += rocmlir_gen_flags or []
    return result

AttentionConfiguration.generateMlirDriverArgs = generateMlirDriverArgs


def toAttentionConfig(params, options: Options) -> AttentionConfiguration:
    """Converts a sampled parameter tuple into a AttentionConfiguration instance."""
    shape, perf = params
    dtype, g, slq, slk, hdqk, hdv, scale, bias, tq, tk, tv, to, causal, rlse = shape
    perfString = f"attn:v1:{','.join(str(x) for x in perf)}"
    return AttentionConfiguration(
        dtype=dtype,
        g=g,
        seq_len_q=slq,
        seq_len_k=slk,
        num_heads_q=hdqk,
        num_heads_kv=hdv,
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
        perfConfig=perfString
    )


async def testAttentionConfig(config: AttentionConfiguration, options: Options, paths) -> TestResult:
    """Runs the given configuration and returns whether it successfully concluded,
    failed validation, or was inapplicable."""
    global CURRENT_SEQ_LEN

    mlirGenOpts = config.generateMlirDriverArgs(options.flags)
    if CURRENT_SEQ_LEN is not None:
        mlirGenOpts.append(f"--current_seq_len={','.join(map(str, CURRENT_SEQ_LEN))}")
    mlirGenOpts.append('-pv')

    proc1 = await asyncio.create_subprocess_exec(
        paths.mlir_paths.rocmlir_gen_path,
        *mlirGenOpts,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    rocmlirGeneratorOutput, rocmlirGeneratorError = await proc1.communicate()

    if proc1.returncode !=0:
        if options.debug:
            print("rocmlir-gen failed:\nError = ", rocmlirGeneratorError.decode().strip())
        return TestResult.FAIL

    proc2 = await asyncio.create_subprocess_exec(
        paths.mlir_paths.rocmlir_driver_path,
        '-c',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    rocmlirDriverOutput, rocmlirDriverError = await proc2.communicate(input=rocmlirGeneratorOutput)

    if proc2.returncode !=0:
        if options.debug:
            print("rocmlir-driver failed:\nError = ", rocmlirDriverError.decode().strip())
        return TestResult.INVALID
    
    proc3 = subprocess.Popen(
        [
            paths.mlir_paths.cpu_runner_path,
            '-O2',
            f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},'
            f'{paths.mlir_paths.libconv_validation_wrappers_path},'
            f'{paths.mlir_paths.libmlir_runtime_utils_path}',
            '--entry-point-result=void'
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        stdout3, stderr3 = proc3.communicate(input=rocmlirDriverOutput, timeout=900)
    except BrokenPipeError:
        if options.debug:
            print("Broken pipe: cpu-runner failed to read input")
        return TestResult.FAIL
    except subprocess.TimeoutExpired:
        proc3.kill()
        if options.debug:
            print("TimeoutExpired: cpu-runner timed out")
        return TestResult.FAIL

    if proc3.returncode != 0:
        if options.debug:
            print("Runner failed:\nOutput = ", stdout3.decode().strip())
            print("\nError = ", stderr3.decode().strip())
        
        if 'hipErrorOutOfMemory' in stderr3.decode().strip():
            if options.debug:
                print("\n---> Classified as INVALID since the reason is memory access fault")
            return TestResult.INVALID

        return TestResult.FAIL

    output = stdout3.decode()

    if 'FAILED' in output or 'nan' in output.lower():
        if options.debug:
            print("FAILED in output or NaN:\nOutput = ", output)
        return TestResult.FAIL
    return TestResult.PASS


IterType = TypeVar('IterType')
def grouper(iterable: Iterable[IterType], n: int):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


async def dropGoodConfig(config: AttentionConfiguration,
        options: Options, paths: Paths) -> Union[TestResult, AttentionConfiguration]:
    """Test the given `params`, returning the corresponding `config` on failure
    and `None` on success or inapplicability."""
    result = await testAttentionConfig(config, options, paths)
    if not options.quiet:
        print(f"{result.name}: {config!r}") if CURRENT_SEQ_LEN is None else print(f"{result.name}: {config!r} - with KV-Cache")

    return config if result == TestResult.FAIL else result


async def sweepParameters(paramIter: Iterable[IterType],
        toConfig: Callable[[IterType, Options], AttentionConfiguration],
        options: Options, paths: Paths) -> Tuple[int, int, List[AttentionConfiguration]]:
    """Iterates over sampled parameter combinations, runs tests and returns passed and
      invalid count and list of failing configs."""
    failingConfigs = []
    passed = 0
    invalid = 0
    configs = (c for c in (toConfig(p, options) for p in paramIter))
    for configs in grouper((dropGoodConfig(c, options, paths) for c in configs),
            options.concurrentTests):
        configsFuture = asyncio.gather(*configs)
        try:
            configsResults = await configsFuture
        except Exception as e:
            configsFuture.cancel()
            raise e
        for result in configsResults:
            if result == TestResult.PASS:
                passed = passed + 1
            elif result == TestResult.INVALID:
                invalid = invalid + 1
            else:
                failingConfigs.append(result)

    return (passed, invalid, failingConfigs)


def genCurrentSeqLens(g: int, maxSeqLen: int) -> list[int]:
    return [random.randint(0, maxSeqLen-1) for _ in range(g)]


def sampleAttentionShape():
    global CURRENT_SEQ_LEN

    g = random.randint(1, 256) # GROUPS
    seqLenK = random.randint(1, 16384) # SEQ_LEN_K 

    useKVCache = random.choice(BOOLS)
    CURRENT_SEQ_LEN = genCurrentSeqLens(g, seqLenK) if useKVCache else None
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
    )


perfConfigSpace = list(itertools.product(
        [32, 64, 128, 256], # M/block G0
        [32, 64, 128, 256], # M/block G1
        [32, 64, 128, 256], # N/block G0
        [8, 16, 32, 64], # Kpack/Block
        [32, 64, 128, 256], # M/Wave
        [4, 16, 32], # MN/Xdl
        [4, 8, 16], # kPack
        [0, 1] # forceUnroll
    ))


def logFailingConfigs(configs: List[AttentionConfiguration], filename: str):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CommandLine'])
        for config in configs:
            writer.writerow([' '.join(config.generateMlirDriverArgs([]))])


def main():
    parser = argparse.ArgumentParser(
            description='Sweep parameter values for attention to detect bugs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--jobs', type=int, default=os.cpu_count())
    parser.add_argument('--mlir-build-dir', type=str, required=True)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--log-failures', action='store_true')

    args = parser.parse_args()
    arch = ','.join(getArch())
    chip = GFX_CHIP_RE.search(arch).group(0)
    paths = createPaths(None, args.mlir_build_dir)
    options = Options(
        debug=args.debug,
        quiet=args.quiet,
        arch=arch,
        flags=[],
        concurrentTests=args.jobs,
        numCu=getNumCU(chip)
    )
   

    if not args.quiet:
        print(f"Sampling {args.samples} configurations from attention space...")

    samples = [
        (sampleAttentionShape(), random.choice(perfConfigSpace))
        for _ in range(args.samples)
    ]

    passed, invalid, failing = asyncio.run(sweepParameters(samples, toAttentionConfig, options, paths))
    if CURRENT_SEQ_LEN is not None:
        print(f"\nCurrent_seq_len in this run: --current_seq_len={','.join(map(str, CURRENT_SEQ_LEN))}")
    print(f"Passed: {passed}, Invalid: {invalid}, Failed: {len(failing)}")
    if failing:
        print("\n*** Failing Configurations ***")
        for fail in failing:
            print(' '.join(fail.generateMlirDriverArgs([])))
        if args.log_failures:
            logFailingConfigs(failing, LOGFILE)
    
    return 0


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
