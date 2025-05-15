#!/usr/bin/env python3
"""Sweeps the parameters of the rocmlir driver for bugs for attention-based kernel configurations.

Usage:
    python3 attentionSweeps.py --mlir-build-dir <path-to-mlir-build-dir> [options]

Options:
    --mlir-build-dir    Path to the MLIR build directory (required)
    --samples           Number of random configuration samples to the test (default: 20)
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

# GLOBAL VARIABLES
DATA_TYPES_ATTENTION = ['i8', 'f32', 'f16', 'bf16']
#SEQ_LENGTHS = range(1, 16385) 
#HEAD_DIMS = range(1, 16385)
#GROUPS = range(1, 16385)
BOOLS = [True, False]
LOGFILE = 'failing_configs.csv'
GFX_CHIP_RE = re.compile(r"gfx[0-9a-z]+")


class PerfConfigVersion(Enum):
    '''Currently, attention uses only v1, so it's hardcoded in file at the moment,
    Enumeration for PerfConfigVersion is not removed in case we reintroduce higher
    perfConfig versions.'''
    V1 = 'v1'
    V2 = 'v2'
    V3 = 'v3'


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
    PerfConfigVersion: PerfConfigVersion


def toAttentionConfig(params, options: Options) -> AttentionConfiguration:
    """Converts a sampled parameter tuple into a AttentionConfiguration instance"""
    shape, perf = params
    dtype, g, slq, slk, hdqk, hdv, scale, bias, tq, tk, tv, to = shape
    perfString = f"attn:v1:{','.join(str(x) for x in perf)}"
    return AttentionConfiguration(
        dtype, g, slq, slk, hdqk, hdv, scale,
        bias, tq, tk, tv, to, options.arch,
        options.numCu, perfString
    )


async def testAttentionConfig(config: AttentionConfiguration, options: Options, paths) -> TestResult:
    """Runs the given configuration and returns whether it successfully concluded,
    failed validation, or was inapplicable."""
    mlirGenOpts = config.generateMlirDriverCommandLine(options.flags)
    mlirGenOpts.append('-pv')

    proc1 = await asyncio.create_subprocess_exec(
        paths.mlir_paths.rocmlir_gen_path, *mlirGenOpts,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    proc2 = await asyncio.create_subprocess_exec(
        paths.mlir_paths.rocmlir_driver_path, '-c',
        sdin=proc1.stdout, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    proc3 = await asyncio.create_subprocess_exec(
        paths.mlir_paths.cpu_runner_path,
        '-02',
        f'--shared-libs{paths.mlir_paths.libmlir_rocm_runtime_path},'
        f'{paths.mlir_paths.libconv_validation_wrappers_path},'
        f'{paths.mlir_paths.libmlir_runtime_utils_path}',
        '--entry-point-result=void',
        stdin=proc2.stdout,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout3, _ = await proc3.communicate()
    output = stdout3.decode('utf-8')
    
    if proc1.returncode not in [None, 0] or proc2.returncode not in [None, 0]:
        return TestResult.INVALID
    if proc3.returncode not in [None, 0]:
        if options.debug:
            print("Runner failed:", output)
            return TestResult.FAILED
    if 'FAILED' in output or 'nan' in output.lower():
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
    and `None` on success or inapplicability"""
    result = await testAttentionConfig(config, options, paths)
    if not options.quiet:
        print(f"{result.name}: {config!r}")
    return config if result == TestResult.FAIL else result


async def sweepParameters(paramIter: Iterable[IterType],
        toConfig: Callable[[IterType, Options], AttentionConfiguration],
        options: Options, paths: Paths) -> Tuple[int, int, List[AttentionConfiguration]]:
    """Iterates over sampled parameter combinations, runs tests and returns passed and
      invalid count and list of failing configs"""
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


def sampleAttentionShape():
    return (
        random.choice(DATA_TYPES_ATTENTION),
        random.randint(1, 16384), # GROUPS
        random.randint(1, 16384), # SEQ_LEN_Q
        random.randint(1, 16384), # SEQ_LEN_K
        random.randint(1, 16384), # HEAD_DIM_QK
        random.randint(1, 16384), # HEAD_DIM_V
        random.choice(BOOLS),   # with_attn_scale
        random.choice(BOOLS),   # with_attn_bias
        random.choice(BOOLS),   # transQ
        random.choice(BOOLS),   # transK
        random.choice(BOOLS),   # transV
        random.choice(BOOLS)    # transO
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
    with open(filename, mode='w', newLine='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CommandLine'])
        for config in configs:
            writer.writerow([' '.join(config.generateMlirDriverCommandLine([]))])


def main():
    parser = argparse.ArgumentParser(
            description='Sweep parameter values for attention to detect bugs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--mlir-build-dir', type=str, required=True)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--log-failures', action='store_true')

    args = parser.parse_args()
    print("args parsed")
    arch = ','.join(getArch())
    chip = GFX_CHIP_RE.search(arch).group(0)
    paths = createPaths(None, args.mlir_build_dir)
    print("paths ok")
    options = Options(
        debug=args.debug,
        quiet=args.quiet,
        arch=arch,
        flags=[],
        concurrentTests=args.jobs,
        numCu=getNumCU(chip),
        PerfConfigVersion=PerfConfigVersion.V1
    )
   
    print("options ok")

    if not args.quiet:
        print(f"Sampling {args.samples} configurations from attention space...")

    samples = [
        (sampleAttentionShape(), random.choice(perfConfigSpace))
        for _ in range(args.samples)
    ]

    print("and after sampliing?")
    passed, invalid, failing = asyncio.run(sweepParameters(samples, toAttentionConfig, options, paths))
    print("after sweep?")
    print(f"Passed: {passed}, Invalid: {invalid}, Failed: {len(failing)}")
    if failing:
        print("\n*** Failing Configurations ***")
        for fail in failing:
            print(' '.join(fail.generateMlirDriverCommandLine([])))
        if args.log_failures:
            logFailingConfigs(failing, LOGFILE)


if __name__ == '__main__':
    ret = main()
    sys.exit(int(not ret))
