#!/usr/bin/env python3
"""Sweeps the parameters of the rocmlir driver for bugs for attention only.
"""
import argparse
import itertools
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union, Iterable, Callable, Tuple, List, TypeVar
from enum import Enum
import sys

from more_itertools import grouper

import perfRunner
from perfRunner import AttentionConfiguration
from perfRunner import Paths
from perfRunner import getArch


class Version(Enum):
    V2 = 2
    V3 = 4


class TestResult(Enum):
    PASS = 1
    INVALID = 2
    FAIL = 3
    

@dataclass(frozen=True)
class Options:
    """Class for keeping option state for the parameter sweep script."""
    debug: bool
    quiet: bool
    arch: str
    flags: list
    concurrent_tests: int


@dataclass
class PerfConfig:
    config: Sequence[int]
    version: Version = Version.V3
    version_map: dict = field(default_factory=lambda: {Version.V2: "v2", Version.V3: "v3"})

    def __str__(self):
        suffix = ",".join(str(v) for v in self.config)
        return f'{self.version_map[self.version]:{suffix}}'


@dataclass
class MLIROnlyAttentionConfig(AttentionConfiguration):
    attentionConfig: Optional[str] = None  # Optional additional attention configuration
 
    def generateMlirDriverCommandLine(self, mlir_gen_flags: Optional[Sequence[str]] = None) -> Sequence[str]:
        command = [
            '--operation', 'attention',
            '-t', self.dataType,
            '--arch', self.arch,
            '-g', str(self.g),
            '-seq_len_q', str(self.seq_len_q),
            '-seq_len_k', str(self.seq_len_k),
            '-head_dim_qk', str(self.head_dim_qk),
            '-head_dim_v', str(self.head_dim_v),
           # '-with_attn_scale', str(self.valueDim), # TODO: check if needed
            '-transQ', str(self.transQ),
            '-transK', str(self.transK),
            '-transV', str(self.transV),
            '-transO', str(self.transO),
        ]

        if mlir_gen_flags:
            command.extend(mlir_gen_flags)

        if self.perfConfig is not None:
            command.extend('--perf_config')
            command.extend(str(self.perfConfig))
    
        return command


async def testAttentionConfig(config: MLIROnlyAttentionConfig, options: Options, paths: Paths) -> TestResult:
    """Runs the given configuration and returns whether it successfully concluded,
    failed validation, or was inapplicable."""
    mlirGenOpts = config.generateMlirDriverCommandLine(options.flags)
    mlirGenOpts.append('-pv')


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
    if result == TestResult.FAIL:
        return config
    return result


async def sweepParameters(paramIter: Iterable[IterType],
        toConfig: Callable[[IterType, Options], MLIROnlyAttentionConfig],
        options: Options, paths: Paths) -> Tuple[int, int, List[MLIROnlyAttentionConfig]]:
    failingConfigs = []
    passed = 0
    invalid = 0
    configs = (c for c in (toConfig(p, options) for p in paramIter))
    for configs in grouper((dropGoodConfig(c, options, paths) for c in configs),
            options.concurrent_tests):
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

# TODO: check if relevant for attention parameterSweeps
def obtainGenFlags(codepath, arch):
    supportedCodepath = ['mfma', 'vanilla', 'wmma']
    # If codepath not provided or not supported, infer it from the arch
    rocmlirGenFlags = []
    if codepath not in supportedCodepath:
        if 'gfx908' in arch or 'gfx90a' in arch or 'gfx94' in arch:
            codepath = 'mfma'
            rocmlirGenFlags = ['-mfma=on', '-dot=on', '-atomic_add=on', '-atomic_add_f16=on']
        elif 'gfx95' in arch:
            codepath = 'mfma'
            rocmlirGenFlags = ['-mfma=on', '-dot=on', '-atomic_add=on', '-atomic_add_f16=on', '-atomic_add_bf16=on']
        elif 'gfx906' in arch:
            codepath = 'vanilla'
            rocmlirGenFlags = ['-mfma=off', '-dot=on', '-atomic_add=off']
        elif 'gfx1030' in arch:
            # Use vanilla codepath for gfx1030 until it has its own perf configs
            codepath = 'vanilla'
            rocmlirGenFlags = ['-mfma=off', '-dot=on', '-atomic_add=off']
        elif 'gfx11' in arch:
            codepath = 'wmma'
            rocmlirGenFlags = ['-mfma=off', '-dot=on', '-atomic_add=on', '-wmma=infer']
        elif 'gfx12' in arch:
            codepath = 'wmma'
            rocmlirGenFlags = ['-mfma=off', '-dot=on', '-atomic_add=on', '-wmma=infer', '-atomic_add_f16=on', '-atomic_add_bf16=on']
        else:
            # unknow arch info
            print(f"""Unknown arch {arch}""", file=sys.stderr)
    
    return rocmlirGenFlags


def main():
    parser = argparse.ArgumentParser(
            description='Sweep parameter values to check correctness of MLIR')
    parser.add_argument('config',
        help="The attention configuration to test")

    # TODO: add modes such as debug, quiet, ... 

    args = parser.parse_args()
    arch = ','.join(getArch())

    options = Options(debug=args.debug, quiet=args.quiet,
        arch=arch, flags=obtainGenFlags(args.codepath, arch), concurrent_tests=args.jobs)
    paths = perfRunner.create_paths(None, args.mlir_build_dir)

if __name__ == '__main__':
    ret = main()
    sys.exit(int(not ret))
