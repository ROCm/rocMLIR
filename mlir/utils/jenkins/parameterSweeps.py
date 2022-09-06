#!/usr/bin/env python3
"""Script to sweep the parameters of the MIOpen driver for bugs

Note: This requires Python 3.7 or newer, use pyenv or the like to install it temporarily

Usage:
$ ninja miopen-gen mlir-miopen-driver mlir-rocm-runner ci-performance-scripts
$ stdbuf --output=L python3 ./bin/parameterSweeps.py [config] | stdbuf --output=L tee [output-file-of-choice]"""

import argparse
import asyncio
import enum
import itertools
import math
import re
import os
import sys

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Optional, Tuple, TypeVar, Union

import MIOpenDriver
from MIOpenDriver import ConvConfiguration
from MIOpenDriver import Paths

@dataclass(frozen=True)
class Options:
    """Class for keeping option state for the parameter sweep script."""
    debug: bool
    quiet: bool
    xdlops: bool
    concurrent_tests: int

class MLIROnlyConfig(ConvConfiguration):
    def __repr__(self):
        return f"""ConvConfiguration(dtype={self.dataType!r}, direction={self.direction!r}, layout={self.inputLayout.upper()!r},
                n={self.n!r}, c={self.c!r}, hi={self.hi!r}, wi={self.wi!r}, k={self.k!r}, y={self.y!r}, x={self.x!r},
                convStrideH={self.convStrideH!r}, convStrideW={self.convStrideW!r}, paddingHL={self.paddingHL!r}, paddingHR={self.paddingHR!r},
                paddingWL={self.paddingWL!r}, paddingWR={self.paddingWR!r}, dilationH={self.dilationH!r}, dilationW={self.dilationW!r},
                group={self.group!r}, xdlops={self.xdlops!r}, perfConfig={self.perfConfig!r})"""

    def generateMlirDriverCommandLine(self) -> Sequence[str]:
        direction = {'fwd': 'conv2d',
                     'bwd': 'conv2d_bwd_data',
                     'wrw':'conv2d_bwd_weight'}[self.direction]

        result = ['--operation', direction,
                    '-t', self.dataType,
                    '--fil_layout', self.filterLayout,
                    '--in_layout', self.inputLayout,
                    '--out_layout', self.outputLayout,
                    '--batchsize', str(self.n),
                    '--in_channels', str(self.c),
                    '--in_h', str(self.hi),
                    '--in_w', str(self.wi),
                    '--out_channels', str(self.k),
                    '--fil_h', str(self.y),
                    '--fil_w', str(self.x),
                    '--dilation_h', str(self.dilationH),
                    '--dilation_w', str(self.dilationW),
                    '--conv_stride_h', str(self.convStrideH),
                    '--conv_stride_w', str(self.convStrideW),
                    '--padding_h_l', str(self.paddingHL),
                    '--padding_h_r', str(self.paddingHR),
                    '--padding_w_l', str(self.paddingWL),
                    '--padding_w_r', str(self.paddingWR)]

        if self.xdlops:
            result.append('-x2')
        if self.perfConfig is not None:
            result.append('--perf_config')
            result.append(','.join(str(v) for v in self.perfConfig))

        return result

    def __init__(self, dtype: str, direction: str, layout: str,
                    n: int, c: int, hi: int, wi: int, k: int, y: int, x: int,
                    convStrideH: int, convStrideW: int,
                    paddingHL: int, paddingHR: int, paddingWL: int, paddingWR: int,
                    dilationH: int, dilationW: int, group: int, xdlops: bool,
                    perfConfig: Optional[Sequence[int]]=None):
        if dtype not in {"f16", "f32", "bf16", "i8"}:
            raise ValueError(f"Invalid datatype: {dtype}")
        if direction not in {"fwd", "bwd", "wrw"}:
            raise ValueError(f"Invalid direction: {direction}")
        if layout not in self.MLIR_OUTPUT_LAYOUTS:
            raise ValueError(f"Invalid layout: {layout}")

        self.dataType = dtype
        self.direction = direction

        self.filterLayout = self.MLIR_FILTER_LAYOUTS[layout]
        self.inputLayout = layout.lower()
        self.outputLayout = self.MLIR_OUTPUT_LAYOUTS[layout]

        self.n = n
        self.c = c
        self.hi = hi
        self.wi = wi
        self.k = k
        self.y = y
        self.x = x

        self.convStrideH = convStrideH
        self.convStrideW = convStrideW
        self.paddingHL = paddingHL
        self.paddingHR = paddingHR
        self.paddingWL = paddingWL
        self.paddingWR = paddingWR
        self.dilationH = dilationH
        self.dilationW = dilationW

        self.group = group
        self.xdlops = xdlops
        self.perfConfig = perfConfig
        self.ho = math.floor((self.hi + self.paddingHL + self.paddingHR - (self.y - 1) * self.dilationH - 1 ) / self.convStrideH) + 1
        self.wo = math.floor((self.wi + self.paddingWL + self.paddingWR * 2 - (self.x - 1) * self.dilationW - 1 ) / self.convStrideW) + 1


CORRECT_RESULT_RE = re.compile('\[1\s*1\s*1\]')

class TestResult(enum.Enum):
    PASS = 1
    INVALID = 2
    FAIL = 3

async def testConfig(config: MLIROnlyConfig, options: Options, paths: Paths) -> TestResult:
    """Runs the given configuration and returns whether it successfully concluded,
    failed validation, or was inapplicable."""
    miopenGenOpts = config.generateMlirDriverCommandLine()
    miopenGenOpts.append('-pv')

    applicableFromGen, genToApplicable = os.pipe()
    generator = await asyncio.create_subprocess_exec(
        paths.miopen_gen_path,
        *miopenGenOpts, stdout=genToApplicable, stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL)
    os.close(genToApplicable)

    applicability = await asyncio.create_subprocess_exec(
        paths.mlir_miopen_driver_path,
        '--kernel-pipeline=applicability', '-', stdin=applicableFromGen,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    os.close(applicableFromGen)
    _, genErrs = await generator.communicate()
    highLevel, tuneErrs = await applicability.communicate()

    if generator.returncode != 0:
        if options.debug:
            print(f"""miopen-gen failed for config {config!r}
Command line = {miopenGenOpts}
Return code = {generator.returncode}
Errors = {genErrs.decode('utf-8')}
""")
        return TestResult.INVALID

    if applicability.returncode != 0:
        if options.debug:
            print(f"""mlir-miopen-driver applicability pipeline failed for config {config!r}
Generator command line = {miopenGenOpts}
Return code = {applicability.returncode}
Errors = {tuneErrs.decode('utf-8')}
""")
        return TestResult.INVALID

    runnerFromLowering, loweringToRunner = os.pipe()
    lowering = await asyncio.create_subprocess_exec(
        paths.mlir_miopen_driver_path,
        '--kernel-pipeline=gpu', '-', stdin=asyncio.subprocess.PIPE,
        stdout=loweringToRunner, stderr=asyncio.subprocess.PIPE)
    os.close(loweringToRunner)

    mlir_rocm_runner_args = [f'--shared-libs={paths.libmlir_rocm_runtime_path},{paths.libconv_validation_wrappers_path},{paths.libmlir_runtime_utils_path}', '--entry-point-result=void']
    runner = await asyncio.create_subprocess_exec(
        paths.rocm_runner_path,
        *mlir_rocm_runner_args, stdin=runnerFromLowering,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    os.close(runnerFromLowering)

    _, loweringErrs = await lowering.communicate(input=highLevel)
    runnerOut, runnerErrs = await runner.communicate()
    runnerOut = runnerOut.decode('utf-8')

    if lowering.returncode != 0:
        if options.debug:
            print(f"""Low-level lowering did not complete succesfully for config {config!r}
Command line = {miopenGenOpts}
Errors = {loweringErrs.decode('utf-8')}
Return code = {lowering.returncode}""")
        return TestResult.FAIL

    if runner.returncode != 0:
        if options.debug:
            print(f"""Runner execution failed for config {config!r}
Output = {runnerOut}
Errors = {runnerErrs.decode('utf-8')}
Return code = {runner.returncode}""", file=sys.stderr)
        return TestResult.FAIL

    if not CORRECT_RESULT_RE.search(runnerOut):
        print(f"""Convolution returned intorrect result
Output = {runnerOut}
Errors = {runnerErrs.decode('utf-8')}""", file=sys.stderr)
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

async def dropGoodConfig(config: ConvConfiguration,
        options: Options, paths: Paths) -> Union[TestResult, ConvConfiguration]:
    """Test the given `params`, returning the corresponding `config` on failure
    and `None` on success or inapplicability"""
    result = await testConfig(config, options, paths)
    if not options.quiet:
        print(f"{result.name}: {config!r}")
    if result == TestResult.FAIL:
        return config
    return result

async def sweepParameters(paramIter: Iterable[IterType],
        toConfig: Callable[[IterType, Options], MLIROnlyConfig],
        options: Options, paths: Paths) -> Tuple[int, int, List[MLIROnlyConfig]]:
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

CONV_STRUCTURE = itertools.product(
    # Small/large - that is, do we have padding
    [False, True],
    # op
    ['fwd', 'wrw', 'bwd'],
    # layout
    ['NCHW', 'NHWC'],
    # dtype
    # TODO(kdrewnia): add bf16 once we're confident in that support
    # and add int8 for fwd only
    ['f32', 'f16'],
    # Padding - hl, hr, wl, wr in [0, 3]
    # [0, 3] hits the cases 0, < y/x, == y/x, > y/x
    range(0, 4),
    range(0, 4),
    range(0, 4),
    range(0, 4),
    # Stride - 1 or 2 - all meaningful strides before breaking past h/w=4
    range(1, 3),
    range(1, 3),
    # Dilation in [1, 2] - all meaningful dilations befor breaking past h/w=4
    range(1, 3),
    range(1, 3))

def to_conv_structure_type_test(params, options: Options) -> MLIROnlyConfig:
    size, op, layout, dtype, phl, phr, pwl, pwr, sh, sw, dh, dw = params
    # Fixed parameters, y = x = 2, hi = wi = 4, g = 1
    g, hi, wi, y, x = 1, 4, 4, 2, 2
    if size:
        # Values of n, c, k that prevent hitting the padding kernel
        n, c, k = 64, 64, 64
    else:
        # Values of n, c, k, meant to be small and to hit the padding kernel
        n, c, k = 1, 7, 7
    return MLIROnlyConfig(dtype, op, layout, n, c, hi, wi, k, y, x, sh, sw,
        phl, phr, pwl, pwr, dh, dw, g, options.xdlops)

XDLOPS_PERF_CONFIG = itertools.product(
    # op
    ['fwd', 'wrw', 'bwd'],
    # layout
    ['NCHW', 'NHWC'],
    # dtype
    ['f32', 'f16'],
    # MPerBlock (exponent)
    range(2, 9),
    # NPerBlock (exponent)
    range(4, 9),
    # KPerBlock (exponent)
    range(0, 4),
    # MPerWave (exponent)
    range(1, 8),
    # NPerWave (exponent)
    range(4, 8),
    # KPack (exponent)
    range(1, 4)
)
def to_xdlops_perf_config_test(params, options: Options) -> MLIROnlyConfig:
    n, g, c, hi, wi, k, y, x, sw, sh, phl, phr, pwl, pwr, dh, dw =\
         512, 1, 512, 1, 1, 512, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1
    op, layout, dtype, m_per_block, n_per_block, k_per_block, m_per_wave,\
         n_per_wave, kpack = params
    perf_config = (1 << m_per_block, 1 << n_per_block, 1 << k_per_block,
        1 << m_per_wave, 1 << n_per_wave, 1 << kpack, 1, 1)
    return MLIROnlyConfig(dtype, op, layout, n, c, hi, wi, k, y, x, sh, sw, phl,
        phr, pwl, pwr, dh, dw, g, True, perf_config)

NON_XDLOPS_PERF_CONFIG = itertools.product(
    # op
    ['fwd', 'wrw', 'bwd'],
    # layout
    ['NCHW', 'NHWC'],
    # dtype
    ['f32', 'f16'],
    # BlockSize (exponent)
    range(6, 9),
    # MPerBlock (exponent)
    range(5, 8),
    # NPerBlock (exponent)
    range(5, 8),
    # KPerBlock (exponent)
    range(2, 4),
    # MPerThread (exponent)
    range(1, 3),
    # NPerThread (exponent)
    range(1, 3)
)
def to_non_xdlops_perf_config_test(params, options: Options) -> MLIROnlyConfig:
    n, g, c, hi, wi, k, y, x, sw, sh, phl, phr, pwl, pwr, dh, dw =\
         512, 1, 512, 1, 1, 512, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1
    op, layout, dtype, block_size, m_per_block, n_per_block, k_per_block,\
        m_per_thread,n_per_thread = params
    perf_config = (1 << block_size, 1 << m_per_block, 1 << n_per_block,
        1 << k_per_block, 1 << m_per_thread, n_per_thread)
    return MLIROnlyConfig(dtype, op, layout, n, c, hi, wi, k, y, x, sh, sw, phl,
        phr, pwl, pwr, dh, dw, g, False, perf_config)

async def runConfig(paramIter: Iterable[IterType],
        toConfig: Callable[[IterType, Options], MLIROnlyConfig],
        options: Options, paths: Paths) -> bool:
    n_passes, n_invalids, failures = \
        await sweepParameters(paramIter, toConfig, options, paths)
    if len(failures) != 0:
        print("*** Summary of failures ***")
        for c in failures:
            print(' '.join(c.generateMlirDriverCommandLine()))
    print(f"Passed: {n_passes}, Invalid: {n_invalids}, Failed: {len(failures)}")
    return len(failures) == 0

def main() -> bool:
    parser = argparse.ArgumentParser(
            description='Sweep parameter values to check correctness of MLIR')
    parser.add_argument('config',
        help="The configuration to test",
        choices=['conv_structure', 'xdlops_perf_config', 'non_xdlops_perf_config', 'perf_config'])
    parser.add_argument('--debug', '-d', action='store_true', default=False,
        help='Turn on debug output (print error messages on failure or inapplicability)')
    parser.add_argument('--no-debug', '-D', dest='debug', action='store_false',
        help='Turn off debug output')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
        help="Quiet mode (don't output each test result")
    parser.add_argument('--no-quiet', '-Q', dest='quiet', action='store_false',
        help='Turn off quiet mode')
    parser.add_argument('--xdlops', '-x', action='store_true', default=False,
        help='Use xdlops when generating kernels (default off)')
    parser.add_argument('--no-xdlops', '-X', dest='xdlops', action='store_false',
        help='Explicitly disable xdlops usage')
    parser.add_argument('--jobs', '-j', type=int,
        default=(len(os.sched_getaffinity(0)) // 2),
        help="Number of jobs to run in parallel (default %(default)s)")
    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=MIOpenDriver.find_mlir_build_dir(),
        help="The build directory of MLIR based kernel generator",
    )
    parser.add_argument(
        "--miopen-build-dir",
        type=str,
        default=MIOpenDriver.find_miopen_build_dir(),
        help="The build directory of MIOpen",
    )
    args = parser.parse_args()
    options = Options(debug=args.debug, quiet=args.quiet, xdlops=args.xdlops,
        concurrent_tests=args.jobs)
    paths = MIOpenDriver.create_paths(args.mlir_build_dir, args.miopen_build_dir)

    config = args.config
    if config == 'perf_config':
        config = ('xdlops_' if options.xdlops else 'non_xdlops_') + config
    succeeded = False
    if config == 'conv_structure':
        succeeded = asyncio.run(runConfig(CONV_STRUCTURE,
            to_conv_structure_type_test, options, paths))
    elif config == 'xdlops_perf_config':
        succeeded = asyncio.run(runConfig(XDLOPS_PERF_CONFIG,
            to_xdlops_perf_config_test, options, paths))
    elif config == 'non_xdlops_perf_config':
        succeeded = asyncio.run(runConfig(NON_XDLOPS_PERF_CONFIG,
            to_non_xdlops_perf_config_test, options, paths))
    else:
        print(f"Unknown config: {config}", file=sys.stderr)
    return succeeded

if __name__ == '__main__':
    ret = main()
    sys.exit(int(not ret))
