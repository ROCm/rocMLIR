#!/usr/bin/env python3
"""Script to sweep the parameters of the rocmlir driver for bugs

Note: This requires Python 3.7 or newer, use pyenv or the like to install it temporarily

Usage:
$ ninja rocmlir-gen rocmlir-driver mlir-runner ci-performance-scripts
$ stdbuf --output=L python3 ./bin/parameterSweeps.py [config] | stdbuf --output=L tee [output-file-of-choice]"""

import argparse
import asyncio
import enum
import itertools
import math
import os
import sys

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Optional, Tuple, TypeVar

import perfRunner
from perfRunner import ConvConfiguration
from perfRunner import Paths
from perfRunner import get_arch
from perfRunner import get_num_cu


@dataclass(frozen=True)
class Options:
    """Class for keeping option state for the parameter sweep script."""
    debug: bool
    quiet: bool
    debugFails: bool
    arch: str
    flags: list
    concurrent_tests: int
    num_cu: int
    log_failures: bool = False


class PerfConfig:

    class Version(enum.Enum):
        V2 = 2
        V3 = 3
        V4 = 4

    def __init__(self, config: Sequence[int], version: Version = Version.V4):
        self._config = config
        self._version = version
        self._version_map = {
            PerfConfig.Version.V2: "v2",
            PerfConfig.Version.V3: "v3",
            PerfConfig.Version.V4: "v4"
        }

    def __str__(self):
        suffix = ','.join(str(v) for v in self._config)
        return f'{self._version_map[self._version]}:{suffix}'


class MLIROnlyConfig(ConvConfiguration):

    def __repr__(self):
        perf_config_str = str(self.perfconfig) if self.perfconfig else ""
        return f"""ConvConfiguration(dtype={self.dataType!r}, direction={self.direction!r}, layout={self.inputLayout.upper()!r},
                n={self.n!r}, c={self.c!r}, hi={self.hi!r}, wi={self.wi!r}, k={self.k!r}, y={self.y!r}, x={self.x!r},
                convStrideH={self.conv_stride_h!r}, convStrideW={self.conv_stride_w!r}, paddingHL={self.padding_hl!r}, paddingHR={self.padding_hr!r},
                paddingWL={self.padding_wl!r}, paddingWR={self.padding_wr!r}, dilationH={self.dilation_h!r}, dilationW={self.dilation_w!r},
                group={self.group!r}, arch={self.arch!r}, perfConfig={perf_config_str!r})"""

    def generate_mlir_driver_commandline(self, rocmlir_gen_flags) -> Sequence[str]:
        direction = {
            'fwd': 'conv',
            'bwd': 'conv_bwd_data',
            'wrw': 'conv_bwd_weight'
        }[self.direction]

        result = [
            '--operation', direction, '-t', self.dataType, '--arch', self.arch, '--fil_layout',
            self.filterLayout, '--in_layout', self.inputLayout, '--out_layout', self.outputLayout,
            '--batchsize',
            str(self.n), '--in_channels',
            str(self.c), '--in_h',
            str(self.hi), '--in_w',
            str(self.wi), '--out_channels',
            str(self.k), '--fil_h',
            str(self.y), '--fil_w',
            str(self.x), '--dilation_h',
            str(self.dilation_h), '--dilation_w',
            str(self.dilation_w), '--conv_stride_h',
            str(self.conv_stride_h), '--conv_stride_w',
            str(self.conv_stride_w), '--padding_h_l',
            str(self.padding_hl), '--padding_h_r',
            str(self.padding_hr), '--padding_w_l',
            str(self.padding_wl), '--padding_w_r',
            str(self.padding_wr)
        ]

        result += rocmlir_gen_flags

        if self.perfconfig is not None:
            result.append('--perf_config')
            result.append(str(self.perfconfig))

        return result

    def __init__(self,
                 dtype: str,
                 direction: str,
                 layout: str,
                 n: int,
                 c: int,
                 hi: int,
                 wi: int,
                 k: int,
                 y: int,
                 x: int,
                 conv_stride_h: int,
                 conv_stride_w: int,
                 padding_hl: int,
                 padding_hr: int,
                 padding_wl: int,
                 padding_wr: int,
                 dilation_h: int,
                 dilation_w: int,
                 group: int,
                 arch: str,
                 perfconfig: Optional[PerfConfig] = None):
        if dtype not in {"f16", "f32", "bf16", "i8"}:
            raise ValueError(f"Invalid datatype: {dtype}")
        if direction not in {"fwd", "bwd", "wrw"}:
            raise ValueError(f"Invalid direction: {direction}")

        self.dataType = dtype
        self.direction = direction

        self.filterLayout = perfRunner.filter_layouts(layout)
        self.inputLayout = layout.lower()
        self.outputLayout = perfRunner.output_layouts(layout)

        self.n = n
        self.c = c
        self.hi = hi
        self.wi = wi
        self.k = k
        self.y = y
        self.x = x

        self.conv_stride_h = conv_stride_h
        self.conv_stride_w = conv_stride_w
        self.padding_hl = padding_hl
        self.padding_hr = padding_hr
        self.padding_wl = padding_wl
        self.padding_wr = padding_wr
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        self.group = group
        self.arch = arch
        self.perfconfig = perfconfig
        self.ho = math.floor((self.hi + self.padding_hl + self.padding_hr -
                              (self.y - 1) * self.dilation_h - 1) / self.conv_stride_h) + 1
        self.wo = math.floor((self.wi + self.padding_wl + self.padding_wr * 2 -
                              (self.x - 1) * self.dilation_w - 1) / self.conv_stride_w) + 1


def multiline_repr(obj, num_fields=4):
    """ Returns a multi-line string representation of the given object,
    inserting a newline after every defined number of comma-separated
    fields in its repr(). Useful for making long configuration
    representations more readable in logs or debug output."""
    s = repr(obj).replace('\n', ' ')  # Flatten to one line
    lines = []
    field = ''
    fields = []
    in_quotes = False
    perf_config_str = None

    i = 0
    while i < len(s):
        # Detect start of perf_config to prevent it from being split
        if s.startswith('perf_config=', i):
            perf_config_str = s[i:]
            break
        c = s[i]
        if c == "'":
            in_quotes = not in_quotes
            field += c
        elif c == ',' and not in_quotes:
            fields.append(field.strip() + ',')
            field = ''
        else:
            field += c
        i += 1
    if field:
        fields.append(field.strip())
    for j in range(0, len(fields), num_fields):
        prefix = '\t' if j > 0 else ''
        group = fields[j:j + num_fields]
        if j + num_fields >= len(fields) and group and group[-1].endswith(','):
            group[-1] = group[-1][:-1]
        lines.append(f"{prefix}{' '.join(group)}")
    if perf_config_str:
        lines.append('\t' + perf_config_str.strip())

    return '\n'.join(lines)


class TestResult(enum.Enum):
    PASS = 1
    INVALID = 2
    FAIL = 3


async def test_config(config, options: Options, paths: Paths) -> TestResult:
    """Runs the given configuration and returns whether it successfully concluded,
    failed validation, or was inapplicable."""
    if isinstance(config, MLIROnlyConfig):
        rocmlir_gen_opts = config.generate_mlir_driver_commandline(options.flags)
    else:
        rocmlir_gen_opts = config.generate_mlir_driver_commandline(' '.join(options.flags),
                                                                   kernel_repeats=None).split()
        if getattr(config, "currentSeqLen") is not None:
            rocmlir_gen_opts.append(f"--current_seq_len={','.join(map(str, config.currentSeqLen))}")
    rocmlir_gen_opts.append('-pv')

    applicable_from_gen, gen_to_applicable = os.pipe()
    generator = await asyncio.create_subprocess_exec(paths.mlir_paths.rocmlir_gen_path,
                                                     *rocmlir_gen_opts,
                                                     stdout=gen_to_applicable,
                                                     stderr=asyncio.subprocess.PIPE,
                                                     stdin=asyncio.subprocess.DEVNULL)
    os.close(gen_to_applicable)

    applicability = await asyncio.create_subprocess_exec(paths.mlir_paths.rocmlir_driver_path,
                                                         '--kernel-pipeline=applicability',
                                                         '-',
                                                         stdin=applicable_from_gen,
                                                         stdout=asyncio.subprocess.PIPE,
                                                         stderr=asyncio.subprocess.PIPE)
    os.close(applicable_from_gen)
    _, gen_errs = await generator.communicate()
    high_level, tune_errs = await applicability.communicate()

    if generator.returncode != 0:
        if options.debug:
            print(f"""rocmlir-gen failed for config {config!r}
Command line = {rocmlir_gen_opts}
Return code = {generator.returncode}
Errors = {gen_errs.decode('utf-8')}
""")
        return TestResult.INVALID

    if applicability.returncode != 0:
        if options.debug:
            print(f"""rocmlir-driver applicability pipeline failed for config {config!r}
Generator command line = {rocmlir_gen_opts}
Return code = {applicability.returncode}
Errors = {tune_errs.decode('utf-8')}
""")
        return TestResult.INVALID

    runner_from_lowering, lowering_to_runner = os.pipe()
    lowering = await asyncio.create_subprocess_exec(paths.mlir_paths.rocmlir_driver_path,
                                                    '--kernel-pipeline=full',
                                                    '--host-pipeline=runner',
                                                    '-',
                                                    stdin=asyncio.subprocess.PIPE,
                                                    stdout=lowering_to_runner,
                                                    stderr=asyncio.subprocess.PIPE)
    os.close(lowering_to_runner)

    mlir_cpu_runner_args = [
        '-O2',
        f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path}',
        '--entry-point-result=void'
    ]
    runner = await asyncio.create_subprocess_exec(paths.mlir_paths.cpu_runner_path,
                                                  *mlir_cpu_runner_args,
                                                  stdin=runner_from_lowering,
                                                  stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.PIPE)
    os.close(runner_from_lowering)

    _, lowering_errs = await lowering.communicate(input=high_level)
    runner_out, runner_errs = await runner.communicate()
    runner_out = runner_out.decode('utf-8')

    if lowering.returncode != 0:
        if options.debug or options.debugFails:
            print(f"""Low-level lowering did not complete succesfully for config {config!r}
Config = {config}
Command line = {rocmlir_gen_opts}
Errors = {lowering_errs.decode('utf-8')}
Return code = {lowering.returncode}""")
        return TestResult.FAIL

    if runner.returncode != 0:
        if options.debug or options.debugFails:
            print(f"""Runner execution failed for config {config!r}
Config = {config}
Output = {runner_out}
Errors = {runner_errs.decode('utf-8')}
Return code = {runner.returncode}""",
                  file=sys.stderr)
        return TestResult.FAIL

    output_lines = [line.strip() for line in runner_out.splitlines() if len(line.strip()) > 0]
    expected_output = "[1 1 1]"
    all_correct = all(line == expected_output for line in output_lines)
    if not all_correct:
        print(f"""Config returned incorrect result
Config = {config}
Output = {runner_out}
Errors = {runner_errs.decode('utf-8')}""",
              file=sys.stderr)
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


async def drop_good_config(config, options: Options, paths: Paths):
    """Test the given `params`, returning the corresponding `config` on failure
    and `None` on success or inapplicability"""
    result = await test_config(config, options, paths)
    if not options.quiet:
        if isinstance(config, MLIROnlyConfig):
            print(f"{result.name}: {config!r}")
        else:
            print("-" * 100)
            print(f"{result.name}: {multiline_repr(config)}")
    if result == TestResult.FAIL:
        if options.log_failures:
            if isinstance(config, perfRunner.AttentionConfiguration):
                with open("failing_attn_configs.txt", "a") as f:
                    f.write(multiline_repr(config) + "\n")
            else:
                with open("failing_conv_configs.txt", "a") as f:
                    f.write(multiline_repr(config) + "\n")
        return config
    return result


async def sweep_parameters(param_iter: Iterable[IterType], to_config: Callable[[IterType, Options],
                                                                               PerfConfig],
                           options: Options, paths: Paths) -> Tuple[int, int, List[PerfConfig]]:
    failing_configs = []
    passed = 0
    invalid = 0
    configs = (c for c in (to_config(p, options) for p in param_iter))
    for configs in grouper((drop_good_config(c, options, paths) for c in configs),
                           options.concurrent_tests):
        configs_future = asyncio.gather(*configs)
        try:
            configs_results = await configs_future
        except Exception as e:
            configs_future.cancel()
            raise e
        for result in configs_results:
            if result == TestResult.PASS:
                passed = passed + 1
            elif result == TestResult.INVALID:
                invalid = invalid + 1
            else:
                failing_configs.append(result)

    return (passed, invalid, failing_configs)


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
    return MLIROnlyConfig(dtype, op, layout, n, c, hi, wi, k, y, x, sh, sw, phl, phr, pwl, pwr, dh,
                          dw, g, options.arch)


WMMA_PERF_CONFIG = itertools.product(
    # op
    ['fwd', 'wrw', 'bwd'],
    # layout
    ['NCHW', 'NHWC'],
    # dtype
    ['f16'],
    # MPerBlock (exponent)
    range(2, 9),
    # NPerBlock (exponent)
    range(4, 9),
    # KPerBlock (exponent)
    range(0, 4),
    # MPerWave (exponent)
    range(2, 8),
    # NPerWave (exponent)
    range(2, 8),
    # MNPerXdl (exponent)
    range(4, 5),  # 16 only
    # KPack (exponent)
    range(2, 5),
    # splitKFactor (exponent)
    range(0, 1),
    # GEMM Schedule Version
    range(1, 3))

MFMA_PERF_CONFIG = itertools.product(
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
    range(2, 8),
    # NPerWave (exponent)
    range(2, 8),
    # MNPerXdl (exponent)
    range(4, 6),
    # KPack (exponent)
    range(1, 4),
    # splitKFactor (exponent)
    range(0, 1),
    # GEMM Schedule Version
    range(1, 3))


def to_wmma_perf_config_test(params, options: Options) -> MLIROnlyConfig:
    n, g, c, hi, wi, k, y, x, sw, sh, phl, phr, pwl, pwr, dh, dw = \
        512, 1, 512, 1, 1, 512, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1
    op, layout, dtype, m_per_block, n_per_block, k_per_block, m_per_wave, \
        n_per_wave, mn_per_xdl, kpack, split_k, gemm_schedule = params
    # no mn_per_xdl for wmma
    perf_config_tuple = (1 << m_per_block, 1 << n_per_block, 1 << k_per_block, 1 << m_per_wave,
                         1 << n_per_wave, 1 << kpack, 1 << split_k, gemm_schedule, 2, 1, 1)
    return MLIROnlyConfig(dtype, op, layout, n, c, hi, wi, k, y, x, sh, sw, phl, phr, pwl, pwr, dh,
                          dw, g, options.arch, PerfConfig(perf_config_tuple, PerfConfig.Version.V3))


def to_mfma_perf_config_test(params, options: Options) -> MLIROnlyConfig:
    n, g, c, hi, wi, k, y, x, sw, sh, phl, phr, pwl, pwr, dh, dw = \
        512, 1, 512, 1, 1, 512, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1
    op, layout, dtype, m_per_block, n_per_block, k_per_block, m_per_wave, \
        n_per_wave, mn_per_xdl, kpack, split_k, gemm_schedule = params
    perf_config_tuple = (1 << m_per_block, 1 << n_per_block, 1 << k_per_block, 1 << m_per_wave,
                         1 << n_per_wave, 1 << mn_per_xdl, 1 << kpack, 1 << split_k, gemm_schedule,
                         2, 1, 1)
    return MLIROnlyConfig(dtype, op, layout, n, c, hi, wi, k, y, x, sh, sw, phl, phr, pwl, pwr, dh,
                          dw, g, options.arch, PerfConfig(perf_config_tuple, PerfConfig.Version.V4))


VANILLA_PERF_CONFIG = itertools.product(
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
    range(1, 3),
    # splitKFactor (exponent)
    range(0, 1),
    # scheduleVersion
    range(1, 5))


def to_vanilla_perf_config_test(params, options: Options) -> MLIROnlyConfig:
    n, g, c, hi, wi, k, y, x, sw, sh, phl, phr, pwl, pwr, dh, dw = \
         512, 1, 512, 1, 1, 512, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1
    op, layout, dtype, block_size, m_per_block, n_per_block, k_per_block, \
        m_per_thread, n_per_thread, split_k, schedule_version = params
    perf_config_tuple = (1 << block_size, 1 << m_per_block, 1 << n_per_block, 1 << k_per_block,
                         1 << m_per_thread, n_per_thread, 1 << split_k, schedule_version, 2)
    return MLIROnlyConfig(dtype, op, layout, n, c, hi, wi, k, y, x, sh, sw, phl, phr, pwl, pwr, dh,
                          dw, g, options.arch, PerfConfig(perf_config_tuple, PerfConfig.Version.V3))


async def run_config(param_iter: Iterable[IterType], to_config: Callable[[IterType, Options],
                                                                         MLIROnlyConfig],
                     options: Options, paths: Paths) -> bool:
    n_passes, n_invalids, failures = \
        await sweep_parameters(param_iter, to_config, options, paths)
    if len(failures) != 0:
        print("*** Summary of failures ***")
        for c in failures:
            print(' '.join(c.generate_mlir_driver_commandline(options.flags)))
    print(f"Passed: {n_passes}, Invalid: {n_invalids}, Failed: {len(failures)}")
    return len(failures) == 0 and n_passes > 0


def main() -> bool:
    parser = argparse.ArgumentParser(
        description='Sweep parameter values to check correctness of MLIR')
    parser.add_argument('config',
                        help="The configuration to test",
                        choices=[
                            'conv_structure', 'mfma_perf_config', 'vanilla_perf_config',
                            'wmma_perf_config', 'perf_config'
                        ])
    parser.add_argument(
        '--debug',
        '-d',
        action='store_true',
        default=False,
        help='Turn on debug output (print error messages on failure or inapplicability)')
    parser.add_argument('--no-debug',
                        '-D',
                        dest='debug',
                        action='store_false',
                        help='Turn off debug output')
    parser.add_argument('--quiet',
                        '-q',
                        action='store_true',
                        default=False,
                        help="Quiet mode (don't output each test result")
    parser.add_argument('--no-quiet',
                        '-Q',
                        dest='quiet',
                        action='store_false',
                        help='Turn off quiet mode')
    parser.add_argument('--xdlops',
                        '-x',
                        action='store_true',
                        default=False,
                        help='Use xdlops when generating kernels (default off)')
    parser.add_argument('--no-xdlops',
                        '-X',
                        dest='xdlops',
                        action='store_false',
                        help='Explicitly disable xdlops usage')
    parser.add_argument('--log-failures',
                        '-L',
                        action='store_true',
                        default=False,
                        help='Save failures to file')
    parser.add_argument('--debug-fails',
                        action='store_true',
                        default=False,
                        help='Enable debug output for failing configurations')
    parser.add_argument('--codepath',
                        type=str,
                        default='none',
                        help="codepath to control kernel generation")
    parser.add_argument('--jobs',
                        '-j',
                        type=int,
                        default=(len(os.sched_getaffinity(0)) // 2),
                        help="Number of jobs to run in parallel (default %(default)s)")
    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=perfRunner.find_mlir_build_dir(),
        help="The build directory of MLIR based kernel generator",
    )
    args = parser.parse_args()
    arch = get_arch()
    supported_codepath = ['mfma', 'vanilla', 'wmma']
    # If codepath not provided or not supported, infer it from the arch
    codepath = args.codepath
    rocmlir_gen_flags = []
    if codepath not in supported_codepath:
        if 'gfx908' in arch or 'gfx90a' in arch:
            codepath = 'mfma'
            rocmlir_gen_flags = ['-mfma=on', '-dot=on', '-atomic_add=on', '-atomic_add_f16=on']
        elif 'gfx942' in arch:
            codepath = 'mfma'
            rocmlir_gen_flags = [
                '-mfma=on', '-dot=on', '-atomic_add=on', '-atomic_add_f16=on',
                '-direct_to_lds_32b=on'
            ]
        elif 'gfx95' in arch:
            codepath = 'mfma'
            rocmlir_gen_flags = [
                '-mfma=on', '-dot=on', '-atomic_add=on', '-atomic_add_f16=on',
                '-atomic_add_bf16=on', '-direct_to_lds_32b=on', '-direct_to_lds_128b=on'
            ]
        elif 'gfx906' in arch:
            codepath = 'vanilla'
            rocmlir_gen_flags = ['-mfma=off', '-dot=on', '-atomic_add=off']
        elif 'gfx1030' in arch:
            # Use vanilla codepath for gfx1030 until it has its own perf configs
            codepath = 'vanilla'
            rocmlir_gen_flags = ['-mfma=off', '-dot=on', '-atomic_add=off']
        elif 'gfx11' in arch:
            codepath = 'wmma'
            rocmlir_gen_flags = ['-mfma=off', '-dot=on', '-atomic_add=on', '-wmma=infer']
        elif 'gfx12' in arch:
            codepath = 'wmma'
            rocmlir_gen_flags = [
                '-mfma=off', '-dot=on', '-atomic_add=on', '-wmma=infer', '-atomic_add_f16=on',
                '-atomic_add_bf16=on'
            ]
        else:
            # unknow arch info
            print(f"""Unknown arch {arch}""", file=sys.stderr)

    options = Options(debug=args.debug,
                      quiet=args.quiet,
                      log_failures=args.log_failures,
                      debugFails=args.debug_fails,
                      arch=arch,
                      flags=rocmlir_gen_flags,
                      concurrent_tests=args.jobs,
                      num_cu=get_num_cu(perfRunner.get_chip()))

    paths = perfRunner.create_paths(None, args.mlir_build_dir)

    config = args.config
    if config == 'perf_config':
        config = codepath + '_' + config
    succeeded = False
    if config == 'conv_structure':
        succeeded = asyncio.run(
            run_config(CONV_STRUCTURE, to_conv_structure_type_test, options, paths))
    elif config == 'mfma_perf_config':
        succeeded = asyncio.run(
            run_config(MFMA_PERF_CONFIG, to_mfma_perf_config_test, options, paths))
    elif config == 'vanilla_perf_config':
        succeeded = asyncio.run(
            run_config(VANILLA_PERF_CONFIG, to_vanilla_perf_config_test, options, paths))
    elif config == 'wmma_perf_config':
        succeeded = asyncio.run(
            run_config(WMMA_PERF_CONFIG, to_wmma_perf_config_test, options, paths))
    else:
        print(f"Unknown config: {config}", file=sys.stderr)
    return succeeded


if __name__ == '__main__':
    ret = main()
    sys.exit(int(not ret))
