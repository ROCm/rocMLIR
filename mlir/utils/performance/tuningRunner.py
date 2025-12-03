#!/usr/bin/env python3

from perfCommonUtils import Operation
from dataclasses import dataclass
import os
import subprocess
import sys
from pathlib import Path
import argparse
import glob
import tempfile

import perfRunner
from perfRunner import PerfConfiguration
from perfRunner import ConvConfiguration
from perfRunner import GemmConfiguration
from perfRunner import AttentionConfiguration
from perfRunner import GemmGemmConfiguration
from perfRunner import ConvGemmConfiguration
from perfRunner import Paths
from perfCommonUtils import CORRECT_RESULT_RE

import numpy as np
import pandas as pd

MLIR_N_REPEATS = 10
WARMUP_ITERATIONS = 1
SLEEP_US = 100  # 0.1 ms


@dataclass(frozen=True)
class Options:
    debug: bool
    tuning_space_kind: str
    quiet: bool
    arch: str
    num_cu: int
    rocmlir_gen_flags: str
    verify_mode: str
    verify_perfconfigs: bool
    tflops: bool
    compact_print: bool


def verify_mode_flags(verify_mode: str) -> str:
    if verify_mode == "none":
        return ""
    if verify_mode == "cpu":
        return " -pv"
    if verify_mode == "gpu":
        return " -pv_with_gpu --verifier-keep-perf-config=false"
    raise ValueError("Unknown verification mode", verify_mode)


# Run a gemm or conv config and verify it
def verify_kernel_with_perfconfig(perfconfig, config, paths: Paths, options: Options) -> float:
    if not options.compact_print:
        print(f"Verifying with perfConfig = {perfconfig}", file=sys.stderr)
    config.set_perfconfig(perfconfig.strip())
    rocmlir_gen_command = paths.mlir_paths.rocmlir_gen_path + \
        verify_mode_flags(options.verify_mode) + \
        ' -print-verify-results=summary ' + \
        config.generate_mlir_driver_commandline(options.rocmlir_gen_flags, kernel_repeats=MLIR_N_REPEATS)
    rocmlir_driver_command = [paths.mlir_paths.rocmlir_driver_path, '-c']
    mlir_cpu_runner_args = [
        '-O2',
        f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path}',
        '--entry-point-result=void'
    ]
    profiler_command = [perfRunner.ROCPROF] + perfRunner.get_metric_args_for_rocprof(
        options.arch) + [
            '--kernel-trace', '--stats', '-f', 'csv', '-o',
            perfRunner.BENCHMARKING_RESULT_FILE_NAME, '--', paths.mlir_paths.cpu_runner_path
        ] + mlir_cpu_runner_args

    if options.debug:
        print(rocmlir_gen_command, file=sys.stderr)

    prevdir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            # invoke rocmlir-gen.
            p1 = subprocess.Popen(rocmlir_gen_command.split(),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
            # pipe to rocmlir-driver
            p2 = subprocess.Popen(rocmlir_driver_command,
                                  stdin=p1.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
            p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
            # pipe to rocprof + mlir-runner.
            p3 = subprocess.Popen(profiler_command,
                                  stdin=p2.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            p2.stdout.close()  # Allow p2 to receive a SIGPIPE if p3 exits.
            # get output.
            try:
                outs, errs = p3.communicate(timeout=600)
                outs = outs.decode('utf-8')
                if p3.returncode != 0 or not CORRECT_RESULT_RE.search(outs):
                    print(f"""Verification failed:
Output = {outs}
Errors = {errs.decode('utf-8')}""",
                          file=sys.stderr)
                    return np.nan
            except subprocess.TimeoutExpired:
                print("Verification timed out", file=sys.stderr)
                p3.kill()
                outs, errs = p3.communicate()
                return np.nan
            nano_seconds = perfRunner.get_nanoseconds(
                perfRunner.get_profiler_output_path(options.arch,
                                                    perfRunner.BENCHMARKING_STATS_FILE_NAME))
        finally:
            os.chdir(prevdir)
    return nano_seconds


def get_winning_config(tuning_output, test_vector, config, all_data, paths: Path, options: Options):
    max_tflops = -np.inf
    min_ns = np.inf
    winning_config = "None"
    for i, result in enumerate(tuning_output):
        result = result.decode('utf-8').strip()
        if not options.quiet and not options.compact_print and i > 0 and i % 100 == 0:
            print(
                f"Tested {i} configs, best perf {max_tflops} TFlops {min_ns} ns on perf_config {winning_config}",
                file=sys.stderr)
        if options.debug:
            print(result, file=sys.stderr)
        # Time is in ns
        perfconfig, time = result.split('\t')
        if time == "N/A":
            nano_seconds = np.nan
        else:
            nano_seconds = float(time)

        config.set_perfconfig(perfconfig)
        entry = config.table_entry(nano_seconds)
        all_data.append(entry)
        these_tflops = entry['TFlops']
        # verify that each perfconfig passes accuracy verification
        if options.verify_perfconfigs:
            if options.verify_mode == "none":
                print(
                    "Use of `--verify-perf-configs` should happen in conjuction with `--verify-mode`. Please pass `--verify-mode=cpu` or `--verify-mode=gpu` flag"
                )
                sys.exit(1)
            else:
                verify_ns = verify_kernel_with_perfconfig(perfconfig, config, paths, options)
                if np.isnan(verify_ns):
                    # Verification failed, abort the loop
                    print(f"verification failed on : {test_vector} : {perfconfig}", file=sys.stderr)
                    sys.exit(1)

        if not np.isnan(these_tflops) and these_tflops > max_tflops:
            max_tflops = these_tflops
            min_ns = nano_seconds
            winning_config = perfconfig
            if options.compact_print and not options.quiet:
                print(
                    f"Tested {i} configs, best perf {max_tflops} TFlops {min_ns} ns on perf_config {winning_config}",
                    file=sys.stderr)

    return winning_config, max_tflops


# Tune MLIR Gemm or Convolution kernels
def tune_mlir_kernels(configs, conf_class, paths: Paths, options: Options):
    all_data = []
    winners = {}
    tuning_driver_args = [
        f"--tuning-space={options.tuning_space_kind}", f"--num-iterations={MLIR_N_REPEATS}",
        f"--warmup-iterations={WARMUP_ITERATIONS}", f"--sleep-us={SLEEP_US}", "--use-median"
    ]
    for test_vector in configs:
        if not test_vector.endswith(".mlir"):
            command_line = test_vector.split(sep=' ')
            config = conf_class.from_command_line(command_line, options.arch, options.num_cu)
            test_vector = config.to_command_line()
            print("Tuning:", test_vector, file=sys.stderr)
            command_line_options = config.generate_mlir_driver_commandline(
                options.rocmlir_gen_flags, kernel_repeats=None)
            # Note, we don't need the -ph, this goes to the tuning driver.
            # Because we don't set -ph, kernel_repeats is set to None.
            # This is because the kernel-repeats flag is only supported with host harness or CPU validation.
            kernel_gen_command = paths.mlir_paths.rocmlir_gen_path + ' ' + command_line_options
            kernel_gen = subprocess.Popen(kernel_gen_command.split(),
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL)
            tuning_loop = subprocess.Popen([paths.mlir_paths.rocmlir_tuning_driver_path] +
                                           tuning_driver_args,
                                           stdin=kernel_gen.stdout,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
            kernel_gen.stdout.close()
        else:
            # pipe to rocmlir_gen --emit-tuning-key
            tuning_key = subprocess.Popen(
                [paths.mlir_paths.rocmlir_gen_path, '--emit-tuning-key', test_vector],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            output, _ = tuning_key.communicate()
            result = output.decode('utf-8').strip().split('\t')
            print(f"Tuning:{result[2]} from {test_vector}", file=sys.stderr)
            command_line = result[2].split(sep=' ')
            config = conf_class.from_command_line(command_line, options.arch, options.num_cu)
            tuning_loop = subprocess.Popen([paths.mlir_paths.rocmlir_tuning_driver_path] +
                                           tuning_driver_args + [test_vector],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

        # Tune, printing progress as we go to avoid CI timeouts
        winning_config, max_tflops = get_winning_config(tuning_loop.stdout, test_vector, config,
                                                        all_data, paths, options)

        if options.verify_mode != "none":
            verify_ns = verify_kernel_with_perfconfig(winning_config, config, paths, options)
            if np.isnan(verify_ns):
                # Verification failed, abort the loop
                return None, None
            verify_tflops = config.compute_tflops(verify_ns)
            print(
                f"Tuned and verified : {test_vector} : {winning_config} with {max_tflops} TFlops and {verify_tflops} on verification",
                file=sys.stderr)
            if options.verify_mode == "gpu":
                print("Note: Verify tflops counts verification kernel", file=sys.stderr)
        else:
            print(f"Tuned : {test_vector} : {winning_config} with {max_tflops} TFlops",
                  file=sys.stderr)
        if options.tflops:
            winners[test_vector] = (winning_config, max_tflops)
        else:
            winners[test_vector] = winning_config
    all_data = pd.DataFrame(all_data)
    return winners, all_data


# Extract gemm or conv configurations from fusion tests
def extract_fusion_configs(test_dir, paths: Paths, options: Options):
    all_configs = []
    op_type = Operation.FUSION
    for filename in glob.glob(test_dir + '/*mlir'):
        print("Extract from:", filename, file=sys.stderr)
        test_entry = perfRunner.get_fusion_test_info(filename, paths)
        if not test_entry:
            continue
        test_vector = test_entry['testVector']
        if not test_vector:
            continue
        # skip if the best config already exists
        if test_vector in all_configs:
            print("An entry already exists in the tuning DB", file=sys.stderr)
            continue
        command_line = test_vector.split(sep=' ')
        if command_line[0].startswith('conv'):
            if op_type == Operation.FUSION:
                op_type = Operation.CONV
            elif op_type != Operation.CONV:
                print("Invalid config op: ", test_vector, file=sys.stderr)
                continue
        else:
            if op_type == Operation.FUSION:
                op_type = Operation.GEMM
            elif op_type != Operation.GEMM:
                print("Invalid config op: ", test_vector, file=sys.stderr)
                continue
        all_configs.append(test_vector)

    with open(paths.configuration_file_path, 'w') as outfile:
        for item in all_configs:
            outfile.write("%s\n" % item)

    return op_type


# Main function.
def main(args=None):
    """
    usage examples:

    python3 tuningRunner.py --op gemm -configs_file=../mlir/utils/performance/configs/tier1-gemm-configs --output=tuning_db.tsv
    python3 tuningRunner.py --op gemm --config="-g 3 -m 1024 -k 769 -n 512 -t f32 -transA 0 -transB 0"
    python3 tuningRunner.py --op conv --tuning-space=quick --config="conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1"
    python3 tuningRunner.py --op fusion -test_dir=../mlir/test/fusion/resnet50-e2e --output=tuning_db.tsv

    """
    if args is None:
        args = sys.argv[1:]

    arch = perfRunner.get_arch()
    num_cu = perfRunner.get_num_cu(perfRunner.get_chip())
    perfRunner.initialize_dtypes_attn()
    root_dir = str(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/performance/configs/tier1-conv-configs'

    parser = argparse.ArgumentParser(
        prog="rocMLIR tuning runner",
        description="A script for tuning MLIR conv or gemm kernels",
        allow_abbrev=False,
    )

    parser.add_argument("--op",
                        "--operation",
                        choices=['conv', 'gemm', 'fusion', 'attention', 'gemm_gemm', 'conv_gemm'],
                        default='conv',
                        help="Operation for tuning")

    parser.add_argument("-c",
                        "--configs_file",
                        type=str,
                        default=default_conv_configs,
                        help="File of configurations to test")

    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default="tuning_results_local.tsv",
                        help="File to output tuning results to. Will append to existing files")

    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=perfRunner.find_mlir_build_dir(),
        help="The build directory of MLIR based kernel generator",
    )

    parser.add_argument("--config",
                        type=str,
                        nargs='*',
                        help="The specific config to test, if you want to test one")

    parser.add_argument("--rocmlir_gen_flags",
                        type=str,
                        default=argparse.SUPPRESS,
                        help="rocmlir-gen flags to toggle each feature")

    parser.add_argument("--debug",
                        "-d",
                        action='store_true',
                        default=False,
                        help="Print debug messages on failure or inapplicability")

    parser.add_argument("--tuning-space",
                        default="full",
                        choices=["quick", "full", "greedy", "exhaustive"],
                        help="Which space of tuning configs should be used while tuning")
    parser.add_argument("--quiet",
                        "-q",
                        action='store_true',
                        default=False,
                        help="Quiet mode (don't output each test result)")

    parser.add_argument(
        "--verify-mode",
        default="gpu",
        choices=["none", "cpu", "gpu"],
        help=
        "Flag to specify if verification of compiled kernel with selected PerfConfig should use CPU based implementation or GPU based implementation"
    )

    parser.add_argument(
        "--verify-perf-configs",
        action='store_true',
        default=False,
        help=
        "Compile and verify given problem with all applicable perf configs. Whether it would use CPU or GPU based verification is controlled by `--verify-mode`. Should be used in conjunction with `--verify-mode`"
    )

    parser.add_argument("--test_dir",
                        default="../mlir/test/fusion/resnet50-e2e",
                        type=str,
                        help="fusion E2E tests directory")

    parser.add_argument('--data-type',
                        nargs='+',
                        choices=[
                            "f32", "f16", "bf16", "i8", "i8_i32", "i8_i8", "fp8", "fp8_f32",
                            "fp8_fp8", "f4E2M1FN"
                        ],
                        default=["f32", "f16", "i8"],
                        help='Force a set of datatypes')

    parser.add_argument(
        '--scale-type',
        nargs='+',
        choices=["f32", "f8E8M0FNU"],
        default=None,
        help=
        'Force a set of scale types for scaled GEMM (only applicable when config includes -scaledGemm)'
    )

    parser.add_argument("--tflops",
                        action='store_true',
                        default=False,
                        help="Include the TFlops along with the winning perf-configs")

    parser.add_argument("--compact-print",
                        action='store_true',
                        default=False,
                        help="Print info only when a change happens")

    parsed_args = parser.parse_args(args)

    rocmlir_gen_flags = ''
    if 'rocmlir_gen_flags' in parsed_args:
        rocmlir_gen_flags = parsed_args.rocmlir_gen_flags

    op_type = Operation.from_name(parsed_args.op)
    if op_type == Operation.FUSION:
        configs_path = "./fusion_config_file"
    else:
        configs_path = None if parsed_args.config else parsed_args.configs_file
    paths = perfRunner.create_paths(configs_path, parsed_args.mlir_build_dir)

    if not paths.mlir_paths:
        raise RuntimeError("MLIR build dir was not provided/found")

    options = Options(arch=arch,
                      num_cu=num_cu,
                      debug=parsed_args.debug,
                      quiet=parsed_args.quiet,
                      tuning_space_kind=parsed_args.tuning_space,
                      rocmlir_gen_flags=rocmlir_gen_flags,
                      verify_mode=parsed_args.verify_mode,
                      verify_perfconfigs=parsed_args.verify_perf_configs,
                      tflops=parsed_args.tflops,
                      compact_print=parsed_args.compact_print)

    if op_type == Operation.FUSION:
        op_type = extract_fusion_configs(parsed_args.test_dir, paths, options)

    conf_class = PerfConfiguration
    if op_type == Operation.CONV:
        conf_class = ConvConfiguration
    elif op_type == Operation.GEMM:
        conf_class = GemmConfiguration
    elif op_type == Operation.ATTENTION:
        conf_class = AttentionConfiguration
    elif op_type == Operation.GEMM_GEMM:
        conf_class = GemmGemmConfiguration
    elif op_type == Operation.CONV_GEMM:
        conf_class = ConvGemmConfiguration
    else:
        raise RuntimeError("Tuning operation was not provided/found")

    if parsed_args.config:
        configs = parsed_args.config
    elif op_type == Operation.CONV:
        configs = perfRunner.get_conv_configurations(paths.configuration_file_path)
    elif op_type == Operation.GEMM:
        datatypes, output_map = perfRunner.parse_data_types(parsed_args.data_type)
        scale_types = parsed_args.scale_type if parsed_args.scale_type else None
        configs = perfRunner.get_gemm_configurations(paths.configuration_file_path, datatypes,
                                                     output_map, scale_types)
    elif op_type == Operation.ATTENTION:
        configs = perfRunner.get_attn_configurations(paths.configuration_file_path)
    elif op_type == Operation.GEMM_GEMM:
        configs = perfRunner.get_gemm_gemm_configurations(paths.configuration_file_path)
    elif op_type == Operation.CONV_GEMM:
        configs = perfRunner.get_conv_gemm_configurations(paths.configuration_file_path)

    winners, all_data = tune_mlir_kernels(configs, conf_class, paths, options)

    if winners is None:
        # Tuning aborted, bail
        print("Tuning aborted")
        return 1

    if parsed_args.debug:
        print(all_data, file=sys.stderr)
        all_data.to_csv(f"{parsed_args.output}.debug", sep='\t', index=False)

    # Note, appending results here to allow multiple config sets
    if parsed_args.output == '-':
        outfile = sys.stdout
    else:
        outfile = open(parsed_args.output, 'a')

    with outfile:
        if parsed_args.tflops:
            print(f"# arch\tnumCUs\ttestVector\tperfConfig\tTFlops ({options.tuning_space_kind})",
                  file=outfile)
            for test_vector, (perfconfig, tflops) in winners.items():
                print(f"Arch = {arch}({num_cu} CUs), vector = '{test_vector}', \
perfConfig = {perfconfig}, TFlops = {tflops}",
                      file=sys.stderr)
                print(f"{arch}\t{num_cu}\t{test_vector}\t{perfconfig}\t{tflops}", file=outfile)
        else:
            print(f"# arch\tnumCUs\ttestVector\tperfConfig ({options.tuning_space_kind})",
                  file=outfile)
            for test_vector, perfconfig in winners.items():
                print(
                    f"Arch = {arch}({num_cu} CUs), vector = '{test_vector}', perfConfig = {perfconfig}",
                    file=sys.stderr)
                print(f"{arch}\t{num_cu}\t{test_vector}\t{perfconfig}", file=outfile)


if __name__ == '__main__':
    sys.exit(main())
