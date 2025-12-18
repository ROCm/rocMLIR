#!/usr/bin/env python3

from perfCommonUtils import Operation
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import subprocess
import sys
import argparse
import glob
import tempfile
import itertools
import math

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
import json

from hip import hip

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
    output: str
    abort_on_error: bool


class TuningError(Exception):
    pass


def log_error(title, message, outfile=None):
    content = f"{title}\n" + '\n'.join(f"\t{line}" for line in message.split('\n'))
    print(content, file=sys.stderr)
    if outfile:
        print('\n'.join(f"### {line}" for line in content.split('\n')), file=outfile)
        outfile.flush()


def verify_mode_flags(verify_mode: str) -> str:
    if verify_mode == "none":
        return ""
    if verify_mode == "cpu":
        return " -pv"
    if verify_mode == "gpu":
        return " -pv_with_gpu --verifier-keep-perf-config=false"
    raise ValueError("Unknown verification mode", verify_mode)


# Run a gemm or conv config and verify it
def verify_kernel_with_perfconfig(perfconfig,
                                  config,
                                  paths: Paths,
                                  options: Options,
                                  env=None) -> float:
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

    if env is None:
        env = os.environ.copy()

    prevdir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = None
        p2 = None
        p3 = None
        try:
            os.chdir(tmpdir)
            # invoke rocmlir-gen.
            p1 = subprocess.Popen(rocmlir_gen_command.split(),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  env=env)
            # pipe to rocmlir-driver
            p2 = subprocess.Popen(rocmlir_driver_command,
                                  stdin=p1.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  env=env)
            p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
            # pipe to rocprof + mlir-runner.
            p3 = subprocess.Popen(profiler_command,
                                  stdin=p2.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=env)
            p2.stdout.close()  # Allow p2 to receive a SIGPIPE if p3 exits.
            # get output.
            debug_info = f"""rocmlir-gen cmd = {rocmlir_gen_command}
rocmlir-driver cmd = {' '.join(rocmlir_driver_command)}
rocprof cmd = {' '.join(profiler_command)}"""

            try:
                outs, errs = p3.communicate(timeout=600)
                outs = outs.decode('utf-8')
                if p3.returncode != 0 or not CORRECT_RESULT_RE.search(outs):
                    raise TuningError(f"""Verification failed
{debug_info}
stdout:
{outs}
stderr:
{errs.decode('utf-8')}""")
            except subprocess.TimeoutExpired:
                p3.kill()
                outs, errs = p3.communicate()
                raise TuningError(f"""Verification timed out
{debug_info}
stdout:
{outs.decode('utf-8')}
stderr:
{errs.decode('utf-8')}""")

            nano_seconds = perfRunner.get_nanoseconds(
                perfRunner.get_profiler_output_path(options.arch,
                                                    perfRunner.BENCHMARKING_STATS_FILE_NAME))
        finally:
            os.chdir(prevdir)
            if p1:
                p1.kill()
                p1.wait()
            if p2:
                p2.kill()
                p2.wait()
            if p3:
                p3.kill()
                p3.wait()
    return nano_seconds


def get_winning_config(tuning_output, config, paths: Paths, options: Options, env=None):
    max_tflops = -np.inf
    winning_config = "None"
    entries = []

    for result in tuning_output:
        result = result.decode('utf-8').strip()
        try:
            parts = result.split('\t')
            if len(parts) < 2:
                print(f"Error parsing tuning driver output line: {result}", file=sys.stderr)
                continue
            perfconfig = parts[0]
            time = parts[-1]
            if time == "N/A":
                nano_seconds = np.nan
                measurements = None
            else:
                nano_seconds = float(time)
                measurements = json.loads(parts[1]) if len(parts) == 3 else None
        except ValueError:
            print(f"Error parsing tuning driver output line: {result}", file=sys.stderr)
            continue

        config.set_perfconfig(perfconfig)
        entry = config.table_entry(nano_seconds)
        if options.debug:
            entry["Measurements"] = measurements
        entries.append(entry)
        these_tflops = entry['TFlops']

        # verify that each perfconfig passes accuracy verification
        if options.verify_perfconfigs and not np.isnan(nano_seconds):
            try:
                verify_ns = verify_kernel_with_perfconfig(perfconfig, config, paths, options, env)
            except TuningError as e:
                raise TuningError(
                    f"Error during verification of perf config {perfconfig}\n{str(e)}")
            if np.isnan(verify_ns):
                raise TuningError(f"Verification failed for perf config {perfconfig}")

        if not np.isnan(these_tflops) and these_tflops > max_tflops:
            max_tflops = these_tflops
            winning_config = perfconfig

    return winning_config, max_tflops, entries


def get_available_gpus():
    for env_var in ["ROCM_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"]:
        visible_devices = os.environ.get(env_var)
        if visible_devices is not None:
            return [int(x.strip()) for x in visible_devices.split(',') if x.strip().isdigit()]

    err, device_count = hip.hipGetDeviceCount()
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        print(f"Warning: hipGetDeviceCount failed with error {err}, defaulting to GPU 0",
              file=sys.stderr)
        return [0]

    return list(range(device_count))


def tune_single_config(test_vector, conf_class, paths: Paths, options: Options, gpu_id,
                       num_compile_threads):
    tuning_driver_args = [
        f"--tuning-space={options.tuning_space_kind}", f"--num-iterations={MLIR_N_REPEATS}",
        f"--warmup-iterations={WARMUP_ITERATIONS}", f"--sleep-us={SLEEP_US}",
        f"--show-all-measurements={'true' if options.debug else 'false'}",
        f"--num-compile-threads={num_compile_threads}", "--use-median"
    ]

    env = os.environ.copy()
    env["ROCM_VISIBLE_DEVICES"] = str(gpu_id)

    kernel_gen = None
    tuning_loop = None
    try:
        if not test_vector.endswith(".mlir"):
            command_line = test_vector.split(sep=' ')
            config = conf_class.from_command_line(command_line, options.arch, options.num_cu)
            test_vector = config.to_command_line()
            command_line_options = config.generate_mlir_driver_commandline(
                options.rocmlir_gen_flags, kernel_repeats=None)
            # Note, we don't need the -ph, this goes to the tuning driver.
            # Because we don't set -ph, kernel_repeats is set to None.
            # This is because the kernel-repeats flag is only supported with host harness or CPU validation.
            kernel_gen_command = paths.mlir_paths.rocmlir_gen_path + ' ' + command_line_options
            kernel_gen = subprocess.Popen(kernel_gen_command.split(),
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL,
                                          env=env)
            tuning_loop = subprocess.Popen([paths.mlir_paths.rocmlir_tuning_driver_path] +
                                           tuning_driver_args,
                                           stdin=kernel_gen.stdout,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           env=env)
            kernel_gen.stdout.close()
        else:
            # pipe to rocmlir_gen --emit-tuning-key
            tuning_key = subprocess.Popen(
                [paths.mlir_paths.rocmlir_gen_path, '--emit-tuning-key', test_vector],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env)
            output, _ = tuning_key.communicate()
            result = output.decode('utf-8').strip().split('\t')
            command_line = result[2].split(sep=' ')
            config = conf_class.from_command_line(command_line, options.arch, options.num_cu)
            tuning_loop = subprocess.Popen([paths.mlir_paths.rocmlir_tuning_driver_path] +
                                           tuning_driver_args + [test_vector],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           env=env)

        # Tune, printing progress as we go to avoid CI timeouts
        winning_config, max_tflops, entries = get_winning_config(tuning_loop.stdout, config, paths,
                                                                 options, env)

    except TuningError as e:
        return {'success': False, 'error': str(e)}
    finally:
        if kernel_gen:
            kernel_gen.kill()
            kernel_gen.wait()
        if tuning_loop:
            tuning_loop.kill()
            tuning_loop.wait()

    if tuning_loop.returncode != 0:
        error_msg = f"rocmlir-tuning-driver failed with return code {tuning_loop.returncode}"
        stderr_content = tuning_loop.stderr.read().decode('utf-8').strip()
        if stderr_content:
            error_msg += f"\nstderr:\n{stderr_content}"
        return {'success': False, 'error': error_msg}

    if winning_config == "None":
        return {'success': False, 'error': "No valid perf config found"}

    verify_tflops = None
    if options.verify_mode != "none":
        try:
            verify_ns = verify_kernel_with_perfconfig(winning_config, config, paths, options, env)
        except TuningError as e:
            return {
                'success': False,
                'error': f"Error during verification of winning config {winning_config}\n{str(e)}"
            }

        if np.isnan(verify_ns):
            return {
                'success': False,
                'error': f"Verification failed for winning config {winning_config}"
            }

        verify_tflops = config.compute_tflops(verify_ns)

    return {
        'success': True,
        'winning_config': winning_config,
        'max_tflops': max_tflops,
        'entries': entries,
        'verify_tflops': verify_tflops
    }


# Tune MLIR Gemm or Convolution kernels
def tune_mlir_kernels(configs, conf_class, paths: Paths, options: Options):
    outfile = None
    debugfile = None
    try:
        if options.output == '-':
            outfile = sys.stdout
        else:
            outfile = open(options.output, 'a')

        if options.debug:
            debugfile = open(f"{options.output}.debug", 'a')

        result_data_template = {
            'arch': options.arch,
            'numCUs': options.num_cu,
            'testVector': '',
            f'perfConfig ({options.tuning_space_kind})': ''
        }
        if options.tflops:
            result_data_template['TFlops'] = 0.0

        # Create a DataFrame to hold results. We will write out one problem config at a time as we go.
        result_df = pd.DataFrame([result_data_template])

        # Print header
        print("# " + "\t".join(result_df.columns), file=outfile)
        outfile.flush()

        debug_header_written = False

        gpu_ids = get_available_gpus()
        if not options.quiet:
            print(f"Using GPUs: {gpu_ids}", file=sys.stderr)
        gpu_cycle = itertools.cycle(gpu_ids)

        num_compile_threads = math.ceil(os.cpu_count() / len(gpu_ids))

        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            future_to_test_vector = {
                executor.submit(tune_single_config, test_vector, conf_class, paths, options,
                                next(gpu_cycle), num_compile_threads):
                    test_vector for test_vector in configs
            }

            for future in as_completed(future_to_test_vector):
                test_vector = future_to_test_vector[future]
                error_title = f"Error tuning test vector: {test_vector}"

                try:
                    result = future.result()
                except Exception as e:
                    log_error(error_title, str(e), outfile)
                    if options.abort_on_error:
                        return False
                    continue

                if not result['success']:
                    log_error(error_title, result['error'], outfile)
                    if options.abort_on_error:
                        return False
                    continue

                winning_config = result['winning_config']
                max_tflops = result['max_tflops']
                entries = result['entries']

                # Eagerly write out results to output file
                result_df.iloc[0, 2] = test_vector
                result_df.iloc[0, 3] = winning_config
                if options.tflops:
                    result_df.iloc[0, 4] = max_tflops
                result_df.to_csv(outfile, sep='\t', mode='a', header=False, index=False)
                outfile.flush()

                if debugfile:
                    pd.DataFrame(entries).to_csv(debugfile,
                                                 sep='\t',
                                                 mode='a',
                                                 header=not debug_header_written,
                                                 index=False)
                    debugfile.flush()
                    debug_header_written = True
        return True
    finally:
        if outfile and outfile != sys.stdout:
            outfile.close()
        if debugfile:
            debugfile.close()


# Extract gemm or conv configurations from fusion tests
def extract_fusion_configs(test_dir, paths: Paths):
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

    parser.add_argument("--abort-on-error",
                        action='store_true',
                        default=False,
                        help="Abort tuning upon first error encounter")

    parsed_args = parser.parse_args(args)

    if parsed_args.verify_perf_configs and parsed_args.verify_mode == "none":
        print(
            "Use of `--verify-perf-configs` is not allowed with `--verify-mode=none`. Please pass `--verify-mode=cpu` or `--verify-mode=gpu`."
        )
        return 1

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
                      compact_print=parsed_args.compact_print,
                      output=parsed_args.output,
                      abort_on_error=parsed_args.abort_on_error)

    if op_type == Operation.FUSION:
        op_type = extract_fusion_configs(parsed_args.test_dir, paths)

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

    if not tune_mlir_kernels(configs, conf_class, paths, options):
        print("Tuning aborted", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
