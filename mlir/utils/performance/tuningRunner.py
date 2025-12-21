#!/usr/bin/env python3

import argparse
import glob
import math
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd
from hip import hip
from tqdm import tqdm

import perfRunner
from perfCommonUtils import CORRECT_RESULT_RE, Operation
from perfRunner import (
    AttentionConfiguration,
    ConvConfiguration,
    ConvGemmConfiguration,
    GemmConfiguration,
    GemmGemmConfiguration,
    Paths,
    PerfConfiguration,
)

# Thread-local storage for GPU assignment
_thread_local = threading.local()

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
    output: str
    abort_on_error: bool
    retune: bool


@dataclass
class TuningResult:
    test_vector: str
    success: bool
    winning_config: Optional[str] = None
    max_tflops: Optional[float] = None
    entries: List[Dict] = field(default_factory=list)
    verify_tflops: Optional[float] = None
    error: Optional[str] = None


class TuningError(Exception):
    pass


def get_available_gpus() -> List[int]:
    for env_var in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
        visible_devices = os.environ.get(env_var)
        if visible_devices is not None:
            device_ids = []
            for device_id in visible_devices.split(','):
                device_id = device_id.strip()
                if not device_id:
                    continue
                try:
                    device_ids.append(int(device_id))
                except ValueError:
                    print(f"Warning: Ignoring invalid device id '{device_id}' in {env_var}",
                          sys.stderr)
            if len(device_ids) > 0:
                return device_ids
            else:
                print(f"Warning: No valid device ids found in {env_var}", file=sys.stderr)

    err, device_count = hip.hipGetDeviceCount()
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        print(f"Warning: hipGetDeviceCount failed with error '{err}', defaulting to GPU 0",
              file=sys.stderr)
        return [0]

    return list(range(device_count)) or [0]


def load_tuned_configs(options: Options) -> Dict[str, TuningResult]:
    tuned_configs = {}
    if options.output == '-' or not os.path.exists(options.output):
        return tuned_configs

    try:
        is_same_tuning_space = False
        with open_output_file(options.output, mode='r') as outfile:
            for line in outfile:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('###'):
                    continue
                if line.startswith('#'):
                    if f'perfConfig ({options.tuning_space_kind})' in line:
                        is_same_tuning_space = True
                    else:
                        is_same_tuning_space = False
                    continue

                if not is_same_tuning_space:
                    continue

                parts = line.split('\t')
                if len(parts) < 4:
                    continue

                test_vector = parts[2]
                winning_config = parts[3] if parts[3] else None
                max_tflops = float(parts[4]) if len(parts) > 4 and parts[4] else None

                if winning_config and winning_config != "None":
                    tuned_configs[test_vector] = TuningResult(test_vector=test_vector,
                                                              success=True,
                                                              winning_config=winning_config,
                                                              max_tflops=max_tflops)
    except Exception as e:
        print(f"Warning: Failed to load existing tuning results from {options.output}: {e}",
              file=sys.stderr)

    return tuned_configs


@contextmanager
def open_output_file(output_path: str, mode='a'):
    if output_path == '-':
        yield sys.stdout
    else:
        f = open(output_path, mode)
        try:
            yield f
        finally:
            f.close()


def write_header(outfile, options: Options):
    columns = ['arch', 'numCUs', 'testVector', f'perfConfig ({options.tuning_space_kind})']
    if options.tflops:
        columns.append('TFlops')
    print("# " + "\t".join(columns), file=outfile)
    outfile.flush()


def write_result(outfile, result: TuningResult, options: Options):
    fields = [options.arch, str(options.num_cu), result.test_vector, result.winning_config or ""]
    if options.tflops:
        fields.append(f"{result.max_tflops}" if result.max_tflops else "")
    print("\t".join(fields), file=outfile)
    outfile.flush()


def log_error(title: str, message: str, outfile=None):
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


def kill_process(proc):
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def verify_kernel_with_perfconfig(perfconfig,
                                  config,
                                  paths: Paths,
                                  options: Options,
                                  env=None) -> float:
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

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = None
        p2 = None
        p3 = None
        try:
            # invoke rocmlir-gen.
            p1 = subprocess.Popen(rocmlir_gen_command.split(),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  env=env,
                                  cwd=tmpdir)
            # pipe to rocmlir-driver
            p2 = subprocess.Popen(rocmlir_driver_command,
                                  stdin=p1.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  env=env,
                                  cwd=tmpdir)
            p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
            # pipe to rocprof + mlir-runner.
            p3 = subprocess.Popen(profiler_command,
                                  stdin=p2.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=env,
                                  cwd=tmpdir)
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
                kill_process(p3)
                outs, errs = p3.communicate()
                raise TuningError(f"""Verification timed out
{debug_info}
stdout:
{outs.decode('utf-8')}
stderr:
{errs.decode('utf-8')}""")

            stats_file = os.path.join(
                tmpdir,
                perfRunner.get_profiler_output_path(options.arch,
                                                    perfRunner.BENCHMARKING_STATS_FILE_NAME))
            nano_seconds = perfRunner.get_nanoseconds(stats_file)

        finally:
            kill_process(p1)
            kill_process(p2)
            kill_process(p3)

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


def tune_single_config(test_vector, conf_class, paths: Paths, options: Options, gpu_id: int,
                       num_compile_threads: int) -> Dict[str, Any]:
    tuning_driver_args = [
        f"--tuning-space={options.tuning_space_kind}", f"--num-iterations={MLIR_N_REPEATS}",
        f"--warmup-iterations={WARMUP_ITERATIONS}", f"--sleep-us={SLEEP_US}",
        f"--show-all-measurements={'true' if options.debug else 'false'}",
        f"--num-compile-threads={num_compile_threads}", "--use-median"
    ]

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

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
        kill_process(kernel_gen)
        kill_process(tuning_loop)

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


def tune_mlir_kernels(configs, conf_class, paths: Paths, options: Options):
    gpu_ids = get_available_gpus()
    num_workers = len(gpu_ids)
    num_compile_threads = math.ceil((os.cpu_count() or 1) / num_workers)

    if not options.quiet:
        print(f"Using {num_workers} GPU(s): {gpu_ids}", file=sys.stderr)
        print(f"Using {num_compile_threads} compile thread(s) per GPU", file=sys.stderr)

    gpu_assignment_lock = threading.Lock()
    available_gpus = list(gpu_ids)

    def get_gpu_id():
        """Ensure each thread gets a unique GPU ID."""
        if not hasattr(_thread_local, 'gpu_id'):
            with gpu_assignment_lock:
                if not available_gpus:
                    raise RuntimeError(
                        f"More workers than available GPUs! Expected {len(gpu_ids)} workers max.")
                _thread_local.gpu_id = available_gpus.pop()
        return _thread_local.gpu_id

    def tune_task(test_vector: str) -> TuningResult:
        try:
            gpu_id = get_gpu_id()
            result = tune_single_config(test_vector, conf_class, paths, options, gpu_id,
                                        num_compile_threads)
            return TuningResult(test_vector=test_vector,
                                success=result.get('success', False),
                                winning_config=result.get('winning_config'),
                                max_tflops=result.get('max_tflops'),
                                entries=result.get('entries', []),
                                verify_tflops=result.get('verify_tflops'),
                                error=result.get('error'))
        except Exception as e:
            return TuningResult(test_vector=test_vector, success=False, error=str(e))

    tuned_configs = {}
    if not options.retune and options.output != '-':
        tuned_configs = load_tuned_configs(options)
        if tuned_configs and not options.quiet:
            print(f"Found {len(tuned_configs)} tuned config(s) in {options.output}",
                  file=sys.stderr)

    configs_to_tune = [config for config in configs if config not in tuned_configs]
    num_tuned_configs = len(configs) - len(configs_to_tune)
    if num_tuned_configs and not options.quiet:
        print(f"Skipping {num_tuned_configs} out of {len(configs)} already tuned config(s)",
              file=sys.stderr)

    debugfile = None
    debug_header_written = False

    has_errors = False

    try:
        pbar = tqdm(total=len(configs),
                    initial=num_tuned_configs,
                    disable=options.quiet,
                    file=sys.stderr,
                    desc=f"Tuning {conf_class.__name__} ({options.tuning_space_kind})",
                    unit="config",
                    leave=False)

        executor = ThreadPoolExecutor(max_workers=num_workers)
        futures = {
            executor.submit(tune_task, test_vector): test_vector for test_vector in configs_to_tune
        }

        if options.debug:
            debugfile = open(f"{options.output}.debug", 'a')

        with open_output_file(options.output) as outfile:
            write_header(outfile, options)

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result.success:
                    write_result(outfile, result, options)

                    if debugfile and result.entries:
                        pd.DataFrame(result.entries).to_csv(debugfile,
                                                            sep='\t',
                                                            mode='a',
                                                            header=not debug_header_written,
                                                            index=False)
                        debugfile.flush()
                        debug_header_written = True
                else:
                    has_errors = True
                    log_error(f"Error tuning {result.test_vector}", result.error or "Unknown error",
                              outfile)
                    if options.abort_on_error:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return False

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return False
    finally:
        if debugfile:
            debugfile.close()
        pbar.close()

    if has_errors:
        print("Encountered errors during tuning", file=sys.stderr)
    else:
        print("Tuning completed successfully", file=sys.stderr)

    return not has_errors


def extract_fusion_configs(test_dir, paths: Paths, options: Options):
    all_configs = []
    op_type = Operation.FUSION
    for filename in glob.glob(test_dir + '/*mlir'):
        if not options.quiet:
            print("Extract from:", filename, file=sys.stderr)
        test_entry = perfRunner.get_fusion_test_info(filename, paths)
        if not test_entry:
            continue
        test_vector = test_entry['testVector']
        if not test_vector:
            continue
        # skip if the best config already exists
        if test_vector in all_configs:
            if not options.quiet:
                print("An entry already exists in the tuning DB", file=sys.stderr)
            continue
        command_line = test_vector.split(sep=' ')
        if command_line[0].startswith('conv'):
            if op_type == Operation.FUSION:
                op_type = Operation.CONV
            elif op_type != Operation.CONV:
                if not options.quiet:
                    print("Invalid config op: ", test_vector, file=sys.stderr)
                continue
        else:
            if op_type == Operation.FUSION:
                op_type = Operation.GEMM
            elif op_type != Operation.GEMM:
                if not options.quiet:
                    print("Invalid config op: ", test_vector, file=sys.stderr)
                continue
        all_configs.append(test_vector)

    with open(paths.configuration_file_path, 'w') as outfile:
        for item in all_configs:
            outfile.write("%s\n" % item)

    return op_type


def main(args=None):
    """
    usage examples:

    python3 tuningRunner.py --op gemm --configs-file=../mlir/utils/performance/configs/tier1-gemm-configs --output=tuning_db.tsv
    python3 tuningRunner.py --op gemm --config="-g 3 -m 1024 -k 769 -n 512 -t f32 -transA 0 -transB 0"
    python3 tuningRunner.py --op conv --tuning-space=quick --config="conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1"
    python3 tuningRunner.py --op fusion --test-dir=../mlir/test/fusion/resnet50-e2e --output=tuning_db.tsv

    """
    if args is None:
        args = sys.argv[1:]

    arch = perfRunner.get_arch()
    num_cu = perfRunner.get_num_cu(perfRunner.get_chip())
    root_dir = str(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/performance/configs/tier1-conv-configs'

    parser = argparse.ArgumentParser(
        prog="tuningRunner.py",
        description="Automated performance tuning for rocMLIR generated kernels",
        allow_abbrev=False,
    )

    parser.add_argument("--op",
                        "--operation",
                        choices=['conv', 'gemm', 'fusion', 'attention', 'gemm_gemm', 'conv_gemm'],
                        default='conv',
                        help="Operation to tune")

    parser.add_argument(
        "-c",
        "--configs-file",
        "--configs_file",  # for backward compatibility
        type=str,
        default=default_conv_configs,
        help="Path to file containing list of configurations to tune")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="tuning_results_local.tsv",
        help=
        "Output file path for tuning results in TSV format. Results will be appended if file exists. Use '-' for stdout."
    )

    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=perfRunner.find_mlir_build_dir(),
        help=
        "Path to rocMLIR build directory containing rocmlir-gen, rocmlir-driver, rocmlir-tuning-driver, and other build artifacts",
    )

    parser.add_argument("--config",
                        type=str,
                        nargs='*',
                        help="Specific config to tune. Format depends on --op type.")

    parser.add_argument(
        "--rocmlir-gen-flags",
        "--rocmlir_gen_flags",  # for backward compatibility
        type=str,
        default=argparse.SUPPRESS,
        help="Additional flags to pass to rocmlir-gen")

    parser.add_argument("-d",
                        "--debug",
                        action='store_true',
                        default=False,
                        help="Enable debug output including detailed measurements")

    parser.add_argument("--tuning-space",
                        default="full",
                        choices=["quick", "full", "greedy", "exhaustive"],
                        help="Tuning space kind to use")

    parser.add_argument(
        "-q",
        "--quiet",
        action='store_true',
        default=False,
        help="Suppress progress bars and informational messages, showing only errors")

    parser.add_argument("--verify-mode",
                        default="gpu",
                        choices=["none", "cpu", "gpu"],
                        help="Verification mode to use when verifying perf configs")

    parser.add_argument(
        "--verify-perf-configs",
        action='store_true',
        default=False,
        help=
        "Verify each perf config during tuning, not just the winning config. Requires --verify-mode to be cpu or gpu."
    )

    parser.add_argument(
        "--test-dir",
        "--test_dir",  # for backward compatibility
        default="../mlir/test/fusion/resnet50-e2e",
        type=str,
        help=
        "Directory containing fusion E2E tests to extract configs from. Only used when --op=fusion."
    )

    parser.add_argument('--data-type',
                        nargs='+',
                        choices=[
                            "f32", "f16", "bf16", "i8", "i8_i32", "i8_i8", "fp8", "fp8_f32",
                            "fp8_fp8", "f4E2M1FN"
                        ],
                        default=["f32", "f16", "i8"],
                        help="Force a set of data types for gemm tuning. Only used when --op=gemm.")

    parser.add_argument(
        '--scale-type',
        nargs='+',
        choices=["f32", "f8E8M0FNU"],
        default=None,
        help="Force a set of scale types for gemm tuning. Only used when --op=gemm.")

    parser.add_argument("--tflops",
                        action='store_true',
                        default=False,
                        help="Include achieved TFLOPS in the output alongside the winning config")

    parser.add_argument("--abort-on-error",
                        action='store_true',
                        default=False,
                        help="Abort tuning upon first error encounter")

    parser.add_argument(
        "--retune",
        action='store_true',
        default=False,
        help="Force retuning of all configs, ignoring existing results in the output file")

    parsed_args = parser.parse_args(args)

    if parsed_args.verify_perf_configs and parsed_args.verify_mode == "none":
        print(
            "Use of `--verify-perf-configs` is not allowed with `--verify-mode=none`. Please pass `--verify-mode=cpu` or `--verify-mode=gpu`.",
            file=sys.stderr)
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
                      output=parsed_args.output,
                      abort_on_error=parsed_args.abort_on_error,
                      retune=parsed_args.retune)

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

    return not tune_mlir_kernels(configs, conf_class, paths, options)


if __name__ == '__main__':
    sys.exit(main())
