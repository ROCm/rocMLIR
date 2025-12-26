#!/usr/bin/env python3
"""Automated performance tuning for rocMLIR generated kernels.

This script tunes MLIR kernels by running them with different performance
configurations and selecting the best one based on execution time.

Usage examples:
    python3 tuningRunner.py --op gemm --configs-file=../mlir/utils/performance/configs/tier1-gemm-configs --output=tuning_db.tsv
    python3 tuningRunner.py --op gemm --config="-g 3 -m 1024 -k 769 -n 512 -t f32 -transA 0 -transB 0"
    python3 tuningRunner.py --op conv --tuning-space=quick --config="conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1"
    python3 tuningRunner.py --op fusion --test-dir=../mlir/test/fusion/resnet50-e2e --output=tuning_db.tsv
"""

import argparse
import glob
import math
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd
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


class UniqueChoicesAction(argparse.Action):
    """Argparse action that ensures no duplicate values."""

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != len(set(values)):
            duplicates = [v for v in values if values.count(v) > 1]
            parser.error(
                f"argument {option_string}: duplicate values not allowed: {set(duplicates)}")
        setattr(namespace, self.dest, values)


@dataclass(frozen=True)
class Options:
    """Configuration options for the tuning process."""
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
    gpu_ids: List[int]


@dataclass
class TuningResult:
    """Result of tuning a single configuration."""
    test_vector: str
    success: bool
    winning_config: Optional[str] = None
    max_tflops: Optional[float] = None
    entries: List[Dict] = field(default_factory=list)
    verify_tflops: Optional[float] = None
    error: Optional[str] = None


class TuningError(Exception):
    """Raised when tuning or verification fails."""
    pass


class OutputFileWriter:
    """Context manager for writing tuning results to TSV file."""

    def __init__(self, filepath: str, options: Options):
        self.filepath = filepath
        self.options = options
        self.file = None
        self.header_written = False

    def __enter__(self):
        if self.filepath == '-':
            self.file = sys.stdout
        else:
            self.file = open(self.filepath, 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file and self.file != sys.stdout:
            self.file.close()

    def _write_header(self):
        if self.header_written:
            return

        commit_hash = get_git_commit_hash()
        print(f"# commit: {commit_hash}", file=self.file)
        columns = ['arch', 'numCUs', 'testVector', f'perfConfig ({self.options.tuning_space_kind})']
        if self.options.tflops:
            columns.append('TFlops')
        print("# " + "\t".join(columns), file=self.file)

        self.file.flush()
        self.header_written = True

    def write_result(self, result: TuningResult):
        self._write_header();

        fields = [
            self.options.arch,
            str(self.options.num_cu), result.test_vector, result.winning_config or ""
        ]
        if self.options.tflops:
            fields.append(f"{result.max_tflops}" if result.max_tflops else "")
        print("\t".join(fields), file=self.file)

        self.file.flush()

    def write_error(self, content: str):
        self._write_header();
        print('\n'.join(f"### {line}" for line in content.split('\n')), file=self.file)
        self.file.flush()


class DebugFileWriter:
    """Context manager for writing debug entries to TSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.header_written = False

    def __enter__(self):
        self.file = open(self.filepath, 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()

    def write_entries(self, entries: List[Dict]):
        if not entries:
            return

        pd.DataFrame(entries).to_csv(self.file,
                                     sep='\t',
                                     mode='a',
                                     header=not self.header_written,
                                     index=False)

        self.file.flush()
        self.header_written = True


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              stderr=subprocess.DEVNULL).decode().strip()
        return commit_hash
    except Exception:
        return "unknown"


def load_tuned_configs(options: Options) -> Dict[str, TuningResult]:
    """Load previously tuned configurations from output file.

    The output file format is TSV with the following structure:
    - Commit line starting with '# commit: ' indicating the git commit hash of the tuning run
    - Header lines starting with '# ' containing tuning space kind in parentheses
      (e.g., '# arch\tnumCUs\ttestVector\tperfConfig (quick)\tTFlops')
    - Multiple header sections can exist in the same file from different tuning runs
    - Data lines with tab-separated fields following each header
    - Error lines starting with '### ' indicating errors during tuning

    Only data lines under headers matching options.tuning_space_kind are loaded.
    For example, if options.tuning_space_kind='quick', only data under headers
    containing '(quick)' will be loaded, ignoring '(full)' or other sections.
    """
    tuned_configs = {}
    if options.output == '-' or not os.path.exists(options.output):
        return tuned_configs

    def is_commit_line(line: str) -> bool:
        return line.startswith('# commit: ')

    def is_header_line(line: str) -> bool:
        return line.startswith('# ')

    def is_error_line(line: str) -> bool:
        return line.startswith('### ')

    current_commit = get_git_commit_hash()

    try:
        file_commit = current_commit
        is_same_tuning_space = False
        with open(options.output, mode='r') as outfile:
            for line in outfile:
                line = line.strip()
                if not line:
                    continue

                if is_commit_line(line):
                    file_commit = line[len('# commit: '):].strip()
                    continue

                if is_header_line(line):
                    is_same_tuning_space = f"({options.tuning_space_kind})" in line
                    if is_same_tuning_space and file_commit != current_commit:
                        print(
                            f"Warning: Loading tuned configs from a different commit (file: {file_commit[:12]}, current: {current_commit[:12]})",
                            file=sys.stderr)
                    continue

                if is_error_line(line) or not is_same_tuning_space:
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


def get_gpu_info() -> Dict[int, str]:
    """Query physical GPU IDs and their SKUs using rocm-smi.

    rocm-smi reports physical device IDs regardless of environment variables (e.g.,
    ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES).
    We are assuming that SKU names are unique per GPU for identification purposes,
    as GFX IDs are not always unique.
    """
    try:
        output = subprocess.check_output(["rocm-smi", "--showproductname", "--json"],
                                         text=True,
                                         timeout=10)
        data = json.loads(output)
        gpu_info = {}
        for key in data.keys():
            if key.startswith("card"):
                gpu_id = int(key.replace("card", ""))
                gpu_info[gpu_id] = data[key].get("Card SKU", "unknown")
        if gpu_info:
            return gpu_info
        print("Warning: rocm-smi returned no GPU cards", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Warning: rocm-smi failed with return code {e.returncode}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("Warning: rocm-smi timed out", file=sys.stderr)
    except FileNotFoundError:
        print("Warning: rocm-smi not found in PATH", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse rocm-smi JSON output: {e}", file=sys.stderr)
    except ValueError as e:
        print(f"Warning: Failed to extract GPU info from rocm-smi output: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Unexpected error querying GPUs: {e}", file=sys.stderr)

    print("Warning: Could not detect GPUs, defaulting to GPU 0", file=sys.stderr)
    return {0: "unknown"}


def validate_gpu_homogeneity(gpu_ids: List[int], gpu_info: Dict[int, str]) -> bool:
    """Validate that all selected GPUs are of the same model."""
    if len(gpu_ids) <= 1:
        return True

    skus = {gpu_info[gpu_id] for gpu_id in gpu_ids}
    return len(skus) == 1


def make_isolated_gpu_env(gpu_id: int) -> Dict[str, str]:
    """Create environment that isolates subprocess to one physical GPU.

    Sets ROCR_VISIBLE_DEVICES at the HSA/ROCr level, providing complete
    isolation for all higher layers including HIP.
    """
    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    if "HIP_VISIBLE_DEVICES" in env:
        del env["HIP_VISIBLE_DEVICES"]  # Remove HIP_VISIBLE_DEVICES to avoid conflicts
    return env


def verify_mode_flags(verify_mode: str) -> str:
    """Convert verify mode to rocmlir-gen flags."""
    if verify_mode == "none":
        return ""
    if verify_mode == "cpu":
        return " -pv"
    if verify_mode == "gpu":
        return " -pv_with_gpu --verifier-keep-perf-config=false"
    raise ValueError("Unknown verification mode", verify_mode)


def kill_process(proc):
    """Terminate a subprocess and wait for cleanup."""
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print(f"Warning: Process {proc.pid} did not terminate in time after kill", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to kill process {proc.pid}: {e}", file=sys.stderr)


def verify_perfconfig(perfconfig, config, paths: Paths, options: Options, gpu_id: int) -> float:
    """Verify a performance config by running with profiling.

    Returns the execution time in nanoseconds, or NaN if verification fails.
    """
    config.set_perfconfig(perfconfig)

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

    env = make_isolated_gpu_env(gpu_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = None
        p2 = None
        p3 = None
        try:
            p1 = subprocess.Popen(rocmlir_gen_command.split(),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  env=env,
                                  cwd=tmpdir)
            p2 = subprocess.Popen(rocmlir_driver_command,
                                  stdin=p1.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL,
                                  env=env,
                                  cwd=tmpdir)
            p1.stdout.close()
            p3 = subprocess.Popen(profiler_command,
                                  stdin=p2.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=env,
                                  cwd=tmpdir)
            p2.stdout.close()

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


def find_best_perfconfig(tuning_output, config, paths: Paths, options: Options,
                         gpu_id: int) -> tuple[str, float, List[Dict]]:
    """Parse tuning driver output and find the best performing perfconfig.

    Returns the winning config, its TFLOPS, and all entries.
    """
    max_tflops = -np.inf
    winning_config = "None"
    entries = []

    for result in tuning_output:
        result = result.decode('utf-8').strip()
        if not result:
            continue
        try:
            parts = result.split('\t')
            if len(parts) < 2:
                continue  # Skip silently - can happen during normal shutdown
            perfconfig = parts[0]
            time = parts[-1]
            if time == "N/A":
                nano_seconds = np.nan
                measurements = None
            else:
                nano_seconds = float(time)
                measurements = json.loads(parts[1]) if len(parts) == 3 else None
        except ValueError:
            continue  # Skip silently - can happen during normal shutdown

        config.set_perfconfig(perfconfig)
        entry = config.table_entry(nano_seconds)
        if options.debug:
            entry["Measurements"] = measurements
        entries.append(entry)
        these_tflops = entry['TFlops']

        if options.verify_perfconfigs and not np.isnan(nano_seconds):
            try:
                verify_ns = verify_perfconfig(perfconfig, config, paths, options, gpu_id)
            except TuningError as e:
                raise TuningError(
                    f"Error during verification of perf config {perfconfig}\n{str(e)}")
            if np.isnan(verify_ns):
                raise TuningError(f"Verification failed for perf config {perfconfig}")

        if not np.isnan(these_tflops) and these_tflops > max_tflops:
            max_tflops = these_tflops
            winning_config = perfconfig

    return winning_config, max_tflops, entries


def tune_config(test_vector, conf_class, paths: Paths, options: Options, gpu_id: int,
                num_compile_threads: int) -> Dict[str, Any]:
    """Tune a single configuration and return the results."""
    tuning_driver_args = [
        f"--tuning-space={options.tuning_space_kind}", f"--num-iterations={MLIR_N_REPEATS}",
        f"--warmup-iterations={WARMUP_ITERATIONS}", f"--sleep-us={SLEEP_US}",
        f"--show-all-measurements={'true' if options.debug else 'false'}",
        f"--num-compile-threads={num_compile_threads}", "--use-median"
    ]

    env = make_isolated_gpu_env(gpu_id)

    kernel_gen = None
    tuning_driver = None
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
            tuning_driver = subprocess.Popen([paths.mlir_paths.rocmlir_tuning_driver_path] +
                                             tuning_driver_args,
                                             stdin=kernel_gen.stdout,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             env=env)
            kernel_gen.stdout.close()
        else:
            tuning_key = subprocess.Popen(
                [paths.mlir_paths.rocmlir_gen_path, '--emit-tuning-key', test_vector],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env)
            output, _ = tuning_key.communicate()
            result = output.decode('utf-8').strip().split('\t')
            command_line = result[2].split(sep=' ')
            config = conf_class.from_command_line(command_line, options.arch, options.num_cu)
            tuning_driver = subprocess.Popen([paths.mlir_paths.rocmlir_tuning_driver_path] +
                                             tuning_driver_args + [test_vector],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             env=env)

        # Tune, printing progress as we go to avoid CI timeouts
        winning_config, max_tflops, entries = find_best_perfconfig(tuning_driver.stdout, config,
                                                                   paths, options, gpu_id)

    except TuningError as e:
        return {'success': False, 'error': str(e)}
    finally:
        kill_process(kernel_gen)
        kill_process(tuning_driver)

    if tuning_driver.returncode != 0:
        error_msg = f"rocmlir-tuning-driver failed with return code {tuning_driver.returncode}"
        stderr_content = tuning_driver.stderr.read().decode('utf-8').strip()
        if stderr_content:
            error_msg += f"\nstderr:\n{stderr_content}"
        return {'success': False, 'error': error_msg}

    if winning_config == "None":
        return {'success': False, 'error': "No valid perf config found"}

    verify_tflops = None
    if options.verify_mode != "none":
        try:
            verify_ns = verify_perfconfig(winning_config, config, paths, options, gpu_id)
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


def tune_configs(configs, conf_class, paths: Paths, options: Options) -> bool:
    """Tune multiple configurations in parallel across available GPUs."""
    gpu_ids = options.gpu_ids
    num_workers = min(len(gpu_ids), len(configs))
    num_compile_threads = math.ceil((os.cpu_count() or 1) / num_workers)
    # Avoid oversubscribing by leaving one CPU thread free for non-compilation work
    num_compile_threads = max(1, num_compile_threads - 1)

    if not options.quiet:
        print(f"Using {num_workers} GPU(s): {gpu_ids[:num_workers]}", file=sys.stderr)
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
            result = tune_config(test_vector, conf_class, paths, options, gpu_id,
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
    if not options.retune:
        tuned_configs = load_tuned_configs(options)
        if tuned_configs and not options.quiet:
            print(f"Found {len(tuned_configs)} tuned config(s) in {options.output}",
                  file=sys.stderr)

    configs_to_tune = [config for config in configs if config not in tuned_configs]
    num_tuned_configs = len(configs) - len(configs_to_tune)
    if num_tuned_configs and not options.quiet:
        print(f"Skipping {num_tuned_configs} out of {len(configs)} already tuned config(s)",
              file=sys.stderr)

    executor = None
    progress_bar = None
    with OutputFileWriter(options.output, options) as writer:
        with DebugFileWriter(
                f"{options.output}.debug") if options.debug else nullcontext() as debug_writer:
            try:  # No context manager for executor because we need to shutdown with Wait=False
                executor = ThreadPoolExecutor(max_workers=num_workers)
                futures = {
                    executor.submit(tune_task, test_vector): test_vector
                    for test_vector in configs_to_tune
                }

                progress_bar = tqdm(
                    total=len(configs),
                    initial=num_tuned_configs,
                    disable=options.quiet,
                    file=sys.stderr,
                    desc=f"Tuning {conf_class.__name__} ({options.tuning_space_kind})",
                    unit="config",
                    leave=False)

                has_errors = False

                for future in as_completed(futures):
                    result = future.result()
                    progress_bar.update(1)

                    if result.success:
                        writer.write_result(result)
                        if debug_writer:
                            debug_writer.write_entries(result.entries)
                    else:
                        has_errors = True
                        error_message = result.error or "Unknown error"
                        content = f"Error tuning {result.test_vector}\n" + '\n'.join(
                            f"\t{line}" for line in error_message.split('\n'))
                        print(content, file=sys.stderr)
                        writer.write_error(content)
                        if options.abort_on_error:
                            return False

                if has_errors:
                    print("Encountered errors during tuning", file=sys.stderr)
                else:
                    print("Tuning completed successfully", file=sys.stderr)

                return not has_errors
            finally:
                if executor:
                    executor.shutdown(wait=False, cancel_futures=True)
                if progress_bar:
                    progress_bar.close()


def extract_fusion_configs(test_dir, paths: Paths, options: Options) -> Operation:
    """Extract tuning configurations from fusion E2E test files."""
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
    """Entry point. Parses arguments and starts tuning process."""
    if args is None:
        args = sys.argv[1:]

    arch = perfRunner.get_arch()
    num_cu = perfRunner.get_num_cu(perfRunner.get_chip())

    root_dir = str(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
    default_conv_configs = root_dir + '/mlir/utils/jenkins/performance/configs/tier1-conv-configs'

    gpu_info = get_gpu_info()
    available_gpus = sorted(gpu_info.keys())

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

    parser.add_argument("--gpus",
                        type=int,
                        nargs='+',
                        choices=available_gpus,
                        action=UniqueChoicesAction,
                        default=available_gpus,
                        metavar='GPU_ID',
                        help=f"GPUs to use for tuning (available: {available_gpus}, default: all)")

    parsed_args = parser.parse_args(args)

    if parsed_args.verify_perf_configs and parsed_args.verify_mode == "none":
        print(
            "Use of `--verify-perf-configs` is not allowed with `--verify-mode=none`. Please pass `--verify-mode=cpu` or `--verify-mode=gpu`.",
            file=sys.stderr)
        return 1

    if (not validate_gpu_homogeneity(parsed_args.gpus, gpu_info)):
        details = ", ".join(f"GPU {g}: {gpu_info[g]}" for g in parsed_args.gpus)
        print(f"Mixed GPU models not supported for parallel tuning. Found: {details}",
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
        print("rocMLIR build dir was not provided/found", file=sys.stderr)
        return 1

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
                      retune=parsed_args.retune,
                      gpu_ids=parsed_args.gpus)

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
        print("Tuning operation was not provided/found", file=sys.stderr)
        return 1

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

    try:
        return not tune_configs(configs, conf_class, paths, options)
    except KeyboardInterrupt:
        print("Tuning interrupted by user", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
