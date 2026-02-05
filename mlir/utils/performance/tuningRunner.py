#!/usr/bin/env python3
"""Automated performance tuning for rocMLIR generated kernels.

This script tunes MLIR kernels by running them with different performance configurations and selecting the best one based on execution time.

Usage examples:
    python3 tuningRunner.py --op gemm --configs-file=../mlir/utils/performance/configs/tier1-gemm-configs --output=tuning_db.tsv
    python3 tuningRunner.py --op gemm --config="-g 3 -m 1024 -k 769 -n 512 -t f32 -transA 0 -transB 0"
    python3 tuningRunner.py --op conv --tuning-space=quick --config="conv -F 1 -f NCHW -I NCHW -O NCHW -n 256 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1"
    python3 tuningRunner.py --op fusion --test-dir=../mlir/test/fusion/resnet50-e2e --output=tuning_db.tsv
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque

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

MLIR_N_REPEATS = 10
WARMUP_ITERATIONS = 1
SLEEP_US = 100  # 0.1 ms

# =============================================================================
# Configuration & Results
# =============================================================================


@dataclass(frozen=True)
class Options:
    """Configuration options for the tuning process."""
    debug: bool
    tuning_space_kind: str
    quiet: bool
    arch: str
    num_cu: int
    num_chiplets: int
    rocmlir_gen_flags: str
    verify_mode: str
    verify_perfconfigs: bool
    tflops: bool
    output: str
    abort_on_error: bool
    retune: bool
    gpu_ids: List[int]
    num_cpus: Optional[int]
    wait_for_compiles: bool


@dataclass
class TuningResult:
    """Result of tuning a single configuration."""
    test_vector: str
    success: bool
    gpu_id: Optional[int] = None
    winning_config: Optional[str] = None
    max_tflops: Optional[float] = None
    entries: List[Dict] = field(default_factory=list)
    verify_tflops: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
# Exceptions
# =============================================================================


class TuningError(Exception):
    """Raised when tuning or verification fails."""
    pass


# =============================================================================
# System Topology Discovery
# =============================================================================


@dataclass
class Gpu:
    """Information about a GPU."""
    gpu_id: int
    sku: str
    numa_node: int


@dataclass
class GpuTopology:
    """System GPU topology with NUMA mappings."""
    gpus: Dict[int, Gpu]  # GPU ID -> Gpu

    def get_numa_node(self, gpu_id: int) -> int:
        """Get NUMA node for a GPU, defaults to 0 if unknown."""
        if gpu_id in self.gpus:
            return self.gpus[gpu_id].numa_node
        return 0

    def validate_homogeneity(self, gpu_ids: List[int]) -> bool:
        """Validate that all selected GPUs are of the same model."""
        if len(gpu_ids) <= 1:
            return True
        skus = {self.gpus[gpu_id].sku for gpu_id in gpu_ids if gpu_id in self.gpus}
        return len(skus) == 1

    @staticmethod
    def discover() -> 'GpuTopology':
        """Query GPU topology using rocm-smi.

        rocm-smi reports physical device IDs regardless of environment variables
        (e.g., ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES).
        """
        try:
            output = subprocess.check_output(
                ["rocm-smi", "--showproductname", "--showtoponuma", "--json"],
                text=True,
                timeout=10)
            data = json.loads(output)
            gpus = {}
            for key, value in data.items():
                if key.startswith("card"):
                    gpu_id = int(key.replace("card", ""))
                    sku = value.get("Card SKU", "unknown")
                    numa_node_str = value.get("(Topology) Numa Node")
                    numa_node = int(numa_node_str) if numa_node_str is not None else 0
                    gpus[gpu_id] = Gpu(gpu_id=gpu_id, sku=sku, numa_node=numa_node)
            if gpus:
                return GpuTopology(gpus=gpus)
            print("Warning: rocm-smi returned no GPU cards", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Warning: rocm-smi failed with return code {e.returncode}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("Warning: rocm-smi timed out", file=sys.stderr)
        except FileNotFoundError:
            print("Warning: rocm-smi not found in PATH", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse rocm-smi JSON output: {e}", file=sys.stderr)
        except (ValueError, KeyError) as e:
            print(f"Warning: Failed to extract GPU info from rocm-smi output: {e}", file=sys.stderr)

        print("Warning: Could not detect GPUs, defaulting to GPU 0", file=sys.stderr)
        return GpuTopology(gpus={0: Gpu(gpu_id=0, sku="unknown", numa_node=0)})


@dataclass
class NumaTopology:
    """System NUMA topology with CPU mappings."""
    numa_to_cpus: Dict[int, List[int]]  # NUMA node -> list of CPU IDs

    def get_cpus_for_numa_node(self, numa_node: int) -> List[int]:
        """Get CPUs belonging to a NUMA node."""
        return self.numa_to_cpus.get(numa_node, [])

    @staticmethod
    def discover() -> 'NumaTopology':
        """Discover NUMA topology for CPUs.

        Returns a topology where all CPUs are on node 0 if discovery fails or system is non-NUMA.
        """
        numa_to_cpus: Dict[int, List[int]] = {}
        numa_base = "/sys/devices/system/node"

        if os.path.exists(numa_base):
            for entry in os.listdir(numa_base):
                if entry.startswith("node") and entry[4:].isdigit():
                    node_id = int(entry[4:])
                    cpulist_path = os.path.join(numa_base, entry, "cpulist")
                    if os.path.exists(cpulist_path):
                        with open(cpulist_path, 'r') as f:
                            numa_to_cpus[node_id] = NumaTopology._parse_cpu_list(f.read())

        # Fallback: single node with all CPUs
        if not numa_to_cpus:
            numa_to_cpus[0] = list(range(os.cpu_count() or 1))

        return NumaTopology(numa_to_cpus=numa_to_cpus)

    @staticmethod
    def _parse_cpu_list(cpu_list_str: str) -> List[int]:
        """Parse CPU list string like '0-55,112-167' into list of CPU IDs."""
        cpus = []
        for part in cpu_list_str.strip().split(','):
            if '-' in part:
                start, end = part.split('-', 1)
                cpus.extend(range(int(start), int(end) + 1))
            else:
                cpus.append(int(part))
        return cpus


# =============================================================================
# Tuning Infrastructure
# =============================================================================


@dataclass
class TunedConfigsCache:
    """Cache for previously tuned configurations loaded from output file."""
    _results: Dict[str, TuningResult] = field(default_factory=dict)

    def contains(self, test_vector: str) -> bool:
        """Check if a test vector has already been tuned."""
        return test_vector in self._results

    def get(self, test_vector: str) -> Optional[TuningResult]:
        """Get cached result for a test vector."""
        return self._results.get(test_vector)

    def count(self) -> int:
        """Return number of cached configurations."""
        return len(self._results)

    @classmethod
    def from_output_file(cls,
                         filepath: str,
                         tuning_space_kind: str,
                         quiet: bool = False) -> 'TunedConfigsCache':
        """Load previously tuned configurations from an output TSV file.

        The output file has the following structure:
        - Commit lines starting with '# commit: ' indicating the git commit hash of the tuning run
        - Header lines starting with '# ' containing tuning space kind in parentheses
          (e.g., '# arch\tnumCUs\ttestVector\tperfConfig (quick)\tTFlops')
        - Multiple commit and header sections can exist in the same file from different tuning runs
        - Data lines with tab-separated fields following each header
        - Error lines starting with '### ' indicating errors during tuning

        Only data lines under headers matching options.tuning_space_kind are loaded.
        For example, if options.tuning_space_kind='quick', only data under headers containing '(quick)'
        will be loaded, ignoring '(full)' or other sections.
        """
        cache = cls()

        if filepath == '-' or not os.path.exists(filepath):
            return cache

        current_commit = get_git_commit_hash()
        file_commit = current_commit
        matching_tuning_space = False

        try:
            with open(filepath, mode='r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Track commit hash for warning about stale results
                    if line.startswith('# commit: '):
                        file_commit = line[len('# commit: '):].strip()
                        continue

                    # Check if this section header matches our tuning space
                    if line.startswith('# '):
                        matching_tuning_space = f"({tuning_space_kind})" in line
                        if matching_tuning_space and file_commit != current_commit and not quiet:
                            print(
                                f"Warning: Loading tuned configs from different commit "
                                f"(file: {file_commit[:12]}, current: {current_commit[:12]})",
                                file=sys.stderr)
                        continue

                    # Skip error lines and lines from non-matching sections
                    if line.startswith('### ') or not matching_tuning_space:
                        continue

                    # Parse data line
                    fields = line.split('\t')
                    if len(fields) < 4:
                        continue

                    test_vector = fields[2]
                    perf_config = fields[3] if fields[3] else None
                    tflops_value = float(fields[4]) if len(fields) > 4 and fields[4] else None

                    if perf_config and perf_config != "None":
                        cache._results[test_vector] = TuningResult(test_vector=test_vector,
                                                                   success=True,
                                                                   winning_config=perf_config,
                                                                   max_tflops=tflops_value)
        except Exception as e:
            if not quiet:
                print(f"Warning: Failed to load existing tuning results from {filepath}: {e}",
                      file=sys.stderr)

        return cache


@dataclass
class TuningContext:
    """Encapsulates all state and configuration needed for tuning operations."""
    configs: List[str]
    conf_class: type
    paths: Paths
    options: Options
    gpu_topology: GpuTopology
    numa_topology: NumaTopology

    _threads_per_gpu: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Compute optimal thread allocation after initialization."""
        self._threads_per_gpu = self._compute_thread_allocation()

    def _compute_thread_allocation(self) -> Dict[int, int]:
        """Determine how many compile threads each GPU should use based on NUMA topology."""
        # Group GPUs by their NUMA node
        gpus_by_node: Dict[int, List[int]] = {}
        for gpu_id in self.options.gpu_ids:
            node = self.gpu_topology.get_numa_node(gpu_id)
            gpus_by_node.setdefault(node, []).append(gpu_id)

        # Allocate CPUs from each node proportionally to GPUs on that node
        allocation: Dict[int, int] = {}
        for node, gpus_on_node in gpus_by_node.items():
            cpus_on_node = len(self.numa_topology.get_cpus_for_numa_node(node))
            threads_each = max(1, cpus_on_node // len(gpus_on_node))
            for gpu_id in gpus_on_node:
                allocation[gpu_id] = threads_each

        # Apply user-specified CPU limit if provided
        if self.options.num_cpus is not None:
            total_allocated = sum(allocation.values())
            if self.options.num_cpus < total_allocated:
                scale_factor = self.options.num_cpus / total_allocated
                for gpu_id in allocation:
                    allocation[gpu_id] = max(1, int(allocation[gpu_id] * scale_factor))
            elif not self.options.quiet:
                print(
                    f"Note: --num-cpus={self.options.num_cpus} exceeds optimal {total_allocated}, "
                    f"using optimal allocation",
                    file=sys.stderr)

        return allocation

    def get_compile_threads(self, gpu_id: int) -> int:
        """Get the number of compile threads allocated to a GPU."""
        return self._threads_per_gpu.get(gpu_id, 1)

    def print_gpu_summary(self):
        """Print summary of GPU allocation to stderr."""
        if self.options.quiet:
            return
        num_active = len(self.options.gpu_ids)
        print(f"Using {num_active} GPU(s):", file=sys.stderr)
        for gpu_id in self.options.gpu_ids[:num_active]:
            node = self.gpu_topology.get_numa_node(gpu_id)
            threads = self._threads_per_gpu.get(gpu_id, 1)
            print(f"  GPU {gpu_id}: NUMA node {node}, {threads} compile threads", file=sys.stderr)


class GpuWorkerPool:
    """Manages assignment of GPUs to worker threads with NUMA-aware CPU affinity."""

    def __init__(self, ctx: TuningContext):
        self._ctx = ctx
        self._assignment_lock = threading.Lock()
        self._unassigned_gpus = deque(ctx.options.gpu_ids)
        self._worker_state = threading.local()

    @property
    def worker_count(self) -> int:
        """Number of parallel workers (one per GPU)."""
        return len(self._ctx.options.gpu_ids)

    def acquire_gpu_for_thread(self) -> int:
        """Assign a GPU to the calling thread if not already assigned.

        Also pins the thread to CPUs on the GPU's NUMA node for better memory locality.
        Returns the assigned GPU ID.
        """
        if hasattr(self._worker_state, 'assigned_gpu'):
            return self._worker_state.assigned_gpu

        with self._assignment_lock:
            if not self._unassigned_gpus:
                raise RuntimeError("No GPUs available - more workers than GPUs")
            self._worker_state.assigned_gpu = self._unassigned_gpus.popleft()

        self._apply_numa_affinity(self._worker_state.assigned_gpu)
        return self._worker_state.assigned_gpu

    def _apply_numa_affinity(self, gpu_id: int) -> None:
        """Pin current thread to CPUs on the same NUMA node as the GPU."""
        node = self._ctx.gpu_topology.get_numa_node(gpu_id)
        cpu_list = self._ctx.numa_topology.get_cpus_for_numa_node(node)

        if cpu_list:
            try:
                os.sched_setaffinity(0, set(cpu_list))
            except OSError:
                if not self._ctx.options.quiet:
                    print(f"Warning: Could not set CPU affinity for GPU {gpu_id}", file=sys.stderr)

        self._set_memory_policy(node)

    def _set_memory_policy(self, numa_node: int) -> None:
        """Set memory allocation policy to prefer the specified NUMA node."""
        try:
            import ctypes
            libnuma = ctypes.CDLL("libnuma.so.1", mode=ctypes.RTLD_GLOBAL)

            # MPOL_PREFERRED = 1 (prefer allocations on this node, fall back to others)
            # MPOL_BIND = 2 (strict, fail if node unavailable)
            mpol_preferred = 1

            # Create a nodemask with just our node
            nodemask = 1 << numa_node

            # int set_mempolicy(int mode, const unsigned long *nodemask, unsigned long maxnode)
            libnuma.set_mempolicy(mpol_preferred,
                                  ctypes.byref(ctypes.c_ulong(nodemask)),
                                  maxnode=64)
        except (OSError, AttributeError):
            pass  # libnuma not available, rely on first-touch policy


# =============================================================================
# Output Writers
# =============================================================================


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
        columns = [
            'arch', 'numCUs', 'numChiplets', 'testVector',
            f'perfConfig ({self.options.tuning_space_kind})'
        ]
        if self.options.tflops:
            columns.append('TFlops')
        print("# " + "\t".join(columns), file=self.file)

        self.file.flush()
        self.header_written = True

    def write_result(self, result: TuningResult):
        self._write_header()

        fields = [
            self.options.arch,
            str(self.options.num_cu),
            str(self.options.num_chiplets), result.test_vector, result.winning_config or ""
        ]
        if self.options.tflops:
            fields.append(f"{result.max_tflops}" if result.max_tflops else "")
        print("\t".join(fields), file=self.file)

        self.file.flush()

    def write_error(self, content: str):
        self._write_header()
        print('\n'.join(f"### {line}" for line in content.splitlines()), file=self.file)
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


# =============================================================================
# Utilities
# =============================================================================


class TuningArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with custom validation for tuning arguments."""

    def __init__(self, *args, gpu_topology: GpuTopology = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gpu_topology = gpu_topology

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)

        op_type = Operation.from_name(parsed.op)

        if op_type == Operation.FUSION and not parsed.test_dir:
            self.error("argument --op=fusion: requires --test-dir to be specified")

        if parsed.test_dir and op_type != Operation.FUSION:
            self.error("argument --test-dir: only allowed with --op=fusion")

        if parsed.verify_perf_configs and parsed.verify_mode == "none":
            self.error("argument --verify-perf-configs: not allowed with --verify-mode=none")

        if self._gpu_topology and not self._gpu_topology.validate_homogeneity(parsed.gpus):
            details = ", ".join(f"GPU {g}: {self._gpu_topology.gpus[g].sku}" for g in parsed.gpus)
            self.error(f"argument --gpus: mixed GPU models not supported. Found: {details}")

        return parsed


class UniqueChoicesAction(argparse.Action):
    """Argparse action that ensures no duplicate values."""

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != len(set(values)):
            duplicates = [v for v in values if values.count(v) > 1]
            parser.error(
                f"argument {option_string}: duplicate values not allowed: {set(duplicates)}")
        setattr(namespace, self.dest, values)


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              stderr=subprocess.DEVNULL).decode().strip()
        return commit_hash
    except Exception:
        return "unknown"


def set_isolated_gpu_env(env: Dict[str, str], gpu_id: int) -> None:
    """Modify environment to isolate subprocess to one physical GPU.

    Sets ROCR_VISIBLE_DEVICES at the HSA/ROCr level, providing complete isolation for all higher layers including HIP.
    """
    env["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("HIP_VISIBLE_DEVICES", None)  # Remove HIP_VISIBLE_DEVICES to avoid conflicts


def make_isolated_gpu_env(gpu_id: int) -> Dict[str, str]:
    """Create environment that isolates subprocess to one physical GPU."""
    env = os.environ.copy()
    set_isolated_gpu_env(env, gpu_id)
    return env


def verify_mode_flags(verify_mode: str) -> str:
    """Convert verify mode to rocmlir-gen flags."""
    if verify_mode == "none":
        return ""
    if verify_mode == "cpu":
        return "-pv"
    if verify_mode == "gpu":
        return "-pv_with_gpu --verifier-keep-perf-config=false"
    raise ValueError("Unknown verification mode", verify_mode)


def kill_process(proc) -> None:
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


# =============================================================================
# Core Tuning Logic
# =============================================================================


def verify_perfconfig(perfconfig, config, paths: Paths, options: Options, gpu_id: int) -> float:
    """Verify a performance config by running with profiling.

    Returns the execution time in nanoseconds, or NaN if verification fails.
    """
    config.set_perfconfig(perfconfig)

    command_line_options = config.generate_mlir_driver_commandline(options.rocmlir_gen_flags,
                                                                   kernel_repeats=MLIR_N_REPEATS)
    rocmlir_gen_command = [
        paths.mlir_paths.rocmlir_gen_path, '-print-verify-results=summary'
    ] + verify_mode_flags(options.verify_mode).split() + command_line_options.split()

    rocmlir_driver_command = [paths.mlir_paths.rocmlir_driver_path, '-c']

    mlir_cpu_runner_args = [
        '-O2',
        f'--shared-libs={paths.mlir_paths.libmlir_rocm_runtime_path},{paths.mlir_paths.libconv_validation_wrappers_path},{paths.mlir_paths.libmlir_runtime_utils_path}',
        '--entry-point-result=void'
    ]
    rocprof_command = [perfRunner.ROCPROF] + perfRunner.get_metric_args_for_rocprof(
        options.arch) + [
            '--kernel-trace', '--stats', '-f', 'csv', '-o',
            perfRunner.BENCHMARKING_RESULT_FILE_NAME, '--', paths.mlir_paths.cpu_runner_path
        ] + mlir_cpu_runner_args

    verification_pipeline = " | ".join([
        ' '.join(rocmlir_gen_command), ' '.join(rocmlir_driver_command), ' '.join(rocprof_command)
    ])

    debug_info = f"[GPU {gpu_id}] Verification pipeline:\n" + verification_pipeline

    if not options.quiet and options.debug:
        print(debug_info, file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = None
        p2 = None
        p3 = None
        env = make_isolated_gpu_env(gpu_id)
        try:
            p1 = subprocess.Popen(rocmlir_gen_command,
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
            p3 = subprocess.Popen(rocprof_command,
                                  stdin=p2.stdout,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=env,
                                  cwd=tmpdir)
            p2.stdout.close()

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

    for line in tuning_output:
        result = line.strip()
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
        f"--warmup-iterations={WARMUP_ITERATIONS}", "--use-median", f"--sleep-us={SLEEP_US}",
        f"--show-all-measurements={options.debug}", f"--num-compile-threads={num_compile_threads}",
        f"--wait-for-compiles={options.wait_for_compiles}"
    ]

    env = make_isolated_gpu_env(gpu_id)

    rocmlir_gen = None
    tuning_driver = None
    try:
        rocmlir_gen_command = [paths.mlir_paths.rocmlir_gen_path]
        tuning_driver_command = [paths.mlir_paths.rocmlir_tuning_driver_path] + tuning_driver_args
        if not test_vector.endswith(".mlir"):
            command_line = test_vector.split(sep=' ')
            try:
                config = conf_class.from_command_line(command_line, options.arch, options.num_cu,
                                                      options.num_chiplets)
            except ValueError as e:
                return {'success': False, 'error': str(e)}
            test_vector = config.to_command_line()
            command_line_options = config.generate_mlir_driver_commandline(
                options.rocmlir_gen_flags, kernel_repeats=None)
            # Note, we don't need the -ph, this goes to the tuning driver.
            # Because we don't set -ph, kernel_repeats is set to None.
            # This is because the kernel-repeats flag is only supported with host harness or CPU validation.
            rocmlir_gen_command += command_line_options.split()
            rocmlir_gen = subprocess.Popen(rocmlir_gen_command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL,
                                           env=env)
            tuning_driver = subprocess.Popen(tuning_driver_command,
                                             stdin=rocmlir_gen.stdout,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             env=env)
            rocmlir_gen.stdout.close()
            tuning_pipeline = " | ".join(
                [' '.join(rocmlir_gen_command), ' '.join(tuning_driver_command)])
        else:
            rocmlir_gen_command += ['--emit-tuning-key', test_vector]
            tuning_key = subprocess.Popen(rocmlir_gen_command,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          env=env)
            output, _ = tuning_key.communicate()
            if tuning_key.returncode != 0:
                return {
                    'success': False,
                    'error': f"rocmlir-gen failed with return code {tuning_key.returncode}"
                }
            result = output.decode('utf-8').strip().split('\t')
            command_line = result[2].split(sep=' ')
            try:
                config = conf_class.from_command_line(command_line, options.arch, options.num_cu,
                                                      options.num_chiplets)
            except ValueError as e:
                return {'success': False, 'error': str(e)}
            tuning_driver_command += [test_vector]
            tuning_driver = subprocess.Popen(tuning_driver_command,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             env=env)
            tuning_pipeline = ' '.join(tuning_driver_command)

        debug_info = f"[GPU {gpu_id}] Tuning '{test_vector}':\n" + tuning_pipeline

        if not options.quiet and options.debug:
            print(debug_info, file=sys.stderr)

        # Note: communicate waits for process to terminate which might cause CI timeouts if tuning takes too long
        tuning_stdout, tuning_stderr = tuning_driver.communicate()

        if tuning_driver.returncode != 0:
            error_msg = f"rocmlir-tuning-driver failed with return code {tuning_driver.returncode}"
            stderr_content = tuning_stderr.decode('utf-8').strip()
            if stderr_content:
                error_msg += f"\nstderr:\n{stderr_content}"
            return {'success': False, 'error': error_msg}

        tuning_output = tuning_stdout.decode('utf-8').splitlines()
        winning_config, max_tflops, entries = find_best_perfconfig(tuning_output, config, paths,
                                                                   options, gpu_id)
    except TuningError as e:
        return {'success': False, 'error': str(e)}
    finally:
        kill_process(rocmlir_gen)
        kill_process(tuning_driver)

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


def tune_configs(ctx: TuningContext) -> bool:
    """Tune multiple configurations in parallel across available GPUs."""
    # Load cached results unless retuning is forced
    cache = TunedConfigsCache()
    if not ctx.options.retune:
        cache = TunedConfigsCache.from_output_file(ctx.options.output,
                                                   ctx.options.tuning_space_kind, ctx.options.quiet)
        if cache.count() > 0 and not ctx.options.quiet:
            print(f"Found {cache.count()} tuned config(s) in {ctx.options.output}", file=sys.stderr)

    # Filter out already-tuned configs
    pending_configs = [c for c in ctx.configs if not cache.contains(c)]
    skipped_count = len(ctx.configs) - len(pending_configs)
    if skipped_count > 0 and not ctx.options.quiet:
        print(f"Skipping {skipped_count} of {len(ctx.configs)} already tuned config(s)",
              file=sys.stderr)

    if not pending_configs:
        print("All configurations already tuned", file=sys.stderr)
        return True

    pool = GpuWorkerPool(ctx)
    num_workers = min(pool.worker_count, len(ctx.configs))
    ctx.print_gpu_summary()

    def execute_tuning_task(test_vector: str) -> TuningResult:
        try:
            gpu_id = pool.acquire_gpu_for_thread()
            compile_threads = ctx.get_compile_threads(gpu_id)
            result = tune_config(test_vector, ctx.conf_class, ctx.paths, ctx.options, gpu_id,
                                 compile_threads)
            return TuningResult(test_vector=test_vector,
                                success=result.get('success', False),
                                gpu_id=gpu_id,
                                winning_config=result.get('winning_config'),
                                max_tflops=result.get('max_tflops'),
                                entries=result.get('entries', []),
                                verify_tflops=result.get('verify_tflops'),
                                error=result.get('error'))
        except Exception as e:
            return TuningResult(test_vector=test_vector, success=False, error=str(e))

    executor = None
    progress_bar = None
    has_errors = False

    with OutputFileWriter(ctx.options.output, ctx.options) as results_writer:
        with DebugFileWriter(f"{ctx.options.output}.debug") if ctx.options.debug else nullcontext(
        ) as debug_writer:
            try:  # No context manager for executor because we need to shutdown with wait=False
                progress_bar = tqdm(
                    total=len(ctx.configs),
                    initial=skipped_count,
                    disable=ctx.options.quiet,
                    file=sys.stderr,
                    desc=f"Tuning {ctx.conf_class.__name__} ({ctx.options.tuning_space_kind})",
                    unit="config",
                    leave=False)

                executor = ThreadPoolExecutor(max_workers=num_workers)
                pending_futures = {
                    executor.submit(execute_tuning_task, test_vector): test_vector
                    for test_vector in pending_configs
                }

                for completed_future in as_completed(pending_futures):
                    result = completed_future.result()

                    if result.success:
                        results_writer.write_result(result)
                        if debug_writer:
                            debug_writer.write_entries(result.entries)
                        progress_bar.update(1)
                    else:
                        has_errors = True
                        error_text = result.error or "Unknown error"
                        gpu_prefix = f"[GPU {result.gpu_id}] " if result.gpu_id is not None else ""
                        formatted_error = f"{gpu_prefix}Error tuning {result.test_vector}\n" + '\n'.join(
                            f"\t{line}" for line in error_text.splitlines())
                        print(formatted_error, file=sys.stderr)
                        results_writer.write_error(formatted_error)

                        if ctx.options.abort_on_error:
                            return False

                        progress_bar.refresh()

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


# =============================================================================
# Configuration Loading
# =============================================================================


def resolve_paths(op_type: Operation, parsed_args) -> Paths:
    """Resolve paths based on operation type and arguments."""
    if op_type == Operation.FUSION:
        configs_path = "./fusion_config_file"
    else:
        configs_path = None if parsed_args.config else parsed_args.configs_file
    return perfRunner.create_paths(configs_path, parsed_args.mlir_build_dir)


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


def get_config_class(op_type: Operation) -> type:
    """Get the configuration class for an operation type."""
    config_classes = {
        Operation.CONV: ConvConfiguration,
        Operation.GEMM: GemmConfiguration,
        Operation.ATTENTION: AttentionConfiguration,
        Operation.GEMM_GEMM: GemmGemmConfiguration,
        Operation.CONV_GEMM: ConvGemmConfiguration,
    }

    return config_classes.get(op_type, PerfConfiguration)


def load_configs(op_type: Operation, parsed_args, paths: Paths) -> List[str]:
    """Load configurations based on operation type and arguments."""
    if parsed_args.config:
        return parsed_args.config

    loaders = {
        Operation.CONV:
            lambda: perfRunner.get_conv_configurations(paths.configuration_file_path),
        Operation.GEMM:
            lambda: perfRunner.get_gemm_configurations(
                paths.configuration_file_path, *perfRunner.parse_data_types(parsed_args.data_type),
                parsed_args.scale_type),
        Operation.ATTENTION:
            lambda: perfRunner.get_attn_configurations(paths.configuration_file_path),
        Operation.GEMM_GEMM:
            lambda: perfRunner.get_gemm_gemm_configurations(paths.configuration_file_path),
        Operation.CONV_GEMM:
            lambda: perfRunner.get_conv_gemm_configurations(paths.configuration_file_path),
    }

    loader = loaders.get(op_type)
    if loader:
        return loader()

    raise ValueError(f"Unsupported operation type: {str(op_type)}")


# =============================================================================
# Entry Point
# =============================================================================


def parse_arguments(gpu_topology: GpuTopology, available_gpus: List[int], args=None):
    """Parse and validate command-line arguments."""
    parser = TuningArgumentParser(
        prog="tuningRunner.py",
        description="Automated performance tuning for rocMLIR generated kernels",
        allow_abbrev=False,
        gpu_topology=gpu_topology)

    config_group = parser.add_mutually_exclusive_group(required=True)

    config_group.add_argument(
        "-c",
        "--configs-file",
        "--configs_file",  # for backward compatibility
        type=str,
        help="Path to file containing list of configurations to tune")

    config_group.add_argument("--config",
                              type=str,
                              nargs='*',
                              help="Specific config to tune. Format depends on --op type.")

    config_group.add_argument(
        "--test-dir",
        "--test_dir",  # for backward compatibility
        type=str,
        help=
        "Directory containing fusion E2E tests to extract configs from. Only used when --op=fusion."
    )

    parser.add_argument("--op",
                        "--operation",
                        choices=['conv', 'gemm', 'fusion', 'attention', 'gemm_gemm', 'conv_gemm'],
                        required=True,
                        help="Operation to tune")

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

    parser.add_argument(
        "--rocmlir-gen-flags",
        "--rocmlir_gen_flags",  # for backward compatibility
        type=str,
        default="",
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

    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        metavar='N',
        help="Maximum CPU threads for compilation (default: auto-detect based on NUMA topology)")

    parser.add_argument("--wait-for-compiles",
                        action='store_true',
                        default=False,
                        help="Wait for all compilation tasks to complete before starting tuning. "
                        "Useful for systems with shared CPU/GPU memory (e.g., APUs).")

    return parser.parse_args(args)


def main(args=None):
    gpu_topology = GpuTopology.discover()
    available_gpus = sorted(gpu_topology.gpus.keys())

    # We call into perfRunner which also queries GPU info using HIP and rocminfo.
    # To ensure consistency, we isolate the process to the first available GPU.
    set_isolated_gpu_env(os.environ, available_gpus[0])

    parsed_args = parse_arguments(gpu_topology, available_gpus, args)

    op_type = Operation.from_name(parsed_args.op)
    paths = resolve_paths(op_type, parsed_args)

    if not paths.mlir_paths:
        print("rocMLIR build dir was not provided/found", file=sys.stderr)
        return 1

    arch = perfRunner.get_arch()
    chip = perfRunner.get_chip()
    num_cu = perfRunner.get_num_cu(chip)
    num_chiplets = perfRunner.get_num_chiplets(chip, num_cu)

    options = Options(arch=arch,
                      num_cu=num_cu,
                      num_chiplets=num_chiplets,
                      debug=parsed_args.debug,
                      quiet=parsed_args.quiet,
                      tuning_space_kind=parsed_args.tuning_space,
                      rocmlir_gen_flags=parsed_args.rocmlir_gen_flags,
                      verify_mode=parsed_args.verify_mode,
                      verify_perfconfigs=parsed_args.verify_perf_configs,
                      tflops=parsed_args.tflops,
                      output=parsed_args.output,
                      abort_on_error=parsed_args.abort_on_error,
                      retune=parsed_args.retune,
                      gpu_ids=parsed_args.gpus,
                      num_cpus=parsed_args.num_cpus,
                      wait_for_compiles=parsed_args.wait_for_compiles)

    if op_type == Operation.FUSION:
        op_type = extract_fusion_configs(parsed_args.test_dir, paths, options)

    try:
        conf_class = get_config_class(op_type)
        configs = load_configs(op_type, parsed_args, paths)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    ctx = TuningContext(configs=configs,
                        conf_class=conf_class,
                        paths=paths,
                        options=options,
                        gpu_topology=gpu_topology,
                        numa_topology=NumaTopology.discover())

    try:
        tuning_succeeded = tune_configs(ctx)
        return 0 if tuning_succeeded else 1
    except KeyboardInterrupt:
        print("Tuning interrupted by user", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
