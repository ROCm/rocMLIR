#!/usr/bin/env python3
"""Automated performance tuning for rocMLIR generated kernels.

This script tunes MLIR kernels by running them with different performance configurations and selecting the best one based on execution time.

Usage examples:
    # Tune GEMM configs from a file
    python3 tuningRunner.py --op gemm -c configs/tier1-gemm-configs -o tuning_db.tsv

    # Tune a single GEMM config
    python3 tuningRunner.py --op gemm --config "-g 3 -m 1024 -k 769 -n 512 -t f32 -transA 0 -transB 0"

    # Quick-tune CONV configs from a file
    python3 tuningRunner.py --op conv -c configs/tier1-conv-configs --tuning-space quick

    # Use a subset of available GPUs
    python3 tuningRunner.py --op gemm -c configs/tier1-gemm-configs --gpus 2 3

    # Tune fusion ops from E2E test directory
    python3 tuningRunner.py --op fusion --test-dir ../mlir/test/fusion/resnet50-e2e

    # Pipe configs from stdin
    cat configs/tier1-gemm-configs | python3 tuningRunner.py --op gemm -c - -o tuning_db.tsv
"""

import argparse
import glob
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import deque

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

# =============================================================================
# Constants
# =============================================================================

MLIR_N_REPEATS = 10
WARMUP_ITERATIONS = 1
SLEEP_US = 100  # 0.1 ms
MAX_FAILURES = 20

# =============================================================================
# Logging Setup
# =============================================================================

# ANSI color codes
_LOG_COLORS = {
    logging.DEBUG: '\033[36m',  # Cyan
    logging.INFO: '\033[34m',  # Blue
    logging.WARNING: '\033[33m',  # Yellow
    logging.ERROR: '\033[91m',  # Red
    logging.CRITICAL: '\033[91m',  # Red
}
_COLOR_RESET = '\033[0m'


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid corrupting progress bars."""

    def __init__(self, use_color: bool = False):
        super().__init__()
        self.use_color = use_color

    def emit(self, record):
        try:
            msg = record.getMessage()
            levelname = record.levelname

            if self.use_color:
                color = _LOG_COLORS.get(record.levelno, '')
                prefix = f"{color}{levelname}{_COLOR_RESET}: "
            else:
                prefix = f"{levelname}: "

            indent = ' ' * 4
            lines = msg.splitlines()
            if len(lines) == 1:
                formatted = prefix + lines[0]
            else:
                formatted = prefix + lines[0] + '\n' + '\n'.join(
                    indent + line for line in lines[1:])

            tqdm.write(formatted, file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logger(quiet: bool = False, verbose: bool = False) -> logging.Logger:
    """Configure and return a logger for tuningRunner."""
    log = logging.getLogger("tuningRunner")

    if quiet:
        log.setLevel(logging.ERROR)
    elif verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    log.handlers.clear()

    use_color = sys.stderr.isatty()
    handler = TqdmLoggingHandler(use_color=use_color)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    log.addHandler(handler)

    return log


# Module-level logger
logger: logging.Logger = setup_logger()

# =============================================================================
# Configuration & Results
# =============================================================================


@dataclass(frozen=True)
class Options:
    """Configuration options for the tuning process."""
    debug: bool
    tuning_space_kind: str
    quiet: bool
    verbose: bool
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
    retry_failed: bool
    gpu_ids: List[int]
    num_cpus: Optional[int]
    wait_for_compiles: bool


@dataclass
class TuningResult:
    """Result of tuning a single configuration."""
    test_vector: str
    success: bool
    gpu_id: int
    elapsed_seconds: float
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

        rocm-smi reports physical device IDs regardless of environment variables (e.g., ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES).
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
            logger.warning("rocm-smi returned no GPU cards")
        except subprocess.CalledProcessError as e:
            logger.warning(f"rocm-smi failed with return code {e.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning("rocm-smi timed out")
        except FileNotFoundError:
            logger.warning("rocm-smi not found in PATH")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse rocm-smi JSON output: {e}")
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to extract GPU info from rocm-smi output: {e}")

        logger.warning("Could not detect GPUs, defaulting to GPU 0")
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
                    else:
                        logger.warning(f"Missing cpulist for NUMA node {node_id}")

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
# State Management
# =============================================================================


class ConfigState(Enum):
    """Possible states for a tuning configuration in the state file.

    State transitions:
        PENDING (implicit) -> RUNNING: Config starts tuning
        RUNNING -> SUCCESS (implicit): Tuning completes successfully (removed from state, written to output)
        RUNNING -> FAILED: Tuning completes with error
        RUNNING -> INTERRUPTED: User interrupted (Ctrl+C) during tuning
        RUNNING -> CRASHED: Detected on next startup (stale RUNNING state)
        FAILED/CRASHED -> PENDING: User requests retry with --retry-failed

    Note: PENDING and SUCCESS are implicit states:
        - PENDING: not in state file AND not in output file
        - SUCCESS: in output file (not tracked in state file)
    """
    RUNNING = "running"  # Currently being tuned
    FAILED = "failed"  # Tuning completed with error
    INTERRUPTED = "interrupted"  # User interrupted during tuning (Ctrl+C)
    CRASHED = "crashed"  # Process crashed while tuning (detected on startup)


@dataclass
class TuningStateContext:
    """Context that identifies a tuning run. State is invalidated if context changes."""
    arch: str
    num_cu: int
    tuning_space: str

    def matches(self, other: 'TuningStateContext') -> bool:
        return (self.arch == other.arch and self.num_cu == other.num_cu and
                self.tuning_space == other.tuning_space)


@dataclass
class TuningState:
    """Persistent state for tuning runs, survives crashes and interrupts."""
    context: TuningStateContext
    configs: Dict[str, ConfigState] = field(default_factory=dict)

    def set_running(self, test_vector: str) -> None:
        """Mark a config as currently running."""
        self.configs[test_vector] = ConfigState.RUNNING

    def set_failed(self, test_vector: str) -> None:
        """Mark a config as failed."""
        self.configs[test_vector] = ConfigState.FAILED

    def set_interrupted(self, test_vector: str) -> None:
        """Mark a config as interrupted by user."""
        self.configs[test_vector] = ConfigState.INTERRUPTED

    def set_crashed(self, test_vector: str) -> None:
        """Mark a config as crashed."""
        self.configs[test_vector] = ConfigState.CRASHED

    def remove(self, test_vector: str) -> None:
        """Remove a config from state (e.g., on success)."""
        self.configs.pop(test_vector, None)

    def should_skip(self, test_vector: str) -> bool:
        """Check if a config should be skipped (failed or crashed)."""
        return self.configs.get(test_vector) in (ConfigState.FAILED, ConfigState.CRASHED)

    def _count_by_state(self, *states: ConfigState) -> int:
        """Count configs in any of the given states."""
        return sum(1 for s in self.configs.values() if s in states)

    def failed_count(self) -> int:
        """Count of failed configs."""
        return self._count_by_state(ConfigState.FAILED)

    def crashed_count(self) -> int:
        """Count of crashed configs."""
        return self._count_by_state(ConfigState.CRASHED)

    def skip_count(self) -> int:
        """Count of configs that should be skipped (failed + crashed)."""
        return self._count_by_state(ConfigState.FAILED, ConfigState.CRASHED)

    def promote_running_to_interrupted(self) -> int:
        """Move all RUNNING configs to INTERRUPTED (clean shutdown). Returns count."""
        count = 0
        for tv in self.configs:
            if self.configs[tv] == ConfigState.RUNNING:
                self.configs[tv] = ConfigState.INTERRUPTED
                count += 1
        return count


class TuningStateFile:
    """Manages reading and writing of tuning state to a JSON file.

    If filepath is None, all operations are no-ops (null object pattern).
    """

    def __init__(self, filepath: Optional[str]):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._state: Optional[TuningState] = None

    def load(self, expected_context: TuningStateContext) -> 'TuningStateFile':
        """Load state from file. Returns self for chaining.

        On load:
        - INTERRUPTED configs are demoted to PENDING (removed from state)
        - RUNNING configs are promoted to CRASHED (indicates previous crash)
        """
        if not self.filepath:
            self._state = TuningState(context=expected_context)
            return self

        if not os.path.exists(self.filepath):
            self._state = TuningState(context=expected_context)
            return self

        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)

            file_context = TuningStateContext(arch=data.get('arch', ''),
                                              num_cu=data.get('numCUs', 0),
                                              tuning_space=data.get('tuningSpace', ''))

            if not file_context.matches(expected_context):
                logger.warning("State file context mismatch, starting fresh")
                self._state = TuningState(context=expected_context)
                return self

            configs = {}
            for tv, state_str in data.get('configs', {}).items():
                try:
                    config_state = ConfigState(state_str)
                    # Demote INTERRUPTED to PENDING (don't add to configs)
                    if config_state == ConfigState.INTERRUPTED:
                        continue
                    # Promote RUNNING to CRASHED (stale running = crash)
                    if config_state == ConfigState.RUNNING:
                        config_state = ConfigState.CRASHED
                    configs[tv] = config_state
                except ValueError:
                    pass  # Skip invalid states

            self._state = TuningState(context=expected_context, configs=configs)
            return self

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load state file: {e}")
            self._state = TuningState(context=expected_context)
            return self

    @property
    def state(self) -> TuningState:
        """Get the current state. Must call load() first."""
        if self._state is None:
            raise RuntimeError("State not loaded. Call load() first.")
        return self._state

    def _save_locked(self) -> None:
        """Save state to file atomically. Assumes lock is held."""
        if not self.filepath or not self._state:
            return

        data = {
            'arch': self._state.context.arch,
            'numCUs': self._state.context.num_cu,
            'tuningSpace': self._state.context.tuning_space,
            'configs': {
                tv: s.value for tv, s in self._state.configs.items()
            }
        }

        # Write to temp file then rename for atomicity
        temp_path = self.filepath + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, self.filepath)

    def save(self) -> None:
        """Save state to file atomically. No-op if filepath is None."""
        with self._lock:
            self._save_locked()

    def delete(self) -> None:
        """Delete the state file. No-op if filepath is None."""
        if not self.filepath:
            return

        with self._lock:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            self._state = None

    def set_running(self, test_vector: str) -> None:
        """Mark a config as running and save."""
        if self._state:
            with self._lock:
                self._state.set_running(test_vector)
                self._save_locked()

    def set_failed(self, test_vector: str) -> None:
        """Mark a config as failed and save."""
        if self._state:
            with self._lock:
                self._state.set_failed(test_vector)
                self._save_locked()

    def set_success(self, test_vector: str) -> None:
        """Remove a config from state (success) and save."""
        if self._state:
            with self._lock:
                self._state.remove(test_vector)
                self._save_locked()

    def finalize_interrupted(self) -> None:
        """Mark any RUNNING configs as INTERRUPTED and save. Called on clean shutdown."""
        if self._state:
            with self._lock:
                interrupted_count = self._state.promote_running_to_interrupted()
                if interrupted_count > 0:
                    logger.info(f"Marked {interrupted_count} running config(s) as interrupted")
                self._save_locked()


def get_state_filepath(output_filepath: str) -> Optional[str]:
    """Get the state file path for a given output file."""
    if output_filepath == '-':
        return None
    return f"{output_filepath}.state"


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

    def get_all_results(self) -> List[TuningResult]:
        """Get all cached tuning results."""
        return list(self._results.values())

    def count(self) -> int:
        """Return number of cached configurations."""
        return len(self._results)

    @classmethod
    def from_output_file(cls, options: Options) -> 'TunedConfigsCache':
        """Load previously tuned configurations from an output TSV file.

        Format: # arch\tnumCUs\ttestVector\tperfConfig (tuning_space)\t[TFlops]\t[elapsedSeconds]
        Only loads entries matching current arch, numCUs, and tuning space.
        """
        cache = cls()

        if options.output == '-' or not os.path.exists(options.output):
            return cache

        current_commit = get_git_commit_hash()

        # Active section state
        metadata: Dict[str, Optional[Any]] = {}
        matching_section = False
        column_indices: Dict[str, int] = {}

        try:
            with open(options.output, mode='r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for metadata line
                    if line.startswith('## '):
                        parts = line[3:].split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            metadata[key] = value
                        continue

                    # Check for header line
                    if cls._is_header_line(line):
                        # Determine if this section matches based on tuning space
                        matching_section = f'({options.tuning_space_kind})' in line

                        if matching_section:
                            column_indices = cls._parse_header_line(line)

                            # Warn if commit hashes differ
                            file_commit = metadata.get('commit', 'unknown')
                            if file_commit != current_commit:
                                logger.warning(
                                    f"Loading tuned configs from different commit (file: {file_commit[:8]}, current: {current_commit[:8]})"
                                )

                        # Reset metadata for next section
                        metadata = {}
                        continue

                    # Skip other comment lines
                    if line.startswith('#'):
                        continue

                    # Skip data lines from non-matching sections
                    if not matching_section or not column_indices:
                        continue

                    # Parse data line
                    result = cls._parse_data_line(line.split('\t'), column_indices, options.arch,
                                                  options.num_cu)
                    if result:
                        cache._results[result.test_vector] = result

        except Exception as e:
            logger.warning(f"Failed to load existing tuning results from {options.output}: {e}")

        return cache

    @staticmethod
    def _is_header_line(line: str) -> bool:
        """Check if line is a column header."""
        return line.startswith('# arch\t')

    @staticmethod
    def _parse_header_line(line: str) -> Dict[str, int]:
        """Parse column header and return name -> index mapping."""
        # Strip leading '# ' if present
        header_text = line[2:] if line.startswith('# ') else line
        indices = {}
        for i, col in enumerate(header_text.split('\t')):
            if col:
                # Exctract base column name (handles 'perfConfig (tuning_space)')
                col_name = col.split()[0]
                indices[col_name] = i
        return indices

    @staticmethod
    def _parse_data_line(fields: List[str], column_indices: Dict[str, int], arch: str,
                         num_cu: int) -> Optional[TuningResult]:
        """Parse a data line and return TuningResult if valid.

        A line is valid if:
        - arch and numCUs match current system (if columns exist, for old format)
        - testVector is present
        - perfConfig is present and not 'None'
        - TFlops is a valid finite number (if column exists)
        """

        def get_field(name: str) -> Optional[str]:
            idx = column_indices.get(name)
            if idx is not None and idx < len(fields) and fields[idx]:
                return fields[idx]
            return None

        if get_field('arch') != arch:
            return None
        if get_field('numCUs') != str(num_cu):
            return None

        test_vector = get_field('testVector')
        if not test_vector:
            return None

        perf_config = get_field('perfConfig')
        if not perf_config or perf_config == 'None':
            return None

        max_tflops = None
        if 'TFlops' in column_indices:
            tflops_str = get_field('TFlops')
            if not tflops_str:
                return None
            try:
                tflops_val = float(tflops_str)
                if np.isnan(tflops_val) or np.isinf(tflops_val):
                    return None
                max_tflops = tflops_val
            except ValueError:
                return None

        elapsed_seconds = 0.0
        elapsed_str = get_field('elapsedSeconds')
        if elapsed_str:
            try:
                elapsed_seconds = float(elapsed_str)
            except ValueError:
                pass

        return TuningResult(test_vector=test_vector,
                            success=True,
                            gpu_id=-1,
                            elapsed_seconds=elapsed_seconds,
                            winning_config=perf_config,
                            max_tflops=max_tflops)


@dataclass
class ETATracker:
    """Track completion times for accurate ETA estimation using median of successful configs."""
    total_configs: int
    num_workers: int
    initial_times: List[float] = field(default_factory=list)
    initial_ok_count: int = 0
    initial_fail_count: int = 0
    _success_times: List[float] = field(default_factory=list, init=False)
    _processed: int = field(default=0, init=False)
    _ok_count: int = field(default=0, init=False)
    _fail_count: int = field(default=0, init=False)

    def __post_init__(self):
        self._success_times = list(self.initial_times)
        self._ok_count = self.initial_ok_count
        self._fail_count = self.initial_fail_count

    def record(self, result: TuningResult) -> None:
        self._processed += 1
        if result.success:
            self._ok_count += 1
            self._success_times.append(result.elapsed_seconds)
        else:
            self._fail_count += 1

    def _format_rate(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s/cfg"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m/cfg"
        else:
            return f"{seconds / 3600:.1f}h/cfg"

    def _format_eta(self, seconds: float) -> str:
        if seconds < 60:
            return "<1m"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d{hours}h"

    def get_postfix_str(self) -> str:
        remaining = self.total_configs - self._processed

        rate = "n/a"
        eta = "n/a"
        if len(self._success_times) >= 3:
            median = statistics.median(self._success_times)
            eta_seconds = (remaining / self.num_workers) * median
            rate = self._format_rate(median)
            eta = self._format_eta(eta_seconds)

        return f"ok={self._ok_count}, fail={self._fail_count}, rate={rate}, eta={eta}"


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
            else:
                logger.info(
                    f"--num-cpus={self.options.num_cpus} exceeds optimal {total_allocated}, using optimal allocation"
                )

        return allocation

    def get_compile_threads(self, gpu_id: int) -> int:
        """Get the number of compile threads allocated to a GPU."""
        return self._threads_per_gpu.get(gpu_id, 1)

    def print_gpu_summary(self):
        """Print summary of GPU allocation."""
        num_active = len(self.options.gpu_ids)
        lines = [f"Using {num_active} GPU(s)"]
        for gpu_id in self.options.gpu_ids[:num_active]:
            node = self.gpu_topology.get_numa_node(gpu_id)
            threads = self._threads_per_gpu.get(gpu_id, 1)
            lines.append(f"GPU {gpu_id}: NUMA node {node}, {threads} compile threads")
        logger.info("\n".join(lines))


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
                logger.warning(f"Could not set CPU affinity for GPU {gpu_id}")

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
        self._header_written = False
        self._is_appending = False

    def __enter__(self):
        if self.filepath == '-':
            self.file = sys.stdout
        else:
            self._is_appending = os.path.exists(self.filepath) and os.path.getsize(
                self.filepath) > 0
            self.file = open(self.filepath, 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file and self.file != sys.stdout:
            self.file.close()

    def _write_header(self):
        if self._header_written:
            return

        if self._is_appending:
            print("", file=self.file)  # Blank line before new section

        # Metadata comments
        print(f"## commit: {get_git_commit_hash()}", file=self.file)

        # TSV header with '# ' prefix
        columns = [
            'arch', 'numCUs', 'numChiplets', 'testVector',
            f'perfConfig ({self.options.tuning_space_kind})'
        ]
        if self.options.tflops:
            columns.append('TFlops')
        columns.append('elapsedSeconds')
        print("# " + "\t".join(columns), file=self.file)

        self.file.flush()
        self._header_written = True

    def write_result(self, result: TuningResult):
        assert result.success and result.winning_config and result.max_tflops, "write_result called with failed result"

        self._write_header()

        fields = [
            self.options.arch,
            str(self.options.num_cu),
            str(self.options.num_chiplets), result.test_vector, result.winning_config
        ]
        if self.options.tflops:
            fields.append(str(result.max_tflops))
        fields.append(f"{result.elapsed_seconds:.1f}")
        print("\t".join(fields), file=self.file)

        self.file.flush()


class DebugFileWriter:
    """Context manager for writing debug entries to TSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self._header_written = False

    def __enter__(self):
        self._header_written = os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0
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
                                     header=not self._header_written,
                                     index=False)

        self.file.flush()
        self._header_written = True


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
        logger.warning(f"Process {proc.pid} did not terminate in time after kill")
    except Exception as e:
        logger.warning(f"Failed to kill process {proc.pid}: {e}")


def format_error(context: str,
                 command: str = None,
                 stdout: str = None,
                 stderr: str = None,
                 exit_code: int = None,
                 gpu_id: int = None,
                 max_lines: int = 10) -> str:
    """Format an error message with optional details."""

    def truncate(text: str) -> str:
        if not text or not text.strip():
            return None
        lines = text.strip().splitlines()
        if len(lines) <= max_lines:
            return text.strip()
        half = max_lines // 2
        return '\n'.join(lines[:half] + [f'... ({len(lines) - max_lines} lines omitted) ...'] +
                         lines[-half:])

    parts = [context]

    if exit_code is not None:
        parts.append(f"Exit code: {exit_code}")

    if command:
        if gpu_id is not None:
            parts.append(f"Reproduce: ROCR_VISIBLE_DEVICES={gpu_id} {command}")
        else:
            parts.append(f"Reproduce: {command}")

    truncated_stdout = truncate(stdout)
    if truncated_stdout:
        parts.append("stdout:\n" + truncated_stdout)

    truncated_stderr = truncate(stderr)
    if truncated_stderr:
        parts.append("stderr:\n" + truncated_stderr)

    return '\n'.join(parts)


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
    logger.debug(f"[GPU {gpu_id}] Verifying perfconfig '{perfconfig}'\n{verification_pipeline}")

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
                    raise TuningError(
                        format_error(f"Verification failed for perfconfig '{perfconfig}'",
                                     command=verification_pipeline,
                                     stdout=outs,
                                     stderr=errs.decode('utf-8'),
                                     exit_code=p3.returncode,
                                     gpu_id=gpu_id))

            except subprocess.TimeoutExpired:
                kill_process(p3)
                outs, errs = p3.communicate()
                raise TuningError(
                    format_error(f"Verification timed out for perfconfig '{perfconfig}'",
                                 command=verification_pipeline,
                                 stdout=outs.decode('utf-8'),
                                 stderr=errs.decode('utf-8'),
                                 gpu_id=gpu_id))

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
            verify_ns = verify_perfconfig(perfconfig, config, paths, options, gpu_id)
            if np.isnan(verify_ns):
                raise TuningError(f"Verification returned NaN for perfconfig '{perfconfig}'")

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
            output, err = tuning_key.communicate()
            if tuning_key.returncode != 0:
                return {
                    'success':
                        False,
                    'error':
                        format_error("Failed to generate tuning key",
                                     command=' '.join(rocmlir_gen_command),
                                     stderr=err.decode('utf-8'),
                                     exit_code=tuning_key.returncode,
                                     gpu_id=gpu_id)
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

        logger.debug(f"[GPU {gpu_id}] Tuning '{test_vector}'\n{tuning_pipeline}")

        # Note: communicate waits for process to terminate which might cause CI timeouts if tuning takes too long
        tuning_stdout, tuning_stderr = tuning_driver.communicate()

        if tuning_driver.returncode != 0:
            return {
                'success':
                    False,
                'error':
                    format_error("Tuning pipeline failed",
                                 command=tuning_pipeline,
                                 stderr=tuning_stderr.decode('utf-8'),
                                 exit_code=tuning_driver.returncode,
                                 gpu_id=gpu_id)
            }
        else:
            # Log any stderr output from tuning driver because it may contain warnings
            tuning_stderr_str = tuning_stderr.decode('utf-8').strip()
            if tuning_stderr_str:
                logger.debug(f"[GPU {gpu_id}] rocmlir-tuning-driver stderr:\n{tuning_stderr_str}")

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
            return {'success': False, 'error': str(e)}

        if np.isnan(verify_ns):
            return {
                'success': False,
                'error': f"Verification returned NaN for winning perfconfig '{winning_config}'"
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
        cache = TunedConfigsCache.from_output_file(ctx.options)
        if cache.count() > 0:
            logger.info(f"Found {cache.count()} tuned config(s) in {ctx.options.output}")

    # Load state file
    state_context = TuningStateContext(arch=ctx.options.arch,
                                       num_cu=ctx.options.num_cu,
                                       tuning_space=ctx.options.tuning_space_kind)
    state_file = TuningStateFile(get_state_filepath(ctx.options.output))
    state_file.load(state_context)
    state = state_file.state

    crashed_count = state.crashed_count()
    if crashed_count > 0:
        logger.warning(f"Detected {crashed_count} crashed config(s) from previous run")

    failed_count = state.failed_count()
    if failed_count > 0:
        logger.info(f"Found {failed_count} failed config(s) in state file")

    state_file.save()

    # Filter out already-tuned configs (unless --retune)
    pending_configs = ctx.configs
    skipped_success = 0
    if not ctx.options.retune:
        pending_configs = [c for c in pending_configs if not cache.contains(c)]
        skipped_success = len(ctx.configs) - len(pending_configs)

    # Filter out failed/crashed configs (unless --retry-failed or --retune)
    skipped_failed = 0
    if not ctx.options.retry_failed and not ctx.options.retune:
        before_filter = len(pending_configs)
        pending_configs = [c for c in pending_configs if not state.should_skip(c)]
        skipped_failed = before_filter - len(pending_configs)

    total_skipped = skipped_success + skipped_failed

    if skipped_success > 0:
        logger.info(f"Skipping {skipped_success} already tuned config(s)")
    if skipped_failed > 0:
        logger.info(f"Skipping {skipped_failed} failed/crashed config(s)")

    if not pending_configs:
        logger.info("No configurations to tune")
        return True

    pool = GpuWorkerPool(ctx)
    num_workers = min(pool.worker_count, len(ctx.configs))
    ctx.print_gpu_summary()

    # Prepare ETA tracker with historical data
    initial_times = [r.elapsed_seconds for r in cache.get_all_results() if r.elapsed_seconds > 0.0]
    eta_tracker = ETATracker(total_configs=len(pending_configs),
                             num_workers=num_workers,
                             initial_times=initial_times,
                             initial_ok_count=skipped_success,
                             initial_fail_count=skipped_failed)

    def execute_tuning_task(test_vector: str) -> TuningResult:
        gpu_id = pool.acquire_gpu_for_thread()

        state_file.set_running(test_vector)

        start_time = time.time()
        compile_threads = ctx.get_compile_threads(gpu_id)
        result = tune_config(test_vector, ctx.conf_class, ctx.paths, ctx.options, gpu_id,
                             compile_threads)
        return TuningResult(test_vector=test_vector,
                            success=result.get('success', False),
                            gpu_id=gpu_id,
                            elapsed_seconds=time.time() - start_time,
                            winning_config=result.get('winning_config'),
                            max_tflops=result.get('max_tflops'),
                            entries=result.get('entries', []),
                            verify_tflops=result.get('verify_tflops'),
                            error=result.get('error'))

    with OutputFileWriter(ctx.options.output, ctx.options) as results_writer:
        with DebugFileWriter(f"{ctx.options.output}.debug") if ctx.options.debug else nullcontext(
        ) as debug_writer:

            executor = None
            progress_bar = None

            try:  # No context manager for executor because we need to shutdown with wait=False
                progress_bar = tqdm(
                    total=len(ctx.configs),
                    initial=total_skipped,
                    disable=ctx.options.quiet or not sys.stderr.isatty(),
                    file=sys.stderr,
                    desc=f"Tuning {ctx.conf_class.__name__} ({ctx.options.tuning_space_kind})",
                    unit="config",
                    leave=False,
                    bar_format=
                    '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [t={elapsed}{postfix}]')
                progress_bar.set_postfix_str(eta_tracker.get_postfix_str())

                executor = ThreadPoolExecutor(max_workers=num_workers)
                pending_futures = {
                    executor.submit(execute_tuning_task, test_vector): test_vector
                    for test_vector in pending_configs
                }

                has_errors = False
                consecutive_failures = 0

                for completed_future in as_completed(pending_futures):
                    result = completed_future.result()

                    if result.success:
                        consecutive_failures = 0
                        results_writer.write_result(result)
                        if debug_writer:
                            debug_writer.write_entries(result.entries)
                        state_file.set_success(result.test_vector)
                    else:
                        has_errors = True
                        consecutive_failures += 1
                        state_file.set_failed(result.test_vector)

                        error_msg = f"[GPU {result.gpu_id}] Tuning failed for '{result.test_vector}'"
                        if result.error:
                            error_msg += "\n" + result.error
                        logger.error(error_msg)

                        if ctx.options.abort_on_error:
                            return False

                        if consecutive_failures >= MAX_FAILURES:
                            logger.error("Aborting due to too many consecutive failures")
                            return False

                    eta_tracker.record(result)
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(eta_tracker.get_postfix_str())

            except KeyboardInterrupt:
                logger.info("Tuning interrupted by user")
                raise
            finally:
                if executor:
                    executor.shutdown(wait=False, cancel_futures=True)
                if progress_bar:
                    progress_bar.close()

                state_file.finalize_interrupted()

    if has_errors:
        logger.error("Encountered errors during tuning")
    else:
        logger.info("Tuning completed successfully")

    return not has_errors


# =============================================================================
# Configuration Loading
# =============================================================================


def resolve_paths(op_type: Operation, parsed_args) -> Paths:
    """Resolve paths based on operation type and arguments."""
    if op_type == Operation.FUSION:
        configs_path = "./fusion_config_file"
    elif parsed_args.config:
        configs_path = None
    else:
        configs_path = parsed_args.configs_file
    return perfRunner.create_paths(configs_path, parsed_args.mlir_build_dir)


def extract_fusion_configs(test_dir, paths: Paths) -> Operation:
    """Extract tuning configurations from fusion E2E test files."""
    all_configs = []
    op_type = Operation.FUSION
    for filename in glob.glob(test_dir + '/*mlir'):
        logger.info(f"Extract from: {filename}")
        test_entry = perfRunner.get_fusion_test_info(filename, paths)
        if not test_entry:
            continue
        test_vector = test_entry['testVector']
        if not test_vector:
            continue
        if test_vector in all_configs:
            logger.info("An entry already exists in the tuning DB")
            continue
        command_line = test_vector.split(sep=' ')
        if command_line[0].startswith('conv'):
            if op_type == Operation.FUSION:
                op_type = Operation.CONV
            elif op_type != Operation.CONV:
                logger.warning(f"Invalid config op: {test_vector}")
                continue
        else:
            if op_type == Operation.FUSION:
                op_type = Operation.GEMM
            elif op_type != Operation.GEMM:
                logger.warning(f"Invalid config op: {test_vector}")
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


def load_configs_from_stdin() -> str:
    """Read configs from stdin and return path to a temporary file."""
    content = sys.stdin.read()
    fd, path = tempfile.mkstemp(suffix='.txt', prefix='tuning_configs_')
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


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

    raise ValueError(f"Unsupported operation type: {op_type}")


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

    parser.add_argument("-q",
                        "--quiet",
                        action='store_true',
                        default=False,
                        help="Suppress non-error output")

    parser.add_argument("-v",
                        "--verbose",
                        action='store_true',
                        default=False,
                        help="Enable verbose output, including commands being executed")

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

    parser.add_argument("--retry-failed",
                        action='store_true',
                        default=False,
                        help="Retry previously failed/crashed configs instead of skipping them")

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

    parser.add_argument(
        "--wait-for-compiles",
        action='store_true',
        default=False,
        help=
        "Wait for all compilation tasks to complete before starting tuning. Useful for systems with shared CPU/GPU memory (e.g., APUs)."
    )

    return parser.parse_args(args)


def main(args=None):
    global logger

    gpu_topology = GpuTopology.discover()
    available_gpus = sorted(gpu_topology.gpus.keys())

    # We call into perfRunner which also queries GPU info using HIP and rocminfo.
    # To ensure consistency, we isolate the process to the first available GPU.
    set_isolated_gpu_env(os.environ, available_gpus[0])

    parsed_args = parse_arguments(gpu_topology, available_gpus, args)

    logger = setup_logger(quiet=parsed_args.quiet, verbose=parsed_args.verbose)

    stdin_temp_file = None
    try:
        # Handle stdin for configs file
        if parsed_args.configs_file == '-':
            stdin_temp_file = load_configs_from_stdin()
            parsed_args.configs_file = stdin_temp_file

        op_type = Operation.from_name(parsed_args.op)
        paths = resolve_paths(op_type, parsed_args)

        if not paths.mlir_paths:
            logger.error("rocMLIR build dir was not provided/found")
            return 1

        if op_type == Operation.FUSION:
            op_type = extract_fusion_configs(parsed_args.test_dir, paths)

        conf_class = get_config_class(op_type)
        configs = load_configs(op_type, parsed_args, paths)

    finally:
        if stdin_temp_file:
            os.unlink(stdin_temp_file)

    arch = perfRunner.get_arch()
    chip = perfRunner.get_chip()
    num_cu = perfRunner.get_num_cu(chip)
    num_chiplets = perfRunner.get_num_chiplets(chip, num_cu)

    options = Options(arch=arch,
                      num_cu=num_cu,
                      num_chiplets=num_chiplets,
                      debug=parsed_args.debug,
                      quiet=parsed_args.quiet,
                      verbose=parsed_args.verbose,
                      tuning_space_kind=parsed_args.tuning_space,
                      rocmlir_gen_flags=parsed_args.rocmlir_gen_flags,
                      verify_mode=parsed_args.verify_mode,
                      verify_perfconfigs=parsed_args.verify_perf_configs,
                      tflops=parsed_args.tflops,
                      output=parsed_args.output,
                      abort_on_error=parsed_args.abort_on_error,
                      retune=parsed_args.retune,
                      retry_failed=parsed_args.retry_failed,
                      gpu_ids=parsed_args.gpus,
                      num_cpus=parsed_args.num_cpus,
                      wait_for_compiles=parsed_args.wait_for_compiles)

    ctx = TuningContext(configs=configs,
                        conf_class=conf_class,
                        paths=paths,
                        options=options,
                        gpu_topology=gpu_topology,
                        numa_topology=NumaTopology.discover())

    try:
        tuning_succeeded = tune_configs(ctx)
    except KeyboardInterrupt:
        return 130  # 128 + SIGINT

    return 0 if tuning_succeeded else 1


if __name__ == '__main__':
    sys.exit(main())
