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
import functools
import glob
import json
import logging
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

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

OUTPUT_HEADER_COLUMNS = [
    'arch', 'numCUs', 'numChiplets', 'testVector', 'perfConfig', 'TFlops', 'tuningSpace',
    'commitId', 'timestamp', 'durationSec'
]

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


class GpuLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes messages with GPU ID."""

    def process(self, msg, kwargs):
        gpu_id = self.extra.get('gpu_id')
        if gpu_id is not None:
            return f"[GPU {gpu_id}] {msg}", kwargs
        return msg, kwargs


def setup_logger(quiet: bool = False, verbose: bool = False) -> None:
    """Configure and return a logger for tuningRunner."""
    if quiet and verbose:
        raise ValueError("quiet and verbose are mutually exclusive")

    if quiet:
        logger.setLevel(logging.ERROR)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.addHandler(TqdmLoggingHandler(use_color=sys.stderr.isatty()))


def get_gpu_logger(gpu_id: int) -> logging.LoggerAdapter:
    """Get a logger adapter for a specific GPU."""
    return GpuLoggerAdapter(logger, {'gpu_id': gpu_id})


# Module-level logger
logger: logging.Logger = logging.getLogger("tuningRunner")

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
    chip: str
    arch: str  # Old arch value for backwards compatibility
    num_cu: int
    num_chiplets: int
    rocmlir_gen_flags: str
    verify_mode: str
    verify_perfconfigs: bool
    output: str
    abort_on_error: bool
    retune: bool
    retry_states: frozenset
    gpu_ids: List[int]
    num_cpus: Optional[int]
    wait_for_compiles: bool
    timeout: Optional[int]


@dataclass
class TuningResult:
    """Result of tuning a single configuration."""
    test_vector: str
    success: bool
    timed_out: bool = False
    gpu_id: int = -1
    duration_seconds: float = 0.0
    timestamp: Optional[str] = None
    winning_config: Optional[str] = None
    max_tflops: Optional[float] = None
    entries: List[Dict] = field(default_factory=list)
    verify_tflops: Optional[float] = None


# =============================================================================
# Exceptions
# =============================================================================


class TuningError(Exception):
    """Raised when tuning or verification fails."""
    pass


# =============================================================================
# System Topology Discovery
# =============================================================================


@dataclass(frozen=True)
class Gpu:
    """Information about a GPU."""
    gpu_id: int
    sku: str
    numa_node: int


@dataclass(frozen=True)
class GpuTopology:
    """System GPU topology with NUMA mappings."""
    gpus: Dict[int, Gpu]  # GPU ID -> Gpu

    def get_numa_node(self, gpu_id: int) -> int:
        """Get NUMA node for a GPU."""
        return self.gpus[gpu_id].numa_node

    def validate_homogeneity(self, gpu_ids: List[int]) -> bool:
        """Validate that all selected GPUs are of the same model."""
        if len(gpu_ids) <= 1:
            return True

        skus = {self.gpus[gpu_id].sku for gpu_id in gpu_ids}
        return len(skus) == 1

    @staticmethod
    def discover() -> 'GpuTopology':
        """Query GPU topology using rocm-smi.

        rocm-smi reports physical device IDs regardless of environment variables (e.g., ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES).
        """
        output = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showtoponuma", "--json"], text=True, timeout=10)
        data = json.loads(output)

        gpus = {}
        for key, value in data.items():
            if key.startswith("card"):
                gpu_id = int(key.replace("card", ""))

                sku = value["Card SKU"]

                numa_node_str = value.get("(Topology) Numa Node")
                numa_node = int(numa_node_str) if numa_node_str is not None else 0

                gpus[gpu_id] = Gpu(gpu_id=gpu_id, sku=sku, numa_node=numa_node)

        if not gpus:
            raise RuntimeError("rocm-smi returned no GPU cards")

        return GpuTopology(gpus=gpus)


@dataclass(frozen=True)
class NumaTopology:
    """System NUMA topology with CPU mappings."""
    numa_to_cpus: Dict[int, List[int]]  # NUMA node -> list of CPU IDs

    def get_cpus_for_numa_node(self, numa_node: int) -> List[int]:
        """Get CPUs belonging to a NUMA node."""
        return self.numa_to_cpus[numa_node]

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
# State Management
# =============================================================================


class ConfigState(Enum):
    """Possible states for a tuning configuration in the state file.

    State transitions:
        PENDING (implicit) -> RUNNING: Config starts tuning
        RUNNING -> SUCCEEDED (implicit): Tuning completes successfully (removed from state, written to output)
        RUNNING -> FAILED: Tuning completes with error
        RUNNING -> TIMED_OUT: Tuning exceeded timeout
        RUNNING -> INTERRUPTED: User interrupted (Ctrl+C) during tuning
        RUNNING -> CRASHED: Detected on next startup (stale RUNNING state)
        <state> -> PENDING: User requests retry with --retry <state>

    Note: PENDING and SUCCEEDED are implicit states:
        - PENDING: not in state file AND not in output file
        - SUCCEEDED: in output file (not tracked in state file)
    """
    RUNNING = "running"  # Currently being tuned
    FAILED = "failed"  # Tuning completed with error
    TIMED_OUT = "timed_out"  # Tuning exceeded timeout
    INTERRUPTED = "interrupted"  # User interrupted during tuning (Ctrl+C)
    CRASHED = "crashed"  # Process crashed while tuning (detected on startup)


# States representing unsuccessful tuning outcomes that are skipped by default
UNSUCCESSFUL_STATES = frozenset({ConfigState.FAILED, ConfigState.TIMED_OUT, ConfigState.CRASHED})


@dataclass
class TuningState:
    """State tracking for configs within a single context."""
    configs: Dict[str, ConfigState] = field(default_factory=dict)
    _pre_running_states: Dict[str, ConfigState] = field(default_factory=dict)

    def set_running(self, test_vector: str) -> None:
        if test_vector in self.configs:
            self._pre_running_states[test_vector] = self.configs[test_vector]
        self.configs[test_vector] = ConfigState.RUNNING

    def set_failed(self, test_vector: str) -> None:
        self.configs[test_vector] = ConfigState.FAILED
        self._pre_running_states.pop(test_vector, None)

    def set_timed_out(self, test_vector: str) -> None:
        self.configs[test_vector] = ConfigState.TIMED_OUT
        self._pre_running_states.pop(test_vector, None)

    def set_interrupted(self, test_vector: str) -> None:
        self.configs[test_vector] = ConfigState.INTERRUPTED
        self._pre_running_states.pop(test_vector, None)

    def remove(self, test_vector: str) -> None:
        self.configs.pop(test_vector, None)
        self._pre_running_states.pop(test_vector, None)

    def should_skip(self, test_vector: str, retry_states: frozenset = frozenset()) -> bool:
        state = self.configs.get(test_vector)
        return state in UNSUCCESSFUL_STATES and state not in retry_states

    def is_empty(self) -> bool:
        return not self.configs

    def failed_count(self) -> int:
        return sum(1 for s in self.configs.values() if s == ConfigState.FAILED)

    def timed_out_count(self) -> int:
        return sum(1 for s in self.configs.values() if s == ConfigState.TIMED_OUT)

    def crashed_count(self) -> int:
        return sum(1 for s in self.configs.values() if s == ConfigState.CRASHED)

    def promote_running_to_interrupted(self) -> int:
        count = 0
        for tv in self.configs:
            if self.configs[tv] == ConfigState.RUNNING:
                prev_state = self._pre_running_states.pop(tv, None)
                self.configs[tv] = prev_state or ConfigState.INTERRUPTED
                count += 1
        return count


class TuningStateFile:
    """Manages multi-context tuning state in a JSON file.

    File format:
    {
        "contexts": {
            "<arch>/<num_cu>/<num_chiplets>/<tuning_space>": {
                "test_vector_1": "failed",
                "test_vector_2": "crashed"
            }
        }
    }

    If filepath is None, all operations are no-ops.
    """

    def __init__(self, filepath: Optional[str], arch: str, num_cu: int, num_chiplets: int,
                 tuning_space: str):
        self.filepath = filepath
        self.context_key = f"{arch}/{num_cu}/{num_chiplets}/{tuning_space}"
        self._lock = threading.Lock()
        self._all_contexts: Dict[str, Dict[str, str]] = {}  # context_key -> {tv -> state_str}
        self._state = TuningState()

        self._load()
        self._save_locked()  # Persist any state transitions from load

    def _load(self) -> None:
        """Load state from file.

        For the active context only:
        - INTERRUPTED configs are removed (will be retried)
        - RUNNING configs become CRASHED (stale = crash)
        """
        if not self.filepath or not os.path.exists(self.filepath):
            return

        with open(self.filepath, 'r') as f:
            data = json.load(f)
        self._all_contexts = data['contexts']

        # Process configs for active context with state transitions
        if self.context_key in self._all_contexts:
            for tv, state_str in self._all_contexts[self.context_key].items():
                try:
                    state = ConfigState(state_str)
                    if state == ConfigState.INTERRUPTED:
                        continue  # Remove - will retry
                    if state == ConfigState.RUNNING:
                        state = ConfigState.CRASHED  # Stale running = crashed
                    self._state.configs[tv] = state
                except ValueError:
                    logger.warning(f"Unknown state '{state_str}' for config '{tv}' in state file")

    @property
    def state(self) -> TuningState:
        return self._state

    def _save_locked(self) -> None:
        if not self.filepath:
            return

        # Update active context in all_contexts
        if not self._state.is_empty():
            self._all_contexts[self.context_key] = {
                tv: s.value for tv, s in self._state.configs.items()
            }
        else:
            self._all_contexts.pop(self.context_key, None)

        # Remove empty contexts
        self._all_contexts = {k: v for k, v in self._all_contexts.items() if v}

        # Delete file if nothing left, otherwise save
        if not self._all_contexts:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            return

        temp_path = self.filepath + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump({'contexts': self._all_contexts}, f, indent=2)
        os.replace(temp_path, self.filepath)

    def set_running(self, test_vector: str) -> None:
        with self._lock:
            self._state.set_running(test_vector)
            self._save_locked()

    def set_failed(self, test_vector: str) -> None:
        with self._lock:
            self._state.set_failed(test_vector)
            self._save_locked()

    def set_timed_out(self, test_vector: str) -> None:
        with self._lock:
            self._state.set_timed_out(test_vector)
            self._save_locked()

    def set_succeeded(self, test_vector: str) -> None:
        with self._lock:
            self._state.remove(test_vector)
            self._save_locked()

    def finalize_interrupted(self) -> None:
        """Mark RUNNING configs as INTERRUPTED on clean shutdown."""
        with self._lock:
            count = self._state.promote_running_to_interrupted()
            if count > 0:
                logger.info(f"Marked {count} running config(s) as interrupted")
            self._save_locked()


def get_state_filepath(output_filepath: str) -> Optional[str]:
    """Get the state file path for a given output file."""
    if output_filepath == '-':
        return None
    return f"{output_filepath}.state"


# =============================================================================
# Tuning Infrastructure
# =============================================================================


@dataclass(frozen=True)
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

        Format (new): # arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig\tTFlops\ttuningSpace\tcommitId\ttimestamp\tdurationSec
        Format (old): # arch\tnumCUs\tnumChiplets\ttestVector\tperfConfig (tuning_space)\t[TFlops]

        Only loads entries matching current arch, num_cu, num_chiplets, and tuning space.
        """
        if options.output == '-' or not os.path.exists(options.output):
            return cls()

        results: Dict[str, TuningResult] = {}

        current_commit = get_git_commit_hash()
        warned_commits: set = set()

        header_tuning_space: Optional[str] = None
        column_indices: Dict[str, int] = {}

        with open(options.output, mode='r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Check for header line
                if cls._is_header_line(line):
                    column_indices = cls._parse_header_line(line)
                    # Extract tuning space from header for old format (perfConfig (tuning_space))
                    header_tuning_space = cls._extract_tuning_space_from_header(line)
                    continue

                # Skip comment lines
                if line.startswith('#'):
                    continue

                # Skip if we haven't seen a header yet
                if not column_indices:
                    continue

                # Parse data line
                result = cls._parse_data_line(line.split('\t'), column_indices, options,
                                              header_tuning_space, current_commit, warned_commits)
                if result:
                    results[result.test_vector] = result

        return cls(_results=results)

    @staticmethod
    def _is_header_line(line: str) -> bool:
        """Check if line is a column header."""
        header_prefix = f"# {OUTPUT_HEADER_COLUMNS[0]}\t"
        return line.startswith(header_prefix)

    @staticmethod
    def _extract_tuning_space_from_header(line: str) -> Optional[str]:
        """Extract tuning space from old format header like 'perfConfig (quick)' or 'TFlops (quick)'."""
        match = re.search(r'\((\w+)\)', line)
        return match.group(1) if match else None

    @staticmethod
    def _parse_header_line(line: str) -> Dict[str, int]:
        """Parse column header and return name -> index mapping."""
        # Strip leading '# ' if present
        header_text = line[2:] if line.startswith('# ') else line

        indices = {}
        for i, col in enumerate(header_text.split('\t')):
            if not col:
                continue
            # Extract base column name (handles 'perfConfig (tuning_space)')
            col_name = col.split()[0]
            indices[col_name] = i

        return indices

    @staticmethod
    def _parse_data_line(fields: List[str], column_indices: Dict[str, int], options: Options,
                         header_tuning_space: Optional[str], current_commit: str,
                         warned_commits: set) -> Optional[TuningResult]:
        """Parse a data line and return TuningResult if valid.

        A line is valid if:
        - arch matches current system (chip or arch for backwards compatibility)
        - numCUs and numChiplets match current system
        - tuning space matches (from column or header)
        - testVector is present
        - perfConfig is present and not 'None'
        """

        def get_field(name: str) -> Optional[str]:
            idx = column_indices.get(name)
            if idx is not None and idx < len(fields) and fields[idx]:
                return fields[idx]
            return None

        # Check arch match (chip or arch)
        file_arch = get_field('arch')
        if file_arch != options.chip and file_arch != options.arch:
            return None

        # Check numCUs match
        file_num_cu = get_field('numCUs')
        if file_num_cu and file_num_cu != str(options.num_cu):
            return None

        # Check numChiplets match
        file_num_chiplets = get_field('numChiplets')
        if file_num_chiplets and file_num_chiplets != str(options.num_chiplets):
            return None

        # Check tuning space match (new format has column, old format used header)
        file_tuning_space = get_field('tuningSpace') or header_tuning_space
        if file_tuning_space != options.tuning_space_kind:
            return None

        test_vector = get_field('testVector')
        if not test_vector:
            return None

        perf_config = get_field('perfConfig')
        if not perf_config or perf_config == 'None':
            return None

        # TFlops (optional)
        max_tflops = None
        tflops_str = get_field('TFlops')
        if tflops_str:
            try:
                tflops_val = float(tflops_str)
                if np.isfinite(tflops_val):
                    max_tflops = tflops_val
            except ValueError:
                pass

        # Duration (optional)
        duration_seconds = 0.0
        duration_str = get_field('durationSec')
        if duration_str:
            try:
                duration_seconds = float(duration_str)
            except ValueError:
                pass

        # Timestamp (optional)
        timestamp = get_field('timestamp')

        # Warn if commit differs (avoid spamming for same commit)
        file_commit = get_field('commitId')
        if file_commit and file_commit != current_commit and file_commit not in warned_commits:
            logger.warning(
                f"Loading tuned configs from different commit (file: {file_commit[:8]}, current: {current_commit[:8]})"
            )
            warned_commits.add(file_commit)

        return TuningResult(test_vector=test_vector,
                            success=True,
                            gpu_id=-1,
                            duration_seconds=duration_seconds,
                            timestamp=timestamp,
                            winning_config=perf_config,
                            max_tflops=max_tflops)


@dataclass
class ETATracker:
    """Track completion times for accurate ETA estimation using median of successful configs."""
    total_configs: int
    num_workers: int
    success_times: List[float] = field(default_factory=list)
    ok_count: int = 0
    fail_count: int = 0
    _processed: int = field(default=0, init=False)

    def record(self, result: TuningResult) -> None:
        self._processed += 1
        if result.success:
            self.ok_count += 1
            self.success_times.append(result.duration_seconds)
        else:
            self.fail_count += 1

    def _format_rate(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s/cfg"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m/cfg"
        else:
            return f"{seconds / 3600:.1f}h/cfg"

    def _format_eta(self, seconds: float) -> str:
        if seconds == 0:
            return "0s"
        elif seconds < 60:
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
        if len(self.success_times) >= 3:
            median = statistics.median(self.success_times)
            eta_seconds = (remaining / self.num_workers) * median if self.num_workers > 0 else 0
            rate = self._format_rate(median)
            eta = self._format_eta(eta_seconds)

        return f"ok={self.ok_count}, fail={self.fail_count}, rate={rate}, eta={eta}"


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
        return self._threads_per_gpu[gpu_id]

    def print_gpu_summary(self, num_workers: Optional[int] = None) -> None:
        """Print summary of GPU allocation."""
        num_active = num_workers or len(self.options.gpu_ids)
        lines = [f"Using {num_active} GPU(s)"]
        for gpu_id in self.options.gpu_ids[:num_active]:
            node = self.gpu_topology.get_numa_node(gpu_id)
            threads = self._threads_per_gpu[gpu_id]
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

        os.sched_setaffinity(0, set(cpu_list))

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
            logger.debug("libnuma not available, skipping memory policy setup")


# =============================================================================
# Output Writers
# =============================================================================


class OutputFileWriter:
    """Context manager for writing tuning results to TSV file."""

    HEADER = "# " + "\t".join(OUTPUT_HEADER_COLUMNS)

    def __init__(self, filepath: str, options: Options):
        self.filepath = filepath
        self.options = options
        self.file = None
        self._header_written = False

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
        print(self.HEADER, file=self.file)
        self.file.flush()
        self._header_written = True

    def write_result(self, result: TuningResult):
        if not result.success:
            raise ValueError("write_result called with unsuccessful result")
        if not result.winning_config:
            raise ValueError("write_result called without winning_config")
        if result.max_tflops is None:
            raise ValueError("write_result called without max_tflops")
        if not result.timestamp:
            raise ValueError("write_result called without timestamp")
        if result.duration_seconds <= 0.0:
            raise ValueError("write_result called with invalid duration_seconds")

        if not self._header_written:
            self._write_header()

        fields = [
            self.options.arch,
            str(self.options.num_cu),
            str(self.options.num_chiplets), result.test_vector, result.winning_config,
            str(result.max_tflops), self.options.tuning_space_kind,
            get_git_commit_hash(), result.timestamp, f"{result.duration_seconds:.1f}"
        ]

        print("\t".join(fields), file=self.file)
        self.file.flush()


class DebugFileWriter:
    """Context manager for writing debug entries to TSV file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self._header_written = False

    def __enter__(self):
        self.file = open(self.filepath, 'a')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()

    def write_result(self, result: TuningResult):
        if not result.success:
            raise ValueError("write_result called with unsuccessful result")
        if not result.entries:
            raise ValueError("write_result called without entries")

        pd.DataFrame(result.entries).to_csv(self.file,
                                            sep='\t',
                                            header=not self._header_written,
                                            index=False)
        self.file.flush()
        self._header_written = True


# =============================================================================
# Utilities
# =============================================================================

# Signals that indicate user/system requested termination (should not be logged as failures)
TERMINATION_SIGNALS = frozenset({
    signal.SIGINT,  # Ctrl+C
    signal.SIGTERM,  # Graceful termination request
    signal.SIGHUP,  # Terminal hangup
    signal.SIGQUIT,  # Quit from keyboard
})


def raise_if_terminated(returncode: int) -> None:
    """Raise KeyboardInterrupt if returncode indicates termination."""
    if -returncode in TERMINATION_SIGNALS:
        raise KeyboardInterrupt()


class TuningArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with custom validation for tuning arguments."""

    def __init__(self, *args, gpu_topology: Optional[GpuTopology] = None, **kwargs):
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


@functools.lru_cache(maxsize=1)
def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        logger.warning(f"Failed to get git commit hash: {e}")
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
    raise ValueError(f"Unknown verification mode: {verify_mode}")


def kill_process(proc: Optional[subprocess.Popen]) -> None:
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
                 command: Optional[str] = None,
                 stdout: Optional[str] = None,
                 stderr: Optional[str] = None,
                 exit_code: Optional[int] = None,
                 gpu_id: Optional[int] = None,
                 max_lines: int = 10) -> str:
    """Format an error message with optional details."""

    def truncate(text: Optional[str]) -> Optional[str]:
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
        parts.append("STDOUT:\n" + truncated_stdout)

    truncated_stderr = truncate(stderr)
    if truncated_stderr:
        parts.append("STDERR:\n" + truncated_stderr)

    return '\n'.join(parts)


# =============================================================================
# Core Tuning Logic
# =============================================================================


def verify_perfconfig(perfconfig: str, config: PerfConfiguration, paths: Paths, options: Options,
                      gpu_id: int) -> float:
    """Verify a performance config by running with profiling.

    Returns the execution time in nanoseconds, or raises TuningError on failure.
    """
    gpu_logger = get_gpu_logger(gpu_id)

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
    gpu_logger.debug(f"Verifying perfconfig '{perfconfig}'\nCommand: {verification_pipeline}")

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
                raise_if_terminated(p3.returncode)
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


def find_best_perfconfig(tuning_output_lines: List[str], config: PerfConfiguration, paths: Paths,
                         options: Options,
                         gpu_id: int) -> Tuple[Optional[str], Optional[float], List[Dict]]:
    """Parse tuning driver output and find the best performing perfconfig.

    Returns the winning config, its TFLOPS, and all entries.
    """
    gpu_logger = get_gpu_logger(gpu_id)

    max_tflops: Optional[float] = None
    winning_config: Optional[str] = None
    entries = []

    for line in tuning_output_lines:
        result = line.strip()
        if not result:
            continue

        parts = result.split('\t')
        if len(parts) < 2:
            gpu_logger.debug(f"Skipping malformed tuning output line: '{result}'")
            continue

        perfconfig = parts[0]
        time = parts[-1]
        try:
            if time == "N/A":
                nano_seconds = np.nan
                measurements = None
            else:
                nano_seconds = float(time)
                measurements = json.loads(parts[1]) if len(parts) == 3 else None
        except (ValueError, json.JSONDecodeError):
            gpu_logger.debug(f"Skipping malformed tuning output line: '{result}'")
            continue

        config.set_perfconfig(perfconfig)
        entry = config.table_entry(nano_seconds)
        if options.debug:
            entry["MeasurementsMs"] = measurements
        entries.append(entry)

        if options.verify_perfconfigs and not np.isnan(nano_seconds):
            verify_ns = verify_perfconfig(perfconfig, config, paths, options, gpu_id)
            if np.isnan(verify_ns):
                raise TuningError(f"Verification returned NaN for perfconfig '{perfconfig}'")

        these_tflops = entry['TFlops']
        if not np.isnan(these_tflops) and (max_tflops is None or these_tflops > max_tflops):
            max_tflops = these_tflops
            winning_config = perfconfig

    return winning_config, max_tflops, entries


def tune_config(test_vector: str, conf_class: type, paths: Paths, options: Options, gpu_id: int,
                num_compile_threads: int) -> TuningResult:
    """Tune a single configuration and return the result."""
    gpu_logger = get_gpu_logger(gpu_id)

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
            config = conf_class.from_command_line(command_line, options.arch, options.num_cu,
                                                  options.num_chiplets)
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
            raise_if_terminated(tuning_key.returncode)
            if tuning_key.returncode != 0:
                gpu_logger.error(
                    format_error("Failed to generate tuning key",
                                 command=' '.join(rocmlir_gen_command),
                                 stderr=err.decode('utf-8'),
                                 exit_code=tuning_key.returncode,
                                 gpu_id=gpu_id))
                return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)
            result = output.decode('utf-8').strip().split('\t')
            command_line = result[2].split(sep=' ')
            config = conf_class.from_command_line(command_line, options.arch, options.num_cu,
                                                  options.num_chiplets)
            tuning_driver_command += [test_vector]
            tuning_driver = subprocess.Popen(tuning_driver_command,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             env=env)
            tuning_pipeline = ' '.join(tuning_driver_command)

        gpu_logger.debug(f"Tuning '{test_vector}'\nCommand: {tuning_pipeline}")

        try:
            tuning_stdout, tuning_stderr = tuning_driver.communicate(timeout=options.timeout)
        except subprocess.TimeoutExpired:
            gpu_logger.error(
                format_error(f"Tuning timed out after {options.timeout}s",
                             command=tuning_pipeline,
                             gpu_id=gpu_id))
            return TuningResult(test_vector=test_vector,
                                success=False,
                                timed_out=True,
                                gpu_id=gpu_id)

        raise_if_terminated(tuning_driver.returncode)

        tuning_output = tuning_stdout.decode('utf-8')
        tuning_errors = tuning_stderr.decode('utf-8')

        if tuning_driver.returncode != 0:
            gpu_logger.error(
                format_error("Tuning pipeline failed",
                             command=tuning_pipeline,
                             stdout=tuning_output,
                             stderr=tuning_errors,
                             exit_code=tuning_driver.returncode,
                             gpu_id=gpu_id))
            return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)
        else:
            # Log any stderr output from tuning driver because it may contain warnings
            if tuning_errors.strip():
                gpu_logger.warning(f"rocmlir-tuning-driver stderr:\n{tuning_errors}")

        winning_config, max_tflops, entries = find_best_perfconfig(tuning_output.splitlines(),
                                                                   config, paths, options, gpu_id)
    except TuningError as e:
        gpu_logger.error(str(e))
        return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)
    finally:
        kill_process(rocmlir_gen)
        kill_process(tuning_driver)

    if winning_config is None:
        gpu_logger.error("No valid perf config found")
        return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)

    verify_tflops = None
    if options.verify_mode != "none":
        try:
            verify_ns = verify_perfconfig(winning_config, config, paths, options, gpu_id)
        except TuningError as e:
            gpu_logger.error(str(e))
            return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)

        if np.isnan(verify_ns):
            gpu_logger.error(f"Verification returned NaN for winning perfconfig '{winning_config}'")
            return TuningResult(test_vector=test_vector, success=False, gpu_id=gpu_id)

        verify_tflops = config.compute_tflops(verify_ns)

    return TuningResult(test_vector=test_vector,
                        success=True,
                        gpu_id=gpu_id,
                        winning_config=winning_config,
                        max_tflops=max_tflops,
                        entries=entries,
                        verify_tflops=verify_tflops)


def tune_configs(ctx: TuningContext, status_only: bool) -> bool:
    """Tune multiple configurations in parallel across available GPUs."""
    # Load tuned configs from output file (unless --retune)
    if ctx.options.retune:
        cache = TunedConfigsCache()
    else:
        cache = TunedConfigsCache.from_output_file(ctx.options)

    # Load state file
    state_file = TuningStateFile(get_state_filepath(ctx.options.output), ctx.options.chip,
                                 ctx.options.num_cu, ctx.options.num_chiplets,
                                 ctx.options.tuning_space_kind)
    state = state_file.state

    if cache.count() > 0:
        logger.info(f"Found {cache.count()} tuned config(s) in {ctx.options.output}")
    if state.crashed_count() > 0:
        logger.warning(f"Found {state.crashed_count()} crashed config(s) in state file")
    if state.timed_out_count() > 0:
        logger.warning(f"Found {state.timed_out_count()} timed out config(s) in state file")
    if state.failed_count() > 0:
        logger.warning(f"Found {state.failed_count()} failed config(s) in state file")

    pending_configs = ctx.configs

    # Filter out already-tuned configs (unless --retune)
    skipped_successful = 0
    if not ctx.options.retune:
        pending_configs = [c for c in pending_configs if not cache.contains(c)]
        skipped_successful = len(ctx.configs) - len(pending_configs)

    # Filter out unsuccessful configs (unless --retry or --retune)
    skipped_unsuccessful = 0
    if not ctx.options.retune:
        before_filter = len(pending_configs)
        pending_configs = [
            c for c in pending_configs if not state.should_skip(c, ctx.options.retry_states)
        ]
        skipped_unsuccessful = before_filter - len(pending_configs)

    total_skipped = skipped_successful + skipped_unsuccessful

    if skipped_successful > 0:
        logger.info(
            f"Skipping {skipped_successful} already tuned config(s) - use '--retune' to retune")
    if skipped_unsuccessful > 0:
        logger.info(
            f"Skipping {skipped_unsuccessful} unsuccessful config(s) - use '--retry <state>' to retry"
        )

    if status_only:
        logger.info(f"{len(pending_configs)}/{len(ctx.configs)} config(s) pending tuning")
        return True

    if not pending_configs:
        logger.info("No configurations to tune")
        return True

    pool = GpuWorkerPool(ctx)
    num_workers = min(pool.worker_count, len(pending_configs))
    ctx.print_gpu_summary(num_workers=num_workers)

    # Prepare ETA tracker with historical data
    initial_times = [
        r.duration_seconds for r in cache.get_all_results() if r.duration_seconds > 0.0
    ]
    eta_tracker = ETATracker(total_configs=len(pending_configs),
                             num_workers=num_workers,
                             success_times=initial_times,
                             ok_count=skipped_successful,
                             fail_count=skipped_unsuccessful)

    has_errors = False

    debug_enabled = ctx.options.debug and ctx.options.output != '-'
    if ctx.options.debug and not debug_enabled:
        logger.warning("Debug output disabled when writing to stdout")

    with (OutputFileWriter(ctx.options.output, ctx.options) as results_writer,
          DebugFileWriter(f"{ctx.options.output}.debug") if debug_enabled else nullcontext() as
          debug_writer):

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

            def execute_tuning_task(test_vector: str) -> TuningResult:
                gpu_id = pool.acquire_gpu_for_thread()

                state_file.set_running(test_vector)

                timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                start_time = time.time()
                compile_threads = ctx.get_compile_threads(gpu_id)
                result = tune_config(test_vector, ctx.conf_class, ctx.paths, ctx.options, gpu_id,
                                     compile_threads)
                result.duration_seconds = time.time() - start_time
                result.timestamp = timestamp

                if result.success:
                    state_file.set_succeeded(result.test_vector)
                elif result.timed_out:
                    state_file.set_timed_out(result.test_vector)
                else:
                    state_file.set_failed(result.test_vector)

                return result

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
                        debug_writer.write_result(result)
                else:
                    has_errors = True
                    logger.error(
                        f"Tuning unsuccessful for '{result.test_vector}' on GPU {result.gpu_id}")

                eta_tracker.record(result)
                progress_bar.update(1)
                progress_bar.set_postfix_str(eta_tracker.get_postfix_str())

                if has_errors and ctx.options.abort_on_error:
                    return False

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


def resolve_paths(op_type: Operation, parsed_args: argparse.Namespace) -> Paths:
    """Resolve paths based on operation type and arguments."""
    if op_type == Operation.FUSION:
        configs_path = "./fusion_config_file"
    elif parsed_args.config:
        configs_path = None
    else:
        configs_path = parsed_args.configs_file
    return perfRunner.create_paths(configs_path, parsed_args.mlir_build_dir)


def extract_fusion_configs(test_dir: str, paths: Paths) -> Operation:
    """Extract tuning configurations from fusion E2E test files.

    Writes extracted configs to paths.configuration_file_path and returns the detected operation type.
    """
    all_configs = []
    op_type = Operation.FUSION

    for filename in glob.glob(test_dir + '/*mlir'):
        logger.info(f"Extracting fusion configs from: {filename}")
        test_entry = perfRunner.get_fusion_test_info(filename, paths)
        if not test_entry:
            continue

        test_vector = test_entry['testVector']
        if not test_vector:
            continue

        if test_vector in all_configs:
            logger.debug("Duplicate entry skipped")
            continue

        command_line = test_vector.split(sep=' ')
        if command_line[0].startswith('conv'):
            if op_type == Operation.FUSION:
                op_type = Operation.CONV
            elif op_type != Operation.CONV:
                logger.warning(f"Mixed operation types, skipping: {test_vector}")
                continue
        else:
            if op_type == Operation.FUSION:
                op_type = Operation.GEMM
            elif op_type != Operation.GEMM:
                logger.warning(f"Mixed operation types, skipping: {test_vector}")
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

    if op_type not in config_classes:
        raise ValueError(f"No config class for operation: {str(op_type)}")
    return config_classes[op_type]


def load_configs_from_stdin() -> str:
    """Read configs from stdin and return path to a temporary file."""
    content = sys.stdin.read()
    fd, path = tempfile.mkstemp(suffix='.txt', prefix='tuning_configs_')
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path


def load_configs(op_type: Operation, parsed_args: argparse.Namespace, paths: Paths) -> List[str]:
    """Load configurations based on operation type and arguments."""
    if parsed_args.config:
        return [parsed_args.config]

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

    if op_type not in loaders:
        raise ValueError(f"No config loader for operation: {str(op_type)}")
    return loaders[op_type]()


# =============================================================================
# Entry Point
# =============================================================================


def parse_arguments(gpu_topology: GpuTopology,
                    available_gpus: List[int],
                    args=None) -> argparse.Namespace:
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
        metavar='FILE',
        help="Path to file containing list of configurations to tune. Use '-' for stdin.")

    config_group.add_argument(
        "--config",
        type=str,
        metavar='CONFIG',
        help="Specific config to tune. Can be a config string or path to an .mlir file.")

    config_group.add_argument(
        "--test-dir",
        "--test_dir",  # for backward compatibility
        type=str,
        metavar='DIR',
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
        metavar='FILE',
        help=
        "Output file path for tuning results in TSV format. Results will be appended if file exists. Use '-' for stdout."
    )

    parser.add_argument(
        "--mlir-build-dir",
        type=str,
        default=perfRunner.find_mlir_build_dir(),
        metavar='DIR',
        help=
        "Path to rocMLIR build directory containing rocmlir-gen, rocmlir-driver, rocmlir-tuning-driver, and other build artifacts",
    )

    parser.add_argument(
        "--rocmlir-gen-flags",
        "--rocmlir_gen_flags",  # for backward compatibility
        type=str,
        default="",
        metavar='FLAGS',
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

    logging_group = parser.add_mutually_exclusive_group()

    logging_group.add_argument("-q",
                               "--quiet",
                               action='store_true',
                               default=False,
                               help="Suppress non-error output")

    logging_group.add_argument("-v",
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

    parser.add_argument('--data-type',
                        nargs='+',
                        choices=[
                            "f32", "f16", "bf16", "i8", "i8_i32", "i8_i8", "fp8", "fp8_f32",
                            "fp8_fp8", "f4E2M1FN"
                        ],
                        default=["f32", "f16", "i8"],
                        metavar='TYPE',
                        help="Force a set of data types for gemm tuning. Only used when --op=gemm.")

    parser.add_argument(
        '--scale-type',
        nargs='+',
        choices=["f32", "f8E8M0FNU"],
        default=None,
        metavar='TYPE',
        help="Force a set of scale types for gemm tuning. Only used when --op=gemm.")

    parser.add_argument(
        "--tflops",
        action='store_true',
        default=False,
        help="[Deprecated, TFlops is always included] Include achieved TFLOPS in the output")

    parser.add_argument("--abort-on-error",
                        action='store_true',
                        default=False,
                        help="Abort tuning upon first error encounter")

    parser.add_argument(
        "--retune",
        action='store_true',
        default=False,
        help="Force retuning of all configs, ignoring existing results in the output file")

    parser.add_argument("--retry",
                        nargs='+',
                        choices=["failed", "timed_out", "crashed"],
                        default=[],
                        metavar='STATE',
                        help="Retry configs in specified states")

    parser.add_argument("--timeout",
                        type=int,
                        default=None,
                        metavar='SECONDS',
                        help="Timeout in seconds for tuning each config")

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

    parser.add_argument("-s",
                        "--status",
                        action='store_true',
                        default=False,
                        help="Only show tuning status without performing any tuning")

    return parser.parse_args(args)


def main(args=None):
    gpu_topology = GpuTopology.discover()
    available_gpus = sorted(gpu_topology.gpus.keys())

    # Capture these before set_isolated_gpu_env overwrites them
    user_rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    user_hip_visible = os.environ.get("HIP_VISIBLE_DEVICES")

    # We call into perfRunner which also queries GPU info using HIP and rocminfo.
    # To ensure consistency, we isolate the process to the first available GPU.
    set_isolated_gpu_env(os.environ, available_gpus[0])

    parsed_args = parse_arguments(gpu_topology, available_gpus, args)

    setup_logger(quiet=parsed_args.quiet, verbose=parsed_args.verbose)

    if user_rocr_visible or user_hip_visible:
        vars_set = []
        if user_rocr_visible:
            vars_set.append(f"ROCR_VISIBLE_DEVICES={user_rocr_visible}")
        if user_hip_visible:
            vars_set.append(f"HIP_VISIBLE_DEVICES={user_hip_visible}")
        logger.warning(
            f"Ignoring {' and '.join(vars_set)}. "
            f"This script manages GPU visibility internally. Use '--gpus' to select specific GPUs.")

    op_type = Operation.from_name(parsed_args.op)

    # Handle stdin for configs file
    stdin_temp_file = None
    if parsed_args.configs_file == '-':
        stdin_temp_file = load_configs_from_stdin()
        parsed_args.configs_file = stdin_temp_file

    try:
        paths = resolve_paths(op_type, parsed_args)
        if not paths.mlir_paths:
            logger.error("rocMLIR build dir was not provided/found")
            return 1

        if op_type == Operation.FUSION:
            op_type = extract_fusion_configs(parsed_args.test_dir, paths)

        configs = load_configs(op_type, parsed_args, paths)
    finally:
        if stdin_temp_file:
            os.unlink(stdin_temp_file)

    arch = perfRunner.get_arch()
    chip = perfRunner.get_chip()
    num_cu = perfRunner.get_num_cu(chip)
    num_chiplets = perfRunner.get_num_chiplets(chip, num_cu)

    options = Options(chip=chip,
                      arch=arch,
                      num_cu=num_cu,
                      num_chiplets=num_chiplets,
                      debug=parsed_args.debug,
                      quiet=parsed_args.quiet,
                      verbose=parsed_args.verbose,
                      tuning_space_kind=parsed_args.tuning_space,
                      rocmlir_gen_flags=parsed_args.rocmlir_gen_flags,
                      verify_mode=parsed_args.verify_mode,
                      verify_perfconfigs=parsed_args.verify_perf_configs,
                      output=parsed_args.output,
                      abort_on_error=parsed_args.abort_on_error,
                      retune=parsed_args.retune,
                      retry_states=frozenset(ConfigState(s) for s in parsed_args.retry),
                      gpu_ids=parsed_args.gpus,
                      num_cpus=parsed_args.num_cpus,
                      wait_for_compiles=parsed_args.wait_for_compiles,
                      timeout=parsed_args.timeout)

    ctx = TuningContext(configs=configs,
                        conf_class=get_config_class(op_type),
                        paths=paths,
                        options=options,
                        gpu_topology=gpu_topology,
                        numa_topology=NumaTopology.discover())

    try:
        tuning_succeeded = tune_configs(ctx, status_only=parsed_args.status)
    except KeyboardInterrupt:
        return 128 + signal.SIGINT

    return 0 if tuning_succeeded else 1


if __name__ == '__main__':
    sys.exit(main())
