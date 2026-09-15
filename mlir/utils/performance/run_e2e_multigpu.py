#!/usr/bin/env python3
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the rocMLIR lit test suite in parallel across all GPUs on a node.

CI nodes often expose several GPUs, but the lit suite historically runs every
test on GPU 0. This driver partitions the suite with lit's `--num-shards` /
`--run-shard`, running one lit process per GPU, each pinned to its device via
ROCR_VISIBLE_DEVICES. Sharding is only used on homogeneous nodes; single-GPU and
mixed-architecture nodes fall back to a single lit run.

Shard output is forwarded line by line rather than collected at the end: CI
watchdogs key off console activity, and a run that dies mid-way must still leave
behind the output that explains why. Note that all shards share one lit exec
root, so they race on `.lit_test_times.txt`; that only perturbs the ordering
heuristic of a later run.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

from gpu_topology import make_isolated_gpu_env, usable_cpu_count


def default_lit_path(build_dir: str) -> str:
    return os.path.join(build_dir, 'external', 'llvm-project', 'llvm', 'bin', 'llvm-lit')


def select_gpu_ids_out_of_process(
        requested: Optional[List[int]]) -> Tuple[List[Optional[int]], Optional[str], str]:
    """Ask a child process which GPUs to use, and what to report about them.

    Enumeration goes through HIP, which leaves the ROCm runtime loaded and
    attached to the KFD for the lifetime of the process. This driver has to
    outlive a wedged shard so it can report and clean up, and ROCr aborts every
    process holding a context when a GPU faults, so the runtime is kept in a
    child that exits immediately. A child that fails means "use one GPU", the
    same fallback taken when enumeration itself fails.
    """
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_topology.py')
    cmd = [sys.executable, helper]
    if requested:
        cmd += ['--gpus'] + [str(g) for g in requested]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
        gpu_ids, arch, message = json.loads(out.stdout)
    except Exception as e:  # noqa: BLE001 - any failure means fall back to one GPU
        return [None], None, f"GPU detection failed ({e}); using the default GPU"
    return gpu_ids, arch, message


def build_shard_command(lit: str, lit_args: List[str], jobs: int, num_shards: int, shard: int,
                        test_paths: List[str]) -> List[str]:
    """Build a single lit invocation for shard `shard` (1-based) of `num_shards`."""
    cmd = [sys.executable, lit, '-j', str(jobs)]
    if num_shards > 1:
        cmd += ['--num-shards', str(num_shards), '--run-shard', str(shard)]
    cmd += lit_args
    cmd += test_paths
    return cmd


def resolve_jobs_per_shard(args: argparse.Namespace, num_shards: int) -> int:
    """Pick the lit worker count for each shard.

    `--jobs-per-gpu` caps concurrency per GPU; `--total-jobs` instead splits a
    machine-wide budget across the shards.
    """
    if args.jobs_per_gpu is not None:
        return max(1, args.jobs_per_gpu)
    if args.total_jobs is not None:
        return max(1, args.total_jobs // num_shards)
    return 8


def cap_jobs_to_host_cpus(jobs_per_shard: int,
                          num_shards: int,
                          budget: Optional[int] = None) -> int:
    """Clamp the per-shard worker count so all shards together fit the host.

    The per-GPU caps exist to protect the GPU, so multiplying one by the number of
    GPUs can ask for far more parallelism than the machine has cores. `budget`
    defaults to the CPUs this process may run on.
    """
    if budget is None:
        budget = usable_cpu_count()
    return max(1, min(jobs_per_shard, budget // num_shards))


def _forward_output(label: str, proc: subprocess.Popen, lock: threading.Lock) -> None:
    """Tag and forward one shard's output so interleaved shards stay tellable apart."""
    for line in proc.stdout:
        with lock:
            sys.stdout.write(f"[{label}] {line}")
            sys.stdout.flush()


def _terminate_tree(proc: subprocess.Popen, sig: int) -> None:
    """Signal a shard's whole process group.

    lit runs each test in its own subprocess, and those hold the GPU contexts.
    Signalling only lit leaves them behind on the node for the next job.
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        proc.send_signal(sig)


def run(args: argparse.Namespace) -> int:
    lit = args.lit or default_lit_path(args.build_dir)
    test_paths = args.test_paths or [os.path.join(args.build_dir, 'mlir', 'test')]

    gpu_ids, _gpu_arch, gpu_msg = select_gpu_ids_out_of_process(args.gpus)
    print(f"[run_e2e_multigpu] {gpu_msg}", flush=True)

    # [None] means one un-pinned lit run (single-GPU / heterogeneous nodes).
    shard_gpus: List[Optional[int]] = gpu_ids
    num_shards = len(shard_gpus)
    requested_jobs = resolve_jobs_per_shard(args, num_shards)
    jobs_per_shard = cap_jobs_to_host_cpus(requested_jobs, num_shards, args.max_total_jobs)
    if jobs_per_shard < requested_jobs:
        budget = args.max_total_jobs if args.max_total_jobs is not None else usable_cpu_count()
        print(
            f"[run_e2e_multigpu] capping {requested_jobs} to {jobs_per_shard} workers per shard "
            f"to stay within {budget} host CPU(s)",
            flush=True)
    print(f"[run_e2e_multigpu] {num_shards} shard(s), {jobs_per_shard} lit workers each",
          flush=True)

    # A single shard needs no tagging, so let it inherit the console directly.
    if num_shards == 1:
        gpu_id = shard_gpus[0]
        cmd = build_shard_command(lit, args.lit_args, jobs_per_shard, 1, 1, test_paths)
        env = make_isolated_gpu_env(gpu_id)
        label = f"GPU {gpu_id}" if gpu_id is not None else "single"
        print(f"[run_e2e_multigpu] shard 1/1 on {label}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            return 0
        return subprocess.call(cmd, env=env)

    labels = []
    procs = []
    for idx, gpu_id in enumerate(shard_gpus):
        cmd = build_shard_command(lit, args.lit_args, jobs_per_shard, num_shards, idx + 1,
                                  test_paths)
        label = f"GPU {gpu_id}" if gpu_id is not None else "single"
        print(f"[run_e2e_multigpu] shard {idx + 1}/{num_shards} on {label}: {' '.join(cmd)}",
              flush=True)
        if args.dry_run:
            continue
        env = make_isolated_gpu_env(gpu_id) or os.environ.copy()
        # lit is itself Python; without this it block-buffers into our pipe and
        # the console would go quiet for minutes at a time.
        env['PYTHONUNBUFFERED'] = '1'
        labels.append(label)
        procs.append(
            subprocess.Popen(cmd,
                             env=env,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             text=True,
                             bufsize=1,
                             start_new_session=True))

    if args.dry_run:
        return 0

    stdout_lock = threading.Lock()
    readers = [
        threading.Thread(target=_forward_output, args=(label, proc, stdout_lock), daemon=True)
        for label, proc in zip(labels, procs)
    ]
    for reader in readers:
        reader.start()

    failures = []
    aborted = []
    pending = set(range(len(procs)))
    while pending:
        time.sleep(1)
        for i in sorted(pending):
            rc = procs[i].poll()
            if rc is None:
                continue
            pending.discard(i)
            if rc != 0:
                failures.append((labels[i], rc))

        # Once a shard fails, stop the rest so the run aborts promptly.
        if args.fail_fast and failures and pending:
            for i in sorted(pending):
                _terminate_tree(procs[i], signal.SIGTERM)
            for i in sorted(pending):
                try:
                    procs[i].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    _terminate_tree(procs[i], signal.SIGKILL)
                    procs[i].wait()
                aborted.append(labels[i])
            pending.clear()

    for reader in readers:
        reader.join(timeout=30)

    if failures:
        summary = ', '.join(f"{label} (exit {rc})" for label, rc in failures)
        print(f"\n[run_e2e_multigpu] FAILED shards: {summary}", flush=True)
        if aborted:
            print(f"[run_e2e_multigpu] aborted (fail-fast): {', '.join(aborted)}", flush=True)
        return 1
    print(f"\n[run_e2e_multigpu] all {num_shards} shard(s) passed", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run the rocMLIR lit suite sharded across all GPUs on the node.')
    parser.add_argument('test_paths',
                        nargs='*',
                        default=None,
                        help='lit test path(s) to run (default: <build-dir>/mlir/test)')
    parser.add_argument('--build-dir',
                        default='build',
                        help='Build directory (default: %(default)s)')
    parser.add_argument('--lit',
                        default=None,
                        help='Path to llvm-lit (default: <build-dir>/external/llvm-project/'
                        'llvm/bin/llvm-lit)')
    parser.add_argument('--jobs-per-gpu',
                        type=int,
                        default=None,
                        help='lit workers per GPU shard (overrides --total-jobs)')
    parser.add_argument('--total-jobs',
                        type=int,
                        default=None,
                        help='Total lit workers split evenly across shards (default per-shard: 8)')
    parser.add_argument('--max-total-jobs',
                        type=int,
                        default=None,
                        help='Upper bound on lit workers across all shards, subject to a '
                        'floor of one worker per shard (default: the number of CPUs this '
                        'process may use)')
    parser.add_argument('--gpus',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Physical GPU ids to shard across (default: auto-detect homogeneous '
                        'GPUs, else a single run)')
    parser.add_argument('--lit-args',
                        type=str,
                        default='-v --time-tests --timeout=3600 --max-failures=1',
                        help='Extra arguments passed to each lit invocation (default: %(default)r)')
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Print the per-shard lit commands without running them')
    parser.add_argument('--fail-fast',
                        dest='fail_fast',
                        action='store_true',
                        default=True,
                        help='Abort remaining shards once any shard fails (default)')
    parser.add_argument('--no-fail-fast',
                        dest='fail_fast',
                        action='store_false',
                        help='Let all shards run to completion even if one fails')
    args = parser.parse_args()
    args.lit_args = args.lit_args.split()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
