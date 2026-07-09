#!/usr/bin/env python3
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the rocMLIR lit test suite in parallel across all GPUs on a node.

CI nodes commonly expose several GPUs but the lit suite (e.g. `check-rocmlir`)
historically runs every test on GPU 0. This driver shards the suite with lit's
native `--num-shards`/`--run-shard` mechanism, launching one lit process per GPU
and isolating each to its device via ROCR_VISIBLE_DEVICES. The total set of
tests is partitioned, so per-GPU concurrency stays bounded (avoiding the
oversubscription hangs seen with a single global `-j`) while all GPUs are used.

Distribution is only enabled on homogeneous nodes (all GPUs share one gfx
architecture); single-GPU and heterogeneous nodes fall back to one lit run,
preserving today's behavior.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

# Reuse the GPU detection helper from the performance scripts.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'performance'))
from gpu_topology import select_gpu_ids  # noqa: E402


def default_lit_path(build_dir: str) -> str:
    return os.path.join(build_dir, 'external', 'llvm-project', 'llvm', 'bin', 'llvm-lit')


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

    `--jobs-per-gpu` wins if given. Otherwise `--total-jobs` is split evenly
    across shards, so a single-GPU node keeps the legacy global concurrency while
    multi-GPU nodes keep per-GPU pressure bounded (total/num_shards). Falls back
    to a conservative default when neither is given.
    """
    if args.jobs_per_gpu is not None:
        return max(1, args.jobs_per_gpu)
    if args.total_jobs is not None:
        return max(1, args.total_jobs // num_shards)
    return 8


def run(args: argparse.Namespace) -> int:
    lit = args.lit or default_lit_path(args.build_dir)
    test_paths = args.test_paths or [os.path.join(args.build_dir, 'mlir', 'test')]

    gpu_ids, _gpu_arch, gpu_msg = select_gpu_ids(args.gpus)
    print(f"[run_e2e_multigpu] {gpu_msg}", flush=True)

    # `select_gpu_ids` returns [None] for the single-GPU / heterogeneous / unknown
    # cases; treat those as one un-pinned lit run (legacy behavior).
    single_gpu = gpu_ids == [None]
    shard_gpus: List[Optional[int]] = [None] if single_gpu else gpu_ids
    num_shards = len(shard_gpus)
    jobs_per_shard = resolve_jobs_per_shard(args, num_shards)
    print(f"[run_e2e_multigpu] {num_shards} shard(s), {jobs_per_shard} lit workers each",
          flush=True)

    # Single shard: stream lit output straight to the console (live per-test
    # progress, keeps CI activity timeouts alive). Multi-shard buffers per-GPU.
    if num_shards == 1:
        gpu_id = shard_gpus[0]
        cmd = build_shard_command(lit, args.lit_args, jobs_per_shard, 1, 1, test_paths)
        env = os.environ.copy()
        if gpu_id is not None:
            env['ROCR_VISIBLE_DEVICES'] = str(gpu_id)
            env.pop('HIP_VISIBLE_DEVICES', None)
        label = f"GPU {gpu_id}" if gpu_id is not None else "single"
        print(f"[run_e2e_multigpu] shard 1/1 on {label}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            return 0
        return subprocess.call(cmd, env=env)

    procs = []
    log_paths = []
    for idx, gpu_id in enumerate(shard_gpus):
        cmd = build_shard_command(lit, args.lit_args, jobs_per_shard, num_shards, idx + 1,
                                  test_paths)
        env = os.environ.copy()
        if gpu_id is not None:
            env['ROCR_VISIBLE_DEVICES'] = str(gpu_id)
            env.pop('HIP_VISIBLE_DEVICES', None)
        label = f"GPU {gpu_id}" if gpu_id is not None else "single"
        print(f"[run_e2e_multigpu] shard {idx + 1}/{num_shards} on {label}: {' '.join(cmd)}",
              flush=True)
        if args.dry_run:
            continue
        log_path = os.path.join(args.build_dir, f"e2e-shard-{idx}.log")
        log_paths.append((label, log_path))
        log_file = open(log_path, 'wb')
        procs.append((label, log_file,
                      subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)))

    if args.dry_run:
        return 0

    failures = []
    aborted = []
    pending = list(range(len(procs)))
    # Heartbeat so the console keeps emitting output during the otherwise-silent
    # buffered run, preventing Jenkins `timeout(activity: true)` from firing.
    start = time.time()
    last_beat = start
    heartbeat_secs = 30
    while pending:
        time.sleep(1)
        now = time.time()
        if now - last_beat >= heartbeat_secs:
            last_beat = now
            print(
                f"[run_e2e_multigpu] still running: {len(pending)}/{len(procs)} "
                f"shard(s) active, {int(now - start)}s elapsed",
                flush=True)
        for i in list(pending):
            label, log_file, proc = procs[i]
            rc = proc.poll()
            if rc is None:
                continue
            log_file.close()
            pending.remove(i)
            if rc != 0:
                failures.append((label, rc))

        # Fail-fast: once any shard fails, terminate the rest so the run aborts
        # promptly (matching the previous single-lit --max-failures=1 behavior).
        if args.fail_fast and failures and pending:
            for i in pending:
                procs[i][2].terminate()
            for i in pending:
                label, log_file, proc = procs[i]
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log_file.close()
                aborted.append(label)
            pending = []

    # Surface every shard's output in the CI console.
    for label, log_path in log_paths:
        print(f"\n===== lit output: {label} ({log_path}) =====", flush=True)
        with open(log_path, 'r', errors='replace') as f:
            sys.stdout.write(f.read())

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
    parser.add_argument('--gpus',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Physical GPU ids to shard across (default: auto-detect homogeneous '
                        'GPUs, else a single run)')
    parser.add_argument('--lit-args',
                        type=str,
                        default='-v --time-tests --timeout=3600',
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
