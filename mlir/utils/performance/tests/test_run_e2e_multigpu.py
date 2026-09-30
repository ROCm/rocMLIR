# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests for run_e2e_multigpu.py.

These cover the pure decisions the driver makes before launching lit: how many
workers each shard gets, how that is clamped to the host's CPUs, and what the
per-shard lit command line looks like. No GPU or build tree is required.
"""
import argparse
import sys
from pathlib import Path

# Ensure we can import run_e2e_multigpu (lives in mlir/utils/performance).
_test_dir = Path(__file__).resolve().parent
_sys_path_parent = str(_test_dir.parent)
if _sys_path_parent not in sys.path:
    sys.path.insert(0, _sys_path_parent)

import run_e2e_multigpu  # noqa: E402


def _args(jobs_per_gpu=None, total_jobs=None):
    return argparse.Namespace(jobs_per_gpu=jobs_per_gpu, total_jobs=total_jobs)


class TestResolveJobsPerShard:
    """Tests for resolve_jobs_per_shard."""

    def test_jobs_per_gpu_wins(self):
        assert run_e2e_multigpu.resolve_jobs_per_shard(_args(jobs_per_gpu=20, total_jobs=8),
                                                       4) == 20

    def test_total_jobs_is_split_across_shards(self):
        assert run_e2e_multigpu.resolve_jobs_per_shard(_args(total_jobs=32), 4) == 8

    def test_default_when_neither_given(self):
        assert run_e2e_multigpu.resolve_jobs_per_shard(_args(), 4) == 8

    def test_never_returns_zero(self):
        assert run_e2e_multigpu.resolve_jobs_per_shard(_args(total_jobs=2), 8) == 1
        assert run_e2e_multigpu.resolve_jobs_per_shard(_args(jobs_per_gpu=0), 1) == 1


class TestCapJobsToHostCpus:
    """Tests for cap_jobs_to_host_cpus."""

    def test_no_cap_when_host_has_room(self):
        assert run_e2e_multigpu.cap_jobs_to_host_cpus(20, 2, budget=256) == 20

    def test_caps_when_shards_would_oversubscribe_the_host(self):
        # 8 shards x 64 workers = 512 requested, but only 128 CPUs are available.
        assert run_e2e_multigpu.cap_jobs_to_host_cpus(64, 8, budget=128) == 16

    def test_never_caps_below_one(self):
        assert run_e2e_multigpu.cap_jobs_to_host_cpus(64, 8, budget=4) == 1

    def test_defaults_to_process_cpu_budget(self, monkeypatch):
        monkeypatch.setattr(run_e2e_multigpu, "usable_cpu_count", lambda: 16)
        assert run_e2e_multigpu.cap_jobs_to_host_cpus(64, 4) == 4


class TestBuildShardCommand:
    """Tests for build_shard_command."""

    def test_single_shard_has_no_sharding_flags(self):
        cmd = run_e2e_multigpu.build_shard_command('llvm-lit', ['-v'], 8, 1, 1, ['tests'])
        assert cmd == [sys.executable, 'llvm-lit', '-j', '8', '-v', 'tests']

    def test_multi_shard_passes_shard_flags(self):
        cmd = run_e2e_multigpu.build_shard_command('llvm-lit', ['-v'], 20, 4, 3, ['tests'])
        assert cmd[cmd.index('--num-shards') + 1] == '4'
        assert cmd[cmd.index('--run-shard') + 1] == '3'
        assert cmd[cmd.index('-j') + 1] == '20'

    def test_lit_args_and_paths_come_last(self):
        cmd = run_e2e_multigpu.build_shard_command('llvm-lit', ['-v', '--time-tests'], 8, 1, 1,
                                                   ['a', 'b'])
        assert cmd[-4:] == ['-v', '--time-tests', 'a', 'b']
