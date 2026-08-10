# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests for parameterSweeps.py.

Covers how sweep configurations are handed out to GPUs. The work itself is
stubbed out, so no GPU or build tree is needed.
"""
import asyncio
import sys
from pathlib import Path

# Ensure we can import parameterSweeps (lives in mlir/utils/performance).
_test_dir = Path(__file__).resolve().parent
_sys_path_parent = str(_test_dir.parent)
if _sys_path_parent not in sys.path:
    sys.path.insert(0, _sys_path_parent)
# Mock hip and amd_arch_db so the module imports without ROCm (CI has no GPU).
exec(
    open(_test_dir / "mock_hip.py").read(), {
        "__file__": str(_test_dir / "mock_hip.py"),
        "sys": sys
    })

import parameterSweeps  # noqa: E402


def _options(gpu_ids, concurrent_tests=2):
    return parameterSweeps.Options(debug=False,
                                   quiet=True,
                                   debug_fails=False,
                                   arch="gfx942",
                                   flags=[],
                                   concurrent_tests=concurrent_tests,
                                   num_cu=304,
                                   num_chiplets=8,
                                   gpu_ids=gpu_ids)


def _run_sweep(monkeypatch, options, num_configs):
    """Run a sweep with the actual test execution stubbed out.

    Returns the GPU id each configuration was handed to, in order.
    """
    assignments = []

    async def fake_drop_good_config(config, options, paths, gpu_id=None):
        assignments.append((config, gpu_id))
        return parameterSweeps.TestResult.PASS

    monkeypatch.setattr(parameterSweeps, "drop_good_config", fake_drop_good_config)
    result = asyncio.run(
        parameterSweeps.sweep_parameters(range(num_configs), lambda p, o: p, options, paths=None))
    return assignments, result


class TestSweepGpuAssignment:
    """Tests for how sweep_parameters spreads configurations over GPUs."""

    def test_configs_cycle_through_selected_gpus(self, monkeypatch):
        assignments, _ = _run_sweep(monkeypatch, _options(gpu_ids=(0, 1, 2)), num_configs=7)
        assert [gpu_id for _, gpu_id in assignments] == [0, 1, 2, 0, 1, 2, 0]

    def test_every_config_is_dispatched_once_in_order(self, monkeypatch):
        assignments, (passed, invalid, failing) = _run_sweep(monkeypatch,
                                                             _options(gpu_ids=(0, 1)),
                                                             num_configs=5)
        assert [config for config, _ in assignments] == [0, 1, 2, 3, 4]
        assert (passed, invalid, failing) == (5, 0, [])

    def test_single_gpu_leaves_every_config_unpinned(self, monkeypatch):
        assignments, _ = _run_sweep(monkeypatch, _options(gpu_ids=(None,)), num_configs=3)
        assert [gpu_id for _, gpu_id in assignments] == [None, None, None]

    def test_assignment_spans_batches(self, monkeypatch):
        # Configurations are dispatched in batches of `concurrent_tests`; the
        # rotation has to continue across batch boundaries, not restart.
        assignments, _ = _run_sweep(monkeypatch,
                                    _options(gpu_ids=(0, 1, 2), concurrent_tests=2),
                                    num_configs=6)
        assert [gpu_id for _, gpu_id in assignments] == [0, 1, 2, 0, 1, 2]
