# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests for gpu_topology.py.

These cover GPU selection (single/multi/heterogeneous, explicit requests, and
error fallbacks) and per-process GPU isolation. They run in CI without a real
GPU: hip is mocked and get_per_device_archs is monkeypatched per scenario.
"""
import sys
from pathlib import Path

# Ensure we can import gpu_topology (lives in mlir/utils/performance).
_test_dir = Path(__file__).resolve().parent
_sys_path_parent = str(_test_dir.parent)
if _sys_path_parent not in sys.path:
    sys.path.insert(0, _sys_path_parent)
# Mock hip so gpu_topology can be imported/exercised without ROCm (CI has no GPU).
exec(
    open(_test_dir / "mock_hip.py").read(), {
        "__file__": str(_test_dir / "mock_hip.py"),
        "sys": sys
    })

import gpu_topology  # noqa: E402 - must run after mock_hip


def _clear_visible_devices(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)


def _fake_archs(monkeypatch, archs):
    monkeypatch.setattr(gpu_topology, "get_per_device_archs", lambda: list(archs))


class TestSelectGpuIds:
    """Tests for select_gpu_ids."""

    def test_respects_preset_rocr_visible_devices(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
        gpu_ids, arch, msg = gpu_topology.select_gpu_ids()
        assert gpu_ids == [None]
        assert arch is None
        assert "VISIBLE_DEVICES" in msg

    def test_respects_preset_hip_visible_devices(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        gpu_ids, arch, _ = gpu_topology.select_gpu_ids()
        assert gpu_ids == [None]
        assert arch is None

    def test_enumeration_failure_falls_back(self, monkeypatch):
        _clear_visible_devices(monkeypatch)

        def _boom():
            raise RuntimeError("no driver")

        monkeypatch.setattr(gpu_topology, "get_per_device_archs", _boom)
        gpu_ids, arch, msg = gpu_topology.select_gpu_ids()
        assert gpu_ids == [None]
        assert arch is None
        assert "failed" in msg

    def test_single_gpu(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        _fake_archs(monkeypatch, ["gfx942"])
        gpu_ids, arch, _ = gpu_topology.select_gpu_ids()
        assert gpu_ids == [None]
        assert arch is None

    def test_homogeneous_multi_gpu(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        _fake_archs(monkeypatch, ["gfx942"] * 8)
        gpu_ids, arch, _ = gpu_topology.select_gpu_ids()
        assert gpu_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert arch == "gfx942"

    def test_mixed_archs_no_request_falls_back(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        _fake_archs(monkeypatch, ["gfx942", "gfx1100"])
        gpu_ids, arch, msg = gpu_topology.select_gpu_ids()
        assert gpu_ids == [None]
        assert arch is None
        assert "mixed" in msg

    def test_requested_valid_homogeneous_subset(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        _fake_archs(monkeypatch, ["gfx942"] * 4)
        gpu_ids, arch, _ = gpu_topology.select_gpu_ids(requested=[1, 3])
        assert gpu_ids == [1, 3]
        assert arch == "gfx942"

    def test_requested_out_of_range_falls_back(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        _fake_archs(monkeypatch, ["gfx942", "gfx942"])
        gpu_ids, arch, msg = gpu_topology.select_gpu_ids(requested=[0, 99])
        assert gpu_ids == [None]
        assert arch is None
        assert "out of range" in msg

    def test_requested_deduplicated(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        _fake_archs(monkeypatch, ["gfx942"] * 4)
        gpu_ids, arch, _ = gpu_topology.select_gpu_ids(requested=[2, 2, 0])
        assert gpu_ids == [2, 0]
        assert arch == "gfx942"

    def test_requested_mixed_archs_falls_back(self, monkeypatch):
        _clear_visible_devices(monkeypatch)
        _fake_archs(monkeypatch, ["gfx942", "gfx1100"])
        gpu_ids, arch, msg = gpu_topology.select_gpu_ids(requested=[0, 1])
        assert gpu_ids == [None]
        assert arch is None
        assert "single arch" in msg


class TestMakeIsolatedGpuEnv:
    """Tests for make_isolated_gpu_env."""

    def test_none_returns_none(self):
        assert gpu_topology.make_isolated_gpu_env(None) is None

    def test_sets_rocr_and_clears_hip(self, monkeypatch):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "5")
        env = gpu_topology.make_isolated_gpu_env(2)
        assert env is not None
        assert env["ROCR_VISIBLE_DEVICES"] == "2"
        assert "HIP_VISIBLE_DEVICES" not in env


class TestGetPerDeviceArchs:
    """Tests for get_per_device_archs (exercises the mocked hip path)."""

    def test_returns_arch_per_device(self):
        archs = gpu_topology.get_per_device_archs()
        assert archs == ["gfx900"]


def _topologies(gpu_to_node, node_to_cpus):
    """Build a (GpuTopology, NumaTopology) pair from plain mappings."""
    gpus = {
        gpu_id: gpu_topology.Gpu(gpu_id=gpu_id, sku="mock-sku", numa_node=node)
        for gpu_id, node in gpu_to_node.items()
    }
    return (gpu_topology.GpuTopology(gpus=gpus),
            gpu_topology.NumaTopology(numa_to_cpus=node_to_cpus))


class TestGpuTopology:
    """Tests for GpuTopology (moved here from tuningRunner)."""

    def test_get_numa_node(self):
        gpus, _ = _topologies({0: 0, 1: 1}, {0: [0], 1: [1]})
        assert gpus.get_numa_node(1) == 1

    def test_homogeneity_single_gpu_is_always_homogeneous(self):
        gpus, _ = _topologies({0: 0}, {0: [0]})
        assert gpus.validate_homogeneity([0])

    def test_homogeneity_mixed_skus(self):
        gpus = gpu_topology.GpuTopology(
            gpus={
                0: gpu_topology.Gpu(gpu_id=0, sku="a", numa_node=0),
                1: gpu_topology.Gpu(gpu_id=1, sku="b", numa_node=0),
            })
        assert not gpus.validate_homogeneity([0, 1])


class TestAllocateCpusPerGpu:
    """Tests for allocate_cpus_per_gpu."""

    def test_splits_node_cpus_between_gpus_on_that_node(self, monkeypatch):
        monkeypatch.setattr(gpu_topology, "_usable_cpus", lambda: set(range(16)))
        gpus, numa = _topologies({0: 0, 1: 0}, {0: list(range(16))})
        assert gpu_topology.allocate_cpus_per_gpu([0, 1], gpus, numa) == {0: 8, 1: 8}

    def test_gpus_on_separate_nodes_get_their_own_cpus(self, monkeypatch):
        monkeypatch.setattr(gpu_topology, "_usable_cpus", lambda: set(range(16)))
        gpus, numa = _topologies({0: 0, 1: 1}, {0: list(range(8)), 1: list(range(8, 16))})
        assert gpu_topology.allocate_cpus_per_gpu([0, 1], gpus, numa) == {0: 8, 1: 8}

    def test_respects_affinity_mask(self, monkeypatch):
        # Only 4 of the node's 16 CPUs are usable (e.g. a container CPU limit).
        monkeypatch.setattr(gpu_topology, "_usable_cpus", lambda: set(range(4)))
        gpus, numa = _topologies({0: 0, 1: 0}, {0: list(range(16))})
        assert gpu_topology.allocate_cpus_per_gpu([0, 1], gpus, numa) == {0: 2, 1: 2}

    def test_never_allocates_zero(self, monkeypatch):
        monkeypatch.setattr(gpu_topology, "_usable_cpus", lambda: {0})
        gpus, numa = _topologies({0: 0, 1: 0, 2: 0, 3: 0}, {0: [0]})
        assert gpu_topology.allocate_cpus_per_gpu([0, 1, 2, 3], gpus, numa) == {
            0: 1,
            1: 1,
            2: 1,
            3: 1
        }


class TestScaleCpuAllocation:
    """Tests for scale_cpu_allocation."""

    def test_limit_above_total_is_unchanged(self):
        assert gpu_topology.scale_cpu_allocation({0: 8, 1: 8}, 32) == {0: 8, 1: 8}

    def test_scales_down_proportionally(self):
        assert gpu_topology.scale_cpu_allocation({0: 8, 1: 8}, 8) == {0: 4, 1: 4}

    def test_never_scales_below_one(self):
        assert gpu_topology.scale_cpu_allocation({0: 8, 1: 8}, 1) == {0: 1, 1: 1}


class TestUsableCpuCount:
    """Tests for usable_cpu_count."""

    def test_matches_affinity_mask(self, monkeypatch):
        monkeypatch.setattr(gpu_topology, "_usable_cpus", lambda: {0, 1, 2})
        assert gpu_topology.usable_cpu_count() == 3
