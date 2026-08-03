# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared GPU and NUMA topology helpers for distributing work across devices.

Collects the primitives used by the tuning runner, the parameter sweeps and the
sharded E2E driver: discovering the GPUs on a node, deciding which of them to
spread work over, isolating a child process to one device, and splitting the
host's CPUs between them.

This module deliberately keeps its imports light (stdlib only at module scope,
hip-python loaded on demand) so that callers which merely want to fall back to a
single GPU do not need a working ROCm stack to start up.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


def _hip_check(call_result):
    """Unwrap a hip-python call result, raising on a non-success status."""
    from hip import hip
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def get_per_device_archs() -> List[str]:
    """Return the gfx architecture name of every visible GPU, indexed by id.

    Uses hip-python rather than rocm-smi because only HIP reports the gfx name;
    rocm-smi reports the card SKU, which is not what kernels are compiled for.
    """
    from hip import hip
    archs = []
    device_count = _hip_check(hip.hipGetDeviceCount())
    for device in range(device_count):
        props = hip.hipDeviceProp_t()
        _hip_check(hip.hipGetDeviceProperties(props, device))
        archs.append(props.gcnArchName.decode('utf-8'))
    return archs


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
        # rocm-smi can take ~20s to enumerate large multi-GPU systems, so allow
        # a generous timeout to avoid spurious TimeoutExpired failures.
        output = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showtoponuma", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=60)
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


def select_gpu_ids(
        requested: Optional[List[int]] = None) -> Tuple[List[Optional[int]], Optional[str], str]:
    """Decide which GPUs to spread work across.

    Returns ``(gpu_ids, arch, message)``. ``gpu_ids == [None]`` means run on a
    single, unpinned GPU (and ``arch`` is ``None``). Otherwise ``gpu_ids`` lists
    the physical devices to isolate work to, all sharing architecture ``arch``.
    """
    # Respect a caller that already pinned visibility (e.g. lit shards, manual
    # runs); don't second-guess their device selection.
    if os.environ.get('ROCR_VISIBLE_DEVICES') or os.environ.get('HIP_VISIBLE_DEVICES'):
        return [None], None, "respecting pre-set *_VISIBLE_DEVICES; using a single GPU"

    try:
        archs = get_per_device_archs()
    except Exception as e:  # noqa: BLE001 - any GPU/runtime issue means fall back
        return [None], None, f"GPU enumeration failed ({e}); using the default GPU"

    count = len(archs)
    if count <= 1:
        return [None], None, "single GPU detected; running on one GPU"

    if requested:
        # Never shard the same device twice.
        unique_requested = list(dict.fromkeys(requested))
        invalid = [i for i in unique_requested if not 0 <= i < count]
        if invalid:
            return [None], None, (f"requested GPU ids {invalid} are out of range "
                                  f"(node has {count} GPU(s)); using the default GPU")
        selected_archs = {archs[i] for i in unique_requested}
        if len(selected_archs) != 1:
            return [None], None, (f"requested GPUs {unique_requested} are not a single arch "
                                  f"({sorted(selected_archs)}); using the default GPU")
        arch = next(iter(selected_archs))
        return unique_requested, arch, f"using requested GPUs {unique_requested} ({arch})"

    if len(set(archs)) > 1:
        return [None], None, (f"mixed GPU architectures ({sorted(set(archs))}); "
                              "using a single GPU")

    return list(range(count)), archs[0], f"distributing across {count} GPUs ({archs[0]})"


def set_isolated_gpu_env(env: Dict[str, str], gpu_id: int) -> None:
    """Modify environment to isolate subprocess to one physical GPU.

    Sets ROCR_VISIBLE_DEVICES at the HSA/ROCr level, providing complete isolation for all higher layers including HIP.
    """
    env["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("HIP_VISIBLE_DEVICES", None)  # Remove HIP_VISIBLE_DEVICES to avoid conflicts


def make_isolated_gpu_env(gpu_id: Optional[int]) -> Optional[Dict[str, str]]:
    """Create environment that isolates subprocess to one physical GPU.

    Returns ``None`` when ``gpu_id`` is ``None`` so callers can pass it straight
    through to ``subprocess``/``asyncio`` APIs to mean "inherit the environment".
    """
    if gpu_id is None:
        return None
    env = os.environ.copy()
    set_isolated_gpu_env(env, gpu_id)
    return env


def _usable_cpus() -> Set[int]:
    """CPUs this process is allowed to run on (respects cpuset restrictions)."""
    try:
        return set(os.sched_getaffinity(0))
    except AttributeError:  # sched_getaffinity is Linux-only
        return set(range(os.cpu_count() or 1))


def usable_cpu_count() -> int:
    """Number of CPUs this process is allowed to run on."""
    return len(_usable_cpus())


def allocate_cpus_per_gpu(gpu_ids: List[int], gpu_topology: GpuTopology,
                          numa_topology: NumaTopology) -> Dict[int, int]:
    """Split the host's CPUs across ``gpu_ids``, keeping each GPU on its own NUMA node.

    CPUs outside this process' affinity mask are left out, so a container that
    was given a slice of the machine does not hand out threads it cannot use.
    """
    usable = _usable_cpus()
    gpus_by_node: Dict[int, List[int]] = {}
    for gpu_id in gpu_ids:
        gpus_by_node.setdefault(gpu_topology.get_numa_node(gpu_id), []).append(gpu_id)

    allocation: Dict[int, int] = {}
    for node, gpus_on_node in gpus_by_node.items():
        cpus_on_node = len(usable.intersection(numa_topology.get_cpus_for_numa_node(node)))
        threads_each = max(1, cpus_on_node // len(gpus_on_node))
        for gpu_id in gpus_on_node:
            allocation[gpu_id] = threads_each

    return allocation


def scale_cpu_allocation(allocation: Dict[int, int], limit: int) -> Dict[int, int]:
    """Scale ``allocation`` down proportionally so its total stays within ``limit``."""
    total = sum(allocation.values())
    if total <= limit:
        return dict(allocation)

    scale = limit / total
    return {gpu_id: max(1, int(count * scale)) for gpu_id, count in allocation.items()}
