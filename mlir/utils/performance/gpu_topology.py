# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared helpers for distributing GPU test workloads across multiple devices.

CI nodes commonly expose several GPUs. These helpers detect the visible GPUs,
confirm they share a single architecture (a prerequisite for safely reusing the
same compiled kernels), and build per-process environments that isolate work to
one device via ROCR_VISIBLE_DEVICES (the same mechanism tuningRunner.py uses).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple


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

    Uses hip-python, matching the rest of the test tooling (the lit configs and
    perfRunner already require it); callers treat a failure as "use one GPU".
    """
    from hip import hip
    archs = []
    device_count = _hip_check(hip.hipGetDeviceCount())
    for device in range(device_count):
        props = hip.hipDeviceProp_t()
        _hip_check(hip.hipGetDeviceProperties(props, device))
        archs.append(props.gcnArchName.decode('utf-8'))
    return archs


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
        # Preserve order but drop duplicates so we never shard the same device twice.
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


def make_isolated_gpu_env(gpu_id: Optional[int]) -> Optional[dict]:
    """Build an environment dict that isolates a child process to one GPU.

    Returns ``None`` when ``gpu_id`` is ``None`` so callers can pass it straight
    through to ``subprocess``/``asyncio`` APIs to mean "inherit the environment".
    ROCR_VISIBLE_DEVICES isolates at the HSA/ROCr level (below HIP); we clear
    HIP_VISIBLE_DEVICES to avoid the two layers disagreeing.
    """
    if gpu_id is None:
        return None
    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("HIP_VISIBLE_DEVICES", None)
    return env
