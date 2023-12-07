//===--- amdgpu/impl/impl.cpp ------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "rt.h"
#include <memory>

/*
 * Data
 */

template <const uint64_t active_timeout = 0>
hsa_status_t active_wait_for_signal(hsa_signal_t signal,
                                    hsa_signal_value_t init,
                                    hsa_signal_value_t success) {
  hsa_signal_value_t got = init;
  if (active_timeout) {
    got = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_NE, init,
                                    active_timeout, HSA_WAIT_STATE_ACTIVE);
    if (got == success)
      return HSA_STATUS_SUCCESS;
    DP("active_timeout %ld exceeded: switching to HSA_WAIT_STATE_BLOCKED.\n",
       active_timeout);
  }
  while (got == init)
    got = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_NE, init,
                                    UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  if (got != success)
    return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t is_locked(void *ptr, void **agentBaseAddress) {
  hsa_status_t err = HSA_STATUS_SUCCESS;
  hsa_amd_pointer_info_t info;
  info.size = sizeof(hsa_amd_pointer_info_t);
  err = hsa_amd_pointer_info(ptr, &info, /*alloc=*/nullptr,
                             /*num_agents_accessible=*/nullptr,
                             /*accessible=*/nullptr);
  if (err != HSA_STATUS_SUCCESS) {
    DP("Error when getting pointer info\n");
    return err;
  }

  if (info.type == HSA_EXT_POINTER_TYPE_LOCKED) {
    // When user passes in a basePtr+offset we need to fix the
    // locked pointer to include the offset: ROCr always returns
    // the base locked address, not the shifted one.
    if ((char *)info.hostBaseAddress <= (char *)ptr &&
        (char *)ptr < (char *)info.hostBaseAddress + info.sizeInBytes)
      *agentBaseAddress =
          (void *)((uint64_t)info.agentBaseAddress + (uint64_t)ptr -
                   (uint64_t)info.hostBaseAddress);
    else // address is already device-agent accessible, no need to compute
         // offset
      *agentBaseAddress = ptr;

  } else
    *agentBaseAddress = nullptr;

  return HSA_STATUS_SUCCESS;
}

// 0 = wait in HSA_WAIT_STATE_BLOCKED till complete
hsa_status_t wait_for_signal(hsa_signal_t signal, hsa_signal_value_t init,
                             hsa_signal_value_t success) {
  return active_wait_for_signal<0>(signal, init, success);
}
// switch to STATE_BLOCKED after 1 sec
hsa_status_t wait_for_signal_kernel(hsa_signal_t signal,
                                    hsa_signal_value_t init,
                                    hsa_signal_value_t success) {
  return active_wait_for_signal<1000000>(signal, init, success);
}
// switch to STATE_BLOCKED after 3 secs
hsa_status_t wait_for_signal_data(hsa_signal_t signal, hsa_signal_value_t init,
                                  hsa_signal_value_t success) {
  return active_wait_for_signal<3000000>(signal, init, success);
}
// Never switch to STATE_BLOCKED, stay in STATE_ACTIVE, not used yet
hsa_status_t wait_for_signal_active(hsa_signal_t signal,
                                    hsa_signal_value_t init,
                                    hsa_signal_value_t success) {
  return active_wait_for_signal<UINT64_MAX>(signal, init, success);
}

// host pointer (either src or dest) must be locked via hsa_amd_memory_lock
static hsa_status_t invoke_hsa_copy(hsa_signal_t signal, void *dest,
                                    hsa_agent_t agent, const void *src,
                                    size_t size) {
  const hsa_signal_value_t init = 1;
  hsa_signal_store_screlease(signal, init);

  hsa_status_t err = hsa_amd_memory_async_copy(dest, agent, src, agent, size, 0,
                                               nullptr, signal);

  return err;
}

struct implUnlockAndFreePtrDeletor {
  void operator()(void *p) {
    hsa_amd_memory_unlock(p);
    core::Runtime::Memfree(p); // ignore failure to free
  }
};

enum CopyDirection { H2D, D2H };

static hsa_status_t locking_async_memcpy(enum CopyDirection direction,
                                         hsa_signal_t signal, void *dest,
                                         hsa_agent_t agent, void *src,
                                         void *lockingPtr, size_t size,
                                         bool *user_locked) {
  void *lockedPtr = nullptr;
  hsa_status_t err = is_locked(lockingPtr, &lockedPtr);
  bool HostPtrIsLocked = true;
  if (err != HSA_STATUS_SUCCESS)
    return err;
  if (!lockedPtr) { // not locked
    HostPtrIsLocked = false;
    hsa_agent_t agents[1] = {agent};
    err = hsa_amd_memory_lock(lockingPtr, size, agents, /*num_agent=*/1,
                              (void **)&lockedPtr);
    if (err != HSA_STATUS_SUCCESS)
      return err;
    DP("locking_async_memcpy: lockingPtr=%p lockedPtr=%p Size = %lu\n",
       lockingPtr, lockedPtr, size);
  } else {
    *user_locked = true;
  }

  switch (direction) {
  case H2D:
    err = invoke_hsa_copy(signal, dest, agent, lockedPtr, size);
    break;
  case D2H:
    err = invoke_hsa_copy(signal, lockedPtr, agent, src, size);
    break;
  }

  if (err != HSA_STATUS_SUCCESS && !HostPtrIsLocked) {
    // do not leak locked host pointers, but discard potential error message
    // because the initial error was in the copy function
    hsa_amd_memory_unlock(lockingPtr);
    return err;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t impl_memcpy_h2d(hsa_signal_t signal, void *deviceDest,
                             void *hostSrc, size_t size,
                             hsa_agent_t device_agent,
                             hsa_amd_memory_pool_t MemoryPool,
                             bool *user_locked) {
  hsa_status_t err;

  err = locking_async_memcpy(CopyDirection::H2D, signal, deviceDest,
                             device_agent, hostSrc, hostSrc, size, user_locked);

  if (err == HSA_STATUS_SUCCESS)
    return err;

  // async memcpy sometimes fails in situations where
  // allocate + copy succeeds. Looks like it might be related to
  // locking part of a read only segment. Fall back for now.
  void *tempHostPtr;
  hsa_status_t ret = core::Runtime::HostMalloc(&tempHostPtr, size, MemoryPool);
  if (ret != HSA_STATUS_SUCCESS) {
    DP("HostMalloc: Unable to alloc %zu bytes for temp scratch\n", size);
    return ret;
  }
  std::unique_ptr<void, implUnlockAndFreePtrDeletor> del(tempHostPtr);
  memcpy(tempHostPtr, hostSrc, size);

  err =
      locking_async_memcpy(CopyDirection::H2D, signal, deviceDest, device_agent,
                           tempHostPtr, tempHostPtr, size, user_locked);
  if (err != HSA_STATUS_SUCCESS)
    return err;

  // free of host malloc (performed at the end of this block)
  // requires h2d memcopy to be completed.
  hsa_signal_value_t init = 1;
  hsa_signal_value_t success = 0;
  return wait_for_signal(signal, init, success);
}

hsa_status_t impl_memcpy_d2h(hsa_signal_t signal, void *hostDest,
                             void *deviceSrc, size_t size,
                             hsa_agent_t deviceAgent,
                             hsa_amd_memory_pool_t MemoryPool,
                             bool *user_locked) {
  hsa_status_t err;

  // device has always visibility over both pointers, so use that
  err = locking_async_memcpy(CopyDirection::D2H, signal, hostDest, deviceAgent,
                             deviceSrc, hostDest, size, user_locked);

  if (err == HSA_STATUS_SUCCESS)
    return err;

  // hsa_memory_copy sometimes fails in situations where
  // allocate + copy succeeds. Looks like it might be related to
  // locking part of a read only segment. Fall back for now.
  void *tempHostPtr;
  hsa_status_t ret = core::Runtime::HostMalloc(&tempHostPtr, size, MemoryPool);
  if (ret != HSA_STATUS_SUCCESS) {
    DP("HostMalloc: Unable to alloc %zu bytes for temp scratch\n", size);
    return ret;
  }
  std::unique_ptr<void, implUnlockAndFreePtrDeletor> del(tempHostPtr);

  err =
      locking_async_memcpy(CopyDirection::D2H, signal, tempHostPtr, deviceAgent,
                           deviceSrc, tempHostPtr, size, user_locked);
  if (err != HSA_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;

  // host-to-host memcopy requires completion of device-to-host memcopy
  hsa_signal_value_t init = 1;
  hsa_signal_value_t success = 0;
  err = wait_for_signal(signal, init, success);
  if (err != HSA_STATUS_SUCCESS)
    return err;

  memcpy(hostDest, tempHostPtr, size);
  return HSA_STATUS_SUCCESS;
}
