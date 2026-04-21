//===- RocmArchRuntime.cpp - Native AMD GPU arch query runtime ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the `mlir_rocm_arch_runtime` shared library. This is the
// only translation unit in rocMLIR that links the AMD HIP / HSA runtimes.
// See `mlir/include/mlir/ExecutionEngine/RocmArchRuntime.h` for rationale.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/RocmArchRuntime.h"

#include "hip/hip_runtime_api.h"

#ifndef _WIN32
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#endif

#include <cstring>

namespace {

// Marker to control symbol visibility. The library is built with
// `-fvisibility=hidden`; entry points are tagged explicitly so we ship a
// minimal, stable symbol surface.
#ifdef _WIN32
#define ROCM_ARCH_RUNTIME_EXPORT __declspec(dllexport)
#else
#define ROCM_ARCH_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

#ifndef _WIN32
struct AgentQuery {
  uint32_t targetDeviceId;
  int numCpus;
  uint32_t simdsPerCU;
  uint32_t maxWavesPerCU;
  uint32_t numXCC;
  bool found;
};

// Adapted from rocminfo.cc; see
// https://github.com/ROCm/rocm-systems/blob/develop/projects/rocminfo/rocminfo.cc
hsa_status_t acquireAgentInfo(hsa_agent_t agent, void *data) {
  auto *q = static_cast<AgentQuery *>(data);

  hsa_device_type_t deviceType;
  if (hsa_status_t err =
          hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &deviceType);
      err != HSA_STATUS_SUCCESS)
    return err;

  if (deviceType != HSA_DEVICE_TYPE_GPU) {
    ++q->numCpus;
    return HSA_STATUS_SUCCESS;
  }

  uint32_t internalNodeId = 0;
  if (hsa_status_t err = hsa_agent_get_info(
          agent,
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
          &internalNodeId);
      err != HSA_STATUS_SUCCESS)
    return err;

  if (internalNodeId < static_cast<uint32_t>(q->numCpus))
    return HSA_STATUS_SUCCESS;

  uint32_t gpuId = internalNodeId - static_cast<uint32_t>(q->numCpus);
  if (gpuId != q->targetDeviceId)
    return HSA_STATUS_SUCCESS;

  if (hsa_status_t err = hsa_agent_get_info(
          agent,
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU),
          &q->simdsPerCU);
      err != HSA_STATUS_SUCCESS)
    return err;
  if (hsa_status_t err = hsa_agent_get_info(
          agent,
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
          &q->maxWavesPerCU);
      err != HSA_STATUS_SUCCESS)
    return err;
  if (hsa_status_t err = hsa_agent_get_info(
          agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_XCC),
          &q->numXCC);
      err != HSA_STATUS_SUCCESS)
    return err;

  q->found = true;
  return HSA_STATUS_SUCCESS;
}
#endif // _WIN32

// Apply the WGP-as-CU correction for Navi-class GPUs.
// See `fixNaviProperties` in AmdArchDb.cpp for the original commentary.
void applyNaviCorrection(uint32_t warpSize, uint32_t &simdsPerCU,
                         uint32_t &maxWavesPerCU, uint64_t &sharedMemPerCU) {
  if (warpSize != 32)
    return;
  simdsPerCU *= 2;
  maxWavesPerCU *= 2;
  sharedMemPerCU *= 2;
}

} // namespace

extern "C" {

ROCM_ARCH_RUNTIME_EXPORT
int32_t mlirRocmArchRuntimeAbiVersion(void) {
  return MLIR_ROCM_ARCH_RUNTIME_ABI_VERSION;
}

ROCM_ARCH_RUNTIME_EXPORT
uint32_t mlirRocmArchRuntimeDeviceCount(void) {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess)
    return 0;
  if (count < 0)
    return 0;
  return static_cast<uint32_t>(count);
}

ROCM_ARCH_RUNTIME_EXPORT
int32_t
mlirRocmArchRuntimeGetProperties(uint32_t deviceId,
                                 struct MlirRocmArchProperties *outProps) {
  if (!outProps)
    return MLIR_ROCM_ARCH_HIP_ERROR;
  std::memset(outProps, 0, sizeof(*outProps));

  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, static_cast<int>(deviceId)) != hipSuccess)
    return MLIR_ROCM_ARCH_HIP_ERROR;

  // gcnArchName is a fixed-size NUL-terminated string in HIP. Copy with an
  // explicit cap so we never overrun the destination if HIP grows the field.
  std::strncpy(outProps->gcnArchName, prop.gcnArchName,
               MLIR_ROCM_ARCH_NAME_MAX - 1);
  outProps->gcnArchName[MLIR_ROCM_ARCH_NAME_MAX - 1] = '\0';

  outProps->multiProcessorCount =
      static_cast<uint32_t>(prop.multiProcessorCount);
  outProps->warpSize = static_cast<uint32_t>(prop.warpSize);
  outProps->sharedMemPerCU =
      static_cast<uint64_t>(prop.maxSharedMemoryPerMultiProcessor);
  outProps->sharedMemPerBlock = static_cast<uint64_t>(prop.sharedMemPerBlock);

#ifdef _WIN32
  // HSA is not supported on Windows; HIP fields above are sufficient and the
  // caller will fall back to the static `AmdArchDb` defaults for HSA-derived
  // values.
  return MLIR_ROCM_ARCH_OK;
#else
  AgentQuery query{};
  query.targetDeviceId = deviceId;
  if (hsa_iterate_agents(acquireAgentInfo, &query) != HSA_STATUS_SUCCESS ||
      !query.found)
    return MLIR_ROCM_ARCH_HSA_ERROR;

  applyNaviCorrection(outProps->warpSize, query.simdsPerCU, query.maxWavesPerCU,
                      outProps->sharedMemPerCU);

  outProps->simdsPerCU = query.simdsPerCU;
  outProps->maxWavesPerCU = query.maxWavesPerCU;
  outProps->numXCC = query.numXCC;
  outProps->hsaValid = 1;
  return MLIR_ROCM_ARCH_OK;
#endif
}

} // extern "C"
