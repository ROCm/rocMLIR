//===- RocmSystemDetect.cpp - Detect ROCm devices -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the system detection of ROCm devices on the current
// system.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/RocmSystemDetect.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"

#include "llvm/Support/Error.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "hip/hip_runtime.h"
#pragma GCC diagnostic pop

#define DEBUG_TYPE "execution-engine-rocm-system-detect"

using namespace mlir;

#define TO_STR(x) llvm::StringRef(std::to_string(x))

RocmSystemDetect::RocmSystemDetect() {
  // collect all GPUs
  int count = 0;
  hipError_t herr = hipGetDeviceCount(&count);
  if (herr != hipSuccess) {
    llvm::errs() << "hipGetDeviceCount() should never fail\n";
    return;
  }

  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t deviceProps;
    herr = hipGetDeviceProperties(&deviceProps, i);
    if (herr == hipSuccess) {
      RocmDeviceName chip(deviceProps.gcnArchName);
      if (chip) {
        llvm::StringRef chipName(chip.getChip());
        llvm::StringRef vendor("AMD");
        llvm::StringRef features(chip.getFeatures());
        llvm::StringRef triple(chip.getTriple());

        auto itr = std::find_if(begin(), end(), [&](const SystemDevice &dev) {
          return dev.chip == chipName;
        });
        if (itr != end()) {
          itr->count++;
        } else {
          push_back(
              {SystemDevice::Type::EGPU,
               chipName,
               1,
               {{"vendor", vendor},
                {"chip", chipName},
                {"features", features},
                {"triple", triple},
                {"major", TO_STR(deviceProps.major)},
                {"minor", TO_STR(deviceProps.minor)},
                {"multiProcessorCount",
                 TO_STR(deviceProps.multiProcessorCount)},
                {"sharedMemPerBlock", TO_STR(deviceProps.sharedMemPerBlock)},
                {"regsPerBlock", TO_STR(deviceProps.regsPerBlock)},
                {"warpSize", TO_STR(deviceProps.warpSize)}}});
        }
      } else {
        llvm::errs() << "ROCm device failed " << deviceProps.gcnArchName
                     << "\n";
      }
    } else {
      llvm::errs() << "hipGetDeviceProperties() failed for Device " << i
                   << "\n";
    }
  }
}

/*
Full list of properties:
  name = '\000' <repeats 255 times>,
  totalGlobalMem = 0x3ff000000,
  sharedMemPerBlock = 0x10000,
  regsPerBlock = 0x10000,
  warpSize = 0x20,
  maxThreadsPerBlock = 0x400,
  maxThreadsDim = {0x400, 0x400, 0x400},
  maxGridSize = {0x7fffffff, 0x7fffffff, 0x7fffffff},
  clockRate = 0x274a98,
  memoryClockRate = 0xf4240,
  memoryBusWidth = 0x100,
  totalConstMem = 0x3ff000000,
  major = 0xa,
  minor = 0x3,
  multiProcessorCount = 0x24,
  l2CacheSize = 0x400000,
  maxThreadsPerMultiProcessor = 0x800,
  computeMode = 0x0,
  clockInstructionRate = 0xf4240,
  arch = {
    hasGlobalInt32Atomics = 0x1,
    hasGlobalFloatAtomicExch = 0x1,
    hasSharedInt32Atomics = 0x1,
    hasSharedFloatAtomicExch = 0x1,
    hasFloatAtomicAdd = 0x1,
    hasGlobalInt64Atomics = 0x1,
    hasSharedInt64Atomics = 0x1,
    hasDoubles = 0x1,
    hasWarpVote = 0x1,
    hasWarpBallot = 0x1,
    hasWarpShuffle = 0x1,
    hasFunnelShift = 0x0,
    hasThreadFenceSystem = 0x1,
    hasSyncThreadsExt = 0x0,
    hasSurfaceFuncs = 0x0,
    has3dGrid = 0x1,
    hasDynamicParallelism = 0x0
  },
  concurrentKernels = 0x1,
  pciDomainID = 0x0,
  pciBusID = 0xa,
  pciDeviceID = 0x0,
  maxSharedMemoryPerMultiProcessor = 0x10000,
  isMultiGpuBoard = 0x0,
  canMapHostMemory = 0x1,
  gcnArch = 0x406,
  gcnArchName = "gfx1030", '\000' <repeats 248 times>,
  integrated = 0x0,
  cooperativeLaunch = 0x0,
  cooperativeMultiDeviceLaunch = 0x0,
  maxTexture1DLinear = 0xfffffff0,
  maxTexture1D = 0x4000,
  maxTexture2D = {0x4000, 0x4000},
  maxTexture3D = {0x4000, 0x4000, 0x2000},
  hdpMemFlushCntl = 0x7fffe7672000,
  hdpRegFlushCntl = 0x7fffe7672004,
  memPitch = 0x3ff000000,
  textureAlignment = 0x100,
  texturePitchAlignment = 0x100,
  kernelExecTimeoutEnabled = 0x0,
  ECCEnabled = 0x0,
  tccDriver = 0x0,
  cooperativeMultiDeviceUnmatchedFunc = 0x0,
  cooperativeMultiDeviceUnmatchedGridDim = 0x0,
  cooperativeMultiDeviceUnmatchedBlockDim = 0x0,
  cooperativeMultiDeviceUnmatchedSharedMem = 0x0,
  isLargeBar = 0x0,
  asicRevision = 0x1,
  managedMemory = 0x0,
  directManagedMemAccessFromHost = 0x48,
  concurrentManagedAccess = 0x0,
  pageableMemoryAccess = 0x0,
  pageableMemoryAccessUsesHostPageTables = 0x0
*/
