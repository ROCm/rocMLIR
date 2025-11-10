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
      RocmDeviceName arch;
      if (succeeded(arch.parse(deviceProps.gcnArchName))) {
        llvm::StringRef chip(arch.getChip());
        llvm::StringRef vendor("AMD");
        llvm::StringMap<bool> features = arch.getFeatures();
        llvm::StringRef triple(arch.getTriple());

        auto itr = std::find_if(begin(), end(), [&](const SystemDevice &dev) {
          return dev.chip == chip && dev.features == features &&
                 dev.llvmTriple == triple;
        });
        if (itr != end()) {
          itr->count++;
        } else {
          push_back(
              {SystemDevice::Type::EGPU,
               triple,
               chip,
               features,
               1,
               {{"vendor", vendor},
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
