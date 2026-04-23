//===- RocmSystemDetect.cpp - Detect ROCm devices -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detects ROCm devices on the current system without link-time
// dependencies on `libamdhip64`. The HIP entry points are resolved via
// the shared helpers in `mlir/ExecutionEngine/RocmDynamicLoader.h`,
// which hide the runtime in a private link-map namespace and therefore
// keep ROCm's libLLVM out of the host process's LLVM scope.
//
// This translation unit is also the canonical *owner* of the HIP
// handle: it exposes the handle through the extern-C function
// `mlirRocmSystemDetectGetHipHandle()` so other loaders
// (`libmlir_rocm_runtime.so`, rocMLIR's `MLIRRockOps`, the tuning
// driver) can reuse it. KFD only allows one user-space HSA session per
// process; a second `dlmopen(LM_ID_NEWLM, ...)` would end up in a
// different namespace and every call from it would return
// `hipErrorNoDevice`.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/RocmSystemDetect.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/ExecutionEngine/RocmRuntimeLoader.h"

#include "llvm/Support/Error.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "hip/hip_runtime.h"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <cstdio>

#define DEBUG_TYPE "execution-engine-rocm-system-detect"

using namespace mlir;

#define TO_STR(x) llvm::StringRef(std::to_string(x))

namespace {

/// Function-pointer table for the (very small) HIP surface this file
/// uses. A null `handle` means HIP could not be loaded at all; the
/// constructor then treats the system as having zero GPUs.
struct HipSymbols {
  rocm_loader::LoadedLibrary lib;
  hipError_t (*getDeviceCount)(int *) = nullptr;
  hipError_t (*getDeviceProperties)(hipDeviceProp_t *, int) = nullptr;
};

const HipSymbols &getHip() {
  static HipSymbols syms = []() {
    HipSymbols s;
    // `CoordinationPolicy::Owned`: RocmSystemDetect is the canonical
    // owner of the HIP handle. We must not consult
    // `mlirRocmSystemDetectGetHipHandle` here; that symbol points back
    // at *us* and would be uninitialised at this point.
    s.lib = rocm_loader::loadRocmLibrary(
        rocm_loader::Library::Hip, /*relatedHandle=*/nullptr,
        rocm_loader::CoordinationPolicy::Owned);
    if (!s.lib.handle) {
      std::fprintf(stderr,
                   "RocmSystemDetect: libamdhip64 not found on the loader "
                   "search path; ROCm device detection disabled.\n");
      return s;
    }
    s.getDeviceCount = reinterpret_cast<hipError_t (*)(int *)>(
        rocm_loader::resolveRocmSymbol(s.lib, "hipGetDeviceCount"));
    // ROCm 6+ ABI-stable variant; fall back to the legacy symbol on
    // older installs.
    s.getDeviceProperties =
        reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
            rocm_loader::resolveRocmSymbol(s.lib,
                                           "hipGetDevicePropertiesR0600"));
    if (!s.getDeviceProperties) {
      s.getDeviceProperties =
          reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
              rocm_loader::resolveRocmSymbol(s.lib, "hipGetDeviceProperties"));
    }
    if (!s.getDeviceCount || !s.getDeviceProperties) {
      std::fprintf(stderr,
                   "RocmSystemDetect: libamdhip64 loaded but required "
                   "symbols are missing; ROCm device detection disabled.\n");
      s.lib.handle = nullptr;
    }
    return s;
  }();
  return syms;
}

} // namespace

// Cross-library coordination export. The full contract lives on the
// declaration in `RocmSystemDetect.h`; the visibility annotations here
// publish the symbol in the host process's dynamic symbol table so
// other loaders can find it via `RTLD_DEFAULT`.
#ifdef _WIN32
#define MLIR_ROCM_SHARED_HIP_EXPORT __declspec(dllexport)
#else
#define MLIR_ROCM_SHARED_HIP_EXPORT __attribute__((visibility("default")))
#endif

extern "C" MLIR_ROCM_SHARED_HIP_EXPORT void *
mlirRocmSystemDetectGetHipHandle() {
  return getHip().lib.handle;
}

RocmSystemDetect::RocmSystemDetect() {
  const HipSymbols &hip = getHip();
  if (!hip.lib.handle)
    return;

  // collect all GPUs
  int count = 0;
  hipError_t herr = hip.getDeviceCount(&count);
  if (herr != hipSuccess) {
    llvm::errs() << "hipGetDeviceCount() should never fail\n";
    return;
  }

  for (int i = 0; i < count; ++i) {
    hipDeviceProp_t deviceProps;
    herr = hip.getDeviceProperties(&deviceProps, i);
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
