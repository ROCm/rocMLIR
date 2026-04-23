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
// Linker discipline: this translation unit does NOT link `libamdhip64` at
// build time. Doing so would make every consumer of
// `MLIRRocmExecutionEngineUtils` (notably `xmir-runner`) transitively pull
// in `libamd_comgr` and ROCm's `libLLVM.so.23`. The dynamic linker would
// then run libLLVM.so.23's static initializers from `_dl_init` BEFORE
// `main()` is reached, and they would unify the `cl::SubCommand`
// singleton against rocMLIR's embedded LLVM, aborting the process with
// "Option '...' already exists!" / SmallPtrSet assertion failures inside
// `CommandLineParser::forEachSubCommand`.
//
// Instead, the two HIP entry points used by `RocmSystemDetect` are
// resolved at construction time via `dlmopen(LM_ID_NEWLM, ...)` on glibc
// (plain `dlopen` on other POSIX, `LoadLibraryA` on Windows). The fresh
// link-map namespace keeps libamd_comgr and libLLVM.so.23 invisible to
// the host process's LLVM. This mirrors the same pattern used by
// `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp` in rocMLIR and by
// `RocmRuntimeWrappers.cpp` next door.
//
//===----------------------------------------------------------------------===//

// glibc's `dlmopen` lives behind `_GNU_SOURCE`; define it before any
// system header is transitively included.
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "mlir/ExecutionEngine/RocmSystemDetect.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"

#include "llvm/Support/Error.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "hip/hip_runtime.h"
#pragma GCC diagnostic pop

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdio>

#define DEBUG_TYPE "execution-engine-rocm-system-detect"

using namespace mlir;

#define TO_STR(x) llvm::StringRef(std::to_string(x))

namespace {

using OsHandle = void *;

/// SONAMEs to try, in order. Mirrors the candidate list in
/// rocMLIR's mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp and
/// external/llvm-project/mlir/lib/ExecutionEngine/RocmRuntimeWrappers.cpp.
constexpr const char *kHipCandidates[] = {
#ifdef _WIN32
    "amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll",
#else
    "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so",
#endif
};

OsHandle openHipRuntime() {
  for (const char *cand : kHipCandidates) {
#ifdef _WIN32
    if (HMODULE h = ::LoadLibraryA(cand))
      return h;
#elif defined(__GLIBC__)
    // Fresh link-map namespace: any libLLVM.so.* that HIP / comgr
    // transitively load resolves only against its own copies and stays
    // invisible to the host process.
    if (void *h = ::dlmopen(LM_ID_NEWLM, cand, RTLD_LAZY))
      return h;
#else
    if (void *h = ::dlopen(cand, RTLD_LAZY | RTLD_LOCAL))
      return h;
#endif
  }
  return nullptr;
}

void *osSym(OsHandle h, const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(
      ::GetProcAddress(static_cast<HMODULE>(h), name));
#else
  return ::dlsym(h, name);
#endif
}

/// Function-pointer table for the (very small) HIP surface this file
/// uses. A null `getDeviceCount` means HIP could not be loaded at all
/// and the constructor should treat the system as having zero GPUs.
struct HipSymbols {
  OsHandle handle = nullptr;
  hipError_t (*getDeviceCount)(int *) = nullptr;
  hipError_t (*getDeviceProperties)(hipDeviceProp_t *, int) = nullptr;
};

const HipSymbols &getHip() {
  static HipSymbols syms = []() {
    HipSymbols s;
    s.handle = openHipRuntime();
    if (!s.handle) {
      // Soft failure: leave function pointers null so the caller can
      // proceed with an empty device list.
      std::fprintf(stderr,
                   "RocmSystemDetect: libamdhip64 not found on the loader "
                   "search path; ROCm device detection disabled.\n");
      return s;
    }
    s.getDeviceCount = reinterpret_cast<hipError_t (*)(int *)>(
        osSym(s.handle, "hipGetDeviceCount"));
    // ROCm 6+ ABI-stable variant; fall back to the legacy symbol on
    // older installs.
    s.getDeviceProperties =
        reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
            osSym(s.handle, "hipGetDevicePropertiesR0600"));
    if (!s.getDeviceProperties) {
      s.getDeviceProperties =
          reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
              osSym(s.handle, "hipGetDeviceProperties"));
    }
    if (!s.getDeviceCount || !s.getDeviceProperties) {
      std::fprintf(stderr,
                   "RocmSystemDetect: libamdhip64 loaded but required "
                   "symbols are missing; ROCm device detection disabled.\n");
      s.handle = nullptr;
    }
    return s;
  }();
  return syms;
}

} // namespace

// Cross-library coordination: `libmlir_rocm_runtime.so` (the JIT
// kernel-launch wrapper) also needs HIP. Loading libamdhip64 a second
// time via `dlmopen(LM_ID_NEWLM, ...)` in another link-map namespace
// makes the second HSA/HIP instance fail every call with
// `hipErrorNoDevice`, because KFD only permits one user-space session
// per process. Expose our dlmopen'd HIP handle so the runtime wrapper
// can dlsym this symbol via RTLD_DEFAULT and reuse our handle instead
// of opening its own. RocmSystemDetect is the canonical owner because
// its TU is link-loaded into xmir-runner via libMLIRRocmExecutionEngineUtils
// and runs first; libmlir_rocm_runtime.so is dlopen'd later as a JIT
// shared lib. An absent symbol means "no shared HIP handle is
// available", and the consumer falls back to its own dlmopen.
//
// The function is `__attribute__((visibility("default")))` so it lands
// in the host process's dynamic symbol table for RTLD_DEFAULT lookup.
#ifdef _WIN32
#define MLIR_ROCM_SHARED_HIP_EXPORT __declspec(dllexport)
#else
#define MLIR_ROCM_SHARED_HIP_EXPORT __attribute__((visibility("default")))
#endif

extern "C" MLIR_ROCM_SHARED_HIP_EXPORT void *
mlirRocmSystemDetectGetHipHandle() {
  return getHip().handle;
}

RocmSystemDetect::RocmSystemDetect() {
  const HipSymbols &hip = getHip();
  if (!hip.handle)
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
