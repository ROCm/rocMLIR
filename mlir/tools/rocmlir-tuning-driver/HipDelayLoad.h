//===- HipDelayLoad.h - Lazy HIP/HIPRTC symbol resolution -------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Resolves the HIP runtime and HIPRTC entry points used by
// `rocmlir-tuning-driver` at run time rather than at link time.
//
// rocMLIR's executables embed their own LLVM (libLLVMSupport.so.*,
// libLLVMCodeGen.so.*, ...). Linking `libamdhip64` directly drags in
// `libamd_comgr` and ROCm's monolithic `libLLVM.so.<major>`, which the
// dynamic linker maps into the host process at startup. When that LLVM
// runs its `cl::opt` static initializers, the dynamic linker unifies
// the rocMLIR-side and ROCm-side `cl::*` symbols across the split-vs-
// monolithic libraries. The result is a SmallPtrSet "Bucket < End"
// assertion (or `LLVM ERROR: Option '...' already exists!`) firing in
// `_dl_init`, before `main()` is even reached. See the dlopenHip
// branch's `[EXTERNAL] Coordinate HIP namespace ...` commit for the
// canonical write-up.
//
// Each consumer .cpp follows this pattern:
//
//   #include <hip/hip_runtime.h>
//   #include <hip/hip_ext.h>     // optional, for hipExtModuleLaunchKernel
//   #include <hip/hiprtc.h>      // optional, for HIPRTC users
//   #include "HipDelayLoad.h"
//   #include "HipDelayLoadMacros.h"  // sequence of #define hipXXX(...) ...
//
// after which bare `hipMalloc(...)`, `hiprtcCreateProgram(...)` etc.
// expand to the function-pointer table dispatch; HIP / HIPRTC types
// (`hipModule_t`, `hipDeviceProp_t`, `hiprtcResult`, ...) come from the
// real headers and are unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_ROCMLIR_TUNING_DRIVER_HIPDELAYLOAD_H
#define MLIR_TOOLS_ROCMLIR_TUNING_DRIVER_HIPDELAYLOAD_H

// We need the full HIP / HIPRTC type definitions to express the
// function-pointer signatures below (`hipModule_t`, `hipDeviceProp_t`,
// `hiprtcProgram`, ...). Pull them in directly so consumers can include
// `HipDelayLoad.h` unconditionally without first including the HIP
// headers themselves. HIP headers are header-only for type purposes
// (no transitive linkage); the dlmopen-based loader in HipDelayLoad.cpp
// is what keeps libamdhip64 out of the build-time link line.
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hiprtc.h>
#endif

#include <cstddef>
#include <cstdint>

namespace rocmlir::tuningdriver {

/// Function pointers for every HIP entry point used by the tuning driver.
/// A null `getDevice` means HIP could not be resolved at all (libamdhip64
/// missing or required symbols absent); callers must check before use.
struct HipSymbols {
  void *handle = nullptr;

  hipError_t (*getDevice)(int *) = nullptr;
  hipError_t (*getDeviceProperties)(hipDeviceProp_t *, int) = nullptr;
  hipError_t (*getLastError)(void) = nullptr;
  const char *(*getErrorString)(hipError_t) = nullptr;

  hipError_t (*malloc_)(void **, size_t) = nullptr;
  hipError_t (*free_)(void *) = nullptr;
  hipError_t (*memsetAsync)(void *, int, size_t, hipStream_t) = nullptr;

  hipError_t (*streamCreate)(hipStream_t *) = nullptr;
  hipError_t (*streamDestroy)(hipStream_t) = nullptr;
  hipError_t (*streamSynchronize)(hipStream_t) = nullptr;

  hipError_t (*eventCreate)(hipEvent_t *) = nullptr;
  hipError_t (*eventDestroy)(hipEvent_t) = nullptr;
  hipError_t (*eventSynchronize)(hipEvent_t) = nullptr;
  hipError_t (*eventElapsedTime)(float *, hipEvent_t, hipEvent_t) = nullptr;

  hipError_t (*moduleLoadData)(hipModule_t *, const void *) = nullptr;
  hipError_t (*moduleUnload)(hipModule_t) = nullptr;
  hipError_t (*moduleGetFunction)(hipFunction_t *, hipModule_t,
                                  const char *) = nullptr;
  hipError_t (*moduleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned,
                                   unsigned, unsigned, unsigned, unsigned,
                                   hipStream_t, void **, void **) = nullptr;
  hipError_t (*extModuleLaunchKernel)(hipFunction_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t, uint32_t,
                                      size_t, hipStream_t, void **, void **,
                                      hipEvent_t, hipEvent_t,
                                      uint32_t) = nullptr;
};

/// Process-wide accessor for the HIP function table. Initialised on first
/// call. The implementation tries to reuse the dlmopen handle owned by
/// `RocmSystemDetect` (via `mlirRocmSystemDetectGetHipHandle` looked up
/// through `RTLD_DEFAULT`) so that we share a single HSA session per
/// process; KFD only permits one. If that symbol is absent (binary built
/// without RocmSystemDetect), falls back to its own dlmopen.
const HipSymbols &getHipSymbols();

#if defined(__HIP_PLATFORM_AMD__)
struct HiprtcSymbols {
  void *handle = nullptr;

  const char *(*getErrorString)(hiprtcResult) = nullptr;
  hiprtcResult (*createProgram)(hiprtcProgram *, const char *, const char *,
                                int, const char **, const char **) = nullptr;
  hiprtcResult (*destroyProgram)(hiprtcProgram *) = nullptr;
  hiprtcResult (*compileProgram)(hiprtcProgram, int, const char **) = nullptr;
  hiprtcResult (*getProgramLogSize)(hiprtcProgram, size_t *) = nullptr;
  hiprtcResult (*getProgramLog)(hiprtcProgram, char *) = nullptr;
  hiprtcResult (*getCodeSize)(hiprtcProgram, size_t *) = nullptr;
  hiprtcResult (*getCode)(hiprtcProgram, char *) = nullptr;
};

/// Process-wide accessor for the HIPRTC function table. HIPRTC ships in
/// its own SONAME (`libhiprtc.so.<major>`), so it is loaded via a
/// separate dlmopen / dlopen call from HIP. HIPRTC is only used by the
/// instruction-cache flush kernel JIT in CacheFlush.cpp.
const HiprtcSymbols &getHiprtcSymbols();
#endif

} // namespace rocmlir::tuningdriver

#endif // MLIR_TOOLS_ROCMLIR_TUNING_DRIVER_HIPDELAYLOAD_H
