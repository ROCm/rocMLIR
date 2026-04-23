//===- HipDelayLoad.cpp - Lazy HIP/HIPRTC symbol resolution ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// glibc's dlmopen() is declared behind _GNU_SOURCE; define it before any
// system header is transitively included.
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hiprtc.h>
#endif

#include "HipDelayLoad.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdio>
#include <cstdlib>

namespace rocmlir::tuningdriver {

namespace {

using OsHandle = void *;

// SONAME candidates, in preference order. Mirrors the lists used by
// `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp`,
// `external/llvm-project/mlir/lib/ExecutionEngine/RocmRuntimeWrappers.cpp`
// and `RocmSystemDetect.cpp` so we always end up referring to the same
// SO on a given install.
constexpr const char *kHipCandidates[] = {
#ifdef _WIN32
    "amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll",
#else
    "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so",
#endif
};

#if defined(__HIP_PLATFORM_AMD__)
constexpr const char *kHiprtcCandidates[] = {
#ifdef _WIN32
    "hiprtc0507.dll", "hiprtc.dll",
#else
    "libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so",
#endif
};
#endif

OsHandle openSharedHip() {
#ifndef _WIN32
  // Coordinate with libMLIRRocmExecutionEngineUtils.so (RocmSystemDetect):
  // KFD permits exactly one user-space HSA session per process. If
  // RocmSystemDetect has already loaded HIP into its own dlmopen
  // namespace -- which it has, transitively, via xmir-runner / any
  // host binary that links MLIRRocmExecutionEngineUtils -- a second
  // dlmopen here would land in a fresh namespace and every call from
  // our table would return hipErrorNoDevice. Reuse the existing
  // handle when the symbol is available.
  if (void *getter = ::dlsym(RTLD_DEFAULT, "mlirRocmSystemDetectGetHipHandle")) {
    using GetHandleFn = void *(*)();
    if (void *shared = reinterpret_cast<GetHandleFn>(getter)()) {
      return shared;
    }
  }
#endif
  return nullptr;
}

OsHandle openHipRuntime() {
  if (OsHandle shared = openSharedHip())
    return shared;

  for (const char *cand : kHipCandidates) {
#ifdef _WIN32
    if (HMODULE h = ::LoadLibraryA(cand))
      return h;
#elif defined(__GLIBC__)
    if (void *h = ::dlmopen(LM_ID_NEWLM, cand, RTLD_LAZY))
      return h;
#else
    if (void *h = ::dlopen(cand, RTLD_LAZY | RTLD_LOCAL))
      return h;
#endif
  }
  return nullptr;
}

#if defined(__HIP_PLATFORM_AMD__)
OsHandle openHiprtcRuntime(OsHandle hipHandle) {
#ifndef _WIN32
  // HIPRTC ships in its own SONAME but it shares HIP's KFD session
  // (HIPRTC just JIT-compiles GPU code; it does not open a separate
  // device). To stay in the same namespace as HIP -- and therefore
  // inherit the coordinated handle from RocmSystemDetect when present
  // -- load HIPRTC into HIP's link-map namespace if we have it.
  if (hipHandle) {
#if defined(__GLIBC__)
    Lmid_t hipNs = 0;
    if (::dlinfo(hipHandle, RTLD_DI_LMID, &hipNs) == 0) {
      for (const char *cand : kHiprtcCandidates) {
        if (void *h = ::dlmopen(hipNs, cand, RTLD_LAZY))
          return h;
      }
    }
#endif
  }
  for (const char *cand : kHiprtcCandidates) {
#if defined(__GLIBC__)
    if (void *h = ::dlmopen(LM_ID_NEWLM, cand, RTLD_LAZY))
      return h;
#else
    if (void *h = ::dlopen(cand, RTLD_LAZY | RTLD_LOCAL))
      return h;
#endif
  }
  return nullptr;
#else
  for (const char *cand : kHiprtcCandidates) {
    if (HMODULE h = ::LoadLibraryA(cand))
      return h;
  }
  return nullptr;
#endif
}
#endif // __HIP_PLATFORM_AMD__

void *osSym(OsHandle h, const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(
      ::GetProcAddress(static_cast<HMODULE>(h), name));
#else
  return ::dlsym(h, name);
#endif
}

HipSymbols loadHipSymbols() {
  HipSymbols s;
  s.handle = openHipRuntime();
  if (!s.handle) {
    std::fprintf(stderr,
                 "rocmlir-tuning-driver: libamdhip64 not found on the loader "
                 "search path; HIP-dependent operations will abort.\n");
    return s;
  }

#define LOAD_HIP_SYM(FIELD, NAME, TYPE)                                        \
  s.FIELD = reinterpret_cast<TYPE>(osSym(s.handle, NAME));                     \
  if (!s.FIELD) {                                                              \
    std::fprintf(stderr,                                                       \
                 "rocmlir-tuning-driver: missing required HIP symbol '%s' "   \
                 "in libamdhip64.\n",                                          \
                 NAME);                                                        \
    s.handle = nullptr;                                                        \
    return s;                                                                  \
  }

  LOAD_HIP_SYM(getDevice, "hipGetDevice", hipError_t (*)(int *));
  // hipGetDeviceProperties was renamed to ...R0600 for ABI stability in
  // ROCm 6.0; prefer the new name and fall back to the legacy alias.
  s.getDeviceProperties =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
          osSym(s.handle, "hipGetDevicePropertiesR0600"));
  if (!s.getDeviceProperties) {
    s.getDeviceProperties =
        reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
            osSym(s.handle, "hipGetDeviceProperties"));
  }
  if (!s.getDeviceProperties) {
    std::fprintf(stderr, "rocmlir-tuning-driver: missing required HIP symbol "
                         "'hipGetDeviceProperties' in libamdhip64.\n");
    s.handle = nullptr;
    return s;
  }
  LOAD_HIP_SYM(getLastError, "hipGetLastError", hipError_t (*)(void));
  LOAD_HIP_SYM(getErrorString, "hipGetErrorString",
               const char *(*)(hipError_t));

  LOAD_HIP_SYM(malloc_, "hipMalloc", hipError_t (*)(void **, size_t));
  LOAD_HIP_SYM(free_, "hipFree", hipError_t (*)(void *));
  LOAD_HIP_SYM(memsetAsync, "hipMemsetAsync",
               hipError_t (*)(void *, int, size_t, hipStream_t));

  LOAD_HIP_SYM(streamCreate, "hipStreamCreate",
               hipError_t (*)(hipStream_t *));
  LOAD_HIP_SYM(streamDestroy, "hipStreamDestroy",
               hipError_t (*)(hipStream_t));
  LOAD_HIP_SYM(streamSynchronize, "hipStreamSynchronize",
               hipError_t (*)(hipStream_t));

  LOAD_HIP_SYM(eventCreate, "hipEventCreate", hipError_t (*)(hipEvent_t *));
  LOAD_HIP_SYM(eventDestroy, "hipEventDestroy", hipError_t (*)(hipEvent_t));
  LOAD_HIP_SYM(eventSynchronize, "hipEventSynchronize",
               hipError_t (*)(hipEvent_t));
  LOAD_HIP_SYM(eventElapsedTime, "hipEventElapsedTime",
               hipError_t (*)(float *, hipEvent_t, hipEvent_t));

  LOAD_HIP_SYM(moduleLoadData, "hipModuleLoadData",
               hipError_t (*)(hipModule_t *, const void *));
  LOAD_HIP_SYM(moduleUnload, "hipModuleUnload",
               hipError_t (*)(hipModule_t));
  LOAD_HIP_SYM(moduleGetFunction, "hipModuleGetFunction",
               hipError_t (*)(hipFunction_t *, hipModule_t, const char *));
  LOAD_HIP_SYM(moduleLaunchKernel, "hipModuleLaunchKernel",
               hipError_t (*)(hipFunction_t, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, unsigned,
                              hipStream_t, void **, void **));
  LOAD_HIP_SYM(extModuleLaunchKernel, "hipExtModuleLaunchKernel",
               hipError_t (*)(hipFunction_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, size_t,
                              hipStream_t, void **, void **, hipEvent_t,
                              hipEvent_t, uint32_t));
#undef LOAD_HIP_SYM
  return s;
}

#if defined(__HIP_PLATFORM_AMD__)
HiprtcSymbols loadHiprtcSymbols(OsHandle hipHandle) {
  HiprtcSymbols s;
  s.handle = openHiprtcRuntime(hipHandle);
  if (!s.handle) {
    std::fprintf(stderr,
                 "rocmlir-tuning-driver: libhiprtc not found on the loader "
                 "search path; runtime kernel compilation disabled.\n");
    return s;
  }

#define LOAD_HIPRTC_SYM(FIELD, NAME, TYPE)                                     \
  s.FIELD = reinterpret_cast<TYPE>(osSym(s.handle, NAME));                     \
  if (!s.FIELD) {                                                              \
    std::fprintf(stderr,                                                       \
                 "rocmlir-tuning-driver: missing required HIPRTC symbol "      \
                 "'%s' in libhiprtc.\n",                                       \
                 NAME);                                                        \
    s.handle = nullptr;                                                        \
    return s;                                                                  \
  }

  LOAD_HIPRTC_SYM(getErrorString, "hiprtcGetErrorString",
                  const char *(*)(hiprtcResult));
  LOAD_HIPRTC_SYM(createProgram, "hiprtcCreateProgram",
                  hiprtcResult (*)(hiprtcProgram *, const char *, const char *,
                                   int, const char **, const char **));
  LOAD_HIPRTC_SYM(destroyProgram, "hiprtcDestroyProgram",
                  hiprtcResult (*)(hiprtcProgram *));
  LOAD_HIPRTC_SYM(compileProgram, "hiprtcCompileProgram",
                  hiprtcResult (*)(hiprtcProgram, int, const char **));
  LOAD_HIPRTC_SYM(getProgramLogSize, "hiprtcGetProgramLogSize",
                  hiprtcResult (*)(hiprtcProgram, size_t *));
  LOAD_HIPRTC_SYM(getProgramLog, "hiprtcGetProgramLog",
                  hiprtcResult (*)(hiprtcProgram, char *));
  LOAD_HIPRTC_SYM(getCodeSize, "hiprtcGetCodeSize",
                  hiprtcResult (*)(hiprtcProgram, size_t *));
  LOAD_HIPRTC_SYM(getCode, "hiprtcGetCode",
                  hiprtcResult (*)(hiprtcProgram, char *));
#undef LOAD_HIPRTC_SYM
  return s;
}
#endif // __HIP_PLATFORM_AMD__

} // namespace

const HipSymbols &getHipSymbols() {
  static HipSymbols syms = loadHipSymbols();
  return syms;
}

#if defined(__HIP_PLATFORM_AMD__)
const HiprtcSymbols &getHiprtcSymbols() {
  static HiprtcSymbols syms = loadHiprtcSymbols(getHipSymbols().handle);
  return syms;
}
#endif

} // namespace rocmlir::tuningdriver
