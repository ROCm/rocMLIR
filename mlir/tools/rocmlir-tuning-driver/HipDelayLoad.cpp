//===- HipDelayLoad.cpp - Lazy HIP/HIPRTC symbol resolution ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HipDelayLoad.h"

#include "mlir/ExecutionEngine/RocmRuntimeLoader.h"

#include <cstdio>
#include <cstdlib>

namespace rocmlir::tuningdriver {

namespace {

// Fail-fast helper. The tuning driver is fundamentally useless without HIP /
// HIPRTC -- every benchmark needs to launch a kernel on an AMD GPU. If we
// returned a partially-initialised symbol table here, the macros in
// `HipDelayLoadMacros.h` would later dereference a null function pointer and
// segfault. Aborting at first detection turns the failure mode into a clear
// diagnostic instead of an undebuggable crash.
[[noreturn]] void abortMissingHip(const char *what) {
  std::fprintf(
      stderr,
      "rocmlir-tuning-driver: %s. The tuning driver requires a working "
      "ROCm install; aborting.\n",
      what);
  std::abort();
}

HipSymbols loadHipSymbols() {
  HipSymbols s;
  s.lib = mlir::rocm_loader::loadRocmLibrary(mlir::rocm_loader::Library::Hip);
  if (!s.lib.handle)
    abortMissingHip("libamdhip64 not found on the loader search path "
                    "(tried unversioned alias and `.so.<MAJOR>` for MAJOR "
                    "in [99..1])");

  auto resolve = [&](const char *name) {
    return mlir::rocm_loader::resolveRocmSymbol(s.lib, name);
  };

#define LOAD_HIP_SYM(FIELD, NAME, TYPE)                                        \
  do {                                                                         \
    s.FIELD = reinterpret_cast<TYPE>(resolve(NAME));                           \
    if (!s.FIELD)                                                              \
      abortMissingHip("missing required HIP symbol '" NAME                     \
                      "' in libamdhip64");                                     \
  } while (false)

  LOAD_HIP_SYM(getDevice, "hipGetDevice", hipError_t (*)(int *));
  // hipGetDeviceProperties was renamed to ...R0600 for ABI stability in
  // ROCm 6.0; prefer the new name and fall back to the legacy alias.
  s.getDeviceProperties =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
          resolve("hipGetDevicePropertiesR0600"));
  if (!s.getDeviceProperties) {
    s.getDeviceProperties =
        reinterpret_cast<hipError_t (*)(hipDeviceProp_t *, int)>(
            resolve("hipGetDeviceProperties"));
  }
  if (!s.getDeviceProperties)
    abortMissingHip("neither 'hipGetDevicePropertiesR0600' nor "
                    "'hipGetDeviceProperties' found in libamdhip64");
  LOAD_HIP_SYM(getLastError, "hipGetLastError", hipError_t (*)(void));
  LOAD_HIP_SYM(getErrorString, "hipGetErrorString",
               const char *(*)(hipError_t));

  LOAD_HIP_SYM(malloc_, "hipMalloc", hipError_t (*)(void **, size_t));
  LOAD_HIP_SYM(free_, "hipFree", hipError_t (*)(void *));
  LOAD_HIP_SYM(memsetAsync, "hipMemsetAsync",
               hipError_t (*)(void *, int, size_t, hipStream_t));

  LOAD_HIP_SYM(streamCreate, "hipStreamCreate", hipError_t (*)(hipStream_t *));
  LOAD_HIP_SYM(streamDestroy, "hipStreamDestroy", hipError_t (*)(hipStream_t));
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
  LOAD_HIP_SYM(moduleUnload, "hipModuleUnload", hipError_t (*)(hipModule_t));
  LOAD_HIP_SYM(moduleGetFunction, "hipModuleGetFunction",
               hipError_t (*)(hipFunction_t *, hipModule_t, const char *));
  LOAD_HIP_SYM(moduleLaunchKernel, "hipModuleLaunchKernel",
               hipError_t (*)(hipFunction_t, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, unsigned,
                              hipStream_t, void **, void **));
  LOAD_HIP_SYM(extModuleLaunchKernel, "hipExtModuleLaunchKernel",
               hipError_t (*)(hipFunction_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, size_t, hipStream_t,
                              void **, void **, hipEvent_t, hipEvent_t,
                              uint32_t));
#undef LOAD_HIP_SYM
  return s;
}

#if defined(__HIP_PLATFORM_AMD__)
[[noreturn]] void abortMissingHiprtc(const char *what) {
  std::fprintf(stderr,
               "rocmlir-tuning-driver: %s. Runtime kernel compilation needs "
               "libhiprtc; aborting.\n",
               what);
  std::abort();
}

HiprtcSymbols loadHiprtcSymbols(void *hipHandle) {
  HiprtcSymbols s;
  // HIPRTC shares HIP's KFD session (it only JIT-compiles GPU code;
  // it does not open a separate device). Load it into HIP's link-map
  // namespace so the same HSA instance satisfies both.
  s.lib = mlir::rocm_loader::loadRocmLibrary(mlir::rocm_loader::Library::Hiprtc,
                                             hipHandle);
  if (!s.lib.handle)
    abortMissingHiprtc("libhiprtc not found on the loader search path "
                       "(tried unversioned alias and `.so.<MAJOR>` for "
                       "MAJOR in [99..1])");

  auto resolve = [&](const char *name) {
    return mlir::rocm_loader::resolveRocmSymbol(s.lib, name);
  };

#define LOAD_HIPRTC_SYM(FIELD, NAME, TYPE)                                     \
  do {                                                                         \
    s.FIELD = reinterpret_cast<TYPE>(resolve(NAME));                           \
    if (!s.FIELD)                                                              \
      abortMissingHiprtc("missing required HIPRTC symbol '" NAME               \
                         "' in libhiprtc");                                    \
  } while (false)

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
  static HiprtcSymbols syms = loadHiprtcSymbols(getHipSymbols().lib.handle);
  return syms;
}
#endif

} // namespace rocmlir::tuningdriver
