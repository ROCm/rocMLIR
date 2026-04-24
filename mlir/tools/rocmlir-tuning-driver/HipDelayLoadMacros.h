//===- HipDelayLoadMacros.h - Macros redirecting hipXXX/hiprtcXXX -*- C++-*-=//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bag of preprocessor `#define`s that turn bare `hipXXX(args)` /
// `hiprtcXXX(args)` call sites into dispatches through the function-
// pointer table provided by `HipDelayLoad.h`. The macros expand to the
// canonical C-call form; HIP's own header definitions are simply
// shadowed.
//
// Include this header LAST in any TU that wants to use the redirects.
// In particular, do NOT include any HIP header *after* this file --
// doing so would either redefine the wrapped function symbols (link
// error) or, worse, silently restore the direct linkage. The expected
// pattern is:
//
//   #include <hip/hip_runtime.h>
//   #include <hip/hip_ext.h>
//   #include <hip/hiprtc.h>
//   #include "HipDelayLoad.h"
//   #include "HipDelayLoadMacros.h"
//   // ... rest of TU uses bare hipMalloc(...), hiprtcCompileProgram(...) ...
//
// This file is intentionally not include-guarded: every consumer that
// includes it gets a fresh expansion at the point of inclusion. Re-
// including it within the same TU would just redefine the same macros
// to the same expansions and is harmless (modulo a -Wmacro-redefined
// warning).
//
//===----------------------------------------------------------------------===//

// HIP runtime API
#define hipGetDevice(...)                                                      \
  (::rocmlir::tuningdriver::getHipSymbols().getDevice(__VA_ARGS__))
#define hipGetDeviceProperties(...)                                            \
  (::rocmlir::tuningdriver::getHipSymbols().getDeviceProperties(__VA_ARGS__))
#define hipGetLastError(...)                                                   \
  (::rocmlir::tuningdriver::getHipSymbols().getLastError(__VA_ARGS__))
#define hipGetErrorString(...)                                                 \
  (::rocmlir::tuningdriver::getHipSymbols().getErrorString(__VA_ARGS__))

#define hipMalloc(...)                                                         \
  (::rocmlir::tuningdriver::getHipSymbols().malloc_(__VA_ARGS__))
#define hipFree(...)                                                           \
  (::rocmlir::tuningdriver::getHipSymbols().free_(__VA_ARGS__))
#define hipMemsetAsync(...)                                                    \
  (::rocmlir::tuningdriver::getHipSymbols().memsetAsync(__VA_ARGS__))

#define hipStreamCreate(...)                                                   \
  (::rocmlir::tuningdriver::getHipSymbols().streamCreate(__VA_ARGS__))
#define hipStreamDestroy(...)                                                  \
  (::rocmlir::tuningdriver::getHipSymbols().streamDestroy(__VA_ARGS__))
#define hipStreamSynchronize(...)                                              \
  (::rocmlir::tuningdriver::getHipSymbols().streamSynchronize(__VA_ARGS__))

#define hipEventCreate(...)                                                    \
  (::rocmlir::tuningdriver::getHipSymbols().eventCreate(__VA_ARGS__))
#define hipEventDestroy(...)                                                   \
  (::rocmlir::tuningdriver::getHipSymbols().eventDestroy(__VA_ARGS__))
#define hipEventSynchronize(...)                                               \
  (::rocmlir::tuningdriver::getHipSymbols().eventSynchronize(__VA_ARGS__))
#define hipEventElapsedTime(...)                                               \
  (::rocmlir::tuningdriver::getHipSymbols().eventElapsedTime(__VA_ARGS__))

#define hipModuleLoadData(...)                                                 \
  (::rocmlir::tuningdriver::getHipSymbols().moduleLoadData(__VA_ARGS__))
#define hipModuleUnload(...)                                                   \
  (::rocmlir::tuningdriver::getHipSymbols().moduleUnload(__VA_ARGS__))
#define hipModuleGetFunction(...)                                              \
  (::rocmlir::tuningdriver::getHipSymbols().moduleGetFunction(__VA_ARGS__))
#define hipModuleLaunchKernel(...)                                             \
  (::rocmlir::tuningdriver::getHipSymbols().moduleLaunchKernel(__VA_ARGS__))
// The real `hipExtModuleLaunchKernel` takes 14 args; the last (`flags`)
// defaults to 0 in the HIP header. Default arguments do not apply when
// calling through a function pointer, so existing 13-arg call sites
// would fail to type-check. Keep the call-site syntax identical to
// upstream HIP code by injecting the trailing `0u` here.
#define hipExtModuleLaunchKernel(...)                                          \
  (::rocmlir::tuningdriver::getHipSymbols().extModuleLaunchKernel(             \
      __VA_ARGS__, 0u))

// HIPRTC API (only on AMD; HIPRTC has no NVIDIA-side stub).
#if defined(__HIP_PLATFORM_AMD__)
#define hiprtcGetErrorString(...)                                              \
  (::rocmlir::tuningdriver::getHiprtcSymbols().getErrorString(__VA_ARGS__))
#define hiprtcCreateProgram(...)                                               \
  (::rocmlir::tuningdriver::getHiprtcSymbols().createProgram(__VA_ARGS__))
#define hiprtcDestroyProgram(...)                                              \
  (::rocmlir::tuningdriver::getHiprtcSymbols().destroyProgram(__VA_ARGS__))
#define hiprtcCompileProgram(...)                                              \
  (::rocmlir::tuningdriver::getHiprtcSymbols().compileProgram(__VA_ARGS__))
#define hiprtcGetProgramLogSize(...)                                           \
  (::rocmlir::tuningdriver::getHiprtcSymbols().getProgramLogSize(__VA_ARGS__))
#define hiprtcGetProgramLog(...)                                               \
  (::rocmlir::tuningdriver::getHiprtcSymbols().getProgramLog(__VA_ARGS__))
#define hiprtcGetCodeSize(...)                                                 \
  (::rocmlir::tuningdriver::getHiprtcSymbols().getCodeSize(__VA_ARGS__))
#define hiprtcGetCode(...)                                                     \
  (::rocmlir::tuningdriver::getHiprtcSymbols().getCode(__VA_ARGS__))
#endif
