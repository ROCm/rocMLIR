//===- CacheFlush.cpp - Cache flush helpers ---------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CacheFlush.h"

#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hiprtc.h>
#endif

using namespace mlir;

namespace rocmlir::tuningdriver {
namespace {

#define CHECK_HIP(expr)                                                        \
  do {                                                                         \
    hipError_t _hip_status = (expr);                                           \
    if (_hip_status != hipSuccess) {                                           \
      llvm::errs() << "HIP error in " << #expr << ": "                         \
                   << hipGetErrorString(_hip_status) << "\n";                  \
      return failure();                                                        \
    }                                                                          \
  } while(0)

#if defined(__HIP_PLATFORM_AMD__)
#define CHECK_HIPRTC(expr)                                                     \
  do {                                                                         \
    hiprtcResult _hiprtc_status = (expr);                                      \
    if (_hiprtc_status != HIPRTC_SUCCESS) {                                    \
      llvm::errs() << "hiprtc error in " << #expr << ": "                      \
                   << hiprtcGetErrorString(_hiprtc_status) << "\n";            \
      return failure();                                                        \
    }                                                                          \
  } while(0)

class HipRtcKernel {
public:
  HipRtcKernel(const char *source, const char *kernelName)
      : source(source), kernelName(kernelName) {}

  bool isBuilt() const { return function != nullptr; }

  LogicalResult build(const char *gcnArchName) {
    if (isBuilt())
      return success();

    hiprtcProgram program;
    CHECK_HIPRTC(
        hiprtcCreateProgram(&program, source, kernelName, 0, nullptr, nullptr));

    std::string archOption = std::string("--gpu-architecture=") + gcnArchName;
    const char *options[] = {archOption.c_str()};
    hiprtcResult compileStatus =
        hiprtcCompileProgram(program, /*numOptions=*/1, options);

    if (compileStatus != HIPRTC_SUCCESS) {
      size_t logSize = 0;
      hiprtcGetProgramLogSize(program, &logSize);
      std::string log;
      log.resize(logSize);
      if (logSize > 0)
        hiprtcGetProgramLog(program, log.data());
      llvm::errs() << "Failed to compile hiprtc kernel '" << kernelName
                   << "': " << hiprtcGetErrorString(compileStatus) << "\n"
                   << log << "\n";
      hiprtcDestroyProgram(&program);
      return failure();
    }

    size_t codeSize = 0;
    CHECK_HIPRTC(hiprtcGetCodeSize(program, &codeSize));

    std::vector<char> codeObject(codeSize);
    CHECK_HIPRTC(hiprtcGetCode(program, codeObject.data()));
    hiprtcDestroyProgram(&program);

    CHECK_HIP(hipModuleLoadData(&module, codeObject.data()));
    CHECK_HIP(hipModuleGetFunction(&function, module, kernelName));

    return success();
  }

  LogicalResult launch(dim3 gridDim, dim3 blockDim, hipStream_t stream,
                       void **kernelParams = nullptr, void **extra = nullptr) {
    if (!isBuilt()) {
      llvm::errs() << "hiprtc kernel '" << kernelName
                   << "' launched before build\n";
      return failure();
    }
    CHECK_HIP(hipModuleLaunchKernel(function, gridDim.x, gridDim.y, gridDim.z,
                                    blockDim.x, blockDim.y, blockDim.z, 0,
                                    stream, kernelParams, extra));
    return success();
  }

  LogicalResult cleanup() {
    if (!module)
      return success();
    CHECK_HIP(hipModuleUnload(module));
    module = nullptr;
    function = nullptr;
    return success();
  }

private:
  const char *source;
  const char *kernelName;
  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
};
#endif

class CacheFlushState {
public:
  LogicalResult flushL2Cache(hipStream_t stream) {
    if (failed(getL2CacheFlushBuffer()))
      return failure();
    CHECK_HIP(hipMemsetAsync(flushBuffer, 0, flushSize, stream));
    return success();
  }

  LogicalResult flushInstructionCache(hipStream_t stream) {
#if defined(__HIP_PLATFORM_AMD__)
    if (failed(buildFlushInstructionCacheKernel()))
      return failure();
    if (failed(icacheKernel.launch(dim3(icacheGridDim, 1, 1), dim3(64, 1, 1),
                                   stream)))
      return failure();
#else
    (void)stream;
#endif
    return success();
  }

  LogicalResult cleanup() {
    if (flushBuffer) {
      CHECK_HIP(hipFree(flushBuffer));
      flushBuffer = nullptr;
      flushSize = 0;
    }
#if defined(__HIP_PLATFORM_AMD__)
    if (failed(icacheKernel.cleanup()))
      return failure();
    icacheGridDim = 0;
#endif
    return success();
  }

private:
  LogicalResult getL2CacheFlushBuffer() {
    if (flushBuffer)
      return success();
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    size_t l2Size = props.l2CacheSize;
    flushSize = l2Size + (l2Size / 5); // 20% margin
    CHECK_HIP(hipMalloc(&flushBuffer, flushSize));
    return success();
  }

#if defined(__HIP_PLATFORM_AMD__)
  LogicalResult buildFlushInstructionCacheKernel() {
    if (icacheKernel.isBuilt())
      return success();

    hipDeviceProp_t deviceProps;
    CHECK_HIP(hipGetDeviceProperties(&deviceProps, 0));

    icacheGridDim = deviceProps.multiProcessorCount * 60;
    if (icacheGridDim <= 0)
      icacheGridDim = 1;

    return icacheKernel.build(deviceProps.gcnArchName);
  }
#endif

  size_t flushSize = 0;
  void *flushBuffer = nullptr;
#if defined(__HIP_PLATFORM_AMD__)
  static constexpr char flushInstructionCacheKernelSource[] = R"(
extern "C" __global__ void flush_icache_kernel() {
  asm volatile("s_icache_inv \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t"
               "s_nop 0 \n\t");
}
)";
  HipRtcKernel icacheKernel{flushInstructionCacheKernelSource,
                            "flush_icache_kernel"};
  int32_t icacheGridDim = 0;
#endif
};

CacheFlushState &getState() {
  static CacheFlushState state;
  return state;
}

} // namespace

LogicalResult flushL2Cache(hipStream_t stream) {
  return getState().flushL2Cache(stream);
}

LogicalResult flushInstructionCache(hipStream_t stream) {
  return getState().flushInstructionCache(stream);
}

LogicalResult cleanupCacheFlushArtifacts() { return getState().cleanup(); }

} // namespace rocmlir::tuningdriver
