//===- CacheFlush.cpp - Cache flush helpers ---------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CacheFlush.h"

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
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
  } while (0)

#if defined(__HIP_PLATFORM_AMD__)
#define CHECK_HIPRTC(expr)                                                     \
  do {                                                                         \
    hiprtcResult _hiprtc_status = (expr);                                      \
    if (_hiprtc_status != HIPRTC_SUCCESS) {                                    \
      llvm::errs() << "hiprtc error in " << #expr << ": "                      \
                   << hiprtcGetErrorString(_hiprtc_status) << "\n";            \
      return failure();                                                        \
    }                                                                          \
  } while (0)

struct HiprtcProgramDeleter {
  void operator()(hiprtcProgram program) const {
    if (!program)
      return;
    hiprtcResult status = hiprtcDestroyProgram(&program);
    if (status != HIPRTC_SUCCESS) {
      llvm::errs() << "hiprtcDestroyProgram failed: "
                   << hiprtcGetErrorString(status) << "\n";
    }
  }
};

using HiprtcProgramHandle =
    std::unique_ptr<std::remove_pointer_t<hiprtcProgram>, HiprtcProgramDeleter>;

class HipModuleHandle {
public:
  HipModuleHandle() = default;
  ~HipModuleHandle() { (void)reset(); }

  hipModule_t get() const { return module; }

  LogicalResult reset(hipModule_t newModule = nullptr) {
    if (module == newModule)
      return success();
    hipModule_t oldModule = module;
    module = nullptr;
    if (oldModule) {
      hipError_t status = hipModuleUnload(oldModule);
      if (status != hipSuccess) {
        llvm::errs() << "HIP error in hipModuleUnload: "
                     << hipGetErrorString(status) << "\n";
        return failure();
      }
    }
    module = newModule;
    return success();
  }

private:
  hipModule_t module = nullptr;
};

class HipRtcKernel {
public:
  HipRtcKernel(const char *source, const char *kernelName)
      : source(source), kernelName(kernelName) {}

  bool isBuilt() const { return function != nullptr; }

  LogicalResult build(const char *gcnArchName) {
    if (isBuilt())
      return success();

    hiprtcProgram rawProgram = nullptr;
    CHECK_HIPRTC(hiprtcCreateProgram(&rawProgram, source, kernelName, 0,
                                     nullptr, nullptr));
    HiprtcProgramHandle program(rawProgram);

    std::string archOption = std::string("--gpu-architecture=") + gcnArchName;
    const char *options[] = {archOption.c_str()};
    hiprtcResult compileStatus =
        hiprtcCompileProgram(program.get(), /*numOptions=*/1, options);

    if (compileStatus != HIPRTC_SUCCESS) {
      size_t logSize = 0;
      hiprtcGetProgramLogSize(program.get(), &logSize);
      std::string log;
      log.resize(logSize);
      if (logSize > 0)
        hiprtcGetProgramLog(program.get(), log.data());
      llvm::errs() << "Failed to compile hiprtc kernel '" << kernelName
                   << "': " << hiprtcGetErrorString(compileStatus) << "\n"
                   << log << "\n";
      return failure();
    }

    size_t codeSize = 0;
    CHECK_HIPRTC(hiprtcGetCodeSize(program.get(), &codeSize));

    std::vector<char> codeObject(codeSize);
    CHECK_HIPRTC(hiprtcGetCode(program.get(), codeObject.data()));
    hiprtcProgram programHandle = program.release();
    CHECK_HIPRTC(hiprtcDestroyProgram(&programHandle));

    hipModule_t rawModule = nullptr;
    CHECK_HIP(hipModuleLoadData(&rawModule, codeObject.data()));
    if (failed(module.reset(rawModule)))
      return failure();
    CHECK_HIP(hipModuleGetFunction(&function, module.get(), kernelName));

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
    hipFunction_t functionHandle = function;
    function = nullptr;
    if (failed(module.reset()))
      return failure();
    (void)functionHandle;
    return success();
  }

private:
  const char *source;
  const char *kernelName;
  HipModuleHandle module;
  hipFunction_t function = nullptr;
};
#endif

class CacheFlushState {
public:
  LogicalResult flushL2Cache(hipStream_t stream) {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (failed(allocL2CacheFlushBuffer()))
      return failure();
    if (skipL2Flush)
      return success();
    CHECK_HIP(hipMemsetAsync(flushBuffer, 0, flushSize, stream));
    return success();
  }

  LogicalResult flushInstructionCache(hipStream_t stream) {
    std::lock_guard<std::mutex> lock(stateMutex);
#if defined(__HIP_PLATFORM_AMD__)
    if (failed(buildFlushInstructionCacheKernel()))
      return failure();
    if (failed(icacheKernel.launch(dim3(icacheGridDim, 1, 1),
                                   dim3(icacheBlockDim, 1, 1), stream)))
      return failure();
#else
    (void)stream;
#endif
    return success();
  }

  LogicalResult cleanup() {
    std::lock_guard<std::mutex> lock(stateMutex);
    LogicalResult result = success();
    if (flushBuffer) {
      hipError_t hipStatus = hipFree(flushBuffer);
      if (hipStatus != hipSuccess) {
        llvm::errs() << "HIP error in hipFree(flushBuffer): "
                     << hipGetErrorString(hipStatus) << "\n";
        result = failure();
      }
      flushBuffer = nullptr;
      flushSize = 0;
    }
    skipL2Flush = false;
#if defined(__HIP_PLATFORM_AMD__)
    if (failed(icacheKernel.cleanup()))
      result = failure();
    icacheGridDim = 0;
    icacheBlockDim = kDefaultWaveSize;
#endif
    return result;
  }

private:
  LogicalResult allocL2CacheFlushBuffer() {
    if (flushBuffer || skipL2Flush)
      return success();
    hipDeviceProp_t props;
    if (failed(populateDeviceProperties(props)))
      return failure();
    size_t l2Size = props.l2CacheSize;
    if (l2Size == 0) {
      llvm::errs() << "Device '" << props.name
                   << "' reported zero-sized L2 cache; skipping L2 flush.\n";
      skipL2Flush = true;
      return success();
    }
    flushSize = l2Size + (l2Size / 5); // 20% margin
    CHECK_HIP(hipMalloc(&flushBuffer, flushSize));
    return success();
  }

  LogicalResult populateDeviceProperties(hipDeviceProp_t &props) {
    int deviceId = 0;
    CHECK_HIP(hipGetDevice(&deviceId));
    CHECK_HIP(hipGetDeviceProperties(&props, deviceId));
    return success();
  }

#if defined(__HIP_PLATFORM_AMD__)
  LogicalResult buildFlushInstructionCacheKernel() {
    if (icacheKernel.isBuilt())
      return success();

    hipDeviceProp_t deviceProps;
    if (failed(populateDeviceProperties(deviceProps)))
      return failure();

    auto archInfo = rock::lookupArchInfo(deviceProps.gcnArchName);
    int64_t waveSize = archInfo.waveSize;
    if (waveSize <= 0)
      waveSize = kDefaultWaveSize;
    if (waveSize > 1024)
      waveSize = 1024;
    icacheBlockDim = static_cast<unsigned>(waveSize);

    static constexpr int32_t wavesPerComputeUnit = 60;
    // Match the wave count used in CK's flush_icache implementation:
    // https://github.com/ROCm/composable_kernel/blob/b38bb492a1a55b5abb0c345962143c0f9c482cfb/include/ck/host_utility/flush_cache.hpp#L383
    icacheGridDim =
        std::max(deviceProps.multiProcessorCount, 1) * wavesPerComputeUnit;

    return icacheKernel.build(deviceProps.gcnArchName);
  }
#endif

  std::mutex stateMutex;
  size_t flushSize = 0;
  void *flushBuffer = nullptr;
  bool skipL2Flush = false;
#if defined(__HIP_PLATFORM_AMD__)
  static constexpr int32_t kDefaultWaveSize = 64;
  // https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/host/flush_icache.hpp
  static constexpr char flushInstructionCacheKernelSource[] = R"(
extern "C" __global__ void flush_icache_kernel() {
  // Issue an instruction cache invalidate and follow it with the 17 cycle delay
  // recommended by CK's flush_icache helper to guarantee that every CU observes
  // the invalidation before the benchmarking kernel launches.
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
  unsigned icacheBlockDim = kDefaultWaveSize;
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
