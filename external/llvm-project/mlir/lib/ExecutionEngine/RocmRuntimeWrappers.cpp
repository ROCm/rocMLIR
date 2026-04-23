//===- RocmRuntimeWrappers.cpp - MLIR ROCM runtime wrapper library --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around the ROCM library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
// Linker discipline: this file does NOT link libamdhip64 at build time.
// The HIP runtime is loaded via the shared helpers in
// `mlir/ExecutionEngine/RocmDynamicLoader.h`, which place libamdhip64
// and its transitive dependencies (most importantly libamd_comgr and
// ROCm's libLLVM) into a private link-map namespace. See that header
// for the rationale (static-initializer collision between the two
// LLVMs, etc.).
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RocmDynamicLoader.h"
#include "llvm/ADT/ArrayRef.h"

#include "hip/hip_runtime.h"

// Export tags match the upstream CudaRuntimeWrappers.cpp convention: every
// mgpu* entry point is default-visible so the JIT runner's dlsym() can
// find it. Internal helpers stay with file-default visibility (the CMake
// target applies a version script to enforce the export set).
#ifdef _WIN32
#define MLIR_HIP_WRAPPERS_EXPORT __declspec(dllexport)
#else
#define MLIR_HIP_WRAPPERS_EXPORT __attribute__((visibility("default")))
#endif

namespace {

namespace rocm_loader = ::mlir::rocm_loader;

/// Table of resolved HIP entry points. All hipXXX wrappers further down
/// call through this table instead of linking against libamdhip64 at
/// build time.
struct HipSymbols {
  rocm_loader::LoadedLibrary lib;

  const char *(*getErrorName)(hipError_t) = nullptr;
  hipError_t (*moduleLoadData)(hipModule_t *, const void *) = nullptr;
  hipError_t (*moduleUnload)(hipModule_t) = nullptr;
  hipError_t (*moduleGetFunction)(hipFunction_t *, hipModule_t,
                                  const char *) = nullptr;
  hipError_t (*moduleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned,
                                   unsigned, unsigned, unsigned, unsigned,
                                   hipStream_t, void **, void **) = nullptr;
  hipError_t (*streamCreate)(hipStream_t *) = nullptr;
  hipError_t (*streamDestroy)(hipStream_t) = nullptr;
  hipError_t (*streamSynchronize)(hipStream_t) = nullptr;
  hipError_t (*streamWaitEvent)(hipStream_t, hipEvent_t, unsigned) = nullptr;
  hipError_t (*eventCreateWithFlags)(hipEvent_t *, unsigned) = nullptr;
  hipError_t (*eventDestroy)(hipEvent_t) = nullptr;
  hipError_t (*eventSynchronize)(hipEvent_t) = nullptr;
  hipError_t (*eventRecord)(hipEvent_t, hipStream_t) = nullptr;
  hipError_t (*malloc_)(void **, size_t) = nullptr;
  hipError_t (*free_)(void *) = nullptr;
  hipError_t (*memcpyAsync)(void *, const void *, size_t, hipMemcpyKind,
                            hipStream_t) = nullptr;
  hipError_t (*memsetD32Async)(hipDeviceptr_t, int, size_t,
                               hipStream_t) = nullptr;
  hipError_t (*memsetD16Async)(hipDeviceptr_t, short, size_t,
                               hipStream_t) = nullptr;
  hipError_t (*hostRegister)(void *, size_t, unsigned) = nullptr;
  hipError_t (*hostUnregister)(void *) = nullptr;
  hipError_t (*hostGetDevicePointer)(void **, void *, unsigned) = nullptr;
  hipError_t (*setDevice)(int) = nullptr;
};

HipSymbols loadHipSymbols() {
  HipSymbols syms;
  // `CoordinationPolicy::Auto` (the default) consults
  // `mlirRocmSystemDetectGetHipHandle` first, so we share the process's
  // single HSA session when RocmSystemDetect is present; otherwise we
  // own the dlmopen ourselves.
  syms.lib = rocm_loader::loadRocmLibrary(rocm_loader::Library::Hip);
  if (!syms.lib.handle) {
    std::fprintf(
        stderr, "mlir_rocm_runtime: failed to load libamdhip64; hip calls will "
                "fail. Ensure a ROCm install with libamdhip64.so is on "
                "LD_LIBRARY_PATH / RPATH.\n");
    std::abort();
  }

#define LOAD_HIP(FIELD, NAME, TYPE)                                            \
  syms.FIELD =                                                                 \
      reinterpret_cast<TYPE>(rocm_loader::resolveRocmSymbol(syms.lib, NAME));  \
  if (!syms.FIELD) {                                                           \
    std::fprintf(stderr,                                                       \
                 "mlir_rocm_runtime: failed to resolve '%s' in "               \
                 "libamdhip64.\n",                                             \
                 NAME);                                                        \
    std::abort();                                                              \
  }

  LOAD_HIP(getErrorName, "hipGetErrorName", const char *(*)(hipError_t));
  LOAD_HIP(moduleLoadData, "hipModuleLoadData",
           hipError_t (*)(hipModule_t *, const void *));
  LOAD_HIP(moduleUnload, "hipModuleUnload", hipError_t (*)(hipModule_t));
  LOAD_HIP(moduleGetFunction, "hipModuleGetFunction",
           hipError_t (*)(hipFunction_t *, hipModule_t, const char *));
  LOAD_HIP(moduleLaunchKernel, "hipModuleLaunchKernel",
           hipError_t (*)(hipFunction_t, unsigned, unsigned, unsigned, unsigned,
                          unsigned, unsigned, unsigned, hipStream_t, void **,
                          void **));
  LOAD_HIP(streamCreate, "hipStreamCreate", hipError_t (*)(hipStream_t *));
  LOAD_HIP(streamDestroy, "hipStreamDestroy", hipError_t (*)(hipStream_t));
  LOAD_HIP(streamSynchronize, "hipStreamSynchronize",
           hipError_t (*)(hipStream_t));
  LOAD_HIP(streamWaitEvent, "hipStreamWaitEvent",
           hipError_t (*)(hipStream_t, hipEvent_t, unsigned));
  LOAD_HIP(eventCreateWithFlags, "hipEventCreateWithFlags",
           hipError_t (*)(hipEvent_t *, unsigned));
  LOAD_HIP(eventDestroy, "hipEventDestroy", hipError_t (*)(hipEvent_t));
  LOAD_HIP(eventSynchronize, "hipEventSynchronize", hipError_t (*)(hipEvent_t));
  LOAD_HIP(eventRecord, "hipEventRecord",
           hipError_t (*)(hipEvent_t, hipStream_t));
  LOAD_HIP(malloc_, "hipMalloc", hipError_t (*)(void **, size_t));
  LOAD_HIP(free_, "hipFree", hipError_t (*)(void *));
  LOAD_HIP(
      memcpyAsync, "hipMemcpyAsync",
      hipError_t (*)(void *, const void *, size_t, hipMemcpyKind, hipStream_t));
  LOAD_HIP(memsetD32Async, "hipMemsetD32Async",
           hipError_t (*)(hipDeviceptr_t, int, size_t, hipStream_t));
  LOAD_HIP(memsetD16Async, "hipMemsetD16Async",
           hipError_t (*)(hipDeviceptr_t, short, size_t, hipStream_t));
  LOAD_HIP(hostRegister, "hipHostRegister",
           hipError_t (*)(void *, size_t, unsigned));
  LOAD_HIP(hostUnregister, "hipHostUnregister", hipError_t (*)(void *));
  LOAD_HIP(hostGetDevicePointer, "hipHostGetDevicePointer",
           hipError_t (*)(void **, void *, unsigned));
  LOAD_HIP(setDevice, "hipSetDevice", hipError_t (*)(int));

#undef LOAD_HIP
  return syms;
}

const HipSymbols &getHip() {
  static HipSymbols syms = loadHipSymbols();
  return syms;
}

} // namespace

// Redirect every bare hipXXX call-site below to go through the table. This
// keeps the rest of the file almost identical to upstream for easy merge.
#define hipGetErrorName(...) (::getHip().getErrorName(__VA_ARGS__))
#define hipModuleLoadData(...) (::getHip().moduleLoadData(__VA_ARGS__))
#define hipModuleUnload(...) (::getHip().moduleUnload(__VA_ARGS__))
#define hipModuleGetFunction(...) (::getHip().moduleGetFunction(__VA_ARGS__))
#define hipModuleLaunchKernel(...) (::getHip().moduleLaunchKernel(__VA_ARGS__))
#define hipStreamCreate(...) (::getHip().streamCreate(__VA_ARGS__))
#define hipStreamDestroy(...) (::getHip().streamDestroy(__VA_ARGS__))
#define hipStreamSynchronize(...) (::getHip().streamSynchronize(__VA_ARGS__))
#define hipStreamWaitEvent(...) (::getHip().streamWaitEvent(__VA_ARGS__))
#define hipEventCreateWithFlags(...)                                           \
  (::getHip().eventCreateWithFlags(__VA_ARGS__))
#define hipEventDestroy(...) (::getHip().eventDestroy(__VA_ARGS__))
#define hipEventSynchronize(...) (::getHip().eventSynchronize(__VA_ARGS__))
#define hipEventRecord(...) (::getHip().eventRecord(__VA_ARGS__))
#define hipMalloc(...) (::getHip().malloc_(__VA_ARGS__))
#define hipFree(...) (::getHip().free_(__VA_ARGS__))
#define hipMemcpyAsync(...) (::getHip().memcpyAsync(__VA_ARGS__))
#define hipMemsetD32Async(...) (::getHip().memsetD32Async(__VA_ARGS__))
#define hipMemsetD16Async(...) (::getHip().memsetD16Async(__VA_ARGS__))
#define hipHostRegister(...) (::getHip().hostRegister(__VA_ARGS__))
#define hipHostUnregister(...) (::getHip().hostUnregister(__VA_ARGS__))
#define hipHostGetDevicePointer(...)                                           \
  (::getHip().hostGetDevicePointer(__VA_ARGS__))
#define hipSetDevice(...) (::getHip().setDevice(__VA_ARGS__))

#define HIP_REPORT_IF_ERROR(expr)                                              \
  [](hipError_t result) {                                                      \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = hipGetErrorName(result);                                \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

thread_local static int32_t defaultDevice = 0;

extern "C" MLIR_HIP_WRAPPERS_EXPORT hipModule_t
mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
  hipModule_t module = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  return module;
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT hipModule_t
mgpuModuleLoadJIT(void *data, int optLevel) {
  assert(false && "This function is not available in HIP.");
  return nullptr;
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuModuleUnload(hipModule_t module) {
  HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT hipFunction_t
mgpuModuleGetFunction(hipModule_t module, const char *name) {
  hipFunction_t function = nullptr;
  HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
  return function;
}

// The wrapper uses intptr_t instead of ROCM's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuLaunchKernel(hipFunction_t function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, hipStream_t stream,
                 void **params, void **extra, size_t /*paramsCount*/) {
  HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, smem,
                                            stream, params, extra));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT hipStream_t mgpuStreamCreate() {
  hipStream_t stream = nullptr;
  HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
  return stream;
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuStreamDestroy(hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuStreamSynchronize(hipStream_t stream) {
  return HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuStreamWaitEvent(hipStream_t stream,
                                                             hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT hipEvent_t mgpuEventCreate() {
  hipEvent_t event = nullptr;
  HIP_REPORT_IF_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  return event;
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuEventDestroy(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventDestroy(event));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuEventSynchronize(hipEvent_t event) {
  HIP_REPORT_IF_ERROR(hipEventSynchronize(event));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuEventRecord(hipEvent_t event,
                                                         hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void *mgpuMemAlloc(uint64_t sizeBytes,
                                                       hipStream_t /*stream*/,
                                                       bool /*isHostShared*/) {
  void *ptr;
  HIP_REPORT_IF_ERROR(hipMalloc(&ptr, sizeBytes));
  return ptr;
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuMemFree(void *ptr,
                                                     hipStream_t /*stream*/) {
  HIP_REPORT_IF_ERROR(hipFree(ptr));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuMemcpy(void *dst, void *src, size_t sizeBytes, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuMemset32(void *dst, int value, size_t count, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuMemset16(void *dst, int short value, size_t count, hipStream_t stream) {
  HIP_REPORT_IF_ERROR(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                        value, count, stream));
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the ROCM runtime. Helpful until we have
// transfer functions implemented.
extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  HIP_REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/0));
}

// Allows to register a MemRef with the ROCm runtime. Helpful until we have
// transfer functions implemented.
extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                          int64_t elementSizeBytes) {

  llvm::SmallVector<int64_t, 4> denseStrides(rank);
  llvm::ArrayRef<int64_t> sizes(descriptor->sizes, rank);
  llvm::ArrayRef<int64_t> strides(sizes.end(), rank);

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto sizeBytes = denseStrides.front() * elementSizeBytes;

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::ArrayRef(denseStrides));

  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

// Allows to unregister byte array with the ROCm runtime. Helpful until we have
// transfer functions implemented.
extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuMemHostUnregister(void *ptr) {
  HIP_REPORT_IF_ERROR(hipHostUnregister(ptr));
}

// Allows to unregister a MemRef with the ROCm runtime. Helpful until we have
// transfer functions implemented.
extern "C" MLIR_HIP_WRAPPERS_EXPORT void
mgpuMemHostUnregisterMemRef(int64_t rank,
                            StridedMemRefType<char, 1> *descriptor,
                            int64_t elementSizeBytes) {
  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}

template <typename T>
void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
  HIP_REPORT_IF_ERROR(hipSetDevice(0));
  HIP_REPORT_IF_ERROR(
      hipHostGetDevicePointer((void **)devicePtr, hostPtr, /*flags=*/0));
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float *allocated, float *aligned, int64_t offset,
                              int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t *allocated, int32_t *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" MLIR_HIP_WRAPPERS_EXPORT void mgpuSetDefaultDevice(int32_t device) {
  defaultDevice = device;
  HIP_REPORT_IF_ERROR(hipSetDevice(device));
}
