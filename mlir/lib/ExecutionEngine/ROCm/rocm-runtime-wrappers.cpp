//===- rocm-runtime-wrappers.cpp - MLIR ROCM runner wrapper library -------===//
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
//===----------------------------------------------------------------------===//

#include <cassert>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "hip/hip_runtime.h"
#include <unordered_map>

namespace {
int32_t reportErrorIfAny(hipError_t result, const char *where) {
  if (result != hipSuccess) {
    llvm::errs() << "HIP failed with " << result << " in " << where << "\n";
  }
  return result;
}
} // anonymous namespace

typedef union bf16_fp32_cvt {
  uint u32;
  unsigned short ushortvec[2];
  float f32;
} bf16_fp32_cvt_t;

static float bfloat16_to_float(ushort src_val) {
  bf16_fp32_cvt_t target_val;
  target_val.ushortvec[1] = src_val;
  target_val.ushortvec[0] = 0;
  return target_val.f32;
}

static unsigned short float_to_bfloat16(float src_val) {
  bf16_fp32_cvt_t target_val;
  target_val.f32 = src_val;
  return target_val.ushortvec[1];
}

// Generate tables for float-to-fp16 conversion
// ref. http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
static void generateTables(unsigned short *basetable,
                           unsigned char *shifttable) {
  unsigned int i;
  int e;
  for (i = 0; i < 256; ++i) {
    e = i - 127;
    if (e < -24) { // Very small numbers map to zero
      basetable[i | 0x000] = 0x0000;
      basetable[i | 0x100] = 0x8000;
      shifttable[i | 0x000] = 24;
      shifttable[i | 0x100] = 24;
    } else if (e < -14) { // Small numbers map to denorms
      basetable[i | 0x000] = (0x0400 >> (-e - 14));
      basetable[i | 0x100] = (0x0400 >> (-e - 14)) | 0x8000;
      shifttable[i | 0x000] = -e - 1;
      shifttable[i | 0x100] = -e - 1;
    } else if (e <= 15) { // Normal numbers just lose precision
      basetable[i | 0x000] = ((e + 15) << 10);
      basetable[i | 0x100] = ((e + 15) << 10) | 0x8000;
      shifttable[i | 0x000] = 13;
      shifttable[i | 0x100] = 13;
    } else if (e < 128) { // Large numbers map to Infinity
      basetable[i | 0x000] = 0x7C00;
      basetable[i | 0x100] = 0xFC00;
      shifttable[i | 0x000] = 24;
      shifttable[i | 0x100] = 24;
    } else { // Infinity and NaN's stay Infinity and NaN's
      basetable[i | 0x000] = 0x7C00;
      basetable[i | 0x100] = 0xFC00;
      shifttable[i | 0x000] = 13;
      shifttable[i | 0x100] = 13;
    }
  }

  return;
}

static unsigned short float_to_fp16(float src_val,
                                    unsigned short const *basetable,
                                    unsigned char const *shifttable) {
  // ref. http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
  bf16_fp32_cvt_t target_val;
  target_val.f32 = src_val;

  unsigned int b = target_val.u32;
  unsigned short h = basetable[(b >> 23) & 0x1ff] +
                     ((b & 0x007fffff) >> shifttable[(b >> 23) & 0x1ff]);
  return h;
}

extern "C" hipModule_t mgpuModuleLoad(void *data) {
  hipModule_t module = nullptr;
  (void)reportErrorIfAny(hipModuleLoadData(&module, data), "ModuleLoad");
  return module;
}

extern "C" void mgpuModuleUnload(hipModule_t module) {
  reportErrorIfAny(hipModuleUnload(module), "ModuleUnload");
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char *name) {
  hipFunction_t function = nullptr;
  (void)reportErrorIfAny(hipModuleGetFunction(&function, module, name),
                         "GetFunction");
  return function;
}

// The wrapper uses intptr_t instead of ROCM's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" int32_t mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                    intptr_t gridY, intptr_t gridZ,
                                    intptr_t blockX, intptr_t blockY,
                                    intptr_t blockZ, int32_t smem,
                                    hipStream_t stream, void **params,
                                    void **extra) {
  return reportErrorIfAny(hipModuleLaunchKernel(function, gridX, gridY, gridZ,
                                                blockX, blockY, blockZ, smem,
                                                stream, params, extra),
                          "LaunchKernel");
}

extern "C" hipStream_t mgpuGetStreamHelper() {
  hipStream_t stream;
  reportErrorIfAny(hipStreamCreate(&stream), "StreamCreate");
  return stream;
}

extern "C" hipStream_t mgpuStreamCreate() { return mgpuGetStreamHelper(); }

extern "C" void mgpuStreamDestroy(hipStream_t stream) {
  reportErrorIfAny(hipStreamDestroy(stream), "StreamDestroy");
}

extern "C" int32_t mgpuStreamSynchronize(hipStream_t stream) {
  return reportErrorIfAny(hipStreamSynchronize(stream), "StreamSync");
}

extern "C" void mgpuStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
  reportErrorIfAny(hipStreamWaitEvent(stream, event, /*flags*/ 0),
                   "StreamWaitEvent");
}

/// Helper functions for writing mlir example code

// Allows to register byte array with the ROCM runtime. Helpful until we have
// transfer functions implemented.
extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  reportErrorIfAny(hipHostRegister(ptr, sizeBytes, /*flags=*/0),
                   "MemHostRegister");
}

// Allows to register a MemRef with the ROCM runtime. Initializes array with
// value. Helpful until we have transfer functions implemented.
template <typename T>
void mgpuMemHostRegisterMemRef(T *pointer, llvm::ArrayRef<int64_t> sizes,
                               llvm::ArrayRef<int64_t> strides, T value) {
  assert(sizes.size() == strides.size());
  llvm::SmallVector<int64_t, 4> denseStrides(strides.size());

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto count = denseStrides.front();

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  std::fill_n(pointer, count, value);
  mgpuMemHostRegister(pointer, count * sizeof(T));
}

extern "C" void mgpuMemHostRegisterFloat(int64_t rank, void *ptr) {
  auto *desc = static_cast<StridedMemRefType<float, 1> *>(ptr);
  auto sizes = llvm::ArrayRef<int64_t>(desc->sizes, rank);
  auto strides = llvm::ArrayRef<int64_t>(desc->sizes + rank, rank);
  mgpuMemHostRegisterMemRef(desc->data + desc->offset, sizes, strides, 1.23f);
}

extern "C" void mgpuMemHostRegisterInt32(int64_t rank, void *ptr) {
  auto *desc = static_cast<StridedMemRefType<int32_t, 1> *>(ptr);
  auto sizes = llvm::ArrayRef<int64_t>(desc->sizes, rank);
  auto strides = llvm::ArrayRef<int64_t>(desc->sizes + rank, rank);
  mgpuMemHostRegisterMemRef(desc->data + desc->offset, sizes, strides, 123);
}

extern "C" void mgpuMemHostRegisterMemRef(int64_t rank,
                                          StridedMemRefType<char, 1> *desc,
                                          int64_t elementSizeBytes) {
  llvm::ArrayRef<int64_t> sizes(desc->sizes, rank);
  llvm::ArrayRef<int64_t> strides(sizes.end(), rank);
  llvm::SmallVector<int64_t, 4> denseStrides(rank);

  std::partial_sum(sizes.rbegin(), sizes.rend(), denseStrides.rbegin(),
                   std::multiplies<int64_t>());
  auto sizeBytes = denseStrides.front() * elementSizeBytes;

  // Only densely packed tensors are currently supported.
  std::rotate(denseStrides.begin(), denseStrides.begin() + 1,
              denseStrides.end());
  denseStrides.back() = 1;
  assert(strides == llvm::makeArrayRef(denseStrides));

  auto ptr = desc->data + desc->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

template <typename T> void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
  reportErrorIfAny(hipSetDevice(0), "hipSetDevice");
  reportErrorIfAny(
      hipHostGetDevicePointer((void **)devicePtr, hostPtr, /*flags=*/0),
      "hipHostGetDevicePointer");
}

extern "C" StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float *allocated, float *aligned, int64_t offset,
                              int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t *allocated, int32_t *aligned,
                              int64_t offset, int64_t size, int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" void mcpuMemset(float *allocated, float *aligned, int64_t offset,
                           int64_t size, int64_t stride, float value) {
  for (unsigned i = 0; i < size; ++i) {
    aligned[i] = value;
  }
}

extern "C" StridedMemRefType<float, 1>
mgpuMemAlloc(float *allocated, float *aligned, int64_t offset, int64_t size,
             int64_t stride) {
  float *gpuPtr;
  hipMalloc((void **)&gpuPtr, size * sizeof(float));
  return {gpuPtr, gpuPtr, offset, {size}, {stride}};
}

extern "C" void mgpuMemDealloc(float *allocated, float *aligned, int64_t offset,
                               int64_t size, int64_t stride) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy(float *sourceAllocated, float *sourceAligned,
                            int64_t sourceOffset, int64_t sourceSize,
                            int64_t sourceStride, float *destAllocated,
                            float *destAligned, int64_t destOffset,
                            int64_t destSize, int64_t destStride,
                            unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned, sourceSize * sizeof(float),
            static_cast<hipMemcpyKind>(copyDirection));
}

extern "C" void mcpuMem5DFloatConvertHalf(
    float *sourceAllocated, float *sourceAligned, int64_t sourceOffset,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, unsigned short *destAllocated, unsigned short *destAligned,
    int64_t destOffset, int64_t size5, int64_t size6, int64_t size7,
    int64_t size8, int64_t size9, int64_t stride5, int64_t stride6,
    int64_t stride7, int64_t stride8, int64_t stride9) {
  assert(size0 * size1 * size2 * size3 * size4 ==
         size5 * size6 * size7 * size8 * size9);

  // Generate tables for converting float to fp16
  unsigned short basetable[512];
  unsigned char shifttable[512];
  generateTables(basetable, shifttable);

  int64_t dataSize = size0 * size1 * size2 * size3 * size4;
  for (int64_t i = 0; i < dataSize; i++) {
    destAligned[i] = float_to_fp16(sourceAligned[i], basetable, shifttable);
  }
}

extern "C" void mcpuMem5DFloatConvertBF16(
    float *sourceAllocated, float *sourceAligned, int64_t sourceOffset,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, unsigned short *destAllocated, unsigned short *destAligned,
    int64_t destOffset, int64_t size5, int64_t size6, int64_t size7,
    int64_t size8, int64_t size9, int64_t stride5, int64_t stride6,
    int64_t stride7, int64_t stride8, int64_t stride9) {
  assert(size0 * size1 * size2 * size3 * size4 ==
         size5 * size6 * size7 * size8 * size9);

  int64_t dataSize = size0 * size1 * size2 * size3 * size4;
  for (int64_t i = 0; i < dataSize; i++) {
    destAligned[i] = float_to_bfloat16(sourceAligned[i]);
  }
}

extern "C" void mcpuMem5DBF16ConvertFloat(
    unsigned short *sourceAllocated, unsigned short *sourceAligned,
    int64_t sourceOffset, int64_t size0, int64_t size1, int64_t size2,
    int64_t size3, int64_t size4, int64_t stride0, int64_t stride1,
    int64_t stride2, int64_t stride3, int64_t stride4, float *destAllocated,
    float *destAligned, int64_t destOffset, int64_t size5, int64_t size6,
    int64_t size7, int64_t size8, int64_t size9, int64_t stride5,
    int64_t stride6, int64_t stride7, int64_t stride8, int64_t stride9) {
  assert(size0 * size1 * size2 * size3 * size4 ==
         size5 * size6 * size7 * size8 * size9);
  int64_t dataSize = size0 * size1 * size2 * size3 * size4;
  for (int64_t i = 0; i < dataSize; i++) {
    destAligned[i] = bfloat16_to_float(sourceAligned[i]);
  }
}

extern "C" void mcpuPrintBF16(unsigned short *allocated,
                              unsigned short *aligned, int64_t offset,
                              int64_t size0, int64_t size1, int64_t size2,
                              int64_t size3, int64_t stride0, int64_t stride1,
                              int64_t stride2, int64_t stride3) {
  int64_t dataSize = size0 * size1 * size2 * size3;
  for (int64_t i = 0; i < dataSize; i++) {
    float fvalue = bfloat16_to_float(aligned[i]);
    printf("%f\t", fvalue);
  }
}
// 2D float memref utility routines.

extern "C" void mcpuMemset2DFloat(float *allocated, float *aligned,
                                  int64_t offset, int64_t size0, int64_t size1,
                                  int64_t stride0, int64_t stride1,
                                  float value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      aligned[i * stride0 + j * stride1] = value;
}

extern "C" StridedMemRefType<float, 2>
mgpuMemAlloc2DFloat(float *allocated, float *aligned, int64_t offset,
                    int64_t size0, int64_t size1, int64_t stride0,
                    int64_t stride1) {
  float *gpuPtr;
  hipMalloc((void **)&gpuPtr, size0 * size1 * sizeof(float));
  return {gpuPtr, gpuPtr, offset, {size0, size1}, {stride0, stride1}};
}

extern "C" void mgpuMemDealloc2DFloat(float *allocated, float *aligned,
                                      int64_t offset, int64_t size0,
                                      int64_t size1, int64_t stride0,
                                      int64_t stride1) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy2DFloat(float *sourceAllocated, float *sourceAligned,
                                   int64_t sourceOffset, int64_t sourceSize0,
                                   int64_t sourceSize1, int64_t sourceStride0,
                                   int64_t sourceStride1, float *destAllocated,
                                   float *destAligned, int64_t destOffset,
                                   int64_t destSize0, int64_t destSize1,
                                   int64_t destStride0, int64_t destStride1,
                                   unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned,
            sourceSize0 * sourceSize1 * sizeof(float),
            static_cast<hipMemcpyKind>(copyDirection));
}

short randomIntegerValue(short min, short max) {
  if (min == max)
    return min;
  return (std::rand() % (max - min)) + min;
}

float randomFloatValue(short min, short max) {
  if (min == max)
    return (float)min;
  return (max - min) * (float)(std::rand()) / RAND_MAX + (float)min;
}

// 3D float memref utility routines.

extern "C" void mcpuMemset3DFloat(float *allocated, float *aligned,
                                  int64_t offset, int64_t size0, int64_t size1,
                                  int64_t size2, int64_t stride0,
                                  int64_t stride1, int64_t stride2,
                                  float value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        aligned[i * stride0 + j * stride1 + k * stride2] = value;
}

extern "C" StridedMemRefType<float, 3>
mgpuMemAlloc3DFloat(float *allocated, float *aligned, int64_t offset,
                    int64_t size0, int64_t size1, int64_t size2,
                    int64_t stride0, int64_t stride1, int64_t stride2) {
  float *gpuPtr;
  hipMalloc((void **)&gpuPtr, size0 * size1 * size2 * sizeof(float));
  return {gpuPtr,
          gpuPtr,
          offset,
          {size0, size1, size2},
          {stride0, stride1, stride2}};
}

extern "C" void mgpuMemDealloc3DFloat(float *allocated, float *aligned,
                                      int64_t offset, int64_t size0,
                                      int64_t size1, int64_t size2,
                                      int64_t stride0, int64_t stride1,
                                      int64_t stride2) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy3DFloat(float *sourceAllocated, float *sourceAligned,
                                   int64_t sourceOffset, int64_t sourceSize0,
                                   int64_t sourceSize1, int64_t sourceSize2,
                                   int64_t sourceStride0, int64_t sourceStride1,
                                   int64_t sourceStride2, float *destAllocated,
                                   float *destAligned, int64_t destOffset,
                                   int64_t destSize0, int64_t destSize1,
                                   int64_t destSize2, int64_t destStride0,
                                   int64_t destStride1, int64_t destStride2,
                                   unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned,
            sourceSize0 * sourceSize1 * sourceSize2 * sizeof(float),
            static_cast<hipMemcpyKind>(copyDirection));
}

// 4D float memref utility routines.

extern "C" void mcpuMemset4DFloat(float *allocated, float *aligned,
                                  int64_t offset, int64_t size0, int64_t size1,
                                  int64_t size2, int64_t size3, int64_t stride0,
                                  int64_t stride1, int64_t stride2,
                                  int64_t stride3, float value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l) {
          aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3] =
              value;
        }
}

extern "C" void mcpuMemset5DFloat(float *allocated, float *aligned,
                                  int64_t offset, int64_t size0, int64_t size1,
                                  int64_t size2, int64_t size3, int64_t size4,
                                  int64_t stride0, int64_t stride1,
                                  int64_t stride2, int64_t stride3,
                                  int64_t stride4, float value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m)
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
}

extern "C" StridedMemRefType<float, 5>
mgpuMemAlloc5DFloat(float *allocated, float *aligned, int64_t offset,
                    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                    int64_t size4, int64_t stride0, int64_t stride1,
                    int64_t stride2, int64_t stride3, int64_t stride4) {
  float *gpuPtr;
  hipMalloc((void **)&gpuPtr,
            size0 * size1 * size2 * size3 * size4 * sizeof(float));
  return {gpuPtr,
          gpuPtr,
          offset,
          {size0, size1, size2, size3, size4},
          {stride0, stride1, stride2, stride3, stride4}};
}

extern "C" void
mcpuMemset5DFloatRandInt(float *allocated, float *aligned, int64_t offset,
                         int64_t size0, int64_t size1, int64_t size2,
                         int64_t size3, int64_t size4, int64_t stride0,
                         int64_t stride1, int64_t stride2, int64_t stride3,
                         int64_t stride4, short min, short max, int64_t seed) {
  if (seed < 0)
    std::srand(time(0));
  else
    std::srand(seed);

  float value;
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m) {
            value = (float)randomIntegerValue(min, max);
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
          }
}

extern "C" void mcpuMemset5DFloatRandFloat(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t size2, int64_t size3, int64_t size4, int64_t stride0,
    int64_t stride1, int64_t stride2, int64_t stride3, int64_t stride4,
    short min, short max, int64_t seed) {
  if (seed < 0)
    std::srand(time(0));
  else
    std::srand(seed);

  float value;
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m) {
            value = randomFloatValue(min, max);
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
          }
}

extern "C" StridedMemRefType<float, 4>
mgpuMemAlloc4DFloat(float *allocated, float *aligned, int64_t offset,
                    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                    int64_t stride0, int64_t stride1, int64_t stride2,
                    int64_t stride3) {
  float *gpuPtr;
  hipMalloc((void **)&gpuPtr, size0 * size1 * size2 * size3 * sizeof(float));
  return {gpuPtr,
          gpuPtr,
          offset,
          {size0, size1, size2, size3},
          {stride0, stride1, stride2, stride3}};
}

extern "C" void mgpuMemDealloc5DFloat(
    float *allocated, float *aligned, int64_t offset, int64_t size0,
    int64_t size1, int64_t size2, int64_t size3, int64_t size4, int64_t stride0,
    int64_t stride1, int64_t stride2, int64_t stride3, int64_t stride4) {
  hipFree(aligned);
}

extern "C" void mgpuMemDealloc4DFloat(float *allocated, float *aligned,
                                      int64_t offset, int64_t size0,
                                      int64_t size1, int64_t size2,
                                      int64_t size3, int64_t stride0,
                                      int64_t stride1, int64_t stride2,
                                      int64_t stride3) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy4DFloat(
    float *sourceAllocated, float *sourceAligned, int64_t sourceOffset,
    int64_t sourceSize0, int64_t sourceSize1, int64_t sourceSize2,
    int64_t sourceSize3, int64_t sourceStride0, int64_t sourceStride1,
    int64_t sourceStride2, int64_t sourceStride3, float *destAllocated,
    float *destAligned, int64_t destOffset, int64_t destSize0,
    int64_t destSize1, int64_t destSize2, int64_t destSize3,
    int64_t destStride0, int64_t destStride1, int64_t destStride2,
    int64_t destStride3, unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned,
            sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 *
                sizeof(float),
            static_cast<hipMemcpyKind>(copyDirection));
}

// Copy Float to Float
extern "C" void mcpuMemCopy5DFloat(
    float *sourceAllocated, float *sourceAligned, int64_t sourceOffset,
    int64_t sourceSize0, int64_t sourceSize1, int64_t sourceSize2,
    int64_t sourceSize3, int64_t sourceSize4, int64_t sourceStride0,
    int64_t sourceStride1, int64_t sourceStride2, int64_t sourceStride3,
    int64_t sourceStride4, float *destAllocated, float *destAligned,
    int64_t destOffset, int64_t destSize0, int64_t destSize1, int64_t destSize2,
    int64_t destSize3, int64_t destSize4, int64_t destStride0,
    int64_t destStride1, int64_t destStride2, int64_t destStride3,
    int64_t destStride4) {

  assert(sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 * sourceSize4 ==
         destSize0 * destSize1 * destSize2 * destSize3 * destSize4);
  int64_t dataSize =
      sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 * sourceSize4;
  for (int64_t i = 0; i < dataSize; i++)
    destAligned[i] = sourceAligned[i];
}

extern "C" void mgpuMemCopy5DFloat(
    float *sourceAllocated, float *sourceAligned, int64_t sourceOffset,
    int64_t sourceSize0, int64_t sourceSize1, int64_t sourceSize2,
    int64_t sourceSize3, int64_t sourceSize4, int64_t sourceStride0,
    int64_t sourceStride1, int64_t sourceStride2, int64_t sourceStride3,
    int64_t sourceStride4, float *destAllocated, float *destAligned,
    int64_t destOffset, int64_t destSize0, int64_t destSize1, int64_t destSize2,
    int64_t destSize3, int64_t destSize4, int64_t destStride0,
    int64_t destStride1, int64_t destStride2, int64_t destStride3,
    int64_t destStride4, unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned,
            sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 *
                sourceSize4 * sizeof(float),
            static_cast<hipMemcpyKind>(copyDirection));
}

// 4D half memref utility routines.

extern "C" void mcpuMemset4DHalf(unsigned short *allocated,
                                 unsigned short *aligned, int64_t offset,
                                 int64_t size0, int64_t size1, int64_t size2,
                                 int64_t size3, int64_t stride0,
                                 int64_t stride1, int64_t stride2,
                                 int64_t stride3, unsigned short value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3] =
              value;
}

extern "C" void mcpuMemset5DHalfRandInt(
    unsigned short *allocated, unsigned short *aligned, int64_t offset,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, short min, short max, int64_t seed) {

  // Generate tables for converting float to fp16
  unsigned short basetable[512];
  unsigned char shifttable[512];
  generateTables(basetable, shifttable);

  if (seed < 0)
    std::srand(time(0));
  else
    std::srand(seed);

  float value;
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m) {
            value = randomIntegerValue(min, max);
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = float_to_fp16(value, basetable, shifttable);
          }
}

extern "C" void mcpuMemset5DHalfRandFloat(
    unsigned short *allocated, unsigned short *aligned, int64_t offset,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, short min, short max, int64_t seed) {

  // Generate tables for converting float to fp16
  unsigned short basetable[512];
  unsigned char shifttable[512];
  generateTables(basetable, shifttable);

  if (seed < 0)
    std::srand(time(0));
  else
    std::srand(seed);

  float value;
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m) {
            value = randomFloatValue(min, max);
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = float_to_fp16(value, basetable, shifttable);
          }
}

extern "C" void mcpuMemset5DHalf(unsigned short *allocated,
                                 unsigned short *aligned, int64_t offset,
                                 int64_t size0, int64_t size1, int64_t size2,
                                 int64_t size3, int64_t size4, int64_t stride0,
                                 int64_t stride1, int64_t stride2,
                                 int64_t stride3, int64_t stride4,
                                 unsigned short value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m)
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
}

extern "C" StridedMemRefType<unsigned short, 4>
mgpuMemAlloc4DHalf(unsigned short *allocated, unsigned short *aligned,
                   int64_t offset, int64_t size0, int64_t size1, int64_t size2,
                   int64_t size3, int64_t stride0, int64_t stride1,
                   int64_t stride2, int64_t stride3) {
  unsigned short *gpuPtr;
  hipMalloc((void **)&gpuPtr,
            size0 * size1 * size2 * size3 * sizeof(unsigned short));
  return {gpuPtr,
          gpuPtr,
          offset,
          {size0, size1, size2, size3},
          {stride0, stride1, stride2, stride3}};
}

extern "C" StridedMemRefType<unsigned short, 5>
mgpuMemAlloc5DHalf(unsigned short *allocated, unsigned short *aligned,
                   int64_t offset, int64_t size0, int64_t size1, int64_t size2,
                   int64_t size3, int64_t size4, int64_t stride0,
                   int64_t stride1, int64_t stride2, int64_t stride3,
                   int64_t stride4) {
  unsigned short *gpuPtr;
  hipMalloc((void **)&gpuPtr,
            size0 * size1 * size2 * size3 * size4 * sizeof(unsigned short));
  return {gpuPtr,
          gpuPtr,
          offset,
          {size0, size1, size2, size3, size4},
          {stride0, stride1, stride2, stride3, stride4}};
}

extern "C" void mgpuMemDealloc4DHalf(unsigned short *allocated,
                                     unsigned short *aligned, int64_t offset,
                                     int64_t size0, int64_t size1,
                                     int64_t size2, int64_t size3,
                                     int64_t stride0, int64_t stride1,
                                     int64_t stride2, int64_t stride3) {
  hipFree(aligned);
}

extern "C" void mgpuMemDealloc5DHalf(unsigned short *allocated,
                                     unsigned short *aligned, int64_t offset,
                                     int64_t size0, int64_t size1,
                                     int64_t size2, int64_t size3,
                                     int64_t size4, int64_t stride0,
                                     int64_t stride1, int64_t stride2,
                                     int64_t stride3, int64_t stride4) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy4DHalf(
    unsigned short *sourceAllocated, unsigned short *sourceAligned,
    int64_t sourceOffset, int64_t sourceSize0, int64_t sourceSize1,
    int64_t sourceSize2, int64_t sourceSize3, int64_t sourceStride0,
    int64_t sourceStride1, int64_t sourceStride2, int64_t sourceStride3,
    unsigned short *destAllocated, unsigned short *destAligned,
    int64_t destOffset, int64_t destSize0, int64_t destSize1, int64_t destSize2,
    int64_t destSize3, int64_t destStride0, int64_t destStride1,
    int64_t destStride2, int64_t destStride3, unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned,
            sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 *
                sizeof(unsigned short),
            static_cast<hipMemcpyKind>(copyDirection));
}

extern "C" void mgpuMemCopy5DHalf(
    unsigned short *sourceAllocated, unsigned short *sourceAligned,
    int64_t sourceOffset, int64_t sourceSize0, int64_t sourceSize1,
    int64_t sourceSize2, int64_t sourceSize3, int64_t sourceSize4,
    int64_t sourceStride0, int64_t sourceStride1, int64_t sourceStride2,
    int64_t sourceStride3, int64_t sourceStride4, unsigned short *destAllocated,
    unsigned short *destAligned, int64_t destOffset, int64_t destSize0,
    int64_t destSize1, int64_t destSize2, int64_t destSize3, int64_t destSize4,
    int64_t destStride0, int64_t destStride1, int64_t destStride2,
    int64_t destStride3, int64_t destStride4, unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned,
            sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 *
                sourceSize4 * sizeof(unsigned short),
            static_cast<hipMemcpyKind>(copyDirection));
}

// 4D bf16 memref utility routines.

extern "C" void mcpuMemset4DBF16(unsigned short *allocated,
                                 unsigned short *aligned, int64_t offset,
                                 int64_t size0, int64_t size1, int64_t size2,
                                 int64_t size3, int64_t stride0,
                                 int64_t stride1, int64_t stride2,
                                 int64_t stride3, unsigned short value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3] =
              value;
}

extern "C" void mcpuMemset5DBF16RandInt(
    unsigned short *allocated, unsigned short *aligned, int64_t offset,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, short min, short max, int64_t seed) {

  if (seed < 0)
    std::srand(time(0));
  else
    std::srand(seed);

  unsigned short value;
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m) {
            value = float_to_bfloat16((float)randomIntegerValue(min, max));
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
          }
}

extern "C" void mcpuMemset5DBF16RandFloat(
    unsigned short *allocated, unsigned short *aligned, int64_t offset,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, short min, short max, int64_t seed) {

  if (seed < 0)
    std::srand(time(0));
  else
    std::srand(seed);

  unsigned short value;
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m) {
            value = float_to_bfloat16(randomFloatValue(min, max));
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
          }
}

extern "C" void mcpuMemset5DBF16(unsigned short *allocated,
                                 unsigned short *aligned, int64_t offset,
                                 int64_t size0, int64_t size1, int64_t size2,
                                 int64_t size3, int64_t size4, int64_t stride0,
                                 int64_t stride1, int64_t stride2,
                                 int64_t stride3, int64_t stride4,
                                 unsigned short value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m)
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
}

extern "C" StridedMemRefType<unsigned short, 4>
mgpuMemAlloc4DBF16(unsigned short *allocated, unsigned short *aligned,
                   int64_t offset, int64_t size0, int64_t size1, int64_t size2,
                   int64_t size3, int64_t stride0, int64_t stride1,
                   int64_t stride2, int64_t stride3) {
  unsigned short *gpuPtr;
  hipMalloc((void **)&gpuPtr,
            size0 * size1 * size2 * size3 * sizeof(unsigned short));
  return {gpuPtr,
          gpuPtr,
          offset,
          {size0, size1, size2, size3},
          {stride0, stride1, stride2, stride3}};
}

extern "C" StridedMemRefType<unsigned short, 5>
mgpuMemAlloc5DBF16(unsigned short *allocated, unsigned short *aligned,
                   int64_t offset, int64_t size0, int64_t size1, int64_t size2,
                   int64_t size3, int64_t size4, int64_t stride0,
                   int64_t stride1, int64_t stride2, int64_t stride3,
                   int64_t stride4) {
  unsigned short *gpuPtr;
  hipMalloc((void **)&gpuPtr,
            size0 * size1 * size2 * size3 * size4 * sizeof(unsigned short));
  return {gpuPtr,
          gpuPtr,
          offset,
          {size0, size1, size2, size3, size4},
          {stride0, stride1, stride2, stride3, stride4}};
}

extern "C" void mgpuMemDealloc4DBF16(unsigned short *allocated,
                                     unsigned short *aligned, int64_t offset,
                                     int64_t size0, int64_t size1,
                                     int64_t size2, int64_t size3,
                                     int64_t stride0, int64_t stride1,
                                     int64_t stride2, int64_t stride3) {
  hipFree(aligned);
}

extern "C" void mgpuMemDealloc5DBF16(unsigned short *allocated,
                                     unsigned short *aligned, int64_t offset,
                                     int64_t size0, int64_t size1,
                                     int64_t size2, int64_t size3,
                                     int64_t size4, int64_t stride0,
                                     int64_t stride1, int64_t stride2,
                                     int64_t stride3, int64_t stride4) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy4DBF16(
    unsigned short *sourceAllocated, unsigned short *sourceAligned,
    int64_t sourceOffset, int64_t sourceSize0, int64_t sourceSize1,
    int64_t sourceSize2, int64_t sourceSize3, int64_t sourceStride0,
    int64_t sourceStride1, int64_t sourceStride2, int64_t sourceStride3,
    unsigned short *destAllocated, unsigned short *destAligned,
    int64_t destOffset, int64_t destSize0, int64_t destSize1, int64_t destSize2,
    int64_t destSize3, int64_t destStride0, int64_t destStride1,
    int64_t destStride2, int64_t destStride3, unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned,
            sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 *
                sizeof(unsigned short),
            static_cast<hipMemcpyKind>(copyDirection));
}

extern "C" void mgpuMemCopy5DBF16(
    unsigned short *sourceAllocated, unsigned short *sourceAligned,
    int64_t sourceOffset, int64_t sourceSize0, int64_t sourceSize1,
    int64_t sourceSize2, int64_t sourceSize3, int64_t sourceSize4,
    int64_t sourceStride0, int64_t sourceStride1, int64_t sourceStride2,
    int64_t sourceStride3, int64_t sourceStride4, unsigned short *destAllocated,
    unsigned short *destAligned, int64_t destOffset, int64_t destSize0,
    int64_t destSize1, int64_t destSize2, int64_t destSize3, int64_t destSize4,
    int64_t destStride0, int64_t destStride1, int64_t destStride2,
    int64_t destStride3, int64_t destStride4, unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned,
            sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 *
                sourceSize4 * sizeof(unsigned short),
            static_cast<hipMemcpyKind>(copyDirection));
}

// Extract proper tensor sizes and strides based on layouts
static void
getSizesAndStrides(int64_t rank1, StridedMemRefType<float, 5> *filter,
                   int64_t rank2, StridedMemRefType<float, 5> *input,
                   int64_t rank3, StridedMemRefType<float, 5> *output,
                   void *f_layout, void *i_layout, void *o_layout,
                   llvm::SmallVector<int64_t, 5> &fSizes,
                   llvm::SmallVector<int64_t, 5> &fStrides,
                   llvm::SmallVector<int64_t, 5> &iSizes,
                   llvm::SmallVector<int64_t, 5> &iStrides,
                   llvm::SmallVector<int64_t, 5> &oSizes,
                   llvm::SmallVector<int64_t, 5> &oStrides) {
  auto filterSizes = llvm::ArrayRef<int64_t>(filter->sizes, rank1);
  auto filterStrides = llvm::ArrayRef<int64_t>(filter->strides, rank1);

  auto inputSizes = llvm::ArrayRef<int64_t>(input->sizes, rank2);
  auto inputStrides = llvm::ArrayRef<int64_t>(input->strides, rank2);

  auto outputSizes = llvm::ArrayRef<int64_t>(output->sizes, rank3);
  auto outputStrides = llvm::ArrayRef<int64_t>(output->strides, rank3);

  auto *layout1 = static_cast<StridedMemRefType<char, 1> *>(f_layout);
  auto *filterLayout = layout1->data + layout1->offset;

  auto *layout2 = static_cast<StridedMemRefType<char, 1> *>(i_layout);
  auto *inputLayout = layout2->data + layout2->offset;

  auto *layout3 = static_cast<StridedMemRefType<char, 1> *>(o_layout);
  auto *outputLayout = layout3->data + layout3->offset;

  // Extract tensor sizes and strides into a map
  std::unordered_map<char, std::pair<int64_t, int64_t>> filterSizeStride,
      inputSizeStride, outputSizeStride;
  for (size_t i = 0; i < 5; i++) {
    filterSizeStride[filterLayout[i]] =
        std::make_pair(filterSizes[i], filterStrides[i]);
    inputSizeStride[inputLayout[i]] =
        std::make_pair(inputSizes[i], inputStrides[i]);
    outputSizeStride[outputLayout[i]] =
        std::make_pair(outputSizes[i], outputStrides[i]);
  }

  // Move sizes and strides into vectors to help compiler optimization
  // filter: g k c y x
  fSizes = {filterSizeStride['g'].first, filterSizeStride['k'].first,
            filterSizeStride['c'].first, filterSizeStride['y'].first,
            filterSizeStride['x'].first};
  fStrides = {filterSizeStride['g'].second, filterSizeStride['k'].second,
              filterSizeStride['c'].second, filterSizeStride['y'].second,
              filterSizeStride['x'].second};
  // input: g n c h w
  iSizes = {inputSizeStride['g'].first, inputSizeStride['n'].first,
            inputSizeStride['c'].first, inputSizeStride['h'].first,
            inputSizeStride['w'].first};
  iStrides = {inputSizeStride['g'].second, inputSizeStride['n'].second,
              inputSizeStride['c'].second, inputSizeStride['h'].second,
              inputSizeStride['w'].second};
  // output: g n k h w
  oSizes = {outputSizeStride['g'].first, outputSizeStride['n'].first,
            outputSizeStride['k'].first, outputSizeStride['h'].first,
            outputSizeStride['w'].first};
  oStrides = {outputSizeStride['g'].second, outputSizeStride['n'].second,
              outputSizeStride['k'].second, outputSizeStride['h'].second,
              outputSizeStride['w'].second};
  return;
}

// A generic forward convolution function that supports random layouts,
// dimensions, strides, paddings, and dilations.
extern "C" void mcpuConv2d(int64_t rank1, void *f_ptr, int64_t rank2,
                           void *i_ptr, int64_t rank3, void *o_ptr,
                           int64_t rank4, void *f_layout, int64_t rank5,
                           void *i_layout, int64_t rank6, void *o_layout,
                           int32_t stride_h, int32_t stride_w,
                           int32_t padding_h_l, int32_t padding_h_r,
                           int32_t padding_w_l, int32_t padding_w_r,
                           int32_t dilation_h, int32_t dilation_w) {
  auto *filter = static_cast<StridedMemRefType<float, 5> *>(f_ptr);
  auto *filterAllocated = filter->data + filter->offset;

  auto *input = static_cast<StridedMemRefType<float, 5> *>(i_ptr);
  auto *inputAllocated = input->data + input->offset;

  auto *output = static_cast<StridedMemRefType<float, 5> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;

  // Extract proper tensor sizes and strides based on layouts
  llvm::SmallVector<int64_t, 5> filterSizes(5), filterStrides(5);
  llvm::SmallVector<int64_t, 5> inputSizes(5), inputStrides(5);
  llvm::SmallVector<int64_t, 5> outputSizes(5), outputStrides(5);

  getSizesAndStrides(rank1, filter, rank2, input, rank3, output, f_layout,
                     i_layout, o_layout, filterSizes, filterStrides, inputSizes,
                     inputStrides, outputSizes, outputStrides);
  // Perform forward convolution
  for (int64_t g = 0; g < outputSizes[0]; g++)
    for (int64_t n = 0; n < outputSizes[1]; n++)
      for (int64_t k = 0; k < outputSizes[2]; k++)
        for (int64_t out_h = 0; out_h < outputSizes[3]; out_h++)
          for (int64_t out_w = 0; out_w < outputSizes[4]; out_w++) {

            float acc = 0.0;
            for (int64_t c = 0; c < inputSizes[2]; c++)
              for (int64_t fil_h = 0; fil_h < filterSizes[3]; fil_h++)
                for (int64_t fil_w = 0; fil_w < filterSizes[4]; fil_w++) {

                  float input;
                  int64_t in_h =
                      out_h * stride_h + fil_h * dilation_h - padding_h_l;
                  int64_t in_w =
                      out_w * stride_w + fil_w * dilation_w - padding_w_l;

                  if (in_h < 0 || in_h >= inputSizes[3] || in_w < 0 ||
                      in_w >= inputSizes[4])
                    input = 0.0;
                  else

                    input = inputAllocated[g * inputStrides[0] +
                                           n * inputStrides[1] +
                                           c * inputStrides[2] +
                                           in_h * inputStrides[3] +
                                           in_w * inputStrides[4]];

                  acc += input * filterAllocated[g * filterStrides[0] +
                                                 k * filterStrides[1] +
                                                 c * filterStrides[2] +
                                                 fil_h * filterStrides[3] +
                                                 fil_w * filterStrides[4]];
                }

            outputAllocated[g * outputStrides[0] + n * outputStrides[1] +
                            k * outputStrides[2] + out_h * outputStrides[3] +
                            out_w * outputStrides[4]] = acc;
          }
}

// A generic backward-weight convolution function that supports random layouts,
// dimensions, strides, paddings, and dilations.
extern "C" void mcpuConv2dBwdWeight(
    int64_t rank1, void *f_ptr, int64_t rank2, void *i_ptr, int64_t rank3,
    void *o_ptr, int64_t rank4, void *f_layout, int64_t rank5, void *i_layout,
    int64_t rank6, void *o_layout, int32_t stride_h, int32_t stride_w,
    int32_t padding_h_l, int32_t padding_h_r, int32_t padding_w_l,
    int32_t padding_w_r, int32_t dilation_h, int32_t dilation_w) {

  auto *filter = static_cast<StridedMemRefType<float, 5> *>(f_ptr);
  auto *filterAllocated = filter->data + filter->offset;

  auto *input = static_cast<StridedMemRefType<float, 5> *>(i_ptr);
  auto *inputAllocated = input->data + input->offset;

  auto *output = static_cast<StridedMemRefType<float, 5> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;

  // Extract proper tensor sizes and strides based on layouts
  llvm::SmallVector<int64_t, 5> filterSizes, filterStrides;
  llvm::SmallVector<int64_t, 5> inputSizes, inputStrides;
  llvm::SmallVector<int64_t, 5> outputSizes, outputStrides;
  getSizesAndStrides(rank1, filter, rank2, input, rank3, output, f_layout,
                     i_layout, o_layout, filterSizes, filterStrides, inputSizes,
                     inputStrides, outputSizes, outputStrides);

  // Perform bwd_weight convolution
  for (int64_t g = 0; g < outputSizes[0]; g++)
    for (int64_t k = 0; k < filterSizes[1]; k++)
      for (int64_t c = 0; c < filterSizes[2]; c++)
        for (int64_t y = 0; y < filterSizes[3]; y++)
          for (int64_t x = 0; x < filterSizes[4]; x++) {

            float acc = 0.0;
            for (int64_t n = 0; n < outputSizes[1]; n++)
              for (int64_t out_h = 0; out_h < outputSizes[3]; out_h++)
                for (int64_t out_w = 0; out_w < outputSizes[4]; out_w++) {
                  int64_t in_h =
                      out_h * stride_h + y * dilation_h - padding_h_l;
                  int64_t in_w =
                      out_w * stride_w + x * dilation_w - padding_w_l;
                  if (in_h >= 0 && in_h < inputSizes[3] && in_w >= 0 &&
                      in_w < inputSizes[4])
                    acc += inputAllocated[g * inputStrides[0] +
                                          n * inputStrides[1] +
                                          c * inputStrides[2] +
                                          in_h * inputStrides[3] +
                                          in_w * inputStrides[4]] *
                           outputAllocated[g * outputStrides[0] +
                                           n * outputStrides[1] +
                                           k * outputStrides[2] +
                                           out_h * outputStrides[3] +
                                           out_w * outputStrides[4]];
                }
            filterAllocated[g * filterStrides[0] + k * filterStrides[1] +
                            c * filterStrides[2] + y * filterStrides[3] +
                            x * filterStrides[4]] = acc;
          }
}

// A generic backward-data convolution function that supports random layouts,
// dimensions, strides, paddings, and dilations.
extern "C" void mcpuConv2dBwdData(int64_t rank1, void *f_ptr, int64_t rank2,
                                  void *i_ptr, int64_t rank3, void *o_ptr,
                                  int64_t rank4, void *f_layout, int64_t rank5,
                                  void *i_layout, int64_t rank6, void *o_layout,
                                  int32_t stride_h, int32_t stride_w,
                                  int32_t padding_h_l, int32_t padding_h_r,
                                  int32_t padding_w_l, int32_t padding_w_r,
                                  int32_t dilation_h, int32_t dilation_w) {

  auto *filter = static_cast<StridedMemRefType<float, 5> *>(f_ptr);
  auto *filterAllocated = filter->data + filter->offset;

  auto *input = static_cast<StridedMemRefType<float, 5> *>(i_ptr);
  auto *inputAllocated = input->data + input->offset;

  auto *output = static_cast<StridedMemRefType<float, 5> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;

  // Extract proper tensor sizes and strides based on layouts
  llvm::SmallVector<int64_t, 5> filterSizes, filterStrides;
  llvm::SmallVector<int64_t, 5> inputSizes, inputStrides;
  llvm::SmallVector<int64_t, 5> outputSizes, outputStrides;
  getSizesAndStrides(rank1, filter, rank2, input, rank3, output, f_layout,
                     i_layout, o_layout, filterSizes, filterStrides, inputSizes,
                     inputStrides, outputSizes, outputStrides);

  // Perform bwd_data convolution
  for (int64_t g = 0; g < outputSizes[0]; g++)
    for (int64_t n = 0; n < inputSizes[1]; n++)
      for (int64_t c = 0; c < inputSizes[2]; c++)
        for (int64_t in_h = 0; in_h < inputSizes[3]; in_h++)
          for (int64_t in_w = 0; in_w < inputSizes[4]; in_w++) {

            float acc = 0.0;
            for (int64_t k = 0; k < filterSizes[1]; k++)
              for (int64_t y = 0; y < filterSizes[3]; y++)
                for (int64_t x = 0; x < filterSizes[4]; x++) {
                  int64_t out_h_tmp = in_h + padding_h_l - y * dilation_h;
                  int64_t out_w_tmp = in_w + padding_w_l - x * dilation_w;
                  int64_t out_h = out_h_tmp / stride_h;
                  int64_t out_w = out_w_tmp / stride_w;
                  if (out_h_tmp % stride_h == 0 && out_w_tmp % stride_w == 0 &&
                      out_h >= 0 && out_h < outputSizes[3] && out_w >= 0 &&
                      out_w < outputSizes[4])
                    acc += filterAllocated[g * filterStrides[0] +
                                           k * filterStrides[1] +
                                           c * filterStrides[2] +
                                           y * filterStrides[3] +
                                           x * filterStrides[4]] *
                           outputAllocated[g * outputStrides[0] +
                                           n * outputStrides[1] +
                                           k * outputStrides[2] +
                                           out_h * outputStrides[3] +
                                           out_w * outputStrides[4]];
                }
            inputAllocated[g * inputStrides[0] + n * inputStrides[1] +
                           c * inputStrides[2] + in_h * inputStrides[3] +
                           in_w * inputStrides[4]] = acc;
          }
}
