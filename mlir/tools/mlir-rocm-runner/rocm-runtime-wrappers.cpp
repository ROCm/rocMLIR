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

namespace {
int32_t reportErrorIfAny(hipError_t result, const char *where) {
  if (result != hipSuccess) {
    llvm::errs() << "HIP failed with " << result << " in " << where << "\n";
  }
  return result;
}
} // anonymous namespace

extern "C" int32_t mgpuModuleLoad(void **module, void *data) {
  int32_t err = reportErrorIfAny(
      hipModuleLoadData(reinterpret_cast<hipModule_t *>(module), data),
      "ModuleLoad");
  return err;
}

extern "C" int32_t mgpuModuleGetFunction(void **function, void *module,
                                         const char *name) {
  return reportErrorIfAny(
      hipModuleGetFunction(reinterpret_cast<hipFunction_t *>(function),
                           reinterpret_cast<hipModule_t>(module), name),
      "GetFunction");
}

// The wrapper uses intptr_t instead of ROCM's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" int32_t mgpuLaunchKernel(void *function, intptr_t gridX,
                                    intptr_t gridY, intptr_t gridZ,
                                    intptr_t blockX, intptr_t blockY,
                                    intptr_t blockZ, int32_t smem, void *stream,
                                    void **params, void **extra) {
  return reportErrorIfAny(
      hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(function), gridX,
                            gridY, gridZ, blockX, blockY, blockZ, smem,
                            reinterpret_cast<hipStream_t>(stream), params,
                            extra),
      "LaunchKernel");
}

extern "C" void *mgpuGetStreamHelper() {
  hipStream_t stream;
  reportErrorIfAny(hipStreamCreate(&stream), "StreamCreate");
  return stream;
}

extern "C" int32_t mgpuStreamSynchronize(void *stream) {
  return reportErrorIfAny(
      hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream)),
      "StreamSync");
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

template <typename T>
void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
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
  hipMalloc((void**)&gpuPtr, size * sizeof(float));
  return {gpuPtr, gpuPtr, offset, {size}, {stride}};
}

extern "C" void mgpuMemDealloc(float *allocated, float *aligned,
                               int64_t offset, int64_t size, int64_t stride) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy(float *sourceAllocated, float *sourceAligned,
                            int64_t sourceOffset, int64_t sourceSize,
                            int64_t sourceStride,
                            float *destAllocated, float *destAligned,
                            int64_t destOffset, int64_t destSize,
                            int64_t destStride,
                            unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned, sourceSize * sizeof(float),
            static_cast<hipMemcpyKind>(copyDirection));
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

// 4D float memref utility routines.

extern "C" void mcpuMemset4DFloat(float *allocated, float *aligned, int64_t offset,
                                  int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                                  int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
                                  float value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3] = value;
}

extern "C" StridedMemRefType<float, 4>
mgpuMemAlloc4DFloat(float *allocated, float *aligned, int64_t offset,
                    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3) {
  float *gpuPtr;
  hipMalloc((void**)&gpuPtr, size0 * size1 * size2 * size3 * sizeof(float));
  return {gpuPtr, gpuPtr, offset, {size0, size1, size2, size3}, {stride0, stride1, stride2, stride3}};
}

extern "C" void mgpuMemDealloc4DFloat(float *allocated, float *aligned,
                                      int64_t offset,
                                      int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                                      int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy4DFloat(float *sourceAllocated, float *sourceAligned,
                                   int64_t sourceOffset,
                                   int64_t sourceSize0, int64_t sourceSize1,
                                   int64_t sourceSize2, int64_t sourceSize3,
                                   int64_t sourceStride0, int64_t sourceStride1,
                                   int64_t sourceStride2, int64_t sourceStride3,
                                   float *destAllocated, float *destAligned,
                                   int64_t destOffset,
                                   int64_t destSize0, int64_t destSize1,
                                   int64_t destSize2, int64_t destSize3,
                                   int64_t destStride0, int64_t destStride1,
                                   int64_t destStride2, int64_t destStride3,
                                   unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned, sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 * sizeof(float),
            static_cast<hipMemcpyKind>(copyDirection));
}

// 4D half memref utility routines.

extern "C" void mcpuMemset4DHalf(unsigned short *allocated, unsigned short *aligned, int64_t offset,
                                 int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                                 int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
                                 unsigned short value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3] = value;
}

extern "C" StridedMemRefType<unsigned short, 4>
mgpuMemAlloc4DHalf(unsigned short *allocated, unsigned short *aligned, int64_t offset,
                   int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                   int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3) {
  unsigned short *gpuPtr;
  hipMalloc((void**)&gpuPtr, size0 * size1 * size2 * size3 * sizeof(unsigned short));
  return {gpuPtr, gpuPtr, offset, {size0, size1, size2, size3}, {stride0, stride1, stride2, stride3}};
}

extern "C" void mgpuMemDealloc4DHalf(unsigned short *allocated, unsigned short *aligned,
                                     int64_t offset,
                                     int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                                     int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy4DHalf(unsigned short *sourceAllocated, unsigned short *sourceAligned,
                                  int64_t sourceOffset,
                                  int64_t sourceSize0, int64_t sourceSize1,
                                  int64_t sourceSize2, int64_t sourceSize3,
                                  int64_t sourceStride0, int64_t sourceStride1,
                                  int64_t sourceStride2, int64_t sourceStride3,
                                  unsigned short *destAllocated, unsigned short *destAligned,
                                  int64_t destOffset,
                                  int64_t destSize0, int64_t destSize1,
                                  int64_t destSize2, int64_t destSize3,
                                  int64_t destStride0, int64_t destStride1,
                                  int64_t destStride2, int64_t destStride3,
                                  unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned, sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 * sizeof(unsigned short),
            static_cast<hipMemcpyKind>(copyDirection));
}

// 4D bf16 memref utility routines.

extern "C" void mcpuMemset4DBF16(unsigned short *allocated, unsigned short *aligned, int64_t offset,
                                 int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                                 int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
                                 unsigned short value) {
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3] = value;
}

extern "C" StridedMemRefType<unsigned short, 4>
mgpuMemAlloc4DBF16(unsigned short *allocated, unsigned short *aligned, int64_t offset,
                   int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                   int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3) {
  unsigned short *gpuPtr;
  hipMalloc((void**)&gpuPtr, size0 * size1 * size2 * size3 * sizeof(unsigned short));
  return {gpuPtr, gpuPtr, offset, {size0, size1, size2, size3}, {stride0, stride1, stride2, stride3}};
}

extern "C" void mgpuMemDealloc4DBF16(unsigned short *allocated, unsigned short *aligned,
                                     int64_t offset,
                                     int64_t size0, int64_t size1, int64_t size2, int64_t size3,
                                     int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3) {
  hipFree(aligned);
}

extern "C" void mgpuMemCopy4DBF16(unsigned short *sourceAllocated, unsigned short *sourceAligned,
                                  int64_t sourceOffset,
                                  int64_t sourceSize0, int64_t sourceSize1,
                                  int64_t sourceSize2, int64_t sourceSize3,
                                  int64_t sourceStride0, int64_t sourceStride1,
                                  int64_t sourceStride2, int64_t sourceStride3,
                                  unsigned short *destAllocated, unsigned short *destAligned,
                                  int64_t destOffset,
                                  int64_t destSize0, int64_t destSize1,
                                  int64_t destSize2, int64_t destSize3,
                                  int64_t destStride0, int64_t destStride1,
                                  int64_t destStride2, int64_t destStride3,
                                  unsigned copyDirection) {
  hipMemcpy(destAligned, sourceAligned, sourceSize0 * sourceSize1 * sourceSize2 * sourceSize3 * sizeof(unsigned short),
            static_cast<hipMemcpyKind>(copyDirection));
}

// A generic forward convolution function that supports random layouts,
// dimensions, strides, paddings, and dilations.

extern "C" void
mcpuConv2d(float *filterAllocated, float *filterAligned, int64_t filterOffset,
           int64_t filterSize0, int64_t filterSize1, int64_t filterSize2,
           int64_t filterSize3, int64_t filterStride0, int64_t filterStride1,
           int64_t filterStride2, int64_t filterStride3, float *inputAllocated,
           float *inputAligned, int64_t inputOffset, int64_t inputSize0,
           int64_t inputSize1, int64_t inputSize2, int64_t inputSize3,
           int64_t inputStride0, int64_t inputStride1, int64_t inputStride2,
           int64_t inputStride3, float *outputAllocated, float *outputAligned,
           int64_t outputOffset, int64_t outputSize0, int64_t outputSize1,
           int64_t outputSize2, int64_t outputSize3, int64_t outputStride0,
           int64_t outputStride1, int64_t outputStride2, int64_t outputStride3,
           int *layouts, int *layoutsAlligned, int64_t layoutOffset,
           int64_t layoutSize, int64_t layoutStride, int32_t stride_h,
           int32_t stride_w, int32_t padding_h, int32_t padding_w,
           int32_t dilation_h, int32_t dilation_w) {

  llvm::SmallVector<int64_t, 4> filterSizes(
      {filterSize0, filterSize1, filterSize2, filterSize3});
  llvm::SmallVector<int64_t, 4> filterStrides(
      {filterStride0, filterStride1, filterStride2, filterStride3});
  llvm::SmallVector<int64_t, 4> inputSizes(
      {inputSize0, inputSize1, inputSize2, inputSize3});
  llvm::SmallVector<int64_t, 4> inputStrides(
      {inputStride0, inputStride1, inputStride2, inputStride3});
  llvm::SmallVector<int64_t, 4> outputSizes(
      {outputSize0, outputSize1, outputSize2, outputSize3});
  llvm::SmallVector<int64_t, 4> outputStrides(
      {outputStride0, outputStride1, outputStride2, outputStride3});

  int batchSize, outChannelSize, inChannelSize;
  int outHeightSize, outWidthSize;
  int inHeightSize, inWidthSize;
  int filterHeightSize, filterWidthSize;
  int inBatchStride, inChannelStride, inHeightStride, inWidthStride;
  int outBatchStride, outChannelStride, outHeightStride, outWidthStride;
  int filterOutChanStride, filterInChanStride, filterHeightStride,
      filterWidthStride;

  // layouts[0..3]: filter layout
  // layouts[4..7]: input layout
  // layouts[8..11]: output layout
  int *filterLayout = layouts;
  int *inputLayout = layouts + 4;
  int *outputLayout = layouts + 8;

  // Extract proper tensor sizes and strides based on layouts
  for (size_t i = 0; i < 4; i++) {
    if (filterLayout[i] == 'k') {
      filterOutChanStride = filterStrides[i];
    } else if (filterLayout[i] == 'c') {
      filterInChanStride = filterStrides[i];
    } else if (filterLayout[i] == 'y') {
      filterHeightSize = filterSizes[i];
      filterHeightStride = filterStrides[i];
    } else if (filterLayout[i] == 'x') {
      filterWidthSize = filterSizes[i];
      filterWidthStride = filterStrides[i];
    }

    if (inputLayout[i] == 'n') {
      inBatchStride = inputStrides[i];
    } else if (inputLayout[i] == 'c') {
      inChannelSize = inputSizes[i];
      inChannelStride = inputStrides[i];
    } else if (inputLayout[i] == 'h') {
      inHeightSize = inputSizes[i];
      inHeightStride = inputStrides[i];
    } else if (inputLayout[i] == 'w') {
      inWidthSize = inputSizes[i];
      inWidthStride = inputStrides[i];
    }

    if (outputLayout[i] == 'n') {
      batchSize = outputSizes[i];
      outBatchStride = outputStrides[i];
    } else if (outputLayout[i] == 'k') {
      outChannelSize = outputSizes[i];
      outChannelStride = outputStrides[i];
    } else if (outputLayout[i] == 'h') {
      outHeightSize = outputSizes[i];
      outHeightStride = outputStrides[i];
    } else if (outputLayout[i] == 'w') {
      outWidthSize = outputSizes[i];
      outWidthStride = outputStrides[i];
    }
  }

  // Perform forward convolution
  for (int n = 0; n < batchSize; n++)
    for (int k = 0; k < outChannelSize; k++)
      for (int out_h = 0; out_h < outHeightSize; out_h++)
        for (int out_w = 0; out_w < outWidthSize; out_w++)
          for (int c = 0; c < inChannelSize; c++)
            for (int fil_h = 0; fil_h < filterHeightSize; fil_h++)
              for (int fil_w = 0; fil_w < filterWidthSize; fil_w++) {
                float input;
                int in_h = out_h * stride_h + fil_h * dilation_h - padding_h;
                int in_w = out_w * stride_w + fil_w * dilation_w - padding_w;

                if (in_h < 0 || in_h >= inHeightSize || in_w < 0 ||
                    in_w >= inWidthSize)
                  input = 0.0;
                else
                  input =
                      inputAllocated[n * inBatchStride + c * inChannelStride +
                                     in_h * inHeightStride +
                                     in_w * inWidthStride];

                outputAllocated[n * outBatchStride + k * outChannelStride +
                                out_h * outHeightStride +
                                out_w * outWidthStride] +=
                    input * filterAllocated[k * filterOutChanStride +
                                            c * filterInChanStride +
                                            fil_h * filterHeightStride +
                                            fil_w * filterWidthStride];
              }
}
