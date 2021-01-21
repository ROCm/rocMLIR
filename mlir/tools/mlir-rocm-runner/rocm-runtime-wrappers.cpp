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
#include "bf16convert.hpp"

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

extern "C" void mcpuMemFloatConvertBf16(float *sourceAllocated, float *sourceAligned,
                            int64_t sourceOffset, int64_t sourceSize,
                            int64_t sourceStride,
                            ushort *destAllocated, ushort *destAligned,
                            int64_t destOffset, int64_t destSize,
                            int64_t destStride) {
  assert(sourceSize == destSize);
  for( int64_t i =0; i < destSize; i++){
    destAligned[i] = float_to_bfloat16(sourceAligned[i]); 
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
extern "C" void mcpuConv2d(int64_t rank1, void *f_ptr, int64_t rank2,
                           void *i_ptr, int64_t rank3, void *o_ptr,
                           int64_t rank4, void *f_layout, int64_t rank5,
                           void *i_layout, int64_t rank6, void *o_layout,
                           int32_t stride_h, int32_t stride_w,
                           int32_t padding_h, int32_t padding_w,
                           int32_t dilation_h, int32_t dilation_w) {

  auto *filter = static_cast<StridedMemRefType<float, 4> *>(f_ptr);
  auto *filterAllocated = filter->data + filter->offset;
  auto filterSizes = llvm::ArrayRef<int64_t>(filter->sizes, rank1);
  auto filterStrides = llvm::ArrayRef<int64_t>(filter->strides, rank1);

  auto *input = static_cast<StridedMemRefType<float, 4> *>(i_ptr);
  auto *inputAllocated = input->data + input->offset;
  auto inputSizes = llvm::ArrayRef<int64_t>(input->sizes, rank2);
  auto inputStrides = llvm::ArrayRef<int64_t>(input->strides, rank2);

  auto *output = static_cast<StridedMemRefType<float, 4> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;
  auto outputSizes = llvm::ArrayRef<int64_t>(output->sizes, rank3);
  auto outputStrides = llvm::ArrayRef<int64_t>(output->strides, rank3);

  auto *layout1 = static_cast<StridedMemRefType<char, 1> *>(f_layout);
  auto *filterLayout = layout1->data + layout1->offset;

  auto *layout2 = static_cast<StridedMemRefType<char, 1> *>(i_layout);
  auto *inputLayout = layout2->data + layout2->offset;

  auto *layout3 = static_cast<StridedMemRefType<char, 1> *>(o_layout);
  auto *outputLayout = layout3->data + layout3->offset;

  // Extract proper tensor sizes and strides based on layouts
  std::unordered_map<char, std::pair<int64_t, int64_t>> filterSizeStride;
  std::unordered_map<char, std::pair<int64_t, int64_t>> inputSizeStride;
  std::unordered_map<char, std::pair<int64_t, int64_t>> outputSizeStride;
  for (size_t i = 0; i < 4; i++) {
    filterSizeStride[filterLayout[i]] =
        std::make_pair(filterSizes[i], filterStrides[i]);
    inputSizeStride[inputLayout[i]] =
        std::make_pair(inputSizes[i], inputStrides[i]);
    outputSizeStride[outputLayout[i]] =
        std::make_pair(outputSizes[i], outputStrides[i]);
  }

  // Perform forward convolution
  for (int64_t n = 0; n < outputSizeStride['n'].first; n++)
    for (int64_t k = 0; k < outputSizeStride['k'].first; k++)
      for (int64_t out_h = 0; out_h < outputSizeStride['h'].first; out_h++)
        for (int64_t out_w = 0; out_w < outputSizeStride['w'].first; out_w++) {

          float acc = 0.0;
          for (int64_t c = 0; c < inputSizeStride['c'].first; c++)
            for (int64_t fil_h = 0; fil_h < filterSizeStride['y'].first;
                 fil_h++)
              for (int64_t fil_w = 0; fil_w < filterSizeStride['x'].first;
                   fil_w++) {

                float input;
                int64_t in_h =
                    out_h * stride_h + fil_h * dilation_h - padding_h;
                int64_t in_w =
                    out_w * stride_w + fil_w * dilation_w - padding_w;

                if (in_h < 0 || in_h >= inputSizeStride['h'].first ||
                    in_w < 0 || in_w >= inputSizeStride['w'].first)
                  input = 0.0;
                else
                  input = inputAllocated[n * inputSizeStride['n'].second +
                                         c * inputSizeStride['c'].second +
                                         in_h * inputSizeStride['h'].second +
                                         in_w * inputSizeStride['w'].second];

                acc += input *
                       filterAllocated[k * filterSizeStride['k'].second +
                                       c * filterSizeStride['c'].second +
                                       fil_h * filterSizeStride['y'].second +
                                       fil_w * filterSizeStride['x'].second];
              }

          outputAllocated[n * outputSizeStride['n'].second +
                          k * outputSizeStride['k'].second +
                          out_h * outputSizeStride['h'].second +
                          out_w * outputSizeStride['w'].second] = acc;
        }
}
