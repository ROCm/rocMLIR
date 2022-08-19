//===- conv-validation-wrappers.cpp - conv validation wrapper library -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C wrappers around convolution validations for easy linking in
// ORC jit.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <mutex>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>
#include <unordered_map>

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
static int generateTables(unsigned short *basetable,
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

  return 1;
}

static unsigned short float_to_fp16(float src_val) {
  // Generate tables for converting float to fp16
  static unsigned short basetable[512];
  static unsigned char shifttable[512];
  static std::once_flag flag;
  std::call_once(flag, generateTables, basetable, shifttable);

  // ref. http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
  bf16_fp32_cvt_t target_val;
  target_val.f32 = src_val;

  unsigned int b = target_val.u32;
  unsigned short h = basetable[(b >> 23) & 0x1ff] +
                     ((b & 0x007fffff) >> shifttable[(b >> 23) & 0x1ff]);
  return h;
}

short randomIntegerValue(short min, short max) {
  if (min == max)
    return min;
  return (std::rand() % (max - min)) + min;
}

float randomFloatValue(short min, short max) {
  auto minAsF = static_cast<float>(min);
  if (min == max)
    return minAsF * 0.1f; // avoid inf
  return static_cast<float>((max - min) * static_cast<double>(std::rand()) /
                            static_cast<double>(RAND_MAX)) +
         minAsF;
}

extern "C" void mcpuMemset(float *allocated, float *aligned, int64_t offset,
                           int64_t size, int64_t stride, float value) {
  for (unsigned i = 0; i < size; ++i) {
    aligned[i] = value;
  }
}

extern "C" void
mcpuMemset5DInt8RandInt(int8_t *allocated, int8_t *aligned, int64_t offset,
                        int64_t size0, int64_t size1, int64_t size2,
                        int64_t size3, int64_t size4, int64_t stride0,
                        int64_t stride1, int64_t stride2, int64_t stride3,
                        int64_t stride4, short min, short max, uint32_t seed) {
  if (seed == 0)
    std::srand(time(0));
  else
    std::srand(seed);

  int8_t value;
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m) {
            value = (int8_t)randomIntegerValue(min, max);
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
          }
}

extern "C" void
mcpuMemset5DInt32RandInt(int32_t *allocated, int32_t *aligned, int64_t offset,
                         int64_t size0, int64_t size1, int64_t size2,
                         int64_t size3, int64_t size4, int64_t stride0,
                         int64_t stride1, int64_t stride2, int64_t stride3,
                         int64_t stride4, short min, short max, uint32_t seed) {
  if (seed == 0)
    std::srand(time(0));
  else
    std::srand(seed);

  int32_t value;
  for (unsigned i = 0; i < size0; ++i)
    for (unsigned j = 0; j < size1; ++j)
      for (unsigned k = 0; k < size2; ++k)
        for (unsigned l = 0; l < size3; ++l)
          for (unsigned m = 0; m < size4; ++m) {
            value = (int32_t)randomIntegerValue(min, max);
            aligned[i * stride0 + j * stride1 + k * stride2 + l * stride3 +
                    m * stride4] = value;
          }
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

  int64_t dataSize = size0 * size1 * size2 * size3 * size4;
  for (int64_t i = 0; i < dataSize; i++) {
    destAligned[i] = float_to_fp16(sourceAligned[i]);
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

extern "C" void mcpuPrintF32(float f1, float f2) {
  printf("Values: %f, %f\n", f1, f2);
}

extern "C" void mcpuPrintInt32(int32_t d1, int32_t d2) {
  printf("Values: %d, %d\n", d1, d2);
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

extern "C" void
mcpuMemset5DFloatRandInt(float *allocated, float *aligned, int64_t offset,
                         int64_t size0, int64_t size1, int64_t size2,
                         int64_t size3, int64_t size4, int64_t stride0,
                         int64_t stride1, int64_t stride2, int64_t stride3,
                         int64_t stride4, short min, short max, uint32_t seed) {
  if (seed == 0)
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
    short min, short max, uint32_t seed) {
  if (seed == 0)
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
    int64_t stride4, short min, short max, uint32_t seed) {

  if (seed == 0)
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
                    m * stride4] = float_to_fp16(value);
          }
}

extern "C" void mcpuMemset5DHalfRandFloat(
    unsigned short *allocated, unsigned short *aligned, int64_t offset,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t stride4, short min, short max, uint32_t seed) {

  if (seed == 0)
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
                    m * stride4] = float_to_fp16(value);
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
    int64_t stride4, short min, short max, uint32_t seed) {

  if (seed == 0)
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
    int64_t stride4, short min, short max, uint32_t seed) {

  if (seed == 0)
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

// Extract proper tensor sizes and strides based on layouts
static void extractSizesAndStrides(
    llvm::ArrayRef<int64_t> filterSizes, llvm::ArrayRef<int64_t> filterStrides,
    llvm::ArrayRef<int64_t> inputSizes, llvm::ArrayRef<int64_t> inputStrides,
    llvm::ArrayRef<int64_t> outputSizes, llvm::ArrayRef<int64_t> outputStrides,
    void *f_layout, void *i_layout, void *o_layout,
    std::array<int64_t, 5> &fSizes, std::array<int64_t, 5> &fStrides,
    std::array<int64_t, 5> &iSizes, std::array<int64_t, 5> &iStrides,
    std::array<int64_t, 5> &oSizes, std::array<int64_t, 5> &oStrides) {
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

template <typename T1, typename T2>
static void getSizesAndStrides(int64_t rank1, StridedMemRefType<T1, 5> *filter,
                               int64_t rank2, StridedMemRefType<T1, 5> *input,
                               int64_t rank3, StridedMemRefType<T2, 5> *output,
                               void *f_layout, void *i_layout, void *o_layout,
                               std::array<int64_t, 5> &fSizes,
                               std::array<int64_t, 5> &fStrides,
                               std::array<int64_t, 5> &iSizes,
                               std::array<int64_t, 5> &iStrides,
                               std::array<int64_t, 5> &oSizes,
                               std::array<int64_t, 5> &oStrides) {
  auto filterSizes = llvm::ArrayRef<int64_t>(filter->sizes, rank1);
  auto filterStrides = llvm::ArrayRef<int64_t>(filter->strides, rank1);

  auto inputSizes = llvm::ArrayRef<int64_t>(input->sizes, rank2);
  auto inputStrides = llvm::ArrayRef<int64_t>(input->strides, rank2);

  auto outputSizes = llvm::ArrayRef<int64_t>(output->sizes, rank3);
  auto outputStrides = llvm::ArrayRef<int64_t>(output->strides, rank3);

  extractSizesAndStrides(filterSizes, filterStrides, inputSizes, inputStrides,
                         outputSizes, outputStrides, f_layout, i_layout,
                         o_layout, fSizes, fStrides, iSizes, iStrides, oSizes,
                         oStrides);
}

template <typename TIn, typename TOut, typename TAcc>
static void performConv2d(
    TIn *filterAllocated, TIn *inputAllocated, TOut *outputAllocated,
    llvm::ArrayRef<int64_t> filterSizes, llvm::ArrayRef<int64_t> filterStrides,
    llvm::ArrayRef<int64_t> inputSizes, llvm::ArrayRef<int64_t> inputStrides,
    llvm::ArrayRef<int64_t> outputSizes, llvm::ArrayRef<int64_t> outputStrides,
    int32_t stride_h, int32_t stride_w, int32_t padding_h_l,
    int32_t padding_h_r, int32_t padding_w_l, int32_t padding_w_r,
    int32_t dilation_h, int32_t dilation_w, int32_t xdlops) {

  // Perform forward convolution
  for (int64_t g = 0; g < outputSizes[0]; g++)
    for (int64_t n = 0; n < outputSizes[1]; n++)
      for (int64_t k = 0; k < outputSizes[2]; k++)
        for (int64_t out_h = 0; out_h < outputSizes[3]; out_h++)
          for (int64_t out_w = 0; out_w < outputSizes[4]; out_w++) {

            TAcc acc = 0.0;
            for (int64_t c = 0; c < inputSizes[2]; c++)
              for (int64_t fil_h = 0; fil_h < filterSizes[3]; fil_h++)
                for (int64_t fil_w = 0; fil_w < filterSizes[4]; fil_w++) {

                  TIn input;
                  int64_t in_h =
                      out_h * stride_h + fil_h * dilation_h - padding_h_l;
                  int64_t in_w =
                      out_w * stride_w + fil_w * dilation_w - padding_w_l;

                  if (in_h < 0 || in_h >= inputSizes[3] || in_w < 0 ||
                      in_w >= inputSizes[4])
                    input = (TIn)0;
                  else

                    input = inputAllocated[g * inputStrides[0] +
                                           n * inputStrides[1] +
                                           c * inputStrides[2] +
                                           in_h * inputStrides[3] +
                                           in_w * inputStrides[4]];

                  acc +=
                      (TAcc)(input * filterAllocated[g * filterStrides[0] +
                                                     k * filterStrides[1] +
                                                     c * filterStrides[2] +
                                                     fil_h * filterStrides[3] +
                                                     fil_w * filterStrides[4]]);
                  if (!xdlops) // || (fil_w + fil_h + c) % 4 == 3)
                    acc = (TOut)acc;
                }

            outputAllocated[g * outputStrides[0] + n * outputStrides[1] +
                            k * outputStrides[2] + out_h * outputStrides[3] +
                            out_w * outputStrides[4]] = (TOut)acc;
          }
}

// A generic forward convolution function that supports random layouts,
// dimensions, strides, paddings, and dilations.
extern "C" void
mcpuConv2dFloat(int64_t rank1, void *f_ptr, int64_t rank2, void *i_ptr,
                int64_t rank3, void *o_ptr, int64_t rank4, void *f_layout,
                int64_t rank5, void *i_layout, int64_t rank6, void *o_layout,
                int32_t stride_h, int32_t stride_w, int32_t padding_h_l,
                int32_t padding_h_r, int32_t padding_w_l, int32_t padding_w_r,
                int32_t dilation_h, int32_t dilation_w, int32_t xdlops) {
  auto *filter = static_cast<StridedMemRefType<float, 5> *>(f_ptr);
  auto *filterAllocated = filter->data + filter->offset;

  auto *input = static_cast<StridedMemRefType<float, 5> *>(i_ptr);
  auto *inputAllocated = input->data + input->offset;

  auto *output = static_cast<StridedMemRefType<float, 5> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;

  // Extract proper tensor sizes and strides based on layouts
  std::array<int64_t, 5> filterSizes, filterStrides;
  std::array<int64_t, 5> inputSizes, inputStrides;
  std::array<int64_t, 5> outputSizes, outputStrides;

  getSizesAndStrides<float, float>(rank1, filter, rank2, input, rank3, output,
                                   f_layout, i_layout, o_layout, filterSizes,
                                   filterStrides, inputSizes, inputStrides,
                                   outputSizes, outputStrides);
  performConv2d<float, float, double>(
      filterAllocated, inputAllocated, outputAllocated, filterSizes,
      filterStrides, inputSizes, inputStrides, outputSizes, outputStrides,
      stride_h, stride_w, padding_h_l, padding_h_r, padding_w_l, padding_w_r,
      dilation_h, dilation_w, xdlops);
}

// A generic backward-weight convolution function that supports random layouts,
// dimensions, strides, paddings, and dilations.
extern "C" void mcpuConv2dBwdWeightFloat(
    int64_t rank1, void *f_ptr, int64_t rank2, void *i_ptr, int64_t rank3,
    void *o_ptr, int64_t rank4, void *f_layout, int64_t rank5, void *i_layout,
    int64_t rank6, void *o_layout, int32_t stride_h, int32_t stride_w,
    int32_t padding_h_l, int32_t padding_h_r, int32_t padding_w_l,
    int32_t padding_w_r, int32_t dilation_h, int32_t dilation_w,
    int32_t xdlops) {

  auto *filter = static_cast<StridedMemRefType<float, 5> *>(f_ptr);
  auto *filterAllocated = filter->data + filter->offset;

  auto *input = static_cast<StridedMemRefType<float, 5> *>(i_ptr);
  auto *inputAllocated = input->data + input->offset;

  auto *output = static_cast<StridedMemRefType<float, 5> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;

  // Extract proper tensor sizes and strides based on layouts
  std::array<int64_t, 5> filterSizes, filterStrides;
  std::array<int64_t, 5> inputSizes, inputStrides;
  std::array<int64_t, 5> outputSizes, outputStrides;
  getSizesAndStrides<float, float>(rank1, filter, rank2, input, rank3, output,
                                   f_layout, i_layout, o_layout, filterSizes,
                                   filterStrides, inputSizes, inputStrides,
                                   outputSizes, outputStrides);

  // Perform bwd_weight convolution
  for (int64_t g = 0; g < outputSizes[0]; g++)
    for (int64_t k = 0; k < filterSizes[1]; k++)
      for (int64_t c = 0; c < filterSizes[2]; c++)
        for (int64_t y = 0; y < filterSizes[3]; y++)
          for (int64_t x = 0; x < filterSizes[4]; x++) {

            double acc = 0.0;
            for (int64_t n = 0; n < outputSizes[1]; n++)
              for (int64_t out_h = 0; out_h < outputSizes[3]; out_h++)
                for (int64_t out_w = 0; out_w < outputSizes[4]; out_w++) {
                  int64_t in_h =
                      out_h * stride_h + y * dilation_h - padding_h_l;
                  int64_t in_w =
                      out_w * stride_w + x * dilation_w - padding_w_l;
                  if (in_h >= 0 && in_h < inputSizes[3] && in_w >= 0 &&
                      in_w < inputSizes[4])
                    acc += (double)(inputAllocated[g * inputStrides[0] +
                                                   n * inputStrides[1] +
                                                   c * inputStrides[2] +
                                                   in_h * inputStrides[3] +
                                                   in_w * inputStrides[4]] *
                                    outputAllocated[g * outputStrides[0] +
                                                    n * outputStrides[1] +
                                                    k * outputStrides[2] +
                                                    out_h * outputStrides[3] +
                                                    out_w * outputStrides[4]]);
                  if (!xdlops) // || (out_w + out_h + n) % 4 == 3)
                    acc = (float)acc;
                }
            filterAllocated[g * filterStrides[0] + k * filterStrides[1] +
                            c * filterStrides[2] + y * filterStrides[3] +
                            x * filterStrides[4]] = (float)acc;
          }
}

// A generic backward-data convolution function that supports random layouts,
// dimensions, strides, paddings, and dilations.
extern "C" void mcpuConv2dBwdDataFloat(
    int64_t rank1, void *f_ptr, int64_t rank2, void *i_ptr, int64_t rank3,
    void *o_ptr, int64_t rank4, void *f_layout, int64_t rank5, void *i_layout,
    int64_t rank6, void *o_layout, int32_t stride_h, int32_t stride_w,
    int32_t padding_h_l, int32_t padding_h_r, int32_t padding_w_l,
    int32_t padding_w_r, int32_t dilation_h, int32_t dilation_w,
    int32_t xdlops) {

  auto *filter = static_cast<StridedMemRefType<float, 5> *>(f_ptr);
  auto *filterAllocated = filter->data + filter->offset;

  auto *input = static_cast<StridedMemRefType<float, 5> *>(i_ptr);
  auto *inputAllocated = input->data + input->offset;

  auto *output = static_cast<StridedMemRefType<float, 5> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;

  // Extract proper tensor sizes and strides based on layouts
  std::array<int64_t, 5> filterSizes, filterStrides;
  std::array<int64_t, 5> inputSizes, inputStrides;
  std::array<int64_t, 5> outputSizes, outputStrides;
  getSizesAndStrides<float, float>(rank1, filter, rank2, input, rank3, output,
                                   f_layout, i_layout, o_layout, filterSizes,
                                   filterStrides, inputSizes, inputStrides,
                                   outputSizes, outputStrides);

  // Perform bwd_data convolution
  for (int64_t g = 0; g < outputSizes[0]; g++)
    for (int64_t n = 0; n < inputSizes[1]; n++)
      for (int64_t c = 0; c < inputSizes[2]; c++)
        for (int64_t in_h = 0; in_h < inputSizes[3]; in_h++)
          for (int64_t in_w = 0; in_w < inputSizes[4]; in_w++) {

            double acc = 0.0;
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
                    acc += (double)(filterAllocated[g * filterStrides[0] +
                                                    k * filterStrides[1] +
                                                    c * filterStrides[2] +
                                                    y * filterStrides[3] +
                                                    x * filterStrides[4]] *
                                    outputAllocated[g * outputStrides[0] +
                                                    n * outputStrides[1] +
                                                    k * outputStrides[2] +
                                                    out_h * outputStrides[3] +
                                                    out_w * outputStrides[4]]);
                  if (!xdlops) // || (x + y + k) % 4 == 3)
                    acc = (float)acc;
                }
            inputAllocated[g * inputStrides[0] + n * inputStrides[1] +
                           c * inputStrides[2] + in_h * inputStrides[3] +
                           in_w * inputStrides[4]] = acc;
          }
}

extern "C" void
mcpuConv2dInt8(int64_t rank1, void *f_ptr, int64_t rank2, void *i_ptr,
               int64_t rank3, void *o_ptr, int64_t rank4, void *f_layout,
               int64_t rank5, void *i_layout, int64_t rank6, void *o_layout,
               int32_t stride_h, int32_t stride_w, int32_t padding_h_l,
               int32_t padding_h_r, int32_t padding_w_l, int32_t padding_w_r,
               int32_t dilation_h, int32_t dilation_w, int32_t xdlops) {
  auto *filter = static_cast<StridedMemRefType<int8_t, 5> *>(f_ptr);
  auto *filterAllocated = filter->data + filter->offset;

  auto *input = static_cast<StridedMemRefType<int8_t, 5> *>(i_ptr);
  auto *inputAllocated = input->data + input->offset;

  auto *output = static_cast<StridedMemRefType<int32_t, 5> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;

  // Extract proper tensor sizes and strides based on layouts
  std::array<int64_t, 5> filterSizes, filterStrides;
  std::array<int64_t, 5> inputSizes, inputStrides;
  std::array<int64_t, 5> outputSizes, outputStrides;

  getSizesAndStrides<int8_t, int32_t>(rank1, filter, rank2, input, rank3,
                                      output, f_layout, i_layout, o_layout,
                                      filterSizes, filterStrides, inputSizes,
                                      inputStrides, outputSizes, outputStrides);

  performConv2d<int8_t, int32_t, int32_t>(
      filterAllocated, inputAllocated, outputAllocated, filterSizes,
      filterStrides, inputSizes, inputStrides, outputSizes, outputStrides,
      stride_h, stride_w, padding_h_l, padding_h_r, padding_w_l, padding_w_r,
      dilation_h, dilation_w, xdlops);
}

int findIdxHistRelDiff(double relDiff) {
  /*
    hist_relDiff[]:
    0: 0
    1: 0 - 1e-6
    2: 1e-6 - 1e-5
    3: 1e-5 - 1e-4
    4: 1e-4 - 1e-3
    5: 1e-3 - 1e-2
    6: 1e-2 - 0.1
    7: 0.1 - 1
    8: >= 1
   */
  if (relDiff == 0.0)
    return 0;
  if (relDiff < 1.0e-06)
    return 1;
  if (relDiff < 1.0e-05)
    return 2;
  if (relDiff < 1.0e-04)
    return 3;
  if (relDiff < 1.0e-03)
    return 4;
  if (relDiff < 1.0e-02)
    return 5;
  if (relDiff < 0.1)
    return 6;
  if (relDiff < 1.0)
    return 7;
  return 8;
}

template <typename T>
void mcpuVerify(T *gpuResults, T *validationResults, int64_t dataSize,
                float thr_RMS, float thr_absDiff, float thr_relDiff)
{
  float valNum, gpuNum;
  // metric maxAbsDiff
  float maxAbsDiff = 0.0f;
  double sumAbsDiff = 0.0;
  float maxVAL_abs = 0.0f;
  float maxGPU_abs = 0.0f;
  // metric maxRelDiff
  double maxRelDiff = 0.0;
  double sumRelDiff = 0.0;
  float maxVAL_rel = 0.0f;
  float maxGPU_rel = 0.0f;
  // Metric RMS
  float maxMag = 0.0f;
  double sumDiffSq = 0.0;
  // histogram
  // hist_relDiff[9]++ when cpuVal == 0
  int hist_relDiff[10] = {0};
  for (int64_t i = 0; i < dataSize; ++i) {
    valNum = static_cast<float>(validationResults[i]);
    gpuNum = static_cast<float>(gpuResults[i]);
    // Update the max magnitutde value
    float maxNum = std::max(fabs(valNum), fabs(gpuNum));
    maxMag = std::max(maxMag, maxNum);

    if (valNum == gpuNum) {
      hist_relDiff[0]++;
    } else {
      float absDiff = fabs(valNum - gpuNum);
      // Update maxAbsDiff and its correspinding pair of values
      if (absDiff > maxAbsDiff) {
        maxVAL_abs = valNum;
        maxGPU_abs = gpuNum;
        maxAbsDiff = absDiff;
      }
      sumAbsDiff += static_cast<double>(absDiff);
      // Update maxRelDiff only if cpuVal != 0
      if (valNum != 0.0f) {
        double relDiff =
            static_cast<double>(absDiff) / (static_cast<double>(fabs(valNum)));
        hist_relDiff[findIdxHistRelDiff(relDiff)]++;
        if (relDiff > maxRelDiff) {
          maxVAL_rel = valNum;
          maxGPU_rel = gpuNum;
          maxRelDiff = relDiff;
        }
        sumRelDiff += relDiff;
      } else {
        hist_relDiff[9]++;
      }
      // Accumulate square root
      sumDiffSq += static_cast<double>(absDiff) * static_cast<double>(absDiff);
    }
  }
  double aveAbsDiff = sumAbsDiff / static_cast<double>(dataSize);
  double aveRelDiff = sumRelDiff / static_cast<double>(dataSize);
  double err_RMS = sqrt(sumDiffSq) / (static_cast<double>(maxMag) *
                                      sqrt(static_cast<double>(dataSize)));
  // Verbose information about the difference
  printf("Number of elements: %ld\n", dataSize);
  printf("maxAbsDiff info: maxAbsDiff = %f (valNum = %.5f, gpuNum = %.5f), "
         "average absDiff = %.1e\n",
         maxAbsDiff, maxVAL_abs, maxGPU_abs, aveAbsDiff);
  printf("maxRelDiff info: maxRelDiff = %.1e (valNum = %.10f, gpuNum = %.10f), "
         "average relDiff = %.1e\n",
         maxRelDiff, maxVAL_rel, maxGPU_rel, aveRelDiff);
  printf("RMS = %.1e\n", err_RMS);
  printf("Histogram of relDiff: \n");
  printf("          relDiff = 0   : %d/%ld (%lf%%)\n", hist_relDiff[0],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[0]) /
             static_cast<double>(dataSize));
  printf("      0 < relDiff < 1e-6: %d/%ld (%lf%%)\n", hist_relDiff[1],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[1]) /
             static_cast<double>(dataSize));
  printf("  1e-6 <= relDiff < 1e-5: %d/%ld (%lf%%)\n", hist_relDiff[2],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[2]) /
             static_cast<double>(dataSize));
  printf("  1e-5 <= relDiff < 1e-4: %d/%ld (%lf%%)\n", hist_relDiff[3],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[3]) /
             static_cast<double>(dataSize));
  printf("  1e-4 <= relDiff < 1e-3: %d/%ld (%lf%%)\n", hist_relDiff[4],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[4]) /
             static_cast<double>(dataSize));
  printf("  1e-3 <= relDiff < 1e-2: %d/%ld (%lf%%)\n", hist_relDiff[5],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[5]) /
             static_cast<double>(dataSize));
  printf("  1e-2 <= relDiff < 0.1 : %d/%ld (%lf%%)\n", hist_relDiff[6],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[6]) /
             static_cast<double>(dataSize));
  printf("   0.1 <= relDiff < 1   : %d/%ld (%lf%%)\n", hist_relDiff[7],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[7]) /
             static_cast<double>(dataSize));
  printf("     1 <= relDiff < inf : %d/%ld (%lf%%)\n", hist_relDiff[8],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[8]) /
             static_cast<double>(dataSize));
  printf("             valNum == 0: %d/%ld (%lf%%)\n", hist_relDiff[9],
         dataSize,
         100.0 * static_cast<double>(hist_relDiff[9]) /
             static_cast<double>(dataSize));
  // Check if pass based on all three metrics: RMS, maxAbsDiff, maxRelDiff
  int RMS_pass = (err_RMS <= thr_RMS) ? 1 : 0;
  int absDiff_pass = (maxAbsDiff <= thr_absDiff) ? 1 : 0;
  int relDiff_pass = (maxRelDiff <= thr_relDiff) ? 1 : 0;
  printf("[%d %d %d]\n", RMS_pass, absDiff_pass, relDiff_pass);
}

// Compare the results in f32
extern "C" void mcpuVerify5DFloat(
    float *gpuAllocated, float *gpuAligned, int64_t gpuOffset, int64_t gpuSize0,
    int64_t gpuSize1, int64_t gpuSize2, int64_t gpuSize3, int64_t gpuSize4,
    int64_t gpuStride0, int64_t gpuStride1, int64_t gpuStride2,
    int64_t gpuStride3, int64_t gpuStride4, float *valAllocated,
    float *valAligned, int64_t valOffset, int64_t valSize0, int64_t valSize1,
    int64_t valSize2, int64_t valSize3, int64_t valSize4, int64_t valStride0,
    int64_t valStride1, int64_t valStride2, int64_t valStride3,
    int64_t valStride4, float thr_RMS, float thr_absDiff, float thr_relDiff) {
  assert(gpuSize0 * gpuSize1 * gpuSize2 * gpuSize3 * gpuSize4 ==
         valSize0 * valSize1 * valSize2 * valSize3 * valSize4);
  int64_t dataSize = valSize0 * valSize1 * valSize2 * valSize3 * valSize4;
  mcpuVerify<float>(gpuAligned, valAligned, dataSize, thr_RMS, thr_absDiff, thr_relDiff);
}

// Compare the results in int32
extern "C" void mcpuVerify5DInt32(
    int32_t *gpuAllocated, int32_t *gpuAligned, int64_t gpuOffset, int64_t gpuSize0,
    int64_t gpuSize1, int64_t gpuSize2, int64_t gpuSize3, int64_t gpuSize4,
    int64_t gpuStride0, int64_t gpuStride1, int64_t gpuStride2,
    int64_t gpuStride3, int64_t gpuStride4, int32_t *valAllocated,
    int32_t *valAligned, int64_t valOffset, int64_t valSize0, int64_t valSize1,
    int64_t valSize2, int64_t valSize3, int64_t valSize4, int64_t valStride0,
    int64_t valStride1, int64_t valStride2, int64_t valStride3,
    int64_t valStride4, float thr_RMS, float thr_absDiff, float thr_relDiff) {
    assert(gpuSize0 * gpuSize1 * gpuSize2 * gpuSize3 * gpuSize4 ==
           valSize0 * valSize1 * valSize2 * valSize3 * valSize4);
    int64_t dataSize = valSize0 * valSize1 * valSize2 * valSize3 * valSize4;
    mcpuVerify<int32_t>(gpuAligned, valAligned, dataSize, thr_RMS, thr_absDiff, thr_relDiff);
}
