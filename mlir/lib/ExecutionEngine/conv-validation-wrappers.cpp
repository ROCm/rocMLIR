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
#include <unordered_map>
#include <cmath>

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

extern "C" void mcpuPrint5DBF16(unsigned short *allocated,
                              unsigned short *aligned, int64_t offset,
                              int64_t size0, int64_t size1, int64_t size2,
                                int64_t size3, int64_t size4,
                                int64_t stride0, int64_t stride1,
                                int64_t stride2, int64_t stride3, int64_t stride4) {
    int64_t dataSize = size0 * size1 * size2 * size3 * size4;
    for (int64_t i = 0; i < dataSize; i++) {
        float fvalue = bfloat16_to_float(aligned[i]);
        printf("%f\t", fvalue);
    }
    printf("\n");
}

extern "C" void mcpuPrintBF161D(unsigned short *allocated,
                              unsigned short *aligned, int64_t offset,
                              int64_t size0, int64_t stride0) {
    int64_t dataSize = size0;
    for (int64_t i = 0; i < dataSize; i++) {
        float fvalue = bfloat16_to_float(aligned[i]);
        printf("%f\t", fvalue);
    }
    printf("\n");
}

extern "C" void printMsg0() {
    printf("Hello world!!\n");
}

extern "C" void printMsgOldVerify() {
    printf("Print GPU results from old verify function (f16->f32):\n");
}

extern "C" void mcpuPrintF32(float f1, float f2) {
  printf("Values: %.10f, %.10f\n", f1, f2);
}

extern "C" void mcpuPrint4F32(float f1, float f2, float f3, float f4) {
    printf("%.10f %.10f %.10f %.10f\n", f1, f2, f3, f4);
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

// Return the inteval of the given val as a half precsion float number
float halfFloatInterval(float val)
{
    float powerTwo[41];
    //int en = 0;
    for (int i = 0; i < 41; ++i){
        powerTwo[i] = pow(2.0f, static_cast<float>(i-24));
        //printf("2^%d = %.10f   ", i-24, powerTwo[i]);
        //en ++;
        //if (en % 3 == 0)
        //    printf("\n");
    }
    //printf("\n");
    float absVal = fabs(val);
    if (absVal <= powerTwo[11]) // 2^-13
        return powerTwo[0]; // 2^-24
    for (int exp = -13; exp <= 15; ++exp){ // 2^-13 ~ 2^15 (32768)
        if ((powerTwo[exp+24] < absVal) && (absVal <= powerTwo[exp+24+1])){
            return powerTwo[exp+14];
        }
    }
    return 0.0f; // absVal = inf
    //return std::pow(2.0f, -10.0f);
}

// Compare the results between gpu kernel (f32) and cpu validation (f32)
//
extern "C" void mcpuVerify5DFloatFloat(float *gpuAllocated,
                                      float *gpuAligned, int64_t gpuOffset,
                                      int64_t gpuSize0, int64_t gpuSize1,
                                      int64_t gpuSize2, int64_t gpuSize3,
                                      int64_t gpuSize4, int64_t gpuStride0,
                                      int64_t gpuStride1, int64_t gpuStride2,
                                      int64_t gpuStride3, int64_t gpuStride4,
                                      float *cpuAllocated, float *cpuAligned,
                                      int64_t cpuOffset, int64_t cpuSize0,
                                      int64_t cpuSize1, int64_t cpuSize2,
                                      int64_t cpuSize3, int64_t cpuSize4,
                                      int64_t cpuStride0, int64_t cpuStride1,
                                      int64_t cpuStride2, int64_t cpuStride3,
                                      int64_t cpuStride4, float threshold) {
    assert(gpuSize0 * gpuSize1 * gpuSize2 * gpuSize3 * gpuSize4 ==
           cpuSize0 * cpuSize1 * cpuSize2 * cpuSize3 * cpuSize4);
    int64_t dataSize = cpuSize0 * cpuSize1 * cpuSize2 * cpuSize3 * cpuSize4;
    printf("gpu output size: (%ldx%ldx%ldx%ldx%ld)\n", gpuSize0, gpuSize1, gpuSize2, gpuSize3, gpuSize4);
    printf("cpu output size: (%ldx%ldx%ldx%ldx%ld)\n", cpuSize0, cpuSize1, cpuSize2, cpuSize3, cpuSize4);

    //float val = 1234.0f;
    //printf("val = %lf, interval = %.10f\n", val, halfFloatInterval(val));
    //return;

    // printf("GPU result in new verify function:\n");
    // for (int64_t h = 0 ; h < gpuSize3; ++h){
    //     for (int64_t w = 0 ; w < gpuSize4; ++w){
    //         printf("%.10f ", gpuAligned[h*gpuStride3+w*gpuStride4]);
    //     }
    //     printf("\n");
    // }

    // printf("CPU result in new verify function:\n");
    // for (int64_t h = 0 ; h < cpuSize3; ++h){
    //     for (int64_t w = 0 ; w < cpuSize4; ++w){
    //         printf("%.10f ", cpuAligned[h*cpuStride3+w*cpuStride4]);
    //     }
    //     printf("\n");
    // }

    // for (int64_t b = 0 ; b < cpuSize0; ++b)
    //     for (int64_t c = 0 ; c < cpuSize2; ++c)
    //         for (int64_t h = 0 ; h < cpuSize3; ++h){
    //             for (int64_t w = 0 ; w < cpuSize4; ++w){
    //                 float cpuVal = cpuAligned[b*cpuStride0+
    //                                           c*cpuStride2+
    //                                           h*cpuStride3+
    //                                           w*cpuStride4];
    //                 float gpuVal = gpuAligned[b*cpuStride0+
    //                                           c*cpuStride2+
    //                                           h*cpuStride3+
    //                                           w*cpuStride4];
    //                 if (cpuVal != gpuVal){
    //                     printf("[%-3ld %d %-3ld %-2ld %-2ld]\tdiff: %.10f\trelDiff: %.10f\tcpu: %lf\tgpu: %f\n",
    //                            b, 0, c, h, w, cpuVal-gpuVal, (double)(cpuVal-gpuVal)/(double)cpuVal,
    //                            cpuVal, gpuVal);
    //                 }
    //             }
    //         }

    float cpuVal, gpuVal;
    float maxAbsDiff = 0.0f;
    float maxEpsilonDiff = 0.0f; // Theoretically, it should be an int
    float thr = 1.0004e-03;
    double sumAbsDiff = 0.0;
    double maxRelDiff = 0.0;
    double maxRelDiff_oldVerifier = 0.0f;
    double sumRelDiff = 0.0;
    float maxMag = 0.0f;
    double sumDiffSq = 0.0;
    int cnt_epsilon = 0;
    int cnt_exact = 0;
    int cnt_large = 0;
    int cnt_unknown = 0;
    float maxCPU = 0.0f;
    float maxGPU = 0.0f;
    float cpuMax_old = 0.0f;
    float gpuMax_old = 0.0f;
    for (int64_t i = 0 ; i < dataSize; ++i){
        cpuVal = cpuAligned[i];
        gpuVal = gpuAligned[i];

        float maxVal = std::max(fabs(cpuVal), fabs(gpuVal));
        maxMag = std::max(maxMag, maxVal);

        if (cpuVal == gpuVal){
            cnt_exact ++;
        } else {
            float absDiff = fabs(cpuVal - gpuVal);
            float epsilonDiff = absDiff / halfFloatInterval(std::min(fabs(cpuVal), fabs(gpuVal)));
            if (epsilonDiff == 1.0f) // diff within one epsilon
                cnt_epsilon ++;
            else if (epsilonDiff >= 2.0f) // diff larger than one epsilon
                cnt_large ++;
            else {
                cnt_unknown ++; // diff between 1 and 2 and between 0 and 1 (not expected)
                printf("cpuVal = %.10f, gpuVal = %.10f, absDiff = %.10f, epsilonDiff = %f, epsilon=%.10f\n",
                       cpuVal, gpuVal, absDiff, epsilonDiff, halfFloatInterval(cpuVal));
            }
            if (epsilonDiff > maxEpsilonDiff) {
                maxCPU = cpuVal;
                maxGPU = gpuVal;
            }
            maxEpsilonDiff = std::max(maxEpsilonDiff, epsilonDiff);
            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            sumAbsDiff += static_cast<double>(absDiff);
            sumDiffSq += static_cast<double>(absDiff) * static_cast<double>(absDiff);
            // old logic for verification
            if (cpuVal != 0.0f) {
                double relDiff = static_cast<double>(absDiff) / (static_cast<double>(fabs(cpuVal)));
                maxRelDiff = std::max(maxRelDiff, relDiff);
                if (fabs(cpuVal) > thr){
                    if (relDiff > maxRelDiff_oldVerifier){
                        maxRelDiff_oldVerifier = std::max (maxRelDiff_oldVerifier, relDiff);
                        cpuMax_old = cpuVal;
                        gpuMax_old = gpuVal;
                    }
                }
                sumRelDiff += relDiff;
            }
        }
    }
    double aveAbsDiff = sumAbsDiff / static_cast<double>(dataSize);
    double aveRelDiff = sumRelDiff / static_cast<double>(dataSize);
    double err_RMS = sqrt(sumDiffSq) /
        (static_cast<double>(maxMag) * sqrt(static_cast<double>(dataSize)));
    printf("%-10ld (%d %d %d %d) %.1f (%.10f %.10f) %f %lf %.1e %.1e (%.10f %.10f) %.1e %.1e\n",
           dataSize, cnt_exact, cnt_epsilon, cnt_large, cnt_unknown, maxEpsilonDiff, maxCPU, maxGPU,
           maxAbsDiff, aveAbsDiff,
           maxRelDiff, maxRelDiff_oldVerifier, cpuMax_old, gpuMax_old,
           aveRelDiff,
           err_RMS);
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

    // printf("Info about input and filter\n");
    // printf("Input sizes: (%ldx%ldx%ldx%ldx%ld)\n",
    //        inputSizes[0], inputSizes[1], inputSizes[2], inputSizes[3], inputSizes[4]);
    // printf("filter sizes: (%ldx%ldx%ldx%ldx%ld)\n",
    //        filterSizes[0], filterSizes[1], filterSizes[2], filterSizes[3], filterSizes[4]);
    // printf("Output sizes: (%ldx%ldx%ldx%ldx%ld)\n",
    //        outputSizes[0], outputSizes[1], outputSizes[2], outputSizes[3], outputSizes[4]);

    // printf("Input image:\n");
    // int64_t in_channels = inputSizes[2];
    // int64_t in_h = inputSizes[3];
    // int64_t in_w = inputSizes[4];
    // for (int64_t h = 0; h < in_h; ++h){
    //     for (int64_t c = 0; c < in_channels; ++c){
    //         for (int64_t w = 0; w < in_w; ++w){
    //             printf("%.10lf ", (double)inputAllocated[c * inputStrides[2] +
    //                                          h * inputStrides[3] +
    //                                          w * inputStrides[4]]);
    //         }
    //         printf("   ");
    //     }
    //     printf("\n");
    // }

    // printf("Filter:\n");
    // int64_t fil_channels = filterSizes[2];
    // int64_t fil_h = filterSizes[3];
    // int64_t fil_w = filterSizes[4];
    // for (int64_t h = 0; h < fil_h; ++h){
    //     for (int64_t c = 0; c < fil_channels; ++c){
    //         for (int64_t w = 0; w < fil_w; ++w){
    //             printf("%.10lf ", (double)filterAllocated[c * filterStrides[2] +
    //                                                      h * filterStrides[3] +
    //                                                      w * filterStrides[4]]);
    //         }
    //         printf("   ");
    //     }
    //     printf("\n");
    // }

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

  // printf("Result:\n");
  // int64_t out_h = outputSizes[3];
  // int64_t out_w = outputSizes[4];
  // for (int64_t h = 0; h < out_h; ++h){
  //     //for (int64_t c = 0; c < fil_channels; ++c){
  //         for (int64_t w = 0; w < out_w; ++w){
  //             printf("%.10lf ", (double)outputAllocated[h * outputStrides[3] +
  //                                                       w * outputStrides[4]]);
  //         }
  //         printf("   ");
  //         //}
  //     printf("\n");
  // }
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
