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
#include <iostream>
#include <mutex>
#include <numeric>

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>
#include <unordered_map>

extern "C" void seedRandomValues(uint32_t seed) {
  if (seed == 0)
    std::srand(time(0));
  else
    std::srand(seed);
}

extern "C" float randomIntegerValue(int16_t min, int16_t max) {
  if (min == max)
    return min;
  int16_t randVal = (std::rand() % (max - min)) + min;
  return static_cast<float>(randVal);
}

extern "C" float randomFloatValue(int16_t min, int16_t max) {
  auto minAsF = static_cast<float>(min);
  if (min == max)
    // Lower float values to prevent inf in big fp16 tests where not all sides
    // are randomized
    return minAsF * 0.1;
  return static_cast<float>((max - min) * static_cast<double>(std::rand()) /
                            static_cast<double>(RAND_MAX)) +
         minAsF;
}

size_t findIdxHistRelDiff(double relDiff, const double *BUCKET_BOUNDARIES,
                          size_t NUM_BOUNDARIES) {
  if (relDiff == 0.0)
    return 0;
  size_t i = 0;
  while (i < NUM_BOUNDARIES && relDiff > BUCKET_BOUNDARIES[i])
    i++;
  return i + 1;
}

void printDebugVerifyResults(long long dataSize, float maxAbsDiff,
                             float maxVAL_abs, float maxGPU_abs,
                             double aveAbsDiff, double maxRelDiff,
                             float maxVAL_rel, float maxGPU_rel,
                             double aveRelDiff, double err_RMS,
                             const double *BUCKET_BOUNDARIES,
                             size_t NUM_BUCKETS, int *hist_relDiff) {
  printf("Number of elements: %lld\n", dataSize);
  printf("maxAbsDiff info: maxAbsDiff = %f (valNum = %.5f, gpuNum = %.5f), "
         "average absDiff = %.1e\n",
         maxAbsDiff, maxVAL_abs, maxGPU_abs, aveAbsDiff);
  printf("maxRelDiff info: maxRelDiff = %.1e (valNum = %.10f, gpuNum = %.10f), "
         "average relDiff = %.1e\n",
         maxRelDiff, maxVAL_rel, maxGPU_rel, aveRelDiff);
  printf("RMS = %.1e\n", err_RMS);
  printf("Histogram of relDiff: \n");
  for (size_t i = 0; i < NUM_BUCKETS; ++i) {
    if (i == 0)
      printf("        relDiff = 0     ");
    else if (i == 1)
      printf("     0 < relDiff < %.0e", BUCKET_BOUNDARIES[i - 1]);
    else if (i == NUM_BUCKETS - 2) // second to the last bucket
      printf("%.0e < relDiff < inf   ", BUCKET_BOUNDARIES[i - 2]);
    else if (i == NUM_BUCKETS - 1) // last bucket
      printf("        relDiff = inf   ");
    else
      printf("%.0e < relDiff <= %.0e", BUCKET_BOUNDARIES[i - 2],
             BUCKET_BOUNDARIES[i - 1]);

    printf(": %d/%lld (%lf%%)\n", hist_relDiff[i], dataSize,
           100.0 * static_cast<double>(hist_relDiff[i]) /
               static_cast<double>(dataSize));
  }
}

enum class PrintOption : char {
  Always = 3,  // always print debug info
  Failure = 2, // print elem-wise diff + summary only if the test fails
  Summary = 1, // print summary info only if the test fails
  Off = 0      // do not print debug info
};

template <typename T>
void mcpuVerify(T *gpuResults, T *validationResults, long long dataSize,
                float thr_RMS, float thr_absDiff, float thr_relDiff,
                char printDebug) {
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
  // histogram of relDiss metric
  // bucket index --> interval:
  //     0: 0
  //     1: 0 - 1e-6
  //     2: 1e-6 - 1e-5
  //     3: 1e-5 - 1e-4
  //     4: 1e-4 - 1e-3
  //     5: 1e-3 - 1e-2
  //     6: 1e-2 - 0.1
  //     7: 0.1 - 1
  //     8: >= 1
  //     9: Inf
  constexpr size_t NUM_BOUNDARIES = 7;
  static const double BUCKET_BOUNDARIES[NUM_BOUNDARIES] = {
      1.0e-06, 1.0e-05, 1.0e-04, 1.0e-03, 1.0e-02, 0.1, 1.0};
  // 3 more buckets compared to the bucket boundaries
  // 1. relDiff = 0
  // 2. largest boundary < relDiff <= inf
  // 3. relDiff = inf
  constexpr size_t NUM_BUCKETS = NUM_BOUNDARIES + 3;
  int hist_relDiff[NUM_BUCKETS] = {0};
  // Obtain print debug info option
  PrintOption print_option = static_cast<PrintOption>(printDebug);

  for (long long i = 0; i < dataSize; ++i) {
    valNum = static_cast<float>(validationResults[i]);
    gpuNum = static_cast<float>(gpuResults[i]);
    // Update the max magnitutde value
    float maxNum = std::max(fabs(valNum), fabs(gpuNum));
    maxMag = std::max(maxMag, maxNum);

    if (valNum == gpuNum) {
      hist_relDiff[0]++;
    } else {
      // We know valNum != gpuNum. If valNum is inf, this branch will simply
      // return nan. Let's instead represent infinite with max<fp16> and let's
      // test for it
      constexpr float fp16MaxVal = 65504;
      if (std::isinf(valNum))
        valNum = (valNum > 0 ? fp16MaxVal : -fp16MaxVal);
      float absDiff = fabs(valNum - gpuNum);
      // Update maxAbsDiff and its correspinding pair of values
      if (absDiff > maxAbsDiff) {
        maxVAL_abs = valNum;
        maxGPU_abs = gpuNum;
        maxAbsDiff = absDiff;
      }
      sumAbsDiff += static_cast<double>(absDiff);
      // Update maxRelDiff only if cpuVal != 0
      double relDiff = 0.0;
      if (valNum != 0.0f) {
        relDiff =
            static_cast<double>(absDiff) / (static_cast<double>(fabs(valNum)));
        hist_relDiff[findIdxHistRelDiff(relDiff, BUCKET_BOUNDARIES,
                                        NUM_BOUNDARIES)]++;
        if (relDiff > maxRelDiff) {
          maxVAL_rel = valNum;
          maxGPU_rel = gpuNum;
          maxRelDiff = relDiff;
        }
        sumRelDiff += relDiff;
      } else {
        // relDiff = inf goes to the last bucket
        hist_relDiff[NUM_BUCKETS - 1]++;
      }
      // Accumulate square root
      sumDiffSq += static_cast<double>(absDiff) * static_cast<double>(absDiff);
      // Print out values if print mode is Always||Failure
      // and difference is larger than threshold
      if ((print_option == PrintOption::Always ||
           print_option == PrintOption::Failure) &&
          (absDiff > thr_absDiff || relDiff > thr_relDiff))
        printf("%lld: %f %f %f %lf\n", i, valNum, gpuNum, absDiff, relDiff);
    }
  }
  double aveAbsDiff = sumAbsDiff / static_cast<double>(dataSize);
  double aveRelDiff = sumRelDiff / static_cast<double>(dataSize);
  double err_RMS = sqrt(sumDiffSq) / (static_cast<double>(maxMag) *
                                      sqrt(static_cast<double>(dataSize)));
  // Check if pass based on all three metrics: RMS, maxAbsDiff, maxRelDiff
  int RMS_pass = (err_RMS <= thr_RMS) ? 1 : 0;
  int absDiff_pass = (maxAbsDiff <= thr_absDiff) ? 1 : 0;
  int relDiff_pass = (maxRelDiff <= thr_relDiff) ? 1 : 0;
  int all_pass = (RMS_pass && absDiff_pass && relDiff_pass) ? 1 : 0;
  // Verbose information about the difference
  if (print_option == PrintOption::Always ||
      ((print_option == PrintOption::Failure ||
        print_option == PrintOption::Summary) &&
       all_pass == 0))
    printDebugVerifyResults(dataSize, maxAbsDiff, maxVAL_abs, maxGPU_abs,
                            aveAbsDiff, maxRelDiff, maxVAL_rel, maxGPU_rel,
                            aveRelDiff, err_RMS, BUCKET_BOUNDARIES, NUM_BUCKETS,
                            hist_relDiff);
  printf("[%d %d %d]\n", RMS_pass, absDiff_pass, relDiff_pass);
}

// Compare the results in f32
extern "C" void mcpuVerifyFloat(float *gpuAllocated, float *gpuAligned,
                                int64_t gpuOffset, int64_t gpuSize,
                                int64_t gpuStride, float *valAllocated,
                                float *valAligned, int64_t valOffset,
                                int64_t valSize, int64_t valStride,
                                float thr_RMS, float thr_absDiff,
                                float thr_relDiff, char printDebug) {
  assert(gpuSize == valSize);
  mcpuVerify<float>(gpuAligned, valAligned, valSize, thr_RMS, thr_absDiff,
                    thr_relDiff, printDebug);
}

// Compare the results in int32
template <typename GPUTYPE, typename VALTYPE>
void mcpuVerifyInt(GPUTYPE *gpuAligned, VALTYPE *valAligned, long long dataSize,
                   char printDebug) {
  long long failure_count = 0;  // the number of incorrect elements
  long long overflow_count = 0; // the number of overflow elements
  long long maxAbsDiff = 0;
  int64_t max = std::numeric_limits<GPUTYPE>::max();
  int64_t min = std::numeric_limits<GPUTYPE>::min();
  PrintOption print_option = static_cast<PrintOption>(printDebug);
  for (long long i = 0; i < dataSize; ++i) {
    auto valNum = static_cast<long long>(valAligned[i]);
    int32_t gpuNum = gpuAligned[i];
    if (valNum > max || valNum < min) {
      overflow_count++;
      if (print_option == PrintOption::Always)
        printf("overflow at element : %lld, gpu=%d, val=%lld\n", i, gpuNum,
               valNum);
    }

    if (gpuNum != valNum) {
      failure_count++;
      long long absDiff = std::abs(valNum - gpuNum);
      if (absDiff > maxAbsDiff)
        maxAbsDiff = absDiff;

      // Print out individual failing elements if print mode is Always||Failure
      if (print_option == PrintOption::Always ||
          print_option == PrintOption::Failure) {
        printf("%lld: gpu=%d val=%lld absDiff=%lld\n", i, gpuNum, valNum,
               absDiff);
      }
    }
  }

  if (failure_count == 0) {
    if ((print_option == PrintOption::Always ||
         print_option == PrintOption::Summary) &&
        overflow_count > 0) {
      printf("Number of elements: %lld\n", dataSize);
      printf("Number of overflow elements: %lld\n", overflow_count);
    }
    printf("[1 1 1]\n");
  } else {
    if (print_option == PrintOption::Always ||
        print_option == PrintOption::Failure ||
        print_option == PrintOption::Summary) {
      printf("Number of elements: %lld\n", dataSize);
      printf("Number of incorrect elements: %lld\n", failure_count);
      printf("maxAbsDiff: %lld\n", maxAbsDiff);
      printf("Number of overflow elements: %lld\n", overflow_count);
    }
    printf("[0 0 0]");
  }
}

extern "C" void mcpuVerifyInt32Int32(int32_t *gpuAllocated, int32_t *gpuAligned,
                                     int64_t gpuOffset, int64_t gpuSize,
                                     int64_t gpuStride, int32_t *valAllocated,
                                     int32_t *valAligned, int32_t valOffset,
                                     int64_t valSize, int64_t valStride,
                                     char printDebug) {

  assert(gpuSize == valSize);
  mcpuVerifyInt<int32_t, int32_t>(gpuAligned, valAligned, valSize, printDebug);
}

extern "C" void mcpuVerifyInt32Int64(int32_t *gpuAllocated, int32_t *gpuAligned,
                                     int64_t gpuOffset, int64_t gpuSize,
                                     int64_t gpuStride, int64_t *valAllocated,
                                     int64_t *valAligned, int64_t valOffset,
                                     int64_t valSize, int64_t valStride,
                                     char printDebug) {

  assert(gpuSize == valSize);
  mcpuVerifyInt<int32_t, int64_t>(gpuAligned, valAligned, valSize, printDebug);
}

extern "C" void mcpuVerifyInt8Int64(int8_t *gpuAllocated, int8_t *gpuAligned,
                                    int64_t gpuOffset, int64_t gpuSize,
                                    int64_t gpuStride, int64_t *valAllocated,
                                    int64_t *valAligned, int64_t valOffset,
                                    int64_t valSize, int64_t valStride,
                                    char printDebug) {

  assert(gpuSize == valSize);
  mcpuVerifyInt<int8_t, int64_t>(gpuAligned, valAligned, valSize, printDebug);
}

template <typename T>
void mcpuVerifyNaive(T *gpuAligned, T *valAligned, long long dataSize,
                     char printDebug) {
  long long failure_count = 0; // the number of incorrect elements
  T maxAbsDiff = 0;

  PrintOption print_option = static_cast<PrintOption>(printDebug);
  for (int64_t i = 0; i < dataSize; ++i) {
    T valNum = valAligned[i];
    T gpuNum = gpuAligned[i];

    if (gpuNum != valNum) {
      failure_count++;
      T absDiff = std::abs(valNum - gpuNum);
      if (absDiff > maxAbsDiff)
        maxAbsDiff = absDiff;

      // Print out individual failing elements if print mode is Always||Failure
      if (print_option == PrintOption::Always ||
          print_option == PrintOption::Failure) {
        std::cout << i << ": gpu=" << gpuNum << " val=" << valNum
                  << " absDiff=" << absDiff << std::endl;
      }
    }
  }

  if (failure_count == 0) {
    printf("[1 1 1]\n");
  } else {
    if (print_option == PrintOption::Always ||
        print_option == PrintOption::Failure ||
        print_option == PrintOption::Summary) {
      printf("Number of elements: %lld\n", dataSize);
      printf("Number of incorrect elements: %lld\n", failure_count);
      std::cout << "maxAbsDiff: " << maxAbsDiff << std::endl;
    }
    printf("[0 0 0]");
  }
}

extern "C" void mcpuVerifyInt8Int8(int8_t *gpuAllocated, int8_t *gpuAligned,
                                   int64_t gpuOffset, int64_t gpuSize,
                                   int64_t gpuStride, int8_t *valAllocated,
                                   int8_t *valAligned, int32_t valOffset,
                                   int64_t valSize, int64_t valStride,
                                   char printDebug) {

  assert(gpuSize == valSize);
  mcpuVerifyNaive(gpuAligned, valAligned, valSize, printDebug);
}
