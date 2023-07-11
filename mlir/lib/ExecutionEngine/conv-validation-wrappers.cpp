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

  auto *output = static_cast<StridedMemRefType<int64_t, 5> *>(o_ptr);
  auto *outputAllocated = output->data + output->offset;

  // Extract proper tensor sizes and strides based on layouts
  std::array<int64_t, 5> filterSizes, filterStrides;
  std::array<int64_t, 5> inputSizes, inputStrides;
  std::array<int64_t, 5> outputSizes, outputStrides;

  getSizesAndStrides<int8_t, int64_t>(rank1, filter, rank2, input, rank3,
                                      output, f_layout, i_layout, o_layout,
                                      filterSizes, filterStrides, inputSizes,
                                      inputStrides, outputSizes, outputStrides);

  performConv2d<int8_t, int64_t, int64_t>(
      filterAllocated, inputAllocated, outputAllocated, filterSizes,
      filterStrides, inputSizes, inputStrides, outputSizes, outputStrides,
      stride_h, stride_w, padding_h_l, padding_h_r, padding_w_l, padding_w_r,
      dilation_h, dilation_w, xdlops);
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
