//===- Conv2dGenerator.h - MLIR to C++ option parsing ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares mlir Conv2dGenerator class
//
//===----------------------------------------------------------------------===//

#ifndef BACKWARD_DATA_VALIDATION_H
#define BACKWARD_DATA_VALIDATION_H

#include "math.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

inline std::tuple<int64_t, int64_t, int64_t> calculateBwdDataGemmSizes(
    int64_t y, int64_t x, int64_t c, int64_t n, int64_t k, int64_t ho,
    int64_t wo, int64_t hi, int64_t wi, int strideH, int strideW, int dilationH,
    int dilationW, int leftPadH, int leftPadW, int64_t gemmId) {
  auto gcdStrideDilationH = math_util::gcd(strideH, dilationH);
  auto gcdStrideDilationW = math_util::gcd(strideW, dilationW);

  auto yTilda = strideH / gcdStrideDilationH;
  auto xTilda = strideW / gcdStrideDilationW;

  auto hTilda =
      ho + math_util::integer_divide_ceil(dilationH * (y - 1), strideH);
  auto wTilda =
      wo + math_util::integer_divide_ceil(dilationW * (x - 1), strideW);

  auto iHTildaLeft = math_util::integer_divide_floor(
      std::max(0, leftPadH - dilationH * (yTilda - 1)), strideH);
  auto iWTildaLeft = math_util::integer_divide_floor(
      std::max(0, leftPadW - dilationW * (xTilda - 1)), strideW);

  auto iHTildaRight = std::min(
      hTilda, math_util::integer_divide_ceil(leftPadH + hi - 1, strideH) + 1);
  auto iWTildaRight = std::min(
      wTilda, math_util::integer_divide_ceil(leftPadW + wi - 1, strideW) + 1);

  auto hTildaSlice = iHTildaRight - iHTildaLeft;
  auto wTildaSlice = iWTildaRight - iWTildaLeft;

  auto iYTilda = gemmId / xTilda;
  auto iXTilda = gemmId % xTilda;

  auto yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
  auto xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);

  auto gemmM = c;
  auto gemmN = n * hTildaSlice * wTildaSlice;
  auto gemmK = k * yDotSlice * xDotSlice;
  return std::make_tuple(gemmM, gemmN, gemmK);
}

inline LogicalResult isValidGridGemmXdlops(int64_t gemmM, int64_t gemmN,
                                           int64_t gemmK, int64_t waveSize) {
  if (gemmM % 16 != 0 && gemmN % 64 != 0) {
    return failure();
  }

  if ((gemmM * gemmN) % 256 == 0 && (gemmK * gemmM) % waveSize == 0 &&
      (gemmK * gemmN) % waveSize == 0 && gemmN % 16 == 0 && gemmM % 4 == 0 &&
      gemmK % 4 == 0) {
    return success();
  }
  return failure();
}

inline LogicalResult isValidGemmNonXdlops(int64_t gemmM, int64_t gemmN,
                                          int64_t gemmK) {
  if (gemmM % 32 == 0 && gemmN % 32 == 0 && gemmK % 4 == 0)
    return success();
  else
    return failure();
}

} // namespace mlir
#endif // BACKWARD_DATA_VALIDATION_H
