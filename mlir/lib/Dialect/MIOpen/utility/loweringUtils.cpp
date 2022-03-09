//===- builderUtils.cpp - MIOpen utility functions ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

namespace mlir {
namespace miopen {
std::tuple<Value, ArrayAttr> untransform(OpBuilder &b, Value transformed,
                                         ArrayAttr existing) {
  SmallVector<Attribute> transformList;
  if (existing)
    transformList.append(existing.begin(), existing.end());
  Value ret = transformed;
  while (auto transform = dyn_cast_or_null<TransformOp>(ret.getDefiningOp())) {
    llvm::copy(transform.transforms(), std::back_inserter(transformList));
    ret = transform.input();
  }
  return {ret, b.getArrayAttr(transformList)};
}

SmallVector<int64_t>
populateBackwardDataGemmIds(int64_t strideHeight, int64_t strideWidth,
                            int64_t dilationHeight, int64_t dilationWidth,
                            int64_t filterHeight, int64_t filterWidth) {
  int64_t gcdStrideDilationH = math_util::gcd(strideHeight, dilationHeight);
  int64_t gcdStrideDilationW = math_util::gcd(strideWidth, dilationWidth);

  int64_t yTilda = strideHeight / gcdStrideDilationH;
  int64_t xTilda = strideWidth / gcdStrideDilationW;

  int64_t y = filterHeight;
  int64_t x = filterWidth;

  // Heuristic to determine if every pixel in the output would be written by the
  // backward data convolution algorithm.
  auto isEveryPixelWritten = [&]() -> bool {
    bool result = false;
    for (int32_t dim = 0; dim < 2; ++dim) {
      int64_t convStride = (dim == 0) ? strideHeight : strideWidth;
      int64_t convDilation = (dim == 0) ? dilationHeight : dilationWidth;
      int64_t filterSize = (dim == 0) ? filterHeight : filterWidth;

      if (!(convDilation == 1 && convStride <= filterSize))
        result = false;
    }
    return result;
  };
  bool needZeroInitKernel = !isEveryPixelWritten();

  llvm::SmallVector<int64_t> gemmIds;
  if (needZeroInitKernel)
    gemmIds.push_back(-1);

  // Populate the gemm IDs according to the current backward data convolution
  // algorithm implementation.
  for (int64_t gemmId = 0; gemmId < yTilda * xTilda; ++gemmId) {
    // gemmK size is different for each GEMM
    const int64_t iYTilda = gemmId / xTilda;
    const int64_t iXTilda = gemmId % xTilda;

    int64_t yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
    int64_t xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);
    // gemmK must > 0, otherwise not need to run
    if (yDotSlice * xDotSlice > 0) {
      gemmIds.push_back(gemmId);
    }
  }
  return gemmIds;
}

} // namespace miopen
} // namespace mlir
