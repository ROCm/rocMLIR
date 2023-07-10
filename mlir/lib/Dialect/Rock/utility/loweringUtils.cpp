//===- loweringUtils.cpp - Rock utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::rock;

bool mlir::rock::isWrWAtomicKernel(GemmFeatures features, Type dataType,
                                   bool requiredPadding) {
  return isAccel(features) &&
         bitEnumContainsAll(features, GemmFeatures::atomic_add) &&
         (dataType.isF32() || dataType.isF16()) && !requiredPadding;
}

bool mlir::rock::isAccel(GemmFeatures features) {
  return bitEnumContainsAny(features, GemmFeatures::wmma | GemmFeatures::mfma);
}

LogicalResult mlir::rock::calculateKBlockNum(const int64_t batchSize,
                                             const GemmSize &gemmSize,
                                             int64_t MPerBlock,
                                             int64_t NPerBlock,
                                             int64_t KPerBlock, int64_t KPack,
                                             int64_t num_cu, int64_t &nKBlock) {
  const int64_t gemmM = gemmSize.m;
  const int64_t gemmN = gemmSize.n;
  const int64_t gemmK = gemmSize.k;

  int64_t gemmKBlock = 1;

  if ((gemmM % MPerBlock != 0) || (gemmN % NPerBlock != 0) ||
      (gemmK % (KPerBlock * KPack) != 0))
    return failure();

  const int64_t gridSize =
      gemmSize.g * (gemmM / MPerBlock) * (gemmN / NPerBlock);
  const int64_t maxGridSize = 20 * num_cu;

  gemmKBlock = std::max(maxGridSize / gridSize, static_cast<int64_t>(1));
  gemmKBlock = std::min(gemmKBlock, batchSize);

  for (; gemmKBlock > 1; --gemmKBlock) {
    if (batchSize % gemmKBlock != 0)
      continue;

    if (gemmK % (gemmKBlock * KPerBlock * KPack) != 0)
      continue;

    break;
  }
  // not more than n
  gemmKBlock = std::min(batchSize, gemmKBlock);
  // not less than 1
  gemmKBlock = std::max((int64_t)1, gemmKBlock);

  nKBlock = gemmKBlock;
  return success();
}

SmallVector<int64_t>
mlir::rock::backwardDataKernelIds(int64_t strideHeight, int64_t strideWidth,
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
    bool result = true;
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

  llvm::SmallVector<int64_t> kernelIds;
  if (needZeroInitKernel)
    kernelIds.push_back(-1);

  // Populate the kernel IDs according to the current backward data convolution
  // algorithm implementation.
  for (int64_t kernelId = 0; kernelId < yTilda * xTilda; ++kernelId) {
    // gemmK size is different for each GEMM
    const int64_t iYTilda = kernelId / xTilda;
    const int64_t iXTilda = kernelId % xTilda;

    int64_t yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
    int64_t xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);
    // gemmK must > 0, otherwise not need to run
    if (yDotSlice * xDotSlice > 0) {
      kernelIds.push_back(kernelId);
    }
  }
  return kernelIds;
}

// TODO(kdrewnia): Could rank-0 vectors clear some of this up?
Type mlir::rock::vectorTypeOrSelf(Type elementType, int64_t len) {
  if (len == 1)
    return elementType;
  return VectorType::get({len}, elementType);
}
