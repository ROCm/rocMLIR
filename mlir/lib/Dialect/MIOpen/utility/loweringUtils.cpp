//===- builderUtils.cpp - MIOpen utility functions ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::miopen;

using IntSet = llvm::SmallDenseSet<uint32_t>;

static void propagateTransformOob(TransformMapAttr transformMap,
                                  const IntSet &upperLeft,
                                  const IntSet &upperRight, IntSet &lowerLeft,
                                  IntSet &lowerRight) {
  for (TransformAttr transform : transformMap.getOps()) {
    ArrayRef<uint32_t> upperDims = transform.getUpperDims();
    ArrayRef<uint32_t> lowerDims = transform.getLowerDims();
    ArrayRef<int64_t> params = transform.getParams();

    switch (transform.getType()) {
    case TransformType::Broadcast: {
      // Broadcast makes non-negative indices in-bounds, only check left bounds
      for (auto pair : llvm::zip(upperDims, lowerDims)) {
        uint32_t upper = std::get<0>(pair);
        uint32_t lower = std::get<1>(pair);
        if (upperLeft.contains(upper))
          lowerLeft.insert(lower);
      }
      break;
    }
    case TransformType::PassThrough:
    case TransformType::Slice:
    case TransformType::AddDim: {
      // Zip ends at end of shortes array, allowing addDim here
      for (auto pair : llvm::zip(upperDims, lowerDims)) {
        uint32_t upper = std::get<0>(pair);
        uint32_t lower = std::get<1>(pair);
        if (upperLeft.contains(upper))
          lowerLeft.insert(lower);
        if (upperRight.contains(upper))
          lowerRight.insert(lower);
      }
      break;
    }
    case TransformType::Pad: {
      for (uint32_t i = 0, e = upperDims.size(); i < e; ++i) {
        uint32_t upper = upperDims[i];
        uint32_t lower = lowerDims[i];
        if (upperLeft.contains(upper) || params[2 * i] > 0)
          lowerLeft.insert(lower);
        if (upperRight.contains(upper) || params[2 * i + 1] > 0)
          lowerRight.insert(lower);
      }
      break;
    }
    case TransformType::Embed: {
      bool shallLeft = false;
      bool shallRight = false;
      ArrayRef<int64_t> upperBounds = transformMap.getUpperBounds();
      uint32_t lower = lowerDims[0];
      int64_t lowerBound = transformMap.getLowerBounds()[lower];
      for (auto pair : llvm::zip(upperDims, params)) {
        uint32_t upper = std::get<0>(pair);
        int64_t coeff = std::get<1>(pair);
        if (coeff == 0)
          continue;
        bool negative = coeff < 0;
        bool isLeft = upperLeft.contains(upper);
        bool isRight = upperRight.contains(upper);
        if (negative) {
          // Pessimistically, substraction always risks underflow
          shallLeft = true;
          // However, the risk of overflow from the subtraction itself occurs
          // only if the negative-coefficient argument could have been negative
          // itself
          if (isLeft)
            shallRight = true;
        } else {
          shallLeft |= isLeft;
          shallRight |= isRight;
        }

        // If the max of a dimension times its coefficient can overshoot
        // the maximum size of the output, check bounds on the right
        if (coeff * upperBounds[upper] > lowerBound)
          shallRight = true;
      }

      if (shallLeft)
        lowerLeft.insert(lower);
      if (shallRight)
        lowerRight.insert(lower);
      break;
    }
    case TransformType::Unmerge: {
      uint32_t lower = lowerDims[0];
      bool shallLeft = false;
      bool shallRight = false;
      for (uint32_t upper : upperDims) {
        shallLeft |= upperLeft.contains(upper);
        shallRight |= upperRight.contains(upper);
      }
      if (shallLeft)
        lowerLeft.insert(lower);
      if (shallRight)
        lowerRight.insert(lower);
      break;
    }
    case TransformType::Merge:
    case TransformType::Unfold: {
      uint32_t upper = upperDims[0];
      // Overflow goes to the biggest dimension. Unfold doesn't to carry checks,
      // but the incoming index diffs (if applicable) are spread out to their
      // respective coordinates before being added, so something that causes
      // oob on the right will be assigned to lowerDims[0], since the point
      // just to the right of the in-bounds region has 0 in the coordinates
      // that aren't first.
      if (upperRight.contains(upper))
        lowerRight.insert(lowerDims[0]);
      if (upperLeft.contains(upper)) {
        assert(transform.getType() != TransformType::Unfold &&
               "Can't corently bounds-check unfold from the left");
        for (uint32_t lower : lowerDims)
          lowerLeft.insert(lower);
      }
      break;
    }
    }
  }
}

namespace mlir {
namespace miopen {
LogicalResult calculateKBlockNum(ConvolutionDims convDims, int64_t MPerBlock,
                                 int64_t NPerBlock, int64_t KPerBlock,
                                 int64_t KPack, int64_t num_cu,
                                 int64_t &nKBlock) {
  const int64_t gemmM = convDims.k;
  const int64_t gemmN = convDims.c * convDims.y * convDims.x;
  const int64_t gemmK = convDims.n * convDims.ho * convDims.wo;

  int64_t gemmKBlock = 1;

  if ((gemmM % MPerBlock != 0) || (gemmN % NPerBlock != 0) ||
      (gemmK % (KPerBlock * KPack) != 0))
    return failure();

  const int64_t gridSize =
      convDims.g * (gemmM / MPerBlock) * (gemmN / NPerBlock);
  const int64_t maxGridSize = 20 * num_cu;

  gemmKBlock = std::max(maxGridSize / gridSize, static_cast<int64_t>(1));
  gemmKBlock = std::min(gemmKBlock, convDims.n);

  for (; gemmKBlock > 1; --gemmKBlock) {
    if (convDims.n % gemmKBlock != 0)
      continue;

    if (gemmK % (gemmKBlock * KPerBlock * KPack) != 0)
      continue;

    break;
  }
  // not more than n
  gemmKBlock = std::min(convDims.n, gemmKBlock);
  // not less than 1
  gemmKBlock = std::max((__int64_t)1, gemmKBlock);

  nKBlock = gemmKBlock;
  return success();
}

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

std::tuple<Value, ArrayAttr> untransform(OpBuilder &b, Value transformed,
                                         ArrayRef<Attribute> existing) {
  return untransform(b, transformed, b.getArrayAttr(existing));
}

std::tuple<ArrayAttr, ArrayAttr> computeOobFromTransforms(
    Builder &b, ArrayAttr transforms,
    Optional<std::tuple<ArrayAttr, ArrayAttr>> initialOob) {
  IntSet upperOobLeft, upperOobRight, lowerOobLeft, lowerOobRight;
  if (initialOob.hasValue()) {
    ArrayAttr initLeft, initRight;
    std::tie(initLeft, initRight) = *initialOob;
    for (APInt l : initLeft.getAsValueRange<IntegerAttr>())
      upperOobLeft.insert(l.getZExtValue());
    for (APInt r : initRight.getAsValueRange<IntegerAttr>())
      upperOobRight.insert(r.getZExtValue());
  }

  for (auto transformMap : transforms.getAsRange<TransformMapAttr>()) {
    propagateTransformOob(transformMap, upperOobLeft, upperOobRight,
                          lowerOobLeft, lowerOobRight);
    upperOobLeft = std::move(lowerOobLeft);
    upperOobRight = std::move(lowerOobRight);
    // Clear after move just in case
    lowerOobLeft.clear();
    lowerOobRight.clear();
  }
  llvm::SmallVector<int32_t> leftValues, rightValues;
  leftValues.reserve(upperOobLeft.size());
  rightValues.reserve(upperOobRight.size());
  llvm::copy(upperOobLeft, std::back_inserter(leftValues));
  llvm::copy(upperOobRight, std::back_inserter(rightValues));

  // Consisten output is nice
  std::sort(leftValues.begin(), leftValues.end());
  std::sort(rightValues.begin(), rightValues.end());

  return {b.getI32ArrayAttr(leftValues), b.getI32ArrayAttr(rightValues)};
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

miopen::ConvOpType obtainConvDirection(Operation *op) {
  miopen::ConvOpType opType = miopen::ConvOpType::Fwd;
  if (isa<miopen::Conv2DOp>(*op)) {
    opType = miopen::ConvOpType::Fwd;
  } else if (isa<miopen::Conv2DBwdDataOp>(*op)) {
    opType = miopen::ConvOpType::BwdData;
  } else if (isa<miopen::Conv2DBwdWeightOp>(*op)) {
    opType = miopen::ConvOpType::BwdWeight;
  }
  return opType;
}

mlir::Type obtainConvDataType(Operation *op) {
  return op->getOperand(1)
      .getType()
      .template cast<MemRefType>()
      .getElementType();
}

TransformOp reshapeBuffer(OpBuilder &b, Location loc, Value buffer,
                          ArrayRef<StringRef> names, ArrayRef<int64_t> shape) {
  MemRefType bufferType = buffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> outShape = bufferType.getShape();
  assert(outShape.size() == 1 && "Buffer being reshaped must start linear");

  SmallVector<int64_t> strides;
  strides.reserve(shape.size());
  int64_t stride = 1;
  for (int64_t v : llvm::reverse(shape)) {
    strides.push_back(stride);
    stride *= v;
  }
  std::reverse(strides.begin(), strides.end());
  assert(stride == outShape[0] && "Strides must multiply to buffer length");

  TopDownTMBuilder transform(b, names, shape, loc);
  transform.embed("raw", 0, outShape[0], names, strides);

  TransformMapAttr transformAttr = transform.get();
  auto ret = b.create<TransformOp>(loc, buffer, transformAttr,
                                   bufferType.getMemorySpaceAsInt());
  return ret;
}
} // namespace miopen
} // namespace mlir
