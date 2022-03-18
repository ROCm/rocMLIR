//===- builderUtils.cpp - MIOpen utility functions ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::miopen;

namespace {
using IntSet = llvm::SmallDenseSet<uint32_t>;

void propagateTransformOob(TransformMapAttr transformMap,
                           const IntSet &upperLeft, const IntSet &upperRight,
                           IntSet &lowerLeft, IntSet &lowerRight) {
  for (TransformAttr transform : transformMap.getOps()) {
    ArrayRef<uint32_t> upperDims = transform.getUpperDims();
    ArrayRef<uint32_t> lowerDims = transform.getLowerDims();
    ArrayRef<int64_t> params = transform.getParams();

    switch (transform.getType()) {
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
    case TransformType::Merge: {
      uint32_t upper = upperDims[0];
      // Overflow goes to the biggest dimension
      if (upperRight.contains(upper))
        lowerRight.insert(lowerDims[0]);
      if (upperLeft.contains(upper))
        for (uint32_t lower : lowerDims)
          lowerLeft.insert(lower);
      break;
    }
    case TransformType::Unfold: {
      uint32_t upper = upperDims[0];
      // Unfold can overflow anywhere due to the lack of wraparound
      bool oobRight = upperRight.contains(upper);
      bool oobLeft = upperLeft.contains(upper);
      for (uint32_t lower : lowerDims) {
        if (oobRight)
          lowerRight.insert(lower);
        if (oobLeft)
          lowerLeft.insert(lower);
      }
      break;
    }
    }
  }
}
} // end anonymous namespace

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

std::tuple<ArrayAttr, ArrayAttr>
computeOobFromTransforms(Builder &b, ArrayAttr transforms) {
  IntSet upperOobLeft, upperOobRight, lowerOobLeft, lowerOobRight;
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

llvm::StringMap<uint32_t>
expandNamesInPlace(ArrayRef<StringRef> original,
                   const llvm::StringMap<SmallVector<StringRef, 2>> expansion) {
  uint32_t offset = 0;
  llvm::StringMap<uint32_t> ret;
  for (auto pair : llvm::enumerate(original)) {
    uint32_t origIndex = pair.index();
    StringRef origName = pair.value();
    if (expansion.count(origName) != 0) {
      for (auto newName : (*expansion.find(origName)).getValue()) {
        bool insertResult = ret.insert({newName, origIndex + offset}).second;
        assert(insertResult && "Duplicate dimension in dimension expansion");
        offset++;
      }
      offset--; // Handle extra count and dropping a dimension
    } else {
      bool insertResult = ret.insert({origName, origIndex + offset}).second;
      assert(insertResult && "Dimsion already defined by expansion");
    }
  }
  return ret;
}

llvm::StringMap<uint32_t>
expandNamesInPlace(CoordTransformsBuilder &builder,
                   const llvm::StringMap<SmallVector<StringRef, 2>> expansion) {
  SmallVector<StringRef, 8> names;
  builder.getEndNames(names);
  return expandNamesInPlace(names, expansion);
}
} // namespace miopen
} // namespace mlir
