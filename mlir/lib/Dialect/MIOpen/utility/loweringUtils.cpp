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
      uint32_t lower = lowerDims[0];
      for (auto pair : llvm::zip(upperDims, params)) {
        uint32_t upper = std::get<0>(pair);
        int64_t coeff = std::get<1>(pair);
        if (coeff == 0)
          continue;
        bool negative = coeff < 0;
        bool positive = coeff > 0;
        bool isLeft = upperLeft.contains(upper);
        bool isRight = upperRight.contains(upper);
        if ((isLeft && positive) || (isRight && negative))
          shallLeft = true;
        if ((isRight && positive) || (isLeft && negative))
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

std::tuple<DenseIntElementsAttr, DenseIntElementsAttr>
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

  return {b.getI32VectorAttr(leftValues), b.getI32VectorAttr(rightValues)};
}
} // namespace miopen
} // namespace mlir
