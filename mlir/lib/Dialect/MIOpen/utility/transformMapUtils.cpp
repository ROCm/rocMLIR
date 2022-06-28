//===- transformMapUtils.cpp - transform map utilities --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/MIOpen/utility/transformMapUtils.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"

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

std::tuple<Value, ArrayAttr>
mlir::miopen::untransform(OpBuilder &b, Value transformed, ArrayAttr existing) {
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

std::tuple<Value, ArrayAttr>
mlir::miopen::untransform(OpBuilder &b, Value transformed,
                          ArrayRef<Attribute> existing) {
  return untransform(b, transformed, b.getArrayAttr(existing));
}

std::tuple<ArrayAttr, ArrayAttr> mlir::miopen::computeOobFromTransforms(
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
  SmallVector<int32_t> leftValues(upperOobLeft.begin(), upperOobLeft.end()),
      rightValues(upperOobRight.begin(), upperOobRight.end());

  // Consisten output is nice
  std::sort(leftValues.begin(), leftValues.end());
  std::sort(rightValues.begin(), rightValues.end());

  return {b.getI32ArrayAttr(leftValues), b.getI32ArrayAttr(rightValues)};
}

TransformOp mlir::miopen::reshapeBuffer(OpBuilder &b, Location loc,
                                        Value buffer, ArrayRef<StringRef> names,
                                        ArrayRef<int64_t> shape) {
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
