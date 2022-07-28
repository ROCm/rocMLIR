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
#include "mlir/Dialect/MIOpen/utility/math.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <numeric>
#include <utility>

#define DEBUG_TYPE "miopen-transform-map-utils"

using namespace mlir;
using namespace mlir::miopen;

using IntSet = llvm::SmallDenseSet<uint32_t>;

//===----------------------------------------------------------------------===//
// Out of bounds check computation.
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// General utilities.
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Vectorization inference.
//===----------------------------------------------------------------------===//
namespace {
/// Information about the vectorization state of a given dimension.
/// Includes the maximum number of elements that can be read in the dimension
/// without potentially having to "jump" (include padding, spill over into the
/// next part of memory, etc.), which is an optimistic guess, and the
/// coefficient that the dimension needs to be multiplied by for the dimension
/// to vectorize properly when it is embeded into a broader context.
struct VectorizationInfo {
  int64_t maxLength = 0, needsCoefficient = 0;

  VectorizationInfo(int64_t maxLength, int64_t needsCoefficient)
      : maxLength(maxLength), needsCoefficient(needsCoefficient) {}

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const VectorizationInfo &info) {
    return os << info.maxLength << "@" << info.needsCoefficient;
  }
};

/// A wrapper around the information about vectorization for various dimensions.
/// If a dimension's VectorizationInfo is `None`, that dimension is
/// assumed to be held constant.
struct VectorizationData {
  llvm::IndexedMap<Optional<VectorizationInfo>> data;
  operator llvm::IndexedMap<Optional<VectorizationInfo>> &() { return data; }

  void grow(size_t n) {
    // The underlying grow() takes the max index, not the size
    data.grow(n - 1);
  }

  Optional<VectorizationInfo> &operator[](uint32_t idx) { return data[idx]; }

  const Optional<VectorizationInfo> &operator[](uint32_t idx) const {
    return data[idx];
  }

  void debugPrint() {
    for (size_t i = 0, e = data.size(); i < e; ++i) {
      if (data[i].hasValue())
        LLVM_DEBUG(llvm::dbgs() << *data[i]);
      else
        LLVM_DEBUG(llvm::dbgs() << "?@?");
      LLVM_DEBUG(llvm::dbgs() << (i == e - 1 ? "\n" : " "));
    }
  }
};
} // end namespace

/// Determine the maximum vectorization length for unmerge-like operations,
/// including the implicit final embedding at the end of the system.
/// Takes an iterator over (length, dimension_index), which must proceed
/// from the fastest-moving dimension to the slowest. It computes the product
/// of the lengths of each dimension (including those held constant) along with
/// the product of the vectorization lengths, until these products diverge.
/// Such a divergence means that the dimension before the one under
/// consideration cannot be completely traversed using vector operations,
/// meaning that the unmerged result isn't guaranteed to be traversable with
/// unit stride (the dimension that broke things could have jumps, padding,
/// etc.).
template <typename T>
static Optional<VectorizationInfo>
propagateUnmergeVectorization(T &&dimAndLength,
                              const VectorizationData &input) {
  Optional<VectorizationInfo> result;
  int64_t previousDimsStride = 1;
  for (auto pair : dimAndLength) {
    uint32_t upperDim = std::get<0>(pair);
    int64_t dimLength = std::get<1>(pair);
    if (input[upperDim].hasValue()) {
      VectorizationInfo upperInfo = *input[upperDim];
      if (!result.hasValue())
        result = VectorizationInfo(1, upperInfo.needsCoefficient);
      // Catch
      // 1. Dimensions merged out of order and then
      // 2. Previous dimensions having incomplete vector lengths
      if (upperInfo.needsCoefficient == previousDimsStride &&
          (result->maxLength * result->needsCoefficient) ==
              previousDimsStride) {
        result->maxLength *= upperInfo.maxLength;
      } else {
        break;
      }
    }
    previousDimsStride *= dimLength;
  }
  return result;
}

static VectorizationData
propagateVectorizationInfo(TransformMapAttr map,
                           const VectorizationData &input) {
  VectorizationData result;
  result.grow(map.getMap().getValue().getNumResults());
  for (TransformAttr transform : map.getOps()) {
    ArrayRef<uint32_t> upperDims = transform.getUpperDims();
    ArrayRef<uint32_t> lowerDims = transform.getLowerDims();
    ArrayRef<int64_t> params = transform.getParams();

    switch (transform.getType()) {
    case TransformType::PassThrough:
    case TransformType::AddDim:
      for (auto pair : llvm::zip(upperDims, lowerDims)) {
        result[std::get<1>(pair)] = input[std::get<0>(pair)];
      }
      break;
    // For a slice, each vector must be either fully in or fully out of the
    // slice which means that the new maximum vectorization length is the gcd of
    // the old length and the slice length
    case TransformType::Slice:
      for (auto data : llvm::enumerate(llvm::zip(upperDims, lowerDims))) {
        uint32_t idx = data.index();
        int64_t sliceBegin = params[2 * idx], sliceEnd = params[2 * idx + 1];
        uint32_t upper, lower;
        std::tie(upper, lower) = data.value();
        if (input[upper].hasValue()) {
          int64_t maxLength =
              math_util::gcd(input[upper]->maxLength, sliceEnd - sliceBegin);
          result[lower] =
              VectorizationInfo(maxLength, input[upper]->needsCoefficient);
        }
      }
      break;
    // Padding has the same "must be all in or all out" requirements as slices
    // but on both the left and right. For example, if we had u <- Pad{2, 2}(v)
    // and v's max vectorization was 4, its new maximum must be 2, because
    // a vector of 4 would include both padded and non-padded elements on the
    // left and the right.
    case TransformType::Pad:
      for (auto data : llvm::enumerate(llvm::zip(upperDims, lowerDims))) {
        uint32_t idx = data.index();
        int64_t leftPad = params[2 * idx], rightPad = params[2 * idx + 1];
        uint32_t upper, lower;
        std::tie(upper, lower) = data.value();
        Optional<VectorizationInfo> upperInfo = input[upper];
        if (upperInfo.hasValue()) {
          int64_t maxUpperLen = upperInfo->maxLength;
          int64_t maxVectorizationLeft =
              math_util::gcd(maxUpperLen, maxUpperLen - leftPad);
          int64_t maxVectorizationRight =
              math_util::gcd(maxUpperLen, maxUpperLen - rightPad);
          int64_t lowerMaxLen =
              std::min(maxVectorizationLeft, maxVectorizationRight);
          result[lower] =
              VectorizationInfo(lowerMaxLen, upperInfo->needsCoefficient);
        }
      }
      break;
      // Broadcast has in/out requirements like slice.
    case TransformType::Broadcast:
      for (auto data : llvm::zip(upperDims, lowerDims, params)) {
        uint32_t upper, lower;
        int64_t modulus;
        std::tie(upper, lower, modulus) = data;
        if (input[upper].hasValue()) {
          int64_t lowerMaxLen =
              math_util::gcd(input[upper]->maxLength, modulus);
          result[lower] =
              VectorizationInfo(lowerMaxLen, input[upper]->needsCoefficient);
        }
      }
      break;
    // The embed rule: as we walk from smaller to larger coefficients, we
    // accumulate the vectorization coefficient by multiplying together the
    // vectorization lengths of dimensions, stopping if the accumulated length
    // ends up not equal to theat dimension's coefficient or if the coefficient
    // isn't equal to the accumulated value times the smallest coefficient seen
    // on a dimension being vectorized.
    case TransformType::Embed: {
      // Since embed coefficients can go in any order, we need them sorted
      auto &&zip = llvm::zip(params, upperDims);
      SmallVector<std::tuple<int64_t, uint32_t>> data(zip.begin(), zip.end());
      std::sort(data.begin(), data.end());

      Optional<VectorizationInfo> ourResult;
      for (auto pair : data) {
        int64_t coefficient = std::get<0>(pair);
        uint32_t upperDim = std::get<1>(pair);
        if (input[upperDim].hasValue()) {
          if (coefficient == 0)
            continue;

          int64_t upperLen = input[upperDim]->maxLength;
          int64_t needsCoeff = input[upperDim]->needsCoefficient;

          if (!ourResult.hasValue())
            ourResult = VectorizationInfo(1, coefficient);
          if (coefficient == needsCoeff &&
              coefficient ==
                  (ourResult->maxLength * ourResult->needsCoefficient)) {
            ourResult->maxLength *= upperLen;
          } else {
            break;
          }
        }
      }
      result[lowerDims[0]] = ourResult;
      break;
    }
    // Like `Embed`, but a bit simpler since we don't have to sort.
    case TransformType::Unmerge:
      result[lowerDims[0]] = propagateUnmergeVectorization(
          llvm::zip(llvm::reverse(upperDims), llvm::reverse(params)), input);
      break;
    // For merges, the input vectorization length is split among the output
    // dimensions, with a dimension getting a vectorization length equal
    // to the gcd of the remaining vectorization length and that dimension's
    // length. The remaining length gets lowered by the full lenght of the
    // output dimension so that Merge{3, 4} with a starting vectorization lenght
    // of 6 gets the results [1, 2] and not [3, 2]. While this might be
    // optimistic, if the dimesions don't get put back togther with the correct
    // coefficients, their vectorization will disappear.
    case TransformType::Merge:
    case TransformType::Unfold: {
      int64_t upperDim = upperDims[0];
      if (!input[upperDim].hasValue()) {
        break;
      }
      int64_t maxLen = input[upperDim]->maxLength;
      int64_t coeff = input[upperDim]->needsCoefficient;
      for (auto pair :
           llvm::zip(llvm::reverse(lowerDims), llvm::reverse(params))) {
        uint32_t lowerDim = std::get<0>(pair);
        int64_t lowerLen = std::get<1>(pair);
        int64_t thisMaxLen = math_util::gcd(maxLen, lowerLen);
        result[lowerDim] = VectorizationInfo(thisMaxLen, coeff);
        maxLen = std::max(maxLen / lowerLen, 1L);
        coeff *= lowerLen;
      }
      break;
    }
    }
  }
  return result;
}

int64_t mlir::miopen::getMaxVectorization(ArrayAttr transforms, uint32_t dim,
                                          int64_t len,
                                          ArrayRef<int64_t> outputShape) {
  int64_t numInitialDims = transforms.empty() ? outputShape.size()
                                              : transforms[0]
                                                    .cast<TransformMapAttr>()
                                                    .getMap()
                                                    .getValue()
                                                    .getNumInputs();
  VectorizationData data;
  // grow() takes the last index, not the length
  data.grow(numInitialDims);
  data[dim] = VectorizationInfo(len, 1);
  for (auto transformMap : transforms.getAsRange<TransformMapAttr>()) {
    LLVM_DEBUG(llvm::dbgs() << "Max vectorization data: ");
    data.debugPrint();
    data = propagateVectorizationInfo(transformMap, data);
  }
  LLVM_DEBUG(llvm::dbgs() << "Final max vectorization data: ");
  data.debugPrint();

  Optional<VectorizationInfo> finalUnmerge = propagateUnmergeVectorization(
      llvm::zip(llvm::reverse(
                    llvm::iota_range<uint32_t>(0, outputShape.size(), false)),
                llvm::reverse(outputShape)),
      data);
  int64_t result = 1;
  if (finalUnmerge && finalUnmerge->needsCoefficient == 1)
    result = finalUnmerge->maxLength;
  // TODO(kdrewnia): Add support for tails
  result = math_util::gcd(len, result);
  return result;
}
