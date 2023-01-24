//===- transformMapUtils.cpp - transform map utilities --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>

#define DEBUG_TYPE "rock-transform-map-utils"

using namespace mlir;
using namespace mlir::rock;

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

std::tuple<ArrayAttr, ArrayAttr> mlir::rock::computeOobFromTransforms(
    Builder &b, ArrayAttr transforms,
    Optional<std::tuple<ArrayAttr, ArrayAttr>> initialOob) {
  IntSet upperOobLeft, upperOobRight, lowerOobLeft, lowerOobRight;
  if (initialOob.has_value()) {
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
mlir::rock::untransform(OpBuilder &b, Value transformed, ArrayAttr existing) {
  SmallVector<Attribute> transformList;
  if (existing)
    transformList.append(existing.begin(), existing.end());
  Value ret = transformed;
  while (auto transform = dyn_cast_or_null<TransformOp>(ret.getDefiningOp())) {
    transformList.push_back(transform.getTransform());
    ret = transform.getInput();
  }
  return {ret, b.getArrayAttr(transformList)};
}

std::tuple<Value, ArrayAttr>
mlir::rock::untransform(OpBuilder &b, Value transformed,
                        ArrayRef<Attribute> existing) {
  return untransform(b, transformed, b.getArrayAttr(existing));
}

TransformOp mlir::rock::reshapeBuffer(OpBuilder &b, Location loc, Value buffer,
                                      ArrayRef<StringRef> names,
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
  auto ret = b.create<TransformOp>(loc, buffer, transformAttr);
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
/// The alignment field tracks what we know about how the values in a particular
/// dimension will be aligned, with the initial assiumption being that the
/// input dimension is aligned to the maximum vectorization length. This
/// information is important in ensuring that a pad() that would otherwise
/// vectorize doesn't have reads that go partly into the padding.
struct VectorizationInfo {
  int64_t maxLength = 0, needsCoefficient = 0, alignment = 0;

  VectorizationInfo(int64_t maxLength, int64_t needsCoefficient,
                    int64_t alignment)
      : maxLength(maxLength), needsCoefficient(needsCoefficient),
        alignment(alignment) {}

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const VectorizationInfo &info) {
    return os << info.maxLength << "@" << info.needsCoefficient << " align("
              << info.alignment << ")";
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
      if (data[i].has_value())
        LLVM_DEBUG(llvm::dbgs() << *data[i]);
      else
        LLVM_DEBUG(llvm::dbgs() << "?@? align(?)");
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
  Optional<int64_t> previousAlign;
  for (auto pair : dimAndLength) {
    uint32_t upperDim = std::get<0>(pair);
    int64_t dimLength = std::get<1>(pair);
    if (input[upperDim].has_value()) {
      VectorizationInfo upperInfo = *input[upperDim];
      if (!result.has_value())
        result = VectorizationInfo(1, upperInfo.needsCoefficient, 1);
      // Catch
      // 1. Dimensions merged out of order and then
      // 2. Previous dimensions having incomplete vector lengths
      if (upperInfo.needsCoefficient == previousDimsStride &&
          (result->maxLength * result->needsCoefficient) ==
              previousDimsStride) {
        result->maxLength *= upperInfo.maxLength;
        if (!previousAlign.has_value())
          previousAlign = upperInfo.alignment * previousDimsStride;
        else
          previousAlign = math_util::gcd(
              *previousAlign, upperInfo.alignment * previousDimsStride);
      } else {
        break;
      }
    } else {
      // Dimensions held constant affect alignment
      if (!previousAlign.has_value())
        previousAlign = previousDimsStride;
      else
        previousAlign = math_util::gcd(*previousAlign, previousDimsStride);
    }
    previousDimsStride *= dimLength;
  }
  if (result.has_value())
    result->alignment = previousAlign.value_or(1);
  return result;
}

// Key data structures for vectorization analysis
struct DimInfo {
  TransformMapAttr transformMap; // Map to which the dimension belongs
  TransformAttr transform;       // Transform to which the dimension belongs
  size_t positionInMerge; // Position of the dimension inside the transform
  bool isBroadcast;       // Is this a broadcast dimension?

  // Utility function to return a transform pair <map,transform>
  std::pair<TransformMapAttr, TransformAttr> transformPair() {
    return std::make_pair(transformMap, transform);
  }
};
using DimToMergeMap = llvm::SmallDenseMap<uint32_t, DimInfo>;

using ContiguousMergesMap =
    llvm::DenseMap<std::pair<TransformMapAttr, TransformAttr>,
                   llvm::EquivalenceClasses<uint32_t>>;

void findCountiguousGroupsUnmerge(const ArrayRef<uint32_t> upperDims,
                                  const ArrayRef<int64_t> params,
                                  DimToMergeMap &dimToMerge,
                                  ContiguousMergesMap &contiguousGroups) {

  size_t i = 0;
  while (i < upperDims.size()) {
    // The i-th dimension (i.e., dimension upperDims[i])
    // uses params[i]

    auto keyI = dimToMerge[upperDims[i]];
    if (!keyI.transform) {
      i++;
      continue;
    }

    SmallVector<uint32_t> groupCandidate;
    SmallVector<size_t> dimPosition;
    for (size_t j = i; j < upperDims.size(); j++) {

      // Unit lengths don't affect the mergeability logic
      if (params[j] == 1)
        continue;

      auto keyJ = dimToMerge[upperDims[j]];

      if (!keyJ.transform) {
        break;
      }
      auto mergeJParams = keyJ.transform.getParams();
      auto mergeJDims = keyJ.transform.getLowerDims();
      size_t posJ = keyJ.positionInMerge;

      // a) Dimensions need to come from the same merge
      // b) Unmerge parameters need to match merge parameters
      if (keyJ.transformPair() != keyI.transformPair() ||
          params[j] != mergeJParams[posJ]) {
        break;
      }

      groupCandidate.push_back(mergeJDims[posJ]);
      dimPosition.push_back(posJ);
    }

    i += std::max(size_t(1), groupCandidate.size());

    // Update the result with the current group for the mergePair key
    if (groupCandidate.size() > 1 &&
        std::is_sorted(dimPosition.begin(), dimPosition.end())) {

      uint32_t lowerDim = groupCandidate.back();
      for (auto d : groupCandidate)
        contiguousGroups[keyI.transformPair()].unionSets(lowerDim, d);

      // We also want to add the singleton/broadcast dimensions of the merge the
      // groupCandidate belongs to. In this way we cover for situations like
      // - Merge{8,1,3}, <AddDim at [1]>, <Unmerge{8,3}>
      // - Merge{8,2,3}, <Broadcast at [1]>, <Unmerge{8,1,3}>
      for (auto pair : llvm::zip(keyI.transform.getLowerDims(),
                                 keyI.transform.getParams())) {
        int64_t d = std::get<0>(pair);
        int64_t p = std::get<1>(pair);
        if (p == 1 || dimToMerge[d].isBroadcast)
          contiguousGroups[keyI.transformPair()].unionSets(lowerDim, d);
      }
    }
  }
}

// This is an analysis pass to group the lower dimensions of a Merge
// transformation into contiguous groups. E.g., if we have a Merge{8,8,3} [0] ->
// [0,2,3] and we know that [2,3] are contiguous in the final representation, we
// can split the merge group in
// [[0] [2,3]]. This information will be used by the vectorizer. E.g., the
// vectorizer can fulfill a vectorization by 4, since 8*3=24 is a multiple of 4.
// In other words, every group of dimensions is treated as a single group
ContiguousMergesMap findContiguousGroups(ArrayAttr transforms,
                                         ArrayRef<int64_t> outputShape) {
  // Transform table. Will be overwritten after processing each transform_map
  DimToMergeMap dimToMerge;
  ContiguousMergesMap contiguousGroups;

  for (TransformMapAttr transformMap :
       transforms.getAsRange<TransformMapAttr>()) {
    ArrayRef<int64_t> upperBounds = transformMap.getUpperBounds();
    DimToMergeMap thisDimToMerge;

    for (TransformAttr transform : transformMap.getOps()) {
      TransformType transformType = transform.getType();

      ArrayRef<uint32_t> lowerDims = transform.getLowerDims();
      ArrayRef<uint32_t> upperDims = transform.getUpperDims();
      ArrayRef<int64_t> params = transform.getParams();

      switch (transformType) {
      case TransformType::Unfold:
      case TransformType::Merge:
        for (size_t i = 0; i < lowerDims.size(); i++) {
          thisDimToMerge[lowerDims[i]] = {transformMap, transform, i, false};
        }
        break;
      case TransformType::AddDim:
        break;
      case TransformType::Embed: {
        // Sort the parameters
        auto &&zip = llvm::zip(params, upperDims);
        SmallVector<std::tuple<int64_t, uint32_t>> data(zip.begin(), zip.end());
        std::sort(data.begin(), data.end());
        bool maybeUnmerge = true;

        // Verify that the Embed is an Unmerge operation and
        // at the same time create the sorted unmerge params
        if (!std::all_of(params.begin(), params.end(),
                         [](auto p) { return p > 0; })) {
          break;
        }

        SmallVector<int64_t> unmergeParams;
        SmallVector<uint32_t> unmergeDims;
        // Embed parameters should be in the form of
        // [1, l_0, l_0*l_1, ...] to be considered a valid
        // unmerge
        for (size_t i = 0; i < data.size() - 1; i++) {
          int64_t paramI = std::get<0>(data[i]);
          int64_t paramI1 = std::get<0>(data[i + 1]);
          uint32_t dim = std::get<1>(data[i]);
          int64_t unmergeParam = paramI1 / paramI;
          if (unmergeParam * paramI != paramI1) {
            maybeUnmerge = false;
            break;
          }
          unmergeParams.push_back(unmergeParam);
          unmergeDims.push_back(dim);
        }

        // We are not done yet. To be sure this is an unmerge
        // the faster parameter needs to be 1 and we should set
        // the upper bound
        auto fasterParam = std::get<0>(data[0]);
        if (maybeUnmerge && fasterParam == 1 && upperBounds.size()) {
          uint32_t slowerDim = std::get<1>(data.back());
          unmergeParams.push_back(upperBounds[slowerDim]);
          unmergeDims.push_back(slowerDim);

          // We have the unmerge dimensions/params going from the faster
          // to the slower, but they should be in the exact reverse order
          std::reverse(unmergeParams.begin(), unmergeParams.end());
          std::reverse(unmergeDims.begin(), unmergeDims.end());

          // Now we can use the parameters as unmerge parameters and see if
          // there are contiguous dimensions to be folded together.
          findCountiguousGroupsUnmerge(unmergeDims, unmergeParams, dimToMerge,
                                       contiguousGroups);
        }
        break;
      }

      case TransformType::PassThrough:
      case TransformType::Pad:
      case TransformType::Slice:
        // We only care about how these transformations shuffle
        // the dimensions
        for (auto pair : llvm::zip(upperDims, lowerDims)) {
          uint32_t u = std::get<0>(pair);
          uint32_t l = std::get<1>(pair);
          thisDimToMerge[l] = dimToMerge[u];
        }
        break;
      case TransformType::Broadcast:
        // Flag the dimensions we are broadcasting, if the broadcast
        // parameter is 1
        for (auto pair : llvm::zip(upperDims, lowerDims, params)) {
          uint32_t u = std::get<0>(pair);
          uint32_t l = std::get<1>(pair);
          uint32_t p = std::get<2>(pair);
          thisDimToMerge[l] = dimToMerge[u];
          if (p == 1)
            thisDimToMerge[l].isBroadcast = true;
        }
        break;
      case TransformType::Unmerge:
        findCountiguousGroupsUnmerge(upperDims, params, dimToMerge,
                                     contiguousGroups);
        break;
      }
    }
    dimToMerge = thisDimToMerge;
  }

  // Last global unmerge
  SmallVector<uint32_t> sortedDims(outputShape.size());
  std::iota(sortedDims.begin(), sortedDims.end(), 0);
  findCountiguousGroupsUnmerge(sortedDims, outputShape, dimToMerge,
                               contiguousGroups);
  return contiguousGroups;
}

static VectorizationData
propagateVectorizationInfo(TransformMapAttr map, const VectorizationData &input,
                           ContiguousMergesMap &contiguousMerges) {
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
    // A slice doesn't affect the vectorization length, as it just shifts around
    // the beginning coordinate, but it can change the alignment of the
    // dimension.
    case TransformType::Slice:
      for (auto data : llvm::enumerate(llvm::zip(upperDims, lowerDims))) {
        uint32_t idx = data.index();
        int64_t sliceBegin =
            params[2 * idx]; // , sliceEnd = params[2 * idx + 1];
        uint32_t upper, lower;
        std::tie(upper, lower) = data.value();
        if (input[upper].has_value()) {
          int64_t alignment =
              sliceBegin == 0
                  ? input[upper]->alignment
                  : math_util::gcd(input[upper]->alignment, sliceBegin);
          result[lower] =
              VectorizationInfo(input[upper]->maxLength,
                                input[upper]->needsCoefficient, alignment);
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
        if (upperInfo.has_value()) {
          int64_t maxUpperLen = upperInfo->maxLength;
          int64_t upperAlign = upperInfo->alignment;
          int64_t maxVectorizationLeft =
              math_util::gcd(maxUpperLen, maxUpperLen - leftPad);
          int64_t maxVectorizationRight =
              math_util::gcd(maxUpperLen, maxUpperLen - rightPad);
          int64_t lowerMaxLen =
              std::min(maxVectorizationLeft, maxVectorizationRight);
          // Padding is unique in that it imposes the requirement that
          // the padded dimension's vectorization lenght be a multiple of
          // the alignment of said dimension, to prevent reads that go partially
          // into the padding. However, this only applies if there's actual
          // padding being applied.
          if (leftPad != 0 || rightPad != 0)
            lowerMaxLen = math_util::gcd(lowerMaxLen, upperAlign);
          result[lower] = VectorizationInfo(
              lowerMaxLen, upperInfo->needsCoefficient, upperAlign);
        }
      }
      break;
    // For a broadcast, each vector must be either fully in or fully out of the
    // broadcast which means that the new maximum vectorization length is the
    // gcd of the old length and the broadcast modulus.
    case TransformType::Broadcast:
      for (auto data : llvm::zip(upperDims, lowerDims, params)) {
        uint32_t upper, lower;
        int64_t modulus;
        std::tie(upper, lower, modulus) = data;
        if (input[upper].has_value()) {
          int64_t lowerMaxLen =
              math_util::gcd(input[upper]->maxLength, modulus);
          int64_t lowerAlignment =
              math_util::gcd(input[upper]->alignment, modulus);
          result[lower] = VectorizationInfo(
              lowerMaxLen, input[upper]->needsCoefficient, lowerAlignment);
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

      // Note: with negative coefficients, we can't reliably vectorize, as
      // the subtraction can push us off the left edge and we don't necessarily
      // have an upper bound on how much will be subtracted. So, we use
      // a length of 1 in the fastest vectorizing dimension we can find and
      // call it a day. This flag is needed so that we have the cases
      // - Negative coefficient, but no vectorization info in any inputs =>
      // unknows
      // - Negative coefficient, at least one input is tracked => vectorization
      // of 1
      bool hasNegativeCoefficients = false;
      Optional<VectorizationInfo> ourResult;
      // We first compute the alignment assuming the held constant dimensions
      // don't matter, then we take the gcd of that result with the coefficients
      // on the held-constant dimensions.
      for (auto pair : data) {
        int64_t coefficient = std::get<0>(pair);
        uint32_t upperDim = std::get<1>(pair);
        if (coefficient < 0)
          hasNegativeCoefficients = true;
        if (input[upperDim].has_value()) {
          if (coefficient == 0)
            continue;

          int64_t upperLen = input[upperDim]->maxLength;
          int64_t needsCoeff = input[upperDim]->needsCoefficient;
          int64_t thisAlignment = input[upperDim]->alignment;

          if (!ourResult.has_value()) {
            ourResult =
                VectorizationInfo(1, coefficient, thisAlignment * coefficient);
            if (hasNegativeCoefficients) {
              ourResult->alignment = 1;
              break;
            }
          }
          if (coefficient == needsCoeff &&
              coefficient ==
                  (ourResult->maxLength * ourResult->needsCoefficient)) {
            ourResult->maxLength *= upperLen;
            ourResult->alignment = math_util::gcd(ourResult->alignment,
                                                  thisAlignment * coefficient);
          } else {
            break;
          }
        }
      }
      // Fix up the alignment for cases like Embed{1, 1}(?@?, 2@1), which should
      // be aligned to 1, not 2
      if (ourResult.has_value()) {
        for (auto pair : data) {
          int64_t coefficient = std::get<0>(pair);
          int64_t upperDim = std::get<1>(pair);
          if (coefficient <= 0)
            continue;
          if (input[upperDim].has_value())
            continue;
          ourResult->alignment =
              math_util::gcd(ourResult->alignment, coefficient);
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
    // Note: unfold is a promise about the fact that the merge dimensions
    // are contiguous in memory, which leads to a better vectorization.
    // We are capable of automatically detecting continuous groups of dimensions
    // so we will treat unfol as merge, but we will assert that the resulting
    // vectorization is the same of the unfold. In a future ticket, Unfold
    // should be completely removed.
    case TransformType::Unfold:
    case TransformType::Merge: {
      int64_t upperDim = upperDims[0];
      if (!input[upperDim].has_value()) {
        break;
      }
      int64_t maxLen = input[upperDim]->maxLength;
      int64_t coeff = input[upperDim]->needsCoefficient;
      int64_t align = input[upperDim]->alignment;
      int64_t stride = 1;

      // Group the lengths of the dimensions of the merge. This is basically
      // saying that if we have Merge{a,b,c,d} and c,d are in the same group
      // they should be treated a single dimension d whose length is
      // param[C]*param[D]
      auto groups = contiguousMerges[{map, transform}];
      uint32_t groupID = lowerDims.back();
      llvm::SmallDenseMap<uint32_t, int64_t> groupLengths;
      groupLengths[groupID] = 1;
      for (auto pair :
           llvm::zip(llvm::reverse(lowerDims), llvm::reverse(params))) {
        uint32_t lowerDim = std::get<0>(pair);
        int64_t lowerLen = std::get<1>(pair);

        if (groups.isEquivalent(lowerDim, groupID)) {
          groupLengths[groupID] *= lowerLen;
        } else {
          groupID = lowerDim;
          groupLengths[groupID] = lowerLen;
        }
      }

      // Use the aggregated lengths to compute the correct vectorization info
      for (auto lowerDim : llvm::reverse(lowerDims)) {
        if (!groupLengths.count(lowerDim)) {
          continue;
        }
        int64_t lowerLen = groupLengths[lowerDim];
        int64_t thisMaxLen = math_util::gcd(maxLen, lowerLen);
        int64_t thisAlignment = std::max(align / stride, 1l);
        result[lowerDim] =
            VectorizationInfo(thisMaxLen, coeff * stride, thisAlignment);
        maxLen = std::max(maxLen / lowerLen, 1L);
        stride *= lowerLen;
      }

      // If the transformation is Unfold, we should still able to treat it
      // as a Merge, automatically finding the contiguous groups. However,
      // we can use the knowledge stemming from Unfold to assert that
      // the contiguous detection actually worked
      if (transform.getType() == TransformType::Unfold) {
        int64_t lowerDimsLen = 1;
        for (int64_t length : params)
          lowerDimsLen *= length;
        int64_t unfoldMaxLen = math_util::gcd(maxLen, lowerDimsLen);
        assert(unfoldMaxLen == maxLen && "Failing to detect the unfold!");
      }
      break;
    }
    }
  }
  return result;
}

int64_t mlir::rock::getMaxVectorization(ArrayAttr transforms, uint32_t dim,
                                        int64_t len,
                                        ArrayRef<int64_t> outputShape) {
  int64_t numInitialDims = transforms.empty() ? outputShape.size()
                                              : transforms[0]
                                                    .cast<TransformMapAttr>()
                                                    .getMap()
                                                    .getValue()
                                                    .getNumInputs();
  VectorizationData data;
  auto contiguousMerges = findContiguousGroups(transforms, outputShape);
  // grow() takes the last index, not the length
  data.grow(numInitialDims);
  data[dim] = VectorizationInfo(/*maxLength=*/len, /*needsCoefficient=*/1,
                                /*alignment=*/len);
  for (auto transformMap : transforms.getAsRange<TransformMapAttr>()) {
    LLVM_DEBUG(llvm::dbgs() << "Max vectorization data: ");
    data.debugPrint();
    data = propagateVectorizationInfo(transformMap, data, contiguousMerges);
  }
  LLVM_DEBUG(llvm::dbgs() << "Final max vectorization data: ");
  data.debugPrint();

  Optional<VectorizationInfo> finalUnmerge = propagateUnmergeVectorization(
      llvm::zip(llvm::reverse(
                    llvm::iota_range<uint32_t>(0, outputShape.size(), false)),
                llvm::reverse(outputShape)),
      data);
  int64_t result = 1;
  if (finalUnmerge.has_value())
    LLVM_DEBUG(llvm::dbgs() << "Final unmerge: " << *finalUnmerge << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "Final unmerge yielded no result\n");
  if (finalUnmerge && finalUnmerge->needsCoefficient == 1)
    result = finalUnmerge->maxLength;
  // TODO(kdrewnia): Add support for tails
  result = math_util::gcd(len, result);
  return result;
}

AffineMap mlir::rock::composeTransforms(ArrayAttr transforms) {
  AffineMap result;
  for (auto attr : llvm::reverse(transforms.getAsRange<TransformMapAttr>())) {
    AffineMap map = attr.getMap().getAffineMap();
    if (result)
      result = result.compose(map);
    else
      result = map;
  }
  return result;
}

// This function will create a permutation that will permute the originalMap to
// be a MinorIdentityWithBroadcast. This is used to add a permutation later in
// the chain.
// e.g. :
// (d0, d1, d2, d4) -> (0, d1) was the map
// This function will return a permutation : [0, 3, 2, 1] s.t.
// apply it to the original map would result in
// (d0, d4, d2, d1) -> (0, d1) in effect.
static void createPermutationForMinorIdentityWithBroadcast(
    const AffineMap &originalMap, SmallVectorImpl<uint32_t> &perm) {
  for (uint32_t i = 0; i < originalMap.getNumInputs(); ++i) {
    perm.push_back(i);
  }

  llvm::SmallSet<uint32_t, 4> foundInputDims;
  for (const auto &idxAndValue : llvm::enumerate(originalMap.getResults())) {
    auto idx = idxAndValue.index();
    AffineExpr resultExpr = idxAndValue.value();
    if (resultExpr.isa<AffineDimExpr>()) {
      foundInputDims.insert(originalMap.getDimPosition(idx));
    }
  }

  for (const auto &idxAndValue : llvm::enumerate(originalMap.getResults())) {
    auto idx = idxAndValue.index();
    AffineExpr resultExpr = idxAndValue.value();
    if (resultExpr.isa<AffineDimExpr>()) {
      auto swap1 = originalMap.getDimPosition(idx);
      auto swap2 =
          originalMap.getNumInputs() - originalMap.getNumResults() + idx;
      perm[swap1] = swap2;
      // Only do swap if the output expr does not define another place for the
      // other input dim
      if (!foundInputDims.contains(swap2)) {
        perm[swap2] = swap1;
      }
    }
  }
}

static unsigned getResultPosition(AffineMap map, unsigned input) {
  for (unsigned i = 0, numResults = map.getNumResults(); i < numResults; i++)
    if (map.getDimPosition(i) == input)
      return i;
  llvm_unreachable("incorrect result request");
  return 0;
}

Value mlir::rock::insertTransposeAndBroadcastTransforms(
    OpBuilder &b, ArrayRef<int64_t> outShape, Value inp, AffineMap inpIdxMap) {
  if (!inpIdxMap.isIdentity()) {
    Location loc = inp.getLoc();
    auto inpType = inp.getType().template cast<MemRefType>();
    ArrayRef<int64_t> inpShape = inpType.getShape();

    int64_t diff = outShape.size() - inpShape.size();
    LLVM_DEBUG(llvm::dbgs() << "Reached makeBroadcast with map " << inpIdxMap
                            << " and diff = " << diff << "\n");

    // first expand/collapse the input to match the output rank
    if (diff < 0) {
      // collapse non-dim exprs
      // inp = rock.transform(inp) {[0, 1], 2, 3}
      MutableAffineMap newInpIdxMap = AffineMap::getMinorIdentityMap(
          outShape.size(), outShape.size(), b.getContext());
      uint32_t newIdx = 0;
      SmallVector<int64_t> newInpShape;
      int64_t newInpDimSize = 1;
      SmallVector<SmallVector<uint32_t>> merges;
      SmallVector<uint32_t> mergeDims;
      for (const auto &idxAndValue : llvm::enumerate(inpIdxMap.getResults())) {
        uint32_t idx = idxAndValue.index();
        newInpDimSize *= inpShape[idx];
        AffineExpr resultExpr = idxAndValue.value();
        mergeDims.push_back(idx);
        if (diff != 0 && resultExpr.isa<AffineConstantExpr>() &&
            inpShape[idx] == 1) {
          diff++;
        } else {
          newInpIdxMap.setResult(newIdx++, resultExpr);
          merges.push_back(mergeDims);
          mergeDims.clear();
          newInpShape.push_back(newInpDimSize);
          newInpDimSize = 1;
        }
      }
      if (mergeDims.size())
        merges.back().append(mergeDims);

      TopDownTMBuilder collapseTransform(b, newInpShape, loc);
      for (auto idxAndMerge : llvm::enumerate(merges)) {
        uint32_t idx = idxAndMerge.index();
        auto merge = idxAndMerge.value();
        if (merge.size() == 1) {
          collapseTransform.passThrough({merge[0]}, {idx});
        } else {
          SmallVector<SmallString<8>> mergeNames;
          SmallVector<int64_t> mergeSizes;
          SmallVector<StringRef> mergeNameRefs;
          for (auto midx : merge) {
            SmallString<8> mname(Twine("m" + Twine(midx)).str());
            mergeNames.push_back(mname);
            mergeNameRefs.push_back(mergeNames.back());
            mergeSizes.push_back(inpShape[midx]);
          }
          collapseTransform.merge(mergeNameRefs, merge,
                                  collapseTransform.startName(idx), mergeSizes);
        }
      }
      inp = b.create<TransformOp>(loc, inp, collapseTransform.get());
      auto inpType = inp.getType().template cast<MemRefType>();
      inpShape = inpType.getShape();
      inpIdxMap = newInpIdxMap.getAffineMap();
    } else if (diff > 0) {
      // map = (d0, d1, d2) -> (d1)
      assert(inpIdxMap.getNumInputs() - inpIdxMap.getNumResults() == diff);
      MutableAffineMap newInpIdxMap(b.getMultiDimIdentityMap(outShape.size()));
      BottomUpTMBuilder addDimtransform(b, inpShape, loc);
      for (uint32_t i = 0; i < outShape.size(); ++i) {
        if (inpIdxMap.isFunctionOfDim(i)) {
          // find location in results
          auto inpIdx = getResultPosition(inpIdxMap, i);
          addDimtransform.passThrough({i}, {inpIdx});
          newInpIdxMap.setResult(i, b.getAffineDimExpr(i));
        } else {
          SmallString<8> name;
          ("exp" + Twine(i)).toVector(name);
          addDimtransform.addDim(name, i, 1);
          newInpIdxMap.setResult(i, b.getAffineConstantExpr(0));
        }
      }
      inp = b.create<TransformOp>(loc, inp, addDimtransform.get());
      inpShape = inp.getType().cast<ShapedType>().getShape();
      inpIdxMap = newInpIdxMap.getAffineMap();
    }

    SmallVector<uint32_t> bcastDims;
    SmallVector<uint32_t> bcastInDims;
    SmallVector<uint32_t> passThroughInDims;
    SmallVector<uint32_t> perm;
    createPermutationForMinorIdentityWithBroadcast(inpIdxMap, perm);
    auto permMap = AffineMap::getPermutationMap(perm, b.getContext());
    inpIdxMap = inpIdxMap.compose(permMap);
    bool isIdentity = inpIdxMap.isMinorIdentityWithBroadcasting(&bcastDims);
    assert(
        isIdentity &&
        "this is guaranteed by createPermutationForMinorIdentityWithBroadcast");

    // Broadcast those dimensions that the original linalg.generic map specifies
    // are broadcast and collect their locations, accounting for the leading
    // dimensions not represented in that map but which are present in the gemm
    // coordinates
    BottomUpTMBuilder bcastTransform(b, inpShape, loc);
    bool hasBcast = false;
    for (uint32_t i = 0; i < inpShape.size(); ++i) {
      if (!llvm::is_contained(bcastDims, i)) {
        // Here the diff correspond to leading dropped dimensions when going
        // from output co-ordinates to input co-ordinates.
        assert(inpIdxMap.getDimPosition(i) == i);
        passThroughInDims.push_back(i);
        bcastTransform.passThrough({i}, {i});
      } else if (outShape[perm[i]] == 1) {
        // We can pass-through if the outshape is 1 and it is not realistically
        // a broadcast.
        passThroughInDims.push_back(i);
        bcastTransform.passThrough({i}, {i});
      } else {
        hasBcast = true;
        bcastInDims.push_back(i);
        bcastTransform.broadcast({i}, {outShape[perm[i]]});
      }
    }
    if (hasBcast) {
      inp = b.create<TransformOp>(loc, inp, bcastTransform.get());
    }

    // Permute the dimensions of the fusion argument to match those of the gemm
    // writeback by applying the inverse of the permutation that would have made
    // the original indexing map into a minor identity with broadcast. The
    // inverse of that permutation takes the gemm writeback coordinates and
    // scatters them into positions that match the non-identity indexing pattern
    // of the fusion argument.
    if (!permMap.isIdentity()) {
      BottomUpTMBuilder permtransform(
          b, inp.getType().cast<ShapedType>().getShape(), loc);
      llvm::SmallVector<uint32_t, 4> identityVec;
      for (uint32_t i = 0; i < outShape.size(); ++i) {
        identityVec.push_back(i);
      }
      permtransform.passThrough(identityVec, perm);
      inp = b.create<TransformOp>(loc, inp, permtransform.get());
    }
  }
  return inp;
}

TransformMapAttr mlir::rock::invertTransformMap(
              OpBuilder &b, mlir::rock::TransformMapAttr transformMap) {

  auto lowShape = transformMap.getLowerBounds();
  auto uppShape = transformMap.getUpperBounds();

  rock::TopDownTMBuilder transform(b, lowShape, b.getUnknownLoc());
  for (auto tattr : transformMap.getOps()) {
    switch (tattr.getType()) {
    case rock::TransformType::PassThrough:
      transform.passThrough(tattr.getUpperDims(), tattr.getLowerDims());
      break;
    case rock::TransformType::Pad:
    case rock::TransformType::Slice:
    case rock::TransformType::Embed:
    case rock::TransformType::AddDim:
    case rock::TransformType::Broadcast: // Unsupported
      return rock::TransformMapAttr();

    case rock::TransformType::Unmerge: {
      auto lowDims = tattr.getLowerDims();
      assert(lowDims.size() == 1);
      SmallVector<uint32_t> uppDims(tattr.getUpperDims());
      SmallVector<SmallString<8>> mergeNames;
      SmallVector<int64_t> mergeSizes;
      SmallVector<StringRef> mergeNameRefs;
      for (auto midx : tattr.getUpperDims()) {
        SmallString<8> mname(Twine("m" + Twine(midx)).str());
        mergeNames.push_back(mname);
        mergeNameRefs.push_back(mergeNames.back());
        mergeSizes.push_back(uppShape[midx]);
      }
      transform.merge(mergeNameRefs, tattr.getUpperDims(),
                      transform.startName(lowDims[0]), mergeSizes);
      break;
    }
    case rock::TransformType::Merge:
    case rock::TransformType::Unfold: {
      auto uppDims = tattr.getUpperDims();
      assert(uppDims.size() == 1);
      SmallVector<SmallString<8>> mergeNames;
      SmallVector<int64_t> mergeSizes;
      SmallVector<StringRef> mergeNameRefs;
      for (auto midx : tattr.getLowerDims()) {
        SmallString<8> mname(Twine("dim" + Twine(midx)).str());
        mergeNames.push_back(mname);
        mergeNameRefs.push_back(mergeNames.back());
        mergeSizes.push_back(lowShape[midx]);
      }
      transform.unmerge(transform.startName(uppDims[0]), uppDims[0],
                        mergeNameRefs, mergeSizes);
      break;
    }
    }
  }

  return transform.get();
}

////////////////////////////////////////////////////////////////////////
static int64_t lookupExact(int64_t val, ArrayRef<int64_t> dims) {
  for (int64_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == val) {
      return i;
    }
  }
  return -1;
}

// finds first combination equal to inpSize
static bool
findCombination(int64_t inpSize,
                SmallVector<std::tuple<int64_t, uint32_t>, 8> &outPairs,
                uint32_t reqLen, uint32_t start, uint32_t curLen, bool check[],
                SmallVector<uint32_t> &mergeDims) {
  if (curLen > reqLen)
    return false;
  else if (curLen == reqLen) {
    int64_t outSize = 1;
    for (size_t i = 0; i < outPairs.size(); i++) {
      if (check[i])
        outSize *= std::get<0>(outPairs[i]);
    }
    if (outSize == inpSize) {
      for (size_t i = 0; i < outPairs.size(); i++) {
        if (check[i])
          mergeDims.push_back(std::get<1>(outPairs[i]));
      }
      return true;
    }
    return false;
  }
  if (curLen > outPairs.size() || start >= 7) {
    // terminate
    return false;
  }
  check[start] = true;
  if (findCombination(inpSize, outPairs, reqLen, start + 1, curLen + 1, check,
                      mergeDims))
    return true;
  check[start] = false;
  if (findCombination(inpSize, outPairs, reqLen, start + 1, curLen, check,
                      mergeDims))
    return true;
  return false;
}

static void collectMerges(ArrayRef<int64_t> inpShape,
                          ArrayRef<int64_t> outShape,
                          SmallVector<SmallVector<uint32_t>> &merges) {
  SmallVector<int64_t> localInpShape(inpShape);
  SmallVector<std::tuple<int64_t, uint32_t>, 8> outPairs;
  for (auto &pair : llvm::enumerate(outShape))
    outPairs.push_back({pair.value(), pair.index()});

  // 0. match all exact and merge 1s from outPairs
  auto oit = outPairs.begin();
  for (size_t i = 0, e = outPairs.size(); i < e; ++i) {
    int64_t outDim = std::get<0>(*oit);
    uint32_t outIdx = std::get<1>(*oit);
    int64_t fid = lookupExact(outDim, localInpShape);
    if (fid != -1) {
      merges[fid].push_back(outIdx);
      localInpShape[fid] = -1;
      outPairs.erase(oit);
    } else {
      ++oit;
    }
  }

  // 1. look for combinations that match
  assert(outPairs.size() <= 8);
  bool check[8] = {
      false,
  };
  for (const auto &pair : llvm::enumerate(inpShape)) {
    auto inpIdx = pair.index();
    if (merges[inpIdx].empty()) {
      auto inpSize = pair.value();
      SmallVector<uint32_t> mergeDims;
      for (uint32_t i = 2; i <= outPairs.size(); ++i) {
        if (findCombination(inpSize, outPairs, i, 0, 0, check, mergeDims)) {
          // remove matches
          for (int32_t i = outPairs.size() - 1; i >= 0; --i) {
            if (check[i])
              outPairs.erase(outPairs.begin() + i);
          }
          break;
        }
      }
      assert(!mergeDims.empty());
      merges[inpIdx].append(mergeDims);
    }
  }

  // 2. the rest are 1s
  for (auto &pair : outPairs) {
    assert(std::get<0>(pair) == 1);
    size_t outIdx = std::get<1>(pair);
    uint32_t fid = std::min(outIdx, inpShape.size() - 1);
    merges[fid].push_back(outIdx);
  }
}

TransformMapAttr mlir::rock::transformExpandShape(OpBuilder &b,
                                                  ArrayRef<int64_t> inpShape,
                                                  ArrayRef<int64_t> outShape) {
  // %3 = "tosa.reshape"(%2) {new_shape = [1, 12, 12, 32]} :
  // (tensor<1x12x384xf32>) -> tensor<1x12x12x32xf32>
  //    - inpShape = [1, 12, 384]
  //    - outShape = [1, 12, 12, 32]

  SmallVector<int64_t> linpShape(inpShape);
  if (linpShape.empty())
    linpShape.push_back(1);
  SmallVector<int64_t> loutShape(outShape);
  if (loutShape.empty())
    loutShape.push_back(1);
  SmallVector<SmallVector<uint32_t>> merges(linpShape.size(), {});
  collectMerges(linpShape, loutShape, merges);

  rock::BottomUpTMBuilder transform(b, linpShape, b.getUnknownLoc());
  for (auto idxAndMerge : llvm::enumerate(merges)) {
    uint32_t idx = idxAndMerge.index();
    auto mergeDims = idxAndMerge.value();
    if (mergeDims.size() == 1) {
      transform.passThrough({mergeDims[0]}, {idx});
    } else {
      SmallVector<SmallString<8>> mergeNames;
      SmallVector<int64_t> mergeSizes;
      SmallVector<StringRef> mergeNameRefs;
      for (auto midx : mergeDims) {
        SmallString<8> mname(Twine("exp" + Twine(midx)).str());
        mergeNames.push_back(mname);
        mergeNameRefs.push_back(mergeNames.back());
        mergeSizes.push_back(loutShape[midx]);
      }
      transform.unmerge(mergeNameRefs, mergeDims, transform.startName(idx),
                        mergeSizes);
    }
  }

  return transform.get();
}

TransformMapAttr
mlir::rock::transformCollapseShape(OpBuilder &b, ArrayRef<int64_t> inpShape,
                                   ArrayRef<int64_t> outShape) {
  // %5 = "tosa.reshape"(%4) {new_shape = [12, 12, 32]} :
  // (tensor<1x12x12x32xf32>) -> tensor<12x12x32xf32>
  //    - inpShape = [1, 12, 12, 32]
  //    - outShape = [12, 12, 32]
  SmallVector<int64_t> linpShape(inpShape);
  if (linpShape.empty())
    linpShape.push_back(1);
  SmallVector<int64_t> loutShape(outShape);
  if (loutShape.empty())
    loutShape.push_back(1);

  SmallVector<SmallVector<uint32_t>> merges(loutShape.size(), {});
  collectMerges(loutShape, linpShape, merges);

  rock::TopDownTMBuilder transform(b, loutShape, b.getUnknownLoc());
  for (auto idxAndMerge : llvm::enumerate(merges)) {
    uint32_t idx = idxAndMerge.index();
    auto mergeDims = idxAndMerge.value();
    if (mergeDims.size() == 1) {
      transform.passThrough({mergeDims[0]}, {idx});
    } else {
      SmallVector<SmallString<8>> mergeNames;
      SmallVector<int64_t> mergeSizes;
      SmallVector<StringRef> mergeNameRefs;
      for (auto midx : mergeDims) {
        SmallString<8> mname(Twine("m" + Twine(midx)).str());
        mergeNames.push_back(mname);
        mergeNameRefs.push_back(mergeNames.back());
        mergeSizes.push_back(linpShape[midx]);
      }
      transform.merge(mergeNameRefs, mergeDims, transform.startName(idx),
                      mergeSizes);
    }
  }

  return transform.get();
}
