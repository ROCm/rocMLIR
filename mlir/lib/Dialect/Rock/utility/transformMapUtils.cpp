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
using MergePairKey = std::pair<TransformMapAttr, TransformAttr>;

using DimToMergeMap =
    llvm::DenseMap<uint32_t, std::pair<TransformMapAttr, TransformAttr>>;

using DimRemapMap = llvm::DenseMap<uint32_t, uint32_t>;

using ContiguousMergesMap =
    llvm::DenseMap<MergePairKey, SmallVector<SmallVector<uint32_t>>>;

// Utility data structure to save information about a specific merge
struct MergeInfo {
  MergeInfo() {}
  MergeInfo(MergePairKey mergePair) {
    auto mergeLowerDims = mergePair.second.getLowerDims();
    for (auto pair : llvm::zip(mergeLowerDims, mergePair.second.getParams())) {
      paramMap[std::get<0>(pair)] = std::get<1>(pair);
    }

    for (size_t j = 0; j < mergeLowerDims.size(); j++) {
      dimPosition[mergeLowerDims[j]] = j;
    }
  }

  // Given a dimension this map tells us what is the parameter
  // associated to that dimension
  llvm::DenseMap<uint32_t, int64_t> paramMap;
  // Given a dimension, this map tells us what is its position
  // in the dimension vector (e.g., if mergeDims = [0,2,3], dimPosition(2)=1)
  llvm::DenseMap<uint32_t, int64_t> dimPosition;
};

// Given dimensions and parameters of an unmerge, scan the merges
// to flag group of dimensions that are known to be contiguous
// in the unmerge
void scanForContiguousDimensions(
    ArrayRef<uint32_t> unmergeDimsMaybe, ArrayRef<int64_t> unmergeParams,
    DimRemapMap &dimRemap, DenseSet<uint32_t> &dimToDelete,
    DimToMergeMap &dimToMerge, DenseMap<MergePairKey, MergeInfo> &mergeInfoMap,
    ContiguousMergesMap &result) {

  // Shrink the dimension space if some dimensions are not real
  SmallVector<size_t> realIndices;
  size_t inc = 0;
  for (size_t i = 0; i < unmergeDimsMaybe.size(); i++) {
    while (dimToDelete.contains(i + inc)) {
      inc++;
    }
    realIndices.push_back(i + inc);
  }

  SmallVector<uint32_t> unmergeDims;
  for (size_t i = 0; i < unmergeDimsMaybe.size(); i++) {
    unmergeDims.push_back(realIndices[unmergeDimsMaybe[i]]);
  }

  size_t i = 0;
  // Analyze the unmerge dimensions, one by one
  while (i < unmergeDims.size()) {
    auto unmergeDimI = dimRemap[i];

    // Get the mergePair
    if (!dimToMerge.count(unmergeDimI)) {
      i++;
      continue;
    }
    auto mergePair = dimToMerge[unmergeDimI];

    auto &paramMatcher = mergeInfoMap[mergePair].paramMap;
    auto &reshuffler = mergeInfoMap[mergePair].dimPosition;
    SmallVector<uint32_t> outputDims;

    // Keep adding dimensions until either:
    // - they don't belong to the same merge anymore
    // - their parameters don't match
    // - they are unmerged in the wrong order
    // - skip over singleton dimensions
    for (size_t j = i; j < unmergeDims.size(); j++) {
      // Protect against broadcasts
      if (unmergeParams[j] == 1) {
        continue;
      }

      auto it = dimToMerge.find(dimRemap[j]);
      if (it == dimToMerge.end() || it->second != mergePair ||
          (paramMatcher[dimRemap[j]] != unmergeParams[j]) ||
          dimRemap[j] != unmergeDims[j]) {
        break;
      }

      outputDims.push_back(dimRemap[j]);
    }
    std::sort(outputDims.begin(), outputDims.end(),
              [&](auto a, auto b) { return reshuffler[a] < reshuffler[b]; });

    // Now we know outputDims contain a group of contiguous dimensions
    i += std::max(size_t(1), outputDims.size());

    // Update the result with the current group for the mergePair key
    if (outputDims.size() > 1) {
      for (auto d : outputDims) {
        dimToMerge.erase(d);
      }
      result[mergePair].push_back(outputDims);
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
ContiguousMergesMap findContiguousMerges(ArrayAttr transforms,
                                         ArrayRef<int64_t> outputShape) {
  DimToMergeMap dimToMerge;
  DimRemapMap dimensionMap;
  DenseMap<MergePairKey, MergeInfo> mergeInfoMap;
  DenseSet<uint32_t> dimToDelete;
  ContiguousMergesMap result;

  for (TransformMapAttr transformMap :
       transforms.getAsRange<TransformMapAttr>()) {
    auto upperBounds = transformMap.getUpperBounds();
    for (TransformAttr transform : transformMap.getOps()) {
      auto transformType = transform.getType();
      ArrayRef<uint32_t> lowerDims = transform.getLowerDims();
      ArrayRef<uint32_t> upperDims = transform.getUpperDims();
      ArrayRef<int64_t> params = transform.getParams();

      switch (transformType) {
      case TransformType::Merge:
        for (auto d : lowerDims) {
          auto key = std::make_pair(transformMap, transform);
          dimToMerge[d] = key;
          mergeInfoMap[key] = MergeInfo(key);
          dimensionMap[d] = d;
        }
        break;
      case TransformType::AddDim:
        dimToDelete.insert(upperDims[0]);
        break;
      case TransformType::Embed: {
        // Sort the parameters
        auto &&zip = llvm::zip(params, upperDims);
        SmallVector<std::tuple<int64_t, uint32_t>> data(zip.begin(), zip.end());
        std::sort(data.begin(), data.end());

        // Verify that the Embed is an Unmerge operation and
        // at the same time create the sorted unmerge params
        bool maybeUnmerge = true;
        SmallVector<int64_t> unmergeParams;
        SmallVector<uint32_t> unmergeDims;
        for (size_t i = 0; i < data.size() - 1; i++) {
          auto paramI = std::get<0>(data[i]);
          auto paramI1 = std::get<0>(data[i + 1]);
          auto dim = std::get<1>(data[i]);
          auto unmergeParam = paramI1 / paramI;
          if (unmergeParam * paramI != paramI1) {
            maybeUnmerge = false;
            break;
          }
          unmergeParams.push_back(unmergeParam);
          unmergeDims.push_back(dim);
        }

        // We are not done yet. To be sure this is an unmerge
        // the faster parameter needs to be 1 and the upper bounds
        // have to be set
        auto fasterParam = std::get<0>(data[0]);
        if (maybeUnmerge && fasterParam == 1 && upperBounds.size()) {
          auto slowerDim = std::get<1>(data.back());
          unmergeParams.push_back(upperBounds[slowerDim]);
          unmergeDims.push_back(slowerDim);

          // Shuffle the sorted unmerge params
          auto &&zip = llvm::zip(unmergeDims, unmergeParams);
          SmallVector<std::tuple<uint32_t, int64_t>> data(zip.begin(),
                                                          zip.end());
          std::sort(data.begin(), data.end());

          SmallVector<int64_t> unmergeParamsReshuffled;
          for (auto pair : data) {
            unmergeParamsReshuffled.push_back(std::get<1>(pair));
          }

          // Now we can use the parameters as unmerge parameters and see if
          // there are contiguous dimensions to be folded together.
          scanForContiguousDimensions(upperDims, unmergeParamsReshuffled,
                                      dimensionMap, dimToDelete, dimToMerge,
                                      mergeInfoMap, result);
        }
        break;
      }

      case TransformType::PassThrough:
      case TransformType::Pad:
      case TransformType::Broadcast:
        // Map the lower dimensions to the dimensions originally
        // used (i.e., upper dimensions)
        for (auto pair : llvm::zip(upperDims, lowerDims)) {
          auto u = std::get<0>(pair);
          auto l = std::get<1>(pair);
          dimensionMap[l] = u;
        }
        break;
      case TransformType::Slice:
      case TransformType::Unfold: // TODO: remove Unfold
        break;
      case TransformType::Unmerge:
        scanForContiguousDimensions(upperDims, params, dimensionMap,
                                    dimToDelete, dimToMerge, mergeInfoMap,
                                    result);
      }
    }
  }

  // Last global unmerge
  SmallVector<uint32_t> outputDims(outputShape.size());
  std::iota(outputDims.begin(), outputDims.end(), 0);
  scanForContiguousDimensions(outputDims, outputShape, dimensionMap,
                              dimToDelete, dimToMerge, mergeInfoMap, result);

  // Add singleton dimensions
  for (auto d : dimToMerge) {
    result[d.second].push_back({uint32_t(d.first)});
  }

  // Sort the groups (if we have [[3], [0,2]], the result should be [[0,2],[3]])
  for (auto &pair : result) {
    auto &groups = pair.second;
    std::sort(groups.begin(), groups.end(),
              [&](auto a, auto b) { return a[0] < b[0]; });
  }

  return result;
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
    case TransformType::Merge: {
      int64_t upperDim = upperDims[0];
      if (!input[upperDim].has_value()) {
        break;
      }
      auto groups = contiguousMerges[{map, transform}];
      int64_t maxLen = input[upperDim]->maxLength;
      int64_t coeff = input[upperDim]->needsCoefficient;
      int64_t align = input[upperDim]->alignment;
      int64_t stride = 1;
      size_t dim_pos = lowerDims.size() - 1;
      for (auto g : llvm::reverse(groups)) {
        int64_t lowerLen = 1;
        for (auto lowerDim : llvm::reverse(g)) {
          lowerLen *= params[dim_pos--];
        }
        uint32_t lowerDim = g.back();
        int64_t thisMaxLen = math_util::gcd(maxLen, lowerLen);
        int64_t thisAlignment = std::max(align / stride, 1l);
        result[lowerDim] =
            VectorizationInfo(thisMaxLen, coeff * stride, thisAlignment);
        maxLen = std::max(maxLen / lowerLen, 1L);
        stride *= lowerLen;
      }
      break;
    }
    // Unfold is a promise to the coordinate transforms engine that
    // the dimensions that are being "merged" are contiguous in the underlying
    // memory. When someone is able to make this promise, we take advantage
    // of it by putting all the vectorization on the fastest-moving of the
    // dimmensions.
    case TransformType::Unfold: {
      int64_t upperDim = upperDims[0];
      if (!input[upperDim].has_value()) {
        break;
      }
      int64_t maxLen = input[upperDim]->maxLength;
      int64_t coeff = input[upperDim]->needsCoefficient;
      int64_t align = input[upperDim]->alignment;
      int64_t lastLowerDim = lowerDims.back();
      int64_t lowerDimsLen = 1;
      for (int64_t length : params)
        lowerDimsLen *= length;
      int64_t resultMaxLen = math_util::gcd(maxLen, lowerDimsLen);
      int64_t resultAlignment = math_util::gcd(align, lowerDimsLen);
      result[lastLowerDim] =
          VectorizationInfo(resultMaxLen, coeff, resultAlignment);
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
  auto contiguousMerges = findContiguousMerges(transforms, outputShape);
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

Value mlir::rock::insertTransposeAndBroadcastTransforms(
    OpBuilder &b, ArrayRef<int64_t> outShape, Value inp, AffineMap inpIdxMap) {
  if (!inpIdxMap.isIdentity()) {
    Location loc = inp.getLoc();
    auto inpType = inp.getType().template cast<MemRefType>();
    ArrayRef<int64_t> inpShape = inpType.getShape();

    int64_t diff = outShape.size() - inpShape.size();
    LLVM_DEBUG(llvm::dbgs() << "Reached makeBroadcast with map " << inpIdxMap
                            << " and diff = " << diff << "\n");

    if (diff < 0) {
      // collapse non-dim exprs
      // inp = rock.transform(inp) {[0, 1], 2, 3}
      MutableAffineMap newInpIdxMap = AffineMap::getMinorIdentityMap(
          outShape.size(), outShape.size(), b.getContext());
      uint32_t newIdx = 0;
      SmallVector<SmallVector<uint32_t>> merges;
      SmallVector<uint32_t> mergeDims;
      for (const auto &idxAndValue : llvm::enumerate(inpIdxMap.getResults())) {
        uint32_t idx = idxAndValue.index();
        AffineExpr resultExpr = idxAndValue.value();
        mergeDims.push_back(idx);
        if (diff != 0 && resultExpr.isa<AffineConstantExpr>() &&
            inpShape[idx] == 1) {
          diff++;
        } else {
          newInpIdxMap.setResult(newIdx++, resultExpr);
          merges.push_back(mergeDims);
          mergeDims.clear();
        }
      }
      if (mergeDims.size())
        merges.back().append(mergeDims);

      TopDownTMBuilder collapseTransform(b, outShape, loc);
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
        assert(inpIdxMap.getDimPosition(i) == diff + i);
        passThroughInDims.push_back(diff + i);
        bcastTransform.passThrough({i}, {i});
      } else if (outShape[perm[diff + i]] == 1) {
        // We can pass-through if the outshape is 1 and it is not realistically
        // a broadcast.
        passThroughInDims.push_back(diff + i);
        bcastTransform.passThrough({i}, {i});
      } else {
        hasBcast = true;
        bcastInDims.push_back(diff + i);
        bcastTransform.broadcast({i}, {outShape[perm[diff + i]]});
      }
    }
    if (hasBcast) {
      inp = b.create<TransformOp>(loc, inp, bcastTransform.get());
    }

    // Then, add dimensions that are present in the writeback coordinates but
    // are not present in the additional fusion argument with matching sizes.
    // This, combined with the previous step, ensures that the view of the
    // fusion argument has the same dimensions as the gemm output, though they
    // are not necessarily in the same order.
    bool isDimAdded = false;
    BottomUpTMBuilder addDimtransform(
        b, inp.getType().cast<ShapedType>().getShape(), loc);
    for (uint32_t i = 0; i < outShape.size(); ++i) {
      unsigned int startIdx = i - diff;
      if (llvm::is_contained(bcastInDims, i)) {
        addDimtransform.passThrough({i}, {startIdx});
      } else if (llvm::is_contained(passThroughInDims, i)) {
        addDimtransform.passThrough({i}, {startIdx});
      } else {
        isDimAdded = true;
        SmallString<8> name;
        ("exp" + Twine(i)).toVector(name);
        addDimtransform.addDim(name, i, outShape[perm[i]]);
      }
    }
    if (isDimAdded) {
      inp = b.create<TransformOp>(loc, inp, addDimtransform.get());
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
