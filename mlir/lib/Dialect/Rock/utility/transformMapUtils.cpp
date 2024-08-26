//===- transformMapUtils.cpp - transform map utilities --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>

#define DEBUG_TYPE "rock-transform-map-utils"

using namespace mlir;
using namespace mlir::rock;

//===----------------------------------------------------------------------===//
// General utilities.
//===----------------------------------------------------------------------===//

template <typename AttrT>
static bool hasBigTransforms(ArrayRef<AttrT> transformList, Value ret) {
  bool isBig = llvm::any_of(transformList, [](Attribute a) {
    return needs64BitIndices(cast<TransformMapAttr>(a));
  });
  auto bufferType = cast<ShapedType>(ret.getType());
  isBig |= is4GBMemoryType(bufferType);
  return isBig;
}

std::tuple<Value, ArrayAttr, bool>
mlir::rock::untransform(OpBuilder &b, Value transformed, ArrayAttr existing) {
  SmallVector<Attribute> transformList;
  if (existing)
    transformList.append(existing.begin(), existing.end());
  Value ret = transformed;
  while (auto transform = ret.getDefiningOp<TransformOp>()) {
    transformList.push_back(transform.getTransform());
    ret = transform.getInput();
  }
  bool isBig = hasBigTransforms(ArrayRef<Attribute>(transformList), ret);
  return {ret, b.getArrayAttr(transformList), isBig};
}

std::tuple<Value, ArrayAttr, bool>
mlir::rock::untransform(OpBuilder &b, Value transformed,
                        ArrayRef<Attribute> existing) {
  return untransform(b, transformed, b.getArrayAttr(existing));
}

std::tuple<Value, bool>
mlir::rock::untransform(Value transformed,
                        SmallVectorImpl<TransformMapAttr> &transforms) {
  Value ret = transformed;
  while (auto transform = ret.getDefiningOp<TransformOp>()) {
    transforms.push_back(transform.getTransform());
    ret = transform.getInput();
  }
  bool isBig = hasBigTransforms(ArrayRef<TransformMapAttr>(transforms), ret);
  return {ret, isBig};
}

std::tuple<Value, bool>
mlir::rock::untransform(Value transformed,
                        SmallVectorImpl<TransformOp> &transforms) {
  Value ret = transformed;
  bool isBig = false;
  while (auto transformOp = ret.getDefiningOp<TransformOp>()) {
    transforms.push_back(transformOp);
    isBig |= needs64BitIndices(transformOp.getTransform());
    ret = transformOp.getInput();
  }
  isBig |= is4GBMemoryType(cast<ShapedType>(ret.getType()));
  return std::make_tuple(ret, isBig);
}

Value mlir::rock::transform(OpBuilder &b, Value toBeTransformed,
                            ArrayAttr transforms) {
  SmallVector<TransformMapAttr, 4> transformsVec =
      llvm::to_vector<4>(transforms.getAsRange<TransformMapAttr>());
  auto reverseTransformVec = llvm::reverse(transformsVec);
  Location loc = toBeTransformed.getLoc();
  Value ret = toBeTransformed;
  for (TransformMapAttr trMap : reverseTransformVec) {
    ret = b.create<TransformOp>(loc, ret, trMap);
  }
  return ret;
}

Value mlir::rock::isolateTransforms(OpBuilder &b, Value transformed) {
  SmallVector<TransformOp> ops;
  Value isolated;
  std::tie(isolated, std::ignore) = untransform(transformed, ops);
  bool needToClone = !llvm::all_of(ops, [](TransformOp op) -> bool {
    return op->hasOneUse() || op->use_empty();
  });
  if (!needToClone)
    return transformed;
  for (TransformOp trOp : llvm::reverse(ops)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfterValue(trOp.getOutput());
    IRMapping cloneMap;
    cloneMap.map(trOp.getInput(), isolated);
    Operation *newOp = b.clone(*trOp, cloneMap);
    isolated = newOp->getResult(0);
  }
  return isolated;
}

bool mlir::rock::needs64BitIndices(TransformMapAttr map) {
  // Negative lengths are some form of dynamic index and therefore big.
  auto isBig = [](int64_t l) -> bool {
    return l < 0 || l > (int64_t)(std::numeric_limits<int32_t>::max());
  };
  return llvm::any_of(map.getUpperBounds().asArrayRef(), isBig) ||
         llvm::any_of(map.getLowerBounds().asArrayRef(), isBig);
}

TransformOp mlir::rock::reshapeBuffer(OpBuilder &b, Location loc, Value buffer,
                                      ArrayRef<StringRef> names,
                                      ArrayRef<int64_t> shape) {
  MemRefType bufferType = cast<MemRefType>(buffer.getType());
  ArrayRef<int64_t> outShape = bufferType.getShape();
  assert(outShape.size() == 1 && "Buffer being reshaped must start linear");

  SmallVector<int64_t> strides;
  strides.reserve(shape.size());
  int64_t stride = 1;
  for (int64_t v : llvm::reverse(shape)) {
    stride *= v;
  }
  assert(stride == outShape[0] && "Strides must multiply to buffer length");

  TopDownTMBuilder transform(b, names, shape, loc);
  transform.unmerge("raw", 0, names, shape);

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
  llvm::IndexedMap<std::optional<VectorizationInfo>> data;
  operator llvm::IndexedMap<std::optional<VectorizationInfo>> &() {
    return data;
  }

  void grow(size_t n) {
    // The underlying grow() takes the max index, not the size
    data.grow(n - 1);
  }

  std::optional<VectorizationInfo> &operator[](uint32_t idx) {
    return data[idx];
  }

  const std::optional<VectorizationInfo> &operator[](uint32_t idx) const {
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
static std::optional<VectorizationInfo>
propagateUnmergeVectorization(T &&dimAndLength,
                              const VectorizationData &input) {
  std::optional<VectorizationInfo> result;
  int64_t previousDimsStride = 1;
  std::optional<int64_t> previousAlign;
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

      auto keyJ = dimToMerge[upperDims[j]];

      // Unit lengths don't affect the mergeability logic, if they come from
      // another transform. If the unit length is in the Merge, this can
      // be swapped during the Unmerge, e.g., Merge{2,16,1}->Unmerge{2,1,16}.
      // In this case we want to carry on the usual checks, since we want to
      // prevent the dimensions to be collapsed. Indeed, if collapsing happens
      // we would end up with a Merge{1,1,...,C} transform which would be
      // uncorrectly scaled in the Unmerge
      //
      // TODO: fix the `collapseContiguousDims` function to properly handle this
      // case
      if (params[j] == 1 && !keyJ.transform)
        continue;

      if (!keyJ.transform)
        break;

      auto mergeJParams = keyJ.transform.getParams();
      auto mergeJDims = keyJ.transform.getLowerDims();
      size_t posJ = keyJ.positionInMerge;

      // a) Dimensions need to come from the same merge
      if (keyJ.transformPair() != keyI.transformPair())
        break;

      // b) Unmerge parameters need to match merge parameters.
      int64_t expectedLength = mergeJParams[posJ];
      if (params[j] != expectedLength)
        break;

      groupCandidate.push_back(mergeJDims[posJ]);
      dimPosition.push_back(posJ);
    }

    i += std::max(size_t(1), groupCandidate.size());

    // Update the result with the current group for the mergePair key
    if (groupCandidate.size() > 1 &&
        std::is_sorted(dimPosition.begin(), dimPosition.end())) {

      uint32_t fastestDim = groupCandidate.back();
      size_t fastestDimPosInMerge = dimPosition.back();
      size_t slowestDimPosInMerge = dimPosition.front();
      for (auto d : groupCandidate) {
        // Make sure that the dimension cannot be reused
        dimToMerge.erase(d);
        contiguousGroups[keyI.transformPair()].unionSets(fastestDim, d);
      }

      // We also want to add the singleton dimensions of the merge the
      // groupCandidate belongs to. In this way we cover for situations like
      // - Merge{8,1,3}, <AddDim at [1]>, <Unmerge{8,3}>. However, it is
      // unsound to collapse onto trailing unit dimensions not otherwise
      // accounted for above, as they could be permuted out to
      // arbitrary other positions. Therefore, we only examine unit dimensions
      // which come earlier in the merge parameters than the first true
      // collapsed dimension.
      ArrayRef<uint32_t> thisMergeDims = keyI.transform.getLowerDims();
      ArrayRef<int64_t> thisMergeParams = keyI.transform.getParams();
      for (const auto [d, p] : llvm::zip(
               thisMergeDims.slice(slowestDimPosInMerge,
                                   fastestDimPosInMerge - slowestDimPosInMerge),
               thisMergeParams.slice(slowestDimPosInMerge,
                                     fastestDimPosInMerge -
                                         slowestDimPosInMerge))) {
        if (p == 1 && dimToMerge.contains(d)) {
          // Make sure that the dimension cannot be reused
          dimToMerge.erase(d);
          contiguousGroups[keyI.transformPair()].unionSets(fastestDim, d);
        }
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
ContiguousMergesMap findContiguousGroups(Value transformed) {
  // Transform table. Will be overwritten after processing each transform_map
  DimToMergeMap dimToMerge;
  ContiguousMergesMap contiguousGroups;

  Value currentVal = transformed;
  while (auto trOp = currentVal.getDefiningOp<TransformOp>()) {
    TransformMapAttr transformMap = trOp.getTransform();
    ArrayRef<int64_t> upperBounds = transformMap.getUpperBounds();
    DimToMergeMap thisDimToMerge;

    for (TransformAttr transform : transformMap.getOps()) {
      TransformType transformType = transform.getType();

      ArrayRef<uint32_t> lowerDims = transform.getLowerDims();
      ArrayRef<uint32_t> upperDims = transform.getUpperDims();
      ArrayRef<int64_t> params = transform.getParams();

      switch (transformType) {
      case TransformType::Merge:
        for (size_t i = 0; i < lowerDims.size(); i++) {
          thisDimToMerge[lowerDims[i]] = {transformMap, transform, i};
        }
        break;
      // AddDim drops dimensions down a hole, while ConstDim conjures them
      // from nowhere. In either case, there is no merge that can be associated
      // with them.
      case TransformType::AddDim:
      case TransformType::ConstDim:
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
      case TransformType::Slice:
      case TransformType::Pad:
      case TransformType::Broadcast:
        // We only care about how these transformations shuffle
        // the dimensions
        for (auto pair : llvm::zip(upperDims, lowerDims)) {
          uint32_t u = std::get<0>(pair);
          uint32_t l = std::get<1>(pair);
          thisDimToMerge[l] = dimToMerge[u];
        }
        break;
      case TransformType::Unmerge:
        findCountiguousGroupsUnmerge(upperDims, params, dimToMerge,
                                     contiguousGroups);
        break;
      }
    }
    currentVal = trOp.getInput();
    dimToMerge = thisDimToMerge;
  }

  // Last global unmerge
  auto outputType = cast<ShapedType>(currentVal.getType());
  SmallVector<uint32_t> sortedDims(outputType.getRank());
  std::iota(sortedDims.begin(), sortedDims.end(), 0);
  findCountiguousGroupsUnmerge(sortedDims, outputType.getShape(), dimToMerge,
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
    case TransformType::ConstDim:
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
        std::optional<VectorizationInfo> upperInfo = input[upper];
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
      // Since embed coefficients can go in any order, we need them sorted.
      // We first sort by the Embed coefficient
      // We tiebreak dimensions with equal coefficients as follows:
      //  - Dimensions that have vectorization info are processed before those
      //    that don't.
      //   - Two vectorization dimensions have ties broken by whether their
      //      needed coefficient is equal to their embed coefficient.
      //   - If both inputs have a matching coefficient, the one with the longer
      //     vectorization length is used (sending equality to input order).
      //   - Failing that, since neither of the inputs being compared are
      //     going to succesfully advance the vectorizaion, we fall back to
      //     order in the input map.
      // - In all otther cases, ties are broken by position in the input map.
      auto &&zip = llvm::zip(params, upperDims);
      SmallVector<std::tuple<int64_t, uint32_t>> data(zip.begin(), zip.end());
      std::sort(data.begin(), data.end(), [&](const auto &a, const auto &b) {
        auto [paramA, dimA] = a;
        auto [paramB, dimB] = b;
        if (paramA != paramB)
          return paramA < paramB;
        bool aHasVec = input[dimA].has_value(),
             bHasVec = input[dimB].has_value();
        if (!aHasVec && !bHasVec)
          return dimA < dimB;
        if (aHasVec && !bHasVec)
          return true;
        if (!aHasVec && bHasVec)
          return false;

        bool aNeedsThisCoeff = input[dimA]->needsCoefficient == paramA,
             bNeedsThisCoeff = input[dimB]->needsCoefficient == paramB;
        if (aNeedsThisCoeff && !bNeedsThisCoeff)
          return true;
        if (!aNeedsThisCoeff && bNeedsThisCoeff)
          return false;
        if (aNeedsThisCoeff && bNeedsThisCoeff) {
          int64_t aLen = input[dimA]->maxLength, bLen = input[dimB]->maxLength;
          if (aLen != bLen)
            return aLen < bLen;
        }
        return dimA < dimB;
      });

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
      std::optional<VectorizationInfo> ourResult;

      // When there are equal embed coefficients
      // we should always refer to the one with lowest
      // alignment as the alignment
      llvm::SmallDenseMap<int64_t, int64_t> coeffToAlignment;
      for (auto [coefficient, upperDim] : data) {
        if (input[upperDim].has_value()) {
          if (coeffToAlignment.count(coefficient)) {
            coeffToAlignment[coefficient] = math_util::gcd(
                coeffToAlignment[coefficient], input[upperDim]->alignment);
          } else {
            coeffToAlignment[coefficient] = input[upperDim]->alignment;
          }
        }
      }

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
          int64_t thisAlignment = coeffToAlignment[coefficient];

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
    // Contiguous groups of merge outputs (that is, outputs from Merge{})
    // that will later be combined into the exact same value as that part of the
    // merge input, are grouped together for analysis purposes,
    // since spillover from one to the next is effectively the same as movement
    // in a larger, contiguous dimension. See also collapseContiguousMerges(),
    // which may someday become a prerequisite for this pass.
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
        int64_t thisAlignment = std::max(align / stride, (int64_t)1);
        result[lowerDim] =
            VectorizationInfo(thisMaxLen, coeff * stride, thisAlignment);
        maxLen = std::max(maxLen / lowerLen, (int64_t)1);
        stride *= lowerLen;
      }
      break;
    }
    }
  }
  return result;
}

static FailureOr<std::pair<Value, Operation *>>
findPostFusionTransforms(Value buffer, Operation *currentUser) {
  Value newTransformed = nullptr;
  Operation *newRoot = nullptr;
  for (Operation *user : buffer.getUsers()) {
    if (user == currentUser)
      continue;
    Value candidate = nullptr;
    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
      if (copyOp.getTarget() == buffer)
        candidate = copyOp.getSource();
      else
        candidate = copyOp.getTarget();
    } else if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
      if (genericOp.getOutputs().size() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[vectorization] Can't process "
                                   "linalg.generic with multiple outputs\n");
        return failure();
      }
      Value genericOut = genericOp.getOutputs().front();
      if (genericOut == buffer) {
        if (auto index = genericOp->getAttrOfType<IntegerAttr>(
                "rock.majorTensorNumber")) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[vectorization] can't analyze linalg.generic "
                        "without rock.majorTensorNumber\n");
          return failure();
        } else
          candidate = genericOp.getInputs()[index.getInt()];
      } else
        candidate = genericOut;
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "[vectorization] Unexpected user of temporary buffer: "
                 << *user << "\n");
      return failure();
    }

    if (newTransformed) {
      LLVM_DEBUG(llvm::dbgs() << "[vectorization] Found multiple users that "
                                 "could be the next one\n");
      return failure();
    }
    newTransformed = candidate;
    newRoot = user;
  }
  if (!newTransformed) {
    LLVM_DEBUG(llvm::dbgs()
               << "[vectorization] This memref.alloc is a dead end\n");
    return failure();
  }
  return std::make_pair(newTransformed, newRoot);
}

VectorizationResult mlir::rock::getMaxVectorization(
    Value transformed, uint32_t dim, std::optional<int64_t> inputDimLen,
    Operation *operationRootForFusionTraversal, bool ignoreDataType) {
  auto upperType = cast<ShapedType>(transformed.getType());
  int64_t numInitialDims = upperType.getRank();
  int64_t initialVecLen = inputDimLen.value_or(upperType.getShape()[dim]);
  VectorizationData data;
  // grow() takes the last index, not the length
  data.grow(numInitialDims);
  data[dim] =
      VectorizationInfo(/*maxLength=*/initialVecLen, /*needsCoefficient=*/1,
                        /*alignment=*/initialVecLen);
  bool traverseFusions = (operationRootForFusionTraversal != nullptr);
  Operation *currentUser = operationRootForFusionTraversal;
  Value currentVal = transformed;
  LogicalResult fusionTraversalStatus = success();
  auto contiguousMerges = findContiguousGroups(transformed);

  // Advance to the next operation to analyze, updating any vectorization
  // analysis state as needed. This function must update currentVal and
  // currentUser, and may update other variables. In the simplest case, this
  // advances to the next rock.transform operation. However, it also handles:
  // - If we recah a memref.alloc() and are following fusions, go to the
  // source of post-fusion transforms (for an output fusion, the output of the
  // fusion) to traverse its transform stack (which involves) recomputing
  // contiguous merge data).
  // - For rock.scalarize, adjust the vectorization data to account for the
  // change in indexing scheme and continue.
  auto advance = [&]() -> bool {
    Operation *definingOp = currentVal.getDefiningOp();
    if (!definingOp)
      return false;
    if (auto trOp = dyn_cast<TransformOp>(definingOp)) {
      currentVal = trOp.getInput();
      currentUser = definingOp;
      return true;
    }
    if (isa<memref::AllocOp>(definingOp)) {
      if (!traverseFusions) {
        definingOp->emitError(
            "vectorization analysis found intermediate allocation but isn't "
            "following fusions, results may be incorrect\n");
        return false;
      }
      FailureOr<std::pair<Value, Operation *>> maybeNewStack =
          findPostFusionTransforms(currentVal, currentUser);
      if (failed(maybeNewStack)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[vectorization] Failed to advance past fusion\n");
        fusionTraversalStatus = failure();
        return false;
      }
      std::tie(currentVal, currentUser) = *maybeNewStack;
      LLVM_DEBUG(llvm::dbgs()
                     << "[vectorization] Advancing past fusion to new value "
                     << currentVal << "with data: ";);
      data.debugPrint();
      contiguousMerges = findContiguousGroups(currentVal);
      return true;
    }
    return false;
  };

  do {
    if (auto trOp = currentVal.getDefiningOp<TransformOp>()) {
      TransformMapAttr transformMap = trOp.getTransform();
      LLVM_DEBUG(llvm::dbgs() << "Max vectorization data: ");
      data.debugPrint();
      LLVM_DEBUG(llvm::dbgs() << "Processing: " << transformMap << "\n");
      data = propagateVectorizationInfo(transformMap, data, contiguousMerges);
    }
  } while (advance());
  LLVM_DEBUG(llvm::dbgs() << "Final max vectorization data: ");
  data.debugPrint();

  auto outputType = cast<ShapedType>(currentVal.getType());
  ArrayRef<int64_t> outputShape = outputType.getShape();
  LLVM_DEBUG(llvm::dbgs() << "Vectorization output shape: ");
  LLVM_DEBUG(llvm::interleaveComma(outputShape, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");
  std::optional<VectorizationInfo> finalUnmerge = propagateUnmergeVectorization(
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
  result = math_util::gcd(initialVecLen, result);

  // Vectorizing more than the physical vector length (128 bits) might
  // be harmful for coalescence and other metrics. Let's limit the maximum
  // amount of data to load to the maximum vector length. This means a
  // warp will issue, if possible, a global_load_dwordx4 instruction
  if (!ignoreDataType) {
    constexpr int64_t maxVectorLenBits = 128;
    int64_t bwidth = upperType.getElementTypeBitWidth();
    result = math_util::gcd(maxVectorLenBits / bwidth, result);
  }
  // bufferVectorSize will become non-trivial once scalarization comes in
  return VectorizationResult{/*max=*/result, /*bufferVectorSize=*/1,
                             /*fusionTraversalStatus=*/fusionTraversalStatus};
}

void mlir::rock::collapseContiguousMerges(Value transformed) {
  ContiguousMergesMap contigousMerges = findContiguousGroups(transformed);
  SmallVector<TransformOp> transformOps;
  std::tie(std::ignore, std::ignore) = untransform(transformed, transformOps);
  for (TransformOp trOp : llvm::reverse(transformOps)) {
    assert((trOp->hasOneUse() || trOp->use_empty()) &&
           "Transform ops whose merges will be collapsed must be isolated to "
           "ensure other IR doesn't break");
    bool changed = false;
    TransformMapAttr map = trOp.getTransform();
    SmallVector<TransformAttr> ops;
    ops.reserve(map.getOps().size());
    for (TransformAttr op : map.getOps()) {
      if (op.getType() != TransformType::Merge) {
        ops.push_back(op);
        continue;
      }
      auto mergeData = contigousMerges.find({map, op});
      if (mergeData == contigousMerges.end()) {
        ops.push_back(op);
        continue;
      }
      const llvm::EquivalenceClasses<uint32_t> &groups = mergeData->getSecond();
      SmallVector<int64_t> newLengths(op.getParams());
      ArrayRef<uint32_t> lowerDims = op.getLowerDims();
      uint32_t currentRep = lowerDims.back();
      size_t currentRepPos = lowerDims.size() - 1;
      // Don't process the fastest merge output twice.
      bool hadConcat = false;
      for (ssize_t idx = lowerDims.size() - 2; idx >= 0; --idx) {
        uint32_t dim = lowerDims[idx];
        if (groups.isEquivalent(dim, currentRep)) {
          hadConcat = true;
          newLengths[currentRepPos] *= newLengths[idx];
          newLengths[idx] = 1;
        } else {
          currentRep = dim;
          currentRepPos = idx;
        }
      }
      if (!hadConcat) { // we went through all this trouble for nothing
        ops.push_back(op);
        continue;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "[collapseContigousMerges] Updating: " << op << " to ";
        llvm::interleaveComma(newLengths, llvm::dbgs());
        llvm::dbgs() << "\n";
      });
      auto newMerge = TransformAttr::get(
          op.getContext(), TransformType::Merge, newLengths, op.getUpperNames(),
          op.getUpperDims(), op.getLowerNames(), op.getLowerDims());
      ops.push_back(newMerge);
      changed = true;
    }
    TransformMapAttr newMap = map;
    if (changed) {
      newMap = TransformMapAttr::get(ops, map.getUpperBounds(),
                                     map.getLowerBounds());
      trOp.setTransformAttr(newMap);
    }
  }
}

/// Embed operations can create some scenarios that lead to the need to
/// check if their output falls with the expected range. The first is any
/// subtraction. The second is the case where
/// [coifficient] * [max size of upper dim] > [lower bound] for any upper
/// dimension.
static bool embedCanBeInvalid(TransformMapAttr map, TransformAttr op) {
  assert(op.getType() == TransformType::Embed);
  int64_t lowerBound = map.getLowerBounds()[op.getLowerDims()[0]];
  ArrayRef<int64_t> dimSizes = map.getUpperBounds();
  return llvm::any_of(llvm::zip(op.getParams(), op.getUpperDims()),
                      [&](const auto &pair) -> bool {
                        int64_t coefficient = std::get<0>(pair);
                        uint32_t dim = std::get<1>(pair);
                        return (coefficient < 0) ||
                               ((dimSizes[dim] * coefficient) > lowerBound);
                      });
}

bool mlir::rock::mapImpactsValidity(TransformMapAttr map) {
  bool result = false;
  for (TransformAttr op : map.getOps()) {
    TransformType type = op.getType();
    ArrayRef<int64_t> params = op.getParams();
    if (type == TransformType::Pad) {
      for (size_t i = 0, e = params.size(); i < e; i += 2) {
        // Trivial padding doesn't impact validity
        result |= (params[i] != 0 || params[i + 1] != 0);
      }
    } else if (type == TransformType::Embed) {
      result |= embedCanBeInvalid(map, op);
    }
  }
  return result;
}

Value mlir::rock::updateValidityAfter(OpBuilder &b, Location loc,
                                      TransformMapAttr map,
                                      ValueRange outputs) {
  Value isValid =
      b.createOrFold<arith::ConstantIntOp>(loc, true, b.getI1Type());
  ArrayRef<int64_t> lowerBounds = map.getLowerBounds();

  // unsigned < catches both negatives (as all negatives are > the bound)
  // and being too large on the right.
  auto addLowerDimUltClamp = [&](uint32_t lowerDim) {
    int64_t bound = lowerBounds[lowerDim];
    Value boundConst = b.createOrFold<arith::ConstantIndexOp>(loc, bound);
    Value output = outputs[lowerDim];
    Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             output, boundConst);
    isValid =
        b.createOrFold<arith::AndIOp>(loc, b.getI1Type(), inBounds, isValid);
  };

  for (TransformAttr op : map.getOps()) {
    TransformType type = op.getType();
    ArrayRef<uint32_t> lowerDims = op.getLowerDims();
    ArrayRef<int64_t> params = op.getParams();
    if (type == TransformType::Pad) {
      for (const auto &pair : llvm::enumerate(lowerDims)) {
        size_t leftParam = 2 * pair.index();
        size_t rightParam = leftParam + 1;
        uint32_t lowerDim = pair.value();

        if (params[leftParam] == 0 && params[rightParam] == 0)
          continue;
        addLowerDimUltClamp(lowerDim);
      }
    }
    if (type == TransformType::Embed) {
      if (!embedCanBeInvalid(map, op))
        continue;
      addLowerDimUltClamp(op.getLowerDims()[0]);
    }
  }
  return isValid;
}

AffineMap mlir::rock::composeTransforms(ArrayRef<TransformMapAttr> transforms) {
  AffineMap result;
  for (auto attr : llvm::reverse(transforms)) {
    AffineMap map = attr.getMap().getAffineMap();
    if (result)
      result = result.compose(map);
    else
      result = map;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Converting general MLIR to transformations.
//===----------------------------------------------------------------------===//

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
    if (isa<AffineDimExpr>(resultExpr)) {
      foundInputDims.insert(originalMap.getDimPosition(idx));
    }
  }

  for (const auto &idxAndValue : llvm::enumerate(originalMap.getResults())) {
    auto idx = idxAndValue.index();
    AffineExpr resultExpr = idxAndValue.value();
    if (isa<AffineDimExpr>(resultExpr)) {
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
    auto inpType = cast<MemRefType>(inp.getType());
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
        if (diff != 0 && isa<AffineConstantExpr>(resultExpr) &&
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
      auto inpType = cast<MemRefType>(inp.getType());
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
      inpShape = cast<ShapedType>(inp.getType()).getShape();
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
          b, cast<ShapedType>(inp.getType()).getShape(), loc);
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

LogicalResult
mlir::rock::makeLinalgGenericWithIdentityAffMaps(PatternRewriter &b,
                                                 linalg::GenericOp laOp) {
  auto idxMaps = laOp.getIndexingMapsArray();
  auto outIdxMap = idxMaps.back();

  auto outs = laOp.getOutputs();
  if (outs.size() > 1)
    return laOp.emitError("Only 1 output supported");
  Value out = outs[0];
  auto outType = cast<ShapedType>(out.getType());

  SmallVector<Value> inps(laOp.getInputs());
  for (auto pair : llvm::zip(inps, idxMaps)) {
    if (auto inp = std::get<0>(pair)) {
      auto imap = std::get<1>(pair);

      if (imap != outIdxMap) {
        // inject a broadcast
        auto invertOutIdxMap = inversePermutation(outIdxMap);
        auto outToInpMap = imap.compose(invertOutIdxMap);
        Value regInp = rock::insertTransposeAndBroadcastTransforms(
            b, outType.getShape(), inp, outToInpMap);
        laOp->replaceUsesOfWith(inp, regInp);
      }
    }
  }

  // reset idxmaps
  b.modifyOpInPlace(laOp, [&]() {
    SmallVector<AffineMap, 5> newIdxMaps(idxMaps.size(), outIdxMap);
    laOp.setIndexingMapsAttr(b.getAffineMapArrayAttr(newIdxMaps));
  });

  return success();
}

TransformMapAttr mlir::rock::invertTransformMap(
    OpBuilder &b, mlir::rock::TransformMapAttr transformMap, Location loc) {
  ArrayRef<int64_t> lowShape = transformMap.getLowerBounds();
  llvm::IndexedMap<StringRef> lowNamesMap;
  if (!lowShape.empty())
    lowNamesMap.grow(lowShape.size() - 1); // grow takes largest index;
  for (auto transform : transformMap.getOps()) {
    for (const auto &[name, dim] :
         llvm::zip(transform.getLowerNames(), transform.getLowerDims())) {
      lowNamesMap[dim] = name;
    }
  }
  SmallVector<StringRef> lowNames;
  lowNames.reserve(lowNamesMap.size());
  for (size_t i = 0, e = lowNamesMap.size(); i < e; ++i) {
    lowNames.push_back(lowNamesMap[i]);
  }

  rock::TopDownTMBuilder transform(b, lowNames, lowShape, loc);
  for (auto tattr : transformMap.getOps()) {
    switch (tattr.getType()) {
    case rock::TransformType::PassThrough:
      transform.passThrough(tattr.getUpperNames(), tattr.getUpperDims(),
                            tattr.getLowerNames());
      break;
    case rock::TransformType::Pad:
    case rock::TransformType::Slice:
    case rock::TransformType::Embed:
    case rock::TransformType::Broadcast: // Unsupported
      return rock::TransformMapAttr();
    case rock::TransformType::AddDim:
      if (tattr.getParams()[0] != 1)
        // AddDim of length > 1 has no coherent inverse.
        return rock::TransformMapAttr();
      transform.constDim(tattr.getUpperNames()[0], tattr.getUpperDims()[0],
                         /*constantVal=*/0, /*lowerSize=*/1);
      break;
    case rock::TransformType::ConstDim:
      for (size_t i = 0, e = tattr.getLowerDims().size(); i < e; ++i) {
        // Only adding in constant unit dimensions is invertible
        if (tattr.getParams()[2 * i] != 0 || tattr.getParams()[2 * i + 1] != 1)
          return rock::TransformMapAttr();
        transform.ignore(tattr.getLowerNames()[i]);
      }
      break;
    case rock::TransformType::Unmerge:
      transform.merge(tattr.getUpperNames(), tattr.getUpperDims(),
                      tattr.getLowerNames()[0], tattr.getParams());
      break;
    case rock::TransformType::Merge:
      transform.unmerge(tattr.getUpperNames()[0], tattr.getUpperDims()[0],
                        tattr.getLowerNames(), tattr.getParams());
      break;
    }
  }

  return transform.get();
}

TransformMapAttr mlir::rock::transformCollapseShape(
    OpBuilder &b, Location loc, ArrayRef<int64_t> inpShape,
    ArrayRef<int64_t> outShape, ArrayRef<ReassociationIndices> reassocs) {
  // %5 = "tosa.reshape"(%4) {new_shape = [12, 12, 32]} :
  // (tensor<1x12x12x32xf32>) -> tensor<12x12x32xf32>
  //    - inpShape = [1, 12, 12, 32]
  //    - outShape = [12, 12, 32]

  // This shouldn't happen, but we're checking anyway
  if (outShape.size() != reassocs.size()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Collapse output shape doesn't match number of reassociations\n");
    return TransformMapAttr();
  }

  llvm::IndexedMap<bool> dimUsed;
  dimUsed.grow(inpShape.size() - 1);

  rock::TopDownTMBuilder transform(b, outShape, loc);
  for (const auto &[outDim, inpDims] : llvm::enumerate(reassocs)) {
    for (int64_t dim : inpDims)
      dimUsed[dim] = true;

    if (inpDims.size() == 1)
      transform.passThrough(inpDims[0], outDim);
    else if (inpDims.empty())
      transform.ignore(transform.startName(outDim));
    else {
      // Create the name store in advance
      llvm::SmallDenseMap<int64_t, SmallString<8>> mergeNamesStore;
      for (int64_t inpDim : inpDims) {
        SmallString<8> inpDimName(Twine("col" + Twine(inpDim)).str());
        mergeNamesStore[inpDim] = inpDimName;
      }
      SmallVector<uint32_t> mergeDims;
      SmallVector<StringRef> mergeNames;
      SmallVector<int64_t> mergeSizes;
      for (int64_t inpDim : inpDims) {
        mergeNames.push_back(mergeNamesStore[inpDim]);
        mergeDims.push_back(inpDim);
        mergeSizes.push_back(inpShape[inpDim]);
      }
      transform.merge(mergeNames, mergeDims, transform.startName(outDim),
                      mergeSizes);
    }
  }

  // Dimensions not mentioned in the collapse are unit dimensions that need
  // constant values.
  for (size_t i = 0, e = inpShape.size(); i < e; ++i) {
    if (dimUsed[i])
      continue;
    if (inpShape[i] != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Collapse omits a non-identity dimension, can't happen\n");
      return TransformMapAttr();
    }
    SmallString<8> constDimNameStore;
    StringRef constDimName =
        (Twine("const") + Twine(i)).toStringRef(constDimNameStore);
    transform.constDim(constDimName, i, /*constantVal=*/0, /*lowerSize=*/1);
  }
  return transform.get();
}

TransformMapAttr mlir::rock::transformExpandShape(
    OpBuilder &b, Location loc, ArrayRef<int64_t> inpShape,
    ArrayRef<int64_t> outShape, ArrayRef<ReassociationIndices> reassocs) {
  // %3 = "tosa.reshape"(%2) {new_shape = [1, 12, 12, 32]} :
  // (tensor<1x12x384xf32>) -> tensor<1x12x12x32xf32>
  //    - inpShape = [1, 12, 384]
  //    - outShape = [1, 12, 12, 32]

  // Shouldn't happen, but let's check anyway
  if (inpShape.size() != reassocs.size()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Expand input shape doesn't match number of reassociations\n");
    return TransformMapAttr();
  }

  llvm::IndexedMap<bool> dimDefined;
  dimDefined.grow(outShape.size() - 1);

  rock::BottomUpTMBuilder transform(b, inpShape, loc);
  for (const auto &[inpDim, outDims] : llvm::enumerate(reassocs)) {
    for (int64_t dim : outDims)
      dimDefined[dim] = true;

    if (outDims.size() == 1)
      transform.passThrough(outDims[0], inpDim);
    else if (outDims.empty()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Empty reassocation list in expand_shape, shouldn't happen\n");
      return TransformMapAttr();
    } else {

      // Create the name store in advance
      llvm::SmallDenseMap<int64_t, SmallString<8>> unmergeNamesStore;
      for (int64_t outDim : outDims) {
        SmallString<8> outDimName(Twine("exp" + Twine(outDim)).str());
        unmergeNamesStore[outDim] = outDimName;
      }

      SmallVector<uint32_t> unmergeDims;
      SmallVector<int64_t> unmergeSizes;
      SmallVector<StringRef> unmergeNames;
      for (int64_t outDim : outDims) {
        unmergeNames.push_back(unmergeNamesStore[outDim]);
        unmergeDims.push_back(outDim);
        unmergeSizes.push_back(outShape[outDim]);
      }
      transform.unmerge(unmergeNames, unmergeDims, transform.startName(inpDim),
                        unmergeSizes);
    }
  }

  // Dimensions not defined by the expansion rules are ignored unit dimensions.
  for (size_t i = 0, e = outShape.size(); i < e; ++i) {
    if (dimDefined[i])
      continue;
    if (outShape[i] != 1) {
      LLVM_DEBUG(llvm::dbgs() << "Memref expansion doesn't define a non-unit "
                                 "dimension in the view, can't happen\n");
      return TransformMapAttr();
    }
    SmallString<8> unitDimNameStore;
    StringRef unitDimName =
        (Twine("unit") + Twine(i)).toStringRef(unitDimNameStore);
    transform.addDim(unitDimName, i, 1);
  }
  return transform.get();
}

TransformMapAttr mlir::rock::transformExtractSlice(OpBuilder &b, Location loc,
                                                   ArrayRef<int64_t> inpShape,
                                                   ArrayRef<int64_t> outShape,
                                                   ArrayRef<int64_t> offsets,
                                                   ArrayRef<int64_t> sizes) {
  rock::BottomUpTMBuilder transform(b, inpShape, loc);
  SmallVector<StringRef, 4> lowerNameRefs;
  transform.getStartNames(lowerNameRefs);
  SmallVector<SmallString<8>> upperNameStores;
  SmallVector<StringRef, 4> upperNameRefs;
  for (StringRef lowerName : lowerNameRefs) {
    upperNameStores.emplace_back();
    upperNameRefs.push_back(
        (lowerName + Twine("_sliced")).toStringRef(upperNameStores.back()));
  }
  SmallVector<int64_t, 4> ends;
  for (auto [offset, size] : llvm::zip(offsets, sizes)) {
    ends.push_back(offset + size);
  }
  transform.slice(upperNameRefs, lowerNameRefs, offsets, ends);
  return transform.get();
}

TopDownTMBuilder mlir::rock::rotateIf(bool condition, TopDownTMBuilder &builder,
                                      TransformMapAttr &attr, int64_t stride,
                                      StringRef dName, int64_t d, int64_t dPos,
                                      StringRef kName, int64_t kOuter,
                                      ArrayRef<StringRef> beforeDims,
                                      ArrayRef<StringRef> afterDims,
                                      SmallVector<Attribute> &transformAttrs) {
  if (condition) {
    // d = (d+stride*k_outer)
    TopDownTMBuilder rotateD0 = TopDownTMBuilder::below(builder, attr);
    if (!beforeDims.empty())
      rotateD0.passThrough(beforeDims);
    rotateD0.embed(dName, dPos, d * kOuter, {kName, dName}, {stride, 1});
    if (!afterDims.empty())
      rotateD0.passThrough(afterDims);
    TransformMapAttr rotateD0Attr = rotateD0.get();
    transformAttrs.push_back(rotateD0Attr);

    // d = (d+stride*k_outer) % d
    TopDownTMBuilder rotateD1 = TopDownTMBuilder::below(rotateD0, rotateD0Attr);
    if (!beforeDims.empty())
      rotateD1.passThrough(beforeDims);
    rotateD1.takeRemainder(dName, d);
    if (!afterDims.empty())
      rotateD1.passThrough(afterDims);
    TransformMapAttr rotateD1Attr = rotateD1.get();
    transformAttrs.push_back(rotateD1Attr);
    TopDownTMBuilder rotated = TopDownTMBuilder::below(rotateD1, rotateD1Attr);
    return rotated;
  } else {
    TopDownTMBuilder unrotated = TopDownTMBuilder::below(builder, attr);
    return unrotated;
  }
}

void mlir::rock::expandFlatFunctionArguments(
    OpBuilder &b, func::FuncOp func, ArrayRef<SmallVector<StringRef>> names,
    TypeRange logicalTypes, SmallVectorImpl<Value> &expanded) {
  expanded.resize_for_overwrite(names.size());
  for (auto [arg, nameList, logicalType, logicalVal] :
       llvm::zip(func.getArguments(), names, logicalTypes, expanded)) {
    Location loc = arg.getLoc();
    auto logicalShapedTy = dyn_cast<ShapedType>(logicalType);
    // Pass scalars through unaltered
    if (!logicalShapedTy) {
      logicalVal = arg;
      continue;
    }
    TopDownTMBuilder flattener(b, nameList, logicalShapedTy.getShape(), loc);
    flattener.unmerge("raw", 0, nameList, logicalShapedTy.getShape());
    TransformMapAttr expandMap = flattener.get();
    logicalVal = b.create<rock::TransformOp>(loc, arg, expandMap);
  }
}

void mlir::rock::convertDimStridestoSizes(ArrayRef<int64_t> orderedDimStrides,
                                          int64_t numElements,
                                          SmallVectorImpl<int64_t> &dimSizes) {
  for (auto [idx, dimStride] : llvm::enumerate(orderedDimStrides)) {
    int64_t immLargerCoeff;
    if (idx != 0) {
      immLargerCoeff = orderedDimStrides[idx - 1];
    } else {
      immLargerCoeff = numElements;
    }
    dimSizes.push_back(immLargerCoeff / dimStride);
  }
}

ArrayAttr mlir::rock::prependUpperViews(OpBuilder &b, ArrayAttr viewsToPrepend,
                                        ArrayAttr existingViews) {
  SmallVector<Attribute, 4> views =
      llvm::to_vector<4>(viewsToPrepend.getAsRange<Attribute>());
  views.append(existingViews.getAsRange<Attribute>().begin(),
               existingViews.getAsRange<Attribute>().end());
  return b.getArrayAttr(views);
}

ArrayAttr mlir::rock::invertTransforms(OpBuilder &b, Location loc,
                                       ArrayAttr transforms) {
  SmallVector<Attribute, 4> invertedTrs;
  for (Attribute tr : llvm::reverse(transforms)) {
    auto trMap = cast<TransformMapAttr>(tr);
    TransformMapAttr invertedTrMap = invertTransformMap(b, trMap, loc);
    if (!invertedTrMap)
      return {};
    invertedTrs.push_back(invertedTrMap);
  }
  return b.getArrayAttr(invertedTrs);
}

ArrayRef<int64_t> mlir::rock::getLowerShape(ArrayAttr transformStack) {
  return cast<TransformMapAttr>(transformStack[transformStack.size() - 1])
      .getLowerBounds();
}

Value mlir::rock::addPassThroughIndices(OpBuilder &b, Value transformed,
                                        ArrayRef<int64_t> lengths,
                                        int64_t pos) {
  MLIRContext *context = transformed.getContext();
  size_t numberOfIndices = lengths.size();

  // No dimensions to add, return
  if (numberOfIndices == 0)
    return transformed;

  SmallVector<TransformOp> opsToWiden;
  Value ret;
  std::tie(ret, std::ignore) = untransform(transformed, opsToWiden);

  /// Add a transform that AddDim{}s in all the extra dimensions to make all the
  /// shapes match up.
  ArrayRef<int64_t> underlyingShape =
      cast<ShapedType>(ret.getType()).getShape();
  BottomUpTMBuilder addDimBuilder(b, underlyingShape, ret.getLoc());
  SmallVector<StringRef> underlyingNames;
  addDimBuilder.getStartNames(underlyingNames);
  addDimBuilder.passThrough(
      ArrayRef<StringRef>(underlyingNames).take_front(pos));
  auto backNames = ArrayRef<StringRef>(underlyingNames).drop_front(pos);
  SmallVector<uint32_t> backPoses(backNames.size());
  std::iota(backPoses.begin(), backPoses.end(), pos + numberOfIndices);
  addDimBuilder.passThrough(backNames, backPoses, backNames);
  for (auto [idx, len] : llvm::enumerate(lengths)) {
    SmallString<8> extraName;
    ("extra_" + Twine(idx)).toVector(extraName);
    addDimBuilder.addDim(extraName, pos + idx, len);
  }
  TransformMapAttr addDimAttr = addDimBuilder.get();
  ret = b.create<TransformOp>(ret.getLoc(), ret, addDimAttr);

  // Start iterating through the old stack
  for (TransformOp trOp : llvm::reverse(opsToWiden)) {
    TransformMapAttr trMap = trOp.getTransform();
    auto ops = llvm::to_vector(trMap.getOps());
    SmallVector<TransformAttr> newOps;

    // Since we are adding a PassThrough at position `pos`
    // we need to shift all the subsequent dimensions of the other
    // transformations by the number of extra indices we insert
    for (auto t : ops) {
      auto lowerDims = llvm::to_vector(t.getLowerDims());
      auto upperDims = llvm::to_vector(t.getUpperDims());
      for (size_t i = 0; i < lowerDims.size(); i++) {
        if (lowerDims[i] >= pos)
          lowerDims[i] += numberOfIndices;
      }
      for (size_t i = 0; i < upperDims.size(); i++) {
        if (upperDims[i] >= pos)
          upperDims[i] += numberOfIndices;
      }
      newOps.push_back(TransformAttr::get(context, t.getType(), t.getParams(),
                                          t.getUpperNames(), upperDims,
                                          t.getLowerNames(), lowerDims));
    }

    // Add the passthrough transforms
    SmallVector<SmallString<8>> extraNames;
    for (unsigned i = 0; i < numberOfIndices; i++) {
      SmallString<8> extraName;
      Twine("extra_" + Twine(i)).toVector(extraName);
      extraNames.push_back(extraName);
      auto passThrough = TransformAttr::get(
          context, TransformType::PassThrough, {}, {extraNames.back()},
          {unsigned(pos) + i}, {extraNames.back()}, {unsigned(pos) + i});
      newOps.push_back(passThrough);
    }

    // Change the lower/upper bounds. Given the old bounds [o0,...,on] the new
    // bounds will be [o0,..oP-1,(n0,..,nL-1),oP,...,on] where P is the position
    // `pos`
    ArrayRef<int64_t> originalUpperBounds = trMap.getUpperBounds();
    ArrayRef<int64_t> originalLowerBounds = trMap.getLowerBounds();
    SmallVector<int64_t, 4> newUpperBounds(originalUpperBounds.begin(),
                                           originalUpperBounds.begin() + pos);
    SmallVector<int64_t, 4> newLowerBounds(originalLowerBounds.begin(),
                                           originalLowerBounds.begin() + pos);
    for (size_t i = 0; i < numberOfIndices; i++) {
      newUpperBounds.push_back(lengths[i]);
      newLowerBounds.push_back(lengths[i]);
    }
    newUpperBounds.insert(newUpperBounds.end(),
                          originalUpperBounds.begin() + pos,
                          originalUpperBounds.end());
    newLowerBounds.insert(newLowerBounds.end(),
                          originalLowerBounds.begin() + pos,
                          originalLowerBounds.end());

    // Add the new transform to the stack
    TransformMapAttr newMap =
        TransformMapAttr::get(newOps, newUpperBounds, newLowerBounds);
    ret = b.create<TransformOp>(trOp.getLoc(), ret, newMap);
  }
  return ret;
}

enum DimType { Upper = 0, Lower = 1 };

/// This is an auxiliary data structure required for `removeUpperDimsFromMap`
/// function implementation (see below). The struct holds the type of a
/// `TransformAttr` as well as modified parameters, upper/lower names and
/// dimension indices.
struct TransformAttrArgs {
  rock::TransformType type;
  std::pair<SmallVector<StringRef>, SmallVector<StringRef>> preservedNames;
  std::pair<SmallVector<uint32_t>, SmallVector<uint32_t>> preservedDims;
  SmallVector<int64_t> params;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &stream,
                              const TransformAttrArgs &args) {
  auto print = [&stream](StringRef name, auto &container) {
    stream << name << " = [";
    llvm::interleaveComma(container, stream);
    stream << "]\n";
  };
  stream << args.type << "\n";
  print("upper preserved dims", std::get<DimType::Upper>(args.preservedDims));
  print("upper preserved names", std::get<DimType::Upper>(args.preservedNames));
  print("lower preserved dims", std::get<DimType::Lower>(args.preservedDims));
  print("lower preserved names", std::get<DimType::Lower>(args.preservedNames));
  print("params", args.params);
  return stream;
}

template <DimType Type>
SmallVector<uint32_t>
getPreservedIndices(rock::TransformAttr tr,
                    const SetVector<int64_t> &globalRemoveIndicesSet) {
  SmallVector<uint32_t> preservedIndices;
  auto upperDims =
      Type == DimType::Upper ? tr.getUpperDims() : tr.getLowerDims();
  for (auto dim : upperDims) {
    if (!globalRemoveIndicesSet.contains(dim)) {
      preservedIndices.push_back(dim);
    }
  }
  return preservedIndices;
}

SetVector<uint32_t>
getRemovedIndicesInTr(rock::TransformAttr tr, DimType type,
                      const SetVector<int64_t> &globalRemoveIndicesSet) {
  SetVector<uint32_t> removedDimsInThisTr;
  ArrayRef<unsigned int> dims =
      type == DimType::Upper ? tr.getUpperDims() : tr.getLowerDims();
  for (unsigned int dim : dims) {
    if (globalRemoveIndicesSet.contains(dim)) {
      removedDimsInThisTr.insert(dim);
    }
  }
  return removedDimsInThisTr;
}

template <DimType Type>
void populatePreservedNames(rock::TransformAttr tr, TransformAttrArgs &args) {
  const auto &preservedDims = std::get<Type>(args.preservedDims);
  auto names = Type == DimType::Upper ? tr.getUpperNames() : tr.getLowerNames();
  auto dims = Type == DimType::Upper ? tr.getUpperDims() : tr.getLowerDims();
  assert(names.size() == dims.size());

  SmallVector<StringRef> preservedNames = {};
  for (auto [idx, dim] : llvm::enumerate(dims)) {
    if (llvm::is_contained(preservedDims, dim)) {
      preservedNames.push_back(names[idx]);
    }
  }
  std::get<Type>(args.preservedNames) = std::move(preservedNames);
}

template <DimType Type>
SmallVector<uint32_t> getDifference(rock::TransformAttr tr,
                                    TransformAttrArgs &args) {
  SmallVector<uint32_t> difference;
  auto dims = Type == DimType::Upper ? tr.getUpperDims() : tr.getLowerDims();
  const auto &preserved = std::get<Type>(args.preservedDims);
  for (auto dim : dims) {
    if (!llvm::is_contained(preserved, dim)) {
      difference.push_back(dim);
    }
  }
  return difference;
}

template <DimType Type>
void remapDims(
    std::vector<TransformAttrArgs> &argsVector,
    std::pair<SmallVector<uint32_t>, SmallVector<uint32_t>> &removedDims) {
  auto &sortedDeletedDims = std::get<Type>(removedDims);
  llvm::sort(sortedDeletedDims);
  for (auto &args : argsVector) {
    auto &preserved = std::get<Type>(args.preservedDims);
    for (auto [idx, dim] : llvm::enumerate(preserved)) {
      size_t leastUpperBoundIdx = 0;
      while ((leastUpperBoundIdx < sortedDeletedDims.size()) &&
             (dim > sortedDeletedDims[leastUpperBoundIdx])) {
        ++leastUpperBoundIdx;
      }
      preserved[idx] -= leastUpperBoundIdx;
    }
  }
}

struct SubDimInfo {
  int64_t size;
  int64_t stride;
};

static SmallVector<int64_t> getStrides(ArrayRef<int64_t> dimLens) {
  SmallVector<int64_t> ret{1};
  for (int64_t dimLen : llvm::reverse(dimLens)) {
    if (ret.size() < dimLens.size()) {
      ret.push_back(dimLen * ret.back());
    }
  }
  ret = to_vector(llvm::reverse(ret));
  return ret;
}

/// Given a single `TransformMapAttr`s and a set of indices, the function
/// re-builds a map by removing dimensions specified by indices. The function
/// operates on the user provided upper dimensions bounds. This allows
/// to take into account changes in upper `TransformMapAttr`s. The function
/// modifies the user provided lower bounds to reflect the changes caused by
/// the re-building process.
/// For each `TransformAttr` in a given `TransformMapAttr`, the function
/// computes which upper dimensions need to be preserved as a set
/// difference between the upper dimension and `remove` indices. Then, the
/// function computes which lower dimensions of each `TransformAttr` need
/// to preserved based on several invariants and assumptions.
/// Afterwards, the functions updates the user provided `removeIndicesSet`
/// with a set difference between the original and preserved lower
/// dimension indices. Then, the function remaps dimension indices to eliminate
/// possible holes in index numbering - e.g., 0, 3, 4 -> 0, 1, 2. After that,
/// the function builds new `TransformAttr` and, at the end, constructs a new
/// `TransformMapAttr`
FailureOr<rock::TransformMapAttr> removeUpperDimsFromMap(
    OpBuilder &b, rock::TransformMapAttr trMap,
    SetVector<int64_t> &removeIndicesSet,
    llvm::SmallVector<int64_t> &origUpperBounds,
    llvm::SmallVector<int64_t> &origLowerBounds,
    DenseMap<int64_t, SmallVector<SubDimInfo>> &removedSubDims) {
  LLVM_DEBUG(llvm::dbgs() << "orig = " << trMap << ", removedSubDims.size="
                          << removedSubDims.size() << "\n");
  origLowerBounds =
      llvm::SmallVector<int64_t>(trMap.getLowerBounds().asArrayRef());

  SetVector<int64_t> newRemoveIndicesSet;
  SmallVector<TransformAttr> newOps;
  std::pair<SmallVector<uint32_t>, SmallVector<uint32_t>> removedDims = {};

  std::vector<TransformAttrArgs> argsVector;
  DenseMap<int64_t, SmallVector<SubDimInfo>> newRemovedSubDims;
  for (auto tr : trMap.getOps()) {
    TransformAttrArgs args;
    args.type = tr.getType();
    SmallVector<uint32_t> &preservedUpperDims =
        std::get<DimType::Upper>(args.preservedDims);
    SmallVector<uint32_t> &preservedLowerDims =
        std::get<DimType::Lower>(args.preservedDims);

    preservedUpperDims =
        getPreservedIndices<DimType::Upper>(tr, removeIndicesSet);

    const bool mustBePreserved =
        preservedUpperDims.size() == tr.getUpperDims().size();
    const bool mustBeCompletelyRemoved = preservedUpperDims.empty();
    const bool mustBeModified = !mustBePreserved && !mustBeCompletelyRemoved;

    // compute which lower dimensions must be preserved
    switch (args.type) {
    case TransformType::AddDim:
    case TransformType::ConstDim:
    case TransformType::Broadcast:
    case TransformType::Merge: {
      assert(!mustBeModified && "must be preserved or removed completely");
      [[fallthrough]];
    }
    case TransformType::Unmerge: {
      if (mustBePreserved || mustBeModified) {
        // preserve all original lower dims
        llvm::copy(tr.getLowerDims(), std::back_inserter(preservedLowerDims));
      }
      break;
    }
    case TransformType::PassThrough: {
      assert(tr.getUpperDims().size() == tr.getLowerDims().size());
      auto localLowerDims = tr.getLowerDims();

      // propagate all preserved upper dims to lower dims
      for (auto [idx, dim] : llvm::enumerate(tr.getUpperDims())) {
        if (llvm::is_contained(preservedUpperDims, dim)) {
          preservedLowerDims.push_back(localLowerDims[idx]);
        }
      }
      break;
    }
    default:
      return failure();
    }

    populatePreservedNames<DimType::Upper>(tr, args);
    populatePreservedNames<DimType::Lower>(tr, args);

    // re-compute transformation parameters
    if (mustBePreserved || mustBeModified) {
      switch (args.type) {
      case TransformType::Unmerge: {
        assert(preservedLowerDims.size() == 1);
        SmallVector<int64_t> subDimStrides = getStrides(tr.getParams());
        // Collect all removedSubDims in upper to the lower dim
        for (auto [upperDim, subDimStride] :
             zip(tr.getUpperDims(), subDimStrides)) {
          for (const SubDimInfo &remSubDimInfo : removedSubDims[upperDim]) {
            LLVM_DEBUG(llvm::dbgs() << "creating newRemovedSubDim /w size = "
                                    << remSubDimInfo.size << ", stride="
                                    << remSubDimInfo.stride * subDimStride
                                    << " @ " << preservedLowerDims[0] << "\n");
            newRemovedSubDims[preservedLowerDims[0]].push_back(
                {remSubDimInfo.size, remSubDimInfo.stride * subDimStride});
          }
        }
        SetVector<uint32_t> removedDimsInTr =
            getRemovedIndicesInTr(tr, DimType::Upper, removeIndicesSet);
        for (auto [idx, subDimSize] : enumerate(tr.getParams())) {
          int64_t upperDim = tr.getUpperDims()[idx];
          if (removedDimsInTr.contains(upperDim)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "creating newRemovedSubDim /w size = " << subDimSize
                       << ", stride=" << subDimStrides[idx] << " @ "
                       << preservedLowerDims[0] << "\n");
            newRemovedSubDims[preservedLowerDims[0]].push_back(
                {subDimSize, subDimStrides[idx]});
          }
        }
        uint32_t total = 1;
        for (auto globalDimIdx : preservedUpperDims) {
          auto dimSize = origUpperBounds[globalDimIdx];
          total *= dimSize;
          args.params.push_back(dimSize);
        }
        origLowerBounds[preservedLowerDims[0]] = total;
        break;
      }
      case TransformType::Merge: {
        SmallVector<int64_t> subDimStrides = getStrides(tr.getParams());
        SmallVector<SubDimInfo> relevantSubDims;
        assert(preservedUpperDims.size() == 1);
        LLVM_DEBUG(llvm::dbgs() << "preservedUpperDim = " << preservedUpperDims[0] << "\n");
        for (size_t subDim = 0; subDim < tr.getParams().size(); subDim++) {
          int64_t lowDim = tr.getLowerDims()[subDim];
          for (const SubDimInfo &removedSubDimInfo :
               removedSubDims[preservedUpperDims[0]]) {
            LLVM_DEBUG(llvm::dbgs() << "lowDim = " << lowDim << "\n");
            LLVM_DEBUG(llvm::dbgs() << "remove.stride = "
                                    << removedSubDimInfo.stride << "\n");
            LLVM_DEBUG(llvm::dbgs()
                       << "remove.size = " << removedSubDimInfo.size << "\n");
            LLVM_DEBUG(llvm::dbgs()
                       << "subdim.stride = " << subDimStrides[subDim] << "\n");
            LLVM_DEBUG(llvm::dbgs()
                       << "subdim.size = " << tr.getParams()[subDim] << "\n");

            if (removedSubDimInfo.stride >=
                subDimStrides[subDim] * tr.getParams()[subDim]) {
              // do nothing
              LLVM_DEBUG(llvm::dbgs()
                         << "The relative stride of removed subDim is larger "
                            "than original subDim\n");
            } else if (removedSubDimInfo.stride * removedSubDimInfo.size <
                       subDimStrides[subDim]) {
              // do nothing
              LLVM_DEBUG(llvm::dbgs()
                         << "The stride of this newly created sub dimension is "
                            "larger than removed subDim\n");
            }
            // Everyother case means removedSubDim at least partially overlaps
            // with this dimension
            else {
              LLVM_DEBUG(llvm::dbgs()
                         << "There is atleast partial overlap between removed "
                            "subDim and new subDim\n");
              int diff = 0;
              int newRemovedSubDimStride = 0;
              // Overlap on right side of removedSubDim
              if (removedSubDimInfo.stride * removedSubDimInfo.size >=
                  subDimStrides[subDim] * tr.getParams()[subDim]) {
                int64_t rhsBoundForRemoval =
                    std::max(removedSubDimInfo.stride, subDimStrides[subDim]);
                diff = (subDimStrides[subDim] * tr.getParams()[subDim]) /
                       rhsBoundForRemoval;
                newRemovedSubDimStride =
                    rhsBoundForRemoval / subDimStrides[subDim];
              }
              // The whole of removedSubDim is within the newly created lowDim
              else if (removedSubDimInfo.stride >= subDimStrides[subDim]) {
                diff = removedSubDimInfo.size;
                newRemovedSubDimStride = removedSubDimInfo.stride;
              }
              // Overlap is left side of removedSubDim
              else {
                diff = (removedSubDimInfo.stride * removedSubDimInfo.size) /
                       subDimStrides[subDim];
                newRemovedSubDimStride = 1;
              }
              LLVM_DEBUG(llvm::dbgs()
                         << "creating newRemovedSubDim /w size = " << diff
                         << ", stride=" << newRemovedSubDimStride << " @ "
                         << lowDim << "\n");
              newRemovedSubDims[lowDim].push_back(
                  {diff, newRemovedSubDimStride});
              origLowerBounds[lowDim] = origLowerBounds[lowDim] / diff;
            }
          }
          args.params.push_back(origLowerBounds[lowDim]);
        }
        break;
      }
      case TransformType::PassThrough: {
        // propagate possibly modified dimensions
        DenseMap<int64_t, int64_t> upperToLower;
        for (auto [idx, upperDim] : llvm::enumerate(tr.getUpperDims())) {
          const auto lowerDim = tr.getLowerDims()[idx];
          upperToLower[upperDim] = lowerDim;
          origLowerBounds[lowerDim] = origUpperBounds[upperDim];
        }
        for (auto [dim, subDimInfo] : removedSubDims) {
          newRemovedSubDims[upperToLower[dim]] = subDimInfo;
        }
        [[fallthrough]];
      }
      default: {
        llvm::copy(tr.getParams(), std::back_inserter(args.params));
        break;
      }
      }
      argsVector.push_back(args);
    }

    std::get<DimType::Upper>(removedDims)
        .append(getDifference<DimType::Upper>(tr, args));
    std::get<DimType::Lower>(removedDims)
        .append(getDifference<DimType::Lower>(tr, args));
  }
  removedSubDims = newRemovedSubDims;


  // todo: use vector instead of set
  // update remove indices set
  for (auto dim : std::get<DimType::Lower>(removedDims)) {
    newRemoveIndicesSet.insert(dim);
  }

  remapDims<DimType::Upper>(argsVector, removedDims);
  remapDims<DimType::Lower>(argsVector, removedDims);

  // build new transformations based on the computer preserved data
  for (auto &args : argsVector) {
    auto newTr =
        TransformAttr::get(b.getContext(), args.type, args.params,
                           std::get<DimType::Upper>(args.preservedNames),
                           std::get<DimType::Upper>(args.preservedDims),
                           std::get<DimType::Lower>(args.preservedNames),
                           std::get<DimType::Lower>(args.preservedDims));
    newOps.push_back(newTr);
  }

  // compute new loop bounds
  std::pair<SmallVector<int64_t>, SmallVector<int64_t>> newBounds;
  std::pair<SmallVector<int64_t> &, SmallVector<int64_t> &> oldBounds = {
      origUpperBounds, origLowerBounds};

  auto genNewBounds = [&](DimType type) {
    auto &newBound = type == DimType::Upper
                         ? std::get<DimType::Upper>(newBounds)
                         : std::get<DimType::Lower>(newBounds);
    auto oldBound = type == DimType::Upper
                        ? std::get<DimType::Upper>(oldBounds)
                        : std::get<DimType::Lower>(oldBounds);
    const auto &deletedDims = type == DimType::Upper
                                  ? std::get<DimType::Upper>(removedDims)
                                  : std::get<DimType::Lower>(removedDims);
    for (auto [dim, bound] : llvm::enumerate(oldBound)) {
      if (!llvm::is_contained(deletedDims, dim)) {
        newBound.push_back(bound);
      }
    }
  };
  genNewBounds(DimType::Upper);
  genNewBounds(DimType::Lower);

  // update the info for the next function invocation
  removeIndicesSet = newRemoveIndicesSet;

  // build a new transformation map
  rock::TransformMapAttr newTrMap;
  if (!newOps.empty()) {
    newTrMap =
        TransformMapAttr::get(newOps, std::get<DimType::Upper>(newBounds),
                              std::get<DimType::Lower>(newBounds));
    LLVM_DEBUG(llvm::dbgs() << "newTrMap = " << newTrMap << "\n");
  }
  return newTrMap;
}

/// Given a stack of `TransformMapAttr`s and a set of indices, the function
/// re-builds the maps by removing dimensions specified by indices. The function
/// operates from top to down. The user provided indices are considered to be
/// the upper dimensions of the top most `TransformMapAttr`. The function
/// propagates the remove indices set from top to bottom, gradually adding or
/// removing affected dimensions during the re-building process. The function
/// expect an input stack of `TransformMapAttr`s to be coherent - i.e.,
/// the lower dimensions for map (i) are the same as the upper dimensions of
/// map (i - 1)
FailureOr<ArrayAttr>
mlir::rock::removeUpperDims(OpBuilder &b, ArrayAttr transformAttrs,
                            SetVector<int64_t> removeIndicesSet) {
  SmallVector<Attribute> results;

  llvm::SmallVector<int64_t> upperBounds = {};
  DenseMap<int64_t, SmallVector<SubDimInfo>> preservedSubDims;
  if (!transformAttrs.empty()) {
    auto first = *(transformAttrs.begin());
    auto trMap = cast<rock::TransformMapAttr>(first);
    upperBounds =
        llvm::SmallVector<int64_t>(trMap.getUpperBounds().asArrayRef());
  }

  for (auto map : transformAttrs) {
    auto trMap = cast<rock::TransformMapAttr>(map);

    assert(upperBounds.size() ==
           static_cast<size_t>(trMap.getUpperBounds().size()));
    llvm::SmallVector<int64_t> lowerBounds = {};
    FailureOr<rock::TransformMapAttr> maybeNewTrMapAttr =
        removeUpperDimsFromMap(b, trMap, removeIndicesSet, upperBounds,
                               lowerBounds, preservedSubDims);
    upperBounds = lowerBounds;
    if (failed(maybeNewTrMapAttr)) {
      return failure();
    }
    if (*maybeNewTrMapAttr) {
      results.push_back(*maybeNewTrMapAttr);
    }
  }

  return b.getArrayAttr(results);
}

SetVector<int64_t>
convertDimNamesToIndices(const ArrayAttr trAttrs,
                         const SetVector<StringRef> &removeDimNamesSet) {
  SetVector<int64_t> indices = {};
  if (trAttrs.empty())
    return indices;

  const auto trMap = cast<TransformMapAttr>(trAttrs[0]);
  for (auto tr : trMap.getOps()) {
    ArrayRef<::llvm::StringRef> names = tr.getUpperNames();
    ArrayRef<uint32_t> dims = tr.getUpperDims();
    for (auto [name, dim] : llvm::zip_equal(names, dims)) {
      if (removeDimNamesSet.contains(name)) {
        indices.insert(dim);
      }
    }
  }
  return indices;
}

/// This function is an overload of the function above. The function converts
/// the user provided dim. names to indices and calls the implementation
/// of `removeUpperDims` from above.
FailureOr<ArrayAttr>
mlir::rock::removeUpperDims(OpBuilder &b, ArrayAttr transformAttrs,
                            const SetVector<StringRef> &removeDimNamesSet) {
  SetVector<int64_t> removeIndicesSet =
      convertDimNamesToIndices(transformAttrs, removeDimNamesSet);
  return removeUpperDims(b, transformAttrs, removeIndicesSet);
}
