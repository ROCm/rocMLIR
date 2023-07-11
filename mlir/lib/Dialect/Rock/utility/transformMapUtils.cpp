//===- transformMapUtils.cpp - transform map utilities --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/math.h"
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
propagateUnmergeVectorization(T &&dimAndLength, const VectorizationData &input,
                              int64_t startStrideFromOtherInfo = 1) {
  std::optional<VectorizationInfo> result;
  int64_t previousDimsStride = startStrideFromOtherInfo;
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
      case TransformType::Merge:
        for (size_t i = 0; i < lowerDims.size(); i++) {
          thisDimToMerge[lowerDims[i]] = {transformMap, transform, i, false};
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
      std::optional<VectorizationInfo> ourResult;
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

int64_t mlir::rock::getMaxVectorization(ArrayAttr transforms, uint32_t dim,
                                        int64_t len,
                                        ArrayRef<int64_t> outputShape,
                                        int64_t implicitStride) {
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
    LLVM_DEBUG(llvm::dbgs() << "Processing: " << transformMap << "\n");
    data = propagateVectorizationInfo(transformMap, data, contiguousMerges);
  }
  LLVM_DEBUG(llvm::dbgs() << "Final max vectorization data: ");
  data.debugPrint();

  LLVM_DEBUG(llvm::dbgs() << "Vectorization output shape: ");
  LLVM_DEBUG(llvm::interleaveComma(outputShape, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");
  std::optional<VectorizationInfo> finalUnmerge = propagateUnmergeVectorization(
      llvm::zip(llvm::reverse(
                    llvm::iota_range<uint32_t>(0, outputShape.size(), false)),
                llvm::reverse(outputShape)),
      data, implicitStride);
  int64_t result = 1;
  if (finalUnmerge.has_value())
    LLVM_DEBUG(llvm::dbgs() << "Final unmerge: " << *finalUnmerge << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "Final unmerge yielded no result\n");
  if (finalUnmerge && finalUnmerge->needsCoefficient == 1)
    result = finalUnmerge->maxLength;
  // TODO(kdrewnia): Add support for tails
  result = math_util::gcd(len, result);
  return result * implicitStride;
}

int64_t mlir::rock::getMaxVectorizationForDatatype(
    ArrayAttr transforms, uint32_t dim, int64_t len,
    ArrayRef<int64_t> outputShape, Type dataType) {

  // Get the number of continuous elements that could be read at once
  int64_t theoreticalVectorLen =
      getMaxVectorization(transforms, dim, len, outputShape);

  // Vectorizing more than the physical vector length (128 bits) might
  // be harmful for coalescence and other metrics. Let's limit the maximum
  // amount of data to load to the maximum vector length. This means a
  // warp will issue, if possible, a global_load_dwordx4 instruction
  const int64_t maxVectorLenBits = 128;
  int64_t bwidth = dataType.getIntOrFloatBitWidth();
  int64_t realVectorLength =
      math_util::gcd(maxVectorLenBits / bwidth, theoreticalVectorLen);
  return realVectorLength;
}

ArrayAttr mlir::rock::collapseContiguousMerges(ArrayAttr transforms,
                                               ArrayRef<int64_t> outputShape) {
  ContiguousMergesMap contigousMerges =
      findContiguousGroups(transforms, outputShape);
  SmallVector<Attribute> newTransformMaps;
  for (auto map : transforms.getAsRange<TransformMapAttr>()) {
    bool changed = false;
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
      auto newMerge = TransformAttr::get(
          op.getContext(), TransformType::Merge, newLengths, op.getUpperNames(),
          op.getUpperDims(), op.getLowerNames(), op.getLowerDims());
      ops.push_back(newMerge);
      changed = true;
    }
    if (changed) {
      auto newMap = TransformMapAttr::get(ops, map.getUpperBounds(),
                                          map.getLowerBounds());
      newTransformMaps.push_back(newMap);
    } else {
      newTransformMaps.push_back(map);
    }
  }
  return ArrayAttr::get(transforms.getContext(), newTransformMaps);
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
