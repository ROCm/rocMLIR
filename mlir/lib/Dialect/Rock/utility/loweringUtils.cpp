//===- loweringUtils.cpp - Rock utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
using namespace mlir;
using namespace mlir::rock;

#define DEBUG_TYPE "rock-lowering-utils"

bool mlir::rock::isValidBlockSize(int64_t blockSize, int64_t kPerBlock,
                                  int64_t mPerBlock, int64_t nPerBlock) {
  int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
  int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;
  return (aCopyPerThread != 0 && bCopyPerThread != 0);
}

bool mlir::rock::is4GBMemoryType(ShapedType type) {
  if (!type.hasStaticShape())
    return true;
  int64_t elemBytes;
  if (auto shapedElemTy = dyn_cast<ShapedType>(type.getElementType()))
    elemBytes = (shapedElemTy.getNumElements() *
                 shapedElemTy.getElementTypeBitWidth()) /
                8;
  else
    elemBytes = type.getElementTypeBitWidth() / 8;

  return (type.getNumElements() * elemBytes) >
         (int64_t)std::numeric_limits<uint32_t>::max();
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

  assert(gemmM > 0 && gemmN > 0 && gemmK > 0);
  assert(MPerBlock > 0 && NPerBlock > 0 && KPerBlock > 0 && KPack > 0 &&
         batchSize > 0);

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

bool mlir::rock::isEveryElementWrittenBwdData(ArrayRef<int64_t> strideDims,
                                              ArrayRef<int64_t> dilationDims,
                                              ArrayRef<int64_t> filterDims) {
  bool result = true;
  for (const auto &[stride, dilation, filterSize] :
       zip(strideDims, dilationDims, filterDims)) {
    if (!(dilation == 1 && stride <= filterSize))
      result = false;
  }
  return result;
}

SmallVector<int64_t>
mlir::rock::backwardDataKernelIds(ArrayRef<int64_t> strideDims,
                                  ArrayRef<int64_t> dilationDims,
                                  ArrayRef<int64_t> filterDims, bool usesV4R1) {
  assert(strideDims.size() == dilationDims.size());
  SmallVector<int64_t, 5> gcdStrideDilations;
  for (const auto &[stride, dilation] : zip(strideDims, dilationDims))
    gcdStrideDilations.push_back(math_util::gcd(stride, dilation));

  SmallVector<int64_t, 5> filTilda;
  for (const auto &[stride, gcdSD] : zip(strideDims, gcdStrideDilations))
    filTilda.push_back(stride / gcdSD);

  // Populate the kernel IDs according to the current backward data convolution
  // algorithm implementation.
  llvm::SmallVector<int64_t> kernelIds;
  int64_t subproduct = 1;
  int64_t product;
  for (size_t i = 1; i < filterDims.size(); i++)
    subproduct *= filTilda[i];
  product = subproduct * filTilda[0];
  for (int64_t kernelId = 0; kernelId < product; ++kernelId) {
    // gemmK size is different for each GEMM
    SmallVector<int64_t, 3> iTilda;
    SmallVector<int64_t, 3> iDotSlice;
    int64_t divisor = 1;
    iTilda.resize(filterDims.size());
    switch (filterDims.size()) {
    default:
      llvm_unreachable("Only 2-D and 3-D have been implemented.");
      break;
    case 3:
      divisor = filTilda[2];
      iTilda[2] = kernelId % divisor;
      [[fallthrough]];
    case 2:
      iTilda[1] = (kernelId % subproduct) / divisor;
      iTilda[0] = kernelId / subproduct;
    }
    for (size_t i = 0; i < filterDims.size(); i++)
      iDotSlice.push_back(math_util::integer_divide_ceil(
          filterDims[i] - iTilda[i], filTilda[i]));

    // gemmK must > 0, otherwise not need to run
    int64_t gemmKproduct = 1;
    for (int64_t ds : iDotSlice)
      gemmKproduct *= ds;
    if (gemmKproduct > 0) {
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

static void makeLoadRegsTidMerge(TopDownTMBuilder &viewBuilder,
                                 StringRef dThreadName, int64_t dThreads,
                                 int64_t kThreads, ArrayRef<unsigned> outDims,
                                 bool isKContiguousDim) {
  if (isKContiguousDim) {
    viewBuilder.merge({dThreadName, "k_thread"}, outDims, "tid",
                      {dThreads, kThreads});
  } else {
    viewBuilder.merge({"k_thread", dThreadName}, outDims, "tid",
                      {kThreads, dThreads});
  }
}

static void makeLoadRegsIterMerge(TopDownTMBuilder &viewBuilder,
                                  StringRef dIterName, int64_t dPerThread,
                                  int64_t kPerThread,
                                  ArrayRef<unsigned> outDims,
                                  bool isKContiguousDim) {
  if (isKContiguousDim) {
    viewBuilder.merge({dIterName, "k_iter"}, outDims, "iter",
                      {dPerThread, kPerThread});
  } else {
    viewBuilder.merge({"k_iter", dIterName}, outDims, "iter",
                      {kPerThread, dPerThread});
  }
}

FailureOr<RegsAsMatrixSubTiles> mlir::rock::getLoadRegsAsTileViews(
    OpBuilder &b, Location loc, Value globalBuffer, StringRef dName,
    ArrayRef<StringRef> bidGridOrder, ArrayRef<int64_t> bidGridLengths,
    int64_t blockSize, int64_t kPerBlock, int64_t dPerBlock, int64_t kPerThread,
    int64_t dPerThread, bool isKContiguousDim, bool directToLDS) {
  if (dName != "m" && dName != "n") {
    return emitError(loc, "expected dName to be m or n but got " + dName);
  }
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  StringRef otherBlockDim = dName == "m" ? "n_block" : "m_block";

  MemRefType matrixType = cast<MemRefType>(globalBuffer.getType());
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
  int64_t kGlobal = matrixShape[1];
  int64_t dGlobal = matrixShape[2];

  int64_t kIters = kGlobal / kPerBlock;
  int64_t dataPerThread = (kPerBlock * dPerBlock) / blockSize;

  SmallString<8> dIterName = llvm::formatv("{0}_iter", dName);
  SmallString<8> dThreadName = llvm::formatv("{0}_thread", dName);

  // Note: (kThreads * dThreads) = (kPerBlock * dPerBlock) / dataPerThread) =
  // blockSize
  if (dPerBlock % dPerThread != 0) {
    return failure();
  }
  int64_t dThreads = dPerBlock / dPerThread;
  int64_t kThreads = blockSize / dThreads;
  if (kThreads * dThreads != blockSize) {
    return failure();
  }

  RegsAsMatrixSubTiles gpuViews;
  {
    TopDownTMBuilder gridwiseSplitId(
        b,
        {"k_loop", bidGridOrder[0], bidGridOrder[1], bidGridOrder[2], "tid",
         "iter"},
        {kIters, bidGridLengths[0], bidGridLengths[1], bidGridLengths[2],
         blockSize, dataPerThread},
        loc);
    gridwiseSplitId.passThrough(
        {"k_loop", bidGridOrder[0], bidGridOrder[1], bidGridOrder[2]});
    makeLoadRegsTidMerge(gridwiseSplitId, dThreadName, dThreads, kThreads,
                         {4, 5}, isKContiguousDim);
    makeLoadRegsIterMerge(gridwiseSplitId, dIterName, dPerThread, kPerThread,
                          {6, 7}, isKContiguousDim);
    TransformMapAttr splitIdAttr = gridwiseSplitId.get();
    auto toGlobalIdx = TopDownTMBuilder::below(gridwiseSplitId, splitIdAttr);
    toGlobalIdx.passThrough({"g"}, {0}, {"g_block"});
    if (directToLDS) {
      if (isKContiguousDim) {
        toGlobalIdx.unmerge("k", 1, {"k_loop", "k_thread", "k_iter"},
                            {kGlobal / kPerBlock, kThreads, kPerThread});
        toGlobalIdx.unmerge(dName, 2, {thisBlockDim, dIterName, dThreadName},
                            {dGlobal / dPerBlock, dPerThread, dThreads});
      } else {
        toGlobalIdx.unmerge("k", 1, {"k_loop", "k_iter", "k_thread"},
                            {kGlobal / kPerBlock, kPerThread, kThreads});
        toGlobalIdx.unmerge(dName, 2, {thisBlockDim, dThreadName, dIterName},
                            {dGlobal / dPerBlock, dThreads, dPerThread});
      }
    } else {
      toGlobalIdx.unmerge("k", 1, {"k_loop", "k_thread", "k_iter"},
                          {kGlobal / kPerBlock, kThreads, kPerThread});
      toGlobalIdx.unmerge(dName, 2, {thisBlockDim, dThreadName, dIterName},
                          {dGlobal / dPerBlock, dThreads, dPerThread});
    }

    toGlobalIdx.ignore(otherBlockDim);
    TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();
    gpuViews.gridSubTile = b.getArrayAttr({splitIdAttr, toGlobalIdxAttr});
  }
  {
    StringSet<> dimensionsToRemove{"k_loop", bidGridOrder[0], bidGridOrder[1],
                                   bidGridOrder[2]};
    FailureOr<ArrayAttr> maybeBlockSubTile =
        removeUpperDims(b, gpuViews.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTile)) {
      return failure();
    }
    gpuViews.blockSubTile = maybeBlockSubTile.value();
  }
  {
    StringSet<> dimensionsToRemove{"k_loop", bidGridOrder[0], bidGridOrder[1],
                                   bidGridOrder[2], "tid"};
    FailureOr<ArrayAttr> maybeThreadSubTile =
        removeUpperDims(b, gpuViews.gridSubTile, dimensionsToRemove);

    if (failed(maybeThreadSubTile)) {
      return failure();
    }
    gpuViews.threadSubTile = maybeThreadSubTile.value();
  }
  return gpuViews;
}

FailureOr<RegsAsMatrixSubTiles> mlir::rock::getPackedRegsAsTileViews(
    OpBuilder &b, Location loc, Value globalBuffer, StringRef dName,
    ArrayRef<StringRef> bidGridOrder, ArrayRef<int64_t> bidGridLengths,
    int64_t blockSize, int64_t kPerBlock, int64_t dPerBlock, int64_t kPerThread,
    int64_t dPerThread, int64_t kpack, bool isKContiguousDim,
    bool doSwapThreadIterSubDimsForD) {
  if (dName != "m" && dName != "n") {
    return emitError(loc, "expected dName to be m or n but got " + dName);
  }
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  StringRef otherBlockDim = dName == "m" ? "n_block" : "m_block";

  MemRefType matrixType = cast<MemRefType>(globalBuffer.getType());
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
  int64_t kGlobal = matrixShape[1];
  int64_t dGlobal = matrixShape[2];

  int64_t kIters = kGlobal / kPerBlock;
  int64_t dataPerThread = (kPerBlock * dPerBlock) / blockSize;

  SmallString<8> dIterName = llvm::formatv("{0}_iter", dName);
  SmallString<8> dThreadName = llvm::formatv("{0}_thread", dName);

  // Note: (kThreads * dThreads) = (kPerBlock * dPerBlock) / dataPerThread) =
  // blockSize
  if (dPerBlock % dPerThread != 0) {
    return failure();
  }
  int64_t dThreads = dPerBlock / dPerThread;
  int64_t kThreads = blockSize / dThreads;
  if (kThreads * dThreads != blockSize) {
    return failure();
  }

  int64_t kpackPerThread = std::min(kPerThread, kpack);
  assert(kPerThread % kpackPerThread == 0);
  int64_t kOuterPerThread = kPerThread / kpackPerThread;

  RegsAsMatrixSubTiles gpuViews;
  {
    TopDownTMBuilder gridwiseSplitId(
        b,
        {"k_loop", bidGridOrder[0], bidGridOrder[1], bidGridOrder[2], "tid",
         "iter"},
        {kIters, bidGridLengths[0], bidGridLengths[1], bidGridLengths[2],
         blockSize, dataPerThread},
        loc);
    gridwiseSplitId.passThrough(
        {"k_loop", bidGridOrder[0], bidGridOrder[1], bidGridOrder[2]});
    makeLoadRegsTidMerge(gridwiseSplitId, dThreadName, dThreads, kThreads,
                         {4, 5}, isKContiguousDim);
    gridwiseSplitId.merge({"kouterPerThread", dIterName, "kpackPerThread"},
                          {6, 7, 8}, "iter",
                          {kOuterPerThread, dPerThread, kpackPerThread});
    TransformMapAttr splitIdAttr = gridwiseSplitId.get();
    auto toGlobalIdx = TopDownTMBuilder::below(gridwiseSplitId, splitIdAttr);
    toGlobalIdx.passThrough({"g"}, {0}, {"g_block"});
    toGlobalIdx.unmerge(
        "k", 1, {"k_loop", "k_thread", "kouterPerThread", "kpackPerThread"},
        {kGlobal / kPerBlock, kThreads, kOuterPerThread, kpackPerThread});
    // if the matrix is KxD swap the iter/thread dimension. This is so that
    // each thread writes in LDS contiguously, minimizing bank conflicts
    if (!doSwapThreadIterSubDimsForD)
      toGlobalIdx.unmerge(dName, 2, {thisBlockDim, dThreadName, dIterName},
                          {dGlobal / dPerBlock, dThreads, dPerThread});
    else
      toGlobalIdx.unmerge(dName, 2, {thisBlockDim, dIterName, dThreadName},
                          {dGlobal / dPerBlock, dPerThread, dThreads});

    toGlobalIdx.ignore(otherBlockDim);
    TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();
    gpuViews.gridSubTile = b.getArrayAttr({splitIdAttr, toGlobalIdxAttr});
  }
  {
    StringSet<> dimensionsToRemove{"k_loop", bidGridOrder[0], bidGridOrder[1],
                                   bidGridOrder[2]};
    FailureOr<ArrayAttr> maybeBlockSubTile =
        removeUpperDims(b, gpuViews.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTile)) {
      return failure();
    }
    gpuViews.blockSubTile = maybeBlockSubTile.value();
  }
  {
    StringSet<> dimensionsToRemove{"k_loop", bidGridOrder[0], bidGridOrder[1],
                                   bidGridOrder[2], "tid"};
    FailureOr<ArrayAttr> maybeThreadSubTile =
        removeUpperDims(b, gpuViews.gridSubTile, dimensionsToRemove);

    if (failed(maybeThreadSubTile)) {
      return failure();
    }
    gpuViews.threadSubTile = maybeThreadSubTile.value();
  }
  return gpuViews;
}

Value mlir::rock::normalizeMatrix(Value matrix, OpBuilder &b, Location loc,
                                  bool doTranspose, StringRef firstDim,
                                  StringRef secondDim) {
  auto matrixType = cast<MemRefType>(matrix.getType());
  bool addGroup = matrixType.getShape().size() != 3;
  if (!addGroup && !doTranspose)
    return matrix;
  SmallVector<StringRef, 3> bottomNames;
  if (!addGroup)
    bottomNames.push_back("gemmG");
  if (doTranspose)
    bottomNames.append({secondDim, firstDim});
  else
    bottomNames.append({firstDim, secondDim});
  BottomUpTMBuilder normalizer(b, bottomNames, matrixType.getShape(), loc);

  if (addGroup)
    normalizer.addDim("gemmG", 0, 1);
  else
    normalizer.passThrough(normalizer.startName(0));

  normalizer.passThrough({firstDim, secondDim}, {1, 2}, {firstDim, secondDim});
  TransformMapAttr normalizeAttr = normalizer.get();
  return TransformOp::create(b, loc, matrix, normalizeAttr);
}

Value mlir::rock::padVector(Value vector, OpBuilder &b, Location loc,
                            StringRef firstDim, int64_t firstDimPad) {
  if (firstDimPad == 0)
    return vector;
  ArrayRef<int64_t> shape = cast<MemRefType>(vector.getType()).getShape();
  assert(shape.size() == 2);
  BottomUpTMBuilder padder(b, {"gemmG", firstDim}, shape, loc);
  padder.passThrough("gemmG");
  SmallString<8> paddedName;
  (firstDim + Twine("Pad")).toVector(paddedName);
  padder.pad(paddedName, firstDim, 0, firstDimPad);
  TransformMapAttr padAttr = padder.get();
  return TransformOp::create(b, loc, vector, padAttr);
}

Value mlir::rock::padMatrix(Value matrix, OpBuilder &b, Location loc,
                            StringRef firstDim, int64_t firstDimPad,
                            StringRef secondDim, int64_t secondDimPad) {
  if (firstDimPad == 0 && secondDimPad == 0)
    return matrix;
  ArrayRef<int64_t> shape = cast<MemRefType>(matrix.getType()).getShape();
  BottomUpTMBuilder padder(b, {"gemmG", firstDim, secondDim}, shape, loc);
  padder.passThrough("gemmG");
  if (firstDimPad == 0) {
    padder.passThrough(firstDim);
  } else {
    SmallString<8> paddedName;
    (firstDim + Twine("Pad")).toVector(paddedName);
    padder.pad(paddedName, firstDim, 0, firstDimPad);
  }
  if (secondDimPad == 0) {
    padder.passThrough(secondDim);
  } else {
    SmallString<8> paddedName;
    (secondDim + Twine("Pad")).toVector(paddedName);
    padder.pad(paddedName, secondDim, 0, secondDimPad);
  }
  TransformMapAttr padAttr = padder.get();
  return TransformOp::create(b, loc, matrix, padAttr);
}

FailureOr<TopDownTMBuilder> mlir::rock::swapThreadIdAndIteration(
    TopDownTMBuilder &toMatrixC, int64_t mBlocks, int64_t nBlocks,
    int64_t copyMPerThread, int64_t copyNPerThread, int64_t mPerBlock,
    int64_t nPerBlock, bool doSwapThreadIterSubDimsForM,
    bool doSwapThreadIterSubDimsForN, bool isBlockwise,
    SmallVector<Attribute> &transformAttr) {
  TransformMapAttr toMatrixCAttr = toMatrixC.get();
  transformAttr.push_back(toMatrixCAttr);

  auto splitAgain = TopDownTMBuilder::below(toMatrixC, toMatrixCAttr);
  {
    unsigned int idx = 0;
    if (!isBlockwise) {
      splitAgain.passThrough({"g_block", "m_block", "n_block"});
      idx += 3;
    }

    if (!doSwapThreadIterSubDimsForM) {
      splitAgain.passThrough({"gemmBlockM"}, {idx}, {"gemmBlockM"});
      idx += 1;
    } else {
      if (mPerBlock % copyMPerThread != 0)
        return failure();
      splitAgain.merge({"m_iter", "m_tid"}, {idx, idx + 1}, "gemmBlockM",
                       {copyMPerThread, mPerBlock / copyMPerThread});
      idx += 2;
    }

    if (!doSwapThreadIterSubDimsForN)
      splitAgain.passThrough({"gemmBlockN"}, {idx}, {"gemmBlockN"});
    else {
      if (nPerBlock % copyNPerThread != 0)
        return failure();
      splitAgain.merge({"n_iter", "n_tid"}, {idx, idx + 1}, "gemmBlockN",
                       {copyNPerThread, nPerBlock / copyNPerThread});
    }
  }
  TransformMapAttr splitAgainAttr = splitAgain.get();
  transformAttr.push_back(splitAgainAttr);

  auto swapBack = TopDownTMBuilder::below(splitAgain, splitAgainAttr);
  {
    unsigned int idx = 0;
    if (!isBlockwise) {
      swapBack.passThrough({"g_block", "m_block", "n_block"});
      idx = 3;
    }

    if (!doSwapThreadIterSubDimsForM)
      swapBack.passThrough({"gemmBlockM"}, {idx}, {"gemmBlockM"});
    else
      swapBack.unmerge("gemmBlockM", idx, {"m_tid", "m_iter"},
                       {mPerBlock / copyMPerThread, copyMPerThread});
    idx += 1;

    if (!doSwapThreadIterSubDimsForN)
      swapBack.passThrough({"gemmBlockN"}, {idx}, {"gemmBlockN"});
    else
      swapBack.unmerge("gemmBlockN", idx, {"n_tid", "n_iter"},
                       {nPerBlock / copyNPerThread, copyNPerThread});
  }
  TransformMapAttr swapBackAttr = swapBack.get();
  transformAttr.push_back(swapBackAttr);

  auto finalUnmerge = TopDownTMBuilder::below(swapBack, swapBackAttr);
  if (!isBlockwise) {
    finalUnmerge.passThrough({"gemmG"}, {0}, {"g_block"});
    finalUnmerge.unmerge("gemmM", 1, {"m_block", "gemmBlockM"},
                         {mBlocks, mPerBlock});
    finalUnmerge.unmerge("gemmN", 2, {"n_block", "gemmBlockN"},
                         {nBlocks, nPerBlock});
  } else {
    finalUnmerge.passThrough({"gemmM"}, {0}, {"gemmBlockM"});
    finalUnmerge.passThrough({"gemmN"}, {1}, {"gemmBlockN"});
  }
  TransformMapAttr finalUnmergeAttr = finalUnmerge.get();
  transformAttr.push_back(finalUnmergeAttr);

  return finalUnmerge;
}

Value mlir::rock::createSliceOfFirstDim(PatternRewriter &rewriter, Location loc,
                                        Value buffer, Value sliceIdx) {
  MemRefType bufType = cast<MemRefType>(buffer.getType());
  ArrayRef<int64_t> originalShape = bufType.getShape().slice(1);
  int64_t mbMemRefTypeRank = bufType.getRank();
  IntegerAttr zero = rewriter.getIndexAttr(0);
  IntegerAttr one = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(mbMemRefTypeRank, zero);
  SmallVector<OpFoldResult> sizes(mbMemRefTypeRank, one);
  SmallVector<OpFoldResult> strides(mbMemRefTypeRank, one);
  // Offset is [bufferIndex, 0 ... 0 ].
  offsets.front() = sliceIdx;
  // Sizes is [1, original_size_0 ... original_size_n ].
  for (int64_t i = 0, e = originalShape.size(); i != e; ++i)
    sizes[1 + i] = rewriter.getIndexAttr(originalShape[i]);
  auto dstMemref =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          originalShape, bufType, offsets, sizes, strides));
  Value subview = memref::SubViewOp::create(rewriter, loc, dstMemref, buffer,
                                            offsets, sizes, strides);
  return subview;
}

template <typename AllocType>
static FailureOr<AllocType> findAlloc(Value value) {
  auto *curOp = value.getDefiningOp();
  auto maybeAllocOp = dyn_cast_or_null<AllocType>(curOp);
  while (!maybeAllocOp) {
    // Keep going until the operation that defines the value is a
    // view-like operation
    if (auto viewOp = dyn_cast_or_null<ViewLikeOpInterface>(curOp)) {
      curOp = viewOp.getViewSource().getDefiningOp();
    } else if (auto extractMultiBufferOp =
                   dyn_cast_or_null<ExtractMultiBufferOp>(curOp)) {
      // If we meet an extract_multibuffer, we need to ensure that we can
      // reroute it to the the single load. Otherwise, return failure
      auto buffers = extractMultiBufferOp.getBuffers();
      auto selectIndex = dyn_cast_or_null<arith::ConstantIndexOp>(
          extractMultiBufferOp.getSelectIndex().getDefiningOp());
      if (buffers.size() > 1 && !selectIndex) {
        return failure();
      } else if (buffers.size() == 1) {
        curOp = buffers.back().getDefiningOp();
      } else {
        int64_t index = selectIndex.value() % buffers.size();
        curOp = buffers[index].getDefiningOp();
      }
    } else {
      return failure();
    }
    maybeAllocOp = dyn_cast_or_null<AllocType>(curOp);
  }
  if (!maybeAllocOp)
    return failure();

  return maybeAllocOp;
}

FailureOr<rock::GpuAllocOp> mlir::rock::findGpuAlloc(Value value) {
  return findAlloc<rock::GpuAllocOp>(value);
}

FailureOr<memref::AllocOp> mlir::rock::findMemrefAlloc(Value value) {
  return findAlloc<memref::AllocOp>(value);
}

// This is similar to findAlloc(), but this function gives you a
// list of allocs. The reason why it's a list is because if we find a
// ExtractMultiBufferOp and the index is non-static, we can't know which one
// will be chosen, so we trace back all of them.
SmallVector<rock::GpuAllocOp> mlir::rock::findAllGpuAllocs(Value value) {
  SmallVector<rock::GpuAllocOp> allocs;
  SmallVector<Operation *> worklist{value.getDefiningOp()};
  while (!worklist.empty()) {
    Operation *curOp = worklist.pop_back_val();
    auto maybeAllocOp = dyn_cast_or_null<rock::GpuAllocOp>(curOp);
    if (maybeAllocOp) {
      allocs.push_back(maybeAllocOp);
    } else {
      // Keep going until the operation that defines the value is a
      // view-like operation
      if (auto viewOp = dyn_cast_or_null<ViewLikeOpInterface>(curOp)) {
        worklist.push_back(viewOp.getViewSource().getDefiningOp());
      } else if (auto extractMultiBufferOp =
                     dyn_cast_or_null<rock::ExtractMultiBufferOp>(curOp)) {
        auto buffers = extractMultiBufferOp.getBuffers();
        auto selectIndex = dyn_cast_or_null<arith::ConstantIndexOp>(
            extractMultiBufferOp.getSelectIndex().getDefiningOp());
        if (buffers.size() > 1 && !selectIndex) {
          for (auto buffer : buffers)
            worklist.push_back(buffer.getDefiningOp());
        } else if (buffers.size() == 1) {
          worklist.push_back(buffers.back().getDefiningOp());
        } else {
          int64_t index = selectIndex.value() % buffers.size();
          worklist.push_back(buffers[index].getDefiningOp());
        }
      }
    }
  }

  return allocs;
}

FailureOr<BlockArgument> mlir::rock::findBlockArgument(Value value) {
  auto maybeBlockArg = dyn_cast_or_null<BlockArgument>(value);
  while (!maybeBlockArg) {
    // Keep going until the operation that defines the value is a
    // view-like operation
    if (auto viewOp =
            dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp())) {
      value = viewOp.getViewSource();
    } else {
      return failure();
    }
    maybeBlockArg = dyn_cast_or_null<BlockArgument>(value);
  }

  return maybeBlockArg;
}

// The rows and columns of subtile view needs to
// be transposed depending on which operand of
// gemm the view is going to be.
RegsAsMatrixSubTiles
mlir::rock::transposeSubTileViews(PatternRewriter &rewriter, Location loc,
                                  RegsAsMatrixSubTiles subTileViews) {
  ArrayAttr threadSubTile = subTileViews.threadSubTile;
  SmallVector<Attribute, 4> threadSubTileMaps =
      llvm::to_vector<4>(threadSubTile.getAsRange<Attribute>());
  {
    ArrayRef<int64_t> subTileShape = getLowerShape(threadSubTile);
    TopDownTMBuilder viewBuilder(rewriter, subTileShape, loc);
    viewBuilder.passThrough({0, 1}, {1, 0});
    threadSubTileMaps.push_back(viewBuilder.get());
  }

  ArrayAttr blockSubTile = subTileViews.blockSubTile;
  SmallVector<Attribute, 4> blockSubTileMaps =
      llvm::to_vector<4>(blockSubTile.getAsRange<Attribute>());
  {
    ArrayRef<int64_t> subTileShape = getLowerShape(blockSubTile);
    TopDownTMBuilder viewBuilder(rewriter, subTileShape, loc);
    viewBuilder.passThrough({0, 1}, {1, 0});
    blockSubTileMaps.push_back(viewBuilder.get());
  }

  ArrayAttr gridSubTile = subTileViews.gridSubTile;
  SmallVector<Attribute, 4> gridSubTileMaps =
      llvm::to_vector<4>(gridSubTile.getAsRange<Attribute>());
  {
    ArrayRef<int64_t> subTileShape = getLowerShape(gridSubTile);
    TopDownTMBuilder viewBuilder(rewriter, subTileShape, loc);
    viewBuilder.passThrough({0, 1, 2}, {0, 2, 1});
    gridSubTileMaps.push_back(viewBuilder.get());
  }

  if (subTileViews.blockSubTileTidSlice.has_value()) {
    SmallVector<Attribute, 4> blockSubTileTidSliceMaps = llvm::to_vector<4>(
        subTileViews.blockSubTileTidSlice.value().getAsRange<Attribute>());
    {
      ArrayRef<int64_t> subTileShape =
          getLowerShape(subTileViews.blockSubTileTidSlice.value());
      TopDownTMBuilder viewBuilder(rewriter, subTileShape, loc);
      viewBuilder.passThrough({0, 1}, {1, 0});
      blockSubTileTidSliceMaps.push_back(viewBuilder.get());
    }
    return RegsAsMatrixSubTiles{
        rewriter.getArrayAttr(gridSubTileMaps),
        rewriter.getArrayAttr(blockSubTileMaps),
        rewriter.getArrayAttr(threadSubTileMaps),
        rewriter.getArrayAttr(blockSubTileTidSliceMaps)};
  } else {
    return RegsAsMatrixSubTiles{rewriter.getArrayAttr(gridSubTileMaps),
                                rewriter.getArrayAttr(blockSubTileMaps),
                                rewriter.getArrayAttr(threadSubTileMaps),
                                std::nullopt};
  }
}

// Helper function to get attributes from parents
template <typename RetAttrType>
FailureOr<RetAttrType> getAttrFromOpOrParents(
    Operation *op, StringRef opAttr,
    std::optional<StringRef> maybeDialectAttr = std::nullopt) {
  StringRef dialectAttr = maybeDialectAttr.value_or(opAttr);
  Operation *func = getParentFuncOp(op);
  RetAttrType attr;
  auto getAnyAttr = [&](ArrayRef<StringRef> attrNames, Operation *op) {
    for (StringRef attrName : attrNames) {
      if (!attr) {
        attr = op->getAttrOfType<RetAttrType>(attrName);
      } else {
        return;
      }
    }
  };

  // First check for the attribute on the op
  getAnyAttr({opAttr}, op);
  if (!attr) {
    // If that fails then try checking for the attribute on the func
    getAnyAttr({opAttr, dialectAttr}, func);
  }

  // If there is no desired attribute on the func, then check the nearest parent
  // with a symbol table (covers both ModuleOp and gpu::GPUModuleOp)
  if (!attr) {
    if (auto symbolTableOp = func->getParentWithTrait<OpTrait::SymbolTable>()) {
      getAnyAttr({opAttr, dialectAttr}, symbolTableOp);
      if (attr)
        return attr;
    }
  }

  if (!attr) {
    return failure();
  }
  return attr;
}

FailureOr<IntegerAttr> mlir::rock::getGridSize(Operation *op) {
  return getAttrFromOpOrParents<IntegerAttr>(op, "grid_size");
}

FailureOr<IntegerAttr> mlir::rock::getBlockSize(Operation *op) {
  return getAttrFromOpOrParents<IntegerAttr>(op, "block_size");
}

ReassociationIndices
mlir::rock::getReassociationForFlattening(ShapedType srcTp) {
  ReassociationIndices reassociation;
  for (int i = 0, e = srcTp.getRank(); i < e; i++)
    reassociation.push_back(i);
  return reassociation;
}

Value mlir::rock::getFlattenedMemref(OpBuilder &b, Value nonFlatMemRef) {
  Location loc = nonFlatMemRef.getLoc();
  MemRefType nonFlatMemRefType = cast<MemRefType>(nonFlatMemRef.getType());
  int64_t numElements = nonFlatMemRefType.getNumElements();
  if (nonFlatMemRefType.getRank() > 1) {
    Type nonFlatMemRefElType = nonFlatMemRefType.getElementType();
    auto flatMemRefType =
        MemRefType::get({numElements}, nonFlatMemRefElType, AffineMap{},
                        nonFlatMemRefType.getMemorySpace());
    auto reassociation = getReassociationForFlattening(nonFlatMemRefType);
    return memref::CollapseShapeOp::create(b, loc, flatMemRefType,
                                           nonFlatMemRef, reassociation);
  }
  return nonFlatMemRef;
}

TypedValue<MemRefType> mlir::rock::viewBufferAs(OpBuilder &b, Value buffer,
                                                Type elementType,
                                                ArrayRef<int64_t> dimensions) {
  Location loc = buffer.getLoc();
  Value zeroByteOffset = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  auto bufferType = cast<MemRefType>(buffer.getType());
  assert(bufferType.getRank() == 1 &&
         "Buffer type must be a 1D memref for viewBufferAs");
  assert(bufferType.getElementType() == b.getI8Type() &&
         "Buffer type must be a i8 memref for viewBufferAs");

#ifdef _DEBUG
  int64_t numBytes = bufferType.getShape()[0];
  int64_t numElements = std::accumulate(dimensions.begin(), dimensions.end(),
                                        int64_t{1}, std::multiplies<>());
  int64_t elementBitWidth =
      getElementTypeOrSelf(elementType).getIntOrFloatBitWidth();
  int64_t vectorLength = isa<VectorType>(elementType)
                             ? cast<VectorType>(elementType).getNumElements()
                             : 1;
  int64_t totalBitWidthRequested = elementBitWidth * numElements * vectorLength;
  int64_t bufferBitWidth = numBytes * 8;
  assert(bufferBitWidth == totalBitWidthRequested &&
         "Can't evenly fit type into buffer");
#endif

  auto newBufferType = MemRefType::get(dimensions, elementType, nullptr,
                                       bufferType.getMemorySpace());
  auto view =
      memref::ViewOp::create(b, loc, newBufferType, buffer, zeroByteOffset,
                             /*dynamic dim sizes=*/ValueRange{});
  return TypedValue<MemRefType>(view.getResult());
}

TypedValue<MemRefType> mlir::rock::viewBufferAs(OpBuilder &b, Value buffer,
                                                Type elementType) {
  auto bufferType = cast<MemRefType>(buffer.getType());
  assert(bufferType.getRank() == 1 &&
         "Buffer type must be a 1D memref for viewBufferAs");
  assert(bufferType.getElementType() == b.getI8Type() &&
         "Buffer type must be a i8 memref for viewBufferAs");
  int64_t numBytes = bufferType.getShape()[0];
  int64_t bufferBitWidth = numBytes * 8;
  int64_t elementBitWidth =
      getElementTypeOrSelf(elementType).getIntOrFloatBitWidth();
  int64_t vectorLength = isa<VectorType>(elementType)
                             ? cast<VectorType>(elementType).getNumElements()
                             : 1;
  assert(bufferBitWidth % (elementBitWidth * vectorLength) == 0 &&
         "Can't evenly fit type into buffer");
  int64_t length = bufferBitWidth / (elementBitWidth * vectorLength);
  return viewBufferAs(b, buffer, elementType, {length});
}

Value mlir::rock::gpuAlloc(OpBuilder &b, Location loc, int64_t bufferDim,
                           Type elementType,
                           gpu::AddressSpace memoryAddressSpace) {
  auto memoryAddressSpaceAttr =
      b.getAttr<gpu::AddressSpaceAttr>(memoryAddressSpace);

  // Note: we don't need to create views for register buffers, since those won't
  // have real memory accesses at the end of the day. This is important when
  // dealing with booleans and sub-byte types.
  if (memoryAddressSpace == gpu::AddressSpace::Private) {
    auto memType = MemRefType::get({bufferDim}, elementType, AffineMap{},
                                   memoryAddressSpaceAttr);
    return GpuAllocOp::create(b, loc, memType);
  }
  auto rawMemType =
      MemRefType::get({getPackedByteSize(bufferDim, elementType)},
                      b.getI8Type(), AffineMap{}, memoryAddressSpaceAttr);
  auto buffer = GpuAllocOp::create(b, loc, rawMemType);

  return viewBufferAs(b, buffer, elementType);
}

LogicalResult mlir::rock::checkLDSSize(StringAttr arch, int64_t ldsBytes) {
  // Check for arch limitations exceede
  const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
  return success(ldsBytes <= ldsSize);
}

static void traceAlloc(memref::AllocOp buffer,
                       const BufferDependencyAnalysis &deps,
                       SmallVector<BlockArgument> &args,
                       SmallVector<OpOperand *> &genericOpOperands) {
  IRRewriter rewriter(buffer.getContext());
  std::optional<llvm::SmallVector<OpOperand *>> readersOperands =
      deps.getReaders(buffer);
  if (!readersOperands.has_value())
    return;
  for (OpOperand *readerOperand : readersOperands.value()) {
    auto readOp = dyn_cast<MemoryEffectOpInterface>(readerOperand->getOwner());
    if (!readOp)
      continue;

    if (auto genericOp = dyn_cast<linalg::GenericOp>(readerOperand->getOwner()))
      genericOpOperands.push_back(readerOperand);

    SmallVector<MemoryEffects::EffectInstance> effects;
    readOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      OpOperand *writerOperand = effect.getEffectValue<OpOperand *>();
      // Test against the write operand to guard against [MemRead, MemWrite]
      if (writerOperand && readerOperand != writerOperand &&
          isa<MemoryEffects::Write>(effect.getEffect())) {
        Value writerOperandValue = writerOperand->get();

        FailureOr<BlockArgument> maybeArg =
            findBlockArgument(writerOperandValue);
        if (succeeded(maybeArg))
          args.push_back(maybeArg.value());
        else if (memref::AllocOp writeBuffer =
                     writerOperandValue.getDefiningOp<memref::AllocOp>())
          traceAlloc(writeBuffer, deps, args, genericOpOperands);
      }
    }
  }
}

FailureOr<SmallVector<BlockArgument>>
mlir::rock::traceGemmOutputToArgs(Value matC, func::FuncOp func,
                                  const BufferDependencyAnalysis &deps) {
  if (func.getNumArguments() == 0)
    return failure();

  SmallVector<BlockArgument> args;
  auto funcArgs = func.getArguments();
  // check if matC is a kernel argument
  for (auto arg : funcArgs) {
    if (findBlockArgument(matC) == arg)
      args.push_back(arg);
  }
  assert(args.empty() || args.size() == 1);
  if (!args.empty())
    return args;

  // trace matC to its alloc
  FailureOr<memref::AllocOp> allocOp = findMemrefAlloc(matC);
  if (failed(allocOp))
    return failure();

  // trace gemm alloc to arg
  SmallVector<OpOperand *> genericOpOperands;
  traceAlloc(allocOp.value(), deps, args, genericOpOperands);
#ifdef _DEBUG
  for (auto arg : args) {
    bool containsArg =
        std::find(funcArgs.begin(), funcArgs.end(), arg) != funcArgs.end();
    assert(containsArg &&
           "Found BlockArgument does not belong to func.getArguments()");
  }
#endif

  if (!args.empty())
    return args;

  return failure();
}

FailureOr<SmallVector<OpOperand *>>
mlir::rock::traceGemmOutputToGenericOps(Value matC, func::FuncOp func,
                                        const BufferDependencyAnalysis &deps) {
  auto funcArgs = func.getArguments();
  // check if matC is a kernel argument
  for (auto arg : funcArgs) {
    // no possible linalg.generic output fusion if matC is a block arg
    if (findBlockArgument(matC) == arg)
      return {};
  }

  // trace matC to its alloc
  FailureOr<memref::AllocOp> allocOp = findMemrefAlloc(matC);
  if (failed(allocOp))
    return failure();

  // trace gemm alloc to arg, saving all genericOps
  SmallVector<OpOperand *> genericOpOperands;
  SmallVector<BlockArgument> args;
  traceAlloc(allocOp.value(), deps, args, genericOpOperands);

  return genericOpOperands;
}

/// Given a copy layout <copyDPerThread, copyKPerThread>, come up with the best
/// vectorization strategy for the layout. For instance, if the layout is <D,K>
/// = <2,16> and K is contiguous, we will vectorize by 16 along K and we will
/// loop over the other dimension
static std::pair<GemmDimension, int64_t>
bestGlobalVectorization(Value matrix, int64_t copyDPerThread,
                        int64_t copyKPerThread, GemmDimension tiebreaker,
                        int64_t kPerBlock, int64_t dPerBlock) {
  // A future commit will account for the underlying buffer's vectorization
  // here.
  VectorizationResult kVectorRes = getMaxVectorization(
      matrix, static_cast<uint32_t>(GemmDimension::K), /*inputDimLen=*/
      math_util::gcd(copyKPerThread * copyDPerThread, kPerBlock),
      matrix.getDefiningOp());
  int64_t kVectorLen = kVectorRes.max;
  VectorizationResult dVectorRes = getMaxVectorization(
      matrix, static_cast<uint32_t>(GemmDimension::MorN), /*inputDimLen=*/
      math_util::gcd(copyDPerThread * copyKPerThread, dPerBlock),
      matrix.getDefiningOp());
  int64_t dVectorLen = dVectorRes.max;

  kVectorLen = math_util::gcd(kVectorLen, copyKPerThread);
  dVectorLen = math_util::gcd(dVectorLen, copyDPerThread);
  if (kVectorLen > dVectorLen)
    return {GemmDimension::K, kVectorLen};

  if (dVectorLen > kVectorLen)
    return {GemmDimension::MorN, dVectorLen};

  return {tiebreaker, tiebreaker == GemmDimension::K ? kVectorLen : dVectorLen};
}

/// Compute a thread copy layout, i.e., how many elements a single thread (or
/// workitem) reads along K and M (independently on how we vectorize the reads).
/// This function is used when we are copying directly to LDS.
static FailureOr<std::tuple<GemmDimension, int64_t, int64_t>>
computeCopyPerThreadDirectToLDS(Value matrix, Type elementType,
                                int64_t copyPerThread, int64_t kPerBlock,
                                int64_t dPerBlock, int64_t kpack,
                                int64_t targetBits, Location loc) {
  int64_t copyKPerThread = 0;
  int64_t copyDPerThread = 0;
  // TODO: we need targetBits=96 if we want direct to LDS for f6
  if (targetBits % elementType.getIntOrFloatBitWidth() != 0)
    return failure();

  int64_t inputDimLen = targetBits / elementType.getIntOrFloatBitWidth();

  VectorizationResult dVectorRes =
      getMaxVectorization(matrix, static_cast<uint32_t>(GemmDimension::MorN),
                          inputDimLen, matrix.getDefiningOp());
  int64_t dVectorLen = dVectorRes.max;
  VectorizationResult kVectorRes =
      getMaxVectorization(matrix, static_cast<uint32_t>(GemmDimension::K),
                          inputDimLen, matrix.getDefiningOp());
  int64_t kVectorLen = kVectorRes.max;
  auto dim = (dVectorLen > kVectorLen) ? GemmDimension::MorN : GemmDimension::K;

  int64_t copyFastestDimPerThread;
  if (dim == GemmDimension::MorN) {
    copyDPerThread = math_util::gcd(dVectorLen, copyPerThread);
    copyKPerThread = copyPerThread / copyDPerThread;
    copyFastestDimPerThread = copyDPerThread;
  } else {
    copyKPerThread = math_util::gcd(kVectorLen, copyPerThread);
    copyDPerThread = copyPerThread / copyKPerThread;
    copyFastestDimPerThread = copyKPerThread;
  }

  // if the fastest dimension doesn't match inputDimLen. We can't use direct to
  // LDS.
  if (copyFastestDimPerThread != inputDimLen) {
    return failure();
  }

  if (copyKPerThread == 0 || copyDPerThread == 0) {
    return failure();
  }
  if (kPerBlock < copyKPerThread || dPerBlock < copyDPerThread) {
    return failure();
  }
  return std::make_tuple(dim, copyKPerThread, copyDPerThread);
}

/// Compute a thread copy layout, i.e., how many elements a single thread (or
/// workitem) reads along K and M (independently on how we vectorize the reads)
static FailureOr<std::tuple<GemmDimension, int64_t, int64_t>>
computeCopyPerThread(Type elementType, int64_t copyPerThread, int64_t kPerBlock,
                     int64_t dPerBlock, int64_t kpack, Location loc) {

  // By default, we try to maximize the LDS store vectorization. So we will try
  // to read as many elements as possible along the contiguous dimension in LDS
  // and `copyPerThread/elements` in the other dimension
  int64_t maxVlen = 128 / elementType.getIntOrFloatBitWidth();
  int64_t copyKPerThread = 0;
  int64_t copyDPerThread = 0;

  GemmDimension dim;
  if (kpack == 1) {
    copyDPerThread = math_util::gcd(maxVlen, copyPerThread);
    copyKPerThread = copyPerThread / copyDPerThread;
    dim = GemmDimension::MorN;
  } else {
    copyKPerThread = math_util::gcd(maxVlen, copyPerThread);
    copyDPerThread = copyPerThread / copyKPerThread;
    dim = GemmDimension::K;
  }

  if (copyKPerThread == 0 || copyDPerThread == 0) {
    return emitError(loc) << "gemmA copy size too small,"
                          << " copyKPerThread: " << copyKPerThread
                          << " copyDPerThread: " << copyDPerThread << "\n";
  }
  if (kPerBlock < copyKPerThread || dPerBlock < copyDPerThread) {
    return mlir::emitError(loc)
           << "gemmA per thread copy smaller than per"
           << " block copy, incoherent tuning parameters\n";
  }
  return std::make_tuple(dim, copyKPerThread, copyDPerThread);
}

FailureOr<Value> mlir::rock::wrapLDSBufferForStore(
    OpBuilder &b, Location loc, Value buffer, Type ldsReadType, int64_t kOuter,
    StringRef dName, int64_t d, int64_t kPerThread, int64_t dPerThread,
    bool rotateDWithK) {
  MemRefType bufferType = cast<MemRefType>(buffer.getType());
  ArrayRef<int64_t> bufferShape = bufferType.getShape();
  Type dataType = ldsReadType;
  if (bufferShape.size() != 1)
    return emitError(loc, "Expected a flat buffer");
  int64_t kpack = 1;
  if (auto vectorDataType = dyn_cast<VectorType>(dataType)) {
    kpack = vectorDataType.getNumElements();
    dataType = vectorDataType.getElementType();
  }
  if (bufferShape[0] != getPackedByteSize(kOuter * d * kpack, dataType)) {
    return emitError(loc, "LDS buffer should have ")
           << getPackedByteSize(kOuter * d * kpack, dataType)
           << " elements but has " << bufferShape[0];
  }
  int64_t kpackPerThread = std::min(kPerThread, kpack);
  assert(kpack % kpackPerThread == 0);
  int64_t threadsPerKpack = kpack / kpackPerThread;

  Type ldsWriteType = vectorTypeOrSelf(dataType, kpackPerThread);
  auto typedBuffer = viewBufferAs(b, buffer, ldsWriteType);

  TopDownTMBuilder mergeKpack{
      b, {"k", "d"}, {kOuter * threadsPerKpack * kpackPerThread, d}};
  mergeKpack.merge({"k_outer", "kpack_idx", "kpack_vec"}, {0, 2, 3}, "k",
                   {kOuter, threadsPerKpack, kpackPerThread});
  mergeKpack.merge({dName}, {1}, "d", {d});

  TransformMapAttr mergeKpackAttr = mergeKpack.get();
  SmallVector<Attribute> transformAttrs{mergeKpackAttr};

  // Rotate the buffer if necessary to minimize bank conflicts. Rotating the
  // buffer has the benefit of minimizing bank conflicts when we are transposing
  // the matrix from global to LDS. I.e., instead of storing different items in
  // position (0,0), (1,0), (2,0), ... we store it in (0,0), (1,1), (2, 2), ...
  int64_t stride = (kpack == 1 ? dPerThread : 1);
  TopDownTMBuilder reshapeBuf = rotateIf(
      rotateDWithK, mergeKpack, mergeKpackAttr, stride, dName, d, 1, "k_outer",
      kOuter, {"k_outer"}, {"kpack_idx", "kpack_vec"}, transformAttrs);

  reshapeBuf.unmerge("raw", 0, {"k_outer", dName, "kpack_idx"},
                     {kOuter, d, threadsPerKpack});
  reshapeBuf.ignore("kpack_vec");
  TransformMapAttr reshapeBufAttr = reshapeBuf.get();
  transformAttrs.push_back(reshapeBufAttr);

  ArrayAttr asMatrix = b.getArrayAttr(transformAttrs);
  return transform(b, typedBuffer, asMatrix);
}

FailureOr<VectorDimInfo>
mlir::rock::getVectorDim(Location loc, Value matrix, Type elemType,
                         int64_t blockSize, int64_t kPerBlock,
                         int64_t dPerBlock, int64_t kpack, bool directToLDS) {
  FailureOr<std::tuple<GemmDimension, int64_t, int64_t>> maybeCopyDPerThread =
      failure();
  int64_t copyPerThread = (kPerBlock * dPerBlock) / blockSize;
  if (directToLDS) {
    // TODO: Implement this for WMMA.
    StringRef archValue = rock::getArchValue(matrix.getDefiningOp());
    if (archValue.contains("gfx1250")) {
      return emitError(loc) << "AsyncDirectToLDS is not implemented for WMMA";
    }
    StringAttr arch = getArchValue(matrix.getDefiningOp());
    auto features = rock::lookupArchInfo(arch).defaultFeatures;
    bool directToLDS128b =
        bitEnumContainsAll(features, GemmFeatures::direct_to_lds_128b);
    bool directToLDS32b =
        bitEnumContainsAll(features, GemmFeatures::direct_to_lds_32b);
    assert(directToLDS128b || directToLDS32b);

    // For direct to LDS, we will try if we can load 128b per thread first.
    // If not possible, we will try 32b. If not possible, we can't use direct to
    // LDS.
    if (directToLDS128b)
      maybeCopyDPerThread = computeCopyPerThreadDirectToLDS(
          matrix, elemType, copyPerThread, kPerBlock, dPerBlock, kpack, 128,
          loc);

    if (failed(maybeCopyDPerThread) && directToLDS32b)
      maybeCopyDPerThread =
          computeCopyPerThreadDirectToLDS(matrix, elemType, copyPerThread,
                                          kPerBlock, dPerBlock, kpack, 32, loc);
  } else {
    maybeCopyDPerThread = computeCopyPerThread(
        elemType, copyPerThread, kPerBlock, dPerBlock, kpack, loc);
  }
  if (failed(maybeCopyDPerThread))
    return failure();

  GemmDimension vectorDim = std::get<0>(maybeCopyDPerThread.value());
  int64_t copyKPerThread = std::get<1>(maybeCopyDPerThread.value());
  int64_t copyDPerThread = std::get<2>(maybeCopyDPerThread.value());
  int64_t vectorLen;
  GemmDimension vectorTiebreaker =
      (kpack > 1) ? GemmDimension::K : GemmDimension::MorN;
  if (directToLDS) {
    // with direct to LDS, we will keep the same fastest dimension
    // computeCopyPerThreadDirectToLDS chose.
    if (vectorDim == GemmDimension::K) {
      VectorizationResult kVectorRes = getMaxVectorization(
          matrix, static_cast<uint32_t>(GemmDimension::K), /*inputDimLen=*/
          math_util::gcd(copyKPerThread * copyDPerThread, kPerBlock),
          matrix.getDefiningOp());
      vectorLen = math_util::gcd(kVectorRes.max, copyKPerThread);
    } else {
      VectorizationResult dVectorRes = getMaxVectorization(
          matrix, static_cast<uint32_t>(GemmDimension::MorN), /*inputDimLen=*/
          math_util::gcd(copyDPerThread * copyKPerThread, dPerBlock),
          matrix.getDefiningOp());
      vectorLen = math_util::gcd(dVectorRes.max, copyDPerThread);
    }
  } else {
    // Find the best way of vectorizing the layout
    std::tie(vectorDim, vectorLen) =
        bestGlobalVectorization(matrix, copyDPerThread, copyKPerThread,
                                vectorTiebreaker, kPerBlock, dPerBlock);
  }

  return VectorDimInfo{vectorDim, vectorLen, copyKPerThread, copyDPerThread,
                       vectorTiebreaker};
}

std::optional<int64_t> mlir::rock::getWorkgroupMemorySize(MemRefType type) {
  auto memSpaceValue =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace()).getValue();
  if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
    return getPackedByteSize(type.getNumElements(), type.getElementType());
  }
  return std::nullopt;
}

FailureOr<ThreadwiseReadIntoLoopInfo> mlir::rock::getThreadwiseReadIntoLoopInfo(
    const ThreadwiseReadIntoLoopConfigInput &input) {
  ThreadwiseReadIntoLoopInfo info;
  info.numValues = input.numValues;
  info.vectorDstLen = 1;
  info.vectorSrcLen = 1;
  info.srcStride = 1;
  info.dstVectorType = nullptr;

  Type elementType = input.elementType;
  Type loadType;

  if (input.isSrcVectorBuffer) {
    loadType = elementType;
    info.vectorSrcLen = dyn_cast<VectorType>(elementType).getNumElements();
    elementType = dyn_cast<VectorType>(elementType).getElementType();
    // Here we would call collapseContiguousMerges
    info.srcStride = 1;
    if (!input.isDstVectorBuffer)
      info.numValues = info.numValues / info.vectorSrcLen;
  } else {
    VectorizationResult vectorSrcRes = getMaxVectorization(
        input.sourceView, input.extraIdxCount, /*inputDimLen=*/info.numValues);
    info.vectorSrcLen = vectorSrcRes.max;
    if (input.isGlobalToLDS &&
        info.vectorSrcLen > input.maxGlobalToLDSVectorLen) {
      LLVM_DEBUG(llvm::dbgs()
                 << "getThreadwiseReadIntoLoopInfo:"
                 << "Constraining vectorization from " << info.vectorSrcLen
                 << " to " << input.maxGlobalToLDSVectorLen.value()
                 << " for Global-to-LDS hardware limits\n");
      info.vectorSrcLen = input.maxGlobalToLDSVectorLen.value();
    }
    // Here we would call collapseContiguousMerges
    info.srcStride = info.vectorSrcLen;
    loadType = vectorTypeOrSelf(elementType, info.vectorSrcLen);
  }

  // Force the dynamic validity case down to a vectorization of 1
  if (input.hasDynamicValidities) {
    info.vectorSrcLen = 1;
    info.srcStride = 1;
    loadType = elementType;
  }

  if (input.isDstVectorBuffer) {
    info.dstVectorType =
        dyn_cast<VectorType>(input.dstBufferType.getElementType());
    info.vectorDstLen =
        dyn_cast<VectorType>(info.dstVectorType).getNumElements();
    info.numValues = info.numValues * info.vectorDstLen;
    if (input.isSrcVectorBuffer) {
      info.numValues = info.numValues / info.vectorSrcLen;
    }
  }

  info.elementType = elementType;
  info.loadType = loadType;
  return info;
}

FailureOr<int64_t>
mlir::rock::predictThreadwiseReadIntoLoopCount(ThreadwiseReadIntoOp op) {
  Value sourceView = op.getSource();
  Value dest = op.getDest();

  auto sourceViewType = cast<MemRefType>(sourceView.getType());
  MemRefType dstBufferType = cast<MemRefType>(dest.getType());

  bool isSrcVectorBuffer = isa<VectorType>(sourceViewType.getElementType());
  bool isDstVectorBuffer = isa<VectorType>(dstBufferType.getElementType());

  // Collect transforms from the source value without creating operations
  SmallVector<TransformMapAttr> transforms;
  Value buffer;
  std::tie(buffer, std::ignore) = untransform(sourceView, transforms);

  // Get buffer type
  auto srcBufferType = dyn_cast<MemRefType>(buffer.getType());
  if (!srcBufferType) {
    return failure();
  }

  // Determine address spaces
  gpu::AddressSpace srcAddrSpace = gpu::AddressSpace::Global;
  if (srcBufferType.getMemorySpace()) {
    srcAddrSpace =
        cast<gpu::AddressSpaceAttr>(srcBufferType.getMemorySpace()).getValue();
  }
  gpu::AddressSpace dstAddrSpace = gpu::AddressSpace::Private;
  if (dstBufferType.getMemorySpace()) {
    dstAddrSpace =
        cast<gpu::AddressSpaceAttr>(dstBufferType.getMemorySpace()).getValue();
  }
  bool isGlobalToLDS = srcAddrSpace == gpu::AddressSpace::Global &&
                       dstAddrSpace == gpu::AddressSpace::Workgroup;

  int64_t numValues = dstBufferType.getNumElements();

  // For GlobalToLDS, get numValues from the transform maps
  // We need to combine extraViews with existing transforms
  ArrayAttr extraViews = op.getExtraViews();
  if (isGlobalToLDS) {
    // Combine extraViews with existing transforms to find the top map
    SmallVector<TransformMapAttr> allTransforms;
    if (extraViews) {
      for (auto attr : extraViews.getAsRange<TransformMapAttr>()) {
        allTransforms.push_back(attr);
      }
    }
    allTransforms.append(transforms.begin(), transforms.end());

    if (allTransforms.empty()) {
      return failure();
    }
    TransformMapAttr topMap = allTransforms[0];
    numValues = topMap.getUpperBounds().asArrayRef().back();
    if (isSrcVectorBuffer || isDstVectorBuffer) {
      return failure();
    }
  }

  size_t extraIdxCount = op.getExtraIndices().size();
  auto elementType = sourceViewType.getElementType();

  std::optional<int64_t> maxGlobalToLDSVectorLen;
  if (isGlobalToLDS) {
    StringAttr arch = rock::getArchValue(op);
    auto archInfo = rock::lookupArchInfo(arch);
    Type scalarElementType = getElementTypeOrSelf(elementType);
    int64_t elementBitWidth = scalarElementType.getIntOrFloatBitWidth();
    maxGlobalToLDSVectorLen = archInfo.getMaxLDSVectorLength(elementBitWidth);
  }

  ThreadwiseReadIntoLoopConfigInput loopInput{
      sourceView,        dstBufferType,
      extraIdxCount,     elementType,
      numValues,         isSrcVectorBuffer,
      isDstVectorBuffer, !op.getDynamicValidities().empty(),
      isGlobalToLDS,     maxGlobalToLDSVectorLen};
  auto maybeLoopInfo = getThreadwiseReadIntoLoopInfo(loopInput);
  if (failed(maybeLoopInfo))
    return failure();

  ThreadwiseReadIntoLoopInfo info = maybeLoopInfo.value();
  if (info.srcStride == 0)
    return failure();
  return info.numValues / info.srcStride;
}
