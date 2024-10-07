//===- loweringUtils.cpp - Rock utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::rock;

bool mlir::rock::isValidBlockSize(int64_t blockSize, int64_t kPerBlock,
                                  int64_t mPerBlock, int64_t nPerBlock) {
  int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
  int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;
  return (aCopyPerThread != 0 && bCopyPerThread != 0);
}

bool mlir::rock::isWrWAtomicKernel(GemmFeatures features, Type dataType,
                                   bool requiredPadding) {
  return isAccel(features) &&
         bitEnumContainsAll(features, GemmFeatures::atomic_add) &&
         (dataType.isF32() || dataType.isF16()) && !requiredPadding;
}

bool mlir::rock::isAccel(GemmFeatures features) {
  return bitEnumContainsAny(features, GemmFeatures::wmma | GemmFeatures::mfma);
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

SmallVector<int64_t>
mlir::rock::backwardDataKernelIds(ArrayRef<int64_t> strideDims,
                                  ArrayRef<int64_t> dilationDims,
                                  ArrayRef<int64_t> filterDims) {
  assert(strideDims.size() == dilationDims.size());
  SmallVector<int64_t, 5> gcdStrideDilations;
  for (const auto &[stride, dilation] : zip(strideDims, dilationDims))
    gcdStrideDilations.push_back(math_util::gcd(stride, dilation));

  SmallVector<int64_t, 5> filTilda;
  for (const auto &[stride, gcdSD] : zip(strideDims, gcdStrideDilations))
    filTilda.push_back(stride / gcdSD);

  // Heuristic to determine if every pixel in the output would be written by the
  // backward data convolution algorithm.
  auto isEveryPixelWritten = [&]() -> bool {
    bool result = true;
    for (const auto &[stride, dilation, filterSize] :
         zip(strideDims, dilationDims, filterDims)) {
      if (!(dilation == 1 && stride <= filterSize))
        result = false;
    }
    return result;
  };
  bool needZeroInitKernel = !isEveryPixelWritten();

  llvm::SmallVector<int64_t> kernelIds;
  if (needZeroInitKernel)
    kernelIds.push_back(-1);

  // Populate the kernel IDs according to the current backward data convolution
  // algorithm implementation.
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
                                 bool isKContigousDim) {
  if (isKContigousDim) {
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
                                  bool isKContigousDim) {
  if (isKContigousDim) {
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
    int64_t dPerThread, bool isKContigousDim, int64_t padK,
    int64_t prevKperBlock, int64_t effectiveBlockSize) {
  if (prevKperBlock == 0)
    prevKperBlock = kPerBlock;
  if (effectiveBlockSize == 0)
    effectiveBlockSize = blockSize;

  if (dName != "m" && dName != "n") {
    return emitError(loc, "expected dName to be m or n but got " + dName);
  }
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  StringRef otherBlockDim = dName == "m" ? "n_block" : "m_block";

  MemRefType matrixType = cast<MemRefType>(globalBuffer.getType());
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
  int64_t kGlobal = matrixShape[1];
  if (padK > 0) {
    kGlobal += padK;
  }
  int64_t dGlobal = matrixShape[2];

  int64_t kIters = kGlobal / kPerBlock;
  int64_t dataPerThread = (kPerBlock * dPerBlock) / effectiveBlockSize;

  SmallString<8> dIterName = llvm::formatv("{0}_iter", dName);
  SmallString<8> dThreadName = llvm::formatv("{0}_thread", dName);

  // Note: (kThreads * dThreads) = (kPerBlock * dPerBlock) / dataPerThread) =
  // blockSize

  int64_t prevKThreads = prevKperBlock / kPerThread;
  int64_t kThreads = kPerBlock / kPerThread;
  int64_t dThreads = dPerBlock / dPerThread;

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
    makeLoadRegsTidMerge(gridwiseSplitId, dThreadName, dThreads, prevKThreads,
                         {4, 5}, isKContigousDim);
    makeLoadRegsIterMerge(gridwiseSplitId, dIterName, dPerThread, kPerThread,
                          {6, 7}, isKContigousDim);
    TransformMapAttr splitIdAttr = gridwiseSplitId.get();

    auto padkThreadsLastIter =
        TopDownTMBuilder::below(gridwiseSplitId, splitIdAttr);
    if (prevKThreads == kThreads) {
      padkThreadsLastIter.passThrough(
          {"k_loop", bidGridOrder[0], bidGridOrder[1], bidGridOrder[2],
           "k_thread", "k_iter", dThreadName, dIterName});

    } else {
      padkThreadsLastIter.passThrough({"k_loop", bidGridOrder[0],
                                       bidGridOrder[1], bidGridOrder[2],
                                       "k_iter", dThreadName, dIterName});
      padkThreadsLastIter.pad("k_thread", "k_thread", 0,
                              prevKThreads - kThreads);
    }
    auto padkThreadsLastIterAttr = padkThreadsLastIter.get();

    auto toGlobalIdx =
        TopDownTMBuilder::below(padkThreadsLastIter, padkThreadsLastIterAttr);
    toGlobalIdx.passThrough({"g"}, {0}, {"g_block"});
    toGlobalIdx.unmerge("k", 1, {"k_loop", "k_thread", "k_iter"},
                        {kGlobal / kPerBlock, kThreads, kPerThread});

    toGlobalIdx.unmerge(dName, 2, {thisBlockDim, dThreadName, dIterName},
                        {dGlobal / dPerBlock, dThreads, dPerThread});

    toGlobalIdx.ignore(otherBlockDim);
    TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();

    if (padK == 0) {
      gpuViews.gridSubTile = b.getArrayAttr(
          {splitIdAttr, padkThreadsLastIterAttr, toGlobalIdxAttr});
    } else {
      auto padder = TopDownTMBuilder::below(toGlobalIdx, toGlobalIdxAttr);
      padder.passThrough({"g"}, {0}, {"g"});
      padder.passThrough({dName}, {2}, {dName});
      padder.pad("k", "k", 0, padK);
      gpuViews.gridSubTile =
          b.getArrayAttr({splitIdAttr, padkThreadsLastIterAttr, toGlobalIdxAttr,
                          padder.get()});
    }
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
    int64_t dPerThread, int64_t kpack, bool isKContigousDim,
    bool doSwapThreadIterSubDimsForD, int64_t padK, int64_t prevKperBlock,
    int64_t effectiveBlockSize) {
  if (prevKperBlock == 0)
    prevKperBlock = kPerBlock;
  if (effectiveBlockSize == 0)
    effectiveBlockSize = blockSize;

  if (dName != "m" && dName != "n") {
    return emitError(loc, "expected dName to be m or n but got " + dName);
  }
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  StringRef otherBlockDim = dName == "m" ? "n_block" : "m_block";

  MemRefType matrixType = cast<MemRefType>(globalBuffer.getType());
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
  int64_t kGlobal = matrixShape[1];
  if (padK > 0) {
    kGlobal += padK;
  }
  int64_t dGlobal = matrixShape[2];

  int64_t kIters = kGlobal / kPerBlock;
  int64_t dataPerThread = (kPerBlock * dPerBlock) / effectiveBlockSize;

  SmallString<8> dIterName = llvm::formatv("{0}_iter", dName);
  SmallString<8> dThreadName = llvm::formatv("{0}_thread", dName);

  // Note: (kThreads * dThreads) = (kPerBlock * dPerBlock) / dataPerThread) =
  // blockSize
  int64_t prevKThreads = prevKperBlock / kPerThread;
  int64_t kThreads = kPerBlock / kPerThread;
  int64_t dThreads = dPerBlock / dPerThread;

  int64_t kpackPerThread = std::min(kPerThread, kpack);
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
    makeLoadRegsTidMerge(gridwiseSplitId, dThreadName, dThreads, prevKThreads,
                         {4, 5}, isKContigousDim);
    gridwiseSplitId.merge({"kouterPerThread", dIterName, "kpackPerThread"},
                          {6, 7, 8}, "iter",
                          {kOuterPerThread, dPerThread, kpackPerThread});
    TransformMapAttr splitIdAttr = gridwiseSplitId.get();

    auto padkThreadsLastIter =
        TopDownTMBuilder::below(gridwiseSplitId, splitIdAttr);
    if (prevKThreads == kThreads) {
      padkThreadsLastIter.passThrough(
          {"k_loop", bidGridOrder[0], bidGridOrder[1], bidGridOrder[2],
           "k_thread", dThreadName, "kouterPerThread", dIterName,
           "kpackPerThread"});

    } else {
      padkThreadsLastIter.passThrough(
          {"k_loop", bidGridOrder[0], bidGridOrder[1], bidGridOrder[2],
           dThreadName, "kouterPerThread", dIterName, "kpackPerThread"});
      padkThreadsLastIter.pad("k_thread", "k_thread", 0,
                              prevKThreads - kThreads);
    }
    auto padkThreadsLastIterAttr = padkThreadsLastIter.get();

    auto toGlobalIdx =
        TopDownTMBuilder::below(padkThreadsLastIter, padkThreadsLastIterAttr);
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

    if (padK == 0) {
      gpuViews.gridSubTile = b.getArrayAttr(
          {splitIdAttr, padkThreadsLastIterAttr, toGlobalIdxAttr});
    } else {
      auto padder = TopDownTMBuilder::below(toGlobalIdx, toGlobalIdxAttr);
      padder.passThrough({"g"}, {0}, {"g"});
      padder.passThrough({dName}, {2}, {dName});
      padder.pad("k", "k", 0, padK);
      gpuViews.gridSubTile =
          b.getArrayAttr({splitIdAttr, padkThreadsLastIterAttr, toGlobalIdxAttr,
                          padder.get()});
    }
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
  return b.create<TransformOp>(loc, matrix, normalizeAttr);
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
  return b.create<TransformOp>(loc, matrix, padAttr);
}

TopDownTMBuilder mlir::rock::swapThreadIdAndIteration(
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
      splitAgain.merge({"m_iter", "m_tid"}, {idx, idx + 1}, "gemmBlockM",
                       {copyMPerThread, mPerBlock / copyMPerThread});
      idx += 2;
    }

    if (!doSwapThreadIterSubDimsForN)
      splitAgain.passThrough({"gemmBlockN"}, {idx}, {"gemmBlockN"});
    else
      splitAgain.merge({"n_iter", "n_tid"}, {idx, idx + 1}, "gemmBlockN",
                       {copyNPerThread, nPerBlock / copyNPerThread});
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
  Value subview = rewriter.create<memref::SubViewOp>(loc, dstMemref, buffer,
                                                     offsets, sizes, strides);
  return subview;
}

template <typename AllocType>
FailureOr<AllocType> findAlloc(Value value) {
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
        int64_t index = selectIndex.value();
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

std::optional<int64_t> mlir::rock::computeConstDiff(Value l, Value u) {
  IntegerAttr clb, cub;
  if (matchPattern(l, m_Constant(&clb)) && matchPattern(u, m_Constant(&cub))) {
    llvm::APInt lbValue = clb.getValue();
    llvm::APInt ubValue = cub.getValue();
    return (ubValue - lbValue).getSExtValue();
  }
  return std::nullopt;
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

template <typename RetAttrType>
static FailureOr<RetAttrType> getAttrFromOpOrParents(
    Operation *op, StringRef opAttr,
    std::optional<StringRef> maybeDialectAttr = std::nullopt) {
  StringRef dialectAttr = maybeDialectAttr.value_or(opAttr);
  Operation *func;
  if (isa<func::FuncOp, gpu::GPUFuncOp>(op)) {
    func = op;
  } else {
    func = op->getParentOfType<func::FuncOp>();
    if (!func) {
      func = op->getParentOfType<gpu::GPUFuncOp>();
    }
  }
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
  getAnyAttr({opAttr}, op);
  if (!attr) {
    getAnyAttr({opAttr, dialectAttr}, func);
  }
  if (!attr) {
    auto mod = func->getParentOfType<ModuleOp>();
    getAnyAttr({opAttr, dialectAttr}, mod);
  }
  if (!attr) {
    if (auto mod = func->getParentOfType<gpu::GPUModuleOp>()) {
      getAnyAttr({opAttr, dialectAttr}, mod);
    }
  }
  if (!attr) {
    return failure();
  }
  return attr;
}

FailureOr<StringAttr> mlir::rock::getArch(Operation *op) {
  return getAttrFromOpOrParents<StringAttr>(op, "arch", "mhal.arch");
}

FailureOr<int64_t> mlir::rock::getNumCU(Operation *op) {
  FailureOr<StringAttr> maybeArch = getArch(op);
  if (failed(maybeArch)) {
    return failure();
  }
  StringAttr arch = maybeArch.value();
  FailureOr<IntegerAttr> maybeNumCU =
      getAttrFromOpOrParents<IntegerAttr>(op, "num_cu");
  if (failed(maybeNumCU)) {
    return failure();
  }
  IntegerAttr numCU = maybeNumCU.value();
  AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (numCU.getValue().getSExtValue() < archInfo.minNumCU) {
    return op->emitError() << "num_cu=" << numCU
                           << " cannot be lower than arch minNumCU="
                           << archInfo.minNumCU;
  }
  return numCU.getValue().getSExtValue();
}

FailureOr<UnitAttr> mlir::rock::getReverseGrid(Operation *op) {
  return getAttrFromOpOrParents<UnitAttr>(
      op, rock::ReverseGridAttrAttr::getMnemonic());
}

FailureOr<IntegerAttr> mlir::rock::getGridSize(Operation *op) {
  return getAttrFromOpOrParents<IntegerAttr>(op, "grid_size");
}

FailureOr<IntegerAttr> mlir::rock::getBlockSize(Operation *op) {
  return getAttrFromOpOrParents<IntegerAttr>(op, "block_size");
}

AffineMap mlir::rock::getIdxReversalMap(OpBuilder &b) {
  auto dimExpr = mlir::getAffineDimExpr(0, b.getContext());
  auto dimSizeExpr = mlir::getAffineSymbolExpr(0, b.getContext());
  auto affineMap = mlir::AffineMap::get(1, 1, dimSizeExpr - 1 - dimExpr);
  return affineMap;
}

ReassociationIndices
mlir::rock::getReassociationForFlattening(ShapedType srcTp) {
  ReassociationIndices reassociation;
  for (int i = 0, e = srcTp.getRank(); i < e; i++)
    reassociation.push_back(i);
  return reassociation;
}

TypedValue<MemRefType> mlir::rock::viewBufferAs(OpBuilder &b, Value buffer,
                                                Type type) {
  Location loc = buffer.getLoc();
  Value zeroByteOffset = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  auto bufferType = cast<MemRefType>(buffer.getType());
  int64_t byteWidth = getByteWidth(type);
  int64_t numBytes = bufferType.getShape()[0];
  assert(numBytes % byteWidth == 0 && "Can't evenly fit type into buffer");
  int64_t length = numBytes / byteWidth;
  auto newBufferType = bufferType.cloneWith({length}, type);
  auto view =
      b.create<memref::ViewOp>(loc, newBufferType, buffer, zeroByteOffset,
                               /*dynamic dim sizes=*/ValueRange{});
  return TypedValue<MemRefType>(view.getResult());
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
    return b.create<GpuAllocOp>(loc, memType);
  }
  auto rawMemType =
      MemRefType::get({bufferDim * getByteWidth(elementType)}, b.getI8Type(),
                      AffineMap{}, memoryAddressSpaceAttr);
  auto buffer = b.create<GpuAllocOp>(loc, rawMemType);

  return viewBufferAs(b, buffer, elementType);
}

LogicalResult mlir::rock::checkLDSSize(StringAttr arch, int64_t ldsBytes) {
  // Check for arch limitations exceede
  const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
  return success(ldsBytes <= ldsSize);
}
