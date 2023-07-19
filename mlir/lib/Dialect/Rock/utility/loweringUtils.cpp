//===- loweringUtils.cpp - Rock utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::rock;

bool mlir::rock::isWrWAtomicKernel(GemmFeatures features, Type dataType,
                                   bool requiredPadding) {
  return isAccel(features) &&
         bitEnumContainsAll(features, GemmFeatures::atomic_add) &&
         (dataType.isF32() || dataType.isF16()) && !requiredPadding;
}

bool mlir::rock::isAccel(GemmFeatures features) {
  return bitEnumContainsAny(features, GemmFeatures::wmma | GemmFeatures::mfma);
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
mlir::rock::backwardDataKernelIds(int64_t strideHeight, int64_t strideWidth,
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

  llvm::SmallVector<int64_t> kernelIds;
  if (needZeroInitKernel)
    kernelIds.push_back(-1);

  // Populate the kernel IDs according to the current backward data convolution
  // algorithm implementation.
  for (int64_t kernelId = 0; kernelId < yTilda * xTilda; ++kernelId) {
    // gemmK size is different for each GEMM
    const int64_t iYTilda = kernelId / xTilda;
    const int64_t iXTilda = kernelId % xTilda;

    int64_t yDotSlice = math_util::integer_divide_ceil(y - iYTilda, yTilda);
    int64_t xDotSlice = math_util::integer_divide_ceil(x - iXTilda, xTilda);
    // gemmK must > 0, otherwise not need to run
    if (yDotSlice * xDotSlice > 0) {
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

static SmallVector<unsigned> getOutDimIndices(int64_t outDimStartIdx, int64_t newDimSize){
  SmallVector<unsigned> outDims;
  for (int i=0; i < newDimSize; i++){
    outDims.push_back(outDimStartIdx + i);
  }
  return std::move(outDims);
}

static std::tuple<SmallVector<StringRef>,SmallVector<int64_t>> getCombinedOrder(const LowerDimPartInfo& lhs, const LowerDimPartInfo& rhs){
  llvm::SmallDenseMap<int64_t,std::tuple<StringRef,int64_t>> ordered;
  for(auto [dimPartName, dimPartOrderIdx, dimPartSize] : llvm::zip(lhs.bottomDimPartNames, lhs.bottomDimPartOrder, lhs.bottomDimPartSizes)){
    ordered[dimPartOrderIdx] = {dimPartName, dimPartSize};
  }
  for(auto [dimPartName, dimPartOrderIdx, dimPartSize] : llvm::zip(rhs.bottomDimPartNames, rhs.bottomDimPartOrder, rhs.bottomDimPartSizes)){
    ordered[dimPartOrderIdx] = {dimPartName, dimPartSize};
  }
  SmallVector<StringRef> orderedNames;
  SmallVector<int64_t> orderedSizes;
  for(int i=0; i < ordered.size(); i++){
    orderedNames.push_back(std::get<0>(ordered[i]));
    orderedSizes.push_back(std::get<1>(ordered[i]));
  }
  return {std::move(orderedNames), std::move(orderedSizes)};
}

FailureOr<GPUViews> mlir::rock::createGemmInputViewsFromGlobal(
    OpBuilder &b, Location loc, Value globalBuffer, StringRef dName,
    ArrayRef<StringRef> bidGridOrder, ArrayRef<int64_t> bidGridLengths,
    int64_t gridSize, int64_t blockSize, int64_t kPerBlock, int64_t dPerBlock,
    LowerDimPartInfo kDimTidPartInfo, 
    LowerDimPartInfo dDimTidPartInfo,
    LowerDimPartInfo kDimIterPartInfo, 
    LowerDimPartInfo dDimIterPartInfo) {
  if (dName != "m" && dName != "n") {
    return emitError(loc, "expected dName to be m or n but got " + dName);
  }
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  StringRef otherBlockDim = dName == "m" ? "n_block" : "m_block";

  MemRefType matrixType = globalBuffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
  int64_t kGlobal = matrixShape[1];
  int64_t dGlobal = matrixShape[2];

  int64_t kIters = kGlobal / kPerBlock;
  int64_t dataPerThread = (kPerBlock * dPerBlock) / blockSize;

  // Note: (kThreads * dThreads) = (kPerBlock * dPerBlock) / dataPerThread) =
  // blockSize
  GPUViews gpuViews;
  {
    TopDownTMBuilder gridwiseSplitId(
        b, {"k_loop", "g_block", "m_block", "n_block", "tid", "iter"},
        {kIters, bidGridLengths[0], bidGridLengths[1], bidGridLengths[2],
         blockSize, dataPerThread},
        loc);
    gridwiseSplitId.passThrough({"k_loop", "g_block", "m_block", "n_block"});
    auto [tidDimNames, tidDimSizes] = getCombinedOrder(kDimTidPartInfo, dDimTidPartInfo);
    gridwiseSplitId.merge(tidDimNames, 
                          getOutDimIndices(4, tidDimNames.size()),
                          "tid", 
                          tidDimSizes);
    auto [iterDimNames, iterDimSizes] = getCombinedOrder(kDimIterPartInfo, dDimIterPartInfo);
    gridwiseSplitId.merge(iterDimNames, 
                          getOutDimIndices(4 + tidDimNames.size(), iterDimNames.size()),
                          "iter", 
                          iterDimSizes);

    TransformMapAttr splitIdAttr = gridwiseSplitId.get();
    auto toGlobalIdx = TopDownTMBuilder::below(gridwiseSplitId, splitIdAttr);
    toGlobalIdx.passThrough({"g"}, {0}, {"g_block"});
    {
      SmallVector<StringRef, 3> kSubDimNames {"k_loop"};
      kSubDimNames.append(kDimTidPartInfo.bottomDimPartNames.begin(), kDimTidPartInfo.bottomDimPartNames.end());
      kSubDimNames.append(kDimIterPartInfo.bottomDimPartNames.begin(), kDimIterPartInfo.bottomDimPartNames.end());
      SmallVector<int64_t, 3> kSubDimSizes {kIters};
      kSubDimSizes.append(kDimTidPartInfo.bottomDimPartSizes.begin(), kDimTidPartInfo.bottomDimPartSizes.end());
      kSubDimSizes.append(kDimIterPartInfo.bottomDimPartSizes.begin(), kDimIterPartInfo.bottomDimPartSizes.end());
      toGlobalIdx.unmerge("k", 1, kSubDimNames, kSubDimSizes);
    }

    {
      SmallVector<StringRef, 3> dSubDimNames {thisBlockDim};
      dSubDimNames.append(dDimTidPartInfo.bottomDimPartNames.begin(), dDimTidPartInfo.bottomDimPartNames.end());
      dSubDimNames.append(dDimIterPartInfo.bottomDimPartNames.begin(), dDimIterPartInfo.bottomDimPartNames.end());
      SmallVector<int64_t, 3> dSubDimSizes {dGlobal / dPerBlock};
      dSubDimSizes.append(dDimTidPartInfo.bottomDimPartSizes.begin(), dDimTidPartInfo.bottomDimPartSizes.end());
      dSubDimSizes.append(dDimIterPartInfo.bottomDimPartSizes.begin(), dDimIterPartInfo.bottomDimPartSizes.end());
      toGlobalIdx.unmerge(dName, 2, dSubDimNames, dSubDimSizes);
    }
    toGlobalIdx.ignore(otherBlockDim);
    TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();
    gpuViews.gridwiseView = b.getArrayAttr({splitIdAttr, toGlobalIdxAttr});
  }
  {
    TopDownTMBuilder blockwiseSplitId(b, {"tid", "iter"},
                                      {blockSize, dataPerThread}, loc);
    auto [tidDimNames, tidDimSizes] = getCombinedOrder(kDimTidPartInfo, dDimTidPartInfo);
    blockwiseSplitId.merge(tidDimNames, 
                          getOutDimIndices(0, tidDimNames.size()),
                          "tid", 
                          tidDimSizes);
    auto [iterDimNames, iterDimSizes] = getCombinedOrder(kDimIterPartInfo, dDimIterPartInfo);
    blockwiseSplitId.merge(iterDimNames, 
                          getOutDimIndices(0 + tidDimNames.size(), iterDimNames.size()),
                          "iter", 
                          iterDimSizes);
    TransformMapAttr splitIdAttr = blockwiseSplitId.get();
    auto toGlobalIdx = TopDownTMBuilder::below(blockwiseSplitId, splitIdAttr);
    {
      SmallVector<StringRef, 3> kSubDimNames;
      kSubDimNames.append(kDimTidPartInfo.bottomDimPartNames.begin(), kDimTidPartInfo.bottomDimPartNames.end());
      kSubDimNames.append(kDimIterPartInfo.bottomDimPartNames.begin(), kDimIterPartInfo.bottomDimPartNames.end());
      SmallVector<int64_t, 3> kSubDimSizes;
      kSubDimSizes.append(kDimTidPartInfo.bottomDimPartSizes.begin(), kDimTidPartInfo.bottomDimPartSizes.end());
      kSubDimSizes.append(kDimIterPartInfo.bottomDimPartSizes.begin(), kDimIterPartInfo.bottomDimPartSizes.end());
      toGlobalIdx.unmerge("k", 0, kSubDimNames, kSubDimSizes);
    }
    {
      SmallVector<StringRef, 3> dSubDimNames;
      dSubDimNames.append(dDimTidPartInfo.bottomDimPartNames.begin(), dDimTidPartInfo.bottomDimPartNames.end());
      dSubDimNames.append(dDimIterPartInfo.bottomDimPartNames.begin(), dDimIterPartInfo.bottomDimPartNames.end());
      SmallVector<int64_t, 3> dSubDimSizes;
      dSubDimSizes.append(dDimTidPartInfo.bottomDimPartSizes.begin(), dDimTidPartInfo.bottomDimPartSizes.end());
      dSubDimSizes.append(dDimIterPartInfo.bottomDimPartSizes.begin(), dDimIterPartInfo.bottomDimPartSizes.end());
      toGlobalIdx.unmerge(dName, 1, dSubDimNames, dSubDimSizes);
    }
    TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();
    gpuViews.blockwiseView = b.getArrayAttr({splitIdAttr, toGlobalIdxAttr});
  }
  {
    TopDownTMBuilder threadwiseSplitId(b, {"iter"}, {dataPerThread}, loc);
    auto [iterDimNames, iterDimSizes] = getCombinedOrder(kDimIterPartInfo, dDimIterPartInfo);
    threadwiseSplitId.merge(iterDimNames, 
                          getOutDimIndices(0, iterDimNames.size()),
                          "iter", 
                          iterDimSizes);
    TransformMapAttr splitIdAttr = threadwiseSplitId.get();
    auto toGlobalIdx = TopDownTMBuilder::below(threadwiseSplitId, splitIdAttr);
    {
      SmallVector<StringRef, 3> kSubDimNames;
      kSubDimNames.append(kDimIterPartInfo.bottomDimPartNames.begin(), kDimIterPartInfo.bottomDimPartNames.end());
      SmallVector<int64_t, 3> kSubDimSizes;
      kSubDimSizes.append(kDimIterPartInfo.bottomDimPartSizes.begin(), kDimIterPartInfo.bottomDimPartSizes.end());
      toGlobalIdx.unmerge("k", 0, kSubDimNames, kSubDimSizes);
    }
    {
      SmallVector<StringRef, 3> dSubDimNames;
      dSubDimNames.append(dDimIterPartInfo.bottomDimPartNames.begin(), dDimIterPartInfo.bottomDimPartNames.end());
      SmallVector<int64_t, 3> dSubDimSizes;
      dSubDimSizes.append(dDimIterPartInfo.bottomDimPartSizes.begin(), dDimIterPartInfo.bottomDimPartSizes.end());
      toGlobalIdx.unmerge(dName, 1, dSubDimNames, dSubDimSizes);
    }
    TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();
    gpuViews.threadwiseView = b.getArrayAttr({splitIdAttr, toGlobalIdxAttr});
  }
  return gpuViews;
}
