//===- ShuffleGemmForReductions - MLIR Rock ops lowering passes -----===//
//
// Copyright 2024 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
//
// This pass will rearrange M & N dimensions of a gemm that
// is being fused with a reduction at the end -- possibly with
// reshapes in between. This pass will re-order M & N dimensions
// such that sub-dimensions of M/N being reduced will be split
// equally across blocks.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSHUFFLEGEMMFORREDUCTIONS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-shuffle-gemm-for-reductions"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockShuffleGemmForReductionsPass
    : public rock::impl::RockShuffleGemmForReductionsBase<
          RockShuffleGemmForReductionsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

ArrayAttr reverse(ArrayAttr attrs) {
  SmallVector<Attribute> attrsReversed = llvm::to_vector(llvm::reverse(attrs));
  IRRewriter rewriter(attrs.getContext());
  return rewriter.getArrayAttr(attrsReversed);
}

ArrayAttr getAllViewsFromSource(OpOperand *operand) {
  Value val = operand->get();
  IRRewriter rewriter(val.getContext());
  ArrayAttr trs;
  Value untransformed;
  std::tie(untransformed, trs, std::ignore) = untransform(rewriter, val);
  return reverse(trs);
}

FailureOr<std::tuple<ArrayAttr, Operation *>>
obtainViewsFromReaderToWriter(memref::AllocOp buffer,
                              const BufferDependencyAnalysis &deps,
                              ArrayAttr currViews) {
  LLVM_DEBUG(llvm::dbgs() << "buffer = " << buffer << "\n");
  IRRewriter rewriter(buffer.getContext());
  std::optional<llvm::SmallVector<OpOperand *>> writersOperands =
      deps.getWriters(buffer);
  if (!writersOperands.has_value())
    return failure();
  for (OpOperand *writerOperand : writersOperands.value()) {
    ArrayAttr viewsFromAllocOp = getAllViewsFromSource(writerOperand);
    currViews = prependUpperViews(rewriter, currViews, viewsFromAllocOp);
    if (isa<GridwiseGemmAccelOp, GridwiseGemmOp>(writerOperand->getOwner())) {
      return std::make_tuple(reverse(currViews), writerOperand->getOwner());
    }
    LLVM_DEBUG(llvm::dbgs()
               << "write op = " << *writerOperand->getOwner() << "\n");
    auto writeOp = dyn_cast<MemoryEffectOpInterface>(writerOperand->getOwner());
    if (!writeOp) {
      LLVM_DEBUG(llvm::dbgs() << "\tit is not a memory effect interface op\n");
      continue;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    writeOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      OpOperand *readOperand = effect.getEffectValue<OpOperand *>();
      LLVM_DEBUG(llvm::dbgs()
                 << "readOperand = " << readOperand->get() << "\n");
      // Test against the write operand to guard against [MemRead, MemWrite]
      if (readOperand && readOperand != writerOperand &&
          isa<MemoryEffects::Read>(effect.getEffect())) {
        if (memref::AllocOp readBuffer =
                dyn_cast<memref::AllocOp>(readOperand->get().getDefiningOp())) {
          FailureOr<std::tuple<ArrayAttr, Operation *>> mayBeViewsAndGemmOp =
              obtainViewsFromReaderToWriter(readBuffer, deps, currViews);
          if (succeeded(mayBeViewsAndGemmOp)) {
            return mayBeViewsAndGemmOp;
          }
        }
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "No writer goes to a gemm op.\n");
  return failure();
}

FailureOr<std::tuple<ArrayAttr, Operation *>>
obtainGemmToReduceViews(ReduceOp rOp, const BufferDependencyAnalysis &deps) {
  IRRewriter rewriter(rOp.getContext());
  memref::AllocOp rSrc = rOp.getIn().getDefiningOp<memref::AllocOp>();
  if (!rSrc)
    return failure();
  ArrayAttr views = rewriter.getArrayAttr({});
  return obtainViewsFromReaderToWriter(rSrc, deps, views);
}

struct MNPerBlock {
  int64_t MPerBlock;
  int64_t NPerBlock;
};

static FailureOr<MNPerBlock> getMNPerBlock(Operation *gemmOp) {
  MNPerBlock ret;
  if (auto xdlGemmOp = dyn_cast<GridwiseGemmAccelOp>(gemmOp)) {
    ret.MPerBlock = xdlGemmOp.getParams().getMPerBlock();
    ret.NPerBlock = xdlGemmOp.getParams().getNPerBlock();
  } else if (auto nonxdlGemmOp = dyn_cast<GridwiseGemmOp>(gemmOp)) {
    ret.MPerBlock = nonxdlGemmOp.getParams().getMPerBlock();
    ret.NPerBlock = nonxdlGemmOp.getParams().getNPerBlock();
  } else {
    return failure();
  }
  return ret;
}

ArrayAttr reorderReductionDims(BottomUpTMBuilder &toReductionSplit,
                               ArrayRef<SubDimInfo> reductionSubDims,
                               StringRef dName, int64_t dLen,
                               int64_t dPerBlock) {
  SmallVector<SubDimInfo> reductionSubDimsVec =
      llvm::to_vector(reductionSubDims);
  llvm::sort(reductionSubDimsVec, [](const SubDimInfo &L, const SubDimInfo &R) {
    return L.stride > R.stride;
  });
  llvm::SmallDenseMap<int64_t, int64_t> reductionDims;
  TransformMapAttr reduceSplit;
  {
    toReductionSplit.passThrough(ArrayRef<unsigned>{0, 1},
                                 ArrayRef<unsigned>{0, 1});
    SmallVector<int64_t> splitSizes;
    SmallVector<SmallString<8>> splitNames;
    SmallVector<unsigned> splitDims;
    int64_t dimInsertionPoint = 2;
    int64_t currSize = dLen;
    LLVM_DEBUG(llvm::dbgs() << "dLen = " << dLen << "\n");
    for (auto [idx, sdInfo] : enumerate(reductionSubDimsVec)) {
      {
        SmallString<8> dimName(Twine("d_nr" + Twine(idx)).str());
        splitNames.push_back(dimName);
      }
      splitDims.push_back(dimInsertionPoint++);
      LLVM_DEBUG(llvm::dbgs()
                 << "\tsplitSize = " << currSize / (sdInfo.size * sdInfo.stride)
                 << "\n");
      splitSizes.push_back(currSize / (sdInfo.size * sdInfo.stride));
      {
        SmallString<8> dimName(Twine("d_r" + Twine(idx)).str());
        splitNames.push_back(dimName);
      }
      reductionDims[dimInsertionPoint] = sdInfo.size;
      splitDims.push_back(dimInsertionPoint++);
      LLVM_DEBUG(llvm::dbgs() << "\tsplitSize = " << sdInfo.size << "\n");
      splitSizes.push_back(sdInfo.size);
      currSize = sdInfo.stride;
    }
    if (currSize > 1) {
      {
        SmallString<8> dimName(Twine("d_nr_end").str());
        splitNames.push_back(dimName);
      }
      splitDims.push_back(dimInsertionPoint++);
      LLVM_DEBUG(llvm::dbgs() << "\tsplitSize = " << currSize << "\n");
      splitSizes.push_back(currSize);
    }
    SmallVector<StringRef> splitNamesRefs = getStringRefsFor(splitNames);
    toReductionSplit.unmerge(splitNamesRefs, splitDims, dName, splitSizes);
    reduceSplit = toReductionSplit.get();
  }
  LLVM_DEBUG(llvm::dbgs() << "reduceSplit = " << reduceSplit << "\n");
  auto toCommonReductionDim =
      BottomUpTMBuilder::above(toReductionSplit, reduceSplit);
  TransformMapAttr commonReduction;
  {
    toCommonReductionDim.passThrough(ArrayRef<unsigned>{0, 1},
                                     ArrayRef<unsigned>{0, 1});
    SmallVector<StringRef, 4> startNames;
    toCommonReductionDim.getStartNames(startNames);
    SmallVector<StringRef, 4> reduceDimNames;
    SmallVector<StringRef, 4> nonReduceDimNames;
    unsigned dimInsertionPoint = 2;
    for (unsigned dim = 2; dim < reduceSplit.getUpperBounds().size(); dim++) {
      if (!reductionDims.contains(dim)) {
        nonReduceDimNames.push_back(startNames[dim]);
      } else {
        reduceDimNames.push_back(startNames[dim]);
      }
    }
    toCommonReductionDim.merge("d_nr", dimInsertionPoint++, nonReduceDimNames);
    toCommonReductionDim.merge("d_r", dimInsertionPoint++, reduceDimNames);
    commonReduction = toCommonReductionDim.get();
  }
  LLVM_DEBUG(llvm::dbgs() << "commonReduction = " << commonReduction << "\n");
  auto toResplitReduction =
      BottomUpTMBuilder::above(toCommonReductionDim, commonReduction);
  TransformMapAttr resplitReduction;
  {
    unsigned upperDimCount = commonReduction.getUpperBounds().size();
    int64_t commonReduceSize =
        commonReduction.getUpperBounds().asArrayRef().back();
    int64_t commonFactor = math_util::gcd(commonReduceSize, dPerBlock);
    toResplitReduction.passThrough(ArrayRef<unsigned>{0, 1, 3},
                                   ArrayRef<unsigned>{0, 1, 2});
    toResplitReduction.unmerge({"d_rh", "d_rl"}, {2, upperDimCount}, "d_r",
                               {commonReduceSize / commonFactor, commonFactor});
    resplitReduction = toResplitReduction.get();
  }
  LLVM_DEBUG(llvm::dbgs() << "resplitReductionAttr = " << resplitReduction
                          << "\n");
  auto toRecombined =
      BottomUpTMBuilder::above(toResplitReduction, resplitReduction);
  TransformMapAttr recombined;
  {
    toRecombined.passThrough(ArrayRef<unsigned>{0, 1},
                             ArrayRef<unsigned>{0, 1});
    toRecombined.merge("d", 2, {"d_rh", "d_nr", "d_rl"});
    recombined = toRecombined.get();
  }
  LLVM_DEBUG(llvm::dbgs() << "recombined = " << recombined << "\n");
  OpBuilder builder(recombined.getContext());
  return builder.getArrayAttr(
      {reduceSplit, commonReduction, resplitReduction, recombined});
}

// This function will shuffle the M & N dimensions so that the
// reductions are uniformly split across block tiles. Note that
// we dont consider G as we dont block tile across G dimension.
std::tuple<ArrayAttr, ArrayAttr> generateShuffledGemmInputViews(
    OpBuilder &builder, int64_t g, int64_t m, int64_t mPerBlock, int64_t k,
    int64_t n, int64_t nPerBlock,
    const llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>>
        &reductionSubDims) {
  BottomUpTMBuilder toReductionSplitA(builder, {"G", "K", "M"}, {g, k, m});
  ArrayAttr additionalViewsA = builder.getArrayAttr({});
  if (reductionSubDims.contains(1) && !reductionSubDims.at(1).empty()) {
    additionalViewsA = reorderReductionDims(
        toReductionSplitA, reductionSubDims.at(1), "M", m, mPerBlock);
  }
  BottomUpTMBuilder toReductionSplitB(builder, {"G", "K", "N"}, {g, k, n});
  ArrayAttr additionalViewsB = builder.getArrayAttr({});
  if (reductionSubDims.contains(2) && !reductionSubDims.at(2).empty()) {
    additionalViewsB = reorderReductionDims(
        toReductionSplitB, reductionSubDims.at(2), "N", n, nPerBlock);
  }
  return {additionalViewsA, additionalViewsB};
}

ArrayAttr generateShuffledGemmOutputViews(
    OpBuilder &builder, int64_t g, int64_t m, int64_t mPerBlock, int64_t n,
    int64_t nPerBlock,
    const llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>>
        &reductionSubDims) {
  // Split the reduction and non-reduction splits
  int64_t totalReductionSizeM = 1;
  if (reductionSubDims.contains(1)) {
    for (const SubDimInfo &sdInfo : reductionSubDims.at(1)) {
      totalReductionSizeM *= sdInfo.size;
    }
  }
  int64_t totalReductionSizeN = 1;
  if (reductionSubDims.contains(2)) {
    for (const SubDimInfo &sdInfo : reductionSubDims.at(2)) {
      totalReductionSizeN *= sdInfo.size;
    }
  }
  int64_t commonMPerBlockReductionFactor =
      math_util::gcd(totalReductionSizeM, mPerBlock);
  int64_t commonNPerBlockReductionFactor =
      math_util::gcd(totalReductionSizeN, nPerBlock);

  // Split the reduction and non-reduction splits
  BottomUpTMBuilder toReductionSplit(builder, {"G", "M", "N"}, {g, m, n});
  TransformMapAttr reductionSplit;
  {
    toReductionSplit.passThrough("G");
    toReductionSplit.unmerge(
        {"m_rh", "m_nr", "m_rl"}, {1, 2, 3}, "M",
        {totalReductionSizeM / commonMPerBlockReductionFactor,
         m / totalReductionSizeM, commonMPerBlockReductionFactor});
    toReductionSplit.unmerge(
        {"n_rh", "n_nr", "n_rl"}, {4, 5, 6}, "N",
        {totalReductionSizeN / commonNPerBlockReductionFactor,
         n / totalReductionSizeN, commonNPerBlockReductionFactor});
    reductionSplit = toReductionSplit.get();
  }
  LLVM_DEBUG(llvm::dbgs() << "reductionSplit = " << reductionSplit << "\n");

  // combine reduction dimension
  auto toCombinedReductionDim =
      BottomUpTMBuilder::above(toReductionSplit, reductionSplit);
  TransformMapAttr combinedReduction;
  {
    toCombinedReductionDim.passThrough("G");
    toCombinedReductionDim.passThrough({1}, {2});
    toCombinedReductionDim.merge("m_r", 2, {"m_rh", "m_rl"});
    toCombinedReductionDim.passThrough({3}, {5});
    toCombinedReductionDim.merge("n_r", 4, {"n_rh", "n_rl"});
    combinedReduction = toCombinedReductionDim.get();
  }
  LLVM_DEBUG(llvm::dbgs() << "combinedReduction = " << combinedReduction
                          << "\n");

  // Split to original sub dimensions
  auto toSplitOriginalSubDims =
      BottomUpTMBuilder::above(toCombinedReductionDim, combinedReduction);
  int64_t nSubDimStartPoint = -1;
  TransformMapAttr splitOriginalSubDims;
  {
    toSplitOriginalSubDims.passThrough("G");
    SmallVector<SubDimInfo> mReductionSubDimInfo;
    if (reductionSubDims.contains(1)) {
      mReductionSubDimInfo = reductionSubDims.at(1);
    }
    SmallVector<SubDimInfo> nReductionSubDimInfo;
    if (reductionSubDims.contains(2)) {
      nReductionSubDimInfo = reductionSubDims.at(2);
    }

    unsigned dimInsertionPoint = 1;
    {
      SmallVector<unsigned> mReductionSubDims;
      SmallVector<int64_t> mReductionSubDimSizes;
      SmallVector<SmallString<8>> mReductionSubDimNames;

      SmallVector<unsigned> mNonReductionSubDims;
      SmallVector<int64_t> mNonReductionSubDimSizes;
      SmallVector<SmallString<8>> mNonReductionSubDimNames;
      int64_t currSize = m;
      for (const auto &[idx, sdInfo] : enumerate(mReductionSubDimInfo)) {
        mNonReductionSubDimSizes.push_back(currSize /
                                           (sdInfo.size * sdInfo.stride));
        {
          SmallString<8> dimName(Twine("m_nr" + Twine(idx)).str());
          mNonReductionSubDimNames.push_back(dimName);
        }
        mNonReductionSubDims.push_back(dimInsertionPoint++);

        mReductionSubDimSizes.push_back(sdInfo.size);
        {
          SmallString<8> dimName(Twine("m_r" + Twine(idx)).str());
          mReductionSubDimNames.push_back(dimName);
        }
        mReductionSubDims.push_back(dimInsertionPoint++);

        currSize = sdInfo.stride;
      }
      if (currSize > 1 || mNonReductionSubDimSizes.empty()) {
        mNonReductionSubDimSizes.push_back(currSize);
        {
          SmallString<8> dimName("m_nr_last");
          mNonReductionSubDimNames.push_back(dimName);
        }
        mNonReductionSubDims.push_back(dimInsertionPoint++);
      }

      SmallVector<StringRef> mNonReductionSubDimNameRefs =
          getStringRefsFor(mNonReductionSubDimNames);
      toSplitOriginalSubDims.unmerge(mNonReductionSubDimNameRefs,
                                     mNonReductionSubDims, "m_nr",
                                     mNonReductionSubDimSizes);
      if (!mReductionSubDimSizes.empty()) {
        SmallVector<StringRef> mReductionSubDimNameRefs =
            getStringRefsFor(mReductionSubDimNames);
        toSplitOriginalSubDims.unmerge(mReductionSubDimNameRefs,
                                       mReductionSubDims, "m_r",
                                       mReductionSubDimSizes);
      } else {
        toSplitOriginalSubDims.passThrough({"m_r"}, {dimInsertionPoint++},
                                           {"m_r"});
      }
    }
    nSubDimStartPoint = dimInsertionPoint;

    {
      SmallVector<unsigned> nReductionSubDims;
      SmallVector<int64_t> nReductionSubDimSizes;
      SmallVector<SmallString<8>> nReductionSubDimNames;

      SmallVector<unsigned> nNonReductionSubDims;
      SmallVector<int64_t> nNonReductionSubDimSizes;
      SmallVector<SmallString<8>> nNonReductionSubDimNames;
      int64_t currSize = n;
      for (const auto &[idx, sdInfo] : enumerate(nReductionSubDimInfo)) {
        nNonReductionSubDimSizes.push_back(currSize /
                                           (sdInfo.size * sdInfo.stride));
        {
          SmallString<8> dimName(Twine("n_nr" + Twine(idx)).str());
          nNonReductionSubDimNames.push_back(dimName);
        }
        nNonReductionSubDims.push_back(dimInsertionPoint++);

        nReductionSubDimSizes.push_back(sdInfo.size);
        {
          SmallString<8> dimName(Twine("n_r" + Twine(idx)).str());
          nReductionSubDimNames.push_back(dimName);
        }
        nReductionSubDims.push_back(dimInsertionPoint++);

        currSize = sdInfo.stride;
      }
      if (currSize > 1 || nNonReductionSubDimSizes.empty()) {
        nNonReductionSubDimSizes.push_back(currSize);
        {
          SmallString<8> dimName("n_nr_last");
          nNonReductionSubDimNames.push_back(dimName);
        }
        nNonReductionSubDims.push_back(dimInsertionPoint++);
      }
      SmallVector<StringRef> nNonReductionSubDimNameRefs =
          getStringRefsFor(nNonReductionSubDimNames);
      toSplitOriginalSubDims.unmerge(nNonReductionSubDimNameRefs,
                                     nNonReductionSubDims, "n_nr",
                                     nNonReductionSubDimSizes);
      if (!nReductionSubDimSizes.empty()) {

        SmallVector<StringRef> nReductionSubDimNameRefs =
            getStringRefsFor(nReductionSubDimNames);
        toSplitOriginalSubDims.unmerge(nReductionSubDimNameRefs,
                                       nReductionSubDims, "n_r",
                                       nReductionSubDimSizes);
      } else {
        toSplitOriginalSubDims.passThrough({"n_r"}, {dimInsertionPoint++},
                                           {"n_r"});
      }
    }
    splitOriginalSubDims = toSplitOriginalSubDims.get();
  }
  LLVM_DEBUG(llvm::dbgs() << "splitOriginalSubDims = " << splitOriginalSubDims
                          << "\n");

  // Recombine into original M & N
  auto toRecombineMN =
      BottomUpTMBuilder::above(toSplitOriginalSubDims, splitOriginalSubDims);
  TransformMapAttr recombineMN;
  {
    toRecombineMN.passThrough("G");
    SmallVector<StringRef, 4> startNames;
    toRecombineMN.getStartNames(startNames);

    // M
    {
      SmallVector<StringRef, 4> mSubDimNames;
      for (int dim = 1; dim < nSubDimStartPoint; dim++) {
        mSubDimNames.push_back(startNames[dim]);
      }
      toRecombineMN.merge("M", 1, mSubDimNames);
    }

    // N
    {
      SmallVector<StringRef, 4> nSubDimNames;
      for (unsigned dim = nSubDimStartPoint; dim < startNames.size(); dim++) {
        nSubDimNames.push_back(startNames[dim]);
      }
      toRecombineMN.merge("N", 2, nSubDimNames);
    }
    recombineMN = toRecombineMN.get();
  }
  LLVM_DEBUG(llvm::dbgs() << "recombineMN = " << recombineMN << "\n");

  return builder.getArrayAttr(
      {reductionSplit, combinedReduction, splitOriginalSubDims, recombineMN});
}

// This function will attempt to shuffle M & N dimensions of the gemm so that
// reductions sub-dimensions within it could be split to blocks equally.
// However, for that to work the transform stack needs to be invertible and
// sub-dimension should be discoverable using "getLowerSubDimensions". If one of
// those fail, we bail and not attempt to use blockwise_reductions in such
// fusions.
static LogicalResult
rearrangeGemmParallelDimsForReduction(ReduceOp rOp,
                                      const BufferDependencyAnalysis &deps) {
  FailureOr<std::tuple<ArrayAttr, Operation *>> maybeViewsAndGemmOp =
      obtainGemmToReduceViews(rOp, deps);
  if (succeeded(maybeViewsAndGemmOp)) {
    auto [views, gemmOp] = maybeViewsAndGemmOp.value();
    LLVM_DEBUG(llvm::dbgs() << "gemmToReduceViews=" << views << "\n");
    FailureOr<MNPerBlock> mnPerBlock = getMNPerBlock(gemmOp);
    if (failed(mnPerBlock)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "m/n per block extraction failed from gemm op.\n");
      return failure();
    }
    IRRewriter rewriter(rOp.getContext());
    ArrayAttr invertedViews = invertTransforms(rewriter, rOp.getLoc(), views);
    LLVM_DEBUG(llvm::dbgs()
               << "inv(gemmToReduceViews)=" << invertedViews << "\n");
    if (!invertedViews || invertedViews.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "gemm to reduce view inversion failed.\n");
      return failure();
    }
    FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<mlir::rock::SubDimInfo>>>
        reductionSubDimsinGemmSpace = getLowerSubDimensions(
            rewriter, invertedViews, rOp.getAxis().getZExtValue());
    if (failed(reductionSubDimsinGemmSpace) ||
        reductionSubDimsinGemmSpace.value().empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "reduce to gemm lower subdimension tracing failed.\n");
      return failure();
    }
    for (auto [dim, subDimInfos] : reductionSubDimsinGemmSpace.value()) {
      LLVM_DEBUG(llvm::dbgs() << "dim=" << dim << ":");
      LLVM_DEBUG(llvm::interleaveComma(subDimInfos, llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }

    TypedValue<MemRefType> gemmInA;
    TypedValue<MemRefType> gemmInB;
    TypedValue<MemRefType> gemmOut;
    if (GridwiseGemmAccelOp gemmAccelOp =
            dyn_cast<GridwiseGemmAccelOp>(gemmOp)) {
      gemmInA = gemmAccelOp.getA();
      gemmInB = gemmAccelOp.getB();
      gemmOut = gemmAccelOp.getC();
    } else if (GridwiseGemmOp gemmNonAccelOp =
                   dyn_cast<GridwiseGemmOp>(gemmOp)) {
      gemmInA = gemmNonAccelOp.getA();
      gemmInB = gemmNonAccelOp.getB();
      gemmOut = gemmNonAccelOp.getC();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "unsupported op:" << *gemmOp << "\n");
      return failure();
    }
    int64_t g = gemmInA.getType().getShape()[0];
    int64_t k = gemmInA.getType().getShape()[1];
    int64_t m = gemmInA.getType().getShape()[2];
    int64_t n = gemmInB.getType().getShape()[2];
    auto [additionalViewsA, additionalViewsB] = generateShuffledGemmInputViews(
        rewriter, g, m, mnPerBlock.value().MPerBlock, k, n,
        mnPerBlock.value().NPerBlock, reductionSubDimsinGemmSpace.value());
    Value trGemmInA = gemmInA;
    rewriter.setInsertionPointAfterValue(gemmInA);
    for (Attribute trMap : additionalViewsA) {
      trGemmInA = rewriter.create<TransformOp>(rOp.getLoc(), trGemmInA,
                                               cast<TransformMapAttr>(trMap));
    }
    Value trGemmInB = gemmInB;
    rewriter.setInsertionPointAfterValue(gemmInB);
    for (Attribute trMap : additionalViewsB) {
      trGemmInB = rewriter.create<TransformOp>(rOp.getLoc(), trGemmInB,
                                               cast<TransformMapAttr>(trMap));
    }
    ArrayAttr additionalOutputViews = generateShuffledGemmOutputViews(
        rewriter, g, m, mnPerBlock.value().MPerBlock, n,
        mnPerBlock.value().NPerBlock, reductionSubDimsinGemmSpace.value());
    rewriter.setInsertionPointAfterValue(gemmOut);
    Value trGemmOut = gemmOut;
    ArrayAttr invertedOutViews =
        invertTransforms(rewriter, rOp.getLoc(), additionalOutputViews);
    for (Attribute trMap : invertedOutViews) {
      trGemmOut = rewriter.create<TransformOp>(rOp.getLoc(), trGemmOut,
                                               cast<TransformMapAttr>(trMap));
    }
    if (GridwiseGemmAccelOp gemmAccelOp =
            dyn_cast<GridwiseGemmAccelOp>(gemmOp)) {
      gemmAccelOp.getAMutable().assign(trGemmInA);
      gemmAccelOp.getBMutable().assign(trGemmInB);
      gemmAccelOp.getCMutable().assign(trGemmOut);
    } else if (GridwiseGemmOp gemmNonAccelOp =
                   dyn_cast<GridwiseGemmOp>(gemmOp)) {
      gemmNonAccelOp.getAMutable().assign(trGemmInA);
      gemmNonAccelOp.getBMutable().assign(trGemmInB);
      gemmNonAccelOp.getCMutable().assign(trGemmOut);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "unsupported op:" << *gemmOp << "\n");
      return failure();
    }
  } else {
    LLVM_DEBUG(llvm::dbgs() << "failed to obtain gemm to reduce views.\n");
    return failure();
  }
  return success();
}

void RockShuffleGemmForReductionsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel")) {
    return;
  }
  ReduceOp largestReductionOp;
  int64_t currReductionDimSize = 0;
  func.walk([&](ReduceOp rOp) -> WalkResult {
    TypedValue<ShapedType> rIn = rOp.getIn();
    int64_t reduceDimSize =
        rIn.getType().getShape()[rOp.getAxis().getZExtValue()];
    if (reduceDimSize > currReductionDimSize) {
      largestReductionOp = rOp;
    }
    return WalkResult::advance();
  });
  if (largestReductionOp) {
    auto &bufferDeps = getAnalysis<BufferDependencyAnalysis>();
    LogicalResult res =
        rearrangeGemmParallelDimsForReduction(largestReductionOp, bufferDeps);
    if (failed(res)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "unable to shuffle the gemm dims for blockwise reductions.\n");
    }
  }
}
