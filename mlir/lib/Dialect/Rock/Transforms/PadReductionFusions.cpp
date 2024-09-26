//===- ReuseLDS - MLIR Rock ops lowering passes -----===//
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
// This pass re-uses LDS memory by using the lifetime annotations (rock.dealloc)
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/BufferDependencyAnalysis.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPADREDUCTIONFUSIONS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-pad-reduction-fusions"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockPadReductionFusionsPass
    : public rock::impl::RockPadReductionFusionsBase<RockPadReductionFusionsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// struct RockReduceRewritePattern
//     : public OpRewritePattern<ReduceOp> {
//   using OpRewritePattern<ReduceOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(ReduceOp op,
//                                 PatternRewriter &b) const override {
//     Location loc = op.getLoc();
//     auto &bufferDeps = getAnalysis<BufferDependencyAnalysis>();
//   }
// };

ArrayAttr reverse(ArrayAttr attrs){
  SmallVector<Attribute> attrsReversed = llvm::to_vector(llvm::reverse(attrs));
  IRRewriter rewriter(attrs.getContext());
  return rewriter.getArrayAttr(attrsReversed);
}

ArrayAttr getAllViewsFromSource(OpOperand* operand){
  Value val = operand->get();
  SmallVector<Attribute> attrs;
  while(TransformOp trOp = dyn_cast<TransformOp>(val.getDefiningOp())){
    attrs.push_back(trOp.getTransformAttr());
    val = trOp.getViewSource();
  }
  IRRewriter rewriter(val.getContext());
  ArrayAttr arrAttrs = rewriter.getArrayAttr({attrs});
  return reverse(arrAttrs);
}

FailureOr<std::tuple<ArrayAttr,Operation*>> obtainViewsFromReaderToWriter(memref::AllocOp buffer, const BufferDependencyAnalysis& deps, ArrayAttr currViews){
  LLVM_DEBUG(llvm::dbgs() << "buffer = " << buffer << "\n");
  IRRewriter rewriter(buffer.getContext());
  std::optional<llvm::SmallVector<OpOperand *>> writersOperands = deps.getWriters(buffer);
  if(!writersOperands.has_value()) return failure();
  for(OpOperand* writerOperand : writersOperands.value()){
    ArrayAttr viewsFromAllocOp = getAllViewsFromSource(writerOperand);
    currViews = prependUpperViews(rewriter, currViews, viewsFromAllocOp);
    if(isa<GridwiseGemmAccelOp,GridwiseGemmOp>(writerOperand->getOwner())){
      return std::make_tuple(reverse(currViews), writerOperand->getOwner());
    }
    LLVM_DEBUG(llvm::dbgs() << "write op = " << *writerOperand->getOwner() << "\n");
    auto writeOp = dyn_cast<MemoryEffectOpInterface>(writerOperand->getOwner());
    if (!writeOp){
      LLVM_DEBUG(llvm::dbgs() << "\tit is not a memory effect interface op\n");
      continue;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    writeOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance& effect : effects){
      OpOperand *readOperand =
          effect.getEffectValue<OpOperand *>();
      LLVM_DEBUG(llvm::dbgs() << "readOperand = " << readOperand->get() << "\n");
      // Test against the write operand to guard against [MemRead, MemWrite]
      if (readOperand &&
          readOperand != writerOperand &&
          isa<MemoryEffects::Read>(effect.getEffect())) {
          if(memref::AllocOp readBuffer = dyn_cast<memref::AllocOp>(readOperand->get().getDefiningOp())){
            FailureOr<std::tuple<ArrayAttr,Operation*>> mayBeViewsAndGemmOp = obtainViewsFromReaderToWriter(readBuffer, deps, currViews);
            if(succeeded(mayBeViewsAndGemmOp)){
              return mayBeViewsAndGemmOp;
            }
          }
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "No writer goes to a gemm op.\n");
  return failure();
}

FailureOr<std::tuple<ArrayAttr,Operation*>> obtainGemmToReduceViews(ReduceOp rOp, const BufferDependencyAnalysis& deps){
  IRRewriter rewriter(rOp.getContext());
  memref::AllocOp rSrc = rOp.getIn().getDefiningOp<memref::AllocOp>();
  if(!rSrc) return failure();
  ArrayAttr views = rewriter.getArrayAttr({});
  return obtainViewsFromReaderToWriter(rSrc, deps, views);
}

struct MNPerBlock {
  int64_t MPerBlock;
  int64_t NPerBlock;
};

static MNPerBlock getMNPerBlock(Operation* gemmOp){
  MNPerBlock ret;
  if(auto xdlGemmOp = dyn_cast<GridwiseGemmAccelOp>(gemmOp)){
    ret.MPerBlock = xdlGemmOp.getParams().getMPerBlock();
    ret.NPerBlock = xdlGemmOp.getParams().getNPerBlock();
  }
  else if (auto nonxdlGemmOp = dyn_cast<GridwiseGemmOp>(gemmOp)){
    ret.MPerBlock = nonxdlGemmOp.getParams().getMPerBlock();
    ret.NPerBlock = nonxdlGemmOp.getParams().getNPerBlock();
  }
  else{
    llvm_unreachable("Only gemm ops are supported!\n");
  }
  return ret;
}

ArrayAttr appendTiledViews(ArrayAttr views, const MNPerBlock& tileSizes){
  OpBuilder builder(views.getContext());
  TransformMapAttr topMap = cast<TransformMapAttr>(views[0]);
  llvm::errs() << "bottomMap=" << topMap << "\n";
  llvm::errs() << "MPerBlock=" << tileSizes.MPerBlock << "\n";
  llvm::errs() << "NPerBlock=" << tileSizes.NPerBlock << "\n";
  ArrayRef<int64_t> upperSizes = topMap.getUpperBounds().asArrayRef();
  BottomUpTMBuilder toTiledViews(builder, {"G", "M", "N"}, upperSizes);
  toTiledViews.passThrough("G");
  unsigned mblock = upperSizes[1] / tileSizes.MPerBlock;
  toTiledViews.unmerge({"mblock", "mperblock"}, {1, 2}, "M", {mblock, (unsigned)tileSizes.MPerBlock});
  unsigned nblock = upperSizes[2] / tileSizes.NPerBlock;
  toTiledViews.unmerge({"nblock", "nperblock"}, {3, 4}, "N", {nblock, (unsigned)tileSizes.NPerBlock});
  TransformMapAttr tiledView = toTiledViews.get();
  llvm::errs() << "tiledView=" << tiledView << "\n"; 
  return prependUpperViews(builder, builder.getArrayAttr({tiledView}), views); 
}

ArrayAttr reorderReductionDims(BottomUpTMBuilder& toReductionSplit, ArrayRef<SubDimInfo> reductionSubDims, StringRef dName, int64_t dLen, int64_t dPerBlock){
  SmallVector<SubDimInfo> reductionSubDimsVec = llvm::to_vector(reductionSubDims);
  llvm::sort(reductionSubDimsVec, [](const SubDimInfo& L, const SubDimInfo& R){return L.stride > R.stride;});
  llvm::SmallDenseMap<int64_t, int64_t> reductionDims;
  {
    toReductionSplit.passThrough(ArrayRef<unsigned>{0, 1}, ArrayRef<unsigned>{0, 1});
    SmallVector<int64_t> splitSizes;
    SmallVector<SmallString<8>> splitNames;
    SmallVector<StringRef> splitNamesRefs;
    SmallVector<unsigned> splitDims;
    int64_t dimInsertionPoint = 2;
    int64_t currSize = dLen;
    for (auto [idx, sdInfo] : enumerate(reductionSubDimsVec)){
      {
        SmallString<8> dimName(Twine("d_nr" + Twine(idx)).str());
        splitNames.push_back(dimName);
      }
      splitNamesRefs.push_back(splitNames.back());
      splitDims.push_back(dimInsertionPoint++);
      splitSizes.push_back(currSize / (sdInfo.size * sdInfo.stride));
      {
        SmallString<8> dimName(Twine("d_r" + Twine(idx)).str());
        splitNames.push_back(dimName);
      }
      splitNamesRefs.push_back(splitNames.back());
      reductionDims[dimInsertionPoint] = sdInfo.size;
      splitDims.push_back(dimInsertionPoint++);
      splitSizes.push_back(sdInfo.size);
      currSize = sdInfo.stride;
    }
    toReductionSplit.unmerge(splitNamesRefs, splitDims, dName, splitSizes);
  }
  TransformMapAttr reduceSplit = toReductionSplit.get();
  llvm::errs() << "reduceSplit = " << reduceSplit << "\n";
  auto toCommonReductionDim = BottomUpTMBuilder::above(toReductionSplit, reduceSplit);
  {
    SmallVector<StringRef, 4> startNames;
    toCommonReductionDim.getStartNames(startNames);
    SmallVector<StringRef, 4> reduceDimNames;
    unsigned dimInsertionPoint = 1;
    for(unsigned dim = 0; dim < reduceSplit.getUpperBounds().size(); dim++){
      if(!reductionDims.contains(dim)){
        toCommonReductionDim.passThrough({dimInsertionPoint++},{dim});
      }
      else {
        reduceDimNames.push_back(startNames[dim]);
      }
    }
    toCommonReductionDim.merge("d_r", dimInsertionPoint++, reduceDimNames);
  }
  TransformMapAttr commonReduction = toCommonReductionDim.get();
  llvm::errs() << "commonReduction = " << commonReduction << "\n";
  auto toResplitReduction = BottomUpTMBuilder::above(toCommonReductionDim, commonReduction);
  {
    unsigned upperDimCount = commonReduction.getUpperBounds().size();
    int64_t commonReduceSize = commonReduction.getUpperBounds().asArrayRef().back();
    int64_t commonFactor = math_util::gcd(commonReduceSize, dPerBlock);
    toResplitReduction.passThrough(ArrayRef<unsigned>{0, 1, 3}, ArrayRef<unsigned>{0, 1, 2});
    toResplitReduction.unmerge({"d_rh", "d_rl"}, {2, upperDimCount}, "d_r", {commonReduceSize / commonFactor, commonFactor});
  }
  TransformMapAttr resplitReduction = toResplitReduction.get();
  llvm::errs() << "resplitReductionAttr = " << resplitReduction << "\n";
  auto toRecombined = BottomUpTMBuilder::above(toResplitReduction, resplitReduction);
  {
    toRecombined.passThrough(ArrayRef<unsigned>{0, 1}, ArrayRef<unsigned>{0, 1});
    toRecombined.merge("d", 2, {"d_rh", "d_nr", "d_rl"});
  }
  TransformMapAttr recombined = toRecombined.get();
  llvm::errs() << "recombined = " << recombined << "\n";
  OpBuilder builder(recombined.getContext());
  builder.getArrayAttr({reduceSplit, commonReduction, resplitReduction, recombined});
}

std::tuple<ArrayAttr, ArrayAttr> generateShuffledGemmInputViews(OpBuilder& builder, int64_t g, int64_t m, int64_t mPerBlock, int64_t k, int64_t n, int64_t nPerBlock, llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>> reductionSubDims){
  BottomUpTMBuilder toReductionSplitA(builder, {"G", "K", "M"}, {g, k, m});
  ArrayAttr additionalViewsA = reorderReductionDims(toReductionSplitA, reductionSubDims[1], "M", m, mPerBlock);
  BottomUpTMBuilder toReductionSplitB(builder, {"G", "K", "N"}, {g, k, n});
  ArrayAttr additionalViewsB = reorderReductionDims(toReductionSplitB, reductionSubDims[2], "N", n, nPerBlock);
  return {additionalViewsA, additionalViewsB};
}

ArrayAttr doGemmParallelShuffle(ArrayAttr views, llvm::SmallDenseMap<int64_t, SmallVector<mlir::rock::SubDimInfo>> reductionSubDims, const MNPerBlock& tileSizes){
  OpBuilder builder(views.getContext());
  TransformMapAttr topMap = cast<TransformMapAttr>(views[0]);
  ArrayRef<int64_t> upperSizes = topMap.getUpperBounds().asArrayRef();

  BottomUpTMBuilder toReductionSplit(builder, {"G", "M", "N"}, upperSizes);
  llvm::SmallDenseMap<int64_t, int64_t> reductionDimsM;
  llvm::SmallDenseMap<int64_t, int64_t> reductionDimsN;
  {
    toReductionSplit.passThrough("G");
    int64_t dimInsertionPoint = 1;
    {
      SmallVector<SubDimInfo> reductionSubDimsM = reductionSubDims[1];
      llvm::sort(reductionSubDimsM, [](const SubDimInfo& L, const SubDimInfo& R){return L.stride > R.stride;});
      SmallVector<int64_t> splitSizesM;
      SmallVector<SmallString<8>> splitNamesM;
      SmallVector<StringRef> splitNamesMRefs;
      SmallVector<unsigned> splitDimsM;
      int64_t currSize = upperSizes[1];
      for (auto [idx, sdInfo] : enumerate(reductionSubDimsM)){
        {
          SmallString<8> dimName(Twine("m_nr" + Twine(idx)).str());
          splitNamesM.push_back(dimName);
        }
        splitNamesMRefs.push_back(splitNamesM.back());
        splitDimsM.push_back(dimInsertionPoint++);
        splitSizesM.push_back(currSize / (sdInfo.size * sdInfo.stride));

        {
          SmallString<8> dimName(Twine("m_r" + Twine(idx)).str());
          splitNamesM.push_back(dimName);
        }
        splitNamesMRefs.push_back(splitNamesM.back());
        reductionDimsM[dimInsertionPoint] = sdInfo.size;
        splitDimsM.push_back(dimInsertionPoint++);
        splitSizesM.push_back(sdInfo.size);
        currSize = sdInfo.stride;
      }
      toReductionSplit.unmerge(splitNamesMRefs, splitDimsM, "M", splitSizesM);
    }

    {
      SmallVector<SubDimInfo> reductionSubDimsN = reductionSubDims[2];
      llvm::sort(reductionSubDimsN, [](const SubDimInfo& L, const SubDimInfo& R) {return L.stride > R.stride;});
      SmallVector<int64_t> splitSizesN;
      SmallVector<SmallString<8>> splitNamesN;
      SmallVector<StringRef> splitNamesNRefs;
      SmallVector<unsigned> splitDimsN;
      int64_t currSize = upperSizes[2];
      for (auto [idx, sdInfo] : enumerate(reductionSubDimsN)){
        {
          SmallString<8> dimName(Twine("n_nr" + Twine(idx)).str());
          splitNamesN.push_back(dimName);
        }
        splitNamesNRefs.push_back(splitNamesN.back());
        splitDimsN.push_back(dimInsertionPoint++);
        splitSizesN.push_back(currSize / (sdInfo.size * sdInfo.stride));

        {
          SmallString<8> dimName(Twine("n_r" + Twine(idx)).str());
          splitNamesN.push_back(dimName);
        }
        splitNamesNRefs.push_back(splitNamesN.back());
        reductionDimsN[dimInsertionPoint] = sdInfo.size;
        splitDimsN.push_back(dimInsertionPoint++);
        splitSizesN.push_back(sdInfo.size);
        currSize = sdInfo.stride;
      }
      toReductionSplit.unmerge(splitNamesNRefs, splitDimsN, "N", splitSizesN);
    }
  }
  TransformMapAttr reduceSplit = toReductionSplit.get();
  llvm::errs() << "reduceSplit = " << reduceSplit << "\n";

  auto toCommonReductionDim = BottomUpTMBuilder::above(toReductionSplit, reduceSplit);
  {
    SmallVector<StringRef, 4> startNames;
    toCommonReductionDim.getStartNames(startNames);
    toCommonReductionDim.passThrough("G");
    SmallVector<StringRef, 4> reduceDimNamesM;
    SmallVector<StringRef, 4> reduceDimNamesN;
    unsigned dimInsertionPoint = 1;
    for(unsigned dim = 1; dim < reduceSplit.getUpperBounds().size(); dim++){
      if(!reductionDimsM.contains(dim) && !reductionDimsN.contains(dim)){
        toCommonReductionDim.passThrough({dimInsertionPoint++},{dim});
      }
      else if(reductionDimsM.contains(dim)){
        reduceDimNamesM.push_back(startNames[dim]);
      }
      else if(reductionDimsN.contains(dim)){
        reduceDimNamesN.push_back(startNames[dim]);
      }
      else{
        llvm_unreachable("above cases cover everything!");
      }
    }
    toCommonReductionDim.merge("m_r", dimInsertionPoint++, reduceDimNamesM);
    toCommonReductionDim.merge("n_r", dimInsertionPoint++, reduceDimNamesN);
  }
  TransformMapAttr commonReduction = toCommonReductionDim.get();
  llvm::errs() << "commonReduction = " << commonReduction << "\n";

  auto resplitReduction = BottomUpTMBuilder::above(toCommonReductionDim, commonReduction);
  {
    unsigned upperDimCount = commonReduction.getUpperBounds().size();
    int64_t commonMReduceSize = commonReduction.getUpperBounds().asArrayRef()[upperDimCount - 2];
    int64_t commonNReduceSize = commonReduction.getUpperBounds().asArrayRef().back();
    int64_t commonMFactor = math_util::gcd(commonMReduceSize, tileSizes.MPerBlock);
    int64_t commonNFactor = math_util::gcd(commonNReduceSize, tileSizes.NPerBlock);
    resplitReduction.unmerge({"m_rh", "m_rl"}, {0, upperDimCount - 2}, "m_r", {commonMReduceSize / commonMFactor, commonMFactor});
  }



  // int64_t reductionSubDimMBlock = math_util::gcd(totalReductionSizeM, tileSizes.MPerBlock);
  return views;
}



void RockPadReductionFusionsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel")){
    return;
  }
  auto &bufferDeps = getAnalysis<BufferDependencyAnalysis>();
  WalkResult walkResult = func.walk([&](ReduceOp rOp) -> WalkResult {
    LLVM_DEBUG(llvm::dbgs() << "rOp = " << rOp << "\n");
    LLVM_DEBUG(llvm::dbgs() << "---------------------\n");
    FailureOr<std::tuple<ArrayAttr,Operation*>> res = obtainGemmToReduceViews(rOp, bufferDeps);
    if(succeeded(res)){
      auto[views, gemmOp] = res.value();
      MNPerBlock mnPerBlock = getMNPerBlock(gemmOp);
      // views = appendTiledViews(views, mnPerBlock);
      llvm::errs() << "views = " << views << "\n";
      OpBuilder builder(views.getContext());
      ArrayAttr invertedViews = invertTransforms(builder, rOp.getLoc(), views);
      llvm::errs() << "inverted_views = " << invertedViews << "\n";
      FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<mlir::rock::SubDimInfo>>> subDimensions = getLowerSubDimensions(builder, invertedViews, 2);
      assert(succeeded(subDimensions));
      for(auto [dim, subDimInfos] : subDimensions.value()){
        llvm::errs() << "dim=" << dim << ":";
        llvm::interleaveComma(subDimInfos, llvm::errs());
        llvm::errs() << "\n";
      }

      TypedValue<MemRefType> gemmInA;
      TypedValue<MemRefType> gemmInB;
      if(GridwiseGemmAccelOp gemmAccelOp = dyn_cast<GridwiseGemmAccelOp>(gemmOp)){
        gemmInA = gemmAccelOp.getA();
        gemmInB = gemmAccelOp.getB();
      }
      int64_t g = gemmInA.getType().getShape()[0];
      int64_t k = gemmInA.getType().getShape()[1];
      int64_t m = gemmInA.getType().getShape()[2];
      int64_t n = gemmInB.getType().getShape()[2];
      auto[additionalViewsA, additionalViewsB] = generateShuffledGemmInputViews(builder, g, m, mnPerBlock.MPerBlock, n, k, mnPerBlock.NPerBlock, subDimensions.value());



      // doGemmParallelShuffle(views, subDimensions.value(), mnPerBlock);


      llvm::errs() << "gemm=" << *gemmOp << "\n";

      IRRewriter rewriter(rOp.getContext());
      SetVector<int64_t> removeIndicesSet;
      removeIndicesSet.insert(1);
      removeIndicesSet.insert(3);
      FailureOr<ArrayAttr> blockSubTileViews =
      removeUpperDims(rewriter, views, removeIndicesSet);
      llvm::errs() << "blockSubTileViews = " << blockSubTileViews.value() << "\n";
    }
    else{
      llvm::errs() << "failed obtaining views from reduce to gemm.\n";
    }
    return WalkResult::advance();
  });
}
