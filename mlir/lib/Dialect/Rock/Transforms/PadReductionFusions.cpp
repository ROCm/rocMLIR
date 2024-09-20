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

ArrayAttr getAllViewsFromSource(OpOperand* operand){
  Value val = operand->get();
  SmallVector<Attribute> attrs;
  while(TransformOp trOp = dyn_cast<TransformOp>(val.getDefiningOp())){
    attrs.push_back(trOp.getTransformAttr());
    val = trOp.getViewSource();
  }
  SmallVector<Attribute> attrsReversed = llvm::to_vector(llvm::reverse(attrs));
  IRRewriter rewriter(val.getContext());
  return rewriter.getArrayAttr(attrsReversed);
}

FailureOr<ArrayAttr> obtainViewsFromReaderToWriter(memref::AllocOp buffer, const BufferDependencyAnalysis& deps, ArrayAttr currViews){
  LLVM_DEBUG(llvm::dbgs() << "buffer = " << buffer << "\n");
  IRRewriter rewriter(buffer.getContext());
  std::optional<llvm::SmallVector<OpOperand *>> writersOperands = deps.getWriters(buffer);
  if(!writersOperands.has_value()) return failure();
  for(OpOperand* writerOperand : writersOperands.value()){
    ArrayAttr viewsFromAllocOp = getAllViewsFromSource(writerOperand);
    currViews = prependUpperViews(rewriter, currViews, viewsFromAllocOp);
    if(isa<GridwiseGemmAccelOp,GridwiseGemmOp>(writerOperand->getOwner())){
      return currViews;
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
            FailureOr<ArrayAttr> mayBeViews = obtainViewsFromReaderToWriter(readBuffer, deps, currViews);
            if(succeeded(mayBeViews)){
              return mayBeViews;
            }
          }
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "No writer goes to a gemm op.\n");
  return failure();
}

FailureOr<ArrayAttr> obtainGemmToReduceViews(ReduceOp rOp, const BufferDependencyAnalysis& deps){
  IRRewriter rewriter(rOp.getContext());
  memref::AllocOp rSrc = rOp.getIn().getDefiningOp<memref::AllocOp>();
  if(!rSrc) return failure();
  ArrayAttr views = rewriter.getArrayAttr({});
  return obtainViewsFromReaderToWriter(rSrc, deps, views);
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
    FailureOr<ArrayAttr> res = obtainGemmToReduceViews(rOp, bufferDeps);
    if(succeeded(res)){
      llvm::errs() << "views = ";
      llvm::errs() << res.value();
      llvm::errs() << "\n";
    }
    else{
      llvm::errs() << "failed obtaining views from reduce to gemm.\n";
    }
    return WalkResult::advance();
  });
}
