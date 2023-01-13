//===- RegularizeKernel.cpp - rewrites to allow Rock kernel fusion  ------===//
//
// Copyright 2022 Advanced Micro Devices.
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEKERNELPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fold-transpose"

using namespace mlir;
using namespace mlir::rock;

namespace {

// This rewrite will rewrite the linalg IO that has view like-ops surrounding
// them to be consumed by the linalg operation itself adjusting the indexing
// maps to faithfully represent them.
struct CollapseRewritePattern
    : public OpRewritePattern<memref::CollapseShapeOp> {
  using OpRewritePattern<memref::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CollapseShapeOp collapse,
                                PatternRewriter &rw) const override {
    auto inpType = collapse.getOperand().getType().cast<ShapedType>();
    auto outType = collapse.getResultType().cast<ShapedType>();
    auto transform = rock::transformCollapseShape(rw, inpType.getShape(),
                                                  outType.getShape());
    rw.replaceOpWithNewOp<rock::TransformOp>(collapse, collapse.getOperand(),
                                             transform);
    return success();
  }
};

struct ExpandRewritePattern : public OpRewritePattern<memref::ExpandShapeOp> {
  using OpRewritePattern<memref::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp expand,
                                PatternRewriter &rw) const override {
    auto inpType = expand.getOperand().getType().cast<ShapedType>();
    auto outType = expand.getResultType().cast<ShapedType>();
    auto transform =
        rock::transformExpandShape(rw, inpType.getShape(), outType.getShape());
    rw.replaceOpWithNewOp<rock::TransformOp>(expand, expand.getOperand(),
                                             transform);
    return success();
  }
};

////////////////////////////////////////////////////////////////////////
////  Shuffle Transforms To Writers
////////////////////////////////////////////////////////////////////////
struct ShuffleTransformsUpRewritePattern
    : public OpRewritePattern<rock::GridwiseGemmOp> {
  using OpRewritePattern<rock::GridwiseGemmOp>::OpRewritePattern;

  static LogicalResult shuffleTransformsUp(PatternRewriter &b, Operation *start,
                                           Value result);
  static LogicalResult applyTransforms(PatternRewriter &b, Value alloc,
                                       Operation *forOp, Operation *backOp);

  LogicalResult matchAndRewrite(rock::GridwiseGemmOp ggemm,
                                PatternRewriter &rewriter) const override {
    // Input:
    // %x = alloc
    // %y = tx0 (%x)
    // %y = gemm (...)
    // %z = tx1 (%x)
    // %a = lag0 (%z)   -- input
    // Result:
    // %x' = alloc
    // %z' = tx1' (%x')
    // %y = tx0 (%z')
    // %y = gemm (...)
    // %a = lag0 (%x')

    return shuffleTransformsUp(rewriter, ggemm, ggemm.getC());
  }
};

LogicalResult ShuffleTransformsUpRewritePattern::applyTransforms(
    PatternRewriter &b, Value alloc, Operation *forwOp, Operation *backOp) {
  LogicalResult lres = failure();
  Value result;
  SmallVector<TransformMapAttr> transforms;
  while (auto top = dyn_cast<rock::TransformOp>(forwOp)) {
    result = top.getResult();
    if (!result.hasOneUse())
      return failure(); // currently restricted to 1 reader
    auto tmap = rock::invertTransformMap(b, top.getTransform());
    if (!tmap)
      return failure(); // not invertible
    transforms.push_back(tmap);
    forwOp = (*result.getUses().begin()).getOwner();
  }
  if (transforms.size()) {
    // check forwOp is reader
    // apply transformation
    PatternRewriter::InsertionGuard guard(b);
    b.setInsertionPoint(alloc.getDefiningOp());
    Value val = b.create<memref::AllocOp>(forwOp->getLoc(),
                                          result.getType().cast<MemRefType>())
                    .getResult();
    forwOp->replaceUsesOfWith(result, val);
    for (auto tx : llvm::reverse(transforms)) {
      auto top = b.create<rock::TransformOp>(forwOp->getLoc(), val, tx);
      val = top.getResult();
    }
    backOp->replaceUsesOfWith(alloc, val);
    lres = success();
  }
  // recurse to the next level
  if (auto laop = dyn_cast<linalg::GenericOp>(forwOp)) {
    if (laop.getNumOutputs() == 1) {
      if (succeeded(shuffleTransformsUp(b, forwOp, laop.getOutputs()[0])))
        lres = success();
    }
  }
  return lres;
}

LogicalResult ShuffleTransformsUpRewritePattern::shuffleTransformsUp(
    PatternRewriter &b, Operation *start, Value result) {
  Operation *prevOp = start;
  // trace back to alloc
  while (auto op = result.getDefiningOp()) {
    if (isa<memref::AllocOp>(op)) {
      // apply
      Operation *readPath = nullptr;
      for (auto &use : result.getUses()) {
        Operation *useOp = use.getOwner();
        if (useOp != prevOp) {
          if (readPath != nullptr)
            return failure(); // multi readers
          readPath = useOp;
        }
      }
      if (readPath == nullptr)
        return failure();
      return applyTransforms(b, result, readPath, prevOp);
    } else if (auto top = dyn_cast<rock::TransformOp>(op)) {
      prevOp = op;
      result = top.getOperand();
    } else {
      // unknown path
      assert(0);
    }
  }
  return failure();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct RockRegularizeKernelPass
    : public rock::impl::RockRegularizeKernelPassBase<
          RockRegularizeKernelPass> {
  void runOnOperation() override;
};
} // end namespace

void RockRegularizeKernelPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    return;
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<CollapseRewritePattern, ExpandRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<ShuffleTransformsUpRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }

#if 0
  {
    RewritePatternSet patterns(ctx);
    patterns.add<ShuffleTransformUpRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
#endif
}
