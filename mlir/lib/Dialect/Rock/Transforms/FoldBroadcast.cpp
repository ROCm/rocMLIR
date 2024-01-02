//===- Regularize.cpp - rewrites to allow Rock kernel fusion  ------===//
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
#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFOLDBROADCASTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-canonicalize-gemm"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct FoldBroadcast : public OpRewritePattern<rock::GemmOp> {
  using OpRewritePattern<rock::GemmOp>::OpRewritePattern;

  // Determine if the first dimension of a view is a broadcast
  bool isBatchDimFoldable(PatternRewriter &rw, Value aView) const {
    auto [buffer, views, needs64BitIdxA] = untransform(rw, aView);

    // There are no views, hence no broadcast is possible
    if (views.empty())
      return false;

    auto trMap = views[0].cast<TransformMapAttr>();

    // There is no batch, hence nothing that can be a broadcast
    if (trMap.getUpperBounds().size() != 3)
      return false;

    auto ops = trMap.getOps();

    // Look for a `Broadcast{1} at [0]` transformation
    for (auto op : ops)
      if (op.getType() == rock::TransformType::Broadcast &&
          op.getUpperDims().back() == 0 && op.getParams().back() == 1)
        return true;

    return false;
  }

  // Merge the batch dimension into either M or N, i.e., transform (d0, d1, d2)
  // into (d0*d1, d2) or (d1, d0*d2)
  Value mergeBatch(PatternRewriter &rw, Location loc,
                   TypedValue<ShapedType> buffer, bool isTransposed) const {
    auto shapeA = buffer.getType().getShape();
    ArrayAttr mergeBatchAttr;
    if (isTransposed) {
      rock::TopDownTMBuilder mergeBatchBuilder(
          rw, {"d0", "gd1"}, {shapeA[1], shapeA[0] * shapeA[2]}, loc);
      mergeBatchBuilder.merge({"g", "d1"}, {0, 2}, "gd1",
                              {shapeA[0], shapeA[2]});
      mergeBatchBuilder.passThrough({"d0"}, {1}, {"d0"});
      mergeBatchAttr = rw.getArrayAttr({mergeBatchBuilder.get()});
    } else {
      rock::TopDownTMBuilder mergeBatchBuilder(
          rw, {"gd0", "d1"}, {shapeA[0] * shapeA[1], shapeA[2]}, loc);
      mergeBatchBuilder.merge({"g", "d0"}, {0, 1}, "gd0",
                              {shapeA[0], shapeA[1]});
      mergeBatchBuilder.passThrough({"d1"}, {2}, {"d1"});
      mergeBatchAttr = rw.getArrayAttr({mergeBatchBuilder.get()});
    }
    return rock::transform(rw, buffer, mergeBatchAttr);
  }

  // Select the 0th slice from a broadcast, de facto removing the broadcast
  // dimension
  Value unbroadcast(PatternRewriter &rw, Location loc,
                    TypedValue<ShapedType> buffer) const {
    auto shape = buffer.getType().getShape();
    rock::TopDownTMBuilder unbroadcastBuilder(rw, {"d0", "d1"},
                                              {shape[1], shape[2]}, loc);
    unbroadcastBuilder.constDim({"g"}, 0, 0, shape[0]);
    unbroadcastBuilder.passThrough({"d0"}, {1}, {"d0"});
    unbroadcastBuilder.passThrough({"d1"}, {2}, {"d1"});
    return rock::transform(rw, buffer,
                           rw.getArrayAttr({unbroadcastBuilder.get()}));
  }

  LogicalResult matchAndRewrite(rock::GemmOp op,
                                PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    bool isABatchBroadcast = isBatchDimFoldable(rw, op.getA());
    bool isBBatchBroadcast = isBatchDimFoldable(rw, op.getB());

    if (!isABatchBroadcast && !isBBatchBroadcast)
      return failure();

    Value newA, newB, newC;
    if (isBBatchBroadcast && isABatchBroadcast) {
      // If both B and C are canonicalizable, simply
      // remove the broadcast from A,B and C
      newA = unbroadcast(rw, loc, op.getA());
      newB = unbroadcast(rw, loc, op.getB());
      newC = unbroadcast(rw, loc, op.getC());
    } else if (isBBatchBroadcast) {
      newA = mergeBatch(rw, loc, op.getA(), op.getATransposed());
      newB = unbroadcast(rw, loc, op.getB());
      newC = mergeBatch(rw, loc, op.getC(), op.getCTransposed());
    } else { // isABatchBroadcast
      newA = unbroadcast(rw, loc, op.getA());
      newB = mergeBatch(rw, loc, op.getB(), op.getBTransposed());
      newC = mergeBatch(rw, loc, op.getC(), op.getCTransposed());
    }

    // Create the new GemmOp
    rw.create<rock::GemmOp>(
        op.getLoc(), op.getResultTypes(), newA, newB, newC, op.getATransposed(),
        op.getBTransposed(), op.getCTransposed(), op.getArch(),
        op.getNumCUAttr(), op.getFeatures(), op.getStoreMethod(),
        op.getDerivedBlockSizeAttr(), op.getGridSizeAttr(), op.getParamsAttr());

    rw.eraseOp(op);
    return success();
  }
};

struct RockFoldBroadcastPass
    : public rock::impl::RockFoldBroadcastPassBase<RockFoldBroadcastPass> {
  void runOnOperation() override;
};
} // end namespace

void RockFoldBroadcastPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    // disable for non-kernels
    return;
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<FoldBroadcast>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
}
