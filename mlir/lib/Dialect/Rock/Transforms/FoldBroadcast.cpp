//===- FoldBroadcast.cpp - fold a broadcasted batch dim  ------===//
//
// Copyright 2024 Advanced Micro Devices.
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

#define DEBUG_TYPE "rock-fold-broadcast"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct FoldBroadcast : public OpRewritePattern<rock::GemmOp> {
  using OpRewritePattern<rock::GemmOp>::OpRewritePattern;

  // Analze the stack to verify if the batch size is a broadcast
  bool isBatchDimFoldableInTheTransformStack(ArrayAttr views) const {
    // We start from the batch-size dimension (which is always the 0-th
    // dimension in gemm)
    DenseSet<uint32_t> workList{0};

    // Let's walk the transform stack backwards (from top to bottom)
    size_t level = 0;

    while (!workList.empty() && level < views.size()) {
      auto curMap = views[level++].cast<TransformMapAttr>();
      auto ops = curMap.getOps();
      DenseSet<uint32_t> newWorkList;
      // Let's consider all the operations of the current transform map
      for (auto tr : ops) {
        for (auto [idx, upperDim] : llvm::enumerate(tr.getUpperDims())) {
          // If the current operation is transforming one of the dimensions
          // we are looking for, then dive into the operation to decide what to
          // do
          if (workList.contains(upperDim)) {
            switch (tr.getType()) {
            // If it is a single-length broadcast, don't do
            // anything, otherwise we need to be sure that the new
            // dimension is a single-length broadcast
            case rock::TransformType::Broadcast:
              if (tr.getParams()[idx] != 1)
                newWorkList.insert(tr.getLowerDims()[idx]);
              break;
            // AddDim and ConstDim are basically broadcasts. No
            // need to go further
            case rock::TransformType::AddDim:
            case rock::TransformType::ConstDim:
              break;
            // Follow the indices for the transformation that reroute them
            case rock::TransformType::PassThrough:
            case rock::TransformType::Slice:
            case rock::TransformType::Pad:
              newWorkList.insert(tr.getLowerDims()[idx]);
              break;
            // For a merge to be a valid broadcast
            // we need to ensure that all their (lower) dimensions
            // bigger than 1 lead to broadcasts
            case rock::TransformType::Merge:
              for (auto [length, dim] :
                   llvm::zip(tr.getParams(), tr.getLowerDims())) {
                if (length != 1)
                  newWorkList.insert(dim);
              }
              break;
            // For an umerge/embed, just follow the single dimension
            // down.
            case rock::TransformType::Unmerge:
            case rock::TransformType::Embed:
              newWorkList.insert(tr.getLowerDims().back());
              break;
            }
          }
        }
      }
      workList = newWorkList;
    }
    // If we want top/down and we determined that the batch dimension
    // led to a broadcast, then return true.
    return workList.empty();
  }

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

    return isBatchDimFoldableInTheTransformStack(views);
  }

  // Merge the batch dimension into either M or N, i.e., transform (d0, d1, d2)
  // into (d0*d1, d2) or (d1, d0*d2)
  Value mergeBatch(PatternRewriter &rw, Location loc,
                   TypedValue<ShapedType> buffer, bool isTransposed) const {
    auto shape = buffer.getType().getShape();
    ArrayAttr mergeBatchAttr;
    if (isTransposed) {
      rock::TopDownTMBuilder mergeBatchBuilder(
          rw, {"d0", "gd1"}, {shape[1], shape[0] * shape[2]}, loc);
      mergeBatchBuilder.merge({"g", "d1"}, {0, 2}, "gd1", {shape[0], shape[2]});
      mergeBatchBuilder.passThrough({"d0"}, {1}, {"d0"});
      mergeBatchAttr = rw.getArrayAttr({mergeBatchBuilder.get()});
    } else {
      rock::TopDownTMBuilder mergeBatchBuilder(
          rw, {"gd0", "d1"}, {shape[0] * shape[1], shape[2]}, loc);
      mergeBatchBuilder.merge({"g", "d0"}, {0, 1}, "gd0", {shape[0], shape[1]});
      mergeBatchBuilder.passThrough({"d1"}, {2}, {"d1"});
      mergeBatchAttr = rw.getArrayAttr({mergeBatchBuilder.get()});
    }
    return rock::transform(rw, buffer, mergeBatchAttr);
  }

  // Select the 0th slice from a broadcast, de facto removing the broadcast
  // dimension
  Value unbroadcastBatch(PatternRewriter &rw, Location loc,
                         TypedValue<ShapedType> buffer) const {
    auto shape = buffer.getType().getShape();
    rock::TopDownTMBuilder unbroadcastBuilder(rw, {"d0", "d1"},
                                              {shape[1], shape[2]}, loc);
    unbroadcastBuilder.constDim({"g"}, 0, 0, shape[0]);
    unbroadcastBuilder.passThrough({"d0", "d1"}, {1, 2}, {"d0", "d1"});
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
      newA = unbroadcastBatch(rw, loc, op.getA());
      newB = unbroadcastBatch(rw, loc, op.getB());
      newC = unbroadcastBatch(rw, loc, op.getC());
    } else if (isBBatchBroadcast) {
      newA = mergeBatch(rw, loc, op.getA(), op.getATransposed());
      newB = unbroadcastBatch(rw, loc, op.getB());
      newC = mergeBatch(rw, loc, op.getC(), op.getCTransposed());
    } else { // isABatchBroadcast
      // When the broadcasted batch is on A, matrix B and C need
      // to be considered as if they were transposed
      newA = unbroadcastBatch(rw, loc, op.getA());
      newB = mergeBatch(rw, loc, op.getB(), !op.getBTransposed());
      newC = mergeBatch(rw, loc, op.getC(), !op.getCTransposed());
    }

    // Create the new GemmOp
    auto gemm = rw.create<rock::GemmOp>(
        op.getLoc(), newC.getType(), newA, newB, newC, op.getATransposed(),
        op.getBTransposed(), op.getCTransposed(), op.getArch(),
        op.getNumCUAttr(), op.getFeatures(), op.getStoreMethod(),
        op.getDerivedBlockSizeAttr(), op.getGridSizeAttr(), op.getParamsAttr());

    // Remove dummy transforms from the gemm output and use it to replace the
    // original op through all the IR
    Value result = rw.create<rock::TensorUntransformCastOp>(
        loc, op.getC().getType().cast<RankedTensorType>(), gemm.getResult(),
        gemm.getC());
    rw.replaceOp(op, result);

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
