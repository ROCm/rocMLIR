//===- ViewToRock.cpp - Lowering Tensor to Rock Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tensor view operations to the Rock
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToRock/TosaToRock.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKVIEWTOTRANSFORMPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-view-to-transform"

using namespace mlir;

namespace {

struct CollapseShapeRewritePattern
    : public OpConversionPattern<tensor::CollapseShapeOp> {
  using OpConversionPattern<tensor::CollapseShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final {
    Location loc = collapseOp.getLoc();
    ArrayRef<int64_t> inpShape = collapseOp.getSrcType().getShape();
    ArrayRef<int64_t> outShape = collapseOp.getResultType().getShape();
    SmallVector<ReassociationIndices, 4> reassocs =
        collapseOp.getReassociationIndices();

    rock::TransformMapAttr collapseAttr =
        rock::transformCollapseShape(b, loc, inpShape, outShape, reassocs);
    if (!collapseAttr)
      return b.notifyMatchFailure(
          loc, "couldn't translate tensor collapse into rock transforms");
    b.replaceOpWithNewOp<rock::TransformOp>(collapseOp, adaptor.getSrc(),
                                            collapseAttr);
    return success();
  }
};

struct ExpandShapeRewritePattern
    : public OpConversionPattern<tensor::ExpandShapeOp> {
  using OpConversionPattern<tensor::ExpandShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final {
    Location loc = expandOp.getLoc();
    ArrayRef<int64_t> inpShape = expandOp.getSrcType().getShape();
    ArrayRef<int64_t> outShape = expandOp.getResultType().getShape();
    SmallVector<ReassociationIndices, 4> reassocs =
        expandOp.getReassociationIndices();

    rock::TransformMapAttr expandAttr =
        rock::transformExpandShape(b, loc, inpShape, outShape, reassocs);
    if (!expandAttr)
      return b.notifyMatchFailure(
          loc, "could not translate tensor expansion into rock transform");
    b.replaceOpWithNewOp<rock::TransformOp>(expandOp, adaptor.getSrc(),
                                            expandAttr);
    return success();
  }
};

struct TransposeRewritePattern : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern<tosa::TransposeOp>::OpRewritePattern;

  LogicalResult getTransposeDims(Value v, SmallVector<int32_t> &perms) const {
    Operation *cval = v.getDefiningOp();
    if (isa<arith::ConstantOp>(cval) || isa<tosa::ConstOp>(cval)) {
      auto cattr = cval->getAttr("value").cast<DenseElementsAttr>();
      auto vals = cattr.tryGetValues<int32_t>();
      if (succeeded(vals)) {
        perms.assign((*vals).begin(), (*vals).end());
        return success();
      }
      auto vals64 = cattr.tryGetValues<int64_t>();
      if (succeeded(vals64)) {
        perms.assign((*vals64).begin(), (*vals64).end());
        return success();
      }
    }
    return failure();
  }

  // Fold transpose ops and convert convolution into changed layout.
  // case #0 : fold TP(NCHW2NHWC)+tosa.conv.NHWC+TP(NHWC2NCHW) back to
  //           rock.conv.NCHW
  // Pattern match start from the output transpose
  LogicalResult matchAndRewrite(tosa::TransposeOp top,
                                PatternRewriter &b) const final {
    SmallVector<int32_t> perms;
    if (failed(getTransposeDims(top.getOperand(1), perms)))
      return failure();

    Location loc = top.getLoc();
    Value inp = top.getOperand(0);
    ShapedType inpType = inp.getType().template cast<ShapedType>();
    ArrayRef<int64_t> inpShape = inpType.getShape();
    assert(perms.size() == inpShape.size());

    SmallVector<uint32_t, 8> endDims;
    SmallVector<uint32_t, 8> startDims;
    for (uint32_t i = 0, e = inpShape.size(); i < e; ++i) {
      startDims.push_back(perms[i]);
      endDims.push_back(i);
    }
    rock::BottomUpTMBuilder transform(b, inpShape, loc);
    transform.passThrough(endDims, startDims);
    b.replaceOpWithNewOp<rock::TransformOp>(top, inp, transform.get());

    return success();
  }
};

struct RockViewToTransform
    : public rock::impl::RockViewToTransformPassBase<RockViewToTransform> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    if (!func->hasAttr("kernel")) {
      return;
    }
    MLIRContext *ctx = &getContext();
    // Split patterns into two stages by bufferization
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    target.addLegalDialect<rock::RockDialect,
                           bufferization::BufferizationDialect>();
    target.addIllegalOp<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
                        tosa::TransposeOp>();

    patterns.add<TransposeRewritePattern, CollapseShapeRewritePattern,
                 ExpandShapeRewritePattern>(ctx);

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
