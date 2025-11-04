//===- TosaOptionalDecompositions.cpp
//------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to apply the Tosa operations decompositions
// exposed as populate functions in
// include/mlir/Dialect/Tosa/Transforms/Passes.h
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXTRANSFORMPASS
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {

class QuantDotDecompose final : public OpRewritePattern<migraphx::QuantDotOp> {
public:
  using OpRewritePattern<migraphx::QuantDotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(migraphx::QuantDotOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    if (!op.getScaleA() || !op.getScaleB()) {
      return failure();
    }
    auto inA = op.getInA();
    auto inB = op.getInB();
    auto inACvt = migraphx::ConvertOp::create(
        rewriter, loc,
        MIXRShapedType::get(inA.getType().getShape(),
                            inA.getType().getStrides(), rewriter.getF32Type()),
        inA);
    auto inBCvt = migraphx::ConvertOp::create(
        rewriter, loc,
        MIXRShapedType::get(inB.getType().getShape(),
                            inB.getType().getStrides(), rewriter.getF32Type()),
        inB);
    auto scaleA = op.getScaleA();
    auto scaleB = op.getScaleB();
    if (scaleA.getType().getElementType() != rewriter.getF32Type()) {
      scaleA = migraphx::ConvertOp::create(
          rewriter, loc,
          MIXRShapedType::get(scaleA.getType().getShape(),
                              scaleA.getType().getStrides(),
                              rewriter.getF32Type()),
          scaleA);
    }
    if (scaleB.getType().getElementType() != rewriter.getF32Type()) {
      scaleB = migraphx::ConvertOp::create(
          rewriter, loc,
          MIXRShapedType::get(scaleB.getType().getShape(),
                              scaleB.getType().getStrides(),
                              rewriter.getF32Type()),
          scaleB);
    }
    auto newA = migraphx::MulOp::create(
        rewriter, loc,
        MIXRShapedType::get(inACvt.getType().getShape(),
                            inACvt.getType().getStrides(),
                            rewriter.getF32Type()),
        inACvt, scaleA);
    auto newB = migraphx::MulOp::create(
        rewriter, loc,
        MIXRShapedType::get(inBCvt.getType().getShape(),
                            inBCvt.getType().getStrides(),
                            rewriter.getF32Type()),
        inBCvt, scaleB);
    auto dotOp = migraphx::DotOp::create(
        rewriter, loc, op.getResult().getType(), newA, newB, nullptr, nullptr);
    rewriter.replaceOp(op, dotOp->getResults()[0]);
    return success();
  }
};

class SqrtDecompose final : public OpConversionPattern<migraphx::SqrtOp> {
public:
  using OpConversionPattern<migraphx::SqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto inA = op->getOperand(0);
    auto outputTy = cast<ShapedType>(op->getResults()[0].getType());
    auto rSop = migraphx::RsqrtOp::create(rewriter, loc, outputTy, inA);
    auto rCop = migraphx::RecipOp::create(rewriter, loc, outputTy, rSop);

    rewriter.replaceOp(op, rCop->getResults()[0]);
    return success();
  }
};

  void populateMIGraphXSqrt(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<SqrtDecompose>(context);
  }

  struct MIGraphXTransforms
      : public migraphx::impl::MIGraphXTransformPassBase<MIGraphXTransforms> {
    void runOnOperation() override {
      auto &ctx = getContext();
      RewritePatternSet patterns(&ctx);
      ConversionTarget target(ctx);
      target.addLegalDialect<migraphx::MIGraphXDialect, func::FuncDialect,
                             tosa::TosaDialect, mhal::MHALDialect>();
      target.addIllegalOp<migraphx::SqrtOp>();
      auto func = getOperation();

      populateMIGraphXSqrt(&ctx, patterns);
      if (failed(applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
      }
      {
        RewritePatternSet patterns(&ctx);
        patterns.add<QuantDotDecompose>(&ctx);
        if (failed(applyPatternsGreedily(func, std::move(patterns))))
          signalPassFailure();
      }
    }
  };

} // namespace
