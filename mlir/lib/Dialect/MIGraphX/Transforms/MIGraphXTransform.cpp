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

    auto inA = op.getInA();
    auto inB = op.getInB();
    auto scaleA = op.getScaleA();
    auto scaleB = op.getScaleB();

    // Only decompose scaled GEMM operations (both scales must be present).
    // The verifier ensures both scales are provided together or neither.
    if (!scaleA && !scaleB) {
      return failure();
    }

    // Determine target output type
    auto resultType = op.getResult().getType();

    // For scaled operations, we always convert to F32 for proper computation
    Type computeElemType = rewriter.getF32Type();

    // Convert input A if needed
    Value processedA = inA;
    if (inA.getType().getElementType() != computeElemType) {
      processedA = migraphx::ConvertOp::create(
          rewriter, loc,
          MIXRShapedType::get(inA.getType().getShape(),
                              inA.getType().getStrides(), computeElemType),
          inA);
    }

    // Convert input B if needed
    Value processedB = inB;
    if (inB.getType().getElementType() != computeElemType) {
      processedB = migraphx::ConvertOp::create(
          rewriter, loc,
          MIXRShapedType::get(inB.getType().getShape(),
                              inB.getType().getStrides(), computeElemType),
          inB);
    }

    // Apply scaleA if present
    if (scaleA) {
      Value convertedScaleA = scaleA;
      if (scaleA.getType().getElementType() != computeElemType) {
        convertedScaleA = migraphx::ConvertOp::create(
            rewriter, loc,
            MIXRShapedType::get(scaleA.getType().getShape(),
                                scaleA.getType().getStrides(), computeElemType),
            scaleA);
      }
      processedA = migraphx::MulOp::create(
          rewriter, loc,
          MIXRShapedType::get(
              cast<MIXRShapedType>(processedA.getType()).getShape(),
              cast<MIXRShapedType>(processedA.getType()).getStrides(),
              computeElemType),
          processedA, convertedScaleA);
    }

    // Apply scaleB if present
    if (scaleB) {
      Value convertedScaleB = scaleB;
      if (scaleB.getType().getElementType() != computeElemType) {
        convertedScaleB = migraphx::ConvertOp::create(
            rewriter, loc,
            MIXRShapedType::get(scaleB.getType().getShape(),
                                scaleB.getType().getStrides(), computeElemType),
            scaleB);
      }
      processedB = migraphx::MulOp::create(
          rewriter, loc,
          MIXRShapedType::get(
              cast<MIXRShapedType>(processedB.getType()).getShape(),
              cast<MIXRShapedType>(processedB.getType()).getStrides(),
              computeElemType),
          processedB, convertedScaleB);
    }

    // Create the dot operation with processed inputs
    auto dotOp = migraphx::DotOp::create(rewriter, loc, resultType, processedA,
                                         processedB, nullptr, nullptr);
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
