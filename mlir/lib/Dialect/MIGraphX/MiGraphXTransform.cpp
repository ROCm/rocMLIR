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

#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MIGraphX/PassDetail.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace migraphx;

namespace {

class SqrtDecompose final : public OpConversionPattern<migraphx::SqrtOp> {
public:
  using OpConversionPattern<migraphx::SqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto inA = op->getOperand(0);
    auto outputTy = op->getResults()[0].getType().cast<ShapedType>();
    auto rSop = rewriter.create<migraphx::RsqrtOp>(loc, outputTy, inA);
    auto rCop = rewriter.create<migraphx::RecipOp>(loc, outputTy, rSop);

    rewriter.replaceOp(op, rCop->getResults()[0]);
    return success();
  }
};

void populateMIGraphXSqrt(MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<SqrtDecompose>(context);
}

struct MIGraphXTransforms
    : public MIGraphXTransformPassBase<MIGraphXTransforms> {
  void runOnOperation() override {
    auto &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<migraphx::MIGraphXDialect, func::FuncDialect>();
    target.addIllegalOp<migraphx::SqrtOp>();
    auto func = getOperation();

    populateMIGraphXSqrt(&ctx, patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
    /*
    if (applyPatternsAndFoldGreedily(func, std::move(patterns)).failed())
      signalPassFailure();
      */
  }
};

} // namespace

std::unique_ptr<Pass> mlir::migraphx::createMIGraphXTransformPass() {
  return std::make_unique<MIGraphXTransforms>();
}
