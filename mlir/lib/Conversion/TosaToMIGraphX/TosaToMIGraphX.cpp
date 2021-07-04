//===- TosaToMIGraphX.cpp - Lowering Tosa to MIGraphX Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToMIGraphX/TosaToMIGraphX.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

class ConstRandomPattern : public OpRewritePattern<tosa::ConstOp> {
public:
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tosa::ConstOp op,
                  PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto results = op->getResults();
    auto resultType = results[0].getType().template cast<ShapedType>();
    auto shape = resultType.getShape();
    SmallVector<IntegerAttr, 5> shapeAttr;
    for(auto dim: shape){
      shapeAttr.push_back(rewriter.getI32IntegerAttr(dim));
    }

    ValueRange args({});
    auto cop = rewriter.create<mlir::migraphx::ConstantOp>(loc, resultType, args);
    cop->setAttr("shape",
                 rewriter.getArrayAttr(ArrayRef<Attribute>(shapeAttr.begin(), shapeAttr.end())));
    rewriter.replaceOp(op, cop->getResults());

    return success();
  }
};

} // namespace

namespace {

class DummyConverter : public OpConversionPattern<tosa::AddOp> {
public:
  using OpConversionPattern<tosa::AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::tosa::AddOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto results = op->getResults();
    auto resultType = results[0].getType().template cast<ShapedType>();
    ValueRange args({operands[0], operands[1]});
    auto dop = rewriter.create<mlir::AddFOp>(loc, resultType, args);
    rewriter.replaceOp(op, dop->getResults());

    return success();
  }
};

} // namespace

void mlir::tosa::populateConstRandomPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<ConstRandomPattern>(context);
}
