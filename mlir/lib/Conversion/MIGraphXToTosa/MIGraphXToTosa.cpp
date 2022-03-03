//===- MIGraphXToTosa.cpp - Lowering MIGraphX to Tosa Dialect
//-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the MIGraphX to the Tos dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MIGraphXToTosa/MIGraphXToTosa.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

class ConvConverter final
    : public OpConversionPattern<migraphx::ConvolutionOp> {
public:
  using OpConversionPattern<migraphx::ConvolutionOp>::OpConversionPattern;

  Value getZero(Location loc, Type elemType,
                ConversionPatternRewriter &rewriter) const {
    auto biasTy = RankedTensorType::get({1}, elemType);
    auto zeroAttr =
        DenseElementsAttr::get(biasTy, rewriter.getZeroAttr(elemType));
    return rewriter.create<arith::ConstantOp>(loc, zeroAttr);
  }

  tosa::TransposeOp getRank4TransposeOp(Location loc, Value input,
                                        ConversionPatternRewriter &rewriter,
                                        SmallVector<int64_t> &permutation,
                                        bool bRoot) const {
    auto permutationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4}, rewriter.getI64Type()), permutation);
    Value permutationValue =
        rewriter.create<arith::ConstantOp>(loc, permutationAttr);
    ShapedType inputTy = input.getType().cast<ShapedType>();
    auto inputShape = inputTy.getShape();
    SmallVector<int64_t> newShape{
        inputShape[permutation[0]], inputShape[permutation[1]],
        inputShape[permutation[2]], inputShape[permutation[3]]};
    Type newTy = RankedTensorType::get(newShape, inputTy.getElementType());

    auto newOp =
        rewriter.create<tosa::TransposeOp>(loc, newTy, input, permutationValue);
    newOp->setAttr("changing_layout_root", rewriter.getBoolAttr(bRoot));
    return newOp;
  }

  LogicalResult
  matchAndRewrite(migraphx::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    Location loc = op->getLoc();
    auto input_t = operands[0];
    auto filter_t = operands[1];
    auto results = op->getResults();
    auto elementTy =
        op->getOperand(0).getType().cast<ShapedType>().getElementType();
    auto outputTy = results[0].getType().cast<ShapedType>();
    SmallVector<int64_t> NCHW2NHWC{0, 2, 3, 1};
    SmallVector<int64_t> NHWC2NCHW{0, 3, 1, 2};

    // insert transpose to input and filter tensors
    input_t = getRank4TransposeOp(loc, input_t, rewriter, NCHW2NHWC, false);
    filter_t = getRank4TransposeOp(loc, filter_t, rewriter, NCHW2NHWC, false);
    auto outShape = outputTy.getShape();

    // original output shape was NCHW, change it into NHWC
    SmallVector<int64_t> newShape{outShape[0], outShape[2], outShape[3],
                                  outShape[1]};
    Type newOutTy = RankedTensorType::get(newShape, outputTy.getElementType());

    // Construct a new Conv2DOp.
    auto cop = rewriter.create<tosa::Conv2DOp>(
        loc, newOutTy,
        ValueRange{input_t, filter_t, getZero(loc, elementTy, rewriter)});

    // translate attributes
    auto padAttr = op->getAttr("padding").cast<ArrayAttr>();
    auto strideAttr = op->getAttr("stride").cast<ArrayAttr>();
    auto dilationAttr = op->getAttr("dilation").cast<ArrayAttr>();
    int64_t padTop = padAttr[0].dyn_cast<IntegerAttr>().getInt();
    int64_t padBottom = padAttr[1].dyn_cast<IntegerAttr>().getInt();
    int64_t padLeft = padAttr[2].dyn_cast<IntegerAttr>().getInt();
    int64_t padRight = padAttr[3].dyn_cast<IntegerAttr>().getInt();
    int64_t strideHeight = strideAttr[0].dyn_cast<IntegerAttr>().getInt();
    int64_t strideWidth = strideAttr[1].dyn_cast<IntegerAttr>().getInt();
    int64_t dilationHeight = dilationAttr[0].dyn_cast<IntegerAttr>().getInt();
    int64_t dilationWidth = dilationAttr[1].dyn_cast<IntegerAttr>().getInt();

    // Record desired layout and set
    // convolution config attributes
    cop->setAttr("expected_filter_layout", rewriter.getStringAttr("kcyx"));
    cop->setAttr("expected_input_layout", rewriter.getStringAttr("nchw"));
    cop->setAttr("expected_output_layout", rewriter.getStringAttr("nkhw"));

    cop->setAttr("dilation", rewriter.getArrayAttr({
                                 rewriter.getI64IntegerAttr(dilationHeight),
                                 rewriter.getI64IntegerAttr(dilationWidth),
                             }));
    cop->setAttr("stride", rewriter.getArrayAttr({
                               rewriter.getI64IntegerAttr(strideHeight),
                               rewriter.getI64IntegerAttr(strideWidth),
                           }));
    cop->setAttr("pad", rewriter.getArrayAttr({
                            rewriter.getI64IntegerAttr(padTop),
                            rewriter.getI64IntegerAttr(padBottom),
                            rewriter.getI64IntegerAttr(padLeft),
                            rewriter.getI64IntegerAttr(padRight),
                        }));

    // transpose the output back to NCHW so that it can match following
    // operators.
    auto top = getRank4TransposeOp(loc, cop, rewriter, NHWC2NCHW, true);
    rewriter.replaceOp(op, {top});
    return success();
  }
};

} // namespace

void migraphx::populateMIGraphXToTosaConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.insert<ConvConverter>(context);
}
