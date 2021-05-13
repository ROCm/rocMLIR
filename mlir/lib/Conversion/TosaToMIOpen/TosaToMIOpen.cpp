//===- TosaToMIOpen.cpp - Lowering Tosa to MIOpen Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the MIOpen dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToMIOpen/TosaToMIOpen.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

class ConvConverter final : public OpConversionPattern<tosa::Conv2DOp> {
public:
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;

  Value expandTensor(tosa::Conv2DOp op, Value operand,
                     ConversionPatternRewriter &rewriter) const {
    auto context = rewriter.getContext();
    auto oprType = operand.getType().template cast<ShapedType>();
    if (!oprType.hasStaticShape()) {
      (void)rewriter.notifyMatchFailure(
          op, "tosa to miopen conversion expects statically shaped tensors");
    }
    auto shape = oprType.getShape();
    SmallVector<int64_t, 5> expShape(shape.begin(), shape.end());
    expShape.push_back(1);
    auto newType = MemRefType::get(expShape, oprType.getElementType());

    SmallVector<linalg::ReassociationExprs, 5> reassociations;
    uint32_t dim = 0;
    for (; dim < shape.size() - 1; ++dim) {
      reassociations.push_back({getAffineDimExpr(dim, context)});
    }
    // last dimension + g dimension
    reassociations.push_back(
        {getAffineDimExpr(dim, context), getAffineDimExpr(dim + 1, context)});

    auto oprExpand = rewriter.create<mlir::linalg::ReshapeOp>(
        op->getLoc(), newType, operand, reassociations);

    return oprExpand;
  }

  LogicalResult
  matchAndRewrite(tosa::Conv2DOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto input_t = operands[0];
    auto filter_t = operands[1];
    auto bias_t = operands[2]; // TODO(sjw): add bias op linalg
    auto results = op->getResults();

    assert(results.size() == 1);

    // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
    auto inputExpanded = expandTensor(op, input_t, rewriter);

    auto filterExpanded = expandTensor(op, filter_t, rewriter);

    auto outputType = getTypeConverter<BufferizeTypeConverter>()
                          ->convertType(results[0].getType())
                          .cast<MemRefType>();
    Value output_mr = rewriter.create<AllocOp>(loc, outputType);
    auto outputExpanded = expandTensor(op, output_mr, rewriter);

    ValueRange args({filterExpanded, inputExpanded, outputExpanded});

    // Construct a new Conv2DOp.
    TypeRange resTypes;
    auto cop = rewriter.create<mlir::miopen::Conv2DOp>(loc, resTypes, args);

    StringRef arch = "gfx906";
    int32_t num_cu = 64;

    // translate attributes
    int32_t padTop = op.pad()[0].dyn_cast<IntegerAttr>().getInt();
    int32_t padBottom = op.pad()[1].dyn_cast<IntegerAttr>().getInt();
    int32_t padLeft = op.pad()[2].dyn_cast<IntegerAttr>().getInt();
    int32_t padRight = op.pad()[3].dyn_cast<IntegerAttr>().getInt();
    int32_t strideHeight = op.stride()[0].dyn_cast<IntegerAttr>().getInt();
    int32_t strideWidth = op.stride()[1].dyn_cast<IntegerAttr>().getInt();
    int32_t dilationHeight = op.dilation()[0].dyn_cast<IntegerAttr>().getInt();
    int32_t dilationWidth = op.dilation()[1].dyn_cast<IntegerAttr>().getInt();

    const char *filterLayout = "kyxcg";
    const char *inputLayout = "nhwcg";
    const char *outputLayout = "nhwkg";
    SmallVector<StringAttr, 5> filterLayoutSpec;
    SmallVector<StringAttr, 5> inputLayoutSpec;
    SmallVector<StringAttr, 5> outputLayoutSpec;
    for (size_t i = 0; i < 5; ++i) {
      filterLayoutSpec.push_back(
          rewriter.getStringAttr(StringRef(&filterLayout[i], 1).str()));
      inputLayoutSpec.push_back(
          rewriter.getStringAttr((StringRef(&inputLayout[i], 1) + "i").str()));
      outputLayoutSpec.push_back(
          rewriter.getStringAttr((StringRef(&outputLayout[i], 1) + "o").str()));
    }

    cop->setAttr("arch", rewriter.getStringAttr(arch));
    cop->setAttr("num_cu", rewriter.getI32IntegerAttr(num_cu));

    cop->setAttr("filter_layout",
                 rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                     filterLayoutSpec.begin(), filterLayoutSpec.end())));
    cop->setAttr("input_layout",
                 rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                     inputLayoutSpec.begin(), inputLayoutSpec.end())));
    cop->setAttr("output_layout",
                 rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                     outputLayoutSpec.begin(), outputLayoutSpec.end())));

    cop->setAttr("dilations", rewriter.getArrayAttr({
                                  rewriter.getI32IntegerAttr(dilationHeight),
                                  rewriter.getI32IntegerAttr(dilationWidth),
                              }));
    cop->setAttr("strides", rewriter.getArrayAttr({
                                rewriter.getI32IntegerAttr(strideHeight),
                                rewriter.getI32IntegerAttr(strideWidth),
                            }));
    cop->setAttr("padding", rewriter.getArrayAttr({
                                rewriter.getI32IntegerAttr(padTop),
                                rewriter.getI32IntegerAttr(padBottom),
                                rewriter.getI32IntegerAttr(padLeft),
                                rewriter.getI32IntegerAttr(padRight),
                            }));

    rewriter.replaceOp(op, output_mr);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToMIOpenOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  static BufferizeTypeConverter bufferizer;
  patterns->insert<ConvConverter>(bufferizer, context);
}
