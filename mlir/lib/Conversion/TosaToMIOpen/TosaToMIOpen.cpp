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
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
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

  LogicalResult matchAndRewrite(tosa::Conv2DOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto results = op->getResults();
    auto input_t = operands[0];
    auto filter_t = operands[1];
    auto bias_t = operands[2];

    assert(results.size() == 1);

    for (Value in : op->getOperands()) {
      auto inType = in.getType().template cast<ShapedType>();
      if (!inType.hasStaticShape())
        return rewriter.notifyMatchFailure(
                                           op,
                                           "tosa to miopen conversion expects statically shaped tensors");
    }

    for (auto result : results) {
      auto resultType = result.getType().template cast<ShapedType>();
      if (!resultType.hasStaticShape())
        return rewriter.notifyMatchFailure(
                                           op,
                                           "tosa to linalg conversion expects statically shaped tensors");
    }

    auto outputType = getTypeConverter<BufferizeTypeConverter>()->convertType(results[0].getType()).cast<MemRefType>();
    
    Value output_t = rewriter.create<AllocOp>(loc, outputType);

    ValueRange args({filter_t, input_t, output_t});

    TypeRange resTypes;
    auto cop = rewriter.create<mlir::miopen::Conv2DOp>(loc, resTypes, args);

    // Construct a new Conv2DOp.
    StringRef arch = "gfx906";
    int32_t num_cu = 64;

    ArrayAttr padArr = op.pad();
    int32_t paddingHeight = padArr[0].dyn_cast<IntegerAttr>().getInt();
    int32_t paddingWidth = padArr[1].dyn_cast<IntegerAttr>().getInt();
    // Add support for 4 side padding
    // assert( padding[2] == padding[0] );
    // assert( padding[3] == padding[1] );
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

    cop->setAttr(
        "filter_layout",
        rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                                  filterLayoutSpec.begin(), filterLayoutSpec.end())));
    cop->setAttr(
        "input_layout", rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                                                  inputLayoutSpec.begin(), inputLayoutSpec.end())));
    cop->setAttr(
        "output_layout",
        rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                                  outputLayoutSpec.begin(), outputLayoutSpec.end())));

    cop->setAttr("dilations",
                 rewriter.getArrayAttr({
                     rewriter.getI32IntegerAttr(dilationHeight),
                         rewriter.getI32IntegerAttr(dilationWidth),
                         }));
    cop->setAttr("strides",
                 rewriter.getArrayAttr({
                     rewriter.getI32IntegerAttr(strideHeight),
                         rewriter.getI32IntegerAttr(strideWidth),
                         }));
    cop->setAttr("padding",
                 rewriter.getArrayAttr({
                     rewriter.getI32IntegerAttr(paddingHeight),
                         rewriter.getI32IntegerAttr(paddingWidth),
                         }));
      
    rewriter.replaceOp(op, output_t);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToMIOpenOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  static BufferizeTypeConverter bufferizer;
  patterns->insert<ConvConverter>(bufferizer, context);
}

