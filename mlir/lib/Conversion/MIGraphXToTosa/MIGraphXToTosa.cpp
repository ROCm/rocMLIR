//===- MIGraphXToTosa.cpp - Lowering MIGraphX to Tosa Dialec -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the MIGraphX to the Tosa dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MIGraphXToTosa/MIGraphXToTosa.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


using namespace mlir;

namespace {

class BiasedConvConverter final : public OpConversionPattern<migraphx::add> {
public:
  using OpConversionPattern<migraphx::add>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::add op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
                    
    auto loc = op->getLoc();
    auto input_t = operands[0];
    auto filter_t = operands[1];
    // TODO(sjw): add bias op linalg.
    // auto bias_t = operands[2];
    auto results = op->getResults();

    // attach kernel attr to parent function
    auto func = op->getParentOfType<FuncOp>();
    func->setAttr("kernel", rewriter.getUnitAttr());

    assert(results.size() == 1);

    // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
    auto inputExpanded = expandMemRef(op, input_t, rewriter);

    auto filterExpanded = expandMemRef(op, filter_t, rewriter);

    auto outputType = getTypeConverter<BufferizeTypeConverter>()
                          ->convertType(results[0].getType())
                          .cast<MemRefType>();
    Value output_mr = rewriter.create<memref::AllocOp>(loc, outputType);
    auto outputExpanded = expandMemRef(op, output_mr, rewriter);

    SmallVector<Value, 4> args({filterExpanded, inputExpanded, outputExpanded});

    // Construct a new Conv2DOp.
    TypeRange resultTypes;
    auto cop = rewriter.create<miopen::Conv2DOp>(
        loc, resultTypes, ValueRange{args[0], args[1], args[2]});

    // TODO(sjw): get these from options
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

    // specify layout attributes
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

    // arch-specific attributes
    cop->setAttr("arch", rewriter.getStringAttr(arch));
    cop->setAttr("num_cu", rewriter.getI32IntegerAttr(num_cu));

    // convolution config attributes
    cop->setAttr("filter_layout",
                 rewriter.getArrayAttr(ArrayRef<Attribute>(
                     filterLayoutSpec.begin(), filterLayoutSpec.end())));
    cop->setAttr("input_layout",
                 rewriter.getArrayAttr(ArrayRef<Attribute>(
                     inputLayoutSpec.begin(), inputLayoutSpec.end())));
    cop->setAttr("output_layout",
                 rewriter.getArrayAttr(ArrayRef<Attribute>(
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

void tosa::populateTosaToMIOpenConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  static BufferizeTypeConverter bufferizer;
  patterns->insert<ConvConverter>(bufferizer, context);
}
