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

#if 0
    std::vector<NamedAttribute> attributes{
      rewriter.getNamedAttr("arch", rewriter.getStringAttr(arch)),
          rewriter.getNamedAttr("num_cu", rewriter.getI32IntegerAttr(num_cu)),

          rewriter.getNamedAttr(
              "filter_layout",
              rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                                       filterLayoutSpec.begin(), filterLayoutSpec.end()))),
          rewriter.getNamedAttr(
              "input_layout", rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                                                       inputLayoutSpec.begin(), inputLayoutSpec.end()))),
          rewriter.getNamedAttr(
              "output_layout",
              rewriter.getArrayAttr(ArrayRef<mlir::Attribute>(
                                       outputLayoutSpec.begin(), outputLayoutSpec.end()))),

          rewriter.getNamedAttr("dilations",
                               rewriter.getArrayAttr({
                                   rewriter.getI32IntegerAttr(dilationHeight),
                                       rewriter.getI32IntegerAttr(dilationWidth),
                                       })),
          rewriter.getNamedAttr("strides",
                               rewriter.getArrayAttr({
                                   rewriter.getI32IntegerAttr(strideHeight),
                                       rewriter.getI32IntegerAttr(strideWidth),
                                       })),
          rewriter.getNamedAttr("padding",
                               rewriter.getArrayAttr({
                                   rewriter.getI32IntegerAttr(paddingHeight),
                                       rewriter.getI32IntegerAttr(paddingWidth),
                                       })),
          };

      // xdlops v2.
  // if (xdlops)
  //   attributes.push_back(
  //       rewriter.getNamedAttr("xdlopsV2", rewriter.getBoolAttr(true)));
    
#endif

      TypeRange resTypes;
      auto cop = rewriter.create<mlir::miopen::Conv2DOp>(loc, resTypes, args);


#if 1
    // Construct a new Conv2DOp.
    StringRef arch = "gfx906";
    int32_t num_cu = 64;

    int32_t dilationHeight=1, dilationWidth=1;
    int32_t strideHeight=1, strideWidth=1;
    int32_t paddingHeight=0, paddingWidth=0;
    
    std::string filterLayout = "kcyx";
    std::string inputLayout = "nchw";
    std::string outputLayout = "nkhw";
    SmallVector<StringAttr, 4> filterLayoutSpec;
    SmallVector<StringAttr, 4> inputLayoutSpec;
    SmallVector<StringAttr, 4> outputLayoutSpec;
    for (size_t i = 0; i < 4; ++i) {
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
#endif
      
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

