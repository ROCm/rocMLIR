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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

class ConvConverter final : public OpConversionPattern<tosa::Conv2DOp> {
public:
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;

  Value expandMemRef(tosa::Conv2DOp op, Value operand,
                     ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto context = rewriter.getContext();
    auto oprType = operand.getType().template cast<ShapedType>();
    if (!oprType.hasStaticShape()) {
      (void)rewriter.notifyMatchFailure(
          op, "tosa to miopen conversion expects statically shaped tensors");
      return Value();
    }
    auto shape = oprType.getShape();
    SmallVector<int64_t, 5> expShape(shape.begin(), shape.end());
    expShape.push_back(1);
    auto newType = MemRefType::get(expShape, oprType.getElementType());

    SmallVector<ReassociationExprs, 5> reassociations;
    uint32_t dim = 0;
    for (; dim < shape.size() - 1; ++dim) {
      reassociations.push_back({getAffineDimExpr(dim, context)});
    }

    // last dimension + g dimension
    reassociations.push_back(
        {getAffineDimExpr(dim, context), getAffineDimExpr(dim + 1, context)});

    auto oprExpand = rewriter.create<memref::ExpandShapeOp>(
        loc, newType, operand, reassociations);
    return oprExpand;
  }

  LogicalResult
  matchAndRewrite(tosa::Conv2DOp op, tosa::Conv2DOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    auto loc = op->getLoc();
    auto context = op->getContext();
    auto input_t = operands[0];
    auto filter_t = operands[1];
    auto bias_mr = operands[2];
    auto results = op->getResults();

    // attach kernel attr to parent function
    auto func = op->getParentOfType<FuncOp>();
    func->setAttr("kernel", rewriter.getUnitAttr());

    assert(results.size() == 1);

    // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
    auto inputExpanded = expandMemRef(op, input_t, rewriter);

    auto filterExpanded = expandMemRef(op, filter_t, rewriter);

    auto outputType = getTypeConverter<mlir::bufferization::BufferizeTypeConverter>()
                          ->convertType(results[0].getType())
                          .cast<MemRefType>();
    Value output = rewriter.create<memref::AllocOp>(loc, outputType);
    auto outputExpanded = expandMemRef(op, output, rewriter);

    SmallVector<Value, 4> args({filterExpanded, inputExpanded, outputExpanded});

    // Construct a new Conv2DOp.
    TypeRange resultTypes;
    auto cop = rewriter.create<miopen::Conv2DOp>(
        loc, resultTypes, ValueRange{args[0], args[1], args[2]});

    // TODO(sjw): get these from options
    StringRef arch = "gfx906";
    uint32_t num_cu = 64;
    bool xdlopsV2 = false;

    if (auto attr = op->getAttrOfType<StringAttr>("arch"))
      arch = attr.getValue();
    else if (auto attr = func->getAttrOfType<StringAttr>("arch"))
      arch = attr.getValue();

    if (auto attr = op->getAttrOfType<IntegerAttr>("num_cu"))
      num_cu = attr.getValue().getZExtValue();
    else if (auto attr = func->getAttrOfType<IntegerAttr>("num_cu"))
      num_cu = attr.getValue().getZExtValue();

    if (auto attr = op->getAttrOfType<BoolAttr>("xdlopsV2"))
      xdlopsV2 = attr.getValue();
    else if (auto attr = func->getAttrOfType<BoolAttr>("xdlopsV2"))
      xdlopsV2 = attr.getValue();

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
    // TODO: remove these
    cop->setAttr("arch", rewriter.getStringAttr(arch));
    cop->setAttr("num_cu", rewriter.getI32IntegerAttr(num_cu));
    cop->setAttr("xdlopsV2", rewriter.getBoolAttr(xdlopsV2));

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

    // test for zero bias, and ignore
    auto bias_t = op.getOperand(2);
    bool zero_bias = false;
    if (auto cst = bias_t.getDefiningOp<arith::ConstantOp>()) {
      auto val = cst.getValue().cast<ElementsAttr>();
      auto valType = val.getType().dyn_cast<ShapedType>();
      auto valElemType = valType.getElementType();
      if (valElemType.isa<FloatType>()) {
        zero_bias = true;
        for (auto ii = val.value_begin<APFloat>();
             zero_bias && ii != val.value_end<APFloat>(); ++ii)
          zero_bias &= (*ii).isZero();
      } else if (valElemType.isa<IntegerType>()) {
        zero_bias = true;
        for (auto ii = val.value_begin<APInt>();
             zero_bias && ii != val.value_end<APInt>(); ++ii)
          zero_bias &= (*ii).isZero();
      }
    }
    if (!zero_bias) {
      // non-zero bias, replace with tosa.add w/ broadcast
      auto conv_output_t = rewriter.create<bufferization::ToTensorOp>(loc, output);

      auto biasType = bias_mr.getType().template cast<ShapedType>();
      if (!biasType.hasStaticShape())
        return failure();

      SmallVector<int64_t, 4> bias_s{1, 1, 1};
      bias_s.push_back(biasType.getShape()[0]);
      auto newType = MemRefType::get(bias_s, biasType.getElementType());

      SmallVector<ReassociationExprs, 1> reassociations;

      // [[0, 1, 2, 3]]
      reassociations.push_back(
          {getAffineDimExpr(0, context), getAffineDimExpr(1, context),
           getAffineDimExpr(2, context), getAffineDimExpr(3, context)});

      auto bias_expand_mr = rewriter.create<memref::ExpandShapeOp>(
          loc, newType, bias_mr, reassociations);

      auto bias_t = rewriter.create<bufferization::ToTensorOp>(loc, bias_expand_mr);
      output = rewriter.create<tosa::AddOp>(loc, op.getType(),
                                            ValueRange{conv_output_t, bias_t});
    }

    rewriter.replaceOp(op, output);

    return success();
  }
};

} // namespace

void tosa::populateTosaToMIOpenConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  static mlir::bufferization::BufferizeTypeConverter bufferizer;
  patterns->insert<ConvConverter>(bufferizer, context);
}
