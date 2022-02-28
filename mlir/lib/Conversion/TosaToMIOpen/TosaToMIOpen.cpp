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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Tell if the given tosa conv2d is supposed to have NCHW layout
static bool checkNCHW(tosa::Conv2DOp &convOp) {
  // Check if the convolution has expected layout
  auto fLayout = convOp->getAttr("expected_filter_layout");
  auto iLayout = convOp->getAttr("expected_input_layout");
  auto oLayout = convOp->getAttr("expected_output_layout");
  if (!fLayout || !iLayout || !oLayout) {
    return false;
  }
  // Test if all match
  if (!(fLayout.dyn_cast<StringAttr>().getValue().equals("kcyx") &&
        iLayout.dyn_cast<StringAttr>().getValue().equals("nchw") &&
        oLayout.dyn_cast<StringAttr>().getValue().equals("nkhw"))) {
    return false;
  }
  return true;
}

static bool isZeroAttribute(Attribute value) {
  if (auto intValue = value.dyn_cast<IntegerAttr>())
    return intValue.getValue().isNullValue();
  if (auto fpValue = value.dyn_cast<FloatAttr>())
    return fpValue.getValue().isZero();
  if (auto splatValue = value.dyn_cast<SplatElementsAttr>())
    return isZeroAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = value.dyn_cast<ElementsAttr>())
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = value.dyn_cast<ArrayAttr>())
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
}

static bool isConstantZero(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    return isZeroAttribute(cst.getValue());
  }
  return false;
}

static Value expandMemRef(ConversionPatternRewriter &rw, Operation *op,
                          Value operand, uint32_t idx = 4) {
  auto context = rw.getContext();
  auto oprType = operand.getType().template cast<ShapedType>();
  if (!oprType.hasStaticShape()) {
    (void)rw.notifyMatchFailure(
        op, "tosa to miopen conversion expects statically shaped tensors");
    return Value();
  }
  auto shape = oprType.getShape();

  idx--;

  // expShape = shape + 1 dim
  //  ex:  <NxHxWxCxf32> -> <Nx1xHxWxCxf32>
  SmallVector<int64_t, 5> expShape;
  // reassocs = shape w/ 1 split
  //  ex:  [0,1,[2,3],4]
  uint32_t d0 = 0;
  SmallVector<ReassociationExprs, 5> reassocs;
  for (uint32_t dim = 0; dim < shape.size(); ++dim) {
    expShape.push_back(shape[dim]);
    if (dim == idx) {
      expShape.push_back(1);
      reassocs.push_back(
          {getAffineDimExpr(d0++, context), getAffineDimExpr(d0++, context)});
    } else {
      reassocs.push_back({getAffineDimExpr(d0++, context)});
    }
  }

  auto newType = MemRefType::get(expShape, oprType.getElementType());
  auto oprExpand = rw.create<memref::ExpandShapeOp>(op->getLoc(), newType,
                                                    operand, reassocs);
  return oprExpand;
}

static LogicalResult
makeMIOpenConv2D(ConversionPatternRewriter &rw, Operation *op, Value input,
                 const char *inputLayout, Value filter,
                 const char *filterLayout, Value output,
                 const char *outputLayout, const ArrayAttr &pad,
                 const ArrayAttr &stride, const ArrayAttr &dilation) {
  auto loc = op->getLoc();
  auto func = op->getParentOfType<FuncOp>();

  // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
  auto inputExpanded = expandMemRef(rw, op, input);
  auto filterExpanded = expandMemRef(rw, op, filter);
  auto outputExpanded = expandMemRef(rw, op, output);

  ValueRange args{filterExpanded, inputExpanded, outputExpanded};

  // Construct a new Conv2DOp.
  TypeRange resultTypes;
  auto cop = rw.create<miopen::Conv2DOp>(loc, resultTypes, args);

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
  int32_t padTop = pad[0].dyn_cast<IntegerAttr>().getInt();
  int32_t padBottom = pad[1].dyn_cast<IntegerAttr>().getInt();
  int32_t padLeft = pad[2].dyn_cast<IntegerAttr>().getInt();
  int32_t padRight = pad[3].dyn_cast<IntegerAttr>().getInt();
  int32_t strideHeight = stride[0].dyn_cast<IntegerAttr>().getInt();
  int32_t strideWidth = stride[1].dyn_cast<IntegerAttr>().getInt();
  int32_t dilationHeight = dilation[0].dyn_cast<IntegerAttr>().getInt();
  int32_t dilationWidth = dilation[1].dyn_cast<IntegerAttr>().getInt();

  // specify layout attributes
  SmallVector<StringAttr, 5> filterLayoutSpec;
  SmallVector<StringAttr, 5> inputLayoutSpec;
  SmallVector<StringAttr, 5> outputLayoutSpec;
  for (size_t i = 0; i < 5; ++i) {
    filterLayoutSpec.push_back(
        rw.getStringAttr(StringRef(&filterLayout[i], 1).str()));
    inputLayoutSpec.push_back(
        rw.getStringAttr((StringRef(&inputLayout[i], 1) + "i").str()));
    outputLayoutSpec.push_back(
        rw.getStringAttr((StringRef(&outputLayout[i], 1) + "o").str()));
  }

  // arch-specific attributes
  // TODO: remove these
  cop->setAttr("arch", rw.getStringAttr(arch));
  cop->setAttr("num_cu", rw.getI32IntegerAttr(num_cu));
  cop->setAttr("xdlopsV2", rw.getBoolAttr(xdlopsV2));

  // convolution config attributes
  cop->setAttr("filter_layout",
               rw.getArrayAttr(ArrayRef<Attribute>(filterLayoutSpec.begin(),
                                                   filterLayoutSpec.end())));
  cop->setAttr("input_layout",
               rw.getArrayAttr(ArrayRef<Attribute>(inputLayoutSpec.begin(),
                                                   inputLayoutSpec.end())));
  cop->setAttr("output_layout",
               rw.getArrayAttr(ArrayRef<Attribute>(outputLayoutSpec.begin(),
                                                   outputLayoutSpec.end())));

  cop->setAttr("dilations", rw.getArrayAttr({
                                rw.getI32IntegerAttr(dilationHeight),
                                rw.getI32IntegerAttr(dilationWidth),
                            }));
  cop->setAttr("strides", rw.getArrayAttr({
                              rw.getI32IntegerAttr(strideHeight),
                              rw.getI32IntegerAttr(strideWidth),
                          }));
  cop->setAttr("padding", rw.getArrayAttr({
                              rw.getI32IntegerAttr(padTop),
                              rw.getI32IntegerAttr(padBottom),
                              rw.getI32IntegerAttr(padLeft),
                              rw.getI32IntegerAttr(padRight),
                          }));

  return success();
}

class ConvConverter final : public OpConversionPattern<tosa::Conv2DOp> {
public:
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp op,
                                tosa::Conv2DOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    auto operands = adaptor.getOperands();
    auto loc = op->getLoc();
    auto context = op->getContext();
    auto input = operands[0];
    auto filter = operands[1];
    auto bias_mr = operands[2];
    auto resultType = op.getType();

    auto outputType =
        getTypeConverter()->convertType(resultType).cast<MemRefType>();
    Value output = rw.create<memref::AllocOp>(loc, outputType);

    bool isNCHW = checkNCHW(op);

    const char *filterLayout = isNCHW ? "kcyxg" : "kyxcg";
    const char *inputLayout  = isNCHW ? "nchwg" : "nhwcg";
    const char *outputLayout = isNCHW ? "nkhwg" : "nhwkg";

    if (failed(makeMIOpenConv2D(rw, op, input, inputLayout, filter,
                                filterLayout, output, outputLayout, op.pad(),
                                op.stride(), op.dilation()))) {
      return failure();
    }

    rw.replaceOp(op, output);

    // test for zero bias, and ignore
    if (!isConstantZero(op.getOperand(2))) {
      // non-zero bias, replace with tosa.add w/ broadcast
      auto conv_output_t = rw.create<bufferization::ToTensorOp>(loc, output);

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

      auto bias_expand_mr = rw.create<memref::ExpandShapeOp>(
          loc, newType, bias_mr, reassociations);

      auto bias_t = rw.create<bufferization::ToTensorOp>(loc, bias_expand_mr);
      output = rw.create<tosa::AddOp>(loc, op.getType(),
                                      ValueRange{conv_output_t, bias_t});
    }
    return success();
  }
};

class MatMulConverter final : public OpConversionPattern<tosa::MatMulOp> {
public:
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                tosa::MatMulOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    // BS must equal 1
    auto operands = adaptor.getOperands();
    auto loc = op->getLoc();
    // A(BS,M,K) -> A(BS,1,M,K)
    auto A = expandMemRef(rw, op, operands[0], 1);
    const char *ALayout = "ghwcn";
    // B(BS,K,N) -> B(BS,K,N,1)
    auto B = expandMemRef(rw, op, operands[1], 3);
    const char *BLayout = "gckyx";

    // C(BS,M,N) -> C(BS,1,M,N)
    auto outputType =
        getTypeConverter()->convertType(op.getType()).cast<MemRefType>();
    Value output = rw.create<memref::AllocOp>(loc, outputType);
    auto C = expandMemRef(rw, op, output, 1);
    const char *CLayout = "ghwkn";

    auto zero = rw.getIndexAttr(0);
    auto pad = rw.getArrayAttr({zero, zero, zero, zero});
    auto one = rw.getIndexAttr(1);
    auto ones = rw.getArrayAttr({one, one});

    if (failed(makeMIOpenConv2D(rw, op, A, ALayout, B, BLayout, C, CLayout, pad,
                                ones, ones))) {
      return failure();
    }

    rw.replaceOp(op, output);

    return success();
  }
};

class TransposeConverter final : public OpConversionPattern<tosa::TransposeOp> {
public:
  using OpConversionPattern<tosa::TransposeOp>::OpConversionPattern;

  bool transposeEqualTo(SmallVector<int64_t> &given,
                        SmallVector<int64_t> &perm) const {
    for (uint32_t i = 0; i < 4; i++) {
      if (perm[i] != given[i])
        return false;
    }
    // all matches.
    return true;
  }

  // Fold transpose ops and convert convolution into changed layout.
  // case #0 : fold TP(NCHW2NHWC)+tosa.conv.NHWC+TP(NHWC2NCHW) back to
  //           miopen.conv.NCHW
  // Pattern match start from the output transpose
  LogicalResult
  matchAndRewrite(tosa::TransposeOp top, tosa::TransposeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    auto loc = top->getLoc();
    auto context = top->getContext();

    SmallVector<int64_t> NCHW2NHWC{0, 2, 3, 1};
    SmallVector<int64_t> NHWC2NCHW{0, 3, 1, 2};

    // Check if output has been transposed NHWC2NCHW
    auto bottomOp = dyn_cast<arith::ConstantOp>(operands[1].getDefiningOp());
    if (!bottomOp) {
      bottomOp = dyn_cast<arith::ConstantOp>(
          operands[1].getDefiningOp()->getOperand(0).getDefiningOp());
    }
    auto bottomAttr = bottomOp->getAttr("value").cast<DenseElementsAttr>();
    SmallVector<int64_t> bottomPerm;
    for (auto i : bottomAttr.getValues<int64_t>()) {
      bottomPerm.push_back(i);
    }
    if (!transposeEqualTo(bottomPerm, NHWC2NCHW)) {
      return rewriter.notifyMatchFailure(
          bottomOp, [&](::mlir::Diagnostic &diag) {
            diag << "Bottom transpose not matching";
          });
    }
    auto convOp = dyn_cast<tosa::Conv2DOp>(operands[0].getDefiningOp());
    if (!(convOp)) {
      return rewriter.notifyMatchFailure(convOp, [&](::mlir::Diagnostic &diag) {
        diag << "Second is not tosa::conv2d";
      });
    }

    if (!checkNCHW(convOp)) {
      return rewriter.notifyMatchFailure(convOp, [&](::mlir::Diagnostic &diag) {
        diag << "Convolution doesn't expect NHWC->NCHW conversion";
      });
    }

    auto opr0 = *convOp.getODSOperands(0).begin();
    auto opr1 = *convOp.getODSOperands(1).begin();
    auto inputTpOp = dyn_cast<tosa::TransposeOp>(opr0.getDefiningOp());
    auto filterTpOp = dyn_cast<tosa::TransposeOp>(opr1.getDefiningOp());
    if (!(inputTpOp) || !(filterTpOp)) {
      return rewriter.notifyMatchFailure(convOp, [&](::mlir::Diagnostic &diag) {
        diag << "No transpose found for input/filter";
      });
    }
    auto inputTpOpr1 = *inputTpOp.getODSOperands(1).begin();
    auto filterTpOpr1 = *filterTpOp.getODSOperands(1).begin();

    // Try to get transpose to input
    auto inTpConstOp = dyn_cast<arith::ConstantOp>(inputTpOpr1.getDefiningOp());
    if (!inTpConstOp) {
      inTpConstOp = dyn_cast<arith::ConstantOp>(
          inputTpOpr1.getDefiningOp()->getOperand(0).getDefiningOp());
    }
    SmallVector<int64_t> inTpPerm;
    auto inTpConstAttr =
        inTpConstOp->getAttr("value").cast<DenseElementsAttr>();
    for (auto i : inTpConstAttr.getValues<int64_t>()) {
      inTpPerm.push_back(i);
    }

    // Try to get transpose to filter
    auto filTpConstOp =
        dyn_cast<arith::ConstantOp>(filterTpOpr1.getDefiningOp());
    if (!filTpConstOp) {
      filTpConstOp = dyn_cast<arith::ConstantOp>(
          filterTpOpr1.getDefiningOp()->getOperand(0).getDefiningOp());
    }
    SmallVector<int64_t> filTpPerm;
    auto filTpConstAttr =
        filTpConstOp->getAttr("value").cast<DenseElementsAttr>();
    for (auto i : filTpConstAttr.getValues<int64_t>()) {
      filTpPerm.push_back(i);
    }

    // Reject if pattern doesn't match
    if (!transposeEqualTo(inTpPerm, NCHW2NHWC) ||
        !transposeEqualTo(filTpPerm, NCHW2NHWC)) {
      return rewriter.notifyMatchFailure(convOp, [&](::mlir::Diagnostic &diag) {
        diag << "No expected input/filter transpose shapes";
      });
    }

    auto input_t = inputTpOp->getOperand(0);
    auto filter_t = filterTpOp->getOperand(0);

    // fold input transpose
    inputTpOp->replaceAllUsesWith(ValueRange(input_t));
    rewriter.replaceOp(inputTpOp, {input_t});

    // fold filter transpose
    filterTpOp->replaceAllUsesWith(ValueRange(filter_t));
    rewriter.replaceOp(filterTpOp, {filter_t});

    // fold result transpose
    auto op = dyn_cast<tosa::Conv2DOp>(operands[0].getDefiningOp());

    // Reshape output dimensions
    // Note - new conv op has NCHW layout so that input/output shapes
    // in the operators chain remains valid during this conversion.
    auto results = op->getResults();
    auto convTy = results[0].getType().cast<ShapedType>();
    auto convElemTy = convTy.getElementType();
    auto convShape = convTy.getShape();
    // NHWC into NCHW
    SmallVector<int64_t> newShape{convShape[0], convShape[3], convShape[1],
                                  convShape[2]};
    Type newOutTy = RankedTensorType::get(newShape, convElemTy);

    // Construct a new Conv2DOp with a new layout
    auto cop = rewriter.create<tosa::Conv2DOp>(
        loc, newOutTy,
        ValueRange{op->getOperand(0), op->getOperand(1), op->getOperand(2)});
    cop->setAttrs(op->getAttrs());

    // Replace ((op)->(top)) with (cop)
    op->replaceAllUsesWith(cop);
    top->replaceAllUsesWith(cop);
    rewriter.replaceOp(top, {cop});
    op->erase();

    return success();
  }
};

} // namespace

void tosa::populateTosaToMIOpenConversionPatterns(
    bufferization::BufferizeTypeConverter &typeConverter, MLIRContext *context,
    RewritePatternSet &patterns) {
  patterns.insert<ConvConverter>(typeConverter, context);
  patterns.insert<MatMulConverter>(typeConverter, context);
}
void tosa::populateTosaToMIOpenTensorConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.insert<TransposeConverter>(context);
}
