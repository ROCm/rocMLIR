//===- TosaToRock.cpp - Lowering Tosa to Rock Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Rock dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToRock/TosaToRock.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

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
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    return isZeroAttribute(cst.getValue());
  if (auto cst = v.getDefiningOp<tosa::ConstOp>())
    return isZeroAttribute(cst->getAttr("value"));
  return false;
}

static Value expandMemRef(ConversionPatternRewriter &rw, Operation *op,
                          Value operand, uint32_t idx = 4) {
  auto loc = op->getLoc();
  auto oprType = operand.getType().template cast<ShapedType>();
  if (!oprType.hasStaticShape()) {
    (void)rw.notifyMatchFailure(
        op, "tosa to rock conversion expects statically shaped tensors");
    return Value();
  }
  ArrayRef<int64_t> shape = oprType.getShape();

  SmallVector<uint32_t, 8> endDims;
  SmallVector<uint32_t, 8> startDims;
  for (uint32_t i = 0, e = shape.size(); i < e; ++i) {
    startDims.push_back(i);
    endDims.push_back(i < idx ? i : i + 1);
  }
  rock::BottomUpTMBuilder transform(rw, shape, loc);
  transform.passThrough(endDims, startDims);
  transform.addDim("g", idx, 1);

  return rw.create<rock::TransformOp>(loc, operand, transform.get());
}

static LogicalResult
makeRockConv2D(ConversionPatternRewriter &rw, Operation *op, Value input,
                 StringRef inputLayout, Value filter, StringRef filterLayout,
                 Value output, StringRef outputLayout, const ArrayAttr &pad,
                 const ArrayAttr &stride, const ArrayAttr &dilation) {
  auto loc = op->getLoc();
  auto func = op->getParentOfType<func::FuncOp>();
  auto mod = func->getParentOfType<ModuleOp>();

  // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
  auto inputExp = expandMemRef(rw, op, input);
  auto filterExp = expandMemRef(rw, op, filter);
  auto outputExp = expandMemRef(rw, op, output);

  // TODO(sjw): get these from options
  StringRef chip = "";
  uint32_t num_cu = 64;
  bool xdlopsV2 = false;

  if (auto attr = op->getAttrOfType<StringAttr>("arch"))
    chip = attr.getValue();
  else if (auto attr = func->getAttrOfType<StringAttr>("arch"))
    chip = attr.getValue();
  else if (auto attr = mod->getAttrOfType<StringAttr>("kernel.chip"))
    chip = attr.getValue();

  if (auto attr = op->getAttrOfType<IntegerAttr>("num_cu"))
    num_cu = attr.getValue().getZExtValue();
  else if (auto attr = func->getAttrOfType<IntegerAttr>("num_cu"))
    num_cu = attr.getValue().getZExtValue();

  if (auto attr = op->getAttrOfType<BoolAttr>("xdlopsV2"))
    xdlopsV2 = attr.getValue();
  else if (auto attr = func->getAttrOfType<BoolAttr>("xdlopsV2"))
    xdlopsV2 = attr.getValue();

  rock::GemmFeatures features = rock::GemmFeatures::none;
  if (xdlopsV2)
    features = features | rock::GemmFeatures::xdlops;
  auto cop = rw.create<rock::Conv2DOp>(
      loc, filterExp, inputExp, outputExp, rw.getStringAttr(chip),
      rw.getI32IntegerAttr(num_cu),
      rw.getAttr<rock::GemmFeaturesAttr>(features),
      /*blockSize=*/nullptr, /*gridSize=*/nullptr, /*params=*/nullptr);
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
    filterLayoutSpec.push_back(rw.getStringAttr(filterLayout.substr(i, 1)));
    inputLayoutSpec.push_back(rw.getStringAttr(inputLayout.substr(i, 1) + "i"));
    outputLayoutSpec.push_back(
        rw.getStringAttr(outputLayout.substr(i, 1) + "o"));
  }

  // arch-specific attributes
  // TODO: remove these
  if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
    cop->setAttr("perf_config", attr);

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

    SmallString<5> filterLayout("kyxcg");
    if (auto attr = op->getAttrOfType<StringAttr>("filter_layout"))
      filterLayout = Twine(attr.getValue() + "g").str();
    SmallString<5> inputLayout("nhwcg");
    if (auto attr = op->getAttrOfType<StringAttr>("input_layout"))
      inputLayout = Twine(attr.getValue() + "g").str();
    SmallString<5> outputLayout("nhwkg");
    if (auto attr = op->getAttrOfType<StringAttr>("output_layout"))
      outputLayout = Twine(attr.getValue() + "g").str();

    if (failed(makeRockConv2D(rw, op, input, inputLayout, filter,
                                filterLayout, output, outputLayout, op.pad(),
                                op.stride(), op.dilation()))) {
      return failure();
    }

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

    rw.replaceOp(op, output);

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

    if (failed(makeRockConv2D(rw, op, A, ALayout, B, BLayout, C, CLayout, pad,
                                ones, ones))) {
      return failure();
    }

    rw.replaceOp(op, output);

    return success();
  }
};

struct TransposeRewritePattern : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern<tosa::TransposeOp>::OpRewritePattern;

  SmallVector<int32_t> getTransposeDims(Value v) const {
    if (Operation *cval = v.getDefiningOp<arith::ConstantOp>()) {
      auto cattr = cval->getAttr("value").cast<DenseElementsAttr>();
      return SmallVector<int32_t>(cattr.getValues<int64_t>());
    }
    if (Operation *cval = v.getDefiningOp<tosa::ConstOp>()) {
      auto cattr = cval->getAttr("value").cast<DenseElementsAttr>();
      return SmallVector<int32_t>(cattr.getValues<int32_t>());
    }
    // May be bufferization cast
    //  but this is no longer a bufferization pass, so assert
    assert(0);
    return getTransposeDims(v.getDefiningOp()->getOperand(0));
  }

  void permuteLayout(Operation *op, const char *attrKey,
                     const char *layoutDefault, ArrayRef<int32_t> permDims,
                     bool isInput = false) const {
    StringRef currentLayout(layoutDefault);
    if (auto attr = op->getAttrOfType<StringAttr>(attrKey))
      currentLayout = attr.getValue();
    SmallString<4> layout(currentLayout);
    if (isInput) {
      for (int i = 0, e = permDims.size(); i < e; ++i)
        layout[permDims[i]] = currentLayout[i];
    } else {
      for (int i = 0, e = permDims.size(); i < e; ++i)
        layout[i] = currentLayout[permDims[i]];
    }
    op->setAttr(attrKey, StringAttr::get(op->getContext(), layout));
  }

  // Fold transpose ops and convert convolution into changed layout.
  // case #0 : fold TP(NCHW2NHWC)+tosa.conv.NHWC+TP(NHWC2NCHW) back to
  //           rock.conv.NCHW
  // Pattern match start from the output transpose
  LogicalResult matchAndRewrite(tosa::TransposeOp top,
                                PatternRewriter &b) const final {
    auto dims = getTransposeDims(top.getOperand(1));

    if (dims.size() != 4) {
      return b.notifyMatchFailure(top, [&](::mlir::Diagnostic &diag) {
        diag << "Bad constant transpose dims";
      });
    }

    Value tInput = top.getOperand(0);
    Value tOutput = top.getResult();

    if (tosa::Conv2DOp convOp = tInput.getDefiningOp<tosa::Conv2DOp>()) {
      // tosa.conv2d output is transpose
      permuteLayout(convOp, "output_layout", "nhwk", dims);
      convOp->getResult(0).setType(tOutput.getType());
      top->replaceAllUsesWith(convOp);
    } else {
      // trace output to tosa.conv2d
      for (auto &use : tOutput.getUses()) {
        if (auto op = dyn_cast<tosa::Conv2DOp>(use.getOwner())) {
          if (convOp)
            return failure();
          convOp = op;
        } else {
          return failure();
        }
      }

      // conv Input Modifier
      if (convOp.getOperand(0) == tOutput) {
        // input feature map
        permuteLayout(convOp, "input_layout", "nhwc", dims, true);
        top.replaceAllUsesWith({tInput});
      } else {
        // filter
        assert(convOp.getOperand(1) == tOutput);
        permuteLayout(convOp, "filter_layout", "kyxc", dims, true);
        top.replaceAllUsesWith({tInput});
      }
    }

    top.erase();
    return success();
  }
};

} // namespace

void tosa::populateTosaToRockConversionPatterns(
    bufferization::BufferizeTypeConverter &typeConverter, MLIRContext *context,
    RewritePatternSet &patterns) {
  patterns.insert<ConvConverter>(typeConverter, context);
  patterns.insert<MatMulConverter>(typeConverter, context);
}
void tosa::populateTosaToRockTensorConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.insert<TransposeRewritePattern>(context);
}
