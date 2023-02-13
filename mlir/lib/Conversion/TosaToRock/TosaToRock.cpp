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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/Generator/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
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

static Value expandTensor(ConversionPatternRewriter &rw, Operation *op,
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

static std::tuple<StringAttr, Optional<uint32_t>, rock::GemmFeatures>
getArchAttributes(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  auto mod = func->getParentOfType<ModuleOp>();

  // TODO(sjw): get these from options
  StringAttr arch = StringAttr::get(op->getContext(), "");
  Optional<uint32_t> num_cu = None;
  Optional<bool> xdlopsV2 = None;

  if (auto attr = op->getAttrOfType<StringAttr>("arch"))
    arch = attr;
  else if (auto attr = func->getAttrOfType<StringAttr>("xmodel.arch"))
    arch = attr;
  else if (auto attr = func->getAttrOfType<StringAttr>("arch"))
    arch = attr;
  else if (auto attr = mod->getAttrOfType<StringAttr>("xmodel.arch"))
    arch = attr;

  if (auto attr = op->getAttrOfType<IntegerAttr>("num_cu"))
    num_cu = attr.getValue().getZExtValue();
  else if (auto attr = func->getAttrOfType<IntegerAttr>("num_cu"))
    num_cu = attr.getValue().getZExtValue();

  if (auto attr = op->getAttrOfType<BoolAttr>("xdlopsV2"))
    xdlopsV2 = attr.getValue();
  else if (auto attr = func->getAttrOfType<BoolAttr>("xdlopsV2"))
    xdlopsV2 = attr.getValue();

  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  rock::GemmFeatures features = archInfo.defaultFeatures;
  if (xdlopsV2.has_value())
    features = rock::bitEnumSet(features, rock::GemmFeatures::mfma, *xdlopsV2);

  return {arch, num_cu, features};
}

static FailureOr<rock::Conv2DOp>
makeRockConv2D(ConversionPatternRewriter &rw, Operation *op, Value input,
               StringRef inputLayout, Value filter, StringRef filterLayout,
               Value output, StringRef outputLayout, const ArrayAttr &pad,
               const ArrayAttr &stride, const ArrayAttr &dilation) {
  Location loc = op->getLoc();

  // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
  auto inputExp = expandTensor(rw, op, input);
  auto filterExp = expandTensor(rw, op, filter);
  auto outputExp = expandTensor(rw, op, output);

  StringAttr arch;
  Optional<uint32_t> num_cu;
  rock::GemmFeatures features;

  std::tie(arch, num_cu, features) = getArchAttributes(op);

  // translate attributes
  int32_t padTop = pad[0].dyn_cast<IntegerAttr>().getInt();
  int32_t padBottom = pad[1].dyn_cast<IntegerAttr>().getInt();
  int32_t padLeft = pad[2].dyn_cast<IntegerAttr>().getInt();
  int32_t padRight = pad[3].dyn_cast<IntegerAttr>().getInt();
  int32_t strideHeight = stride[0].dyn_cast<IntegerAttr>().getInt();
  int32_t strideWidth = stride[1].dyn_cast<IntegerAttr>().getInt();
  int32_t dilationHeight = dilation[0].dyn_cast<IntegerAttr>().getInt();
  int32_t dilationWidth = dilation[1].dyn_cast<IntegerAttr>().getInt();

  SmallVector<int32_t, 4> paddingArray{padTop, padBottom, padLeft, padRight};
  SmallVector<int32_t, 2> strideArray{strideHeight, strideWidth};
  SmallVector<int32_t, 2> dilationArray{dilationHeight, dilationWidth};

  auto cop = rw.create<rock::Conv2DOp>(
      loc, outputExp.getType(), filterExp, inputExp, outputExp, arch,
      rw.getAttr<rock::GemmFeaturesAttr>(features),
      /*blockSize=*/nullptr, /*gridSize=*/nullptr,
      rw.getI32ArrayAttr(paddingArray), rw.getI32ArrayAttr(strideArray),
      rw.getI32ArrayAttr(dilationArray),
      /*params=*/nullptr,
      num_cu.has_value() ? rw.getI32IntegerAttr(num_cu.value()) : nullptr);

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

  return cop;
}

class ConvConverter final : public OpConversionPattern<tosa::Conv2DOp> {
public:
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp op,
                                tosa::Conv2DOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    auto operands = adaptor.getOperands();
    auto loc = op->getLoc();
    auto *context = op->getContext();
    auto input = operands[0];
    auto filter = operands[1];
    auto bias = operands[2];
    auto outputType = op.getType().cast<RankedTensorType>();

    Value output =
        rw.create<bufferization::AllocTensorOp>(loc, outputType, ValueRange{});

    SmallString<5> filterLayout("kyxcg");
    if (auto attr = op->getAttrOfType<StringAttr>("filter_layout"))
      filterLayout = Twine(attr.getValue() + "g").str();
    SmallString<5> inputLayout("nhwcg");
    if (auto attr = op->getAttrOfType<StringAttr>("input_layout"))
      inputLayout = Twine(attr.getValue() + "g").str();
    SmallString<5> outputLayout("nhwkg");
    if (auto attr = op->getAttrOfType<StringAttr>("output_layout"))
      outputLayout = Twine(attr.getValue() + "g").str();

    FailureOr<rock::Conv2DOp> rockConv = makeRockConv2D(
        rw, op, input, inputLayout, filter, filterLayout, output, outputLayout,
        op.getPad(), op.getStride(), op.getDilation());
    if (failed(rockConv))
      return failure();

    Value result = rw.create<rock::TensorUntransformCastOp>(
        loc, outputType, rockConv->getResult(), rockConv->getOutput());
    // test for zero bias, and ignore
    if (!isConstantZero(op.getOperand(2))) {
      // non-zero bias, replace with tosa.add w/ broadcast
      auto biasType = bias.getType().template cast<ShapedType>();
      if (!biasType.hasStaticShape())
        return failure();

      SmallVector<int64_t, 4> biasShape{1, 1, 1};
      biasShape.push_back(biasType.getShape()[0]);
      auto newType =
          RankedTensorType::get(biasShape, biasType.getElementType());

      SmallVector<ReassociationExprs, 1> reassociations;

      // [[0, 1, 2, 3]]
      reassociations.push_back(
          {getAffineDimExpr(0, context), getAffineDimExpr(1, context),
           getAffineDimExpr(2, context), getAffineDimExpr(3, context)});

      auto biasExpand =
          rw.create<tensor::ExpandShapeOp>(loc, newType, bias, reassociations);

      result = rw.create<tosa::AddOp>(loc, op.getType(),
                                      ValueRange{result, biasExpand});
    }

    rw.replaceOp(op, result);

    return success();
  }
};

class MatMulConverter final : public OpConversionPattern<tosa::MatMulOp> {
public:
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

  UnitAttr getTranspose(tosa::MatMulOp op, StringRef name) const {
    if (auto attr = op->getAttrOfType<BoolAttr>(name)) {
      if (attr.getValue())
        return UnitAttr::get(op->getContext());
    }
    return nullptr;
  }

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                tosa::MatMulOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    Location loc = op->getLoc();
    auto outputType = op.getType().cast<RankedTensorType>();
    Value output =
        rw.create<bufferization::AllocTensorOp>(loc, outputType, ValueRange{});

    UnitAttr transposeA = getTranspose(op, "transpose_a"),
             transposeB = getTranspose(op, "transpose_b"),
             transposeC = getTranspose(op, "transpose_c");

    StringAttr arch;
    Optional<uint32_t> num_cu;
    rock::GemmFeatures features;
    std::tie(arch, num_cu, features) = getArchAttributes(op);

    auto rockGemm = rw.create<rock::GemmOp>(
        loc, outputType, adaptor.getA(), adaptor.getB(), output, transposeA,
        transposeB, transposeC, arch,
        num_cu.has_value() ? rw.getI32IntegerAttr(num_cu.value()) : nullptr,
        rw.getAttr<rock::GemmFeaturesAttr>(features),
        rw.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
        /*blockSize=*/nullptr, /*gridSize=*/nullptr, /*params=*/nullptr);

    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      rockGemm->setAttr("perf_config", attr);

    rw.replaceOp(op, rockGemm.getResult());

    return success();
  }
};

struct TransposeRewritePattern : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern<tosa::TransposeOp>::OpRewritePattern;

  LogicalResult getTransposeDims(Value v, SmallVector<int32_t> &perms) const {
    Operation *cval = v.getDefiningOp();
    if (isa<arith::ConstantOp>(cval) || isa<tosa::ConstOp>(cval)) {
      auto cattr = cval->getAttr("value").cast<DenseElementsAttr>();
      auto vals = cattr.tryGetValues<int32_t>();
      if (succeeded(vals)) {
        perms.assign((*vals).begin(), (*vals).end());
        return success();
      }
      auto vals64 = cattr.tryGetValues<int64_t>();
      if (succeeded(vals64)) {
        perms.assign((*vals64).begin(), (*vals64).end());
        return success();
      }
    }
    return failure();
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

  void setTranspose(Operation *op, StringRef name, bool isNonTrivial) const {
    bool currentValue = false;
    if (auto attr = op->getAttrOfType<BoolAttr>(name))
      currentValue = attr.getValue();
    bool newValue = currentValue ^ isNonTrivial;
    op->setAttr(name, BoolAttr::get(op->getContext(), newValue));
  }

  // Fold transpose ops and convert convolution into changed layout.
  // case #0 : fold TP(NCHW2NHWC)+tosa.conv.NHWC+TP(NHWC2NCHW) back to
  //           rock.conv.NCHW
  // Pattern match start from the output transpose
  LogicalResult matchAndRewrite(tosa::TransposeOp top,
                                PatternRewriter &b) const final {
    SmallVector<int32_t> dims;
    if (failed(getTransposeDims(top.getOperand(1), dims)))
      return failure();

    bool isConvDims = dims.size() == 4;
    bool isMatmulDims = dims.size() == 3;
    if (!(isConvDims || isMatmulDims)) {
      return b.notifyMatchFailure(top, [&](::mlir::Diagnostic &diag) {
        diag << "Bad constant transpose dims";
      });
    }
    bool matmulNonTrivial = false;
    if (isMatmulDims) {
      if (dims[0] != 0) {
        return b.notifyMatchFailure(top, [&](Diagnostic &diag) {
          diag << "Can't transpose the batch dimension out of place";
        });
      }
      matmulNonTrivial = (dims[1] == 2 && dims[2] == 1);
    }

    Value tInput = top.getOperand(0);
    Value tOutput = top.getResult();

    if (tosa::Conv2DOp convOp = tInput.getDefiningOp<tosa::Conv2DOp>()) {
      // tosa.conv2d output is transpose
      permuteLayout(convOp, "output_layout", "nhwk", dims);
      convOp->getResult(0).setType(tOutput.getType());
      top->replaceAllUsesWith(convOp);
    } else if (tosa::MatMulOp matMulOp =
                   tInput.getDefiningOp<tosa::MatMulOp>()) {
      setTranspose(matMulOp, "transpose_c", matmulNonTrivial);
      matMulOp->getResult(0).setType(tOutput.getType());
      top->replaceAllUsesWith(matMulOp);
    } else {
      // trace output to tosa.conv2d
      for (auto &use : tOutput.getUses()) {
        if (auto op = dyn_cast<tosa::Conv2DOp>(use.getOwner())) {
          if (convOp || matMulOp)
            return failure();
          convOp = op;
        } else if (auto op = dyn_cast<tosa::MatMulOp>(use.getOwner())) {
          if (convOp || matMulOp)
            return failure();
          matMulOp = op;
        } else {
          return failure();
        }
      }

      // conv Input Modifier
      if (convOp && convOp.getOperand(0) == tOutput) {
        // input feature map
        permuteLayout(convOp, "input_layout", "nhwc", dims, true);
        top.replaceAllUsesWith({tInput});
      } else if (convOp) {
        // filter
        assert(convOp.getOperand(1) == tOutput);
        permuteLayout(convOp, "filter_layout", "kyxc", dims, true);
        top.replaceAllUsesWith({tInput});
      } else if (matMulOp && matMulOp.getA() == tOutput) {
        setTranspose(matMulOp, "transpose_a", matmulNonTrivial);
        top.replaceAllUsesWith({tInput});
      } else if (matMulOp) {
        assert(matMulOp.getB() == tOutput);
        setTranspose(matMulOp, "transpose_b", matmulNonTrivial);
        top.replaceAllUsesWith({tInput});
      }
    }

    top.erase();
    return success();
  }
};

} // namespace

void tosa::populateTosaToRockConversionPatterns(MLIRContext *context,
                                                RewritePatternSet &patterns) {
  patterns.add<ConvConverter, MatMulConverter>(context);
}

void tosa::populateTosaToRockTensorConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<TransposeRewritePattern>(context);
}
