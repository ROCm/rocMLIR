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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
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
    return intValue.getValue().isZero();
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

static std::tuple<StringAttr, std::optional<uint32_t>, rock::GemmFeatures>
getArchAttributes(Operation *op, Type inputType) {
  auto func = op->getParentOfType<func::FuncOp>();
  auto mod = func->getParentOfType<ModuleOp>();

  // TODO(sjw): get these from options
  StringAttr arch = StringAttr::get(op->getContext(), "");
  std::optional<uint32_t> num_cu = std::nullopt;
  std::optional<bool> xdlopsV2 = std::nullopt;

  if (auto attr = op->getAttrOfType<StringAttr>("arch"))
    arch = attr;
  else if (auto attr = func->getAttrOfType<StringAttr>("mhal.arch"))
    arch = attr;
  else if (auto attr = func->getAttrOfType<StringAttr>("arch"))
    arch = attr;
  else if (auto attr = mod->getAttrOfType<StringAttr>("mhal.arch"))
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
  rock::GemmFeatures features = archInfo.getDefaultFeatures(inputType);
  if (xdlopsV2.has_value())
    features = rock::bitEnumSet(features, rock::GemmFeatures::mfma, *xdlopsV2);

  return {arch, num_cu, features};
}

static FailureOr<rock::Conv2DOp>
makeRockConv2D(ConversionPatternRewriter &rw, Operation *op, Value input,
               StringRef inputLayout, Value filter, StringRef filterLayout,
               Value output, StringRef outputLayout,
               const DenseI64ArrayAttr &pad, const DenseI64ArrayAttr &stride,
               const DenseI64ArrayAttr &dilation) {
  Location loc = op->getLoc();

  // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
  auto inputExp = expandTensor(rw, op, input);
  auto filterExp = expandTensor(rw, op, filter);
  auto outputExp = expandTensor(rw, op, output);

  StringAttr arch;
  std::optional<uint32_t> num_cu;
  rock::GemmFeatures features;
  std::tie(arch, num_cu, features) = getArchAttributes(op, input.getType());

  ArrayRef<int64_t> pad64 = pad;
  ArrayRef<int64_t> stride64 = stride;
  ArrayRef<int64_t> dilation64 = dilation;
  SmallVector<int32_t, 4> paddingArray;
  SmallVector<int32_t, 2> strideArray;
  SmallVector<int32_t, 2> dilationArray;
  for (auto i : pad64)
    paddingArray.push_back(i);
  for (auto i : stride64)
    strideArray.push_back(i);
  for (auto i : dilation64)
    dilationArray.push_back(i);

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
        op.getPadAttr(), op.getStrideAttr(), op.getDilationAttr());
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

  Value insertBroadcast(Value inp, ArrayRef<int64_t> outShape, Location loc,
                        ConversionPatternRewriter &rw) const {
    ArrayRef<int64_t> inpShape = inp.getType().cast<ShapedType>().getShape();
    bool broadcastDone = false;
    rock::BottomUpTMBuilder broadcastDims(rw, inpShape, loc);
    for (unsigned int i = 0; i < outShape.size(); i++) {
      if (inpShape[i] == 1 && outShape[i] != 1) {
        broadcastDims.broadcast({i}, {outShape[i]});
        broadcastDone = true;
      } else {
        broadcastDims.passThrough({i}, {i});
      }
    }
    if (!broadcastDone) {
      return inp;
    }
    return rw.create<rock::TransformOp>(loc, inp, broadcastDims.get());
  }

  std::tuple<int64_t, int64_t> getLastDims(UnitAttr transposed,
                                           RankedTensorType type) const {
    ArrayRef<int64_t> shape = type.getShape();
    int64_t rank = type.getRank();
    if (transposed) {
      return {shape[rank - 1], shape[rank - 2]};
    }
    return {shape[rank - 2], shape[rank - 1]};
  }

  void setLastDims(UnitAttr transposed, SmallVectorImpl<int64_t> &shape,
                   std::pair<int64_t, int64_t> lastDims) const {
    size_t rank = shape.size();
    if (transposed) {
      shape[rank - 1] = lastDims.first;
      shape[rank - 2] = lastDims.second;
    } else {
      shape[rank - 2] = lastDims.first;
      shape[rank - 1] = lastDims.second;
    }
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
    std::optional<uint32_t> num_cu;
    rock::GemmFeatures features;
    std::tie(arch, num_cu, features) =
        getArchAttributes(op, op.getA().getType());

    auto [mDim, nDim] = getLastDims(transposeC, outputType);

    int64_t kDimOfA;
    std::tie(std::ignore, kDimOfA) =
        getLastDims(transposeA, op.getA().getType().cast<RankedTensorType>());
    int64_t kDimOfB;
    std::tie(kDimOfB, std::ignore) =
        getLastDims(transposeB, op.getB().getType().cast<RankedTensorType>());
    int kDim = (kDimOfA > kDimOfB) ? kDimOfA : kDimOfB;

    SmallVector<int64_t, 3> aShape = llvm::to_vector<3>(
        op.getA().getType().cast<RankedTensorType>().getShape());
    setLastDims(transposeA, aShape, {mDim, kDim});
    Value brA = insertBroadcast(adaptor.getA(), aShape, loc, rw);

    SmallVector<int64_t, 3> bShape = llvm::to_vector<3>(
        op.getB().getType().cast<RankedTensorType>().getShape());
    setLastDims(transposeB, bShape, {kDim, nDim});
    Value brB = insertBroadcast(adaptor.getB(), bShape, loc, rw);

    auto rockGemm = rw.create<rock::GemmOp>(
        loc, outputType, brA, brB, output, transposeA, transposeB, transposeC,
        arch,
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

  LogicalResult checkMatMulTransposeValid(tosa::MatMulOp matmulOp,
                                          ArrayRef<int32_t> dims) const {
    // batch dimension is expected to be 3rd from the last.
    if (dims.size() >= 3 && dims[dims.size() - 3] != (int32_t)dims.size() - 3) {
      return matmulOp.emitError(
          "Can't transpose the batch dimension out of place");
    }
    return success();
  }

  bool isMatMulNonTrivial(ArrayRef<int32_t> dims) const {
    int32_t lastDim = dims.size() - 1;
    int32_t prevLastDim = dims.size() - 2;
    return (dims[prevLastDim] == lastDim && dims[lastDim] == prevLastDim);
  }

  // This function traverses the uses of tOutput and then modifies
  // the uses to indicate the input are transposed and replaces them
  // with tInput. If there are collapse shapes encountered, the collapse
  // is applied on the tInput.
  LogicalResult mergeTransposeWithGemmLikeOp(PatternRewriter &rewriter,
                                             Value tOutput,
                                             ArrayRef<int32_t> dims,
                                             Value tInput) const {
    for (auto &use : tOutput.getUses()) {
      if (auto op = dyn_cast<tensor::CollapseShapeOp>(use.getOwner())) {
        SmallVector<ReassociationIndices, 4> reassocIndices =
            op.getReassociationIndices();
        ArrayRef<int64_t> inShape = op.getSrcType().getShape();

        // This loops maps reassociated dims back to pre transposed dims.
        SmallVector<int32_t, 4> newDims;
        for (ReassociationIndices indices : reassocIndices) {
          int32_t minIdx = dims[indices[0]];
          for (size_t i = 0; i < indices.size(); i++) {
            // We can ignore unit dim collapses
            if (inShape[indices[i]] == 1) {
              continue;
            }
            int32_t transposedDim = dims[indices[i]];
            if (std::abs(minIdx - transposedDim) > 1) {
              return rewriter.notifyMatchFailure(
                  op, "CollapseShape op following transpose collapses "
                      "non-contigous pre-transpose dims.");
            }
            // Where dims are collapsed, we take the minimum as a
            // representative.
            minIdx = std::min(minIdx, dims[indices[i]]);
          }
          newDims.push_back(minIdx);
        }

        // Assign the ordering index of reassociated dims as the dim index
        SmallVector<int32_t, 4> newDimsSorted = newDims;
        llvm::sort(newDimsSorted);
        DenseMap<int32_t, int32_t> dimMap;
        for (size_t i = 0; i < newDimsSorted.size(); i++) {
          dimMap[newDimsSorted[i]] = i;
        }
        for (size_t i = 0; i < newDims.size(); i++) {
          newDims[i] = dimMap[newDims[i]];
        }

        tensor::CollapseShapeOp newCollapseShapeOp =
            rewriter.create<tensor::CollapseShapeOp>(op.getLoc(), tInput,
                                                     reassocIndices);

        if (mergeTransposeWithGemmLikeOp(rewriter, op.getResult(), newDims,
                                         newCollapseShapeOp.getResult())
                .failed()) {
          rewriter.eraseOp(newCollapseShapeOp);
          return failure();
        }
        rewriter.eraseOp(op);
      } else if (auto op = dyn_cast<tensor::ExpandShapeOp>(use.getOwner())) {
        return rewriter.notifyMatchFailure(
            op, "We dont support expand shapes yet.");
      } else if (auto convOp = dyn_cast<tosa::Conv2DOp>(use.getOwner())) {
        if (convOp.getInput() == tOutput) {
          permuteLayout(convOp, "input_layout", "nhwc", dims, true);
          convOp.getInputMutable().assign(tInput);
        } else if (convOp.getWeight() == tOutput) {
          permuteLayout(convOp, "filter_layout", "kyxc", dims, true);
          convOp.getWeightMutable().assign(tInput);
        } else {
          return convOp.emitError("transpose found leading to a conv2D input "
                                  "other than data or weight");
        }
      } else if (auto matMulOp = dyn_cast<tosa::MatMulOp>(use.getOwner())) {
        if (checkMatMulTransposeValid(matMulOp, dims).failed()) {
          return failure();
        }
        bool mmNonTrivial = isMatMulNonTrivial(dims);
        if (matMulOp.getA() == tOutput) {
          setTranspose(matMulOp, "transpose_a", mmNonTrivial);
          matMulOp.getAMutable().assign(tInput);
        } else if (matMulOp.getB() == tOutput) {
          setTranspose(matMulOp, "transpose_b", mmNonTrivial);
          matMulOp.getBMutable().assign(tInput);
        } else {
          return matMulOp.emitError(
              "transpose found leading to a matmul input other than A or B");
        }
      } else {
        return failure();
      }
    }
    return success();
  }

  // Fold transpose ops and convert convolution into changed layout.
  // case #0 : fold TP(NCHW2NHWC)+tosa.conv.NHWC+TP(NHWC2NCHW) back to
  //           rock.conv.NCHW
  // Pattern match start from the output transpose
  LogicalResult matchAndRewrite(tosa::TransposeOp top,
                                PatternRewriter &b) const final {
    SmallVector<int32_t> dims;
    if (failed(getTransposeDims(top.getOperand(1), dims))) {
      return failure();
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
      if (checkMatMulTransposeValid(matMulOp, dims).failed()) {
        return failure();
      }
      setTranspose(matMulOp, "transpose_c", isMatMulNonTrivial(dims));
      matMulOp->getResult(0).setType(tOutput.getType());
      top->replaceAllUsesWith(matMulOp);
    } else {
      if (mergeTransposeWithGemmLikeOp(b, tOutput, dims, tInput).failed()) {
        return failure();
      }
    }

    b.eraseOp(top);
    return success();
  }
};

struct AttentionRewritePattern : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern<tosa::MatMulOp>::OpRewritePattern;

  FailureOr<Value> maybeSoftmaxNumerator(Value val) const {
    tosa::ExpOp exp = val.getDefiningOp<tosa::ExpOp>();
    if (!exp) {
      return failure();
    }
    tosa::SubOp sub = exp.getInput1().getDefiningOp<tosa::SubOp>();
    if (!sub) {
      return failure();
    }
    tosa::ReduceMaxOp rmax = sub.getInput2().getDefiningOp<tosa::ReduceMaxOp>();
    if (!rmax) {
      return failure();
    }
    if (rmax.getInput() != sub.getInput1()) {
      return failure();
    }
    return rmax.getInput();
  }

  FailureOr<Value> maybeSoftmaxDenominator(Value val) const {
    tosa::ReduceSumOp rsum = val.getDefiningOp<tosa::ReduceSumOp>();
    if (!rsum) {
      return failure();
    }
    return maybeSoftmaxNumerator(rsum.getInput());
  }

  FailureOr<Value> maybeSoftmax(Value val) const {
    tosa::MulOp mul = val.getDefiningOp<tosa::MulOp>();
    if (!mul) {
      return failure();
    }
    if (tosa::ReciprocalOp rec =
            mul.getInput1().getDefiningOp<tosa::ReciprocalOp>()) {
      return maybeSoftmaxDenominator(rec.getInput1());
    } else if (tosa::ReciprocalOp rec =
                   mul.getInput2().getDefiningOp<tosa::ReciprocalOp>()) {
      return maybeSoftmaxDenominator(rec.getInput1());
    } else {
      return failure();
    }
  }

  FailureOr<std::tuple<tosa::MatMulOp, TypedValue<TensorType>>>
  getMatMulAndScaleInputs(tosa::MulOp scale) const {
    if (tosa::MatMulOp mm = scale.getInput1().getDefiningOp<tosa::MatMulOp>()) {
      return std::make_tuple(mm, scale.getInput2());
    }
    if (tosa::MatMulOp mm = scale.getInput2().getDefiningOp<tosa::MatMulOp>()) {
      return std::make_tuple(mm, scale.getInput1());
    }
    return failure();
  }

  LogicalResult match(tosa::MatMulOp op) const override {
    FailureOr<Value> softmaxInput = maybeSoftmax(op.getA());
    if (failed(softmaxInput)) {
      return failure();
    }
    if (tosa::MulOp scale = softmaxInput.value().getDefiningOp<tosa::MulOp>()) {
      FailureOr<std::tuple<tosa::MatMulOp, TypedValue<TensorType>>>
          scaledMatMul = getMatMulAndScaleInputs(scale);
      if (succeeded(scaledMatMul)) {
        return success();
      }
    }
    return failure();
  }

  void rewrite(tosa::MatMulOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value softmaxInput = maybeSoftmax(op.getA()).value();
    tosa::MulOp scale = softmaxInput.getDefiningOp<tosa::MulOp>();
    auto [firstMatMulOp, scaleInput] = getMatMulAndScaleInputs(scale).value();
    auto outputType = op.getType().template cast<RankedTensorType>();
    Value output = rewriter.create<bufferization::AllocTensorOp>(
        loc, outputType, ValueRange{});
    StringAttr arch;
    std::optional<uint32_t> num_cu;
    rock::GemmFeatures features;
    std::tie(arch, num_cu, features) = getArchAttributes(op, op.getType());
    rock::AttentionOp attnOp = rewriter.create<rock::AttentionOp>(
        loc, outputType, firstMatMulOp.getA(), firstMatMulOp.getB(), op.getB(),
        scaleInput, output, rewriter.getAttr<rock::GemmFeaturesAttr>(features),
        /*blockSize=*/nullptr,
        /*gridSize=*/nullptr);
    rewriter.replaceOp(op, attnOp.getResult());
  }
};

template <typename TosaReduceOp>
typename std::enable_if_t<
    std::is_same<TosaReduceOp, tosa::ReduceSumOp>::value ||
        std::is_same<TosaReduceOp, tosa::ReduceMaxOp>::value,
    LogicalResult> static matchAndRewriteReductions(TosaReduceOp op,
                                                    rock::ReduceMethod rMethod,
                                                    Attribute outputInitVal,
                                                    ConversionPatternRewriter
                                                        &rw) {
  Location loc = op->getLoc();
  auto outputType = op.getType().template cast<RankedTensorType>();
  Value output =
      rw.create<bufferization::AllocTensorOp>(loc, outputType, ValueRange{});
  StringAttr arch;
  std::optional<uint32_t> num_cu;
  rock::GemmFeatures features;
  std::tie(arch, num_cu, features) = getArchAttributes(op, op.getType());

  int32_t blockSize = 256;
  auto elementCount =
      op.getInput().getType().template cast<ShapedType>().getNumElements();
  int32_t gridSize = (elementCount + blockSize - 1) / blockSize;
  if (num_cu.has_value()) {
    gridSize = std::min((int32_t)(20 * num_cu.value()), gridSize);
  }

  auto rockReduce = rw.create<rock::ReduceOp>(
      loc, outputType, op.getInput(), output,
      rw.getAttr<rock::GemmFeaturesAttr>(features),
      rw.getAttr<rock::ReduceMethodAttr>(rMethod),
      rw.getIndexAttr(op.getAxis()), rw.getI32IntegerAttr(blockSize),
      rw.getI32IntegerAttr(gridSize),
      /*useLDS=*/nullptr,
      /*useDPP=*/nullptr);

  func::FuncOp func = op->template getParentOfType<func::FuncOp>();
  func.setResultAttr(0, rock::PrefillAttr::getMnemonic(), outputInitVal);
  func.setResultAttr(0, func::FuncOp::getReadAccessAttrName(),
                     rw.getUnitAttr());
  // The original function also need the read access attr for the output.
  if (func->hasAttr("original_func")) {
    if (ModuleOp rootMod =
            func->getParentOfType<ModuleOp>()->getParentOfType<ModuleOp>()) {
      SymbolTable symTable(rootMod);
      SymbolRefAttr originalFuncAttr =
          func->getAttrOfType<SymbolRefAttr>("original_func");
      if (func::FuncOp originalFunc = dyn_cast<func::FuncOp>(
              symTable.lookupSymbolIn(rootMod, originalFuncAttr))) {
        originalFunc.setResultAttr(0, func::FuncOp::getReadAccessAttrName(),
                                   rw.getUnitAttr());
      }
    }
  }
  rw.replaceOp(op, rockReduce.getResult());
  return success();
}

class ReduceSumConverter final : public OpConversionPattern<tosa::ReduceSumOp> {
public:
  using OpConversionPattern<tosa::ReduceSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::ReduceSumOp op,
                                tosa::ReduceSumOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    StringAttr arch;
    std::optional<uint32_t> num_cu;
    rock::GemmFeatures features;
    std::tie(arch, num_cu, features) =
        getArchAttributes(op, op.getInput().getType());
    if (!rock::bitEnumContainsAll(features, rock::GemmFeatures::atomic_add)) {
      op.emitError("Currently, we only support ReduceSum operators on GPUs "
                   "with atomic add support.!.");
    }
    Type elementType =
        op.getInput().getType().cast<ShapedType>().getElementType();
    if (!elementType.isF32()) {
      return rw.notifyMatchFailure(op, "We only support F32 reductions, yet.");
    }
    Attribute outputInitVal = rw.getFloatAttr(elementType, 0.0000);
    return matchAndRewriteReductions(op, rock::ReduceMethod::Sum, outputInitVal,
                                     rw);
  }
};

class ReduceMaxConverter final : public OpConversionPattern<tosa::ReduceMaxOp> {
public:
  using OpConversionPattern<tosa::ReduceMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::ReduceMaxOp op,
                                tosa::ReduceMaxOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    Type elementType =
        op.getInput().getType().cast<ShapedType>().getElementType();
    Attribute outputInitVal;
    if (elementType.isF32()) {
      outputInitVal = rw.getFloatAttr(
          elementType, APFloat::getInf(APFloat::IEEEsingle(), true));
    } else {
      return rw.notifyMatchFailure(op, "We only support F32 reductions, yet.");
    }
    return matchAndRewriteReductions(op, rock::ReduceMethod::Max, outputInitVal,
                                     rw);
  }
};

} // namespace

void tosa::populateTosaToRockConversionPatterns(MLIRContext *context,
                                                RewritePatternSet &patterns) {
  patterns.add<ConvConverter, MatMulConverter, ReduceSumConverter,
               ReduceMaxConverter>(context);
}

void tosa::populateTosaToRockTensorConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<AttentionRewritePattern, TransposeRewritePattern>(context);
}
