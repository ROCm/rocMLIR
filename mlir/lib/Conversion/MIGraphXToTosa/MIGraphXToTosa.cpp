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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

template <typename TosaOp, typename... Args>
static TosaOp createOpAndInfer(PatternRewriter &rewriter, Location loc,
                               Type elemType, Args &&...args) {
  auto op =
      rewriter.create<TosaOp>(loc, UnrankedTensorType::get(elemType), args...);
  InferShapedTypeOpInterface shapeInterface =
      cast<InferShapedTypeOpInterface>(op.getOperation());
  SmallVector<ShapedTypeComponents> returnShape;
  LogicalResult shapeInferenceStatus = shapeInterface.inferReturnTypeComponents(
      op.getContext(), op.getLoc(), op->getOperands(), op->getAttrDictionary(),
      op->getPropertiesStorage(), op->getRegions(), returnShape);
  assert(shapeInferenceStatus.succeeded());
  Type newOutTy = RankedTensorType::get({returnShape[0].getDims()}, elemType);
  auto result = op->getResult(0);
  result.setType(newOutTy);
  return op;
}

static bool assignExpandedShapeVal(Operation *use, Operation *originalOp,
                                   Value maybeExpandedVal) {
  // Tosa only broadcast implicitly on the second input of the binary
  // elementwise operators. Try to swap the operands if trying to broadcast on
  // the first input.
  if (use->hasTrait<OpTrait::IsCommutative>() && use->getNumOperands() == 2) {
    if (use->getOperand(0) == originalOp->getResult(0)) {
      use->setOperand(0, use->getOperand(1));
      use->setOperand(1, maybeExpandedVal);
    } else {
      use->setOperand(1, maybeExpandedVal);
    }
    return true;
  }

  // Do the in place broadcast replacement as-is, assuming it will work.
  // If not, it will fail the Tosa lowering.
  use->replaceUsesOfWith(originalOp->getResult(0), maybeExpandedVal);
  return true;
}

static tosa::CastOp createCastOp(PatternRewriter &rewriter, Location loc,
                                 Type resElementType, Value input) {
  ShapedType inputType = input.getType().cast<ShapedType>();
  Type resType = inputType.cloneWith({}, resElementType);

  auto op = rewriter.create<tosa::CastOp>(loc, resType, input);
  return op;
}

static Type getShapedElementTy(Value v) {
  return v.getType().cast<ShapedType>().getElementType();
}

template <typename ConvType>
class ConvConverter : public OpConversionPattern<ConvType> {
public:
  using OpConversionPattern<ConvType>::OpConversionPattern;

  Value getZeroBias(Location loc, Type elemType, int64_t filterOutputChannels,
                    ConversionPatternRewriter &rewriter) const {
    auto biasTy = RankedTensorType::get({filterOutputChannels}, elemType);
    return rewriter.create<arith::ConstantOp>(loc,
                                              rewriter.getZeroAttr(biasTy));
  }

  tosa::TransposeOp
  getRank4TransposeOp(Location loc, Value input,
                      ConversionPatternRewriter &rewriter,
                      SmallVector<int64_t> &permutation) const {
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
    return newOp;
  }

  // Note, this lowering pattern works for both migraphx.convolution and
  // migraphx.quant_convolution. The only difference between the two ops
  // is that quant_convolution allows convolution input and output to be
  // different types. Because of this, we use same lowering pattern but
  // different tablegen to capture the difference between the two ops.
  LogicalResult
  matchAndRewrite(ConvType op,
                  typename OpConversionPattern<ConvType>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto input_t = op.getInput();
    auto filter_t = op.getFilter();
    auto results = op->getResults();
    auto elementTy = input_t.getType().getElementType();
    auto outputTy = results[0].getType().template cast<ShapedType>();
    SmallVector<int64_t> NCHW2NHWC{0, 2, 3, 1};
    SmallVector<int64_t> NHWC2NCHW{0, 3, 1, 2};

    // insert transpose to input and filter tensors
    input_t = cast<TypedValue<TensorType>>(
        getRank4TransposeOp(loc, input_t, rewriter, NCHW2NHWC).getResult());
    filter_t = cast<TypedValue<TensorType>>(
        getRank4TransposeOp(loc, filter_t, rewriter, NCHW2NHWC).getResult());
    auto outShape = outputTy.getShape();

    // original output shape was NCHW, change it into NHWC
    SmallVector<int64_t> newShape{outShape[0], outShape[2], outShape[3],
                                  outShape[1]};
    Type newOutTy = RankedTensorType::get(newShape, outputTy.getElementType());

    // Construct a new Conv2DOp.
    auto cop = rewriter.create<tosa::Conv2DOp>(
        loc, newOutTy,
        ValueRange{
            input_t, filter_t,
            getZeroBias(
                loc, outputTy.getElementType(),
                filter_t.getType().template cast<ShapedType>().getShape()[0],
                rewriter)});

    // translate attributes
    auto padAttr = op->getAttr("padding").template cast<ArrayAttr>();
    auto strideAttr = op->getAttr("stride").template cast<ArrayAttr>();
    auto dilationAttr = op->getAttr("dilation").template cast<ArrayAttr>();
    int64_t padTop = padAttr[0].template dyn_cast<IntegerAttr>().getInt();
    int64_t padBottom = padAttr[1].template dyn_cast<IntegerAttr>().getInt();
    int64_t padLeft = padAttr[2].template dyn_cast<IntegerAttr>().getInt();
    int64_t padRight = padAttr[3].template dyn_cast<IntegerAttr>().getInt();
    int64_t strideHeight =
        strideAttr[0].template dyn_cast<IntegerAttr>().getInt();
    int64_t strideWidth =
        strideAttr[1].template dyn_cast<IntegerAttr>().getInt();
    int64_t dilationHeight =
        dilationAttr[0].template dyn_cast<IntegerAttr>().getInt();
    int64_t dilationWidth =
        dilationAttr[1].template dyn_cast<IntegerAttr>().getInt();

    // convolution config attributes

    cop->setAttr("dilation", rewriter.getDenseI64ArrayAttr(
                                 {dilationHeight, dilationWidth}));
    cop->setAttr("stride",
                 rewriter.getDenseI64ArrayAttr({strideHeight, strideWidth}));
    cop->setAttr("pad", rewriter.getDenseI64ArrayAttr(
                            {padTop, padBottom, padLeft, padRight}));

    // Convert optional attributes
    if (auto attr = (*op).template getAttrOfType<BoolAttr>("xdlopsV2"))
      cop->setAttr("xdlopsV2", attr);
    if (auto attr = (*op).template getAttrOfType<StringAttr>("perf_config"))
      cop->setAttr("perf_config", attr);

    // Note: For TOSA convolution, a non-float type is considered as a
    // quantized convolution. For quantized convolution, it is required
    // to carry the "quantization_info" as attribute. Adding this
    // attribute help us populate the correct TOSA IR.
    //
    // When we add support to quantized types and TOSA.rescale Op, we
    // should make the quantized attribute to accept actual zero point
    // values from intput and filter.
    if (elementTy.isInteger(8)) {
      auto quantAttr = rewriter.getAttr<tosa::ConvOpQuantizationAttr>(
          /*inputZp =*/0, /*weightZp =*/0);
      cop->setAttr("quantization_info", quantAttr);
    }

    // transpose the output back to NCHW so that it can match following
    // operators.
    auto top = getRank4TransposeOp(loc, cop, rewriter, NHWC2NCHW);
    rewriter.replaceOp(op, top);
    return success();
  }
};

class BroadcastConverter final
    : public OpConversionPattern<migraphx::BroadcastOp> {
public:
  using OpConversionPattern<migraphx::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ArrayRef<int64_t> inShape = op.getInput().getType().getShape();
    uint32_t outRank = op.getOutput().getType().getRank();
    Type elemType = op.getOutput().getType().getElementType();
    int64_t axis = op->getAttr("axis").cast<IntegerAttr>().getInt();

    SmallVector<int64_t, 5> newShape;
    for (uint32_t i = 0; i < outRank; i++) {
      if (i == axis) {
        newShape.push_back(inShape[0]);
      } else {
        newShape.push_back(1);
      }
    }
    tosa::ReshapeOp sameRankReshapedOp = createOpAndInfer<tosa::ReshapeOp>(
        rewriter, loc, elemType, op.getInput(),
        rewriter.getDenseI64ArrayAttr(newShape));
    SmallVector<Operation *, 4> users =
        llvm::to_vector<4>(op.getOutput().getUsers());
    for (Operation *use : users) {
      if (!assignExpandedShapeVal(use, op, sameRankReshapedOp.getResult())) {
        return op.emitError() << use << " does not support broadcasting\n";
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class MultiBroadcastConverter final
    : public OpConversionPattern<migraphx::MultiBroadcastOp> {
public:
  using OpConversionPattern<migraphx::MultiBroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::MultiBroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ArrayRef<int64_t> inShape = op.getInput().getType().getShape();
    uint32_t inRank = op.getInput().getType().getRank();
    uint32_t outRank = op.getOutput().getType().getRank();
    Type elemType = op.getOutput().getType().getElementType();

    if (outRank < inRank) {
      return op.emitError("MultiBroadcastOp shouldn't reduce rank.\n");
    }

    Value replacingValue = op.getInput();
    if (outRank > inRank) {
      SmallVector<int64_t, 5> newShape = llvm::to_vector<5>(inShape);
      for (uint32_t i = inRank; i < outRank; i++) {
        newShape.push_back(1);
      }
      tosa::ReshapeOp sameRankReshapedOp = createOpAndInfer<tosa::ReshapeOp>(
          rewriter, loc, elemType, op.getInput(),
          rewriter.getDenseI64ArrayAttr(newShape));
      replacingValue = sameRankReshapedOp.getResult();
    }

    SmallVector<Operation *, 4> users =
        llvm::to_vector<4>(op.getOutput().getUsers());
    for (Operation *use : users) {
      if (!assignExpandedShapeVal(use, op, replacingValue) && use != op) {
        return op.emitError() << use << " does not support broadcasting\n";
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename DotType>
class DotConverter final : public OpConversionPattern<DotType> {
public:
  using OpConversionPattern<DotType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DotType op,
                  typename OpConversionPattern<DotType>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    TypedValue<TensorType> in_A = op.getInA();
    TypedValue<TensorType> in_B = op.getInB();
    auto results = op->getResults();
    auto elementTy = in_A.getType().getElementType();
    ShapedType outputTy = cast<ShapedType>(results[0].getType());

    // check batch dimension. Tosa matmul only allow a single dimension for it,
    // add reshape ops to flatten and restore the original dimension.
    ArrayRef<int64_t> orgOutDims = outputTy.getShape();
    RankedTensorType newOutType =
        RankedTensorType::get(orgOutDims, outputTy.getElementType());
    size_t outRank = orgOutDims.size();
    ArrayRef<int64_t> orgDimsA = in_A.getType().getShape();
    ArrayRef<int64_t> orgDimsB = in_B.getType().getShape();
    size_t rankA = orgDimsA.size();
    size_t rankB = orgDimsB.size();

    // A, B, Out have the same rank. rank=2 assumes batch=1.
    // Here handling special cases.
    if (outRank != 3 || rankA != rankB ||
        (outRank == 3 && orgDimsA[0] != orgDimsB[0])) {
      int64_t batchSizeA = 1, batchSizeB = 1, batchSizeC = 1;
      for (size_t i = 0; i < outRank - 2; i++) {
        batchSizeC *= orgOutDims[i];
      }
      for (size_t i = 0; i < rankA - 2; i++) {
        batchSizeA *= orgDimsA[i];
      }
      for (size_t i = 0; i < rankB - 2; i++) {
        batchSizeB *= orgDimsB[i];
      }

      int64_t newDimsA[3] = {batchSizeA, orgDimsA[outRank - 2],
                             orgDimsA[outRank - 1]};
      int64_t newDimsB[3] = {batchSizeB, orgDimsB[outRank - 2],
                             orgDimsB[outRank - 1]};
      int64_t newDimsOut[3] = {batchSizeC, orgOutDims[outRank - 2],
                               orgOutDims[outRank - 1]};
      if (batchSizeA != batchSizeB) {
        // support when batchB dimension is broadcast
        if (batchSizeB == 1) {
          // modify [g, m, k, n] to [1, g*m, k, n]
          newDimsA[0] = 1;
          newDimsA[1] *= batchSizeA;
          newDimsOut[0] = 1;
          newDimsOut[1] *= batchSizeC;
        } else {
          // currently not supporting the other case, broadcast A could be
          // supported with an additional transpose.
          return op->emitError("tosa.matmul can't broadcast input.");
        }
      }
      RankedTensorType newAType = RankedTensorType::get(newDimsA, elementTy);
      RankedTensorType newBType = RankedTensorType::get(newDimsB, elementTy);
      newOutType = RankedTensorType::get(newDimsOut, outputTy.getElementType());
      auto reshapeAOp = rewriter.create<tosa::ReshapeOp>(
          loc, newAType, in_A, rewriter.getDenseI64ArrayAttr(newDimsA));
      auto reshapeBOp = rewriter.create<tosa::ReshapeOp>(
          loc, newBType, in_B, rewriter.getDenseI64ArrayAttr(newDimsB));

      // reassign inputs.
      in_A = cast<TypedValue<TensorType>>(reshapeAOp.getResult());
      in_B = cast<TypedValue<TensorType>>(reshapeBOp.getResult());
    }
    // Construct tosa.matmul.
    auto mop = rewriter.create<tosa::MatMulOp>(loc, newOutType, in_A, in_B);

    // Convert optional attributes
    if (auto attr = (*op).template getAttrOfType<BoolAttr>("xdlopsV2"))
      mop->setAttr("xdlopsV2", attr);
    if (auto attr = (*op).template getAttrOfType<StringAttr>("perf_config"))
      mop->setAttr("perf_config", attr);

    // Note: For TOSA matmul, a non-float type is considered as a
    // quantized convolution. For quantized convolution, it is required
    // to carry the "quantization_info" as attribute. Adding this
    // attribute help us populate the correct TOSA IR.
    //
    // When we add support to quantized types and TOSA.rescale Op, we
    // should make the quantized attribute to accept actual zero point
    // values from intput and filter.
    if (elementTy.isInteger(8)) {
      auto quantAttr = rewriter.getAttr<tosa::MatMulOpQuantizationAttr>(
          /*a_zp =*/0, /*b_zp =*/0);
      mop->setAttr("quantization_info", quantAttr);
    }

    if (outRank != 3 || rankA != rankB ||
        (outRank == 3 && orgDimsA[0] != orgDimsB[0])) {
      auto rop = rewriter.create<tosa::ReshapeOp>(
          loc, outputTy, mop, rewriter.getDenseI64ArrayAttr(orgOutDims));
      rewriter.replaceOp(op, rop);
      return success();
    }
    rewriter.replaceOp(op, mop);
    return success();
  }
};

class SoftmaxConverter final : public OpConversionPattern<migraphx::SoftmaxOp> {
public:
  using OpConversionPattern<migraphx::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    auto input = operands[0];
    auto axisAttr = op->getAttr("axis").cast<IntegerAttr>();
    ShapedType inputType = input.getType().cast<ShapedType>();
    auto elementType = inputType.getElementType();
    Location loc = op->getLoc();

    auto tosaMax = createOpAndInfer<tosa::ReduceMaxOp>(
        rewriter, loc, elementType, input, axisAttr);
    auto tosaSub = createOpAndInfer<tosa::SubOp>(rewriter, loc, elementType,
                                                 input, tosaMax);
    auto tosaExp =
        createOpAndInfer<tosa::ExpOp>(rewriter, loc, elementType, tosaSub);
    auto tosaReduceSum = createOpAndInfer<tosa::ReduceSumOp>(
        rewriter, loc, elementType, tosaExp, axisAttr);
    auto tosaReciprocal = createOpAndInfer<tosa::ReciprocalOp>(
        rewriter, loc, elementType, tosaReduceSum);
    auto tosaMul = createOpAndInfer<tosa::MulOp>(
        rewriter, loc, elementType, tosaExp, tosaReciprocal, /*shift=*/0);

    rewriter.replaceOp(op, tosaMul);
    return success();
  }
};

class ReshapeConverter final : public OpConversionPattern<migraphx::ReshapeOp> {
public:
  using OpConversionPattern<migraphx::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ArrayAttr dims = adaptor.getDims();
    Location loc = op->getLoc();
    auto input = adaptor.getInput();
    auto results = op->getResults();
    ShapedType outputTy = results[0].getType().cast<ShapedType>();
    SmallVector<int64_t, 5> newShape;
    for (auto dim : dims) {
      newShape.push_back(dim.dyn_cast<IntegerAttr>().getInt());
    }

    auto rop = rewriter.create<tosa::ReshapeOp>(
        loc, outputTy, input, rewriter.getDenseI64ArrayAttr(newShape));

    rewriter.replaceOp(op, rop);
    return success();
  }
};

class SliceConverter final : public OpConversionPattern<migraphx::SliceOp> {
public:
  using OpConversionPattern<migraphx::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    SmallVector<int64_t, 5> start;
    SmallVector<int64_t, 5> size;
    ArrayAttr axes = op.getAxes();
    ArrayAttr axesStarts = op.getStarts();
    ArrayAttr axesEnds = op.getEnds();

    ArrayRef<int64_t> inShape = op.getInput().getType().getShape();
    for (size_t i = 0; i < inShape.size(); i++) {
      start.push_back(0);
      size.push_back(inShape[i]);
    }

    for (auto [axis, axisS, axisE] : llvm::zip(axes, axesStarts, axesEnds)) {
      int64_t axisInt = axis.cast<IntegerAttr>().getInt();
      int64_t axisSInt = axisS.cast<IntegerAttr>().getInt();
      int64_t axisEInt = axisE.cast<IntegerAttr>().getInt();
      start[axisInt] = axisSInt;
      size[axisInt] = axisEInt - axisSInt;
    }

    auto sliceOp = createOpAndInfer<tosa::SliceOp>(
        rewriter, loc, op.getInput().getType().getElementType(), op.getInput(),
        rewriter.getDenseI64ArrayAttr(start),
        rewriter.getDenseI64ArrayAttr(size));
    rewriter.replaceOp(op, sliceOp);
    return success();
  }
};

class ReduceMeanConverter final
    : public OpConversionPattern<migraphx::ReduceMeanOp> {
public:
  using OpConversionPattern<migraphx::ReduceMeanOp>::OpConversionPattern;

  tosa::ConstOp
  createNumElementsTosaConst(Location loc, TypedValue<TensorType> inputTensor,
                             IntegerAttr axisAttr,
                             ConversionPatternRewriter &rewriter) const {
    Type elementType = inputTensor.getType().getElementType();
    int64_t axis = axisAttr.getValue().getSExtValue();
    Attribute numElements;
    if (elementType.isIntOrIndex()) {
      numElements = rewriter.getIntegerAttr(
          elementType, inputTensor.getType().getShape()[axis]);
    } else {
      numElements = rewriter.getFloatAttr(
          elementType,
          (static_cast<double>(inputTensor.getType().getShape()[axis])));
    }
    RankedTensorType tensorType = RankedTensorType::get({1}, elementType);
    return rewriter.create<tosa::ConstOp>(
        loc, tensorType, DenseElementsAttr::get(tensorType, {numElements}));
  }

  LogicalResult
  matchAndRewrite(migraphx::ReduceMeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ArrayRef<Attribute> axes = op.getAxes().getValue();
    if (axes.size() != 1) {
      return op.emitError("We only support single axes reductions!");
    }
    IntegerAttr axis = axes[0].cast<IntegerAttr>();
    Type elementType = op.getInput().getType().getElementType();

    tosa::ConstOp tosaConstantNumElements = createNumElementsTosaConst(
        loc, cast<TypedValue<TensorType>>(op.getInput()), axis, rewriter);
    auto tosaReciprocal = createOpAndInfer<tosa::ReciprocalOp>(
        rewriter, loc, elementType, tosaConstantNumElements);
    auto tosaMul = createOpAndInfer<tosa::MulOp>(
        rewriter, loc, elementType, op.getInput(), tosaReciprocal, /*shift=*/0);
    auto tosaReduceSum = createOpAndInfer<tosa::ReduceSumOp>(
        rewriter, loc, elementType, tosaMul, axis);
    rewriter.replaceOp(op, tosaReduceSum);
    return success();
  }
};

class QuantizeLinearConverter final
    : public OpConversionPattern<migraphx::QuantizeLinearOp> {
public:
  using OpConversionPattern<migraphx::QuantizeLinearOp>::OpConversionPattern;

  // MIGraphX pseudo code:
  // int64_t quantized = static_cast<int32>(
  //      std::round(input[i] / scales[i])) + zero_pts[i];
  // output[i] = std::max(-128, std::min(127, quantized));
  LogicalResult
  matchAndRewrite(migraphx::QuantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input = op.getInput();
    Value scale = op.getScale();
    Value output = op.getOutput();
    Location loc = op->getLoc();

    Type elementType = getShapedElementTy(input);
    Value inverseScale =
        createOpAndInfer<tosa::ReciprocalOp>(rewriter, loc, elementType, scale);
    if (getShapedElementTy(inverseScale) != getShapedElementTy(input)) {
      inverseScale = createCastOp(rewriter, loc, elementType, inverseScale);
    }
    Value scaled = createOpAndInfer<tosa::MulOp>(
        rewriter, loc, elementType, input, inverseScale, /*shift=*/0);

    Value shifted = scaled;
    if (auto bias = op.getBias()) {
      Value biasCast = createCastOp(rewriter, loc, elementType, bias);
      shifted = createOpAndInfer<tosa::AddOp>(rewriter, loc, elementType,
                                              scaled, biasCast);
    }

    Type outputType = getShapedElementTy(output);
    Value downCast = createCastOp(rewriter, loc, outputType, shifted);
    rewriter.replaceOp(op, downCast);

    return success();
  }
};

class DeQuantizeLinearConverter final
    : public OpConversionPattern<migraphx::DeQuantizeLinearOp> {
public:
  using OpConversionPattern<migraphx::DeQuantizeLinearOp>::OpConversionPattern;

  // MIGraphX pseudo code:
  // output[i] = static_cast<fp32>(input[i] - zero_pts[i]) * scales[i];
  LogicalResult
  matchAndRewrite(migraphx::DeQuantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input = op.getInput();
    Value scale = op.getScale();
    Value output = op.getOutput();
    Location loc = op->getLoc();

    Value shifted = input;
    if (auto bias = op.getBias()) {
      Type elementType = getShapedElementTy(input);
      shifted = createOpAndInfer<tosa::SubOp>(rewriter, loc, elementType, input,
                                              bias);
    }

    Type outputType = getShapedElementTy(output);
    Value upCast = createCastOp(rewriter, loc, outputType, shifted);

    if (getShapedElementTy(scale) != getShapedElementTy(output)) {
      scale = createCastOp(rewriter, loc, outputType, scale);
    }
    Value scaled = createOpAndInfer<tosa::MulOp>(rewriter, loc, outputType,
                                                 upCast, scale, /*shift=*/0);

    rewriter.replaceOp(op, scaled);
    return success();
  }
};

class DivConverter final : public OpConversionPattern<migraphx::DivOp> {
public:
  using OpConversionPattern<migraphx::DivOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(migraphx::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto inATensor = cast<TypedValue<TensorType>>(op.getInA());
    auto inBTensor = cast<TypedValue<TensorType>>(op.getInB());
    Type elementType = inATensor.getType().getElementType();
    if (isa<IntegerType>(elementType)) {
      Value div = createOpAndInfer<tosa::DivOp>(rewriter, loc, elementType,
                                                inATensor, inBTensor);
      rewriter.replaceOp(op, div);
      return success();
    }
    Value recip = createOpAndInfer<tosa::ReciprocalOp>(rewriter, loc,
                                                       elementType, inBTensor);
    Value mul = createOpAndInfer<tosa::MulOp>(rewriter, loc, elementType,
                                              inATensor, recip, /*shift=*/0);
    rewriter.replaceOp(op, mul);
    return success();
  }
};
} // namespace

void migraphx::populateMIGraphXToTosaConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<ConvConverter<ConvolutionOp>, ConvConverter<QuantConvolutionOp>,
               BroadcastConverter, MultiBroadcastConverter, ReshapeConverter,
               SoftmaxConverter, DotConverter<DotOp>, DotConverter<QuantDotOp>,
               ReduceMeanConverter, QuantizeLinearConverter,
               DeQuantizeLinearConverter, SliceConverter, DivConverter>(
      context);
}
