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
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using mlir::migraphx::MIXRShapedType;

#define DEBUG_TYPE "migraphx-to-tosa"

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

migraphx::MIXRShapedToTensorConverter::MIXRShapedToTensorConverter() {
  addConversion([](Type type) { return type; });
  addConversion([](MIXRShapedType shaped) { return shaped.asTensor(); });

  addSourceMaterialization([](OpBuilder &b, MIXRShapedType shapedResType,
                              ValueRange tensorResult,
                              Location loc) -> std::optional<Value> {
    if (tensorResult.size() != 1)
      return Value(); // 1-1 conversions only.
    return b.create<migraphx::AsUnderlyingShapeOp>(loc, shapedResType,
                                                   tensorResult[0]);
  });

  addTargetMaterialization([](OpBuilder &b, Type wantedInputType,
                              ValueRange shapedInput,
                              Location loc) -> std::optional<Value> {
    if (shapedInput.size() != 1)
      return Value(); // 1-1 conversions only.
    return b.create<migraphx::AsLogicalShapeOp>(loc, wantedInputType,
                                                shapedInput[0]);
  });
}

migraphx::MIXRShapedToMemoryLayoutConverter::
    MIXRShapedToMemoryLayoutConverter() {
  addConversion([](Type type) { return type; });
  addConversion(
      [](MIXRShapedType shaped) { return shaped.asFlatMemoryTensor(); });
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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

static tosa::CastOp createCastOp(PatternRewriter &rewriter, Location loc,
                                 Type resElementType, Value input) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  Type resType = inputType.cloneWith({}, resElementType);

  auto op = rewriter.create<tosa::CastOp>(loc, resType, input);
  return op;
}

static Type getShapedElementTy(Value v) {
  return cast<ShapedType>(v.getType()).getElementType();
}

static Value getZeroTensor(Location loc, RankedTensorType type,
                           ConversionPatternRewriter &rewriter) {
  auto value = cast<ElementsAttr>(rewriter.getZeroAttr(type));
  return rewriter.create<tosa::ConstOp>(loc, type, value);
}

static Value getZeroTensor(Location loc, Type elemType, ArrayRef<int64_t> shape,
                           ConversionPatternRewriter &rewriter) {
  auto tensorTy = RankedTensorType::get(shape, elemType);
  return rewriter.create<tosa::ConstOp>(
      loc, tensorTy, cast<ElementsAttr>(rewriter.getZeroAttr(tensorTy)));
}

static tosa::TransposeOp getTransposeOp(Location loc, Value input,
                                        ConversionPatternRewriter &rewriter,
                                        ArrayRef<int64_t> permutation) {
  int64_t len = permutation.size();

  auto permutationAttrType =
      RankedTensorType::get({len}, rewriter.getI64Type());
  auto permutationAttr =
      DenseIntElementsAttr::get(permutationAttrType, permutation);
  Value permutationValue =
      rewriter.create<tosa::ConstOp>(loc, permutationAttrType, permutationAttr);
  ShapedType inputTy = cast<ShapedType>(input.getType());
  auto inputShape = inputTy.getShape();
  SmallVector<int64_t> newShape;
  newShape.reserve(len);
  for (int64_t fromIdx : permutation)
    newShape.push_back(inputShape[fromIdx]);
  Type newTy = RankedTensorType::get(newShape, inputTy.getElementType());

  auto newOp =
      rewriter.create<tosa::TransposeOp>(loc, newTy, input, permutationValue);
  return newOp;
}

//===----------------------------------------------------------------------===//
// The general one-to-one conversion and
//===----------------------------------------------------------------------===//

namespace {
template <typename MIGraphXOp, typename TosaOp>
struct TrivialConverter final : public OpConversionPattern<MIGraphXOp> {
  using OpConversionPattern<MIGraphXOp>::OpConversionPattern;
  using OpConversionPattern<MIGraphXOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<MIGraphXOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(MIGraphXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

template <typename MIGraphXOp, typename TosaOp>
LogicalResult TrivialConverter<MIGraphXOp, TosaOp>::matchAndRewrite(
    MIGraphXOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  SmallVector<Type, 1> types;
  if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
    return failure();
  SmallVector<NamedAttribute> filteredAttrs =
      llvm::to_vector(op->getDiscardableAttrDictionary());
  rewriter.replaceOpWithNewOp<TosaOp>(op, types, adaptor.getOperands(),
                                      filteredAttrs);
  return success();
}

//===----------------------------------------------------------------------===//
// Base kernels (convolution, gemm)
//===----------------------------------------------------------------------===//

namespace {
template <typename ConvType>
struct ConvConverter final : public OpConversionPattern<ConvType> {
  using OpConversionPattern<ConvType>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ConvType>::OpAdaptor;

  // Note, this lowering pattern works for both migraphx.convolution and
  // migraphx.quant_convolution. The only difference between the two ops
  // is that quant_convolution allows convolution input and output to be
  // different types. Because of this, we use same lowering pattern but
  // different tablegen to capture the difference between the two ops.
  LogicalResult
  matchAndRewrite(ConvType op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename DotType>
struct DotConverter final : public OpConversionPattern<DotType> {
  using OpConversionPattern<DotType>::OpConversionPattern;
  using OpConversionPattern<DotType>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<DotType>::OpAdaptor;

  LogicalResult
  matchAndRewrite(DotType op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

template <typename ConvType>
LogicalResult ConvConverter<ConvType>::matchAndRewrite(
    ConvType op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getInput();
  auto inputType = cast<ShapedType>(input.getType());
  Value filter = adaptor.getFilter();
  ValueRange results = op->getResults();
  Type elementTy = inputType.getElementType();
  auto outputTy = cast<MIXRShapedType>(results[0].getType());
  SmallVector<int64_t> NCHW2NHWC{0, 2, 3, 1};
  SmallVector<int64_t> NHWC2NCHW{0, 3, 1, 2};

  int dims = outputTy.getShape().size() - 2;
  SmallVector<int64_t> toChannelLast{0};
  SmallVector<int64_t> fromChannelLast{0, dims + 1};
  for (int i = 0; i < dims; i++) {
    toChannelLast.push_back(i + 2);
    fromChannelLast.push_back(i + 1);
  }
  toChannelLast.push_back(1);

  // insert transpose to input and filter tensors
  input = getTransposeOp(loc, input, rewriter, toChannelLast);
  filter = getTransposeOp(loc, filter, rewriter, toChannelLast);
  ArrayRef<int64_t> outShape = outputTy.getShape();

  // original output shape was NCHW, change it into NHWC
  SmallVector<int64_t> newShape{outShape[0]};
  for (int i = 0; i < dims; i++)
    newShape.push_back(outShape[i + 2]);
  newShape.push_back(outShape[1]);
  Type newOutTy = RankedTensorType::get(newShape, outputTy.getElementType());

  // There is no tosa.conv1d, so instead we'll add a dummy x1 dimension
  // to the input tensors, and make a tosa.conv2d.  We'll also add the
  // ExpandedFrom1D attribute so we can undo it in tosa-to-rock.
  auto expandTo2D = [&rewriter, loc](mlir::Value value) {
    ArrayRef<int64_t> origShape = cast<ShapedType>(value.getType()).getShape();
    SmallVector<int64_t> expShape(origShape.drop_back());
    expShape.push_back(1);
    expShape.push_back(origShape.back());
    Value reshaped = rewriter.create<tosa::ReshapeOp>(
        loc, value, rewriter.getDenseI64ArrayAttr(expShape));
    return reshaped;
  };

  // Construct a new Conv2DOp.
  Operation *cop;
  Type new1DOutTy;
  switch (dims) {
  case 1:
    // Expand to do a conv2d, because there's no conv1d op.
    newShape.insert(std::prev(newShape.end()), 1);
    new1DOutTy = RankedTensorType::get(newShape, outputTy.getElementType());
    input = expandTo2D(input);
    filter = expandTo2D(filter);

    cop = rewriter.create<tosa::Conv2DOp>(
        loc, new1DOutTy,
        ValueRange{
            input, filter,
            getZeroTensor(loc, outputTy.getElementType(),
                          cast<ShapedType>(filter.getType()).getShape()[0],
                          rewriter)});
    cop->setAttr(rock::ExpandedFrom1DAttr::getMnemonic(),
                 rewriter.getUnitAttr());
    break;

  case 2:
    cop = rewriter.create<tosa::Conv2DOp>(
        loc, newOutTy,
        ValueRange{
            input, filter,
            getZeroTensor(loc, outputTy.getElementType(),
                          cast<ShapedType>(filter.getType()).getShape()[0],
                          rewriter)});
    break;
  case 3:
    cop = rewriter.create<tosa::Conv3DOp>(
        loc, newOutTy,
        ValueRange{
            input, filter,
            getZeroTensor(loc, outputTy.getElementType(),
                          cast<ShapedType>(filter.getType()).getShape()[0],
                          rewriter)});
    break;
  default:
    return op->emitError("Only 1-D, 2-D, and 3-D have been implemented.");
    break;
  }

  // translate attributes
  auto padAttr = cast<ArrayAttr>(op->getAttr("padding"));
  auto strideAttr = cast<ArrayAttr>(op->getAttr("stride"));
  auto dilationAttr = cast<ArrayAttr>(op->getAttr("dilation"));
  // MIGraphX padAttr is [hlow, wlow, hhigh, whigh] while TOSA padAttr
  // is [hlow, hhigh, wlow, whigh].
  SmallVector<int64_t> pads;
  for (int i = 0; i < dims; i++) {
    pads.push_back(dyn_cast<IntegerAttr>(padAttr[i]).getInt());
    pads.push_back(dyn_cast<IntegerAttr>(padAttr[i + dims]).getInt());
  }

  SmallVector<int64_t> strides;
  SmallVector<int64_t> dilations;
  for (size_t i = 0; i < strideAttr.size(); i++) {
    strides.push_back(dyn_cast<IntegerAttr>(strideAttr[i]).getInt());
    dilations.push_back(dyn_cast<IntegerAttr>(dilationAttr[i]).getInt());
  }

  int64_t group = op.getGroup();

  // convolution config attributes

  if (dims == 1) {
    if ((dilations.size() != 1) || (strides.size() != 1) ||
        (pads.size() != 2)) {
      return op->emitError(
          "1-D convolution has improper dilation, stride, or pad.");
    }
    dilations.push_back(0);
    strides.push_back(1);
    pads.push_back(0);
    pads.push_back(0);
  }

  cop->setAttr("dilation", rewriter.getDenseI64ArrayAttr(dilations));
  cop->setAttr("stride", rewriter.getDenseI64ArrayAttr(strides));
  cop->setAttr("pad", rewriter.getDenseI64ArrayAttr(pads));
  cop->setAttr("group", rewriter.getI64IntegerAttr(group));

  // Convert optional attributes
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

  if (dims == 1) {
    cop = rewriter.create<tosa::ReshapeOp>(
        loc, cop->getResult(0),
        rewriter.getDenseI64ArrayAttr(cast<ShapedType>(newOutTy).getShape()));
  }

  // transpose the output back to NCHW so that it can match following
  // operators.
  auto top = getTransposeOp(loc, cop->getResult(0), rewriter, fromChannelLast);
  rewriter.replaceOp(op, top);
  return success();
}

template <typename DotType>
LogicalResult DotConverter<DotType>::matchAndRewrite(
    DotType op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  auto inA = cast<TypedValue<RankedTensorType>>(adaptor.getInA());
  auto inB = cast<TypedValue<RankedTensorType>>(adaptor.getInB());
  auto results = op->getResults();
  Type elementTy = inA.getType().getElementType();
  auto origOutputTy = cast<MIXRShapedType>(results[0].getType());

  // check batch dimension. Tosa matmul only allow a single dimension for it,
  // add reshape ops to flatten and restore the original dimension.
  ArrayRef<int64_t> origOutDims = origOutputTy.getShape();
  RankedTensorType newOutType =
      RankedTensorType::get(origOutDims, origOutputTy.getElementType());
  size_t outRank = origOutDims.size();
  ArrayRef<int64_t> orgDimsA = inA.getType().getShape();
  ArrayRef<int64_t> orgDimsB = inB.getType().getShape();
  size_t rankA = orgDimsA.size();
  size_t rankB = orgDimsB.size();

  // A, B, Out have the same rank. rank=2 assumes batch=1.
  // Here handling special cases.
  if (outRank != 3 || rankA != rankB ||
      (outRank == 3 && orgDimsA[0] != orgDimsB[0])) {
    int64_t batchSizeA = 1, batchSizeB = 1, batchSizeC = 1;
    for (size_t i = 0; i < outRank - 2; i++) {
      batchSizeC *= origOutDims[i];
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
    int64_t newDimsOut[3] = {batchSizeC, origOutDims[outRank - 2],
                             origOutDims[outRank - 1]};
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
    newOutType =
        RankedTensorType::get(newDimsOut, origOutputTy.getElementType());
    auto reshapeAOp = rewriter.create<tosa::ReshapeOp>(
        loc, newAType, inA, rewriter.getDenseI64ArrayAttr(newDimsA));
    auto reshapeBOp = rewriter.create<tosa::ReshapeOp>(
        loc, newBType, inB, rewriter.getDenseI64ArrayAttr(newDimsB));

    // reassign inputs.
    inA = cast<TypedValue<RankedTensorType>>(reshapeAOp.getResult());
    inB = cast<TypedValue<RankedTensorType>>(reshapeBOp.getResult());
  }
  // Construct tosa.matmul.
  auto mop = rewriter.create<tosa::MatMulOp>(loc, newOutType, inA, inB);

  // Convert optional attributes
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
        loc, getTypeConverter()->convertType(origOutputTy), mop,
        rewriter.getDenseI64ArrayAttr(origOutDims));
    rewriter.replaceOp(op, rop);
    return success();
  }
  rewriter.replaceOp(op, mop);
  return success();
}

//===----------------------------------------------------------------------===//
// Tensor views and shape manipulation
//===----------------------------------------------------------------------===//
namespace {
struct BroadcastConverter final
    : public OpConversionPattern<migraphx::BroadcastOp> {
  using OpConversionPattern<migraphx::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct MultiBroadcastConverter final
    : public OpConversionPattern<migraphx::MultiBroadcastOp> {
  using OpConversionPattern<migraphx::MultiBroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::MultiBroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct TransposeConverter final
    : public OpConversionPattern<migraphx::TransposeOp> {
  using OpConversionPattern<migraphx::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ReshapeConverter final
    : public OpConversionPattern<migraphx::ReshapeOp> {
  using OpConversionPattern<migraphx::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct SliceConverter final : public OpConversionPattern<migraphx::SliceOp> {
  using OpConversionPattern<migraphx::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
BroadcastConverter::matchAndRewrite(migraphx::BroadcastOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  ArrayRef<int64_t> inShape = op.getInput().getType().getShape();
  ArrayRef<int64_t> outShape = op.getOutput().getType().getShape();
  uint32_t outRank = op.getOutput().getType().getRank();
  Type elemType = op.getOutput().getType().getElementType();
  auto axis =
      static_cast<size_t>(cast<IntegerAttr>(op->getAttr("axis")).getInt());

  SmallVector<int64_t, 5> newShape;
  for (uint32_t i = 0; i < outRank; i++) {
    if (i >= axis && (i - axis) < inShape.size()) {
      newShape.push_back(inShape[i - axis]);
    } else {
      newShape.push_back(1);
    }
  }
  tosa::ReshapeOp sameRankReshapedOp = createOpAndInfer<tosa::ReshapeOp>(
      rewriter, loc, elemType, adaptor.getInput(),
      rewriter.getDenseI64ArrayAttr(newShape));

  auto outType = RankedTensorType::get(outShape, elemType);
  // We create a dummy zero addition with implicit broadcasting
  // because tosa does not have an explicit broadcast op
  auto zeroTensor = getZeroTensor(loc, outType, rewriter);
  auto addWithZero = createOpAndInfer<tosa::AddOp>(
      rewriter, loc, elemType, zeroTensor, sameRankReshapedOp);

  rewriter.replaceOp(op, addWithZero);
  return success();
}

LogicalResult MultiBroadcastConverter::matchAndRewrite(
    migraphx::MultiBroadcastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  auto inType = cast<RankedTensorType>(adaptor.getInput().getType());
  auto outType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getOutput().getType()));
  ArrayRef<int64_t> outShape = outType.getShape();
  ArrayRef<int64_t> outStrides = op.getOutput().getType().getStrides();
  uint32_t inRank = inType.getRank();
  uint32_t outRank = outType.getRank();
  Type elemType = outType.getElementType();

  // If its a splat constant, we can broadcast it trivially
  if (tosa::ConstOp constOp =
          adaptor.getInput().getDefiningOp<tosa::ConstOp>()) {
    if (constOp.getValueAttr().isSplat()) {
      auto outTy = RankedTensorType::get(outShape, elemType);
      auto bcastConstAttr = DenseElementsAttr::get(
          outTy, cast<DenseElementsAttr>(constOp.getValueAttr())
                     .getSplatValue<Attribute>());
      tosa::ConstOp newConstOp =
          rewriter.create<tosa::ConstOp>(loc, outTy, bcastConstAttr);
      rewriter.replaceOp(op, newConstOp);
      return success();
    }
  }

  if (outRank < inRank) {
    return op.emitError("MultiBroadcastOp shouldn't reduce rank.\n");
  }

  Value replacingValue = adaptor.getInput();
  if (outRank > inRank) {
    SmallVector<int64_t, 5> newShape;
    newShape.reserve(outRank);
    for (auto [len, stride] : llvm::zip_equal(outShape, outStrides))
      newShape.push_back(stride == 0 ? 1 : len);
    tosa::ReshapeOp sameRankReshapedOp = createOpAndInfer<tosa::ReshapeOp>(
        rewriter, loc, elemType, adaptor.getInput(),
        rewriter.getDenseI64ArrayAttr(newShape));
    replacingValue = sameRankReshapedOp.getResult();
  }

  // We create a dummy zero addition with implicit broadcasting
  // because tosa does not have an explicit broadcast op
  auto zeroTensor = getZeroTensor(loc, outType, rewriter);
  auto addWithZero = createOpAndInfer<tosa::AddOp>(rewriter, loc, elemType,
                                                   zeroTensor, replacingValue);

  rewriter.replaceOp(op, addWithZero);
  return success();
}

LogicalResult
TransposeConverter::matchAndRewrite(migraphx::TransposeOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  SmallVector<int64_t, 4> permutation;
  permutation.reserve(op.getPermutation().size());
  for (auto permElem : op.getPermutation().getAsRange<IntegerAttr>())
    permutation.push_back(permElem.getInt());
  Value result = getTransposeOp(loc, adaptor.getInput(), rewriter, permutation);
  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult
ReshapeConverter::matchAndRewrite(migraphx::ReshapeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getInput();
  ArrayAttr dims = adaptor.getDims();
  Type outputTy = getTypeConverter()->convertType(op.getOutput().getType());
  SmallVector<int64_t, 5> newShape;
  for (auto dim : dims) {
    newShape.push_back(dyn_cast<IntegerAttr>(dim).getInt());
  }

  auto rop = rewriter.create<tosa::ReshapeOp>(
      loc, outputTy, input, rewriter.getDenseI64ArrayAttr(newShape));

  rewriter.replaceOp(op, rop);
  return success();
}

LogicalResult
SliceConverter::matchAndRewrite(migraphx::SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  SmallVector<int64_t, 5> start;
  SmallVector<int64_t, 5> size;
  ArrayAttr axes = op.getAxes();
  ArrayAttr axesStarts = op.getStarts();
  ArrayAttr axesEnds = op.getEnds();

  Value input = adaptor.getInput();
  auto newInType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> inShape = newInType.getShape();
  for (int64_t dim : inShape) {
    start.push_back(0);
    size.push_back(dim);
  }

  for (auto [axis, axisS, axisE] : llvm::zip(axes, axesStarts, axesEnds)) {
    int64_t axisInt = cast<IntegerAttr>(axis).getInt();
    int64_t axisSInt = cast<IntegerAttr>(axisS).getInt();
    int64_t axisEInt = cast<IntegerAttr>(axisE).getInt();
    start[axisInt] = axisSInt;
    size[axisInt] = axisEInt - axisSInt;
  }

  auto sliceOp = createOpAndInfer<tosa::SliceOp>(
      rewriter, loc, newInType.getElementType(), input,
      rewriter.getDenseI64ArrayAttr(start),
      rewriter.getDenseI64ArrayAttr(size));
  rewriter.replaceOp(op, sliceOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Reductions
//===----------------------------------------------------------------------===//
namespace {
struct ReduceMeanConverter final
    : public OpConversionPattern<migraphx::ReduceMeanOp> {
  using OpConversionPattern<migraphx::ReduceMeanOp>::OpConversionPattern;

  tosa::ConstOp createNumElementsTosaConst(
      Location loc, TypedValue<RankedTensorType> inputTensor,
      IntegerAttr axisAttr, ConversionPatternRewriter &rewriter) const;

  LogicalResult
  matchAndRewrite(migraphx::ReduceMeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ReduceSumConverter final
    : public OpConversionPattern<migraphx::ReduceSumOp> {
  using OpConversionPattern<migraphx::ReduceSumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

tosa::ConstOp ReduceMeanConverter::createNumElementsTosaConst(
    Location loc, TypedValue<RankedTensorType> inputTensor,
    IntegerAttr axisAttr, ConversionPatternRewriter &rewriter) const {
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

LogicalResult ReduceMeanConverter::matchAndRewrite(
    migraphx::ReduceMeanOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  ArrayRef<Attribute> axes = op.getAxes().getValue();
  if (axes.size() != 1) {
    return op.emitError("We only support single axes reductions!");
  }
  IntegerAttr axis =
      rewriter.getI32IntegerAttr(cast<IntegerAttr>(axes[0]).getInt());
  auto input = cast<TypedValue<RankedTensorType>>(adaptor.getInput());
  Type elementType = input.getType().getElementType();

  tosa::ConstOp tosaConstantNumElements =
      createNumElementsTosaConst(loc, input, axis, rewriter);
  auto tosaReciprocal = createOpAndInfer<tosa::ReciprocalOp>(
      rewriter, loc, elementType, tosaConstantNumElements);
  auto tosaMul = createOpAndInfer<tosa::MulOp>(rewriter, loc, elementType,
                                               adaptor.getInput(),
                                               tosaReciprocal, /*shift=*/0);
  auto tosaReduceSum = createOpAndInfer<tosa::ReduceSumOp>(
      rewriter, loc, elementType, tosaMul, axis);
  rewriter.replaceOp(op, tosaReduceSum);
  return success();
}

LogicalResult
ReduceSumConverter::matchAndRewrite(migraphx::ReduceSumOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  ArrayRef<Attribute> axes = op.getAxes().getValue();
  if (axes.size() != 1) {
    return op.emitError("We only support single axes reductions!");
  }
  IntegerAttr axis =
      rewriter.getI32IntegerAttr(cast<IntegerAttr>(axes[0]).getInt());
  auto input = cast<TypedValue<RankedTensorType>>(adaptor.getInput());
  Type elementType = input.getType().getElementType();
  auto tosaReduceSum = createOpAndInfer<tosa::ReduceSumOp>(
      rewriter, loc, elementType, input, axis);
  rewriter.replaceOp(op, tosaReduceSum);
  return success();
}

//===----------------------------------------------------------------------===//
// Binary operations
//===----------------------------------------------------------------------===//
namespace {
struct DivConverter final : public OpConversionPattern<migraphx::DivOp> {
  using OpConversionPattern<migraphx::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct MulConverter final : public OpConversionPattern<migraphx::MulOp> {
  using OpConversionPattern<migraphx::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
DivConverter::matchAndRewrite(migraphx::DivOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  auto inATensor = cast<TypedValue<RankedTensorType>>(adaptor.getInA());
  auto inBTensor = cast<TypedValue<RankedTensorType>>(adaptor.getInB());
  Type elementType = inATensor.getType().getElementType();
  if (isa<IntegerType>(elementType)) {
    Value div = createOpAndInfer<tosa::IntDivOp>(rewriter, loc, elementType,
                                                 inATensor, inBTensor);
    rewriter.replaceOp(op, div);
    return success();
  }
  Value recip = createOpAndInfer<tosa::ReciprocalOp>(rewriter, loc, elementType,
                                                     inBTensor);
  Value mul = createOpAndInfer<tosa::MulOp>(rewriter, loc, elementType,
                                            inATensor, recip, /*shift=*/0);
  rewriter.replaceOp(op, mul);
  return success();
}

LogicalResult
MulConverter::matchAndRewrite(migraphx::MulOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<tosa::MulOp>(
      op, getTypeConverter()->convertType(op.getResult().getType()),
      adaptor.getInA(), adaptor.getInB(), /*shift=*/0);
  return success();
}

//===----------------------------------------------------------------------===//
// Unary operations
//===----------------------------------------------------------------------===//
namespace {
struct DeQuantizeLinearConverter final
    : public OpConversionPattern<migraphx::DeQuantizeLinearOp> {
  using OpConversionPattern<migraphx::DeQuantizeLinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::DeQuantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct QuantizeLinearConverter final
    : public OpConversionPattern<migraphx::QuantizeLinearOp> {
  using OpConversionPattern<migraphx::QuantizeLinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::QuantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ConvertConverter final
    : public OpConversionPattern<migraphx::ConvertOp> {
  using OpConversionPattern<migraphx::ConvertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::ConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct NegConverter final : public OpConversionPattern<migraphx::NegOp> {
  using OpConversionPattern<migraphx::NegOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ReluConverter final : public OpConversionPattern<migraphx::ReluOp> {
  using OpConversionPattern<migraphx::ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct SoftmaxConverter final
    : public OpConversionPattern<migraphx::SoftmaxOp> {
  using OpConversionPattern<migraphx::SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

// MIGraphX implements:
// Let T = scale element type
// output[i] = (convert<T>(input[i]) - convert<T>(zero_pts[i])) * scales[i];
// For f32, this matches ONNX reference, dequantizing to f16, if it's ever done
// will be less precise than the reference but that's probably fine.
LogicalResult DeQuantizeLinearConverter::matchAndRewrite(
    migraphx::DeQuantizeLinearOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getInput();
  Value scale = adaptor.getScale();
  Value output = op.getOutput();
  Location loc = op->getLoc();

  Type outputType = getShapedElementTy(output);
  Value upcastInput = createCastOp(rewriter, loc, outputType, input);

  Value shifted = upcastInput;
  if (auto bias = adaptor.getBias()) {
    Value upcastBias = createCastOp(rewriter, loc, outputType, bias);
    shifted = createOpAndInfer<tosa::SubOp>(rewriter, loc, outputType,
                                            upcastInput, upcastBias);
  }

  Value scaled = createOpAndInfer<tosa::MulOp>(rewriter, loc, outputType,
                                               shifted, scale, /*shift=*/0);

  rewriter.replaceOp(op, scaled);
  return success();
}

// MIGraphX pseudo code:
// int32_t quantized = static_cast<int32>(
//      std::round(input[i] / scales[i])) + zero_pts[i];
// output[i] = std::max(-128, std::min(127, quantized));
LogicalResult QuantizeLinearConverter::matchAndRewrite(
    migraphx::QuantizeLinearOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.getInput();
  Value scale = adaptor.getScale();
  Value bias = adaptor.getBias();
  Value output = op.getOutput();

  Type elementType = getShapedElementTy(input);
  Value inverseScale =
      createOpAndInfer<tosa::ReciprocalOp>(rewriter, loc, elementType, scale);
  Value scaled = createOpAndInfer<tosa::MulOp>(
      rewriter, loc, elementType, input, inverseScale, /*shift=*/0);

  Type outputType = getShapedElementTy(output);
  // If there is a bias, we upcast to the larger of the bias type and int32_t
  // or float (which is what the bias type is in dequantize, the MLIR
  // quantization implementation, and other ML frameworks) and then do a
  // clamping truncation to the output type so that adding a bias saturates
  // instead of overflowing.
  Type biasType = outputType;
  if (bias) {
    biasType = getShapedElementTy(bias);
    if (biasType.getIntOrFloatBitWidth() < 32) {
      biasType = isa<IntegerType>(biasType) ? cast<Type>(rewriter.getI32Type())
                                            : cast<Type>(rewriter.getF32Type());
      bias = createCastOp(rewriter, loc, biasType, bias);
    }
  }
  Value asShort = createCastOp(rewriter, loc, biasType, scaled);
  Value biased = asShort;
  if (bias)
    biased =
        createOpAndInfer<tosa::AddOp>(rewriter, loc, biasType, asShort, bias);

  Value result = biased;
  if (biasType != outputType) {
    unsigned width = outputType.getIntOrFloatBitWidth();
    APInt minI(width, 0), maxI(width, 0);
    // Must be floats because tosa.clamp expects a f32 attribute specifically.
    APFloat minF(0.0f), maxF(0.0f);
    if (auto outFloatType = dyn_cast<FloatType>(outputType)) {
      const llvm::fltSemantics &outSem = outFloatType.getFloatSemantics();
      const llvm::fltSemantics &biasSem =
          cast<FloatType>(biasType).getFloatSemantics();
      minF = APFloat::getLargest(outSem, /*Negative=*/true);
      maxF = APFloat::getLargest(outSem, /*Negative=*/false);
      bool itsExtendNoWayWeCanLoseInfo = false;
      std::ignore = minF.convert(biasSem, APFloat::rmNearestTiesToEven,
                                 &itsExtendNoWayWeCanLoseInfo);
      std::ignore = maxF.convert(biasSem, APFloat::rmNearestTiesToEven,
                                 &itsExtendNoWayWeCanLoseInfo);
      minI = APInt(64, (int64_t)(minF.convertToFloat()));
      maxI = APInt(64, (int64_t)(minF.convertToFloat()));
    } else {
      minI = APInt::getSignedMinValue(width);
      maxI = APInt::getSignedMaxValue(width);
      minF.convertFromAPInt(minI, /*IsSigned=*/true,
                            APFloat::rmNearestTiesToEven);
      maxF.convertFromAPInt(maxI, /*IsSigned=*/true,
                            APFloat::rmNearestTiesToEven);
    }

    FloatAttr minFatt = rewriter.getFloatAttr(rewriter.getF32Type(), minF);
    FloatAttr maxFatt = rewriter.getFloatAttr(rewriter.getF32Type(), maxF);
    result = createOpAndInfer<tosa::ClampOp>(
        rewriter, loc, biasType, result, minI.getSExtValue(),
        maxI.getSExtValue(), minFatt, maxFatt);
    result = createCastOp(rewriter, loc, outputType, result);
  }
  rewriter.replaceOp(op, result);

  return success();
}

LogicalResult
ConvertConverter::matchAndRewrite(migraphx::ConvertOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (!op.getZeroExtend()) {
    rewriter.replaceOpWithNewOp<tosa::CastOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getInA());
    return success();
  }
  rewriter.replaceOpWithNewOp<tosa::CustomOp>(
      op, getTypeConverter()->convertType(op.getResult().getType()),
      "unsigned_cast", "rocmlir", "", adaptor.getInA());
  return success();
}

LogicalResult
NegConverter::matchAndRewrite(migraphx::NegOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<tosa::NegateOp>(
      op, getTypeConverter()->convertType(op.getResult().getType()),
      adaptor.getInA(), nullptr);
  return success();
}

LogicalResult
ReluConverter::matchAndRewrite(migraphx::ReluOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value inA = adaptor.getInA();
  auto outType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));
  auto zero = getZeroTensor(op.getLoc(), outType, rewriter);
  // Since the zero is second, this handles any implicit broadcast.
  rewriter.replaceOpWithNewOp<tosa::MaximumOp>(op, outType, inA, zero);
  return success();
}

LogicalResult
SoftmaxConverter::matchAndRewrite(migraphx::SoftmaxOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.getInput();
  IntegerAttr axisAttr = rewriter.getI32IntegerAttr(op.getAxisAttr().getInt());
  ShapedType inputType = cast<ShapedType>(input.getType());
  Type elementType = inputType.getElementType();

  auto tosaMax = createOpAndInfer<tosa::ReduceMaxOp>(rewriter, loc, elementType,
                                                     input, axisAttr);
  auto tosaSub =
      createOpAndInfer<tosa::SubOp>(rewriter, loc, elementType, input, tosaMax);
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

//===----------------------------------------------------------------------===//
// Misc. ops
//===----------------------------------------------------------------------===//
namespace {
struct LiteralConverter final
    : public OpConversionPattern<migraphx::LiteralOp> {
  using OpConversionPattern<migraphx::LiteralOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::LiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ClipConverter final : public OpConversionPattern<migraphx::ClipOp> {
  using OpConversionPattern<migraphx::ClipOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::ClipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct WhereConverter final : public OpConversionPattern<migraphx::WhereOp> {
  using OpConversionPattern<migraphx::WhereOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
LiteralConverter::matchAndRewrite(migraphx::LiteralOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  MIXRShapedType type = op.getResult().getType();
  RankedTensorType newType = type.asTensor();
  ElementsAttr value = op.getValue();
  if (value.isSplat() && value.getType() != newType)
    value = SplatElementsAttr::get(newType, value.getSplatValue<Attribute>());
  rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, type.asTensor(), value);
  return success();
}

LogicalResult
ClipConverter::matchAndRewrite(migraphx::ClipOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value x = adaptor.getX();
  Value minVals = adaptor.getMinVals();
  Value maxVals = adaptor.getMaxVals();
  auto outType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));
  Value atLeastMin = rewriter.create<tosa::MaximumOp>(loc, outType, x, minVals);
  rewriter.replaceOpWithNewOp<tosa::MinimumOp>(op, outType, atLeastMin,
                                               maxVals);
  return success();
}

LogicalResult
WhereConverter::matchAndRewrite(migraphx::WhereOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value rawCond = adaptor.getCond();
  Value inA = adaptor.getInA();
  Value inB = adaptor.getInB();
  Value cond = createCastOp(rewriter, loc, rewriter.getI1Type(), rawCond);
  rewriter.replaceOpWithNewOp<tosa::SelectOp>(
      op, getTypeConverter()->convertType(op.getResult().getType()), cond, inA,
      inB);
  return success();
}

//===----------------------------------------------------------------------===//
// Function boundaries
//===----------------------------------------------------------------------===//
namespace {
struct AsLogicalShapeConverter final
    : public OpConversionPattern<migraphx::AsLogicalShapeOp> {
  using OpConversionPattern<migraphx::AsLogicalShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::AsLogicalShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct AsUnderlyingShapeConverter final
    : public OpConversionPattern<migraphx::AsUnderlyingShapeOp> {
  using OpConversionPattern<migraphx::AsUnderlyingShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::AsUnderlyingShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

/// This mirrors the call op conversion pattern but works for mhal.launch.
struct MHALLaunchConverter final : public OpConversionPattern<mhal::LaunchOp> {
  using OpConversionPattern<mhal::LaunchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhal::LaunchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult AsLogicalShapeConverter::matchAndRewrite(
    migraphx::AsLogicalShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MIXRShapedType inType = op.getIn().getType();
  RankedTensorType resultType = op.getOut().getType();
  Value in = adaptor.getIn();

  // First, expand ourselves back out to the N-D type that we're logically
  // working with in memory.
  RankedTensorType memoryLayoutType = inType.asMemoryLayoutTensor();
  Value expanded = in;
  if (in.getType() != memoryLayoutType)
    expanded =
        rewriter.create<tosa::ReshapeOp>(loc, in, memoryLayoutType.getShape());

  // This is the permutation that reorders the strides into standard shape.
  // Equivalently, it is the permutation that, when applied to a standard
  // shape, produces its in-memory layout. So, to get back to standard/logical
  // shape, we need to invert it.
  SmallVector<int64_t, 4> inversePermutation;
  inType.getStridePermutation(inversePermutation);
  SmallVector<int64_t> permutation;
  permutation.resize_for_overwrite(inversePermutation.size());
  bool hasTranspose = false;
  for (auto [to, from] : llvm::enumerate(inversePermutation)) {
    permutation[from] = to;
    hasTranspose |= (from != static_cast<int64_t>(to));
  }
  Value transposed = expanded;
  if (hasTranspose)
    transposed = getTransposeOp(loc, expanded, rewriter, permutation);
  auto transposedType = cast<RankedTensorType>(transposed.getType());
  if (transposedType == resultType) {
    rewriter.replaceOp(op, transposed);
    return success();
  }

  SmallVector<int64_t, 4> slicingShape(resultType.getShape());
  for (auto [dim, stride] :
       llvm::zip_equal(slicingShape, inType.getStrides())) {
    if (stride == 0)
      dim = 1;
  }

  Value maybeSliced = transposed;
  if (transposedType.getShape() != ArrayRef(slicingShape)) {
    SmallVector<int64_t, 4> starts(permutation.size(), 0);
    RankedTensorType sliceType = resultType.clone(slicingShape);
    maybeSliced = rewriter.create<tosa::SliceOp>(
        loc, sliceType, transposed, rewriter.getDenseI64ArrayAttr(starts),
        rewriter.getDenseI64ArrayAttr(slicingShape));
  }
  Value maybeBroadcast = maybeSliced;
  if (maybeSliced.getType() != resultType) {
    // We need a broadcast
    Value zeroTensor = getZeroTensor(loc, resultType, rewriter);
    maybeBroadcast =
        rewriter.create<tosa::AddOp>(loc, resultType, zeroTensor, maybeSliced);
  }
  rewriter.replaceOp(op, maybeBroadcast);
  return success();
}

LogicalResult AsUnderlyingShapeConverter::matchAndRewrite(
    migraphx::AsUnderlyingShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MIXRShapedType resultType = op.getOut().getType();
  RankedTensorType memoryLayoutType = resultType.asMemoryLayoutTensor();
  auto resultTensorType =
      cast<RankedTensorType>(getTypeConverter()->convertType(resultType));
  if (!resultTensorType)
    return op.emitOpError("unsupported conversion to underlying shape");
  Value in = adaptor.getIn();
  SmallVector<int64_t, 4> permutation;
  // This is the permutation that reorderd strides into the order they'd be in
  // in a standard shape. So, applying it to a logically-shaped tensor gets
  // you the tensor in in-memory layout.
  resultType.getStridePermutation(permutation);
  Value transposed = in;
  if (!llvm::is_sorted(permutation))
    transposed = getTransposeOp(loc, in, rewriter, permutation);
  if (transposed.getType() != memoryLayoutType) {
    rewriter.eraseOp(transposed.getDefiningOp());
    return op.emitOpError(
        "writing to tensors with long strides or broadcasts is unsupported");
  }

  Value collapsed = transposed;
  if (transposed.getType() != resultTensorType)
    collapsed = rewriter.create<tosa::ReshapeOp>(loc, transposed,
                                                 resultTensorType.getShape());
  rewriter.replaceOp(op, collapsed);
  return success();
}

LogicalResult MHALLaunchConverter::matchAndRewrite(
    mhal::LaunchOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Convert the original function results.
  SmallVector<Type, 2> resultTypes;
  if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
    return failure();

  // If this isn't a one-to-one type mapping, we don't know how to aggregate
  // the results.
  if (op->getNumResults() != resultTypes.size())
    return failure();

  // Substitute with the new result types from the corresponding FuncType
  // conversion.
  rewriter.replaceOpWithNewOp<mhal::LaunchOp>(
      op, op.getCalleeAttr(), resultTypes, adaptor.getOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// External interface
//===----------------------------------------------------------------------===//

void migraphx::populateMIGraphXToTosaConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns
      .add<ConvConverter<ConvolutionOp>, ConvConverter<QuantConvolutionOp>,
           DotConverter<DotOp>, DotConverter<QuantDotOp>, BroadcastConverter,
           MultiBroadcastConverter, TransposeConverter, ReshapeConverter,
           SliceConverter, ReduceMeanConverter, ReduceSumConverter,
           TrivialConverter<AddOp, tosa::AddOp>,
           TrivialConverter<SubOp, tosa::SubOp>,
           TrivialConverter<PowOp, tosa::PowOp>, DivConverter, MulConverter,
           TrivialConverter<AbsOp, tosa::AbsOp>,
           TrivialConverter<CeilOp, tosa::CeilOp>,
           TrivialConverter<ErfOp, tosa::ErfOp>,
           TrivialConverter<ExpOp, tosa::ExpOp>,
           TrivialConverter<FloorOp, tosa::FloorOp>,
           TrivialConverter<LogOp, tosa::LogOp>,
           TrivialConverter<RecipOp, tosa::ReciprocalOp>,
           TrivialConverter<RsqrtOp, tosa::RsqrtOp>,
           TrivialConverter<SigmoidOp, tosa::SigmoidOp>,
           TrivialConverter<TanhOp, tosa::TanhOp>, DeQuantizeLinearConverter,
           QuantizeLinearConverter, DeQuantizeLinearConverter, ConvertConverter,
           NegConverter, ReluConverter, SoftmaxConverter, LiteralConverter,
           ClipConverter, WhereConverter>(typeConverter, patterns.getContext());
}

void mlir::migraphx::populateMIGraphXFuncBoundaryToTosaConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AsLogicalShapeConverter, AsUnderlyingShapeConverter,
               TrivialConverter<func::ReturnOp, func::ReturnOp>,
               MHALLaunchConverter>(typeConverter, patterns.getContext());
  // Add upstream patterns that take care of func.func and its friends.
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
}
