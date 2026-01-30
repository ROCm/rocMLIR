#include "mlir/Conversion/MIGraphXToLinalg/MIGraphXToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

linalg::BoundaryTypeConverter::BoundaryTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion([](migraphx::MIXRShapedType shaped) {
    return shaped.asFlatMemoryTensor();
  });
}

// FIXME: this is taken from MIGraphXToTosa.cpp. It may be a good idea
// to refactor some code so we can share code easier.
linalg::MIXRShapedToTensorConverter::MIXRShapedToTensorConverter() {
  addConversion([](Type type) {
    if (type.isInteger() && !type.isSignlessInteger()) {
      type = IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth(),
                              IntegerType::SignednessSemantics::Signless);
    }
    return type;
  });
  addConversion([](migraphx::MIXRShapedType shaped) {
    RankedTensorType newType = shaped.asTensor();
    Type elementType = newType.getElementType();

    // Convert to signless if the element type is a signed integer
    if (elementType.isInteger() && !elementType.isSignlessInteger()) {
      elementType = IntegerType::get(
          shaped.getContext(), elementType.getIntOrFloatBitWidth(),
          IntegerType::SignednessSemantics::Signless);
      // Create a new tensor type with the signless element type
      newType = RankedTensorType::get(newType.getShape(), elementType);
    }
    return newType;
  });

  addSourceMaterialization([](OpBuilder &b,
                              migraphx::MIXRShapedType shapedResType,
                              ValueRange tensorResult, Location loc) -> Value {
    if (tensorResult.size() != 1)
      return Value(); // 1-1 conversions only.
    return migraphx::AsUnderlyingShapeOp::create(b, loc, shapedResType,
                                                 tensorResult[0]);
  });

  addTargetMaterialization([](OpBuilder &b, Type wantedInputType,
                              ValueRange shapedInput, Location loc) -> Value {
    if (shapedInput.size() != 1)
      return Value(); // 1-1 conversions only.
    return migraphx::AsLogicalShapeOp::create(b, loc, wantedInputType,
                                              shapedInput[0]);
  });
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

// FIXME: this is taken from MIGraphXToTosa
//===----------------------------------------------------------------------===//
// !migraphx.shaped materialization, and FuncBoundary
//===----------------------------------------------------------------------===//
namespace {
struct AsUnderlyingShapeConverter final
    : public OpConversionPattern<migraphx::AsUnderlyingShapeOp> {
  using OpConversionPattern<migraphx::AsUnderlyingShapeOp>::OpConversionPattern;
  using OpConversionPattern<migraphx::AsUnderlyingShapeOp>::getTypeConverter;
  using OpAdaptor =
      typename OpConversionPattern<migraphx::AsUnderlyingShapeOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(migraphx::AsUnderlyingShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct AsLogicalShapeOpConverter final
    : public OpConversionPattern<migraphx::AsLogicalShapeOp> {
  using OpConversionPattern<migraphx::AsLogicalShapeOp>::OpConversionPattern;
  using OpConversionPattern<migraphx::AsLogicalShapeOp>::getTypeConverter;
  using OpAdaptor =
      typename OpConversionPattern<migraphx::AsLogicalShapeOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(migraphx::AsLogicalShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

static bool isPermutationStandardForm(ArrayRef<int64_t> permutation) {
  for (std::size_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != (int64_t)i) {
      return false;
    }
  }

  return true;
}

LogicalResult AsLogicalShapeOpConverter::matchAndRewrite(
    migraphx::AsLogicalShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  migraphx::MIXRShapedType inType = op.getIn().getType();
  RankedTensorType resultType = op.getOut().getType();
  Value in = adaptor.getIn(); // The shape we are casting from

  SmallVector<int64_t, 4> permutation;
  inType.getStridePermutation(permutation);
  if (isPermutationStandardForm(permutation)) {
    // FIXME: add assertion too!
    // FIXME: is there a range in c++?
    SmallVector<ReassociationIndices, 4> reassociationIndex;
    reassociationIndex.push_back({});
    for(int64_t i = 0;i<resultType.getShape().size(); ++i){
      reassociationIndex[0].push_back(i);
    }
    auto newShape = tensor::ExpandShapeOp::create(rewriter, loc, resultType, in, reassociationIndex);
    rewriter.replaceOp(op, newShape);
    return success();
  }

  // FIXME: convert non standard form and broadcasting into logical shape
  return op.emitError("cannot convert this into logical shape for now");
}

LogicalResult AsUnderlyingShapeConverter::matchAndRewrite(
    migraphx::AsUnderlyingShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value in = adaptor.getIn();
  migraphx::MIXRShapedType resultType = op.getResult().getType();
  auto resultTensorType =
      cast<RankedTensorType>(getTypeConverter()->convertType(resultType));

  SmallVector<int64_t, 4> permutation;
  resultType.getStridePermutation(permutation);
  if (isPermutationStandardForm(permutation)) {
     SmallVector<ReassociationIndices, 4> reassociationIndex;
    reassociationIndex.push_back({});
    for(int64_t i = 0;i<resultType.getShape().size(); ++i){
      reassociationIndex[0].push_back(i);
    }
    auto reshape =
        tensor::CollapseShapeOp::create(rewriter, loc, resultTensorType, in, reassociationIndex);
    rewriter.replaceOp(op, reshape);
    return success();
  }

  return op.emitError("cannot convert non standard shape for now");
}

//===----------------------------------------------------------------------===//
// Base kernels (convolution, gemm)
//===----------------------------------------------------------------------===//
namespace {
struct DotConverter final : public OpConversionPattern<migraphx::DotOp> {
  using OpConversionPattern<migraphx::DotOp>::OpConversionPattern;
  using OpConversionPattern<migraphx::DotOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<migraphx::DotOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(migraphx::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
DotConverter::matchAndRewrite(migraphx::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value aIn = adaptor.getInA();
  Value bIn = adaptor.getInB();
  RankedTensorType aType = cast<TypedValue<RankedTensorType>>(aIn).getType();
  RankedTensorType bType = cast<TypedValue<RankedTensorType>>(bIn).getType();
  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  int64_t dimension = aShape.size();
  assert(aShape.size() == bShape.size());

  // don't emit linalg.generic for 2D and 3D case to preserver type sugar
  if (dimension == 2) {
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(aType.getElementType()));
    Value empty = tensor::EmptyOp::create(rewriter, loc, {aShape[0], bShape[1]},
                                          aType.getElementType());
    Value init =
        linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);
    auto matMulOp =
        linalg::MatmulOp::create(rewriter, loc, {aIn, bIn}, init, {});
    rewriter.replaceOp(op, matMulOp);
    return success();
  }

  if (dimension == 3) {
    SmallVector<int64_t, 3> shape{aShape[0], aShape[1], bShape[2]};
    Value init =
        arith::ConstantOp::create(rewriter, loc,
                                  rewriter.getZeroAttr(RankedTensorType::get(
                                      shape, aType.getElementType())));
    auto matMulOp =
        linalg::BatchMatmulOp::create(rewriter, loc, {aIn, bIn}, init, {});
    rewriter.replaceOp(op, matMulOp);
    return success();
  }

  // emit Linalg.generic for the general case
  return op.emitError("Only support 2D/3D for now");
}

//===----------------------------------------------------------------------===//
// populateMIGrpahXToLinalg* method
//===----------------------------------------------------------------------===//
void mlir::linalg::populateMIGraphXToLinalgConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<DotConverter>(converter, patterns.getContext());
}

void mlir::linalg::populateMIGraphXFuncBoundaryToLinalgConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AsUnderlyingShapeConverter, AsLogicalShapeOpConverter,
               TrivialConverter<func::ReturnOp, func::ReturnOp>>(
      typeConverter, patterns.getContext());
}
