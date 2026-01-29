//===- MIGraphXToLinalg.cpp - Lowering MIGraphX to Linalg Dialect ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the MIGraphX to the Linalg dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/MIGraphXToLinalg/MIGraphXToLinalg.h"
#include "mlir/Conversion/MIGraphXToTosa/MIGraphXToTosa.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

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
    SmallVector<ReassociationIndices, 4> reassociationIndex;
    reassociationIndex.push_back({});
    for (std::size_t i = 0; i < resultType.getShape().size(); ++i) {
      reassociationIndex[0].push_back(i);
    }
    auto newShape = tensor::ExpandShapeOp::create(rewriter, loc, resultType, in,
                                                  reassociationIndex);
    rewriter.replaceOp(op, newShape);
    return success();
  }

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
    for (std::size_t i = 0; i < resultType.getShape().size(); ++i) {
      reassociationIndex[0].push_back(i);
    }
    auto reshape = tensor::CollapseShapeOp::create(
        rewriter, loc, resultTensorType, in, reassociationIndex);
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

  // don't emit linalg.generic for 2D and 3D case to preserver type sugar
  if (dimension == 2) {
    SmallVector<int64_t, 2> outputShape{aShape[0], bShape[1]};
    Value zero =
        arith::ConstantOp::create(rewriter, loc,
                                  rewriter.getZeroAttr(RankedTensorType::get(
                                      outputShape, aType.getElementType())));
    auto matMulOp =
        linalg::MatmulOp::create(rewriter, loc, {aIn, bIn}, zero, {});
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

  return op.emitError("only support 2D/3D for now");
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
  patterns.add<AsUnderlyingShapeConverter, AsLogicalShapeOpConverter>(
      typeConverter, patterns.getContext());
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
  migraphx::populateMIGrpahXToLinalgTrivialConverter(patterns, typeConverter);
}
