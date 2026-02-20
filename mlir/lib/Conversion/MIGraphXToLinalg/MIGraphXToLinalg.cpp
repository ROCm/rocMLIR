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
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
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

/// Checking to see if the permutation vector is like (0, 1, 2, 3, 4, 5, ...)
static bool isPermutationStandardForm(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t, 4> increasingVec(permutation.size(), 0);
  std::iota(increasingVec.begin(), increasingVec.end(), 0);
  return llvm::equal(permutation, increasingVec);
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
    SmallVector<ReassociationIndices, 4> reassociationIndex(
        1, ReassociationIndices(resultType.getRank(), 0));
    std::iota(reassociationIndex[0].begin(), reassociationIndex[0].end(), 0);
    auto newShape = tensor::ExpandShapeOp::create(rewriter, loc, resultType, in,
                                                  reassociationIndex);
    rewriter.replaceOp(op, newShape);
    return success();
  }

  return op.emitError(
      "input shape is non standard or broadcast; cannot convert this shape");
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
    SmallVector<ReassociationIndices, 4> reassociationIndex(
        1, ReassociationIndices(resultType.getRank(), 0));
    std::iota(reassociationIndex[0].begin(), reassociationIndex[0].end(), 0);
    auto reshape = tensor::CollapseShapeOp::create(
        rewriter, loc, resultTensorType, in, reassociationIndex);
    rewriter.replaceOp(op, reshape);
    return success();
  }

  return op.emitError(
      "input shape is non standard or broadcast; cannot convert this shape");
}

// TODO: add support for scaled gemms, and migraphx::DeQuantizeLinearConverter
//===----------------------------------------------------------------------===//
// Base kernels (gemm)
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
  Location loc = op->getLoc();
  Value inA = adaptor.getInA();
  Value inB = adaptor.getInB();
  auto results = op->getResults();
  if (!isa<RankedTensorType>(inA.getType()) ||
      !isa<RankedTensorType>(inB.getType())) {
    return op.emitError("expected both operands to be RankedTensorType");
  }
  RankedTensorType aRankedType = cast<RankedTensorType>(inA.getType());
  RankedTensorType bRankedType = cast<RankedTensorType>(inB.getType());
  Type elementTy = aRankedType.getElementType();
  auto origOutputTy = cast<migraphx::MIXRShapedType>(results[0].getType());
  Type outElementTy = origOutputTy.getElementType();
  Type newOutElementTy = getTypeConverter()->convertType(outElementTy);

  ArrayRef<int64_t> origOutDims = origOutputTy.getShape();
  RankedTensorType newOutType =
      RankedTensorType::get(origOutDims, newOutElementTy);
  size_t outRank = origOutDims.size();
  ArrayRef<int64_t> orgDimsA = aRankedType.getShape();
  ArrayRef<int64_t> orgDimsB = bRankedType.getShape();
  size_t rankA = orgDimsA.size();
  size_t rankB = orgDimsB.size();

  if (rankA != rankB || rankB != outRank) {
    // It is possible to support rank of different dimensions such as
    // A = (3,2,2,2), B = (6,2,2,2), and C = (1,3,2,2,2). The tosa
    // lowering path doesn't seems to support it for now so we error
    // for now.
    return op.emitError("operands must have the same rank");
  }

  if (!aRankedType.hasStaticShape() || !bRankedType.hasStaticShape()) {
    return op.emitError("only static shape is supported for now");
  }

  // A nice help function that collapse and expand the shape when necessary
  auto getReassociationIndices = [](int64_t rank) {
    assert(rank >= 3 && "this help only works for rank greater than 3");
    SmallVector<ReassociationIndices, 4> reassociation(3,
                                                       ReassociationIndices());
    reassociation[0].resize(rank - 2);
    std::iota(reassociation[0].begin(), reassociation[0].end(), 0);
    reassociation[1] = {rank - 2};
    reassociation[2] = {rank - 1};
    return reassociation;
  };

  // nice help function to reshape the input
  auto reshapeToDimThree = [&](int64_t rank, Type newType, Value in) -> Value {
    return (rank == 2)
               ? tensor::ExpandShapeOp::create(rewriter, loc, newType, in,
                                               {{0, 1}, {2}})
               : tensor::CollapseShapeOp::create(rewriter, loc, newType, in,
                                                 getReassociationIndices(rank))
                     .getResult();
  };

  auto getBatchSize = [](ArrayRef<int64_t> shape) {
    return std::accumulate(shape.begin(), std::prev(shape.end(), 2), 1,
                           std::multiplies<int>());
  };

  // Handle special cases. Here we are going to compute the new shape of the
  // inputs and the outputs so that we can use linalg.batch_matmul which expects
  // the rank of the input and output to be 3.
  bool needToReshape = outRank != 3 || rankA != rankB ||
                       (outRank == 3 && orgDimsA[0] != orgDimsB[0]);
  if (needToReshape) {
    // reshape the (d0, d1, d2, ..., dn-1, dn) into (d0*d1*d2,...,dn-1,dn)
    int64_t batchSizeA = getBatchSize(orgDimsA);
    int64_t batchSizeB = getBatchSize(orgDimsB);
    int64_t batchSizeC = getBatchSize(origOutDims);

    int64_t newDimsA[3] = {batchSizeA, orgDimsA[rankA - 2],
                           orgDimsA[rankA - 1]};
    int64_t newDimsB[3] = {batchSizeB, orgDimsB[rankB - 2],
                           orgDimsB[rankB - 1]};
    int64_t newDimsOut[3] = {batchSizeC, origOutDims[outRank - 2],
                             origOutDims[outRank - 1]};
    if (batchSizeA != batchSizeB || batchSizeC != batchSizeB) {
      return op.emitError("cannot handle this broadcast for now");
    }

    assert(batchSizeA == batchSizeB && batchSizeB == batchSizeC &&
           "have to be like this for now");
    // Casting the original input into their new shape
    RankedTensorType newAType = RankedTensorType::get(newDimsA, elementTy);
    RankedTensorType newBType = RankedTensorType::get(newDimsB, elementTy);
    newOutType = RankedTensorType::get(newDimsOut, newOutElementTy);
    inA = reshapeToDimThree(rankA, newAType, inA);
    inB = reshapeToDimThree(rankB, newBType, inB);
  }

  auto init = arith::ConstantOp::create(rewriter, loc, newOutType,
                                        rewriter.getZeroAttr(newOutType))
                  .getResult();
  Value result = linalg::BatchMatmulOp::create(rewriter, loc, {inA, inB}, init)
                     .getResult(0);

  // Convert optional attributes
  if (auto attr = (*op).template getAttrOfType<StringAttr>("perf_config"))
    result.getDefiningOp()->setAttr("perf_config", attr);

  if (needToReshape) {
    // We have to reshape the output of linalg.batch_matmul to match the
    // original output in some cases. We have reshaped the input from before
    RankedTensorType finalResultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(origOutputTy));
    SmallVector<ReassociationIndices, 4> reasociation;
    result =
        (finalResultType.getRank() == 2)
            ? tensor::CollapseShapeOp::create(rewriter, loc, finalResultType,
                                              result, {{0, 1}, {2}})
                  .getResult()
            : tensor::ExpandShapeOp::create(rewriter, loc, finalResultType,
                                            result,
                                            getReassociationIndices(outRank));
  }

  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// One to One MIGraphX to Linalg Ops
//===----------------------------------------------------------------------===//
namespace {
template <class MIGraphXOp, class LinalgOp>
struct ElementwiseConverter final : public OpConversionPattern<MIGraphXOp> {
  using OpConversionPattern<MIGraphXOp>::OpConversionPattern;
  using OpConversionPattern<MIGraphXOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<MIGraphXOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(MIGraphXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

template <class MIGraphXOp, class LinalgOp>
LogicalResult ElementwiseConverter<MIGraphXOp, LinalgOp>::matchAndRewrite(
    MIGraphXOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Check that all operands are RankedTensorType
  auto operands = adaptor.getOperands();
  if (operands.size() == 0) {
    return op.emitError("cannot have zero operands");
  }

  RankedTensorType aType = cast<RankedTensorType>(operands[0].getType());
  // Check all operands have RankedTensorType and the same shape
  if (!llvm::all_of(operands, [&](Value v) {
        return isa<RankedTensorType>(v.getType()) &&
               cast<RankedTensorType>(v.getType()) == aType;
      })) {
    return op.emitError("all operands must have the same RankedTensorType");
  }

  RankedTensorType resultType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType());
  auto result = LinalgOp::create(rewriter, loc, operands, init);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// populateMIGraphXToLinalg* method
//===----------------------------------------------------------------------===//
void mlir::migraphx::populateMIGraphXToLinalgConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {
  patterns
      .add<DotConverter, ElementwiseConverter<migraphx::AddOp, linalg::AddOp>,
           ElementwiseConverter<migraphx::SubOp, linalg::SubOp>,
           ElementwiseConverter<migraphx::MulOp, linalg::MulOp>,
           ElementwiseConverter<migraphx::DivOp, linalg::DivOp>,
           ElementwiseConverter<migraphx::PowOp, linalg::PowFOp>,
           ElementwiseConverter<migraphx::AbsOp, linalg::AbsOp>,
           ElementwiseConverter<migraphx::CeilOp, linalg::CeilOp>,
           ElementwiseConverter<migraphx::ExpOp, linalg::ExpOp>,
           ElementwiseConverter<migraphx::FloorOp, linalg::FloorOp>,
           ElementwiseConverter<migraphx::LogOp, linalg::LogOp>,
           ElementwiseConverter<migraphx::NegOp, linalg::NegFOp>,
           ElementwiseConverter<migraphx::SqrtOp, linalg::SqrtOp>,
           ElementwiseConverter<migraphx::TanhOp, linalg::TanhOp>,
           ElementwiseConverter<migraphx::RecipOp, linalg::ReciprocalOp>>(
          converter, patterns.getContext());
}

void mlir::migraphx::populateMIGraphXFuncBoundaryToLinalgConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AsUnderlyingShapeConverter, AsLogicalShapeOpConverter>(
      typeConverter, patterns.getContext());
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
}
