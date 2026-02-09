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
  Type elementTy = cast<RankedTensorType>(inA.getType()).getElementType();
  auto origOutputTy = cast<migraphx::MIXRShapedType>(results[0].getType());
  Type outElementTy = origOutputTy.getElementType();
  Type newOutElementTy = getTypeConverter()->convertType(outElementTy);

  // check batch dimension. Tosa matmul only allow a single dimension for it,
  // add reshape ops to flatten and restore the original dimension.
  ArrayRef<int64_t> origOutDims = origOutputTy.getShape();
  RankedTensorType newOutType =
      RankedTensorType::get(origOutDims, newOutElementTy);
  size_t outRank = origOutDims.size();
  ArrayRef<int64_t> orgDimsA = cast<RankedTensorType>(inA.getType()).getShape();
  ArrayRef<int64_t> orgDimsB = cast<RankedTensorType>(inB.getType()).getShape();
  size_t rankA = orgDimsA.size();
  size_t rankB = orgDimsB.size();

  if (!cast<RankedTensorType>(inA.getType()).hasStaticShape() ||
      !cast<RankedTensorType>(inB.getType()).hasStaticShape()) {
    return op.emitError("only static shape is supported for now");
  }

  auto getReassociationIndices = [](int64_t rank) {
    assert(rank >= 3 && "this help only works for rank greater than 3");
    SmallVector<ReassociationIndices, 4> reassociation(3,
                                                       ReassociationIndices());
    reassociation[0].insert(reassociation[0].begin(), rank - 2, 0);
    std::iota(reassociation[0].begin(), reassociation[0].end(), 0);
    reassociation[1] = {rank - 2};
    reassociation[2] = {rank - 1};
    return reassociation;
  };

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
    if (batchSizeA != batchSizeB || batchSizeC != batchSizeB) {
      return op.emitError("cannot handle this broadcast for now");
    }

    assert(batchSizeA == batchSizeB && batchSizeB == batchSizeC &&
           "have to be like this for now");
    RankedTensorType newAType = RankedTensorType::get(newDimsA, elementTy);
    RankedTensorType newBType = RankedTensorType::get(newDimsB, elementTy);
    newOutType = RankedTensorType::get(newDimsOut, newOutElementTy);
    inA = (rankA == 2)
              ? tensor::ExpandShapeOp::create(rewriter, loc, newAType, inA,
                                              {{0, 1}, {2}})
                    .getResult()
              : tensor::CollapseShapeOp::create(rewriter, loc, newAType, inA,
                                                getReassociationIndices(rankA))
                    .getResult();
    inB = (rankB == 2)
              ? tensor::ExpandShapeOp::create(rewriter, loc, newBType, inB,
                                              {{0, 1}, {2}})
                    .getResult()
              : tensor::CollapseShapeOp::create(rewriter, loc, newBType, inB,
                                                getReassociationIndices(rankB));
  }

  auto init = arith::ConstantOp::create(rewriter, loc, newOutType,
                                        rewriter.getZeroAttr(newOutType))
                  .getResult();
  Value result = linalg::BatchMatmulOp::create(rewriter, loc, {inA, inB}, init)
                     .getResult(0);

  // Convert optional attributes
  if (auto attr = (*op).template getAttrOfType<StringAttr>("perf_config"))
    result.getDefiningOp()->setAttr("perf_config", attr);

  if (outRank != 3 || rankA != rankB ||
      (outRank == 3 && orgDimsA[0] != orgDimsB[0])) {
    RankedTensorType finalResultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(origOutputTy));
    SmallVector<ReassociationIndices, 4> reasociation;
    finalResultType.dump();
    result =
        (finalResultType.getRank() == 2)
            ? tensor::CollapseShapeOp::create(rewriter, loc, finalResultType,
                                              result, {{0, 1}, {2}})
                  .getResult()
            : tensor::ExpandShapeOp::create(rewriter, loc, finalResultType,
                                            result,
                                            getReassociationIndices(outRank));
    rewriter.replaceOp(op, result);
    return success();
  }
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// populateMIGraphXToLinalg* method
//===----------------------------------------------------------------------===//
void mlir::migraphx::populateMIGraphXToLinalgConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<DotConverter>(converter, patterns.getContext());
}

void mlir::migraphx::populateMIGraphXFuncBoundaryToLinalgConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AsUnderlyingShapeConverter, AsLogicalShapeOpConverter>(
      typeConverter, patterns.getContext());
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
}
