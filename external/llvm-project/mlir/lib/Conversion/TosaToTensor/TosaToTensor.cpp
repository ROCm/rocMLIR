//===- TosaToTensor.cpp - Lowering Tosa to Tensor Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Tensor dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace tosa;

static bool findIntermediateShape(ArrayRef<int64_t> lhsShape,
                                  ArrayRef<int64_t> rhsShape,
                                  SmallVector<int64_t> &intermediateShape,
                                  bool isDynamic) {
  if (isDynamic) {
    // TODO (natashaknk): Make dynamic intermediate shape not always be rank-1
    intermediateShape = {-1};
    return true;
  }

  if (lhsShape.empty() || rhsShape.empty()) {
    intermediateShape = {};
    return true;
  }

  unsigned currLhsDim = 0, currRhsDim = 0;
  while (currLhsDim < lhsShape.size() && currRhsDim < rhsShape.size()) {
    int64_t rhsSize = rhsShape[currRhsDim];
    int64_t lhsSize = lhsShape[currLhsDim];
    while (lhsSize != rhsSize && currLhsDim < lhsShape.size() &&
           currRhsDim < rhsShape.size()) {
      if (lhsSize < rhsSize) {
        currLhsDim++;
        lhsSize *= lhsShape[currLhsDim];
      } else {
        currRhsDim++;
        rhsSize *= rhsShape[currRhsDim];
      }
    }
    if (lhsSize == rhsSize) {
      intermediateShape.push_back(lhsSize);
    }
    currRhsDim++;
    currLhsDim++;
  }

  // If the iterators didn't reach the end and their leftover dimensions are not
  // equal to 1 an intermediate shape was not found.
  while (currLhsDim < lhsShape.size()) {
    if (lhsShape[currLhsDim++] != 1) {
      return false;
    }
  }

  while (currRhsDim < rhsShape.size()) {
    if (rhsShape[currRhsDim++] != 1) {
      return false;
    }
  }

  return true;
}

static bool createReassociationMapsForCollapse(
    PatternRewriter &rewriter, ArrayRef<int64_t> srcShape,
    ArrayRef<int64_t> dstShape,
    SmallVector<ReassociationExprs, 4> &reassociationMap, bool isDynamic) {

  // If the shape is dynamic, create a map for collapsing into one dimension.
  if (isDynamic) {
    SmallVector<AffineExpr, 2> exprs;
    for (int i = 0, s = srcShape.size(); i < s; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(i));
    reassociationMap = {exprs};
    return true;
  }

  if (dstShape.empty()) {
    reassociationMap = {};
    return true;
  }

  reassociationMap.resize(dstShape.size());
  unsigned currSrcDim = 0, currDstDim = 0;
  while (currSrcDim < srcShape.size() && currDstDim < dstShape.size()) {
    int64_t dstSize = dstShape[currDstDim];
    int64_t srcSize = srcShape[currSrcDim];
    while (srcSize < dstSize && currSrcDim < srcShape.size()) {
      reassociationMap[currDstDim].push_back(
          rewriter.getAffineDimExpr(currSrcDim++));
      srcSize *= srcShape[currSrcDim];
    }
    if (srcSize == dstSize) {
      reassociationMap[currDstDim].push_back(
          rewriter.getAffineDimExpr(currSrcDim++));
      // If the next dim in collapsedShape is not 1, treat subsequent dims in
      // expandedShape which are 1 to be collapsed.
      if (currDstDim == dstShape.size() - 1 || dstShape[currDstDim + 1] != 1) {
        while (currSrcDim < srcShape.size() && srcShape[currSrcDim] == 1) {
          reassociationMap[currDstDim].push_back(
              rewriter.getAffineDimExpr(currSrcDim++));
        }
      }
    }
    currDstDim++;
  }

  // If both iterators didn't reach the end, we have leftover dimentions which
  // implies that we have a mismatch in shape.
  return currSrcDim == srcShape.size() && currDstDim == dstShape.size();
}

namespace {
class ReshapeConverterCollapse : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ShapedType operandTy = adaptor.getInput1().getType().cast<ShapedType>();
    ShapedType resultTy = reshape.getType().template cast<ShapedType>();
    bool isDynamic = !operandTy.hasStaticShape();

    if (isDynamic && resultTy.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          reshape, "Cannot collapse dynamic dims to more than one dimension");
    }

    if (operandTy == resultTy) {
      rewriter.replaceOp(reshape, adaptor.getOperands()[0]);
      return success();
    }

    SmallVector<ReassociationExprs, 4> reassociationMap;
    if (!createReassociationMapsForCollapse(rewriter, operandTy.getShape(),
                                            resultTy.getShape(),
                                            reassociationMap, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape,
          "tosa.reshape Attempting to collapse into an incompatible shape");
    }

    SmallVector<int64_t> intermediateShape;
    if (!findIntermediateShape(operandTy.getShape(), resultTy.getShape(),
                               intermediateShape, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape, "tosa.reshape Cannot collapse into given shape");
    }

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        reshape, resultTy, adaptor.getOperands()[0], reassociationMap);
    return success();
  }
};

class ReshapeConverterExpand : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ShapedType operandTy = adaptor.getInput1().getType().cast<ShapedType>();
    ShapedType resultTy = reshape.getType().template cast<ShapedType>();
    bool isDynamic = !operandTy.hasStaticShape();

    if (operandTy == resultTy) {
      rewriter.replaceOp(reshape, adaptor.getOperands()[0]);
      return success();
    }

    if (isDynamic && operandTy.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          reshape, "Cannot expand dynamic dims from more than one dimension");
    }

    SmallVector<ReassociationExprs, 4> reassociationMap;
    if (!createReassociationMapsForCollapse(rewriter, resultTy.getShape(),
                                            operandTy.getShape(),
                                            reassociationMap, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape,
          "tosa.reshape Attempting to expand into an incompatible shape");
    }

    SmallVector<int64_t> intermediateShape;
    if (!findIntermediateShape(operandTy.getShape(), resultTy.getShape(),
                               intermediateShape, isDynamic) ||
        intermediateShape != operandTy.getShape()) {
      return rewriter.notifyMatchFailure(
          reshape, "tosa.reshape Cannot expand into given shape");
    }
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        reshape, resultTy, adaptor.getOperands()[0], reassociationMap);
    return success();
  }
};

class ReshapeConverterCollapseExpand
    : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ShapedType operandTy = adaptor.getInput1().getType().cast<ShapedType>();
    ShapedType resultTy = reshape.getType().template cast<ShapedType>();
    bool isDynamic = !operandTy.hasStaticShape();

    if (operandTy == resultTy) {
      rewriter.replaceOp(reshape, adaptor.getOperands()[0]);
      return success();
    }

    SmallVector<int64_t> intermediateShape;
    if (!findIntermediateShape(resultTy.getShape(), operandTy.getShape(),
                               intermediateShape, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape, "tosa.reshape Cannot identify an intermediate shape between "
                   "the given two shapes");
    }

    Value collapse = rewriter.create<tosa::ReshapeOp>(
        reshape.getLoc(),
        RankedTensorType::get(intermediateShape,
                              reshape.getType().getElementType()),
        adaptor.getInput1());
    Value expand =
        rewriter.create<tosa::ReshapeOp>(reshape.getLoc(), resultTy, collapse);
    rewriter.replaceOp(reshape, expand);

    return success();
  }
};

class SliceOpConverter : public OpConversionPattern<tosa::SliceOp> {
public:
  using OpConversionPattern<tosa::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::SliceOp sliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = sliceOp.getLoc();
    Value input = adaptor.getInput();
    SmallVector<int64_t> strides;
    auto starts = sliceOp.getStart();
    auto sizes = sliceOp.getSize();
    strides.resize(sliceOp.getType().template cast<ShapedType>().getRank(), 1);

    SmallVector<Value> dynSizes;
    for (const auto &i : llvm::enumerate(sizes)) {
      int64_t size = i.value().cast<IntegerAttr>().getInt();
      size_t index = i.index();
      if (size != ShapedType::kDynamicSize)
        continue;

      auto dim = rewriter.create<tensor::DimOp>(loc, input, index);
      auto offset = rewriter.create<arith::ConstantOp>(
          loc,
          rewriter.getIndexAttr(starts[index].cast<IntegerAttr>().getInt()));
      dynSizes.push_back(rewriter.create<arith::SubIOp>(loc, dim, offset));
    }

    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(), sliceOp.getType(), input, ValueRange({}), dynSizes,
        ValueRange({}), starts, sizes, rewriter.getI64ArrayAttr(strides));

    rewriter.replaceOp(sliceOp, newSliceOp.getResult());
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToTensorConversionPatterns(
    RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
      ReshapeConverterCollapse,
      ReshapeConverterExpand,
      ReshapeConverterCollapseExpand,
      SliceOpConverter>(patterns->getContext());
  // clang-format on
}
