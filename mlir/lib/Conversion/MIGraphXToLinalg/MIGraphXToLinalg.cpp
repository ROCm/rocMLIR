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
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
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

namespace {
struct ConvConverter final
    : public OpConversionPattern<migraphx::ConvolutionOp> {
  using OpConversionPattern<migraphx::ConvolutionOp>::OpConversionPattern;
  using OpConversionPattern<migraphx::ConvolutionOp>::getTypeConverter;
  using OpAdaptor =
      typename OpConversionPattern<migraphx::ConvolutionOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(migraphx::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  LogicalResult emitConv(ConversionPatternRewriter &rewriter,
                         migraphx::ConvolutionOp op, Value input,
                         Value filter) const;
};
} // namespace

// Nice helper function for the linalg.generic op region
static void convBodyBuilder(OpBuilder &b, Location loc, ValueRange blockArgs) {
  Value inputVal = blockArgs[0];
  Value filterVal = blockArgs[1];
  Value outputVal = blockArgs[2];
  Value mul = arith::MulFOp::create(b, loc, inputVal, filterVal);
  Value add = arith::AddFOp::create(b, loc, outputVal, mul);
  linalg::YieldOp::create(b, loc, add);
}

/// Emit convolution attributes on the newly created operation.
static void emitConvAttributes(migraphx::ConvolutionOp op, Value convOp,
                               Attribute strides, Attribute dilation,
                               Attribute pad, Attribute convOpName) {
  Operation *newOp = convOp.getDefiningOp();
  newOp->setAttr("pad", pad);
  newOp->setAttr("group", op.getGroupAttr());
  newOp->setAttr("stride", strides);
  newOp->setAttr("dilation", dilation);

  // Convert optional attributes
  if (auto attr = (*op).template getAttrOfType<StringAttr>("perf_config"))
    newOp->setAttr("perf_config", attr);
  newOp->setAttr("conv_op", convOpName);
}

/// Emit a grouped convolution of any spatial rank (1D, 2D, or 3D).
/// Input shape: (batch, group, channel, spatial...),
/// filter shape: (group, k, channel, kernel_spatial...)
///
/// clang-format off
///   for n in batch:
///     for g in group:
///       for k in outChannels:
///         for oh_0 in output_spatial_0:
///           for oh_1 in output_spatial_1:
///             // ...
///             for oh_{dim-1} in output_spatial_{dim-1}:
///               for c in channels:                          // reduction
///                 for kh_0 in kernel_spatial_0:              // reduction
///                   for kh_1 in kernel_spatial_1:            // reduction
///                     // ...
/// clang-format on
static Value emitGroupedConv(ConversionPatternRewriter &rewriter, Location loc,
                             RankedTensorType resultType, Value input,
                             Value filter, Value zero, ArrayAttr strides,
                             ArrayAttr dilation) {
  MLIRContext *ctx = rewriter.getContext();
  int64_t dim = cast<RankedTensorType>(input.getType()).getRank() - 3;
  SmallVector<int64_t, 4> strideVals;
  SmallVector<int64_t, 4> dilationVals;
  llvm::transform(
      strides.getValue(), std::back_inserter(strideVals),
      [](Attribute attr) { return cast<IntegerAttr>(attr).getInt(); });
  llvm::transform(
      dilation.getValue(), std::back_inserter(dilationVals),
      [](Attribute attr) { return cast<IntegerAttr>(attr).getInt(); });

  // Iteration domain layout:
  //   parallel:  batch, group, filter, oh_0 .. oh_{dim-1}
  //   reduction: channel, kh_0 .. kh_{dim-1}
  const int64_t nonDimensionalIterationCount =
      4; // includes the batch, group, filter, and channel dimensions
  const int64_t totalDims = nonDimensionalIterationCount +
                            2 * dim; // The two comes from the fact that we are
                                     // iterating both the filters and inputs
  SmallVector<AffineExpr> d;
  for (int64_t i = 0; i < totalDims; ++i)
    d.push_back(getAffineDimExpr(i, ctx));

  AffineExpr batch = d[0], group = d[1], outChannel = d[2];
  AffineExpr inChannel = d[3 + dim];

  SmallVector<AffineExpr> inputExprs = {batch, group, inChannel};
  for (int64_t i = 0; i < dim; ++i) {
    // see the comment above to see what this is referring to
    AffineExpr oh_i = d[3 + i];
    AffineExpr kh_i = d[4 + dim + i];
    inputExprs.push_back(oh_i * strideVals[i] + kh_i * dilationVals[i]);
  }

  SmallVector<AffineExpr> filterExprs = {group, outChannel, inChannel};
  for (int64_t i = 0; i < dim; ++i) {
    // see the comment above to see what this is referring to
    AffineExpr kh_i = d[4 + dim + i];
    filterExprs.push_back(kh_i);
  }

  SmallVector<AffineExpr> outputExprs = {batch, group, outChannel};
  for (int64_t i = 0; i < dim; ++i) {
    // see the comment above to see what this is referring to
    AffineExpr oh_i = d[3 + i];
    outputExprs.push_back(oh_i);
  }

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(totalDims, 0, inputExprs, ctx),
      AffineMap::get(totalDims, 0, filterExprs, ctx),
      AffineMap::get(totalDims, 0, outputExprs, ctx)};

  SmallVector<utils::IteratorType> iteratorTypes(
      3 + dim, // The 3 comes from batch, filter, and group dimensions
      utils::IteratorType::parallel);
  // The one comes from the channel as the reduction iterator types
  iteratorTypes.append(1 + dim, utils::IteratorType::reduction);

  return linalg::GenericOp::create(rewriter, loc, resultType,
                                   ValueRange{input, filter}, zero,
                                   indexingMaps, iteratorTypes, convBodyBuilder)
      .getResult(0);
}

LogicalResult ConvConverter::emitConv(ConversionPatternRewriter &rewriter,
                                      migraphx::ConvolutionOp op, Value input,
                                      Value filter) const {
  // Input and filter are already in NGC* and GKC* form (group dimension
  // expanded). Build the result type as NGK* (with explicit G), emit the
  // grouped linalg conv (1D/2D/3D), then collapse back to NK* for the type
  // converter.
  Location loc = op.getLoc();
  int64_t group = op.getGroupAttr().getInt();
  int64_t dim = cast<RankedTensorType>(input.getType()).getRank() -
                3; // exclude batch (N), group (G), channel (C)
  assert(dim >= 1 && dim <= 3 && "this should be checked at matchAndRewrite");

  // Result type from the op is NK*; expand to NGK* for the linalg conv.
  RankedTensorType resultType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getResult()));
  ArrayRef<int64_t> resultShape = resultType.getShape();
  int64_t n = resultType.getDimSize(0);
  int64_t newK = resultType.getDimSize(1) / group;
  assert(resultType.getDimSize(1) % group == 0 &&
         "output channel must be divisible by group");
  SmallVector<int64_t, 4> newShape{n, group, newK};
  newShape.insert(newShape.end(), std::next(resultShape.begin(), 2),
                  resultShape.end());
  auto newResultType =
      RankedTensorType::get(newShape, resultType.getElementType());
  Value zero = arith::ConstantOp::create(rewriter, loc, newResultType,
                                         rewriter.getZeroAttr(newResultType));

  ArrayAttr strides = op.getStride();
  ArrayAttr dilation = op.getDilation();

  rock::LinalgConvType convLayout =
      (dim == 1)   ? rock::LinalgConvType::Conv1dNgchGkch
      : (dim == 2) ? rock::LinalgConvType::Conv2dNgchwGkchw
                   : rock::LinalgConvType::Conv3dNgchwdGkchwd;
  auto resultConvOpName =
      rock::LinalgConvTypeAttr::get(rewriter.getContext(), convLayout);
  Value result = emitGroupedConv(rewriter, loc, newResultType, input, filter,
                                 zero, strides, dilation);

  emitConvAttributes(op, result, strides, dilation, op.getPaddingAttr(),
                     resultConvOpName);

  // we must reshape the operand to what the type converter expects
  SmallVector<ReassociationIndices, 4> reassociation{{0}, {1, 2}};
  llvm::for_each(llvm::seq<int64_t>(3, dim + 3),
                 [&](int64_t index) { reassociation.push_back({index}); });
  auto finalResult =
      tensor::CollapseShapeOp::create(rewriter, loc, result, reassociation);

  rewriter.replaceOp(op, finalResult);
  return success();
}

/// Expand the channel dimension of `input` into (group, channel_per_group).
/// For a filter:
///     if (isFilter == true):  KCHW  -> GKCHW
/// For an input  (isFilter == false): NCHW -> NGCHW
static Value expandGroupDim(ConversionPatternRewriter &rewriter, Location loc,
                            Value input, bool isFilter, int64_t group,
                            int64_t dim) {
  RankedTensorType originalType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> originalShape = originalType.getShape();
  SmallVector<int64_t, 4> newShape;

  if (isFilter) {
    int64_t newK = originalType.getDimSize(0) / group;
    assert(originalType.getDimSize(0) % group == 0 &&
           "output channel must be divisible by group");
    newShape.push_back(group);
    newShape.push_back(newK);
    newShape.push_back(originalType.getDimSize(1));
    newShape.insert(newShape.end(), std::next(originalShape.begin(), 2),
                    originalShape.end());
    RankedTensorType newType =
        RankedTensorType::get(newShape, originalType.getElementType());

    SmallVector<ReassociationIndices, 4> reassociation;
    reassociation.push_back({0, 1});
    llvm::for_each(llvm::seq<int64_t>(2, dim + 3),
                   [&](int64_t i) { reassociation.push_back({i}); });
    return tensor::ExpandShapeOp::create(rewriter, loc, newType, input,
                                         reassociation);
  }

  int64_t newC = originalType.getDimSize(1) / group;
  assert(originalType.getDimSize(1) % group == 0 &&
         "input channel must be divisible by group");
  newShape.push_back(originalType.getDimSize(0));
  newShape.push_back(group);
  newShape.push_back(newC);
  newShape.insert(newShape.end(), std::next(originalShape.begin(), 2),
                  originalShape.end());

  RankedTensorType newType =
      RankedTensorType::get(newShape, originalType.getElementType());
  SmallVector<ReassociationIndices, 4> reassociation;
  reassociation.push_back({0});
  reassociation.push_back({1, 2});
  llvm::for_each(llvm::seq<int64_t>(3, dim + 3),
                 [&](int64_t i) { reassociation.push_back({i}); });
  return tensor::ExpandShapeOp::create(rewriter, loc, newType, input,
                                       reassociation);
}

/// Apply symmetric padding to the spatial dimensions of `input` when any
/// padding value in `padAttr` is non-zero.  Returns the (possibly padded)
/// input.
static Value applyConvPadding(ConversionPatternRewriter &rewriter, Location loc,
                              Value input, ArrayAttr padAttr, int64_t dim) {
  if (llvm::all_of(padAttr, [](Attribute pad) {
        return cast<IntegerAttr>(pad).getValue() == 0;
      }))
    return input;

  RankedTensorType inputType = cast<RankedTensorType>(input.getType());
  SmallVector<OpFoldResult, 4> low(inputType.getRank(),
                                   rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult, 4> high(inputType.getRank(),
                                    rewriter.getIndexAttr(0));
  assert(2 * dim == (int64_t)padAttr.size() && "padding is symmetric");

  // MIGraphX padAttr is [dim0_low, dim1_low,..., dim0_high, dim1_high, ...]
  SmallVector<int64_t, 4> newShape(inputType.getShape());
  auto lowAttrs = padAttr.getValue().drop_back(dim);
  auto highAttrs = padAttr.getValue().drop_front(dim);
  //  The first spatial dimension (H) is always located at index 2 in the
  //  NC* layout (after batch and channel), regardless of convolution rank.
  int64_t dimHOffset = 2;
  llvm::for_each(llvm::seq<int64_t>(dim), [&](int64_t index) {
    int64_t lowPad = cast<IntegerAttr>(lowAttrs[index]).getInt();
    int64_t highPad = cast<IntegerAttr>(highAttrs[index]).getInt();
    newShape[dimHOffset + index] += lowPad + highPad;
    low[dimHOffset + index] = rewriter.getIndexAttr(lowPad);
    high[dimHOffset + index] = rewriter.getIndexAttr(highPad);
  });

  RankedTensorType newInputType =
      RankedTensorType::get(newShape, inputType.getElementType());
  Value padValue = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(inputType.getElementType()));
  return tensor::PadOp::create(rewriter, loc, newInputType, input, low, high,
                               padValue)
      .getResult();
}

LogicalResult
ConvConverter::matchAndRewrite(migraphx::ConvolutionOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  // Forward convolution is lowered in three steps:
  // 1. Apply padding to the input when the op has non-zero padding.
  // 2. Expand the channel dimension into (group, channel_per_group),
  // introducing
  //    a group dimension G. Input becomes NGC* (e.g. NGCL, NGCHW, NGCDHW) and
  //    filter becomes GKC* (e.g. GKCL, GKCHW, GKCDHW), matching the group attr.
  // 3. Emit the grouped linalg convolution (1D/2D/3D), then collapse the
  //    result back to the original NFHW/NFDHW shape for the type converter.
  Location loc = op.getLoc();
  Value input = adaptor.getInput();
  Value filter = adaptor.getFilter();
  ArrayAttr padAttr = adaptor.getPaddingAttr();
  RankedTensorType inputType = cast<RankedTensorType>(input.getType());
  int64_t dim = inputType.getRank() - 2;
  int64_t group = op.getGroupAttr().getInt();

  if (dim > 3 || dim < 1) {
    return op.emitError(Twine(dim) + "D conv is not supported for now");
  }

  Type inputElementType = inputType.getElementType();
  // For now, the linalg.generic region only supports floating point of the same
  // type.
  if (!llvm::all_of(
          op.getOperandTypes(),
          [&](Type type) {
            assert(isa<migraphx::MIXRShapedType>(type) &&
                   "Convolution must have migraphx::MIXRShapedType");
            return inputElementType ==
                   cast<migraphx::MIXRShapedType>(type).getElementType();
          }) ||
      op.getResult().getType().getElementType() != inputElementType) {
    return op.emitError(
        "all operands and outputs must be floating-point values");
  }

  // Step 1: apply padding when any padding value is non-zero.
  input = applyConvPadding(rewriter, loc, input, padAttr, dim);

  // Step 2: expand group dimension (NCHW -> NGCHW, KCHW -> GKCHW). In theory,
  // one can have an implementation where you don't expand the group dimension
  // to compute convolution with group attribute greater than > 1 (emitting
  // multiple conv2d convolution and concatenating it). Expanding group
  // dimension makes linalg.generic a lot easier to implement, and hence why it
  // is done this way. Also, we don't emit special ops like
  // (conv_2d_nchw_fchw, conv_1d_ncw_fcw, etc) for cases when G=1, because we
  // want to be consistent and make it easier to Linalg to Rock lowering.
  input = expandGroupDim(rewriter, loc, input, /*isFilter=*/false, group, dim);
  filter = expandGroupDim(rewriter, loc, filter, /*isFilter=*/true, group, dim);

  // Step 3: emit linalg conv and collapse result to match type converter.
  return emitConv(rewriter, op, input, filter);
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
// Other elementwise operations
//===----------------------------------------------------------------------===//
namespace {
struct ReluConverter final : public OpConversionPattern<migraphx::ReluOp> {
  using OpConversionPattern<migraphx::ReluOp>::OpConversionPattern;
  using OpConversionPattern<migraphx::ReluOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<migraphx::ReluOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(migraphx::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Used by GenericElementwiseOpConverter:
/// - isValidGenericElementwiseOp: return ture if the operation is valid for the
/// generic elementwise converter
/// - elementwiseBodyBuilder: build the linalg.generic body for the operation
/// These method should be provided through partial specialization. As of
/// current, it is highly likely that you will error out if you don't provide
/// this trait.
template <typename ElementwiseOp>
struct GenericElementwiseTrait {};

/// The GenericElementwiseOpConverter is a template class that is used to
/// convert all elementwise operations (i.e. all iterator_types are parallel).
/// It takes in a GenericElementwiseTrait by partial specialization to check if
/// the operations is valid, and if it is valid, how to construct the
/// linalg.generic body.
template <typename ElementwiseOp>
struct GenericElementwiseOpConverter final
    : public OpConversionPattern<ElementwiseOp> {
  using OpConversionPattern<ElementwiseOp>::OpConversionPattern;
  using OpConversionPattern<ElementwiseOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<ElementwiseOp>::OpAdaptor;

  GenericElementwiseOpConverter(const TypeConverter &typeConverter,
                                MLIRContext *context)
      : OpConversionPattern<ElementwiseOp>(typeConverter, context),
        loweringTrait(GenericElementwiseTrait<ElementwiseOp>()) {}

  LogicalResult
  matchAndRewrite(ElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  GenericElementwiseTrait<ElementwiseOp> loweringTrait;
};
} // namespace

// Generic elementwise precondition checks and body builders
template <>
struct GenericElementwiseTrait<migraphx::SigmoidOp> {
  static bool isValidGenericElementwiseOp(Operation *op) {
    // most of these checks are done by the verifier
    return true;
  }

  static void elementwiseBodyBuilder(OpBuilder &builder, Location loc,
                                     ValueRange inputs) {
    Value x = inputs[0];
    assert(x.getType().isFloat() && "verifier should have checked this!");
    auto getOne = [&]() {
      assert(x.getType().isFloat() && "only support floating point for now");
      return arith::ConstantOp::create(builder, loc, x.getType(),
                                       builder.getFloatAttr(x.getType(), 1));
    };
    Value negX = arith::NegFOp::create(builder, loc, x);
    Value expNegX = math::ExpOp::create(builder, loc, negX.getType(), negX);
    Value denominator =
        arith::AddFOp::create(builder, loc, getOne(), expNegX).getResult();
    Value sigmoid =
        arith::DivFOp::create(builder, loc, getOne(), denominator).getResult();
    linalg::YieldOp::create(builder, loc, sigmoid);
  }
};

template <>
struct GenericElementwiseTrait<migraphx::WhereOp> {
  static bool isValidGenericElementwiseOp(Operation *op) {
    // traits in where op already checked for most of these cases
    return true;
  }

  static void elementwiseBodyBuilder(OpBuilder &builder, Location loc,
                                     ValueRange inputs) {
    Value cond = inputs[0];
    Value inA = inputs[1];
    Value inB = inputs[2];

    IntegerType condShape = dyn_cast<IntegerType>(cond.getType());
    assert(condShape && "should be checked in verifier");
    Value castedCond = convertScalarToDtype(
        builder, loc, cond, builder.getI1Type(), /*isUnsignedCast=*/false);
    Value result = arith::SelectOp::create(builder, loc, castedCond, inA, inB);
    linalg::YieldOp::create(builder, loc, result);
  }
};

template <typename ElementwiseOp>
LogicalResult GenericElementwiseOpConverter<ElementwiseOp>::matchAndRewrite(
    ElementwiseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  ValueRange inputs = adaptor.getOperands();
  RankedTensorType resultType = dyn_cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult()));
  assert(resultType &&
         "The TypeConverter should convert type into RankedTensorType");

  if (llvm::any_of(inputs.getTypes(), [&](Type current) {
        assert(isa<RankedTensorType>(current) &&
               "inputs should be RankedTensorType");
        RankedTensorType casted = dyn_cast<RankedTensorType>(current);
        return casted.getShape() != resultType.getShape();
      })) {
    return op.emitError("expect all inputs and outputs to have the same shape");
  }

  if (!loweringTrait.isValidGenericElementwiseOp(op))
    return failure();

  int64_t rank = resultType.getRank();
  SmallVector<AffineMap> indexingMaps(inputs.size() + op->getNumResults(),
                                      rewriter.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  Value outputs = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                          resultType.getElementType());
  auto result = linalg::GenericOp::create(
      rewriter, loc, TypeRange{resultType}, inputs, ValueRange{outputs},
      indexingMaps, iteratorTypes, loweringTrait.elementwiseBodyBuilder);
  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult
ReluConverter::matchAndRewrite(migraphx::ReluOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  if (adaptor.getOperands().size() != 1) {
    return op.emitError("only expected one operand");
  }

  RankedTensorType resultType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getResult()));
  Location loc = op.getLoc();
  Value in = adaptor.getInA();
  Value zero = arith::ConstantOp::create(rewriter, loc, resultType,
                                         rewriter.getZeroAttr(resultType));
  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType());

  // relu(x) = max(0, x)
  auto result = linalg::MaxOp::create(rewriter, loc, {in, zero}, init);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// elementwise boolean operator
//===----------------------------------------------------------------------===//

namespace {
template <class MIXRBooleanOp>
struct BooleanElementwiseConverter : public OpConversionPattern<MIXRBooleanOp> {
  using OpConversionPattern<MIXRBooleanOp>::OpConversionPattern;
  using OpConversionPattern<MIXRBooleanOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<MIXRBooleanOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(MIXRBooleanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  constexpr std::enable_if_t<std::is_same_v<MIXRBooleanOp, migraphx::Greater> ||
                                 std::is_same_v<MIXRBooleanOp, migraphx::Equal>,
                             arith::CmpIPredicate>
  getIPredicate() const;

  constexpr std::enable_if_t<std::is_same_v<MIXRBooleanOp, migraphx::Greater> ||
                                 std::is_same_v<MIXRBooleanOp, migraphx::Equal>,
                             arith::CmpFPredicate>
  getFPredicate() const;
};
} // namespace

template <class MIXRBooleanOp>
constexpr std::enable_if_t<std::is_same_v<MIXRBooleanOp, migraphx::Greater> ||
                               std::is_same_v<MIXRBooleanOp, migraphx::Equal>,
                           arith::CmpIPredicate>
BooleanElementwiseConverter<MIXRBooleanOp>::getIPredicate() const {
  if (std::is_same_v<MIXRBooleanOp, migraphx::Greater>) {
    return arith::CmpIPredicate::sgt;
  }

  return arith::CmpIPredicate::eq;
}

template <class MIXRBooleanOp>
constexpr std::enable_if_t<std::is_same_v<MIXRBooleanOp, migraphx::Greater> ||
                               std::is_same_v<MIXRBooleanOp, migraphx::Equal>,
                           arith::CmpFPredicate>
BooleanElementwiseConverter<MIXRBooleanOp>::getFPredicate() const {
  if (std::is_same_v<MIXRBooleanOp, migraphx::Greater>) {
    return arith::CmpFPredicate::OGT;
  }

  return arith::CmpFPredicate::OEQ;
}

template <class MIXRBooleanOp>
LogicalResult BooleanElementwiseConverter<MIXRBooleanOp>::matchAndRewrite(
    MIXRBooleanOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value a = adaptor.getInA();
  Value b = adaptor.getInB();
  RankedTensorType resultType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getResult()));
  int64_t rank = resultType.getRank();
  Value emptyTensor = tensor::EmptyOp::create(rewriter, loc, resultType,
                                              /*dynamic_sizes=*/ValueRange{});
  SmallVector<AffineMap> indexingMaps(3, rewriter.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);

  // the block for the linalg generic
  auto buildLinalgInnerBlock = [&](OpBuilder &b, Location loc,
                                   ValueRange blockArgs) {
    Value first = blockArgs[0];
    Value second = blockArgs[1];
    Value cmp =
        (first.getType().isInteger())
            ? arith::CmpIOp::create(b, loc, getIPredicate(), first, second)
                  .getResult()
            : arith::CmpFOp::create(b, loc, getFPredicate(), first, second)
                  .getResult();

    Location cmpLoc = cmp.getLoc();
    Type yieldType = resultType.getElementType();
    // migraphx expects the result type to have the same type as the input. So
    // we must cast the cmp result into the desired type.
    Value result =
        (first.getType().isInteger())
            ? arith::ExtUIOp::create(b, cmpLoc, yieldType, cmp).getResult()
            : arith::UIToFPOp::create(b, cmpLoc, yieldType, cmp).getResult();
    linalg::YieldOp::create(b, loc, result);
  };

  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, resultType, ValueRange{a, b}, emptyTensor, indexingMaps,
      iteratorTypes, buildLinalgInnerBlock);
  rewriter.replaceOp(op, genericOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//
namespace {
struct ClipConverter final : public OpConversionPattern<migraphx::ClipOp> {
  using OpConversionPattern<migraphx::ClipOp>::OpConversionPattern;
  using OpConversionPattern<migraphx::ClipOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<migraphx::ClipOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(migraphx::ClipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
ClipConverter::matchAndRewrite(migraphx::ClipOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value x = adaptor.getX();
  Value minVals = adaptor.getMinVals();
  Value maxVals = adaptor.getMaxVals();
  RankedTensorType outType = dyn_cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));
  if (!outType) {
    return op.emitError("expected a RankedTensorType type");
  }

  if (outType != adaptor.getMaxVals().getType() ||
      maxVals.getType() != x.getType() || x.getType() != minVals.getType()) {
    return op.emitError("expected all operands and result type to be the same");
  }

  // clip(x, min, max) = min(max(x, minvals), maxvals)
  Value initOne = tensor::EmptyOp::create(rewriter, loc, outType.getShape(),
                                          outType.getElementType());
  Value initTwo = tensor::EmptyOp::create(rewriter, loc, outType.getShape(),
                                          outType.getElementType());
  Value atLeastMin =
      linalg::MaxOp::create(rewriter, loc, {x, minVals}, initOne).getResult(0);
  auto result =
      linalg::MinOp::create(rewriter, loc, {atLeastMin, maxVals}, initTwo);
  rewriter.replaceOp(op, result);
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

struct ReshapeConverter final
    : public OpConversionPattern<migraphx::ReshapeOp> {
  using OpConversionPattern<migraphx::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct TransposeConverter final
    : public OpConversionPattern<migraphx::TransposeOp> {
  using OpConversionPattern<migraphx::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
TransposeConverter::matchAndRewrite(migraphx::TransposeOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  RankedTensorType outputType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  assert(outputType && "MIXRShapedToTensorConverter TypeConverter should "
                       "convert this into a RankedTensorType");
  auto init = tensor::EmptyOp::create(rewriter, loc, outputType, {});
  SmallVector<int64_t, 4> permutation;
  llvm::transform(
      op.getPermutation().getValue(), std::back_inserter(permutation),
      [](Attribute attr) { return cast<IntegerAttr>(attr).getInt(); });
  auto result = linalg::TransposeOp::create(rewriter, loc, adaptor.getInput(),
                                            init, permutation);
  rewriter.replaceOp(op, result);
  return success();
}
/// Reshape the input Value into a new RankedTensorType with newShape
/// The input must have type RankedTensorType.
static Value reshapeValue(ConversionPatternRewriter &rewriter, Value input,
                          ArrayRef<int64_t> newShape) {
  // Although there is a tensor.reshape op, we use tensor.collapse_shape
  // and tensor.expand_shape since rock-view-to-transform pass doesn't
  // support tensor.reshape
  RankedTensorType currentType = cast<RankedTensorType>(input.getType());
  Location loc = input.getLoc();
  int64_t inputRank = currentType.getRank();
  int64_t outputRank = static_cast<int64_t>(newShape.size());

  if (currentType.getShape() == newShape) {
    return input;
  }

  SmallVector<ReassociationIndices> collapseReassociation(1);
  SmallVector<ReassociationIndices> expandReassociation(1);
  collapseReassociation[0].resize(inputRank);
  expandReassociation[0].resize(outputRank);
  std::iota(collapseReassociation[0].begin(), collapseReassociation[0].end(),
            0);
  std::iota(expandReassociation[0].begin(), expandReassociation[0].end(), 0);
  input = tensor::CollapseShapeOp::create(rewriter, loc, input,
                                          collapseReassociation);
  if (cast<RankedTensorType>(input.getType()).getShape() == newShape) {
    return input;
  }
  RankedTensorType resultType =
      RankedTensorType::get(newShape, currentType.getElementType());
  input = tensor::ExpandShapeOp::create(rewriter, loc, resultType, input,
                                        expandReassociation);
  return input;
}

LogicalResult
ReshapeConverter::matchAndRewrite(migraphx::ReshapeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getInput();
  ArrayAttr dims = adaptor.getDims();
  SmallVector<int64_t, 5> newShape;
  for (auto dim : dims) {
    newShape.push_back(dyn_cast<IntegerAttr>(dim).getInt());
  }
  auto output = reshapeValue(rewriter, input, newShape);
  rewriter.replaceOp(op, output);
  return success();
}

LogicalResult
BroadcastConverter::matchAndRewrite(migraphx::BroadcastOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  migraphx::MIXRShapedType input = op.getInput().getType();
  migraphx::MIXRShapedType output = op.getOutput().getType();

  RankedTensorType outputType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(output));
  if (!outputType) {
    return op.emitError("cannot convert output type to ranked tensor type");
  }

  uint64_t axis = op.getAxis();
  uint64_t outputRank = output.getRank();

  uint64_t inputRank = input.getRank();
  SmallVector<int64_t, 4> dimensionAttr;
  llvm::transform(llvm::seq<int64_t>(0, axis),
                  std::back_inserter(dimensionAttr),
                  [](int64_t val) { return val; });
  for (auto [index, dim] : llvm::enumerate(input.getShape())) {
    // the one in the input dimension can also be broadcasted
    if (dim == 1) {
      dimensionAttr.push_back(index + axis);
    }
  }
  llvm::transform(llvm::seq<int64_t>(axis + inputRank, outputRank),
                  std::back_inserter(dimensionAttr),
                  [](int64_t val) { return val; });

  // We have to remove the one dimension because it is possible that we are
  // broadcasting that to a different dimension
  auto reshaped =
      reshapeValue(rewriter, adaptor.getInput(),
                   llvm::filter_to_vector(
                       input.getShape(), [](int64_t val) { return val != 1; }));
  auto init = tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                      outputType.getElementType());
  auto result =
      linalg::BroadcastOp::create(rewriter, loc, reshaped, init, dimensionAttr);
  rewriter.replaceOp(op, result);

  return success();
}

LogicalResult MultiBroadcastConverter::matchAndRewrite(
    migraphx::MultiBroadcastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  migraphx::MIXRShapedType outMIXRType = op.getOutput().getType();
  RankedTensorType outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(outMIXRType));
  ArrayRef<int64_t> outShape = outType.getShape();
  ArrayRef<int64_t> outStrides = outMIXRType.getStrides();
  uint32_t inRank =
      cast<RankedTensorType>(adaptor.getInput().getType()).getRank();
  uint32_t outRank = outType.getRank();
  Type elemType = outType.getElementType();

  assert(outRank >= inRank && "MultiBroadcastOp shouldn't reduce rank. This "
                              "should be an invariant of this operation");

  // If it's a splat constant, broadcast it trivially
  if (auto constOp = adaptor.getInput().getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (denseAttr && denseAttr.isSplat()) {
        auto bcastConstAttr = DenseElementsAttr::get(
            outType, denseAttr.getSplatValue<Attribute>());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, outType,
                                                       bcastConstAttr);
        return success();
      }
    }
  }

  // Determine broadcast dimensions (stride == 0) and non-broadcast shape
  SmallVector<int64_t, 4> broadcastDimensions;
  SmallVector<int64_t, 4> nonBroadcastShape;
  for (auto [i, stride, shape] : llvm::enumerate(outStrides, outShape)) {
    if (stride == 0) {
      broadcastDimensions.push_back(i);
    } else {
      nonBroadcastShape.push_back(shape);
    }
  }

  // If no dimensions need broadcasting, just reshape to match output shape
  if (broadcastDimensions.empty()) {
    Value result = reshapeValue(rewriter, adaptor.getInput(), outShape);
    rewriter.replaceOp(op, result);
    return success();
  }

  // Reshape input to match the non-broadcast dimensions of the output
  Value input = reshapeValue(rewriter, adaptor.getInput(), nonBroadcastShape);

  auto init = tensor::EmptyOp::create(rewriter, loc, outShape, elemType);
  auto result = linalg::BroadcastOp::create(rewriter, loc, input, init,
                                            broadcastDimensions);
  rewriter.replaceOp(op, result);
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
} // namespace

LogicalResult
LiteralConverter::matchAndRewrite(migraphx::LiteralOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  migraphx::MIXRShapedType type = op.getResult().getType();
  RankedTensorType newType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(type));
  if (!newType) {
    return op.emitError("expected RankedTensorType as output");
  }

  ElementsAttr value = op.getValue();
  if (value.getType() != newType) {
    if (value.isSplat()) {
      // Get the original splat value (for example SI8 value)
      Attribute splatValue = value.getSplatValue<Attribute>();

      // Reinterpret the splatValue under the new type (for example SI8 -> I8),
      // preserving bytes
      Attribute newSplatValue;
      if (auto intAttr = dyn_cast<IntegerAttr>(splatValue))
        newSplatValue =
            IntegerAttr::get(newType.getElementType(), intAttr.getValue());
      else if (auto floatAttr = dyn_cast<FloatAttr>(splatValue))
        newSplatValue =
            FloatAttr::get(newType.getElementType(), floatAttr.getValue());
      else if (auto boolAttr = dyn_cast<BoolAttr>(splatValue))
        // Convert BoolAttr into IntegerAttr so we don't run target
        // materialization for type conversion. Match the result type of
        // TypeConverter
        newSplatValue =
            IntegerAttr::get(newType.getElementType(), boolAttr.getValue());
      else
        return failure();

      // Create the new SplatElementsAttr (for example I8 type) with preserved
      // value bytes
      value = SplatElementsAttr::get(newType, newSplatValue);
    } else {
      // For non-splat attributes, we need to convert each element to the new
      // type
      SmallVector<Attribute> convertedElements;
      convertedElements.reserve(value.getNumElements());

      for (auto it : value.getValues<Attribute>()) {
        Attribute convertedElement;
        if (auto intAttr = dyn_cast<IntegerAttr>(it))
          convertedElement =
              IntegerAttr::get(newType.getElementType(), intAttr.getValue());
        else if (auto floatAttr = dyn_cast<FloatAttr>(it))
          convertedElement =
              FloatAttr::get(newType.getElementType(), floatAttr.getValue());
        else if (auto boolAttr = dyn_cast<BoolAttr>(it))
          // Convert BoolAttr into IntegerAttr so we don't run target
          // materialization for type conversion. Match the result type of
          // TypeConverter
          convertedElement =
              IntegerAttr::get(newType.getElementType(), boolAttr.getValue());
        else
          return failure();

        convertedElements.push_back(convertedElement);
      }

      value = DenseElementsAttr::get(newType, convertedElements);
    }
  }

  rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newType, value);
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
           ElementwiseConverter<migraphx::RecipOp, linalg::ReciprocalOp>,
           ElementwiseConverter<migraphx::ErfOp, linalg::ErfOp>, ReluConverter,
           GenericElementwiseOpConverter<migraphx::WhereOp>,
           GenericElementwiseOpConverter<migraphx::SigmoidOp>, ReluConverter,
           ClipConverter, BroadcastConverter, MultiBroadcastConverter,
           LiteralConverter, ReshapeConverter,
           BooleanElementwiseConverter<migraphx::Greater>,
           BooleanElementwiseConverter<migraphx::Equal>, ClipConverter,
           TransposeConverter, ConvConverter>(converter, patterns.getContext());
}

void mlir::migraphx::populateMIGraphXFuncBoundaryToLinalgConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AsUnderlyingShapeConverter, AsLogicalShapeOpConverter>(
      typeConverter, patterns.getContext());

  // mhal.launch can be generated through rocmlir-gen, so we need a way to
  // legalize it
  populateMIGraphXToLinalgMHALLauncherConversion(patterns, typeConverter);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
}