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
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
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
} // namespace

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
} // namespace

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
           ReluConverter, ClipConverter, BroadcastConverter,
           MultiBroadcastConverter, LiteralConverter, ReshapeConverter,
           BooleanElementwiseConverter<migraphx::Greater>,
           BooleanElementwiseConverter<migraphx::Equal>, ClipConverter>(
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
