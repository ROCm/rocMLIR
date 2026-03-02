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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"

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

LogicalResult AsLogicalShapeOpConverter::matchAndRewrite(
    migraphx::AsLogicalShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  migraphx::MIXRShapedType inType = op.getIn().getType();
  RankedTensorType resultType = op.getOut().getType();
  Value result = adaptor.getIn(); // The shape we are casting from

  // reshape into memory layout type
  RankedTensorType memoryType = inType.asMemoryLayoutTensor();
  if (result.getType() != inType.asMemoryLayoutTensor()) {
    SmallVector<ReassociationIndices, 4> reassociationIndex(
        1, ReassociationIndices(memoryType.getRank(), 0));
    std::iota(reassociationIndex[0].begin(), reassociationIndex[0].end(), 0);
    result = tensor::ExpandShapeOp::create(rewriter, loc, memoryType, result,
                                           reassociationIndex);
  }

  // This is the permutation that reorders the strides into standard shape.
  // Equivalently, it is the permutation that, when applied to a standard
  // shape, produces its in-memory layout. So, to get back to standard/logical
  // shape, we need to invert it.
  SmallVector<int64_t, 4> inversePermutation, permutation, transposedShape;
  inType.getStridePermutation(inversePermutation);
  size_t nDims = inversePermutation.size();
  permutation.resize_for_overwrite(nDims);
  transposedShape.resize_for_overwrite(nDims);
  bool hasTranspose = false;
  for (auto [to, from] : llvm::enumerate(inversePermutation)) {
    permutation[from] = to;
    transposedShape[from] = memoryType.getShape()[to];
    hasTranspose |= (from != static_cast<int32_t>(to));
  }

  if (hasTranspose) {
    Value init = tensor::EmptyOp::create(rewriter, loc, transposedShape,
                                         memoryType.getElementType())
                     .getResult();
    result =
        linalg::TransposeOp::create(rewriter, loc, result, init, permutation)
            .getResult()[0];
  }

  if (result.getType() == resultType) {
    rewriter.replaceOp(op, result);
    return success();
  }

  SmallVector<int64_t, 4> slicingShape(resultType.getShape());
  for (auto [dim, stride] :
       llvm::zip_equal(slicingShape, inType.getStrides())) {
    if (stride == 0)
      dim = 1;
  }

  RankedTensorType transposedType =
      dyn_cast<RankedTensorType>(result.getType());
  if (transposedType.getShape() != ArrayRef(slicingShape)) {
    SmallVector<int64_t, 4> starts(permutation.size(), 0);
    RankedTensorType sliceType = resultType.clone(slicingShape);
    SmallVector<OpFoldResult, 4> offset(sliceType.getRank(),
                                        rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult, 4> sizes;
    llvm::transform(sliceType.getShape(), std::back_inserter(sizes),
                    [&](int64_t size) { return rewriter.getIndexAttr(size); });
    SmallVector<OpFoldResult, 4> strides(sliceType.getRank(),
                                         rewriter.getIndexAttr(1));
    result = tensor::ExtractSliceOp::create(rewriter, loc, result, offset,
                                            sizes, strides)
                 .getResult();
  }

  if (result.getType() != resultType) {
    SmallVector<int64_t, 4> linalgInputShape, broadcastDimension;
    for (auto [index, stride, shape] :
         llvm::enumerate(inType.getStrides(), inType.getShape())) {
      if (stride != 0) {
        linalgInputShape.push_back(shape);
      } else {
        broadcastDimension.push_back(index);
      }
    }

    SmallVector<ReassociationIndices, 4> reassocationOne(
        1, ReassociationIndices(resultType.getRank(), 0));
    SmallVector<ReassociationIndices, 4> reassocationTwo(
        1, ReassociationIndices(linalgInputShape.size(), 0));
    std::iota(reassocationOne[0].begin(), reassocationOne[0].end(), 0);
    std::iota(reassocationTwo[0].begin(), reassocationTwo[0].end(), 0);
    result =
        tensor::CollapseShapeOp::create(rewriter, loc, result, reassocationOne);
    result = tensor::ExpandShapeOp::create(
        rewriter, loc,
        RankedTensorType::get(linalgInputShape, resultType.getElementType()),
        result, reassocationTwo);
    auto init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                        resultType.getElementType());
    result = linalg::BroadcastOp::create(rewriter, loc, result, init,
                                         broadcastDimension)
                 .getResult()[0];
  }

  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult AsUnderlyingShapeConverter::matchAndRewrite(
    migraphx::AsUnderlyingShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  migraphx::MIXRShapedType resultType = op.getOut().getType();
  RankedTensorType memoryLayoutType = resultType.asMemoryLayoutTensor();
  Value in = adaptor.getIn();
  RankedTensorType inTensorType = dyn_cast<RankedTensorType>(in.getType());
  if (!memoryLayoutType || !in)
    return op.emitOpError(
        "output or input type has strides that cannot be represented");

  RankedTensorType resultTensorType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(resultType));
  if (!resultTensorType)
    return op.emitOpError("unsupported conversion to underlying shape");
  SmallVector<int64_t, 4> permutation;
  // This is the permutation that reorderd strides into the order they'd be in
  // in a standard shape. So, applying it to a logically-shaped tensor gets
  // you the tensor in in-memory layout.
  resultType.getStridePermutation(permutation);

  Value transposed = in;
  if (!llvm::is_sorted(permutation)) {
    SmallVector<int64_t, 4> transposedShape;
    llvm::transform(permutation, std::back_inserter(transposedShape),
                    [&](int64_t permutation) {
                      return inTensorType.getShape()[permutation];
                    });

    auto init = tensor::EmptyOp::create(rewriter, loc, transposedShape,
                                        inTensorType.getElementType())
                    .getResult();
    transposed =
        linalg::TransposeOp::create(rewriter, loc, in, init, permutation)
            .getResult()[0];
  }

  if (transposed.getType() != memoryLayoutType) {
    // Check for broadcasts, which we don't support.
    if (resultType.hasBroadcast()) {
      return op.emitOpError(
          "writing to tensors with broadcasts is unsupported");
    }

    // Verify that memoryLayoutType is >= transposedType in all dimensions.
    RankedTensorType transposedType =
        cast<RankedTensorType>(transposed.getType());
    if (llvm::any_of(llvm::enumerate(memoryLayoutType.getShape(),
                                     transposedType.getShape()),
                     [&](auto data) {
                       auto [index, memDim, transDim] = data;
                       if (memDim < transDim) {
                         op.emitOpError("memory layout dimension ")
                             << memDim << " is smaller than logical dimension "
                             << transDim << "; this indicates invalid strides";
                       }
                       return memDim < transDim;
                     })) {
      return failure();
    }

    auto empty =
        tensor::EmptyOp::create(rewriter, loc, memoryLayoutType.getShape(),
                                memoryLayoutType.getElementType());
    int64_t rank = transposedType.getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (int64_t dim : transposedType.getShape())
      sizes.push_back(rewriter.getIndexAttr(dim));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    transposed = tensor::InsertSliceOp::create(rewriter, loc, transposed, empty,
                                               offsets, sizes, strides);
  }

  // collapsed shape in the end
  assert(transposed.getType() == memoryLayoutType &&
         "we should have either insert a slice to match the memory layout or "
         "the transposed shape is the memory layout");
  SmallVector<ReassociationIndices, 4> reassociationIndex(
      1, ReassociationIndices(resultType.getRank(), 0));
  std::iota(reassociationIndex[0].begin(), reassociationIndex[0].end(), 0);
  auto reshape = tensor::CollapseShapeOp::create(
      rewriter, loc, resultTensorType, transposed, reassociationIndex);

  rewriter.replaceOp(op, reshape);
  return success();
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
} // namespace

LogicalResult
BroadcastConverter::matchAndRewrite(migraphx::BroadcastOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  migraphx::MIXRShapedType input = op.getInput().getType();
  migraphx::MIXRShapedType output = op.getOutput().getType();

  RankedTensorType outputType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(output));
  if (!outputType) {
    return op.emitError("cannot convert output type to ranked tesnor type");
  }

  uint64_t axis = op.getAxis();
  uint64_t outputRank = output.getRank();
  uint64_t inputRank = input.getRank();
  SmallVector<int64_t, 4> dimensionAttr;
  llvm::transform(llvm::seq<int64_t>(0, axis),
                  std::back_inserter(dimensionAttr),
                  [](int64_t val) { return val; });
  llvm::transform(llvm::seq<int64_t>(axis + inputRank, outputRank),
                  std::back_inserter(dimensionAttr),
                  [](int64_t val) { return val; });

  auto init = tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                      outputType.getElementType());
  auto result = linalg::BroadcastOp::create(rewriter, loc, adaptor.getInput(),
                                            init, dimensionAttr);
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
           ElementwiseConverter<migraphx::RecipOp, linalg::ReciprocalOp>,
           ReluConverter, ClipConverter, BroadcastConverter>(
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
