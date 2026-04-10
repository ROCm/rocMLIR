//===- LinalgToRock.cpp - Lowering Linalg to Rock Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Linalg to the Rock dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/LinalgToRock/LinalgToRock.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"

#include <tuple>

using namespace mlir;

namespace {
template <typename LinalgMatOp>
struct MatmulConverter final : public OpConversionPattern<LinalgMatOp> {
  struct MatmulContext {
    Value aMatrix, bMatrix, scaleA, scaleB;
    UnitAttr aTransposedAttr, bTransposedAttr, aScaleTransposedAttr,
        bScaleTransposedAttr;
  };

  using OpConversionPattern<LinalgMatOp>::OpConversionPattern;
  using OpConversionPattern<LinalgMatOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<LinalgMatOp>::OpAdaptor;

  FailureOr<MatmulContext>
  getRockMatmulContext(LinalgMatOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const;

  LogicalResult
  matchAndRewrite(LinalgMatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

/// Check if a matrix operand in a matmul operation is transposed.
/// operandIndex is 0 for A matrix and 1 for B matrix
/// Returns false if identity map, true if last two dims swapped, failure
/// otherwise.
static FailureOr<bool> isMatrixTransposed(AffineMapAttr indexingMap,
                                          bool isAMatrix) {
  if (!indexingMap || indexingMap.getAffineMap().getNumResults() < 2) {
    // it is possible for the result of the affine map to have one dimension in
    // the case of broadcasting
    return failure();
  }

  AffineMap map = indexingMap.getAffineMap();
  unsigned numDims = map.getNumResults();
  unsigned numInputs = map.getNumInputs();
  auto secondLast = dyn_cast<AffineDimExpr>(map.getResult(numDims - 2));
  auto last = dyn_cast<AffineDimExpr>(map.getResult(numDims - 1));

  if (numDims < 2 || !secondLast || !last) {
    return failure();
  }

  // Verify all dimensions except the last two are identity-mapped.
  // For example, in (d0, d1, d2) -> (d0, d2, d1), we check that d0 maps to
  // position 0. This ensures only the last two dimensions are potentially
  // swapped.
  if (!llvm::all_of(llvm::seq<unsigned>(0, numDims - 2), [&](unsigned i) {
        auto expr = dyn_cast<AffineDimExpr>(map.getResult(i));
        return expr && expr.getPosition() == i;
      })) {
    return failure();
  }

  // Define expected positions based on operand type and iteration space
  // For batch_matmul with iteration space (d0, d1, d2, d3) = (batch, m, n, k):
  //
  // A matrix (operandIndex=0):
  //   - Transposed:     (d0, d1, d2, d3) -> (d0, d3, d1)  i.e., (batch, k, m)
  //     Last two results map to positions: d3->3, d1->1 (swapped)
  //
  // B matrix (operandIndex=1):
  //   - Transposed:     (d0, d1, d2, d3) -> (d0, d2, d3)  i.e., (batch, n, k)
  //     Last two results map to positions: d2->2, d3->3 (swapped)
  unsigned transposedSecond = isAMatrix ? numInputs - 1 : numInputs - 2;
  unsigned transposedLast = isAMatrix ? numInputs - 3 : numInputs - 1;
  bool isTransposed = (secondLast.getPosition() == transposedSecond &&
                       last.getPosition() == transposedLast);
  return isTransposed;
}

template <typename LinalgMatOp>
FailureOr<typename MatmulConverter<LinalgMatOp>::MatmulContext>
MatmulConverter<LinalgMatOp>::getRockMatmulContext(
    LinalgMatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Nice wrapper around isMatrixTransposed to reduce code duplication
  auto getTransposeAttrs = [&](AffineMapAttr matrixAIndexingMap,
                               AffineMapAttr matrixBIndexingMap)
      -> FailureOr<std::tuple<UnitAttr, UnitAttr>> {
    FailureOr<bool> maybeATransposed =
        isMatrixTransposed(matrixAIndexingMap, /*isAMatrix=*/true);
    FailureOr<bool> maybeBTransposed =
        isMatrixTransposed(matrixBIndexingMap, /*isAMatrix=*/false);
    if (failed(maybeATransposed) || failed(maybeBTransposed))
      return failure();
    UnitAttr aTransposedAttr =
        *maybeATransposed ? rewriter.getAttr<UnitAttr>() : nullptr;
    UnitAttr bTransposedAttr =
        *maybeBTransposed ? rewriter.getAttr<UnitAttr>() : nullptr;
    return std::make_tuple(aTransposedAttr, bTransposedAttr);
  };

  MatmulContext context;
  if (isa<linalg::GenericOp>(op) && op->hasAttr("rock.quant_dot")) {
    // The linalg.generic op from migraphx-to-linalg place this operand in this
    // way. This operation doesn't have support for transpose as of current
    context.aMatrix = op.getInputs()[0];
    context.scaleA = op.getInputs()[1];
    context.bMatrix = op.getInputs()[2];
    context.scaleB = op.getInputs()[3];

    auto maybeTranspose =
        getTransposeAttrs(dyn_cast<AffineMapAttr>(op.getIndexingMaps()[0]),
                          dyn_cast<AffineMapAttr>(op.getIndexingMaps()[2]));
    auto maybeScaleTranspose =
        getTransposeAttrs(dyn_cast<AffineMapAttr>(op.getIndexingMaps()[1]),
                          dyn_cast<AffineMapAttr>(op.getIndexingMaps()[3]));
    if (failed(maybeTranspose) || failed(maybeScaleTranspose))
      return failure();
    auto [aTransposedAttr, bTransposedAttr] = *maybeTranspose;
    auto [aScaleTransposedAttr, bScaleTransposedAttr] = *maybeScaleTranspose;

    context.aTransposedAttr = aTransposedAttr;
    context.aScaleTransposedAttr = aScaleTransposedAttr;
    context.bTransposedAttr = bTransposedAttr;
    context.bScaleTransposedAttr = bScaleTransposedAttr;
    return success(context);
  }

  // only expect either linalg.matmul or linalg.batch_matmul
  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Value a = op.getOperand(0);
  Value b = op.getOperand(1);
  Value cOriginal = op.getOutputs()[0];

  if (!isa<RankedTensorType>(cOriginal.getType()) ||
      !cast<RankedTensorType>(cOriginal.getType()).hasStaticShape()) {
    return op.emitError(
        "expected the output to have RankedTensorType and static shape");
  }

  auto maybeTranspose =
      getTransposeAttrs(dyn_cast<AffineMapAttr>(op.getIndexingMaps()[0]),
                        dyn_cast<AffineMapAttr>(op.getIndexingMaps()[1]));
  if (failed(maybeTranspose))
    return op.emitError("cannot determine if input matrix is transposed");
  auto [aTransposedAttr, bTransposedAttr] = *maybeTranspose;

  context.aMatrix = a;
  context.scaleA = nullptr;
  context.bMatrix = b;
  context.scaleB = nullptr;
  context.aTransposedAttr = aTransposedAttr;
  context.bTransposedAttr = bTransposedAttr;
  return success(context);
}

template <typename LinalgMatOp>
LogicalResult MatmulConverter<LinalgMatOp>::matchAndRewrite(
    LinalgMatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  FailureOr<MatmulContext> maybeContext =
      getRockMatmulContext(op, adaptor, rewriter);
  if (failed(maybeContext)) {
    return failure();
  }
  MatmulContext context = maybeContext.value();

  // TODO: handle split K attributes as well
  // TODO: handle broadcasting for matrix A and B
  RankedTensorType outputType =
      cast<RankedTensorType>(op.getOutputs()[0].getType());
  rock::StoreMethodAttr method =
      rewriter.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set);
  Value c = bufferization::AllocTensorOp::create(rewriter, op.getLoc(),
                                                 outputType, {});
  rock::GemmOp result = rock::GemmOp::create(
      rewriter, loc, c.getType(), context.aMatrix, context.bMatrix, c,
      /*scaleA=*/context.scaleA,
      /*scaleB=*/context.scaleB, /*aTransposed=*/context.aTransposedAttr,
      /*bTransposed=*/context.bTransposedAttr,
      /*cTransposed=*/nullptr, /*aScaleTransposed=*/context.aScaleTransposedAttr,
      /*bScaleTransposed=*/context.bScaleTransposedAttr, /*features=*/nullptr,
      /*storeMethod=*/method, /*derivedBlockSize=*/nullptr,
      /*gridSize=*/nullptr, /*params=*/nullptr);

  if (auto attr = op->template getAttrOfType<StringAttr>("perf_config"))
    result->setAttr("perf_config", attr);

  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// shape related changes
//===----------------------------------------------------------------------===//
namespace {
struct ExpandStrideConverter final
    : public OpConversionPattern<tensor::InsertSliceOp> {
  using OpConversionPattern<tensor::InsertSliceOp>::OpConversionPattern;
  using OpConversionPattern<tensor::InsertSliceOp>::getTypeConverter;
  using OpAdaptor =
      typename OpConversionPattern<tensor::InsertSliceOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

bool mlir::rock::isRockExpandStride(tensor::InsertSliceOp op) {
  return op->hasAttr("rock.is_expand_strides") &&
         isa<tensor::EmptyOp>(op.getOperand(1).getDefiningOp());
}

LogicalResult ExpandStrideConverter::matchAndRewrite(
    tensor::InsertSliceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // The migraphx-to-linalg passes emits the rock.is_expand_stride attribute
  // to indicate that the insert_slice is an expand_stride. In that case, we
  // transform it into a rock.expand_strides.
  if (!rock::isRockExpandStride(op)) {
    return failure();
  }
  tensor::EmptyOp tensorEmpty =
      dyn_cast<tensor::EmptyOp>(op.getOperand(1).getDefiningOp());
  assert(tensorEmpty && "Should have been checked by isRockExpandStride");

  Location loc = op.getLoc();
  auto alloc = bufferization::AllocTensorOp::create(
      rewriter, loc, tensorEmpty.getResult().getType(), {});
  auto expandOp = rock::ExpandStridesOp::create(rewriter, loc, op.getType(),
                                                adaptor.getSource(), alloc);
  rewriter.replaceOp(op, expandOp);
  return success();
}

void mlir::rock::populateLinalgToRockConversionPattern(
    RewritePatternSet &pattern, MLIRContext *context) {
  pattern.add<MatmulConverter<linalg::BatchMatmulOp>,
              MatmulConverter<linalg::MatmulOp>, ExpandStrideConverter,
              MatmulConverter<linalg::GenericOp>>(context);
}
