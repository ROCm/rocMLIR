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

using namespace mlir;

namespace {
template <typename LinalgMatOp>
struct MatmulConverter final : public OpConversionPattern<LinalgMatOp> {
  using OpConversionPattern<LinalgMatOp>::OpConversionPattern;
  using OpConversionPattern<LinalgMatOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<LinalgMatOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(LinalgMatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

/// Check if a matrix operand in a matmul operation is transposed.
/// operandIndex is 0 for A matrix and 1 for B matrix
/// Returns false if identity map, true if last two dims swapped, failure
/// otherwise.
template <typename LinalgOp>
static FailureOr<bool> isMatrixTransposed(LinalgOp op, unsigned operandIndex) {
  auto indexingMap =
      dyn_cast<AffineMapAttr>(op.getIndexingMaps()[operandIndex]);
  if (!indexingMap || (operandIndex != 1 && operandIndex != 0) ||
      indexingMap.getAffineMap().getNumResults() < 2) {
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
  unsigned transposedSecond = operandIndex == 0 ? numInputs - 1 : numInputs - 2;
  unsigned transposedLast = operandIndex == 0 ? numInputs - 3 : numInputs - 1;
  bool isTransposed = (secondLast.getPosition() == transposedSecond &&
                       last.getPosition() == transposedLast);
  return isTransposed;
}

template <typename LinalgMatOp>
LogicalResult MatmulConverter<LinalgMatOp>::matchAndRewrite(
    LinalgMatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value a = op.getOperand(0);
  Value b = op.getOperand(1);
  Value cOriginal = op.getOutputs()[0];

  if (!isa<RankedTensorType>(cOriginal.getType()) ||
      !cast<RankedTensorType>(cOriginal.getType()).hasStaticShape()) {
    return op.emitError(
        "expected the output to have RankedTensorType and static shape");
  }

  RankedTensorType outputType = cast<RankedTensorType>(cOriginal.getType());
  Value c = bufferization::AllocTensorOp::create(rewriter, op.getLoc(),
                                                 outputType, {});

  // Setting the A and B matrix transpose attribute
  FailureOr<bool> maybeAMatrixTransposed =
      isMatrixTransposed<LinalgMatOp>(op, 0);
  FailureOr<bool> maybeBMatrixTransposed =
      isMatrixTransposed<LinalgMatOp>(op, 1);
  if (failed(maybeAMatrixTransposed) || failed(maybeBMatrixTransposed)) {
    return op.emitError("cannot determine if input matrix is transposed");
  }
  UnitAttr aTransposedAttr =
      (maybeAMatrixTransposed.value()) ? rewriter.getAttr<UnitAttr>() : nullptr;
  UnitAttr bTransposedAttr =
      (maybeBMatrixTransposed.value()) ? rewriter.getAttr<UnitAttr>() : nullptr;

  // TODO: handle split K attributes as well
  // TODO: handle broadcasting for matrix A and B
  // TODO: Scaled GEMM not yet supported (scaleA/scaleB currently null)
  rock::StoreMethodAttr method =
      rewriter.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set);
  rock::GemmOp result = rock::GemmOp::create(
      rewriter, loc, c.getType(), a, b, c, /*scaleA=*/nullptr,
      /*scaleB=*/nullptr, /*aTransposed=*/aTransposedAttr,
      /*bTransposed=*/bTransposedAttr,
      /*cTransposed=*/nullptr, /*aScaleTransposed=*/nullptr,
      /*bScaleTransposed=*/nullptr, /*features=*/nullptr,
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
              MatmulConverter<linalg::MatmulOp>, ExpandStrideConverter>(
      context);
}
