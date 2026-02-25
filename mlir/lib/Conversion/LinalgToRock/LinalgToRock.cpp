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
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
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
// ConvLinalgConverter: linalg.generic (conv) -> rock.conv
//===----------------------------------------------------------------------===//

namespace {
struct ConvFields {
  rock::LinalgConvType type;
  int64_t spatialDim;
  ArrayAttr padding, stride, dilation;
  StringAttr perfConfig;
};
} // namespace

static int64_t getSpatialDim(rock::LinalgConvType type) {
  switch (type) {
  case rock::LinalgConvType::Conv1dNgchGfch:
    return 1;
  case rock::LinalgConvType::Conv2dNgchwGfchw:
    return 2;
  case rock::LinalgConvType::Conv3dNgchwdGfchwd:
    return 3;
  }
  llvm_unreachable("unknown LinalgConvType");
}

/// Set filter_layout, input_layout, and output_layout on a rock.conv op.
static void setConvLayoutAttrs(OpBuilder &builder, rock::ConvOp cop,
                               rock::LinalgConvType type) {
  auto set = [&](StringRef name, ArrayRef<StringRef> layout) {
    cop->setAttr(name, builder.getStrArrayAttr(layout));
  };
  switch (type) {
  case rock::LinalgConvType::Conv3dNgchwdGfchwd:
    set("filter_layout", {"g", "k", "0", "1", "2", "c"});
    set("input_layout", {"ni", "0i", "1i", "2i", "gi", "ci"});
    set("output_layout", {"no", "0o", "1o", "2o", "go", "ko"});
    break;
  case rock::LinalgConvType::Conv2dNgchwGfchw:
    set("filter_layout", {"g", "k", "c", "y", "x"});
    set("input_layout", {"ni", "gi", "ci", "hi", "wi"});
    set("output_layout", {"no", "go", "ko", "ho", "wo"});
    break;
  case rock::LinalgConvType::Conv1dNgchGfch:
    set("filter_layout", {"g", "k", "y", "x", "c"});
    set("input_layout", {"ni", "hi", "wi", "gi", "ci"});
    set("output_layout", {"no", "ho", "wo", "go", "ko"});
    break;
  }
}

/// Transform filter from GFC* layout to GF*C layout for rock.conv.
/// 2D is already in the correct layout.
static Value transformFilter(OpBuilder &builder, Location loc, Value filter,
                             int64_t spatialDim) {
  ArrayRef<int64_t> shape =
      cast<RankedTensorType>(filter.getType()).getShape();
  switch (spatialDim) {
  case 3: {
    rock::BottomUpTMBuilder b(builder, {"g", "f", "c", "h", "w", "d"}, shape,
                              loc);
    b.passThrough({"gk", "fk"}, {0, 1}, {"g", "f"});
    b.passThrough({"hk", "wk", "dk"}, {2, 3, 4}, {"h", "w", "d"});
    b.passThrough({"ck"}, {5}, {"c"});
    return rock::TransformOp::create(builder, loc, filter, b.get());
  }
  case 2:
    return filter;
  case 1: {
    // Conv1D is expanded into Conv2D (matching migraphx-to-tosa): unmerge
    // H into (H, W=1).
    rock::BottomUpTMBuilder b(builder, {"g", "f", "c", "h"}, shape, loc);
    b.passThrough({"gk", "fk"}, {0, 1}, {"g", "f"});
    b.unmerge({"hk", "wk"}, {2, 3}, {"h"}, {shape[3], 1});
    b.passThrough({"ck"}, {4}, {"c"});
    return rock::TransformOp::create(builder, loc, filter, b.get());
  }
  default:
    llvm_unreachable("unsupported spatial dim for filter transform");
  }
}

/// Remove the tensor.pad + tensor.expand_shape pattern emitted by
/// migraphx-to-linalg, replacing it with just tensor.expand_shape on the
/// unpadded source. rock.conv handles padding internally.
///
/// Expected IR structure:
///   %padded = tensor.pad %original ...
///   %expanded = tensor.expand_shape %padded ...
/// Replaced with:
///   %expanded = tensor.expand_shape %original ...
static FailureOr<Value>
removePaddingFromInput(ConversionPatternRewriter &rewriter,
                       linalg::GenericOp op, Value in, ArrayAttr padding) {
  bool hasPadding = llvm::any_of(padding.getValue(), [](Attribute attr) {
    return cast<IntegerAttr>(attr).getInt() != 0;
  });
  if (!hasPadding)
    return in;

  auto expanded = in.getDefiningOp<tensor::ExpandShapeOp>();
  if (!expanded) {
    op.emitError("unexpected padding code structure");
    return failure();
  }
  auto padded = expanded->getOperand(0).getDefiningOp<tensor::PadOp>();
  if (!padded) {
    op.emitError("unexpected padding code structure");
    return failure();
  }

  SmallVector<int64_t, 6> resultShape(expanded.getResultType().getShape());
  auto lowPad = padded.getStaticLow();
  auto highPad = padded.getStaticHigh();
  int64_t numPadDims = lowPad.size();
  int64_t numExpandedDims = resultShape.size();

  // Padding is defined in pre-expand space. The spatial dims are at the
  // tail of both tensors (expand_shape only splits an earlier dim), so
  // align from the end.
  for (int64_t i = numPadDims - 1, j = numExpandedDims - 1;
       i >= 0 && j >= 0; --i, --j) {
    resultShape[j] -= (lowPad[i] + highPad[i]);
  }

  RankedTensorType newResultType = RankedTensorType::get(
      resultShape, padded.getResultType().getElementType());
  Value result = tensor::ExpandShapeOp::create(
      rewriter, expanded.getLoc(), newResultType, padded.getOperand(0),
      expanded.getReassociationIndices());
  rewriter.replaceOp(expanded, result);
  rewriter.eraseOp(padded);
  return result;
}

/// Transform input from NGC* layout to N*GC layout for rock.conv.
/// 2D is already in the correct layout.
static Value transformInput(OpBuilder &builder, Location loc, Value input,
                            int64_t spatialDim) {
  ArrayRef<int64_t> shape =
      cast<RankedTensorType>(input.getType()).getShape();
  switch (spatialDim) {
  case 3: {
    rock::BottomUpTMBuilder b(builder, {"n", "g", "c", "h", "w", "d"}, shape,
                              loc);
    b.passThrough({"ni"}, {0}, {"n"});
    b.passThrough({"hi", "wi", "di"}, {1, 2, 3}, {"h", "w", "d"});
    b.passThrough({"gi", "ci"}, {4, 5}, {"g", "c"});
    return rock::TransformOp::create(builder, loc, input, b.get());
  }
  case 2:
    return input;
  case 1: {
    // Conv1D is expanded into Conv2D (matching migraphx-to-tosa): unmerge
    // H into (H, W=1).
    int64_t h = shape[3];
    rock::BottomUpTMBuilder b(builder, {"n", "g", "c", "h"}, shape, loc);
    b.passThrough({"ni"}, {0}, {"n"});
    b.unmerge({"hi", "wi"}, {1, 2}, {"h"}, {h, 1});
    b.passThrough({"gi", "ci"}, {3, 4}, {"g", "c"});
    return rock::TransformOp::create(builder, loc, input, b.get());
  }
  default:
    llvm_unreachable("unsupported spatial dim for input transform");
  }
}

/// Compute the rock output shape from the linalg output shape.
/// Linalg layout is NGF* while rock needs N*GF (with extra W=1 for 1D).
static SmallVector<int64_t, 6>
computeRockOutputShape(ArrayRef<int64_t> linalgShape, int64_t spatialDim) {
  if (spatialDim == 2)
    return SmallVector<int64_t, 6>(linalgShape);
  SmallVector<int64_t, 6> shape;
  shape.push_back(linalgShape[0]);
  shape.insert(shape.end(), std::next(linalgShape.begin(), 3),
               linalgShape.end());
  if (spatialDim == 1)
    shape.push_back(1); // Conv1D expanded to Conv2D: extra W=1
  shape.push_back(linalgShape[1]);
  shape.push_back(linalgShape[2]);
  return shape;
}

/// Transform rock.conv output back to the linalg output layout.
/// 2D needs no transform.
static Value transformOutput(OpBuilder &builder, Location loc, Value convResult,
                             int64_t spatialDim) {
  if (spatialDim == 2)
    return convResult;
  ArrayRef<int64_t> shape =
      cast<RankedTensorType>(convResult.getType()).getShape();
  switch (spatialDim) {
  case 3: {
    rock::BottomUpTMBuilder b(builder, {"n", "h", "w", "d", "g", "f"}, shape,
                              loc);
    b.passThrough({"go", "fo"}, {1, 2}, {"g", "f"});
    b.passThrough({"no"}, {0}, {"n"});
    b.passThrough({"ho", "wo", "do"}, {3, 4, 5}, {"h", "w", "d"});
    return rock::TransformOp::create(builder, loc, convResult, b.get());
  }
  case 1: {
    // Conv1D was expanded into Conv2D: merge (H, W=1) back into H.
    rock::BottomUpTMBuilder b(builder, {"n", "h", "w", "g", "f"}, shape, loc);
    b.passThrough({"no"}, {0}, {"n"});
    b.passThrough({"go", "fo"}, {1, 2}, {"g", "f"});
    b.merge("ho", 3, {"h", "w"});
    return rock::TransformOp::create(builder, loc, convResult, b.get());
  }
  default:
    llvm_unreachable("unsupported spatial dim for output transform");
  }
}

namespace {
struct ConvLinalgConverter final
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;
  using OpConversionPattern<linalg::GenericOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<linalg::GenericOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  FailureOr<ConvFields> isConv(ConversionPatternRewriter &rewriter,
                               linalg::GenericOp op) const;
};
} // namespace

FailureOr<ConvFields>
ConvLinalgConverter::isConv(ConversionPatternRewriter &rewriter,
                            linalg::GenericOp op) const {
  // FIXME: In the future, strides, dilation, and padding can be extracted
  // by matching the AffineExpr syntax tree. The convolution dimension and
  // layout could also be inferred from the affine_map.
  auto name = op->getAttrOfType<rock::LinalgConvTypeAttr>("conv_op");
  if (!name)
    return failure();
  rock::LinalgConvType convType = name.getValue();
  int64_t spatialDim = getSpatialDim(convType);

  auto convertToArrayAttr =
      [&](Attribute arr, ArrayRef<int64_t> dimOneDefaults = {}) -> ArrayAttr {
    SmallVector<int64_t, 4> values;
    llvm::transform(
        cast<ArrayAttr>(arr).getValue(), std::back_inserter(values),
        [](Attribute val) { return cast<IntegerAttr>(val).getInt(); });
    // Conv1D is expanded into Conv2D to match the migraphx-to-tosa pipeline.
    // Append identity defaults (stride=1, dilation=1, pad=0) for the extra
    // spatial dimension.
    if (spatialDim == 1)
      values.insert(values.end(), dimOneDefaults.begin(),
                    dimOneDefaults.end());
    return rewriter.getIndexArrayAttr(values);
  };

  auto dilation =
      convertToArrayAttr(op->getAttr("dilation"), /*dimOneDefaults=*/1);
  auto stride =
      convertToArrayAttr(op->getAttr("stride"), /*dimOneDefaults=*/1);

  // Input format:  [dim0_low, dim1_low, ..., dim0_high, dim1_high, ...]
  // Rock  format:  [dim0_low, dim0_high, dim1_low, dim1_high, ...]
  auto originalPadding = convertToArrayAttr(op->getAttr("pad")).getValue();
  int64_t numSpatial = originalPadding.size() / 2;
  SmallVector<Attribute, 8> interleavedPad;
  for (int64_t i = 0; i < numSpatial; ++i) {
    interleavedPad.push_back(originalPadding[i]);
    interleavedPad.push_back(originalPadding[numSpatial + i]);
  }
  // For Conv1D is expanded into Conv2D like the tosa pipeline, so
  // we set the last dimension have 0 padding to stay consistent.
  if (spatialDim == 1) {
    interleavedPad.push_back(rewriter.getIndexAttr(0));
    interleavedPad.push_back(rewriter.getIndexAttr(0));
  }
  auto padding = rewriter.getArrayAttr(interleavedPad);
  if (!padding || !dilation || !stride)
    return failure();

  StringAttr perfConfig = op->getAttrOfType<StringAttr>("perf_config");
  return ConvFields{convType, spatialDim, padding, stride, dilation,
                    perfConfig};
}

LogicalResult ConvLinalgConverter::matchAndRewrite(
    linalg::GenericOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  FailureOr<ConvFields> maybeConv = isConv(rewriter, op);
  if (failed(maybeConv))
    return failure();

  ConvFields conv = *maybeConv;
  Location loc = op.getLoc();

  auto maybeInput =
      removePaddingFromInput(rewriter, op, op.getOperand(0), conv.padding);
  if (failed(maybeInput))
    return failure();

  Value input = transformInput(rewriter, loc, *maybeInput, conv.spatialDim);
  Value filter =
      transformFilter(rewriter, loc, op.getOperand(1), conv.spatialDim);

  RankedTensorType linalgResultType =
      cast<RankedTensorType>(op.getResult(0).getType());
  SmallVector<int64_t, 6> rockShape =
      computeRockOutputShape(linalgResultType.getShape(), conv.spatialDim);
  RankedTensorType rockResultType =
      RankedTensorType::get(rockShape, linalgResultType.getElementType());
  Value output =
      bufferization::AllocTensorOp::create(rewriter, loc, rockResultType, {});

  auto cop = rock::ConvOp::create(rewriter, loc, rockResultType, filter, input,
                                  output, /*features=*/nullptr,
                                  /*blockSize=*/nullptr, /*gridSize=*/nullptr,
                                  conv.padding, conv.stride, conv.dilation,
                                  /*params=*/nullptr);
  // TODO: add splitk
  if (conv.perfConfig)
    cop->setAttr("perf_config", conv.perfConfig);

  setConvLayoutAttrs(rewriter, cop, conv.type);

  Value result =
      transformOutput(rewriter, loc, cop.getResult(), conv.spatialDim);
  rewriter.replaceOp(op, result);
  return success();
}

void mlir::rock::populateLinalgToRockConversionPattern(
    RewritePatternSet &pattern, MLIRContext *context) {
  pattern.add<MatmulConverter<linalg::BatchMatmulOp>,
              MatmulConverter<linalg::MatmulOp>, ConvLinalgConverter>(context);
}
