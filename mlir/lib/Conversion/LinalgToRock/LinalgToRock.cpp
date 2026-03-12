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
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
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
  case rock::LinalgConvType::Conv1dBWDNgchGcfh:
  case rock::LinalgConvType::Conv1dNgchGfch:
    return 1;
  case rock::LinalgConvType::Conv2dBWDNgchwGcfhw:
  case rock::LinalgConvType::Conv2dNgchwGfchw:
    return 2;
  case rock::LinalgConvType::Conv3dBWDNgchwdGcfhwd:
  case rock::LinalgConvType::Conv3dNgchwdGfchwd:
    return 3;
  }
  llvm_unreachable("unknown LinalgConvType");
}

/// Set filter_layout, input_layout, and output_layout on a rock.conv op.
/// Layouts match the linalg convention: GKC*, NGC*, NGK*.
static void setConvLayoutAttrs(OpBuilder &builder, Operation *cop,
                               int64_t spatialDim) {
  auto *ctx = builder.getContext();
  auto setLayout = [&](StringRef attrName, ArrayRef<StringRef> prefix,
                       StringRef suffix) {
    SmallVector<Attribute> layout;
    for (StringRef dim : prefix)
      layout.push_back(StringAttr::get(ctx, dim));
    for (int64_t i = 0; i < spatialDim; ++i)
      layout.push_back(StringAttr::get(ctx, Twine(i) + suffix));
    cop->setAttr(attrName, builder.getArrayAttr(layout));
  };
  setLayout("filter_layout", {"g", "k", "c"}, "");
  setLayout("input_layout", {"ni", "gi", "ci"}, "i");
  setLayout("output_layout", {"no", "go", "ko"}, "o");
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
  if (!padded || !padded->hasOneUse()) {
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
  for (int64_t i = numPadDims - 1, j = numExpandedDims - 1; i >= 0 && j >= 0;
       --i, --j) {
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

namespace {
struct ConvLinalgConverter final
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;
  using OpConversionPattern<linalg::GenericOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<linalg::GenericOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct BwdConvLinalgConverter final
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;
  using OpConversionPattern<linalg::GenericOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<linalg::GenericOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

static FailureOr<ConvFields> isConv(ConversionPatternRewriter &rewriter,
                                    linalg::GenericOp op) {
  auto name = op->getAttrOfType<rock::LinalgConvTypeAttr>("conv_op");
  if (!name)
    return failure();
  rock::LinalgConvType convType = name.getValue();
  int64_t spatialDim = getSpatialDim(convType);
  // Conv1D is broadcasted into Conv2D. To check for error, we
  // use effectiveDim instead because it one more stride/dilation
  // in the expanded dimension
  int64_t effectiveDim = (spatialDim == 1) ? spatialDim + 1 : spatialDim;

  auto convertToArrayAttr =
      [&](Attribute arr, ArrayRef<int64_t> dimOneDefaults = {}) -> ArrayAttr {
    if (!arr || !isa<ArrayAttr>(arr)) {
      return ArrayAttr{};
    }

    SmallVector<int64_t, 4> values;
    llvm::transform(
        cast<ArrayAttr>(arr).getValue(), std::back_inserter(values),
        [](Attribute val) { return cast<IntegerAttr>(val).getInt(); });
    // Conv1D is expanded into Conv2D: append identity defaults for the
    // extra spatial dimension (stride=1, dilation=1, pad=0).
    if (spatialDim == 1)
      values.insert(values.end(), dimOneDefaults.begin(), dimOneDefaults.end());
    return rewriter.getIndexArrayAttr(values);
  };

  auto dilation =
      convertToArrayAttr(op->getAttr("dilation"), /*dimOneDefaults=*/{1});
  auto stride =
      convertToArrayAttr(op->getAttr("stride"), /*dimOneDefaults=*/{1});
  if (!dilation || !stride || (int64_t)dilation.size() != effectiveDim ||
      (int64_t)stride.size() != effectiveDim) {
    op.emitError("invalid dilation or stride");
    return failure();
  }

  // Input format:  [dim0_low, dim1_low, ..., dim0_high, dim1_high, ...]
  // Rock  format:  [dim0_low, dim0_high, dim1_low, dim1_high, ...]
  auto originalPadding = convertToArrayAttr(op->getAttr("pad"));
  if (!originalPadding) {
    op.emitError("no padding found");
    return failure();
  }
  int64_t numSpatial = originalPadding.size() / 2;
  SmallVector<Attribute, 8> interleavedPad;
  for (int64_t i = 0; i < numSpatial; ++i) {
    interleavedPad.push_back(originalPadding[i]);
    interleavedPad.push_back(originalPadding[numSpatial + i]);
  }
  // Conv1D is expanded into Conv2D
  if (spatialDim == 1) {
    interleavedPad.push_back(rewriter.getIndexAttr(0));
    interleavedPad.push_back(rewriter.getIndexAttr(0));
  }
  auto padding = rewriter.getArrayAttr(interleavedPad);
  // note that Conv1D is expanded into Conv2D
  if (effectiveDim * 2 != (int64_t)padding.size()) {
    op.emitError("invalid number of padding");
    return failure();
  }

  StringAttr perfConfig = op->getAttrOfType<StringAttr>("perf_config");
  return ConvFields{convType, spatialDim, padding,
                    stride,   dilation,   perfConfig};
}

LogicalResult BwdConvLinalgConverter::matchAndRewrite(
    linalg::GenericOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  FailureOr<ConvFields> maybeConv = isConv(rewriter, op);
  if (failed(maybeConv))
    return failure();

  ConvFields conv = *maybeConv;
  Location loc = op.getLoc();

  // Making sure this is a forward conv only
  switch (conv.type) {
  case rock::LinalgConvType::Conv1dBWDNgchGcfh:
  case rock::LinalgConvType::Conv2dBWDNgchwGcfhw:
  case rock::LinalgConvType::Conv3dBWDNgchwdGcfhwd:
    break;
  default:
    return failure();
  }
  bool hasPadding = llvm::any_of(conv.padding, [](Attribute attr) {
    return cast<IntegerAttr>(attr).getInt() != 0;
  });

  // FIXME: add a check here!
  RankedTensorType resultShape =
      cast<RankedTensorType>(adaptor.getOutputs()[0].getType());
  tensor::ExtractSliceOp extractSlicePadding = nullptr;
  tensor::CollapseShapeOp collapseGroupPadding = nullptr;
  if (hasPadding) {
    // To handle padding, the migraphx to linalg pipeline
    // and it should look something like the following:
    // linalg.generic ins(...) outs(%output)
    // %collapse_group = tensor.collapse_shape %output ....
    // %output = tensor.extract_slice %collapse_shape ...
    if (!op->hasOneUse())
      return op.emitError("invalid padding code structure");
    collapseGroupPadding = dyn_cast<tensor::CollapseShapeOp>(*op->user_begin());
    if (!collapseGroupPadding || !collapseGroupPadding->hasOneUse())
      return op.emitError("invalid padding code structure");

    extractSlicePadding =
        dyn_cast<tensor::ExtractSliceOp>(*collapseGroupPadding->user_begin());
    if (!extractSlicePadding)
      return op.emitError("invalid padding code structure");

    // Take the padded output shape - HWD
    auto lastFewShape = cast<RankedTensorType>(extractSlicePadding.getType())
                            .getShape()
                            .drop_front(2);
    // Take the first NGK
    SmallVector<int64_t, 4> newShape(resultShape.getShape().take_front(3));
    newShape.insert(newShape.end(), lastFewShape.begin(), lastFewShape.end());
    resultShape = RankedTensorType::get(newShape, resultShape.getElementType());
  }

  Value filter = adaptor.getOperands()[1];
  Value input = adaptor.getOperands()[0];
  auto output =
      bufferization::AllocTensorOp::create(rewriter, loc, resultShape, {});
  auto cop = rock::ConvBwdDataOp::create(
      rewriter, loc, output.getType(), filter, output, input,
      /*features=*/nullptr,
      /*blockSize=*/nullptr,
      /*gridSize=*/nullptr, conv.padding, conv.stride, conv.dilation,
      /*params=*/nullptr, rewriter.getIndexAttr(0),
      /*usesV4R1=*/rewriter.getBoolAttr(false));
  setConvLayoutAttrs(rewriter, cop, getSpatialDim(conv.type));

  rock::ConvolutionContext ctx = rock::populateConvContext(cop);
  auto strideDims = ctx.getStrideVal();
  auto dilationDims = ctx.getDilationVal();
  auto filterDims = ctx.getConvDims().fil;
  auto numKernels =
      rock::backwardDataKernelIds(strideDims, dilationDims, filterDims,
                                  /*usesV4R1=*/true);
  // If there is no zeroinit kernel needed, then there is nothing more we need
  // to do here.
  if (!rock::isEveryElementWrittenBwdData(strideDims, dilationDims,
                                          filterDims)) {
    // FIXME: don't hard code this - see PR#1687
    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    Attribute outputInitVal;
    Type funcResType = func.getFunctionType().getResult(0);
    auto shapedResType = cast<ShapedType>(funcResType);
    Type elementType = shapedResType.getElementType();
    if (isa<FloatType>(elementType)) {
      outputInitVal = rewriter.getFloatAttr(elementType, 0.0);
    } else if (isa<IntegerType>(elementType)) {
      outputInitVal = rewriter.getIntegerAttr(elementType, 0);
    } else {
      // We only expect integer and float types for now
      assert(false && "Unsupported element type for prefill attribute");
    }

    func.setResultAttr(0, rock::PrefillAttr::getMnemonic(), outputInitVal);
  }

  if (hasPadding) {
    assert(extractSlicePadding && collapseGroupPadding &&
           "these op should have be found from before");
    SmallVector<ReassociationIndices, 4> reassocations{{0}, {1, 2}};
    llvm::transform(llvm::seq<int64_t>(3, 3 + conv.spatialDim),
                    std::back_inserter(reassocations),
                    [](int64_t index) { return ReassociationIndices{index}; });
    tensor::CollapseShapeOp collapseGroupDim =
        tensor::CollapseShapeOp::create(rewriter, loc, output, reassocations);
    rewriter.eraseOp(op);
    rewriter.eraseOp(collapseGroupPadding);
    rewriter.replaceOp(extractSlicePadding, collapseGroupDim);
    return success();
  }

  rewriter.replaceOp(op, output);
  return success();
}

LogicalResult ConvLinalgConverter::matchAndRewrite(
    linalg::GenericOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  FailureOr<ConvFields> maybeConv = isConv(rewriter, op);
  if (failed(maybeConv))
    return failure();

  ConvFields conv = *maybeConv;
  Location loc = op.getLoc();

  // Making sure this is a forward conv only
  switch (conv.type) {
  case rock::LinalgConvType::Conv1dNgchGfch:
  case rock::LinalgConvType::Conv2dNgchwGfchw:
  case rock::LinalgConvType::Conv3dNgchwdGfchwd:
    break;
  default:
    return failure();
  }

  auto maybeInput =
      removePaddingFromInput(rewriter, op, op.getOperand(0), conv.padding);
  if (failed(maybeInput))
    return failure();

  Value input = *maybeInput;
  Value filter = op.getOperand(1);

  // Conv1D is expanded into Conv2D: unmerge the single spatial dim
  // into (spatial, W=1) for filter and input.
  int64_t effectiveSpatialDim = conv.spatialDim;
  if (conv.spatialDim == 1) {
    effectiveSpatialDim = 2;
    auto filterShape = cast<RankedTensorType>(filter.getType()).getShape();
    rock::BottomUpTMBuilder builder(rewriter, {"g", "k", "c", "0"}, filterShape,
                                    loc);
    builder.passThrough({"gf", "kf", "cf"}, {0, 1, 2}, {"g", "k", "c"});
    builder.unmerge({"0f", "1f"}, {3, 4}, "0", {filterShape[3], 1});
    filter = rock::TransformOp::create(rewriter, loc, filter, builder.get());

    auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
    rock::BottomUpTMBuilder b(rewriter, {"n", "g", "c", "0"}, inputShape, loc);
    b.passThrough({"nu", "gu", "cu"}, {0, 1, 2}, {"n", "g", "c"});
    b.unmerge({"0u", "1u"}, {3, 4}, "0", {inputShape[3], 1});
    input = rock::TransformOp::create(rewriter, loc, input, b.get());
  }

  RankedTensorType linalgResultType =
      cast<RankedTensorType>(op.getResult(0).getType());
  SmallVector<int64_t> rockShape(linalgResultType.getShape());
  if (conv.spatialDim == 1)
    rockShape.push_back(1);
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
  setConvLayoutAttrs(rewriter, cop, effectiveSpatialDim);

  Value result = cop.getResult();
  if (conv.spatialDim == 1) {
    auto shape = cast<RankedTensorType>(result.getType()).getShape();
    rock::BottomUpTMBuilder b(rewriter, {"n", "g", "k", "0", "1"}, shape, loc);
    b.passThrough({"no", "go", "ko"}, {0, 1, 2}, {"n", "g", "k"});
    b.merge("0o", 3, {"0", "1"});
    result = rock::TransformOp::create(rewriter, loc, result, b.get());
  }

  rewriter.replaceOp(op, result);
  return success();
}

void mlir::rock::populateLinalgToRockConversionPattern(
    RewritePatternSet &pattern, MLIRContext *context) {
  pattern.add<MatmulConverter<linalg::BatchMatmulOp>,
              MatmulConverter<linalg::MatmulOp>, ConvLinalgConverter,
              BwdConvLinalgConverter>(context);
}
