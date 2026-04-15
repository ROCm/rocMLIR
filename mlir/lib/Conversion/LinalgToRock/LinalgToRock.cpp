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
  if (isa<linalg::GenericOp>(op) && op->hasAttr("rock.quant_dot") &&
      op.getInputs().size() == 4 && op.getOutputs().size() == 1) {
    // The linalg.generic op from migraphx-to-linalg place this operand in this
    // way.
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
      return op.emitError("cannot determine if input matrix is transposed");
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

  // TODO: see (AIROCMLIR-696)
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
      /*cTransposed=*/nullptr,
      /*aScaleTransposed=*/context.aScaleTransposedAttr,
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
  case rock::LinalgConvType::Conv1dBWDNgchGckh:
  case rock::LinalgConvType::Conv1dNgchGkch:
    return 1;
  case rock::LinalgConvType::Conv2dBWDNgchwGckhw:
  case rock::LinalgConvType::Conv2dNgchwGkchw:
    return 2;
  case rock::LinalgConvType::Conv3dBWDNgchwdGckhwd:
  case rock::LinalgConvType::Conv3dNgchwdGkchwd:
    return 3;
  default:
    llvm_unreachable("unknown LinalgConvType");
  }
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

  // The input layout in the operand of linalg.generic is NGC*, and
  // the filter layout is GKC*. We have to transfer these attribute
  // because later on in the pass, ConvToGemm expect them to be attached.
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
  auto padded = (expanded != nullptr)
                    ? expanded->getOperand(0).getDefiningOp<tensor::PadOp>()
                    : nullptr;
  // We require padding here to have one use because the code structure emitted
  // by the MIGraphX -> Linalg have one use. In theory, you don't need this
  // check, but better be safe than sorry. This goes with expanded as well
  if (!padded || !padded->hasOneUse()) {
    op.emitError("unexpected padding code structure");
    return failure();
  }

  if (!expanded || !expanded->hasOneUse()) {
    return op.emitError("unexpected group expansion shape code structure");
  }

  SmallVector<int64_t> resultShape(expanded.getResultType().getShape());
  // The tensor.pad operand has no group dimension: [N, G*C, spatial...].
  // The expanded result has [N, G, C, spatial_padded...]. Take the first 3
  // dims (N, G, C) from the expanded shape and append the unpadded spatial
  // dims directly from the pad source starting at position 2.
  auto padSourceShape =
      cast<RankedTensorType>(padded.getOperand(0).getType()).getShape();
  resultShape.resize(3);
  resultShape.insert(resultShape.begin() + 3, padSourceShape.begin() + 2,
                     padSourceShape.end());

  RankedTensorType newResultType = RankedTensorType::get(
      resultShape, padded.getResultType().getElementType());
  Value result = tensor::ExpandShapeOp::create(
      rewriter, expanded.getLoc(), newResultType, padded.getOperand(0),
      expanded.getReassociationIndices());
  // erase the operations as well
  rewriter.eraseOp(expanded);
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
  auto name =
      op->getAttrOfType<rock::LinalgConvTypeAttr>(rock::linalgConvOpAttrName);
  if (!name)
    return failure();
  rock::LinalgConvType convType = name.getValue();
  int64_t spatialDim = getSpatialDim(convType);
  // Conv1D is broadcasted into Conv2D. To check for error, we
  // use effectiveDim instead because it one has more stride/dilation
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
  if (!dilation || static_cast<int64_t>(dilation.size()) != effectiveDim) {
    op.emitError("invalid dilation");
    return failure();
  }

  if (!stride || static_cast<int64_t>(stride.size()) != effectiveDim) {
    op.emitError("invalid stride");
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

  // Making sure this is a backwards conv only
  switch (conv.type) {
  case rock::LinalgConvType::Conv1dBWDNgchGckh:
    return op.emitError("conv1d backward conv is not supported for now");
  case rock::LinalgConvType::Conv2dBWDNgchwGckhw:
  case rock::LinalgConvType::Conv3dBWDNgchwdGckhwd:
    break;
  default:
    return failure();
  }
  bool hasPadding = llvm::any_of(conv.padding, [](Attribute attr) {
    return cast<IntegerAttr>(attr).getInt() != 0;
  });

  RankedTensorType resultShape =
      cast<RankedTensorType>(adaptor.getOutputs()[0].getType());
  tensor::ExtractSliceOp extractSlicePadding = nullptr;
  tensor::CollapseShapeOp collapseGroupPadding = nullptr;
  if (hasPadding) {
    // To handle padding, the migraphx to linalg pipeline
    // emits code that looks like the following:
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
  if (conv.perfConfig)
    cop->setAttr("perf_config", conv.perfConfig);
  setConvLayoutAttrs(rewriter, cop, getSpatialDim(conv.type));

  rock::ConvolutionContext ctx = rock::populateConvContext(cop);
  auto strideDims = ctx.getStrideVal();
  auto dilationDims = ctx.getDilationVal();
  auto filterDims = ctx.getConvDims().fil;
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
      return op.emitError("unsupported element type for prefill attribute");
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

  rewriter.replaceOp(op, cop);
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
  case rock::LinalgConvType::Conv1dNgchGkch:
  case rock::LinalgConvType::Conv2dNgchwGkchw:
  case rock::LinalgConvType::Conv3dNgchwdGkchwd:
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
  // TODO: add splitk see (AIROCMLIR-696)
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
              MatmulConverter<linalg::MatmulOp>, ExpandStrideConverter,
              MatmulConverter<linalg::GenericOp>, ConvLinalgConverter,
              BwdConvLinalgConverter>(context);
}
