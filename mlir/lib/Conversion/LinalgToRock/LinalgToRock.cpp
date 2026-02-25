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
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringMap.h"

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
}

namespace {
struct ConvFields {
  rock::LinalgConvType type;
  ArrayAttr padding, stride, dilation;
  StringAttr perfConfig;
};

struct ConvLinalgConverter final
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;
  using OpConversionPattern<linalg::GenericOp>::getTypeConverter;
  using OpAdaptor = typename OpConversionPattern<linalg::GenericOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// Returns strides, dilation, and padding if any
  FailureOr<ConvFields> isConv(ConversionPatternRewriter &rewriter,
                               linalg::GenericOp op) const;
};
} // namespace

FailureOr<ConvFields>
ConvLinalgConverter::isConv(ConversionPatternRewriter &rewriter,
                            linalg::GenericOp op) const {
  // FIXME: In the future, it is possible to extract strides, dilation, and
  // padding by matching the AffineExpr syntax tree. We can also infer the
  // dimension and layout of the convolution from the affine_map.
  rock::LinalgConvTypeAttr name = op->getAttrOfType<rock::LinalgConvTypeAttr>("conv_op");
  if (!name) {
    return failure();
  }
  rock::LinalgConvType convType = name.getValue();

  auto convertToArrayAttr =
      [&](Attribute arr, ArrayRef<int64_t> dimOneDefaults = {}) -> ArrayAttr {
        ArrayAttr casted = dyn_cast<ArrayAttr>(arr);
    SmallVector<int64_t, 4> values;
    llvm::transform(casted.getValue(), std::back_inserter(values),
                    [&](Attribute val) { return cast<IntegerAttr>(val).getInt(); });
    if (convType == rock::LinalgConvType::Conv1dNgchGfch) {
      values.insert(values.end(), dimOneDefaults.begin(), dimOneDefaults.end());
    }
    return rewriter.getIndexArrayAttr(values);
  };

  auto dilation =
      convertToArrayAttr(op->getAttr("dilation"), /*dimOneDefaults=*/1);
  auto stride = convertToArrayAttr(op->getAttr("stride"), /*dimOneDefaults=*/1);

  // We are given padding in format [dim0low, dim1low, ..., dim1high,
  // dim2high,...] but rock expects [dim0low, dim1low, dim2low, ...]
  SmallVector<Attribute, 4> newPaddingOrder;
  auto originalPaddingOrder = convertToArrayAttr(op->getAttr("pad")).getValue();
  int64_t dim = originalPaddingOrder.size() / 2;
  for (int64_t i = 0; i < dim; ++i) {
    newPaddingOrder.push_back(originalPaddingOrder[i]);
    newPaddingOrder.push_back(originalPaddingOrder[i]);
  }
  if (convType == rock::LinalgConvType::Conv1dNgchGfch) {
    newPaddingOrder.push_back(rewriter.getIndexAttr(0));
    newPaddingOrder.push_back(rewriter.getIndexAttr(0));
  }
  auto padding = rewriter.getArrayAttr(newPaddingOrder);
  if (!padding || !dilation || !stride) {
    return failure();
  }

  StringAttr perfConfig = op->getAttrOfType<StringAttr>("perf_config");
  return ConvFields{convType, padding, stride, dilation, perfConfig};
}

LogicalResult ConvLinalgConverter::matchAndRewrite(
    linalg::GenericOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  FailureOr<ConvFields> maybeConvParams = isConv(rewriter, op);
  if (failed(maybeConvParams))
    return failure();

  ConvFields convParams = maybeConvParams.value();
  Location loc = op.getLoc();

  // We have layout filter = GFC* but we need GF*C
  auto getFilter = [&](Value startFilter) -> Value {
    ArrayRef<int64_t> startFilterShape =
        cast<RankedTensorType>(startFilter.getType()).getShape();
    int64_t dim = startFilterShape.size() - 3;
    switch (dim) {
    case 3: {
      rock::BottomUpTMBuilder filterBuilder(
          rewriter, {"g", "f", "c", "h", "w", "d"}, startFilterShape, loc);
      filterBuilder.passThrough({"gk", "fk"}, {0, 1}, {"g", "f"});
      filterBuilder.passThrough({"hk", "wk", "dk"}, {2, 3, 4}, {"h", "w", "d"});
      filterBuilder.passThrough({"ck"}, {5}, {"c"});
      auto attr = filterBuilder.get();
      auto filter = rock::TransformOp::create(rewriter, loc, startFilter, attr);
      return filter;
    }
    case 2: {
      return startFilter;
    }
    case 1: {
      rock::BottomUpTMBuilder filterBuilder(rewriter, {"g", "f", "c", "h"},
                                            startFilterShape, loc);
      filterBuilder.passThrough({"gk", "fk"}, {0, 1}, {"g", "f"});
      filterBuilder.unmerge({"hk", "wk"}, {2, 3}, {"h"},
                            {startFilterShape[3], 1});
      filterBuilder.passThrough({"ck"}, {4}, {"c"});
      auto attr = filterBuilder.get();
      auto filter = rock::TransformOp::create(rewriter, loc, startFilter, attr);
      return filter;
    }
    default:
      llvm_unreachable("seen unsupported cases");
    }
  };

  // We have input filter = NGC* but we need N*GC
  auto getInput = [&](Value in) -> FailureOr<Value> {
    // dealing with padding
    if (llvm::any_of(convParams.padding.getValue(), [](Attribute attr) {
          return cast<IntegerAttr>(attr).getInt() != 0;
        })) {
      // clang-format off
      // Here we are essentially removing the padding while keeping the group
      // dimension expansion. We remove the padding because the rock.conv handles
      // padding for us This code structure comes from what migraphx-to-linalg
      // emits. In theory, there can be other code structure that are emitted in 
      // linalg pipeline to handle padding.
      // Original: 
      // %padded = tensor.pad %original ... 
      // %group_expansion = tensor.expand_shape %padded ... 
      // New: 
      // %group_expansion = tensor.expand_shape %original
      // clang-format on
      if (auto expanded = in.getDefiningOp<tensor::ExpandShapeOp>();
          auto padded =
              expanded->getOperand(0).getDefiningOp<tensor::PadOp>()) {
        SmallVector<int64_t, 6> resultShape(
            expanded.getResultType().getShape());
        auto lowPad = padded.getStaticLow();
        auto highPad = padded.getStaticHigh();
        int64_t numPadDims = lowPad.size();
        int64_t numExpandedDims = resultShape.size();

        // Padding is defined in pre-expand space. The spatial dims are at the
        // tail of both the pre-expand and post-expand tensors (expand_shape
        // only splits an earlier dim), so align from the end.
        for (int64_t i = numPadDims - 1, j = numExpandedDims - 1;
             i >= 0 && j >= 0; --i, --j) {
          resultShape[j] -= (lowPad[i] + highPad[i]);
        }

        RankedTensorType newResultType = RankedTensorType::get(
            resultShape, padded.getResultType().getElementType());
        auto temp = padded.getOperand(0);
        in = tensor::ExpandShapeOp::create(rewriter, expanded.getLoc(),
                                           newResultType, temp,
                                           expanded.getReassociationIndices());
        rewriter.replaceOp(expanded, in);
        rewriter.eraseOp(padded);
      } else {
        op.emitError("unexpected padding code structure");
        return failure();
      }
    }

    ArrayRef<int64_t> startInputShape =
        cast<RankedTensorType>(in.getType()).getShape();
    int64_t dim = startInputShape.size() - 3;
    switch (dim) {
    case 3: {
      rock::BottomUpTMBuilder inputBuilder(
          rewriter, {"n", "g", "c", "h", "w", "d"}, startInputShape, loc);
      inputBuilder.passThrough({"ni"}, {0}, {"n"});
      inputBuilder.passThrough({"hi", "wi", "di"}, {1, 2, 3}, {"h", "w", "d"});
      inputBuilder.passThrough({"gi", "ci"}, {4, 5}, {"g", "c"});
      auto inputAttr = inputBuilder.get();
      auto input = rock::TransformOp::create(rewriter, loc, in, inputAttr);
      return input.getResult();
    }
    case 2: {
      return in;
    }
    case 1: {
      // migraphx-to-tosa pipeline handles 1d convolution by converting
      // 1 dimensional input into 2 dimensional. 1x1x3x10 (NGCH) becomes
      // 1x1x3x1x10 (NHWGC). We are reproducing that here
      int64_t h = startInputShape[3];
      rock::BottomUpTMBuilder filterBuilder(rewriter, {"n", "g", "c", "h"},
                                            startInputShape, loc);
      filterBuilder.passThrough({"ni"}, {0}, {"n"});
      filterBuilder.unmerge({"hi", "wi"}, {1, 2}, {"h"}, {h, 1});
      filterBuilder.passThrough({"gi", "ci"}, {3, 4}, {"g", "c"});
      auto attr = filterBuilder.get();
      return rock::TransformOp::create(rewriter, loc, in, attr).getResult();
    }
    default:
      llvm_unreachable("unsupported cases");
    }
  };

  // Creating the final result shape
  RankedTensorType linalgResultType =
      cast<RankedTensorType>(op.getResult(0).getType());
  ArrayRef<int64_t> linalgOutputShape = linalgResultType.getShape();
  SmallVector<int64_t, 4> rockOutputShape(linalgOutputShape);
  if (linalgOutputShape.size() - 3 == 3 || linalgOutputShape.size() - 3 == 1) {
    rockOutputShape.clear();
    rockOutputShape.push_back(linalgOutputShape[0]);
    rockOutputShape.insert(rockOutputShape.end(),
                           std::next(linalgOutputShape.begin(), 3),
                           linalgOutputShape.end());
    if (linalgOutputShape.size() - 3 == 1)
      rockOutputShape.push_back(1);
    rockOutputShape.push_back(linalgOutputShape[1]);
    rockOutputShape.push_back(linalgOutputShape[2]);
  }
  RankedTensorType rockResultType =
      RankedTensorType::get(rockOutputShape, linalgResultType.getElementType());
  Value output = bufferization::AllocTensorOp::create(rewriter, op.getLoc(),
                                                      rockResultType, {});

  auto maybeInput = getInput(op.getOperand(0));
  if (failed(maybeInput)) {
    return failure();
  }
  auto input = *maybeInput;
  auto filter = getFilter(op.getOperand(1));
  auto cop = rock::ConvOp::create(rewriter, loc, rockResultType, filter, input,
                                  output, /*features=*/nullptr,
                                  /*blockSize=*/nullptr, /*gridSize=*/nullptr,
                                  convParams.padding, convParams.stride,
                                  convParams.dilation, /*params=*/nullptr);
  // TODO: add splitk
  if (convParams.perfConfig) {
    cop->setAttr("perf_config", convParams.perfConfig);
  }

  // Here we are going to emit layouts
  switch (convParams.type) {
    case rock::LinalgConvType::Conv3dNgchwdGfchwd:
    cop->setAttr("filter_layout",
                 rewriter.getStrArrayAttr({"g", "k", "0", "1", "2", "c"}));
    cop->setAttr("input_layout", rewriter.getStrArrayAttr(
                                     {"ni", "0i", "1i", "2i", "gi", "ci"}));
    cop->setAttr("output_layout", rewriter.getStrArrayAttr(
                                      {"no", "0o", "1o", "2o", "go", "ko"}));
    break;
    case rock::LinalgConvType::Conv2dNgchwGfchw:
    cop->setAttr("filter_layout",
                 rewriter.getStrArrayAttr({"g", "k", "c", "y", "x"}));
    cop->setAttr("input_layout",
                 rewriter.getStrArrayAttr({"ni", "gi", "ci", "hi", "wi"}));
    cop->setAttr("output_layout",
                 rewriter.getStrArrayAttr({"no", "go", "ko", "ho", "wo"}));
    break;
    case rock::LinalgConvType::Conv1dNgchGfch:
    cop->setAttr("filter_layout",
                 rewriter.getStrArrayAttr({"g", "k", "y", "x", "c"}));
    cop->setAttr("input_layout",
                 rewriter.getStrArrayAttr({"ni", "hi", "wi", "gi", "ci"}));
    cop->setAttr("output_layout",
                 rewriter.getStrArrayAttr({"no", "ho", "wo", "go", "ko"}));
    break;
  default:
    llvm_unreachable("edge case one");
  }

  // output has type ["no", "0o", "1o", "2o", "go", "ko"]
  // We need to reshape to ngfhwd
  ArrayRef<int64_t> startResultShape = rockResultType.getShape();
  Value finalReshaped;
  switch (convParams.type) {
    case rock::LinalgConvType::Conv3dNgchwdGfchwd: {
    rock::BottomUpTMBuilder resultBuilder(
        rewriter, {"n", "h", "w", "d", "g", "f"}, startResultShape, loc);
    resultBuilder.passThrough({"go", "fo"}, {1, 2}, {"g", "f"});
    resultBuilder.passThrough({"no"}, {0}, {"n"});
    resultBuilder.passThrough({"ho", "wo", "do"}, {3, 4, 5}, {"h", "w", "d"});
    auto resultAttr = resultBuilder.get();
    finalReshaped =
        rock::TransformOp::create(rewriter, loc, cop.getResult(), resultAttr);
    break;
  }
    case rock::LinalgConvType::Conv2dNgchwGfchw: {
    finalReshaped = cop.getResult();
    break;
  }
  case rock::LinalgConvType::Conv1dNgchGfch: {
    rock::BottomUpTMBuilder resultBuilder(rewriter, {"n", "h", "w", "g", "f"},
                                          startResultShape, loc);
    resultBuilder.passThrough({"no"}, {0}, {"n"});
    resultBuilder.passThrough({"go", "fo"}, {1, 2}, {"g", "f"});
    resultBuilder.merge("ho", 3, {"h", "w"});
    auto resultAttr = resultBuilder.get();
    finalReshaped =
        rock::TransformOp::create(rewriter, loc, cop.getResult(), resultAttr);
    break;
  }
  default: {
    return op.emitError("unimplemented final reshape");
  }
  }

  rewriter.replaceOp(op, finalReshaped);
  return success();
}

void mlir::rock::populateLinalgToRockConversionPattern(
    RewritePatternSet &pattern, MLIRContext *context) {
  pattern.add<MatmulConverter<linalg::BatchMatmulOp>,
              MatmulConverter<linalg::MatmulOp>, ExpandStrideConverter,
              MatmulConverter<linalg::GenericOp>, ConvLinalgConverter>(
      context);
}
