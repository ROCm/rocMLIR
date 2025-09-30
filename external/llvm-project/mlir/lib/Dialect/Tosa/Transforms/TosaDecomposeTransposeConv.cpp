//===- TosaDecomposeTransposeConv.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decompose TOSA TransposeConv operation to a series of TOSA Ops specifically
// (1) Convert a Dilated TransposeConv2D to Conv2D including reversing/reshaping
// etc.. of the weights (2) Convert a Strided TransposeConv2D to Conv2D
// including transposing/reversing/reshaping etc..
//     of the weights and input/output tenors and reversing/reshaping etc .. of
//     the weights
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

// If this is a backward-data (transpose) conv lowered from MIGraphX, its
// filter logical in/out channels are reversed relative to forward Conv2D.
FailureOr<std::tuple<Value, ShapedType>>
swapInputOutputDimensions(OpBuilder &rewriter, Operation *op,
                          Value weight, ShapedType weightTy) {
  if (auto kindAttr = op->getAttrOfType<StringAttr>("conv_kind");
      kindAttr && kindAttr.getValue() == "bwd_data") {
    // Expected current shape: [K, H, W, C] but Conv2D expects [O, H, W, I]
    // Swap K<->C => permutation {3,1,2,0}.
    auto wShape = weightTy.getShape();
    SmallVector<int64_t, 4> swappedShape{
        wShape[3], // C becomes O
        wShape[1],
        wShape[2],
        wShape[0]  // K becomes I
    };
    auto swappedTy =
        RankedTensorType::get(swappedShape, weightTy.getElementType());
    weight = rewriter.create<tosa::TransposeOp>(
        op->getLoc(), swappedTy, weight,
        rewriter.getDenseI32ArrayAttr({3, 1, 2, 0}));
    weightTy = cast<ShapedType>(weight.getType());

    return std::make_tuple(weight, weightTy);
  }

  return failure();
}

class TransposeConvNonStridedConverter
    : public OpRewritePattern<tosa::TransposeConv2DOp> {
public:
  using OpRewritePattern<tosa::TransposeConv2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::TransposeConv2DOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = cast<ShapedType>(input.getType());
    ShapedType weightTy = cast<ShapedType>(weight.getType());
    ShapedType biasTy = cast<ShapedType>(bias.getType());
    ShapedType resultTy = cast<ShapedType>(op->getResult(0).getType());

    llvm::ArrayRef<int64_t> stride = op.getStride();
    llvm::ArrayRef<int64_t> pad = op.getOutPad();

    // Fetch dilation (default {1,1})
    SmallVector<int64_t, 2> dilationVals = {1, 1};
    if (auto dilOpt = op.getDilation()) {
      dilationVals[0] = (*dilOpt)[0];
      dilationVals[1] = (*dilOpt)[1];
    }

    // Fetch input pads (default zeros)
    SmallVector<int64_t, 4> inPadVals(4, 0);
    if (auto padOpt = op.getPad()) {
      inPadVals[0] = (*padOpt)[0]; // top
      inPadVals[1] = (*padOpt)[1]; // bottom
      inPadVals[2] = (*padOpt)[2]; // left
      inPadVals[3] = (*padOpt)[3]; // right
    }

    // If striding is all 1 we can modify padding and reverse the kernel along
    // the x/y direction to make it a regular convolution. This is much simpler
    // then handling striding....
    if (llvm::any_of(stride, [](int64_t v) { return v != 1; }))
      return failure();

    if (!inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        !biasTy.hasStaticShape() || !resultTy.hasStaticShape())
      return failure();

    // Swap dimensions if needed
    auto swapOr = swapInputOutputDimensions(rewriter, op, weight, weightTy);
    if (succeeded(swapOr)) {
      weight = std::get<0>(swapOr.value());
      weightTy = std::get<1>(swapOr.value());
    }

    int64_t kernelHeight = weightTy.getDimSize(1);
    int64_t kernelWidth = weightTy.getDimSize(2);
    int64_t effKHm1 = (kernelHeight - 1) * dilationVals[0];
    int64_t effKWm1 = (kernelWidth - 1) * dilationVals[1];

    // Conv2D padding derived from ConvTranspose (ONNX/PyTorch style)
    // convPadTop    = effKHm1 - inPadTop    + outPadTop
    // convPadBottom = effKHm1 - inPadBottom + outPadBottom
    // convPadLeft   = effKWm1 - inPadLeft   + outPadLeft
    // convPadRight  = effKWm1 - inPadRight  + outPadRight
    SmallVector<int64_t, 4> convPad = {
        effKHm1 - inPadVals[0] + pad[0],
        effKHm1 - inPadVals[1] + pad[1],
        effKWm1 - inPadVals[2] + pad[2],
        effKWm1 - inPadVals[3] + pad[3]
    };

    bool needSlice = false;
    SmallVector<int64_t,4> negExcess(4,0);
    for (int i=0;i<4;++i) {
      if (convPad[i] < 0) {
        negExcess[i] = -convPad[i];
        convPad[i] = 0;
        needSlice = true;
      }
    }

    if (needSlice)
      return rewriter.notifyMatchFailure(op, "Cannot currently handle negative "
                                             "padding values.");

    auto reverse1 =
        tosa::ReverseOp::create(rewriter, loc, weightTy, weight,
                                /* axis = */ rewriter.getI32IntegerAttr(1));
    auto reverse2 =
        tosa::ReverseOp::create(rewriter, loc, weightTy, reverse1,
                                /* axis = */ rewriter.getI32IntegerAttr(2));

    Value conv2d = tosa::Conv2DOp::create(
        rewriter, loc, resultTy, input, reverse2, bias, op.getInputZp(),
        op.getWeightZp(), rewriter.getDenseI64ArrayAttr(convPad),
        rewriter.getDenseI64ArrayAttr(stride),
        rewriter.getDenseI64ArrayAttr(dilationVals),
        /* acc_type = */ op.getAccType(),
        op->getAttrOfType<IntegerAttr>("group"));

    rewriter.replaceOp(op, conv2d);
    return success();
  }
};

class TransposeConvStridedConverter
    : public OpRewritePattern<tosa::TransposeConv2DOp> {
public:
  using OpRewritePattern<tosa::TransposeConv2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::TransposeConv2DOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = cast<ShapedType>(input.getType());
    ShapedType weightTy = cast<ShapedType>(weight.getType());
    ShapedType biasTy = cast<ShapedType>(bias.getType());
    ShapedType resultTy = cast<ShapedType>(op->getResult(0).getType());

    Type inputETy = inputTy.getElementType();
    Type weightETy = weightTy.getElementType();
    Type biasETy = biasTy.getElementType();
    Type resultETy = resultTy.getElementType();

    llvm::ArrayRef<int64_t> pad = op.getOutPad();
    llvm::ArrayRef<int64_t> stride = op.getStride();

    // Swap dimensions if needed
    auto swapOr = swapInputOutputDimensions(rewriter, op, weight, weightTy);
    if (succeeded(swapOr)) {
      weight = std::get<0>(swapOr.value());
      weightTy = std::get<1>(swapOr.value());
    }

    // Fetch dilation (default {1,1})
    SmallVector<int64_t, 2> dilationVals = {1, 1};
    if (auto dilOpt = op.getDilation()) {
      dilationVals[0] = (*dilOpt)[0];
      dilationVals[1] = (*dilOpt)[1];
    }

    // Fetch input padding (default {0, 0, 0, 0})
    SmallVector<int64_t, 4> inPadVals = {0, 0, 0, 0};
    if (auto inPadOpt = op.getPad()) {
      inPadVals[0] = (*inPadOpt)[0];
      inPadVals[1] = (*inPadOpt)[1];
      inPadVals[2] = (*inPadOpt)[2];
      inPadVals[3] = (*inPadOpt)[3];
    }

    // If strides are all 1 we dont need to use this one.
    if (llvm::all_of(stride, [](int64_t v) { return v == 1; }))
      return rewriter.notifyMatchFailure(op, "non-one stride found.");

    if (!inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        !biasTy.hasStaticShape() || !resultTy.hasStaticShape())
      return failure();

    int64_t batch = inputTy.getDimSize(0);

    int64_t outputChannels = weightTy.getDimSize(0);
    int64_t weightHeight = weightTy.getDimSize(1);
    int64_t weightWidth = weightTy.getDimSize(2);
    int64_t inputChannels = weightTy.getDimSize(3);

    // Get and verify zero points.
    FailureOr<int64_t> maybeIZp = op.getInputZeroPoint();
    if (failed(maybeIZp))
      return rewriter.notifyMatchFailure(
          op, "input zero point cannot be statically determined");

    FailureOr<int64_t> maybeWZp = op.getWeightZeroPoint();
    if (failed(maybeWZp))
      return rewriter.notifyMatchFailure(
          op, "weight zero point cannot be statically determined");

    int64_t inputZpVal = *maybeIZp;
    int64_t weightZpVal = *maybeWZp;

    if (op.verifyInputZeroPoint(inputZpVal).failed())
      return rewriter.notifyMatchFailure(
          op, "input zero point must be zero for non-int8 integer types");

    if (op.verifyWeightZeroPoint(weightZpVal).failed())
      return rewriter.notifyMatchFailure(
          op, "weight zero point must be zero for non-int8 integer types");

    // construct pad_const values from zp values
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    const Value inputPadConst =
        createPadConstTensor(builder, op->getLoc(), input, inputZpVal);
    const Value weightPadConst =
        createPadConstTensor(builder, op->getLoc(), input, weightZpVal);

    // Helper to create a scalar pad value (weight zero-point)
    auto createWeightPadConst = [&](Value like) -> Value {
      return createPadConstTensor(builder, loc, like, weightZpVal);
    };

    // Explicitly materialize dilation in the weight tensor by inserting
    // (d-1) rows / columns of zero between the original kernel rows and
    // columns. After this expansion we can treat dilation as 1
    // for the remainder of the lowering.
    if (dilationVals[0] > 1 || dilationVals[1] > 1) {
      int64_t dH = dilationVals[0];
      int64_t dW = dilationVals[1];

      // Expand height: iterate original over original rows, slice each 1-row
      // slab, and (except for the final row) pad (dH-1) zero rows below it.
      SmallVector<Value> heightPieces;
      for (int64_t h = 0; h < weightHeight; ++h) {
        llvm::SmallVector<int64_t, 4> begin = {0, h, 0, 0};
        llvm::SmallVector<int64_t, 4> size = {outputChannels, 1, weightWidth,
                                              inputChannels};
        Value beginVal = getTosaConstShape(rewriter, loc, begin);
        Value sizeVal = getTosaConstShape(rewriter, loc, size);
        Value slice = CreateOpAndInferShape<tosa::SliceOp>(
                          rewriter, loc, UnrankedTensorType::get(weightETy),
                          weight, beginVal, sizeVal)
                          .getResult();
        int64_t padRows = (h == weightHeight - 1) ? 0 : (dH - 1);
        if (padRows > 0) {
          llvm::SmallVector<int64_t, 8> padSpec =
                            {0, 0, 0, padRows, 0, 0, 0, 0};
          Value padSpecVal = getTosaConstShape(rewriter, loc, padSpec);
          slice = CreateOpAndInferShape<tosa::PadOp>(
                      rewriter, loc, UnrankedTensorType::get(weightETy), slice,
                      padSpecVal, createWeightPadConst(weight))
                      .getResult();
        }
        heightPieces.push_back(slice);
      }
      weight = CreateOpAndInferShape<tosa::ConcatOp>(
                   rewriter, loc, UnrankedTensorType::get(weightETy),
                   SmallVector<Value>(heightPieces.begin(), heightPieces.end()),
                   rewriter.getI32IntegerAttr(1))
                   .getResult();

      // Update dims after height expansion
      weightTy = cast<ShapedType>(weight.getType());
      weightHeight = weightTy.getDimSize(1); // now (origH-1)*dH + 1

      // Expand width similarly if horizontal dilation > 1.
      if (dW > 1) {
        SmallVector<Value> widthPieces;
        for (int64_t w = 0; w < weightWidth; ++w) {
          llvm::SmallVector<int64_t, 4> begin = {0, 0, w, 0};
          llvm::SmallVector<int64_t, 4> size = {outputChannels, weightHeight,
                                                1, inputChannels};
          Value beginVal = getTosaConstShape(rewriter, loc, begin);
          Value sizeVal = getTosaConstShape(rewriter, loc, size);
          Value slice = CreateOpAndInferShape<tosa::SliceOp>(
                            rewriter, loc, UnrankedTensorType::get(weightETy),
                            weight, beginVal, sizeVal)
                            .getResult();
          int64_t padCols = (w == weightWidth - 1) ? 0 : (dW - 1);
          if (padCols > 0) {
            llvm::SmallVector<int64_t, 8> padSpec =
                              {0, 0, 0, 0, 0, padCols, 0, 0};
            Value padSpecVal = getTosaConstShape(rewriter, loc, padSpec);
            slice = CreateOpAndInferShape<tosa::PadOp>(
                        rewriter, loc, UnrankedTensorType::get(weightETy),
                        slice, padSpecVal, createWeightPadConst(weight))
                        .getResult();
          }
          widthPieces.push_back(slice);
        }
        weight = CreateOpAndInferShape<tosa::ConcatOp>(
                     rewriter, loc, UnrankedTensorType::get(weightETy),
                     SmallVector<Value>(widthPieces.begin(), widthPieces.end()),
                     rewriter.getI32IntegerAttr(2))
                     .getResult();
      }

      // Update type/dims post width expansion.
      weightTy = cast<ShapedType>(weight.getType());
      weightWidth = weightTy.getDimSize(2);

      // After explicit expansion, treat dilation as 1 for the remainder.
      dilationVals = {1, 1};
    }

    // We want to capture the height and width values after dilation expansion,
    // but before padding is added later on.
    int64_t origWeightHeight = weightHeight;
    int64_t origWeightWidth  = weightWidth;

    // Pad the weight so that it is modulo of the striding.
    llvm::SmallVector<int64_t, 8> weightPadding = {0, 0, 0, 0, 0, 0, 0, 0};
    weightPadding[3] =
        (weightHeight % stride[0]) ? (stride[0] - weightHeight % stride[0]) : 0;
    weightPadding[5] =
        (weightWidth % stride[1]) ? (stride[1] - weightWidth % stride[1]) : 0;

    Value weightPaddingVal =
        getTosaConstShape(rewriter, op->getLoc(), weightPadding);

    weight = CreateOpAndInferShape<tosa::PadOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        weightPaddingVal, weightPadConst);

    weightTy = cast<ShapedType>(weight.getType());
    weightHeight = weightTy.getDimSize(1);
    weightWidth = weightTy.getDimSize(2);

    // Split out the width / height by the stride dimensions.
    llvm::SmallVector<int64_t, 6> weightReshapeDims0 = {
        outputChannels, weightHeight / stride[0],
        stride[0],      weightWidth / stride[1],
        stride[1],      inputChannels};

    weight = CreateOpAndInferShape<tosa::ReshapeOp>(
        builder, UnrankedTensorType::get(weightETy), weight,
        getTosaConstShape(rewriter, loc, weightReshapeDims0));

    // Transpose the factored-out stride to the output channels.
    weight = CreateOpAndInferShape<tosa::TransposeOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        rewriter.getDenseI32ArrayAttr({2, 4, 0, 1, 3, 5}));

    // Collapse the strides and output channels into a single dimension.
    llvm::SmallVector<int64_t, 4> weightReshapeDims1 = {
        outputChannels * stride[0] * stride[1], weightHeight / stride[0],
        weightWidth / stride[1], inputChannels};

    weight = CreateOpAndInferShape<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        getTosaConstShape(rewriter, loc, weightReshapeDims1));
    ShapedType restridedWeightTy = cast<ShapedType>(weight.getType());

    weight = CreateOpAndInferShape<tosa::ReverseOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        /* axis = */ rewriter.getI32IntegerAttr(1));
    weight = CreateOpAndInferShape<tosa::ReverseOp>(
        rewriter, loc, UnrankedTensorType::get(weightETy), weight,
        /* axis = */ rewriter.getI32IntegerAttr(2));

    // We need to pad the input far enough that we can pull all values.
    llvm::SmallVector<int64_t, 8> inputPadding = {0, 0, 0, 0, 0, 0, 0, 0};

    // If the op has input padding, make sure to use that. If not, default back
    // to using the legacy logic.
    if (op.getPad().has_value()) {
      inputPadding[2] = inPadVals[0];
      inputPadding[3] = inPadVals[1];
      inputPadding[4] = inPadVals[2];
      inputPadding[5] = inPadVals[3];
    } else {
      inputPadding[2] += restridedWeightTy.getDimSize(1) - 1;
      inputPadding[3] += restridedWeightTy.getDimSize(1) - 1;
      inputPadding[4] += restridedWeightTy.getDimSize(2) - 1;
      inputPadding[5] += restridedWeightTy.getDimSize(2) - 1;
    }

    Value inputPaddingVal =
        getTosaConstShape(rewriter, op->getLoc(), inputPadding);

    input = CreateOpAndInferShape<tosa::PadOp>(
        rewriter, loc, UnrankedTensorType::get(inputETy), input,
        inputPaddingVal, inputPadConst);

    // We use a zero bias as we need to broadcast the bias.
    auto zeroBias = tosa::ConstOp::create(
        rewriter, loc,
        RankedTensorType::get({outputChannels * stride[0] * stride[1]},
                              biasETy),
        DenseElementsAttr::get(
            RankedTensorType::get({outputChannels * stride[0] * stride[1]},
                                  biasETy),
            rewriter.getZeroAttr(biasETy)));

    auto inputZp =
        createZeroPointTensor(rewriter, loc, input.getType(), inputZpVal);
    auto weightZp =
        createZeroPointTensor(rewriter, loc, weight.getType(), weightZpVal);

    if (!inputZp.has_value() || !weightZp.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "fail to create a const zero point tensor");
    }

    // Perform the convolution using the zero bias.
    Value conv2d = CreateOpAndInferShape<tosa::Conv2DOp>(
                       rewriter, loc, UnrankedTensorType::get(resultETy), input,
                       weight, zeroBias, inputZp.value(), weightZp.value(),
                       /*pad=*/rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
                       /*stride=*/rewriter.getDenseI64ArrayAttr({1, 1}),
                       /*dilation=*/rewriter.getDenseI64ArrayAttr({1, 1}),
                       op.getAccType(), op->getAttrOfType<IntegerAttr>("group"))
                       .getResult();

    // Factor the resulting width / height.
    ShapedType convTy = cast<ShapedType>(conv2d.getType());
    Type convETy = convTy.getElementType();

    int64_t convHeight = convTy.getDimSize(1);
    int64_t convWidth = convTy.getDimSize(2);

    // Factor striding out of the convolution result.
    llvm::SmallVector<int64_t, 6> convReshapeDims0 = {
        batch, convHeight, convWidth, stride[0], stride[1], outputChannels};

    auto convReshapeDims0Value =
        getTosaConstShape(rewriter, loc, convReshapeDims0);

    conv2d = CreateOpAndInferShape<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(resultETy), conv2d,
        convReshapeDims0Value);

    // Transpose the factored-out stride to the output channels.
    conv2d = CreateOpAndInferShape<tosa::TransposeOp>(
        rewriter, loc, UnrankedTensorType::get(convETy), conv2d,
        rewriter.getDenseI32ArrayAttr({0, 1, 3, 2, 4, 5}));

    // Fuse striding behavior back into width / height.
    llvm::SmallVector<int64_t, 6> convReshapeDims1 = {
        batch, convHeight * stride[0], convWidth * stride[1], outputChannels};

    auto convReshapeDims1Value =
        getTosaConstShape(rewriter, loc, convReshapeDims1);

    conv2d = CreateOpAndInferShape<tosa::ReshapeOp>(
        rewriter, loc, UnrankedTensorType::get(resultETy), conv2d,
        convReshapeDims1Value);

    // Effective pad = outPad + (k - 1) - (inPad * stride)
    // Each input padded row/col expands to stride rows/cols in the upsampled
    // domain.
    int64_t effPadTop  = pad[0] + (origWeightHeight - stride[0]) - inPadVals[0]*stride[0];
    int64_t effPadLeft = pad[2] + (origWeightWidth  - stride[1]) - inPadVals[2]*stride[1];

    // When we shrink from the orignal size to kPrime by grouping stride phases,
    // we discard some positions that existed in the conceptual upsampled view.
    // The total span of the original field is kOrig -1, and the span
    // represented after factoring is kPrime - 1. The difference is the values
    // that have been lost
    int64_t kHPrime = restridedWeightTy.getDimSize(1);
    int64_t kWPrime = restridedWeightTy.getDimSize(2);
    auto lost = [](int64_t Korig, int64_t kPrime, int64_t S) {
      return (Korig - 1) - (kPrime - 1)*S;
    };
    int64_t lostH = lost(origWeightHeight, kHPrime, stride[0]);
    int64_t lostW = lost(origWeightWidth,  kWPrime, stride[1]);

    // If stride factoring compresses a dimension to a single spatial position,
    // i.e., kPrime == 1, then we dropped a ring of values around that position.
    // To keep the result centered, update effPad by the half the lost distance.
    if (kHPrime == 1 && lostH > 0)
      effPadTop += lostH / 2;
    if (kWPrime == 1 && lostW > 0)
      effPadLeft += lostW / 2;

    int64_t resultSliceTop;
    int64_t resultSliceLeft;
    int64_t resultPadTop;
    int64_t resultPadLeft;
    // Convert effective padding into slice (crop) and post-pad just like the
    // prior logic but now using effPad*.
    if (op.getPad().has_value()) {
      resultSliceTop = std::max<int64_t>(0, -effPadTop);
      resultSliceLeft = std::max<int64_t>(0, -effPadLeft);
      resultPadTop = std::max<int64_t>(0, effPadTop);
      resultPadLeft = std::max<int64_t>(0, effPadLeft);
    } else {
      // Default to using legacy logic if input padding is not present
      resultSliceTop = std::max<int64_t>(0, -pad[0]);
      resultSliceLeft = std::max<int64_t>(0, -pad[2]);
      resultPadTop = std::max<int64_t>(0, pad[0]);
      resultPadLeft = std::max<int64_t>(0, pad[2]);
    }

    // Try to slice the targetted result size, cap to the convolutions width.
    int64_t resultSliceHeight =
        std::min<int64_t>(convReshapeDims1[1] - resultSliceTop,
                          resultTy.getDimSize(1) - resultPadTop);
    int64_t resultSliceWidth =
        std::min<int64_t>(convReshapeDims1[2] - resultSliceLeft,
                          resultTy.getDimSize(2) - resultPadLeft);

    llvm::SmallVector<int64_t, 4> sliceBegin = {0, resultSliceTop,
                                                resultSliceLeft, 0};
    llvm::SmallVector<int64_t, 4> sliceSize(convReshapeDims1.begin(),
                                            convReshapeDims1.end());
    sliceSize[1] = resultSliceHeight;
    sliceSize[2] = resultSliceWidth;

    auto slice = CreateOpAndInferShape<tosa::SliceOp>(
                     rewriter, loc, UnrankedTensorType::get(resultETy), conv2d,
                     getTosaConstShape(rewriter, loc, sliceBegin),
                     getTosaConstShape(rewriter, loc, sliceSize))
                     .getResult();

    llvm::SmallVector<int64_t, 8> resultPadding = {0, 0, 0, 0, 0, 0, 0, 0};
    resultPadding[2] = resultPadTop;
    resultPadding[3] = resultTy.getDimSize(1) - resultPadTop - sliceSize[1];
    resultPadding[4] = resultPadLeft;
    resultPadding[5] = resultTy.getDimSize(2) - resultPadLeft - sliceSize[2];

    Value resultPaddingVal =
        getTosaConstShape(rewriter, op->getLoc(), resultPadding);

    Value resultPad = CreateOpAndInferShape<tosa::PadOp>(
        rewriter, loc, UnrankedTensorType::get(resultETy), slice,
        resultPaddingVal);

    if (EqualizeRanks(rewriter, op.getLoc(), resultPad, bias).failed()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, op.getType(), resultPad, bias);
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaDecomposeTransposeConv(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TransposeConvNonStridedConverter>(ctx);
  patterns.add<TransposeConvStridedConverter>(ctx);
}
