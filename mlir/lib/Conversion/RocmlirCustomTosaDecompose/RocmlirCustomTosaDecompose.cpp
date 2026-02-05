//===- RocmlirCustomTosaDecompose.cpp - Decompose custom Tosa ops --===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// This pass is a downstream version of
// mlir/lib/Dialect/Tosa/Transforms/TosaDecomposeTransposeConv.cpp
// (code was
// copied from rocMLIR commit ec067ce842b1580e02e222ec444b877f0f861e1b)
// with the added functionality of supporting transposeConv in 3D.
// Compared to upstream, we have the following changes:
// - We expect tosa::CustomOp instead of tosa::TransposeConv2DOp
// - We support 2D (like upstream) but also 3D transposed convolutions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RocmlirCustomTosaDecompose/RocmlirCustomTosaDecompose.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/RockTosaCustomOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_ROCMLIRCUSTOMTOSADECOMPOSEPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {
struct RocmlirCustomTosaDecomposePass
    : public impl::RocmlirCustomTosaDecomposePassBase<
          RocmlirCustomTosaDecomposePass> {
  void runOnOperation() override;
};

// This is mostly a copy of the verification op that exists for
// tosa::TransposeConv2DOp in upstream with the output shape checks updated
// to properly account for input padding and dilation.
// See here for the formula being used:
// https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
LogicalResult verifyConvTranspose(tosa::CustomOp op,
                                  PatternRewriter &rewriter) {
  // Make sure this is a transpose conv op.
  if (op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME)
    return rewriter.notifyMatchFailure(op, "domain isn't rocmlir");
  if (op.getOperatorName() != ROCK_CUSTOMOP_CONV_BWD_DATA)
    return rewriter.notifyMatchFailure(op, "isn't a conv_bwd_data");
  if (op.getNumOperands() < 5)
    return rewriter.notifyMatchFailure(op, "should have 5 or more operands");
  if (op.getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "should have 1 result");

  const auto outputType =
      llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!outputType)
    return failure();

  llvm::ArrayRef<int64_t> strides =
      cast<DenseI64ArrayAttr>(op->getAttr("stride"));
  int convDims = strides.size();

  // Right now we can only support 2D values
  if (convDims == 3)
    return failure();

  const int64_t strideY = strides[0];
  const int64_t strideX = strides[1];
  int64_t strideZ = 1;
  if (convDims == 3)
    strideZ = strides[2];

  if (strideY < 1 || strideX < 1 || strideZ < 1)
    return op.emitOpError("expect all stride values to be >= 1, got [")
           << strides << "]";

  const auto checkPadAgainstKernelDim =
      [&](int64_t pad_value, int64_t kernel_dim_size, llvm::StringRef pad_name,
          llvm::StringRef kernel_dim_name) -> LogicalResult {
    if (pad_value <= -kernel_dim_size)
      return op.emitOpError("expected ")
             << pad_name << " > -" << kernel_dim_name
             << ", but got: " << pad_name << "=" << pad_value << " and "
             << kernel_dim_name << "=" << kernel_dim_size;
    return success();
  };

  llvm::ArrayRef<int64_t> outPad =
      cast<DenseI64ArrayAttr>(op->getAttr("out_pad"));
  const int64_t outPadTop = outPad[0];
  const int64_t outPadBottom = outPad[1];
  const int64_t outPadLeft = outPad[2];
  const int64_t outPadRight = outPad[3];

  Value weight = op->getOperand(1);
  const auto weightType = llvm::dyn_cast<RankedTensorType>(weight.getType());

  if (weightType) {
    const int64_t kernelHeight = weightType.getDimSize(1);
    if (ShapedType::isStatic(kernelHeight)) {
      if (failed(checkPadAgainstKernelDim(outPadTop, kernelHeight,
                                          "out_pad_top", "KH")))
        return failure();

      if (failed(checkPadAgainstKernelDim(outPadBottom, kernelHeight,
                                          "out_pad_bottom", "KH")))
        return failure();
    }

    const int64_t kernelWidth = weightType.getDimSize(2);
    if (ShapedType::isStatic(kernelWidth)) {
      if (failed(checkPadAgainstKernelDim(outPadLeft, kernelWidth,
                                          "out_pad_left", "KW")))
        return failure();

      if (failed(checkPadAgainstKernelDim(outPadRight, kernelWidth,
                                          "out_pad_right", "KW")))
        return failure();
    }
  }

  // Fetch pad & dilation;
  SmallVector<int64_t, 2> dilation = {1, 1};
  if (auto dilOpt = cast<DenseI64ArrayAttr>(op->getAttr("dilation"))) {
    dilation[0] = (dilOpt)[0];
    dilation[1] = (dilOpt)[1];
  }

  // Fetch input pads (default zeros)
  SmallVector<int64_t, 4> inPad(4, 0);
  if (auto padOpt = cast<DenseI64ArrayAttr>(op->getAttr("pad"))) {
    inPad[0] = (padOpt)[0]; // top
    inPad[1] = (padOpt)[1]; // bottom
    inPad[2] = (padOpt)[2]; // left
    inPad[3] = (padOpt)[3]; // right
  }

  Value input = op->getOperand(0);
  const auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  if (inputType && weightType) {
    const int64_t inputHeight = inputType.getDimSize(1);
    const int64_t kernelHeight = weightType.getDimSize(1);
    const int64_t outputHeight = outputType.getDimSize(1);

    if (!ShapedType::isDynamic(inputHeight) &&
        !ShapedType::isDynamic(outputHeight)) {
      if (outputHeight !=
          (inputHeight - 1) * strideY + outPadTop + outPadBottom +
              ((kernelHeight - 1) * dilation[0]) + 1 - inPad[0] - inPad[1]) {
        return op.emitOpError(
                   "dimension mismatch: expected OH = (IH - 1) * "
                   "stride_y + out_pad_top + out_pad_bottom + ((KH - 1) * "
                   "dilation_y + 1) - pad_top - pad_bottom, but got: ")
               << outputHeight << " != (" << inputHeight << " - 1) * "
               << strideY << " + " << outPadTop << " + " << outPadBottom
               << " + ((" << kernelHeight << " - 1) * " << dilation[0]
               << " + 1) - " << inPad[0] << " - " << inPad[1];
      }
    }

    const int64_t inputWidth = inputType.getDimSize(2);
    const int64_t kernelWidth = weightType.getDimSize(2);
    const int64_t outputWidth = outputType.getDimSize(2);

    if (!ShapedType::isDynamic(inputWidth) &&
        !ShapedType::isDynamic(outputWidth)) {
      if (outputWidth != (inputWidth - 1) * strideX + outPadLeft + outPadRight +
                             ((kernelWidth - 1) * dilation[1] + 1) - inPad[2] -
                             inPad[3]) {
        return op.emitOpError(
                   "dimension mismatch: expected OW = (IW - 1) * "
                   "stride_x + out_pad_left + out_pad_right + (KW - 1) * "
                   "dilation_x + 1 - pad_left - pad_right, but got: ")
               << outputWidth << " != (" << inputWidth << " - 1) * " << strideX
               << " + " << outPadLeft << " + " << outPadRight << " + (("
               << kernelWidth << " - 1) * " << dilation[1] << " + 1) - "
               << inPad[2] - inPad[3];
      }
    }
  }

  Value bias = op->getOperand(2);
  const auto biasType = llvm::dyn_cast<RankedTensorType>(bias.getType());

  if (!biasType)
    return success();

  const int64_t biasChannels = biasType.getDimSize(0);

  // Skip further checks if bias is dynamic
  if (biasChannels == ShapedType::kDynamic)
    return success();

  const int64_t outputChannels = outputType.getDimSize(3);
  if (!ShapedType::isDynamic(outputChannels) &&
      biasChannels != outputChannels && biasChannels != 1)
    return op.emitOpError(
               "bias channels expected to be equal to output channels (")
           << outputChannels << ") or 1, got " << biasChannels;

  return success();
}

// If this is a backward-data (transpose) conv lowered from MIGraphX, its
// filter logical in/out channels are reversed relative to forward Conv2D.
FailureOr<std::tuple<Value, ShapedType>>
swapInputOutputDimensions(OpBuilder &rewriter, tosa::CustomOp op, Value weight,
                          ShapedType weightTy) {
  if (op.getOperatorName() == ROCK_CUSTOMOP_CONV_BWD_DATA) {
    // Expected current shape: [K, H, W, C] but Conv2D expects [O, H, W, I]
    // Swap K<->C => permutation {3,1,2,0}.
    auto wShape = weightTy.getShape();
    SmallVector<int64_t, 4> swappedShape{
        wShape[3], // C becomes O
        wShape[1], wShape[2],
        wShape[0] // K becomes I
    };
    auto swappedTy =
        RankedTensorType::get(swappedShape, weightTy.getElementType());
    weight =
        tosa::TransposeOp::create(rewriter, op.getLoc(), swappedTy, weight,
                                  rewriter.getDenseI32ArrayAttr({3, 1, 2, 0}));
    weightTy = cast<ShapedType>(weight.getType());

    return std::make_tuple(weight, weightTy);
  }

  return failure();
}

class TransposeConvNonStridedConverter
    : public OpRewritePattern<tosa::CustomOp> {
public:
  using OpRewritePattern<tosa::CustomOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::CustomOp op,
                                PatternRewriter &rewriter) const final {
    if (failed(verifyConvTranspose(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weight = op->getOperand(1);
    Value bias = op->getOperand(2);

    ShapedType inputTy = cast<ShapedType>(input.getType());
    ShapedType weightTy = cast<ShapedType>(weight.getType());
    ShapedType biasTy = cast<ShapedType>(bias.getType());
    ShapedType resultTy = cast<ShapedType>(op->getResult(0).getType());

    // Translate acc_type, padding and stride attributes.
    llvm::ArrayRef<int64_t> outPad =
        cast<DenseI64ArrayAttr>(op->getAttr("out_pad"));
    llvm::ArrayRef<int64_t> stride =
        cast<DenseI64ArrayAttr>(op->getAttr("stride"));
    auto accTypeAttr = cast<TypeAttr>(op->getAttr("acc_type"));
    Type accType = accTypeAttr.getValue();

    int convDims = stride.size();
    if (convDims != 2 && convDims != 3)
      return rewriter.notifyMatchFailure(op, "conv op must be 2D or 3D");

    // Fetch dilation (default all ones)
    SmallVector<int64_t> dilationVals(convDims, 1);
    if (auto dilOpt = cast<DenseI64ArrayAttr>(op->getAttr("dilation"))) {
      dilationVals[0] = (dilOpt)[0];
      dilationVals[1] = (dilOpt)[1];
      if (convDims == 3)
        dilationVals[2] = (dilOpt)[2];
    }

    // Fetch input pads (default zeros)
    SmallVector<int64_t, 4> inPadVals(convDims * 2, 0);
    if (auto padOpt = cast<DenseI64ArrayAttr>(op->getAttr("pad"))) {
      inPadVals[0] = (padOpt)[0];
      inPadVals[1] = (padOpt)[1];
      inPadVals[2] = (padOpt)[2];
      inPadVals[3] = (padOpt)[3];
      if (convDims == 3) {
        inPadVals[4] = (padOpt)[4];
        inPadVals[5] = (padOpt)[5];
      }
    }

    // Get inputZp and weightZp operands.
    Value inputZp = op.getOperands()[3];
    Value weightZp = op.getOperands()[4];

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
    int64_t effKDm1 = 0;
    if (convDims == 3) {
      int64_t kernelDepth = weightTy.getDimSize(3);
      effKDm1 = (kernelDepth - 1) * dilationVals[2];
    }

    // Conv2D/Conv3D padding derived from ConvTranspose (ONNX/PyTorch style)
    // convPad = effK - inPad + outPad
    SmallVector<int64_t, 4> convPad(convDims * 2, 0);
    convPad[0] = effKHm1 - inPadVals[0] + outPad[0];
    convPad[1] = effKHm1 - inPadVals[1] + outPad[1];
    convPad[2] = effKWm1 - inPadVals[2] + outPad[2];
    convPad[3] = effKWm1 - inPadVals[3] + outPad[3];
    if (convDims == 3) {
      convPad[4] = effKDm1 - inPadVals[4] + outPad[4];
      convPad[5] = effKDm1 - inPadVals[5] + outPad[5];
    }

    // A negative padding value would require cropping, "slicing", the result
    // to properly emulate negative padding.
    bool needSlice = false;
    SmallVector<int64_t, 4> negExcess(convDims * 2, 0);
    for (int i = 0; i < convDims * 2; ++i) {
      if (convPad[i] < 0) {
        negExcess[i] = -convPad[i];
        convPad[i] = 0;
        needSlice = true;
      }
    }

    if (needSlice)
      return rewriter.notifyMatchFailure(op, "Cannot currently handle negative "
                                             "padding values.");

    Value convOp;
    Value reverse1 =
        tosa::ReverseOp::create(rewriter, loc, weightTy, weight,
                                /* axis = */ rewriter.getI32IntegerAttr(1));
    Value reverse2 =
        tosa::ReverseOp::create(rewriter, loc, weightTy, reverse1,
                                /* axis = */ rewriter.getI32IntegerAttr(2));

    if (convDims == 2) {
      convOp = tosa::Conv2DOp::create(
          rewriter, loc, resultTy, input, reverse2, bias, inputZp, weightZp,
          rewriter.getDenseI64ArrayAttr(convPad),
          rewriter.getDenseI64ArrayAttr(stride),
          rewriter.getDenseI64ArrayAttr(dilationVals),
          /* acc_type = */ accType, op->getAttrOfType<IntegerAttr>("group"));
    } else {
      Value reverse3 =
          tosa::ReverseOp::create(rewriter, loc, weightTy, reverse2,
                                  /* axis = */ rewriter.getI32IntegerAttr(3));
      convOp = tosa::Conv3DOp::create(
          rewriter, loc, resultTy, input, reverse3, bias, inputZp, weightZp,
          rewriter.getDenseI64ArrayAttr(convPad),
          rewriter.getDenseI64ArrayAttr(stride),
          rewriter.getDenseI64ArrayAttr(dilationVals),
          /* acc_type = */ accType);
    }

    rewriter.replaceOp(op, convOp);
    return success();
  }
};

class TransposeConvStridedConverter : public OpRewritePattern<tosa::CustomOp> {
public:
  // Copy-pasted from mlir/lib/Dialect/Tosa/IR/TosaOps.cpp
  static FailureOr<int64_t> getZeroPoint(Value val, bool signExtend) {
    ElementsAttr zpAttr;
    if (!matchPattern(val, m_Constant(&zpAttr))) {
      return failure();
    }

    Type zpElemType = zpAttr.getElementType();

    if (llvm::isa<FloatType>(zpElemType)) {
      if (zpAttr.getValues<APFloat>()[0].isZero()) {
        return 0;
      }
      // return non-zero value to trigger error check
      return -1;
    }

    if (llvm::isa<IntegerType>(zpElemType)) {
      if (signExtend)
        return zpAttr.getValues<APInt>()[0].getSExtValue();
      else
        return zpAttr.getValues<APInt>()[0].getZExtValue();
    }

    // return non-zero value to trigger error check
    return -1;
  }

  using OpRewritePattern<tosa::CustomOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::CustomOp op,
                                PatternRewriter &rewriter) const final {
    if (failed(verifyConvTranspose(op, rewriter)))
      return failure();

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

    // Translate acc_type, padding and stride attributes.
    llvm::ArrayRef<int64_t> outPad =
        cast<DenseI64ArrayAttr>(op->getAttr("out_pad"));
    llvm::ArrayRef<int64_t> stride =
        cast<DenseI64ArrayAttr>(op->getAttr("stride"));
    auto accTypeAttr = cast<TypeAttr>(op->getAttr("acc_type"));
    Type accType = accTypeAttr.getValue();

    // Get inputZp and weightZp operands.
    Value inputZpOperand = op.getOperands()[3];
    Value weightZpOperand = op.getOperands()[4];

    // Swap dimensions if needed
    auto swapOr = swapInputOutputDimensions(rewriter, op, weight, weightTy);
    if (succeeded(swapOr)) {
      weight = std::get<0>(swapOr.value());
      weightTy = std::get<1>(swapOr.value());
    }

    // Fetch dilation (default {1,1})
    SmallVector<int64_t, 2> dilationVals = {1, 1};
    if (auto dilOpt = cast<DenseI64ArrayAttr>(op->getAttr("dilation"))) {
      dilationVals[0] = (dilOpt)[0];
      dilationVals[1] = (dilOpt)[1];
    }

    // Fetch input padding (default {0, 0, 0, 0})
    SmallVector<int64_t, 4> inPadVals = {0, 0, 0, 0};
    if (auto inPadOpt = cast<DenseI64ArrayAttr>(op->getAttr("pad"))) {
      inPadVals[0] = (inPadOpt)[0];
      inPadVals[1] = (inPadOpt)[1];
      inPadVals[2] = (inPadOpt)[2];
      inPadVals[3] = (inPadOpt)[3];
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
    FailureOr<int64_t> maybeIZp = getZeroPoint(inputZpOperand, false);
    if (failed(maybeIZp))
      return rewriter.notifyMatchFailure(
          op, "input zero point cannot be statically determined");

    FailureOr<int64_t> maybeWZp = getZeroPoint(weightZpOperand, false);
    if (failed(maybeWZp))
      return rewriter.notifyMatchFailure(
          op, "weight zero point cannot be statically determined");

    int64_t inputZpVal = *maybeIZp;
    int64_t weightZpVal = *maybeWZp;

    // Here TOSA would call verifyInputZeroPoint and verifyWeightZeroPoint,
    // skip this for simplicity.

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
          llvm::SmallVector<int64_t, 8> padSpec = {0, 0, 0, padRows,
                                                   0, 0, 0, 0};
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
          llvm::SmallVector<int64_t, 4> size = {outputChannels, weightHeight, 1,
                                                inputChannels};
          Value beginVal = getTosaConstShape(rewriter, loc, begin);
          Value sizeVal = getTosaConstShape(rewriter, loc, size);
          Value slice = CreateOpAndInferShape<tosa::SliceOp>(
                            rewriter, loc, UnrankedTensorType::get(weightETy),
                            weight, beginVal, sizeVal)
                            .getResult();
          int64_t padCols = (w == weightWidth - 1) ? 0 : (dW - 1);
          if (padCols > 0) {
            llvm::SmallVector<int64_t, 8> padSpec = {0, 0,       0, 0,
                                                     0, padCols, 0, 0};
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
    int64_t origWeightWidth = weightWidth;

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
    if (op->hasAttr("pad")) {
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
    Value conv2d =
        CreateOpAndInferShape<tosa::Conv2DOp>(
            rewriter, loc, UnrankedTensorType::get(resultETy), input, weight,
            zeroBias, inputZp.value(), weightZp.value(),
            /*pad=*/rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
            /*stride=*/rewriter.getDenseI64ArrayAttr({1, 1}),
            /*dilation=*/rewriter.getDenseI64ArrayAttr({1, 1}),
            /* acc_type = */ accType, op->getAttrOfType<IntegerAttr>("group"))
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

    // Effective pad = outPad + (paddedK - stride - 1) - (inPad * stride)
    int64_t effPadTop = outPad[0] + (origWeightHeight - stride[0] - 1) -
                        inPadVals[0] * stride[0];
    int64_t effPadLeft = outPad[2] + (origWeightWidth - stride[1] - 1) -
                         inPadVals[2] * stride[1];

    // When we shrink from the orignal size to kPrime by grouping stride phases,
    // we discard some positions that existed in the conceptual upsampled view.
    // The total span of the original field is kOrig -1, and the span
    // represented after factoring is kPrime - 1. The difference is the values
    // that have been lost
    int64_t kHPrime = restridedWeightTy.getDimSize(1);
    int64_t kWPrime = restridedWeightTy.getDimSize(2);
    auto lost = [](int64_t Korig, int64_t kPrime, int64_t S) {
      return (Korig - 1) - (kPrime - 1) * S;
    };
    int64_t lostH = lost(origWeightHeight, kHPrime, stride[0]);
    int64_t lostW = lost(origWeightWidth, kWPrime, stride[1]);

    // If stride factoring compresses a dimension to a single spatial position,
    // i.e., kPrime == 1, then we dropped a ring of values around that position.
    // The adjustment pattern depends on which dimension has asymmetric padding.

    // Height dimension compressed (kHPrime==1)
    if (kHPrime == 1 && lostH > 0) {
      int64_t adjustment = lostH / 2;
      bool hasAsymmetricWidth = (weightPadding[4] != weightPadding[5]);
      if (hasAsymmetricWidth) {
        effPadTop -= adjustment;
        effPadLeft += adjustment;
      } else {
        effPadLeft += adjustment;
      }
    }

    // Width dimension compressed (kWPrime==1)
    if (kWPrime == 1 && lostW > 0) {
      effPadTop += lostW / 2;
    }

    int64_t resultSliceTop;
    int64_t resultSliceLeft;
    int64_t resultPadTop;
    int64_t resultPadLeft;
    // Convert effective padding into slice (crop) and post-pad just like the
    // prior logic but now using effPad*.
    if (op->hasAttr("pad")) {
      resultSliceTop = std::max<int64_t>(0, -effPadTop);
      resultSliceLeft = std::max<int64_t>(0, -effPadLeft);
      resultPadTop = std::max<int64_t>(0, effPadTop);
      resultPadLeft = std::max<int64_t>(0, effPadLeft);
    } else {
      // Default to using legacy logic if input padding is not present
      resultSliceTop = std::max<int64_t>(0, -outPad[0]);
      resultSliceLeft = std::max<int64_t>(0, -outPad[2]);
      resultPadTop = std::max<int64_t>(0, outPad[0]);
      resultPadLeft = std::max<int64_t>(0, outPad[2]);
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

    // We verified early that op has exactly one result, so getType(0) is safe.
    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, op.getType(0), resultPad,
                                             bias);
    return success();
  }
};

// Convert expand_strides custom op to tensor.empty + tensor.insert_slice
class ExpandStridesDecomposeConverter final
    : public OpRewritePattern<tosa::CustomOp> {
public:
  using OpRewritePattern<tosa::CustomOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::CustomOp op,
                                PatternRewriter &rewriter) const final {
    // Match only expand_strides custom ops
    if (op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME)
      return rewriter.notifyMatchFailure(op, "domain isn't rocmlir");
    if (op.getOperatorName() != ROCK_CUSTOMOP_EXPAND_STRIDES)
      return rewriter.notifyMatchFailure(op, "isn't an expand_strides op");
    if (op.getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "should have 1 operand");
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "should have 1 result");

    Location loc = op.getLoc();
    Value input = op->getOperand(0);
    auto inputType = cast<RankedTensorType>(input.getType());
    auto outputType = cast<RankedTensorType>(op.getResult(0).getType());
    int64_t rank = inputType.getRank();

    // Create an empty tensor with the padded output shape
    Value emptyDest = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), outputType.getElementType());

    // Build the offsets, sizes, and strides for insert_slice
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(rank);
    for (int64_t dim : inputType.getShape())
      sizes.push_back(rewriter.getIndexAttr(dim));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    // Insert the input into the beginning of the padded buffer
    Value result = rewriter.create<tensor::InsertSliceOp>(
        loc, input, emptyDest, offsets, sizes, strides);

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::rock::populateRocmlirCustomTosaDecomposeTarget(
    ConversionTarget &target) {
  target.addLegalDialect<tosa::TosaDialect>();
  target.addLegalOp<tensor::EmptyOp, tensor::InsertSliceOp>();
  target.addDynamicallyLegalOp<tosa::CustomOp>([](tosa::CustomOp op) {
    return op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME ||
           (op.getOperatorName() != ROCK_CUSTOMOP_CONV_BWD_DATA &&
            op.getOperatorName() != ROCK_CUSTOMOP_CONV_BWD_WEIGHT &&
            op.getOperatorName() != ROCK_CUSTOMOP_EXPAND_STRIDES);
  });
}

void mlir::rock::populateRocmlirCustomTosaDecomposeConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TransposeConvNonStridedConverter, TransposeConvStridedConverter,
               ExpandStridesDecomposeConverter>(patterns.getContext());
}

void RocmlirCustomTosaDecomposePass::runOnOperation() {
  Operation *op = getOperation();

  ConversionTarget target(getContext());
  rock::populateRocmlirCustomTosaDecomposeTarget(target);

  RewritePatternSet patterns(&getContext());
  rock::populateRocmlirCustomTosaDecomposeConversionPatterns(patterns);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();
}
