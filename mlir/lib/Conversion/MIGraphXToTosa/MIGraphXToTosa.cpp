//===- MIGraphXToTosa.cpp - Lowering MIGraphX to Tosa Dialect
//-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the MIGraphX to the Tos dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MIGraphXToTosa/MIGraphXToTosa.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

static bool isBroadcastable(Operation *op, Operation *operand) {
  // tosa only broadcast implicitly on the second input of the binary operators.
  if (op->getNumOperands() != 2)
    return false;
  if (op->getOperand(1) != operand->getResult(0)) {
    // swap, if possible
    op->setOperand(0, op->getOperand(1));
    op->setOperand(1, operand->getResult(0));
  }
  return true;
}

class ConvConverter final
    : public OpConversionPattern<migraphx::ConvolutionOp> {
public:
  using OpConversionPattern<migraphx::ConvolutionOp>::OpConversionPattern;

  Value getZero(Location loc, Type elemType,
                ConversionPatternRewriter &rewriter) const {
    auto biasTy = RankedTensorType::get({1}, elemType);
    auto zeroAttr =
        DenseElementsAttr::get(biasTy, rewriter.getZeroAttr(elemType));
    return rewriter.create<arith::ConstantOp>(loc, zeroAttr);
  }

  tosa::TransposeOp getRank4TransposeOp(Location loc, Value input,
                                        ConversionPatternRewriter &rewriter,
                                        SmallVector<int64_t> &permutation,
                                        bool bRoot) const {
    auto permutationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({4}, rewriter.getI64Type()), permutation);
    Value permutationValue =
        rewriter.create<arith::ConstantOp>(loc, permutationAttr);
    ShapedType inputTy = input.getType().cast<ShapedType>();
    auto inputShape = inputTy.getShape();
    SmallVector<int64_t> newShape{
        inputShape[permutation[0]], inputShape[permutation[1]],
        inputShape[permutation[2]], inputShape[permutation[3]]};
    Type newTy = RankedTensorType::get(newShape, inputTy.getElementType());

    auto newOp =
        rewriter.create<tosa::TransposeOp>(loc, newTy, input, permutationValue);
    newOp->setAttr("changing_layout_root", rewriter.getBoolAttr(bRoot));
    return newOp;
  }

  LogicalResult
  matchAndRewrite(migraphx::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    Location loc = op->getLoc();
    auto input_t = operands[0];
    auto filter_t = operands[1];
    auto results = op->getResults();
    auto elementTy =
        op->getOperand(0).getType().cast<ShapedType>().getElementType();
    auto outputTy = results[0].getType().cast<ShapedType>();
    SmallVector<int64_t> NCHW2NHWC{0, 2, 3, 1};
    SmallVector<int64_t> NHWC2NCHW{0, 3, 1, 2};

    // insert transpose to input and filter tensors
    input_t = getRank4TransposeOp(loc, input_t, rewriter, NCHW2NHWC, false);
    filter_t = getRank4TransposeOp(loc, filter_t, rewriter, NCHW2NHWC, false);
    auto outShape = outputTy.getShape();

    // original output shape was NCHW, change it into NHWC
    SmallVector<int64_t> newShape{outShape[0], outShape[2], outShape[3],
                                  outShape[1]};
    Type newOutTy = RankedTensorType::get(newShape, outputTy.getElementType());

    // Construct a new Conv2DOp.
    auto cop = rewriter.create<tosa::Conv2DOp>(
        loc, newOutTy,
        ValueRange{input_t, filter_t, getZero(loc, elementTy, rewriter)});

    // translate attributes
    auto padAttr = op->getAttr("padding").cast<ArrayAttr>();
    auto strideAttr = op->getAttr("stride").cast<ArrayAttr>();
    auto dilationAttr = op->getAttr("dilation").cast<ArrayAttr>();
    int64_t padTop = padAttr[0].dyn_cast<IntegerAttr>().getInt();
    int64_t padBottom = padAttr[1].dyn_cast<IntegerAttr>().getInt();
    int64_t padLeft = padAttr[2].dyn_cast<IntegerAttr>().getInt();
    int64_t padRight = padAttr[3].dyn_cast<IntegerAttr>().getInt();
    int64_t strideHeight = strideAttr[0].dyn_cast<IntegerAttr>().getInt();
    int64_t strideWidth = strideAttr[1].dyn_cast<IntegerAttr>().getInt();
    int64_t dilationHeight = dilationAttr[0].dyn_cast<IntegerAttr>().getInt();
    int64_t dilationWidth = dilationAttr[1].dyn_cast<IntegerAttr>().getInt();

    // Record desired layout and set
    // convolution config attributes
    cop->setAttr("expected_filter_layout", rewriter.getStringAttr("kcyx"));
    cop->setAttr("expected_input_layout", rewriter.getStringAttr("nchw"));
    cop->setAttr("expected_output_layout", rewriter.getStringAttr("nkhw"));

    cop->setAttr("dilation", rewriter.getArrayAttr({
                                 rewriter.getI64IntegerAttr(dilationHeight),
                                 rewriter.getI64IntegerAttr(dilationWidth),
                             }));
    cop->setAttr("stride", rewriter.getArrayAttr({
                               rewriter.getI64IntegerAttr(strideHeight),
                               rewriter.getI64IntegerAttr(strideWidth),
                           }));
    cop->setAttr("pad", rewriter.getArrayAttr({
                            rewriter.getI64IntegerAttr(padTop),
                            rewriter.getI64IntegerAttr(padBottom),
                            rewriter.getI64IntegerAttr(padLeft),
                            rewriter.getI64IntegerAttr(padRight),
                        }));

    // Convert optional attributes
    if (auto attr = op->getAttrOfType<BoolAttr>("xdlopsV2"))
      cop->setAttr("xdlopsV2", attr);
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      cop->setAttr("perf_config", attr);

    // transpose the output back to NCHW so that it can match following
    // operators.
    auto top = getRank4TransposeOp(loc, cop, rewriter, NHWC2NCHW, true);
    rewriter.replaceOp(op, {top});
    return success();
  }
};

class BroadcastConverter final
    : public OpConversionPattern<migraphx::BroadcastOp> {
public:
  using OpConversionPattern<migraphx::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    Location loc = op->getLoc();
    auto input_t = operands[0];
    auto axis = op->getAttr("axis").cast<IntegerAttr>().getInt();

    // get shape of the use
    auto outShape = op->getResultTypes()[0].cast<ShapedType>().getShape();
    auto outElemType =
        op->getResultTypes()[0].cast<ShapedType>().getElementType();
    uint32_t outRank = outShape.size();
    auto inShape = input_t.getType().cast<ShapedType>().getShape();

    Value newOperand = input_t;
    if (outRank != inShape.size()) {
      SmallVector<int64_t, 5> newShape;
      SmallVector<Attribute, 5> newShapeAttr;

      // align the dimensions - by the given axis
      for (uint32_t i = 0; i < outRank; i++) {
        newShapeAttr.push_back(rewriter.getI64IntegerAttr(1));
        newShape.push_back(1);
      }
      newShapeAttr[axis] = rewriter.getI64IntegerAttr(inShape[0]);
      newShape[axis] = inShape[0];

      // reshape
      auto outType = RankedTensorType::get(newShape, outElemType);
      newOperand = rewriter.create<tosa::ReshapeOp>(
          loc, outType, input_t, rewriter.getArrayAttr(newShapeAttr));
    }

    for (auto &use : op->getResult(0).getUses()) {
      auto expandedOp = use.getOwner();
      // isa binary operation,
      if (isBroadcastable(expandedOp, op)) {
        // replace the uses
        for (auto &operand : expandedOp->getOpOperands()) {
          if (operand.get() == op) {
            operand.set(newOperand);
            break;
          }
        }
      } else {
        return failure();
      }
    }
    // erase broadcast
    rewriter.eraseOp(op);
    return success();
  }
};

class MultiBroadcastConverter final
    : public OpConversionPattern<migraphx::MultiBroadcastOp> {
public:
  using OpConversionPattern<migraphx::MultiBroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::MultiBroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    Location loc = op->getLoc();
    auto input_t = operands[0];
    auto inShape = input_t.getType().cast<ShapedType>().getShape();
    uint32_t inRank = inShape.size();

    for (auto &use : op->getResult(0).getUses()) {
      auto expandedOp = use.getOwner();
      if (expandedOp == op)
        continue;
      // isa binary operation,
      if (isBroadcastable(expandedOp, op)) {
        // get shape of the use
        auto outShape =
            expandedOp->getResultTypes()[0].cast<ShapedType>().getShape();
        auto outElemType =
            expandedOp->getResultTypes()[0].cast<ShapedType>().getElementType();
        uint32_t outRank = outShape.size();

        Value newOperand = input_t;
        if (outRank != inShape.size()) {
          SmallVector<int64_t, 5> newShape;
          SmallVector<Attribute, 5> newShapeAttr;

          // align the dimensions - by the given in/out shape
          uint32_t i = 0;
          for (; i < outRank - inRank; i++) {
            newShapeAttr.push_back(rewriter.getI64IntegerAttr(inShape[i]));
            newShape.push_back(inShape[i]);
          }
          for (; i < outRank; i++) {
            newShapeAttr.push_back(rewriter.getI64IntegerAttr(1));
            newShape.push_back(1);
          }

          // reshape
          auto outType = RankedTensorType::get(newShape, outElemType);
          newOperand = rewriter.create<tosa::ReshapeOp>(
              loc, outType, input_t, rewriter.getArrayAttr(newShapeAttr));
        }

        // replace the uses
        for (auto &operand : expandedOp->getOpOperands()) {
          if (operand.get() == op) {
            operand.set(newOperand);
            break;
          }
        }
      } else {
        return failure();
      }
    }
    // erase broadcast
    rewriter.eraseOp(op);
    return success();
  }
};

class DotConverter final : public OpConversionPattern<migraphx::DotOp> {
public:
  using OpConversionPattern<migraphx::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(migraphx::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    Location loc = op->getLoc();
    auto in_A = operands[0];
    auto in_B = operands[1];
    auto results = op->getResults();
    auto elementTy =
        op->getOperand(0).getType().cast<ShapedType>().getElementType();
    ShapedType outputTy = results[0].getType().cast<ShapedType>();

    // check batch dimension. Tosa matmul only allow a single dimension for it,
    // add reshape ops to flatten and restore the original dimension.
    ArrayRef<int64_t> orgOutDims = outputTy.getShape();
    RankedTensorType newOutType = RankedTensorType::get(orgOutDims, elementTy);
    size_t outRank = orgOutDims.size();
    if (outRank > 3) { // A, B, Out have the same rank
      ArrayRef<int64_t> orgDimsA = in_A.getType().cast<ShapedType>().getShape();
      ArrayRef<int64_t> orgDimsB = in_B.getType().cast<ShapedType>().getShape();
      int64_t batchSize = 1;
      for (int i = 0; i < outRank - 2; i++) {
        batchSize *= orgOutDims[i];
      }
      int64_t newDimsA[3] = {batchSize, orgDimsA[outRank - 2],
                             orgDimsA[outRank - 1]};
      int64_t newDimsB[3] = {batchSize, orgDimsB[outRank - 2],
                             orgDimsB[outRank - 1]};
      int64_t newDimsOut[3] = {batchSize, orgOutDims[outRank - 2],
                               orgOutDims[outRank - 1]};
      RankedTensorType newAType = RankedTensorType::get(newDimsA, elementTy);
      RankedTensorType newBType = RankedTensorType::get(newDimsB, elementTy);
      newOutType = RankedTensorType::get(newDimsOut, elementTy);
      auto reshapeAOp = rewriter.create<tosa::ReshapeOp>(
          loc, newAType, in_A, rewriter.getI64ArrayAttr(newDimsA));
      auto reshapeBOp = rewriter.create<tosa::ReshapeOp>(
          loc, newBType, in_B, rewriter.getI64ArrayAttr(newDimsB));

      // reassign inputs.
      in_A = reshapeAOp;
      in_B = reshapeBOp;
    }
    // Construct tosa.matmul.
    auto mop = rewriter.create<tosa::MatMulOp>(loc, newOutType, in_A, in_B);

    if (outRank > 3) {
      auto rop = rewriter.create<tosa::ReshapeOp>(
          loc, outputTy, mop, rewriter.getI64ArrayAttr(orgOutDims));
      rewriter.replaceOp(op, {rop});
      return success();
    }
    rewriter.replaceOp(op, {mop});
    return success();
  }
};
} // namespace

void migraphx::populateMIGraphXToTosaConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<ConvConverter, BroadcastConverter, MultiBroadcastConverter,
               DotConverter>(context);
}
