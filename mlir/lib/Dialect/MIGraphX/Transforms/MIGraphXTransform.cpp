//===- TosaOptionalDecompositions.cpp
//------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to apply the Tosa operations decompositions
// exposed as populate functions in
// include/mlir/Dialect/Tosa/Transforms/Passes.h
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXTRANSFORMPASS
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {

/// Decompose quant_dot when one or both inputs are int4 tensors.
/// Pattern:  quant_dot(act_fp16, weights_int4) → dot(act_fp16, convert(weights_int4 → fp16))
/// This implements AWQ-style int4 weight dequantization fused into the GEMM.
/// The actual scale/zero-point dequantization is expected to have been fused
/// into the int4 weights upstream (by a dequantize_linear-folding pass); here
/// we simply promote the int4 to fp16 via a cast and compute a standard dot.
class QuantDotInt4Decompose final
    : public OpRewritePattern<migraphx::QuantDotOp> {
public:
  using OpRewritePattern<migraphx::QuantDotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(migraphx::QuantDotOp op,
                                PatternRewriter &rewriter) const final {
    // Only handle the non-scaled case (no explicit scaleA/scaleB).
    if (op.getScaleA() || op.getScaleB())
      return failure();

    auto isInt4Elem = [](Type t) -> bool {
      if (auto iType = dyn_cast<IntegerType>(t))
        return iType.getWidth() == 4;
      return false;
    };

    Value inA = op.getInA();
    Value inB = op.getInB();
    auto inAElem = cast<MIXRShapedType>(inA.getType()).getElementType();
    auto inBElem = cast<MIXRShapedType>(inB.getType()).getElementType();

    bool aIsInt4 = isInt4Elem(inAElem);
    bool bIsInt4 = isInt4Elem(inBElem);

    if (!aIsInt4 && !bIsInt4)
      return failure(); // No int4 inputs — not our pattern.

    Location loc = op->getLoc();
    auto resultType = op.getResult().getType();
    Type outElem = cast<MIXRShapedType>(resultType).getElementType();

    // Promote int4 inputs to the output element type (fp16 or fp32).
    auto promoteToOutputType = [&](Value v) -> Value {
      auto shaped = cast<MIXRShapedType>(v.getType());
      if (!isInt4Elem(shaped.getElementType()))
        return v;
      auto newType = MIXRShapedType::get(shaped.getShape(), shaped.getStrides(),
                                         outElem);
      return migraphx::ConvertOp::create(rewriter, loc, newType, v).getOutput();
    };

    Value promotedA = promoteToOutputType(inA);
    Value promotedB = promoteToOutputType(inB);

    auto dotType = MIXRShapedType::get(
        cast<MIXRShapedType>(resultType).getShape(),
        cast<MIXRShapedType>(resultType).getStrides(), outElem);
    auto dotOp =
        migraphx::DotOp::create(rewriter, loc, dotType, promotedA, promotedB);
    rewriter.replaceOp(op, dotOp.getOutput());
    return success();
  }
};

class QuantDotDecompose final : public OpRewritePattern<migraphx::QuantDotOp> {
public:
  using OpRewritePattern<migraphx::QuantDotOp>::OpRewritePattern;

private:
  // Helper function to convert a value to the target element type if needed
  Value convertToType(PatternRewriter &rewriter, Location loc, Value val,
                      Type targetElemType) const {
    auto shapedType = cast<MIXRShapedType>(val.getType());
    if (shapedType.getElementType() == targetElemType) {
      return val;
    }
    return migraphx::ConvertOp::create(
        rewriter, loc,
        MIXRShapedType::get(shapedType.getShape(), shapedType.getStrides(),
                            targetElemType),
        val);
  }

  // Helper function to apply scaling: converts input and scale to target type,
  // then multiplies them
  Value applyScale(PatternRewriter &rewriter, Location loc, Value input,
                   Value scale, Type targetElemType) const {
    Value convertedInput = convertToType(rewriter, loc, input, targetElemType);
    Value convertedScale = convertToType(rewriter, loc, scale, targetElemType);

    auto shapedType = cast<MIXRShapedType>(convertedInput.getType());
    return migraphx::MulOp::create(rewriter, loc,
                                   MIXRShapedType::get(shapedType.getShape(),
                                                       shapedType.getStrides(),
                                                       targetElemType),
                                   convertedInput, convertedScale);
  }

public:
  LogicalResult matchAndRewrite(migraphx::QuantDotOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    auto inA = op.getInA();
    auto inB = op.getInB();
    auto scaleA = op.getScaleA();
    auto scaleB = op.getScaleB();

    // Only decompose scaled GEMM operations (both scales must be present).
    // The verifier ensures both scales are provided together or neither.
    if (!scaleA && !scaleB) {
      return failure();
    }

    // Determine target output type
    auto resultType = op.getResult().getType();

    // For scaled operations, we always convert to F32 for proper computation
    Type computeElemType = rewriter.getF32Type();

    // Convert and scale inputs
    Value processedA = applyScale(rewriter, loc, inA, scaleA, computeElemType);
    Value processedB = applyScale(rewriter, loc, inB, scaleB, computeElemType);

    // Create the dot operation with processed inputs
    auto dotOp = migraphx::DotOp::create(rewriter, loc, resultType, processedA,
                                         processedB);
    if (auto attr = (*op).template getAttrOfType<StringAttr>("perf_config"))
      dotOp->setAttr("perf_config", attr);
    rewriter.replaceOp(op, dotOp->getResults()[0]);
    return success();
  }
};

class SqrtDecompose final : public OpConversionPattern<migraphx::SqrtOp> {
public:
  using OpConversionPattern<migraphx::SqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto inA = op->getOperand(0);
    auto outputTy = cast<ShapedType>(op->getResults()[0].getType());
    auto rSop = migraphx::RsqrtOp::create(rewriter, loc, outputTy, inA);
    auto rCop = migraphx::RecipOp::create(rewriter, loc, outputTy, rSop);

    rewriter.replaceOp(op, rCop->getResults()[0]);
    return success();
  }
};

void populateMIGraphXSqrt(MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<SqrtDecompose>(context);
}

struct MIGraphXTransforms
    : public migraphx::impl::MIGraphXTransformPassBase<MIGraphXTransforms> {
  void runOnOperation() override {
    auto &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<migraphx::MIGraphXDialect, func::FuncDialect,
                           tosa::TosaDialect, mhal::MHALDialect>();
    target.addIllegalOp<migraphx::SqrtOp>();
    auto func = getOperation();

    populateMIGraphXSqrt(&ctx, patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
    {
      RewritePatternSet patterns(&ctx);
      patterns.add<QuantDotDecompose, QuantDotInt4Decompose>(&ctx);
      if (failed(applyPatternsGreedily(func, std::move(patterns))))
        signalPassFailure();
    }
  }
};

} // namespace
