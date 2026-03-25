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
#include "mlir/IR/IRMapping.h"
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
  // then multiplies them.
  // When the scale is not f8E8M0FNU, it is first roundtripped through
  // f8E8M0FNU (e.g. f32 -> f8E8M0FNU -> f32) so that the host decomposition
  // matches the kernel path, which casts scales to f8E8M0FNU block exponents
  // for tosa.matmul_t_block_scaled / rock.gemm.
  Value applyScale(PatternRewriter &rewriter, Location loc, Value input,
                   Value scale, Type targetElemType) const {
    Type scaleElemType = cast<MIXRShapedType>(scale.getType()).getElementType();
    Type f8E8M0Type = Float8E8M0FNUType::get(rewriter.getContext());
    if (scaleElemType != f8E8M0Type)
      scale = convertToType(rewriter, loc, scale, f8E8M0Type);
    Value convertedScale = convertToType(rewriter, loc, scale, targetElemType);
    Value convertedInput = convertToType(rewriter, loc, input, targetElemType);

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

/// Broadcast a 4D K or V tensor from [batch, numHeadsKV, D1, D2] to
/// [batch, numHeadsQ, D1, D2] for GQA by inserting a broadcast dimension
/// and reshaping: broadcast to [batch, numHeadsKV, repeat, D1, D2] then
/// reshape to [batch, numHeadsQ, D1, D2].
static Value broadcastForGQA(PatternRewriter &rewriter, Location loc, Value val,
                             int64_t numHeadsQ) {
  auto valType = cast<MIXRShapedType>(val.getType());
  ArrayRef<int64_t> shape = valType.getShape();
  int64_t numHeadsKV = shape[1];
  int64_t repeat = numHeadsQ / numHeadsKV;

  SmallVector<int64_t> bcShape = {shape[0], numHeadsKV, repeat, shape[2],
                                  shape[3]};
  SmallVector<int64_t> bcStrides = {valType.getStrides()[0],
                                    valType.getStrides()[1], 0,
                                    valType.getStrides()[2],
                                    valType.getStrides()[3]};
  auto bcType =
      MIXRShapedType::get(bcShape, bcStrides, valType.getElementType());
  Value bc = migraphx::MultiBroadcastOp::create(rewriter, loc, bcType, val,
                                                rewriter.getI64ArrayAttr(bcShape));

  SmallVector<int64_t> newShape = {shape[0], numHeadsQ, shape[2], shape[3]};
  SmallVector<int64_t> newStrides(newShape.size());
  int64_t s = 1;
  for (int64_t i = newShape.size() - 1; i >= 0; --i) {
    newStrides[i] = s;
    s *= newShape[i];
  }
  auto newType =
      MIXRShapedType::get(newShape, newStrides, valType.getElementType());
  return migraphx::ReshapeOp::create(rewriter, loc, newType, bc,
                                     rewriter.getI64ArrayAttr(newShape));
}

class AttentionDecompose final
    : public OpRewritePattern<migraphx::AttentionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(migraphx::AttentionOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    auto qType = cast<MIXRShapedType>(op.getQueries().getType());
    auto kType = cast<MIXRShapedType>(op.getKeys().getType());
    auto vType = cast<MIXRShapedType>(op.getValues().getType());

    int64_t qRank = qType.getRank();
    int64_t kRank = kType.getRank();

    // Handle GQA: if Q has more heads than K/V, broadcast K/V to match.
    Value queries = op.getQueries();
    Value keys = op.getKeys();
    Value values = op.getValues();

    if (qRank == 4 && kRank == 4 &&
        qType.getDimSize(1) != kType.getDimSize(1)) {
      int64_t numHeadsQ = qType.getDimSize(1);
      keys = broadcastForGQA(rewriter, loc, keys, numHeadsQ);
      values = broadcastForGQA(rewriter, loc, values, numHeadsQ);
      kType = cast<MIXRShapedType>(keys.getType());
      vType = cast<MIXRShapedType>(values.getType());
      kRank = kType.getRank();
    }

    // Compute Q*K shape: batch dims from Q, seq_q from Q, seq_k from K
    SmallVector<int64_t> qkShape(qType.getShape().begin(),
                                 qType.getShape().end());
    qkShape[qRank - 1] = kType.getShape()[kRank - 1];
    SmallVector<int64_t> qkStrides(qkShape.size());
    int64_t stride = 1;
    for (int64_t i = qkShape.size() - 1; i >= 0; --i) {
      qkStrides[i] = stride;
      stride *= qkShape[i];
    }
    Type elemType = qType.getElementType();
    auto qkType = MIXRShapedType::get(qkShape, qkStrides, elemType);

    // 1. First GEMM: Q * K
    Value qk = migraphx::DotOp::create(rewriter, loc, qkType, queries, keys);

    // 2. Inline preSoftmaxBody elementwise ops.
    // The verifier guarantees that if preSoftmaxElemWiseInputs are present,
    // the body contains non-terminator ops, and vice versa.
    if (!op.getPreSoftmaxElemWiseInputs().empty()) {
      Block &block = op.getPreSoftmaxBody().front();
      IRMapping mapping;
      mapping.map(block.getArgument(0), qk);
      auto preSoftmaxInputs = op.getPreSoftmaxElemWiseInputs();
      for (unsigned i = 0; i < preSoftmaxInputs.size(); ++i)
        mapping.map(block.getArgument(i + 1), preSoftmaxInputs[i]);

      Value lastResult = qk;
      for (Operation &bodyOp : block) {
        if (bodyOp.hasTrait<OpTrait::IsTerminator>())
          continue;
        Operation *cloned = rewriter.clone(bodyOp, mapping);
        if (cloned->getNumResults() > 0) {
          lastResult = cloned->getResult(0);
          for (auto [oldRes, newRes] :
               llvm::zip(bodyOp.getResults(), cloned->getResults()))
            mapping.map(oldRes, newRes);
        }
      }
      qk = lastResult;
    }

    // 3. Handle softmaxType: convert before softmax if needed
    Type softmaxElemType = elemType;
    if (op.getSoftmaxType())
      softmaxElemType = *op.getSoftmaxType();

    if (softmaxElemType != elemType) {
      auto qkShaped = cast<MIXRShapedType>(qk.getType());
      auto convertedType = MIXRShapedType::get(
          qkShaped.getShape(), qkShaped.getStrides(), softmaxElemType);
      qk = migraphx::ConvertOp::create(rewriter, loc, convertedType, qk);
    }

    auto qkSoftmaxType = cast<MIXRShapedType>(qk.getType());
    int64_t softmaxAxis = qkSoftmaxType.getRank() - 1;

    // Compute reduced shape (last dim becomes 1) for reduce ops
    auto computeReducedType = [&](MIXRShapedType fullType) {
      SmallVector<int64_t> rShape(fullType.getShape());
      rShape[softmaxAxis] = 1;
      SmallVector<int64_t> rStrides(rShape.size());
      int64_t s = 1;
      for (int64_t i = rShape.size() - 1; i >= 0; --i) {
        rStrides[i] = s;
        s *= rShape[i];
      }
      return MIXRShapedType::get(rShape, rStrides,
                                 fullType.getElementType());
    };

    Value softmaxResult;
    Value lseValue;

    bool needLse = op.getLse() != nullptr;

    if (needLse) {
      // Decompose softmax manually to extract LSE intermediates:
      //   max = reduce_max(qk, axis=-1)
      //   norm = qk - max
      //   exp_val = exp(norm)
      //   sum_exp = reduce_sum(exp_val, axis=-1)
      //   recip = recip(sum_exp)
      //   softmax_result = exp_val * recip
      //   lse = log(sum_exp) + max
      auto axisAttr = rewriter.getI64ArrayAttr({softmaxAxis});
      auto reducedType = computeReducedType(qkSoftmaxType);

      Value maxVal = migraphx::ReduceMaxOp::create(
          rewriter, loc, reducedType, qk, axisAttr);
      Value norm = migraphx::SubOp::create(
          rewriter, loc, qkSoftmaxType, qk, maxVal);
      Value expVal = migraphx::ExpOp::create(
          rewriter, loc, qkSoftmaxType, norm);
      Value sumExp = migraphx::ReduceSumOp::create(
          rewriter, loc, reducedType, expVal, axisAttr);
      Value recip = migraphx::RecipOp::create(
          rewriter, loc, reducedType, sumExp);
      softmaxResult = migraphx::MulOp::create(
          rewriter, loc, qkSoftmaxType, expVal, recip);

      // LSE = log(sum_exp) + max
      Value logSumExp = migraphx::LogOp::create(
          rewriter, loc, reducedType, sumExp);
      lseValue = migraphx::AddOp::create(
          rewriter, loc, reducedType, logSumExp, maxVal);
    } else {
      // 4. Use migraphx.softmax when LSE is not needed
      softmaxResult = migraphx::SoftmaxOp::create(
          rewriter, loc, qkSoftmaxType, qk,
          rewriter.getI64IntegerAttr(softmaxAxis));
    }

    // 5. Convert back if softmaxType differs from values element type
    if (softmaxElemType != vType.getElementType()) {
      auto smShaped = cast<MIXRShapedType>(softmaxResult.getType());
      auto convertedBack = MIXRShapedType::get(
          smShaped.getShape(), smShaped.getStrides(), vType.getElementType());
      softmaxResult = migraphx::ConvertOp::create(rewriter, loc, convertedBack,
                                                   softmaxResult);
    }

    // 6. Second GEMM: softmax(QK) * V
    auto resultType = cast<MIXRShapedType>(op.getResult().getType());

    Value result = migraphx::DotOp::create(rewriter, loc, resultType,
                                            softmaxResult, values);

    SmallVector<Value> results;
    results.push_back(result);

    // 7. Add LSE output if requested
    if (needLse) {
      auto lseOutputType = cast<MIXRShapedType>(op.getLse().getType());

      // The reduce ops keep the reduced axis as size 1 (e.g., [2, 64, 1])
      // but the verifier enforces LSE shape without the trailing 1
      // (e.g., [2, 64]). Reshape to drop it.
      SmallVector<int64_t> lseShape(lseOutputType.getShape());
      SmallVector<int64_t> lseStrides(lseShape.size());
      int64_t ls = 1;
      for (int64_t i = lseShape.size() - 1; i >= 0; --i) {
        lseStrides[i] = ls;
        ls *= lseShape[i];
      }
      auto reshapedLseType = MIXRShapedType::get(
          lseShape, lseStrides,
          cast<MIXRShapedType>(lseValue.getType()).getElementType());
      lseValue = migraphx::ReshapeOp::create(
          rewriter, loc, reshapedLseType, lseValue,
          rewriter.getI64ArrayAttr(lseShape));

      // Convert LSE element type if needed (e.g., f16 -> f32)
      if (reshapedLseType.getElementType() != lseOutputType.getElementType()) {
        lseValue = migraphx::ConvertOp::create(
            rewriter, loc, lseOutputType, lseValue);
      }

      results.push_back(lseValue);
    }

    rewriter.replaceOp(op, results);
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
    // Only run with non-kernel functions:
    // - QuantDotDecompose: tosa.matmul_t_block_scaled doesn't have conversion
    //   to linalg in the upstream passes. For the kernel side, it is converted
    //   to rock.gemm with scales in TosaToRock.cpp.
    //   TODO: Remove once upstream TOSA adds this conversion.
    // - AttentionDecompose: migraphx.attention is decomposed into primitive
    //   migraphx ops (dot, softmax, etc.) for the host/CPU path. For the
    //   kernel side, migraphx.attention is preserved and lowered directly
    //   to rock.attention via the MIGraphXAttentionToRock pass.
    if (!func->hasAttr("rock.kernel")) {
      RewritePatternSet patterns(&ctx);
      patterns.add<QuantDotDecompose>(&ctx);
      patterns.add<AttentionDecompose>(&ctx);
      if (failed(applyPatternsGreedily(func, std::move(patterns))))
        signalPassFailure();
    }
  }
};

} // namespace
