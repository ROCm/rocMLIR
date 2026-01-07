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

// A generic pattern that detects when any operation produces output with
// larger strides than its input (stride expansion) and makes this explicit
// by inserting an expand_strides op.
//
// For example, transforms:
//   %1 = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16,
//   1152x24x1>
// Into:
//   %1 = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16,
//   576x24x1> %2 = migraphx.expand_strides %1 : <4x24x24xf16, 576x24x1> ->
//   <4x24x24xf16, 1152x24x1>
class InsertExpandStrides final : public RewritePattern {
public:
  InsertExpandStrides(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  // Per-result expansion information.
  struct ResultExpansionInfo {
    MIXRShapedType originalType;
    MIXRShapedType compactType;
    bool needsExpansion() const { return compactType != nullptr; }
  };

  // Compute standard row-major strides for a given shape.
  static SmallVector<int64_t> computeStandardStrides(ArrayRef<int64_t> shape) {
    SmallVector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  // Check if strides indicate actual memory expansion. Transposed layouts have
  // non-standard strides but same memory footprint.
  static bool hasExpandedMemoryFootprint(ArrayRef<int64_t> shape,
                                         ArrayRef<int64_t> strides) {
    if (shape.empty())
      return false;

    // Compute max memory offset: sum((shape[i] - 1) * stride[i])
    // For standard/transposed layouts, this equals numElements - 1
    // For expanded layouts, this is larger
    int64_t maxOffset = 0;
    int64_t numElements = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      // Skip broadcast dimensions (stride 0)
      if (strides[i] != 0)
        maxOffset += (shape[i] - 1) * strides[i];
      numElements *= shape[i];
    }

    return maxOffset > (numElements - 1);
  }

  // Check if any operand already has the same expanded strides as the result.
  // If so, the operation is just preserving the stride layout from its input,
  // not creating new expansion.
  static bool inputAlreadyHasExpandedStrides(Operation *op,
                                             MIXRShapedType resultType) {
    ArrayRef<int64_t> resultShape = resultType.getShape();
    ArrayRef<int64_t> resultStrides = resultType.getStrides();

    for (Value operand : op->getOperands()) {
      auto inputType = dyn_cast<MIXRShapedType>(operand.getType());
      if (!inputType)
        continue;

      // Check if input has same shape and same (or larger) strides
      if (inputType.getShape() == resultShape &&
          inputType.getStrides() == resultStrides) {
        return true;
      }
    }
    return false;
  }
};

LogicalResult
InsertExpandStrides::matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {
  // Skip expand_strides ops themselves to avoid infinite loops
  if (isa<migraphx::ExpandStridesOp>(op))
    return failure();

  // Only handle ops in the MIGraphX dialect
  if (!isa<MIGraphXDialect>(op->getDialect()))
    return failure();

  // Skip operations that legitimately produce non-standard strides as part of
  // their semantics. These are view-creating or stride-manipulating operations
  // where the output strides are determined by the operation, not by memory
  // expansion.
  if (isa<migraphx::ReshapeOp, migraphx::TransposeOp, migraphx::BroadcastOp,
      migraphx::MultiBroadcastOp, migraphx::SliceOp, migraphx::FlattenOp,
      migraphx::AsLogicalShapeOp, migraphx::AsUnderlyingShapeOp,
      migraphx::LiteralOp, migraphx::ConvertOp>(op))
  return failure();

  // Must have at least one result
  if (op->getNumResults() == 0)
    return failure();

  // Analyze each result to see if it needs stride expansion
  SmallVector<ResultExpansionInfo> resultInfos;
  resultInfos.reserve(op->getNumResults());
  bool anyNeedsExpansion = false;

  for (OpResult result : op->getResults()) {
    ResultExpansionInfo info;
    info.originalType = dyn_cast<MIXRShapedType>(result.getType());

    // Non-MIXRShapedType results or broadcasts don't need expansion
    if (!info.originalType || info.originalType.hasBroadcast()) {
      info.compactType = nullptr;
    } else {
      ArrayRef<int64_t> shape = info.originalType.getShape();
      ArrayRef<int64_t> strides = info.originalType.getStrides();

      // Only insert expand_strides when there's actual memory expansion,
      // not just non-standard layouts like transposition. Transposed layouts
      // have the same memory footprint as standard layouts. Also skip if the
      // expanded strides are inherited from an input.
      if (!hasExpandedMemoryFootprint(shape, strides) ||
          inputAlreadyHasExpandedStrides(op, info.originalType)) {
        info.compactType = nullptr;
      } else {
        // Compute what the compact (standard row-major) strides should be
        SmallVector<int64_t> standardStrides = computeStandardStrides(shape);
        info.compactType = MIXRShapedType::get(
            shape, standardStrides, info.originalType.getElementType());
        anyNeedsExpansion = true;
      }
    }

    resultInfos.push_back(info);
  }

  // Nothing to do if no results need expansion
  if (!anyNeedsExpansion)
    return failure();

  // Clone the operation and update result types for those needing expansion
  rewriter.setInsertionPoint(op);
  Operation *newOp = rewriter.clone(*op);

  for (auto [idx, info] : llvm::enumerate(resultInfos)) {
    if (info.needsExpansion())
      newOp->getResult(idx).setType(info.compactType);
  }

  // Build replacement values, inserting expand_strides where needed
  Location loc = op->getLoc();
  SmallVector<Value> replacements;
  replacements.reserve(op->getNumResults());

  for (auto [idx, info] : llvm::enumerate(resultInfos)) {
    if (info.needsExpansion()) {
      auto expandOp = migraphx::ExpandStridesOp::create(
          rewriter, loc, info.originalType, newOp->getResult(idx));
      replacements.push_back(expandOp.getResult());
    } else {
      replacements.push_back(newOp->getResult(idx));
    }
  }

  rewriter.replaceOp(op, replacements);
  return success();
}

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
      patterns.add<QuantDotDecompose>(&ctx);
      patterns.add<InsertExpandStrides>(&ctx);
      if (failed(applyPatternsGreedily(func, std::move(patterns))))
        signalPassFailure();
    }
  }
};

} // namespace
