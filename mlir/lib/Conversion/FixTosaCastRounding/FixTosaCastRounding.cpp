//===- FixTosaCastRounding.cpp - Fix tosa.cast rounding -------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// The upstream tosa-to-linalg pass inserts math.roundeven before arith.fptosi
// when lowering tosa.cast from float to integer. This implements TOSA's
// "round to nearest, ties to even" semantics.
//
// However, ONNX and PyTorch define float-to-int cast as truncation (round
// towards zero), which is what arith.fptosi already does natively. Since
// rocMLIR primarily serves ONNX/MIGraphX workloads, this pass removes the
// math.roundeven ops to restore RTZ semantics without modifying the upstream
// LLVM code.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FixTosaCastRounding/FixTosaCastRounding.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_FIXTOSACASTROUNDINGPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Returns true when `roundeven`'s result participates *exclusively* in the
/// upstream tosa-to-linalg float-to-int cast chain. The chain has two parts:
///   1. Float clamp: optional `arith.minimumf`/`maximumf` on the rounded
///      value, ending at `arith.fptosi`.
///   2. Integer saturation merge (i32 case): the rounded value also feeds
///      `arith.cmpf` to produce an i1 mask, which feeds an `arith.select`
///      that picks between an integer saturation constant and the
///      `arith.fptosi` result. The merged i32 then flows to `linalg.yield`.
///
/// We follow both branches and accept `linalg.yield` as a terminal. Removing
/// the `math.roundeven` is safe even for the saturation comparison: at the
/// i32 saturation boundary (|2^31|) every f32 value is already an integer
/// (f32 ULP is >= 1 above 2^23), so `roundeven` is a no-op there and the
/// `arith.cmpf` result is unchanged.
///
/// This is intentionally strict: if any user of a value in the chain is not
/// recognized, we return false (do not strip the `roundeven`) even when a
/// sibling user does reach an `arith.fptosi`. This prevents miscompiles
/// where the rounded value is also consumed by an unrelated op that depends
/// on RNE semantics.
static bool reachesFPToSI(math::RoundEvenOp op) {
  SmallVector<Value, 8> worklist(op->getResults());
  bool foundFPToSI = false;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (isa<arith::FPToSIOp>(user)) {
        foundFPToSI = true;
        continue;
      }
      if (isa<linalg::YieldOp>(user))
        continue;
      if (isa<arith::MinimumFOp, arith::MaximumFOp, arith::CmpFOp,
              arith::SelectOp>(user)) {
        for (Value r : user->getResults())
          worklist.push_back(r);
        continue;
      }
      return false;
    }
  }
  return foundFPToSI;
}

static bool hasRtzCastLocTag(Location loc) {
  if (auto fused = dyn_cast<FusedLoc>(loc))
    if (auto meta = dyn_cast_or_null<StringAttr>(fused.getMetadata()))
      return meta.getValue() == rock::kRtzCastLocTag;
  return false;
}

/// True when this `math.roundeven` is part of an RTZ-tagged
/// `migraphx.convert` lowering. The tag is set on the `tosa.cast` and ends
/// up on the parent `linalg.generic`'s loc and on its output region block
/// argument (the one carved out from the `tensor.empty()` that this cast
/// writes into). Upstream `tosa-to-linalg` may assign the inner
/// `math.roundeven` a different `Location`, so we don't rely on the op's
/// own loc alone.
///
/// We deliberately do NOT scan input block arguments: those inherit the
/// loc of their incoming SSA value, and if that value comes from a
/// previously-tagged cast the tag would propagate forward, causing this
/// pass to wrongly strip an unrelated `math.roundeven` in a downstream
/// generic.
static bool isRtzTaggedCastLowering(math::RoundEvenOp op,
                                    linalg::GenericOp generic) {
  if (hasRtzCastLocTag(op->getLoc()) || hasRtzCastLocTag(generic.getLoc()))
    return true;
  return llvm::any_of(generic.getRegionOutputArgs(), [](BlockArgument arg) {
    return hasRtzCastLocTag(arg.getLoc());
  });
}

struct RemoveRoundEvenBeforeFPToSI
    : public OpRewritePattern<math::RoundEvenOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::RoundEvenOp op,
                                PatternRewriter &rewriter) const override {
    auto generic = op->getParentOfType<linalg::GenericOp>();
    if (!generic)
      return failure();

    // The RTZ-tagged cast lowering corresponds to a single tosa.cast and
    // therefore produces exactly one integer output from the generic. Use
    // getOutputs() rather than getResultTypes() so this still works in
    // buffer semantics (where there are no SSA results).
    //
    // Bail on multi-output generics (e.g. produced by linalg fusion) to
    // avoid stripping a math.roundeven that also feeds a sibling result --
    // for instance an i1 yielded directly from arith.cmpf, which would
    // silently flip if we removed the rounding.
    //
    // Bail on i1 outputs as well: ONNX/PyTorch float-to-bool semantics is
    // "non-zero" rather than truncation, so removing roundeven would be
    // unsafe even if upstream tosa-to-linalg ever emitted it for an i1
    // cast. Today MIGraphXToTosa does not tag float-to-i1 casts, but this
    // guard is defense-in-depth.
    ValueRange outs = generic.getOutputs();
    if (outs.size() != 1)
      return failure();
    Type outElemTy = getElementTypeOrSelf(outs[0].getType());
    if (!isa<IntegerType>(outElemTy) || outElemTy.isInteger(1))
      return failure();

    if (!isRtzTaggedCastLowering(op, generic))
      return failure();

    if (!reachesFPToSI(op))
      return failure();

    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

struct FixTosaCastRoundingPass
    : public impl::FixTosaCastRoundingPassBase<FixTosaCastRoundingPass> {
  using FixTosaCastRoundingPassBase::FixTosaCastRoundingPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveRoundEvenBeforeFPToSI>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
