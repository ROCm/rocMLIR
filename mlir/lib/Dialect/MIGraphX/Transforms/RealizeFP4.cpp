//===- MIGraphXRealizeFP4.cpp - Convert MIGraphX FP8 Unpacks to FP4 -------===//
//
// This file implements a pass to convert migraphx.unpack operations with FP8
// input and output element types to instead produce FP4 results, by adjusting
// the output shaped type (doubling the axis dimension & strides) and
// replacing the operation with an equivalent one producing FP4.
//
// After this pass, all remaining unpacks should have FP4 results, and the
// function signature types will be updated to match.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXREALIZEFP4PASS
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir
using namespace mlir;
using namespace mlir::migraphx;

namespace {

/// Records block arguments that originally had >1 uses (before any rewrite).
/// We avoid (a) folding their associated unpack into the signature and
/// (b) moving an unpack ahead of a transpose/reshape/broadcast if doing so
/// would make that block argument single-use (thereby enabling an unwanted
/// fold).
struct Fp4RealizeAnalysis {
  DenseSet<BlockArgument> multiUseArgs;
  bool isMultiUseArg(Value v) const {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return multiUseArgs.contains(ba);
    return false;
  }
};

// Helper retained from earlier version.
static bool isFp8(Type t) {
  return isa<Float8E4M3FNType, Float8E5M2FNUZType>(t);
}

static MIXRShapedType buildUnpackedFp4(MIXRShapedType inType, int64_t axis) {
  if (!isFp8(inType.getElementType()))
    return {};
  auto rank = (int64_t)inType.getShape().size();
  if (axis < 0 || axis >= rank)
    return {};

  SmallVector<int64_t> lens(inType.getShape().begin(), inType.getShape().end());
  SmallVector<int64_t> strides(inType.getStrides().begin(),
                               inType.getStrides().end());
  lens[axis] *= 2;
  for (int64_t i = 0; i < rank; ++i)
    if (i != axis)
      strides[i] *= 2;

  auto fp4 = Float4E2M1FNType::get(inType.getContext());
  return MIXRShapedType::get(lens, strides, fp4);
}

//===----------------------------------------------------------------------===//
// Pattern: Convert raw fp8->fp8 unpack result type to fp4 (does NOT fold).
//===----------------------------------------------------------------------===//
struct ConvertFp8UnpackToFp4Pattern : OpRewritePattern<UnpackOp> {
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto inTy = dyn_cast<MIXRShapedType>(op.getIn().getType());
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    if (!inTy || !outTy)
      return failure();
    // We only care about incorrect fp8->fp8 unpacks.
    if (!isFp8(inTy.getElementType()) || !isFp8(outTy.getElementType()))
      return failure();

    int64_t axis = op.getAxis();
    auto fp4Ty = buildUnpackedFp4(inTy, axis);
    if (!fp4Ty || fp4Ty == outTy)
      return failure();

    auto newOp = rewriter.create<UnpackOp>(op.getLoc(), fp4Ty, op.getIn(),
                                           rewriter.getI64IntegerAttr(axis));
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Base class to access analysis inside patterns.
struct Fp4PatternBase {
  const Fp4RealizeAnalysis &analysis;
  Fp4PatternBase(const Fp4RealizeAnalysis &a) : analysis(a) {}
};

//===----------------------------------------------------------------------===//
// Fold argument unpack into function signature (only if argument was
// *originally* single-use). We rely on analysis.multiUseArgs to avoid folding
// test case 2.
//===----------------------------------------------------------------------===//
struct FoldFp4ArgUnpackPattern : OpRewritePattern<UnpackOp>, Fp4PatternBase {
  FoldFp4ArgUnpackPattern(MLIRContext *ctx, const Fp4RealizeAnalysis &a)
      : OpRewritePattern<UnpackOp>(ctx), Fp4PatternBase(a) {}
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto blockArg = dyn_cast<BlockArgument>(op.getIn());
    if (!blockArg)
      return failure();
    auto func = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!func)
      return failure();

    auto inTy = dyn_cast<MIXRShapedType>(blockArg.getType());
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    if (!inTy || !outTy)
      return failure();

    // Must already be an fp4 unpack result (Convert pattern should have run).
    if (!isa<Float4E2M1FNType>(outTy.getElementType()))
      return failure();

    // If argument already fp4 with the same shape, nothing to fold.
    if (isa<Float4E2M1FNType>(inTy.getElementType()) &&
        inTy.getShape() == outTy.getShape() &&
        inTy.getStrides() == outTy.getStrides())
      return failure();

    // Only fold if this argument was NOT originally multi-use.
    if (analysis.isMultiUseArg(blockArg))
      return failure();

    // Ensure (current) block argument has only this unpack as user to be safe.
    if (!blockArg.hasOneUse())
      return failure();

    // Form the desired fp4 argument type from original pre-unpacked fp8 type.
    // (If inTy is still fp8: we derive; if a prior move changed it, abort.)
    if (!isFp8(inTy.getElementType()))
      return failure();
    int64_t axis = op.getAxis();
    auto newArgType = buildUnpackedFp4(inTy, axis);
    if (!newArgType)
      return failure();

    FunctionType fType = func.getFunctionType();
    SmallVector<Type> inputs(fType.getInputs().begin(),
                             fType.getInputs().end());
    inputs[blockArg.getArgNumber()] = newArgType;

    rewriter.modifyOpInPlace(func, [&] {
      func.setFunctionType(
          FunctionType::get(func.getContext(), inputs, fType.getResults()));
      blockArg.setType(newArgType);
    });

    rewriter.replaceOp(op, blockArg);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Transpose + Unpack interchange (guarded so we don't create foldable args).
// Skip if transpose input is a multi-use argument (test2 scenario).
//===----------------------------------------------------------------------===//
struct TransposeFp4UnpackInterchange : OpRewritePattern<UnpackOp>,
                                       Fp4PatternBase {
  TransposeFp4UnpackInterchange(MLIRContext *ctx, const Fp4RealizeAnalysis &a)
      : OpRewritePattern<UnpackOp>(ctx), Fp4PatternBase(a) {}
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    auto inTy = dyn_cast<MIXRShapedType>(op.getIn().getType());
    if (!outTy || !inTy)
      return failure();
    if (!isa<Float4E2M1FNType>(outTy.getElementType()))
      return failure();

    auto trOp = op.getIn().getDefiningOp<TransposeOp>();
    if (!trOp)
      return failure();

    auto trInTy = dyn_cast<MIXRShapedType>(trOp.getInput().getType());
    if (!trInTy || !isFp8(trInTy.getElementType()))
      return failure();

    // Prevent creating a single-use argument enabling fold later.
    if (analysis.isMultiUseArg(trOp.getInput()))
      return failure();

    int64_t postAxis = op.getAxis();
    auto permAttr = trOp.getPermutation();
    if (postAxis < 0 || postAxis >= (int64_t)permAttr.size())
      return failure();

    int64_t preAxis = cast<IntegerAttr>(permAttr[postAxis]).getInt();
    auto preUnpackTy = buildUnpackedFp4(trInTy, preAxis);
    if (!preUnpackTy)
      return failure();

    // Compute expected final transposed type.
    SmallVector<int64_t> outShape(preUnpackTy.getRank());
    SmallVector<int64_t> outStrides(preUnpackTy.getRank());
    for (auto en : llvm::enumerate(permAttr.getAsRange<IntegerAttr>())) {
      int64_t src = en.value().getInt();
      outShape[en.index()] = preUnpackTy.getDimSize(src);
      outStrides[en.index()] = preUnpackTy.getStrides()[src];
    }
    auto expectedTy =
        MIXRShapedType::get(outShape, outStrides, preUnpackTy.getElementType());
    if (expectedTy != outTy)
      return failure();

    rewriter.setInsertionPoint(trOp);
    auto u =
        rewriter.create<UnpackOp>(op.getLoc(), preUnpackTy, trOp.getInput(),
                                  rewriter.getI64IntegerAttr(preAxis));
    auto newT = rewriter.create<TransposeOp>(trOp.getLoc(), expectedTy,
                                             u.getResult(), permAttr);

    rewriter.replaceOp(op, newT.getResult());
    rewriter.eraseOp(trOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reshape + Unpack interchange (guarded).
//===----------------------------------------------------------------------===//
struct ReshapeFp4UnpackInterchange : OpRewritePattern<UnpackOp>,
                                     Fp4PatternBase {
  ReshapeFp4UnpackInterchange(MLIRContext *ctx, const Fp4RealizeAnalysis &a)
      : OpRewritePattern<UnpackOp>(ctx), Fp4PatternBase(a) {}
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    auto reshapedTy = dyn_cast<MIXRShapedType>(op.getIn().getType());
    if (!outTy || !reshapedTy)
      return failure();
    if (!isa<Float4E2M1FNType>(outTy.getElementType()))
      return failure();

    auto reshapeOp = op.getIn().getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();

    auto preTy = dyn_cast<MIXRShapedType>(reshapeOp.getInput().getType());
    if (!preTy || !isFp8(preTy.getElementType()))
      return failure();

    // Guard: if reshape input is multi-use argument skip.
    if (analysis.isMultiUseArg(reshapeOp.getInput()))
      return failure();

    int64_t postAxis = op.getAxis();
    if (postAxis < 0 || postAxis >= (int64_t)reshapedTy.getRank())
      return failure();
    if (reshapedTy.getStrides()[postAxis] != 1)
      return failure();

    // Pick last stride==1 axis pre-reshape.
    int64_t preAxis = -1;
    for (auto en : llvm::enumerate(preTy.getStrides()))
      if (en.value() == 1)
        preAxis = en.index();
    if (preAxis < 0)
      return failure();

    auto preUnpackTy = buildUnpackedFp4(preTy, preAxis);
    if (!preUnpackTy)
      return failure();

    rewriter.setInsertionPoint(reshapeOp);
    auto u = rewriter.create<UnpackOp>(op.getLoc(), preUnpackTy,
                                       reshapeOp.getInput(),
                                       rewriter.getI64IntegerAttr(preAxis));
    SmallVector<int64_t> finalShape(outTy.getShape().begin(),
                                    outTy.getShape().end());
    auto newReshape =
        rewriter.create<ReshapeOp>(reshapeOp.getLoc(), outTy, u.getResult(),
                                   rewriter.getI64ArrayAttr(finalShape));

    rewriter.replaceOp(op, newReshape.getResult());
    rewriter.eraseOp(reshapeOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MultiBroadcast + Unpack interchange (guarded).
//===----------------------------------------------------------------------===//
struct MultiBroadcastFp4UnpackInterchange : OpRewritePattern<UnpackOp>,
                                            Fp4PatternBase {
  MultiBroadcastFp4UnpackInterchange(MLIRContext *ctx,
                                     const Fp4RealizeAnalysis &a)
      : OpRewritePattern<UnpackOp>(ctx), Fp4PatternBase(a) {}
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    auto mbOutTy = dyn_cast<MIXRShapedType>(op.getIn().getType());
    if (!outTy || !mbOutTy)
      return failure();
    if (!isa<Float4E2M1FNType>(outTy.getElementType()))
      return failure();

    auto mb = op.getIn().getDefiningOp<MultiBroadcastOp>();
    if (!mb)
      return failure();

    auto preTy = dyn_cast<MIXRShapedType>(mb.getInput().getType());
    if (!preTy || !isFp8(preTy.getElementType()))
      return failure();

    if (analysis.isMultiUseArg(mb.getInput()))
      return failure();

    int64_t axis = op.getAxis();
    if (axis < 0 || axis >= (int64_t)mbOutTy.getRank())
      return failure();

    auto preUnpackTy = buildUnpackedFp4(preTy, axis);
    if (!preUnpackTy)
      return failure();

    // Adjust out_lens (double axis lens).
    SmallVector<Attribute> newOutLens;
    newOutLens.reserve(mb.getOutLens().size());
    for (auto en : llvm::enumerate(mb.getOutLens())) {
      auto ia = cast<IntegerAttr>(en.value());
      int64_t v = ia.getInt();
      if ((int64_t)en.index() == axis)
        v *= 2;
      newOutLens.push_back(IntegerAttr::get(ia.getType(), v));
    }

    rewriter.setInsertionPoint(mb);
    auto u = rewriter.create<UnpackOp>(op.getLoc(), preUnpackTy, mb.getInput(),
                                       rewriter.getI64IntegerAttr(axis));
    auto newMB = rewriter.create<MultiBroadcastOp>(
        mb.getLoc(), outTy, u.getResult(), rewriter.getArrayAttr(newOutLens));

    rewriter.replaceOp(op, newMB.getResult());
    rewriter.eraseOp(mb);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The Pass
//===----------------------------------------------------------------------===//
struct MIGraphXRealizeFP4Pass
    : public migraphx::impl::MIGraphXRealizeFP4PassBase<MIGraphXRealizeFP4Pass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Build analysis BEFORE any rewrites.
    Fp4RealizeAnalysis analysis;
    for (BlockArgument arg : func.getArguments())
      if (!arg.use_empty() && !arg.hasOneUse())
        analysis.multiUseArgs.insert(arg);

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertFp8UnpackToFp4Pattern>(&getContext());
    patterns.add<TransposeFp4UnpackInterchange, ReshapeFp4UnpackInterchange,
                 MultiBroadcastFp4UnpackInterchange>(&getContext(), analysis);
    patterns.add<FoldFp4ArgUnpackPattern>(&getContext(), analysis);

    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return signalPassFailure();

    // Harmonize function result types with actual return operand types.
    auto fType = func.getFunctionType();
    SmallVector<Type> newResults(fType.getResults().begin(),
                                 fType.getResults().end());
    bool changed = false;
    for (auto ret : func.getOps<func::ReturnOp>()) {
      for (auto en : llvm::enumerate(ret.getOperands())) {
        Type actual = en.value().getType();
        if (en.index() < newResults.size() &&
            newResults[en.index()] != actual) {
          newResults[en.index()] = actual;
          changed = true;
        }
      }
    }
    if (changed)
      func.setFunctionType(
          FunctionType::get(func.getContext(), fType.getInputs(), newResults));
  }
};

} // end anonymous namespace
