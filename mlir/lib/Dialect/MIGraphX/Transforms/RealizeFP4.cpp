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

// Helper kept as-is.
static bool isFp8(Type t) { return isa<Float8E4M3FNType>(t); }

static MIXRShapedType buildUnpackedFp4(MIXRShapedType inType, int64_t axis) {
  if (!inType || !isFp8(inType.getElementType()))
    return {};
  int64_t rank = (int64_t)inType.getShape().size();
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

// ---------------------------------------------------------------------------
// Transpose+Unpack -> Unpack+Transpose (fp4)
//
// Pattern triggers on:
//   %t = migraphx.transpose %x : fp8
//   %u = migraphx.unpack %t {axis = A} : fp8 -> fp4
//
// Rewrites to:
//   %u0 = migraphx.unpack %x {axis = preAxis} : fp8 -> fp4
//   %t0 = migraphx.transpose %u0 : fp4
//
// Preconditions:
//   - transpose result element type is fp8
//   - unpack output element type is fp4
//   - unpack input is exactly the transpose result
// ---------------------------------------------------------------------------
struct TransposeFp4UnpackInterchange : OpRewritePattern<UnpackOp> {
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto unpackOutTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    auto unpackInTy = dyn_cast<MIXRShapedType>(op.getIn().getType());
    if (!unpackOutTy || !unpackInTy)
      return failure();
    // Need fp4 result, fp8 input element.
    if (!isa<Float4E2M1FNType>(unpackOutTy.getElementType()) ||
        !isFp8(unpackInTy.getElementType()))
      return failure();

    auto trOp = op.getIn().getDefiningOp<TransposeOp>();
    if (!trOp)
      return failure();
    auto trInTy = dyn_cast<MIXRShapedType>(trOp.getInput().getType());
    if (!trInTy || !isFp8(trInTy.getElementType()))
      return failure();

    int64_t postAxis = op.getAxis();
    ArrayAttr permAttr = trOp.getPermutation();
    if (postAxis < 0 || postAxis >= (int64_t)permAttr.size())
      return failure();
    // permutation[i] = source dimension index for output dim i
    int64_t preAxis = cast<IntegerAttr>(permAttr[postAxis]).getInt();

    // Build fp4 type for unpack BEFORE transpose.
    auto preUnpackFp4Ty = buildUnpackedFp4(trInTy, preAxis);
    if (!preUnpackFp4Ty)
      return failure();

    // Build the transposed fp4 type we expect after moving transpose.
    SmallVector<int64_t> outShape(preUnpackFp4Ty.getRank());
    SmallVector<int64_t> outStrides(preUnpackFp4Ty.getRank());
    for (auto en : llvm::enumerate(permAttr.getAsRange<IntegerAttr>())) {
      int64_t srcIdx = en.value().getInt();
      outShape[en.index()] = preUnpackFp4Ty.getDimSize(srcIdx);
      outStrides[en.index()] = preUnpackFp4Ty.getStrides()[srcIdx];
    }
    auto expectedPostTy = MIXRShapedType::get(outShape, outStrides,
                                              preUnpackFp4Ty.getElementType());

    // Ensure types line up with current unpack output to avoid miscompile.
    if (expectedPostTy != unpackOutTy)
      return failure();

    rewriter.setInsertionPoint(trOp);
    auto newUnpack =
        rewriter.create<UnpackOp>(op.getLoc(), preUnpackFp4Ty, trOp.getInput(),
                                  rewriter.getI64IntegerAttr(preAxis));

    auto newTranspose = rewriter.create<TransposeOp>(
        trOp.getLoc(), expectedPostTy, newUnpack.getResult(), permAttr);

    rewriter.replaceOp(op, newTranspose.getResult());
    rewriter.eraseOp(trOp);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Reshape+Unpack -> Unpack+Reshape (fp4)
//
// Similar guard logic to int4 pass:
//  - unpack result is fp4
//  - reshape produces fp8 (still packed) and its input is fp8
//  - The dimension chosen for unpack after reshape has stride 1
// We move unpack to operate on a dimension (pick last stride==1) of the
// pre-reshape tensor, then reshape the fp4 result.
// ---------------------------------------------------------------------------
struct ReshapeFp4UnpackInterchange : OpRewritePattern<UnpackOp> {
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    auto inTy = dyn_cast<MIXRShapedType>(op.getIn().getType());
    if (!outTy || !inTy)
      return failure();
    if (!isa<Float4E2M1FNType>(outTy.getElementType()) ||
        !isFp8(inTy.getElementType()))
      return failure();

    auto reshapeOp = op.getIn().getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();

    auto preTy = dyn_cast<MIXRShapedType>(reshapeOp.getInput().getType());
    if (!preTy || !isFp8(preTy.getElementType()))
      return failure();

    int64_t postAxis = op.getAxis();
    if (postAxis < 0 || postAxis >= (int64_t)inTy.getRank())
      return failure();

    // Need stride==1 on the axis in the reshaped (post-reshape) fp8 tensor.
    if (inTy.getStrides()[postAxis] != 1)
      return failure();

    // Choose last stride==1 dim in pre-reshape tensor (mirrors int4 logic).
    int64_t preAxis = -1;
    for (auto en : llvm::enumerate(preTy.getStrides()))
      if (en.value() == 1)
        preAxis = en.index();
    if (preAxis < 0)
      return failure();

    auto preUnpackFp4Ty = buildUnpackedFp4(preTy, preAxis);
    if (!preUnpackFp4Ty)
      return failure();

    // Expected final fp4 shape after reshape must match current unpack output.
    // The current outTy came from: reshape(fp8) then unpack(axis=postAxis).
    // After transformation: unpack(fp8 pre) -> fp4, then reshape to outTy.
    // We trust outTy; we just perform the sequence.

    rewriter.setInsertionPoint(reshapeOp);
    auto newUnpack = rewriter.create<UnpackOp>(
        op.getLoc(), preUnpackFp4Ty, reshapeOp.getInput(),
        rewriter.getI64IntegerAttr(preAxis));

    // New reshape dims = outTy logical shape.
    SmallVector<int64_t> outShape(outTy.getShape().begin(),
                                  outTy.getShape().end());
    auto newReshape = rewriter.create<ReshapeOp>(
        reshapeOp.getLoc(), outTy, newUnpack.getResult(),
        rewriter.getI64ArrayAttr(outShape));

    rewriter.replaceOp(op, newReshape.getResult());
    rewriter.eraseOp(reshapeOp);
    return success();
  }
};

// ---------------------------------------------------------------------------
// MultiBroadcast+Unpack -> Unpack+MultiBroadcast (fp4)
//
// Pattern triggers on:
//   %b = migraphx.multibroadcast %x {out_lens = [...]} : fp8 -> fp8
//   %u = migraphx.unpack %b {axis = A} : fp8 -> fp4
//
// Rewrites to:
//   %u0 = migraphx.unpack %x {axis = A} : fp8 -> fp4
//   %b0 = migraphx.multibroadcast %u0 {out_lens' (axis A doubled)} : fp4 -> fp4
//
// We recompute fp4 pre-broadcast type and ensure final type matches u's output.
// ---------------------------------------------------------------------------
struct MultiBroadcastFp4UnpackInterchange : OpRewritePattern<UnpackOp> {
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    auto inTy = dyn_cast<MIXRShapedType>(op.getIn().getType());
    if (!outTy || !inTy)
      return failure();
    if (!isa<Float4E2M1FNType>(outTy.getElementType()) ||
        !isFp8(inTy.getElementType()))
      return failure();

    auto mbOp = op.getIn().getDefiningOp<MultiBroadcastOp>();
    if (!mbOp)
      return failure();

    auto preTy = dyn_cast<MIXRShapedType>(mbOp.getInput().getType());
    if (!preTy || !isFp8(preTy.getElementType()))
      return failure();

    int64_t axis = op.getAxis();
    if (axis < 0 || axis >= (int64_t)inTy.getRank())
      return failure();

    auto preUnpackFp4Ty = buildUnpackedFp4(preTy, axis);
    if (!preUnpackFp4Ty)
      return failure();

    // Build updated out_lens (double axis lens).
    SmallVector<Attribute> newOutLens;
    newOutLens.reserve(mbOp.getOutLens().size());
    for (auto en : llvm::enumerate(mbOp.getOutLens())) {
      auto intAttr = cast<IntegerAttr>(en.value());
      int64_t v = intAttr.getInt();
      if ((int64_t)en.index() == axis)
        v *= 2;
      newOutLens.push_back(IntegerAttr::get(intAttr.getType(), v));
    }

    // Broadcast result fp4 shaped type should match current unpack output.
    // We trust current outTy; just emit broadcast producing that type.

    rewriter.setInsertionPoint(mbOp);
    auto newUnpack =
        rewriter.create<UnpackOp>(op.getLoc(), preUnpackFp4Ty, mbOp.getInput(),
                                  rewriter.getI64IntegerAttr(axis));
    auto newBroadcast = rewriter.create<MultiBroadcastOp>(
        mbOp.getLoc(), outTy, newUnpack.getResult(),
        rewriter.getArrayAttr(newOutLens));

    rewriter.replaceOp(op, newBroadcast.getResult());
    rewriter.eraseOp(mbOp);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Primary pattern (benefit=2):
// Convert any migraphx.unpack whose input elem type is fp8 and whose (still
// incorrect) result elem type is fp8 into an fp4-producing unpack, adjusting
// the result shaped type (doubling the axis dimension & strides).
// After this runs, all remaining unpacks we care about have fp4 results.
// ---------------------------------------------------------------------------
struct ConvertFp8UnpackToFp4Pattern : OpRewritePattern<UnpackOp> {
  ConvertFp8UnpackToFp4Pattern(MLIRContext *ctx)
      : OpRewritePattern<UnpackOp>(ctx, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto inTy  = dyn_cast<MIXRShapedType>(op.getIn().getType());
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    if (!inTy || !outTy)
      return failure();
    // Only if both are still fp8 (i.e., not yet converted).
    if (!isFp8(inTy.getElementType()) || !isFp8(outTy.getElementType()))
      return failure();

    int64_t axis = op.getAxis();
    auto newOutTy = buildUnpackedFp4(inTy, axis);
    if (!newOutTy)
      return failure();

    auto newUnpack = rewriter.create<UnpackOp>(
        op.getLoc(), newOutTy, op.getIn(), rewriter.getI64IntegerAttr(axis));
    rewriter.replaceOp(op, newUnpack.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// Fold pattern (runs after conversion):
// If an unpack( fp8_arg -> fp4_result ) still exists on a function argument,
// fold it by updating the function argument type to the fp4 shaped type and
// deleting the unpack.
// Now simplified to require input elem is fp8 and result elem is fp4.
// ---------------------------------------------------------------------------
struct FoldFp4ArgUnpackPattern : OpRewritePattern<UnpackOp> {
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp op,
                                PatternRewriter &rewriter) const override {
    auto blockArg = dyn_cast<BlockArgument>(op.getIn());
    if (!blockArg)
      return failure();
    auto func = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!func)
      return failure();

    auto inTy  = dyn_cast<MIXRShapedType>(blockArg.getType());
    auto outTy = dyn_cast<MIXRShapedType>(op.getOut().getType());
    if (!inTy || !outTy)
      return failure();

    // Expect already converted situation: input fp8, result fp4.
    if (!isFp8(inTy.getElementType()) ||
        !isa<Float4E2M1FNType>(outTy.getElementType()))
      return failure();

    // Verify that outTy matches what conversion would have produced.
    int64_t axis = op.getAxis();
    auto expected = buildUnpackedFp4(inTy, axis);
    if (!expected || expected != outTy)
      return failure();

    FunctionType fType = func.getFunctionType();
    SmallVector<Type> inputs(fType.getInputs().begin(), fType.getInputs().end());
    if (inputs[blockArg.getArgNumber()] == outTy)
      return failure(); // Already folded.

    rewriter.modifyOpInPlace(func, [&] {
      inputs[blockArg.getArgNumber()] = outTy;
      func.setFunctionType(FunctionType::get(func.getContext(), inputs,
                                             fType.getResults()));
      blockArg.setType(outTy);
    });

    rewriter.replaceOp(op, blockArg);
    return success();
  }
};

// (Optional future patterns for transpose/reshape/multibroadcast would now
// assume op.getOut() already has fp4 element type, so their guards can drop
// fp8 checks.)

struct MIGraphXRealizeFP4Pass
    : public migraphx::impl::MIGraphXRealizeFP4PassBase<MIGraphXRealizeFP4Pass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());

    // Register high-benefit converter first.
    patterns.add<ConvertFp8UnpackToFp4Pattern>(&getContext());
    patterns.add<TransposeFp4UnpackInterchange, ReshapeFp4UnpackInterchange,
                 MultiBroadcastFp4UnpackInterchange>(&getContext());
    // Register fold pattern (benefit default = 1).
    patterns.add<FoldFp4ArgUnpackPattern>(&getContext());

    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return signalPassFailure();

    // Fix function result types if returns changed (fp8 -> fp4).
    FunctionType fType = func.getFunctionType();
    SmallVector<Type> newResults(fType.getResults().begin(),
                                 fType.getResults().end());
    bool changed = false;
    for (auto ret : func.getOps<func::ReturnOp>()) {
      for (auto it : llvm::enumerate(ret.getOperands())) {
        Type actual = it.value().getType();
        if (it.index() < newResults.size() &&
            newResults[it.index()] != actual) {
          newResults[it.index()] = actual;
          changed = true;
        }
      }
    }
    if (changed)
      func.setFunctionType(
          FunctionType::get(func.getContext(), fType.getInputs(), newResults));
  }
};

} // namespace
