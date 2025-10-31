#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPACK4BITGPUOPSTO8BITPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {

static bool isTarget4Bit(Type t) { return t.getIntOrFloatBitWidth() == 4; }

struct GpuMemcpyRewritePattern : public OpRewritePattern<gpu::MemcpyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::MemcpyOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getDst();
    if (src.getDefiningOp<mlir::UnrealizedConversionCastOp>() == nullptr ||
        dst.getDefiningOp<mlir::UnrealizedConversionCastOp>() == nullptr) {
      // no casts, nothing to do
      return failure();
    }
    while (auto cast = src.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1) {
        return rewriter.notifyMatchFailure(
            op, "cannot handle unrealized_conversion_cast with multiple inputs "
                "for source operand");
      }
      src = cast.getInputs()[0];
    }
    while (auto cast = dst.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1) {
        return rewriter.notifyMatchFailure(
            op, "cannot handle unrealized_conversion_cast with multiple inputs "
                "for destination operand");
      }
      dst = cast.getInputs()[0];
    }
    if (dst.getType() != src.getType()) {
      return failure();
    }
    // Preserve async token type and dependencies from the original memcpy
    SmallVector<Type> resultTypes;
    if (op.getAsyncToken())
      resultTypes.push_back(op.getAsyncToken().getType());
    rewriter.replaceOpWithNewOp<gpu::MemcpyOp>(
        op, resultTypes, op.getAsyncDependencies(), dst, src);
    return success();
  }
};

struct GpuDeallocRewritePattern : public OpRewritePattern<gpu::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    Value buffer = op.getMemref();
    if (buffer.getDefiningOp<gpu::AllocOp>()) {
      return failure();
    }
    while (auto cast =
               buffer.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1) {
        return rewriter.notifyMatchFailure(
            op, "cannot handle unrealized_conversion_cast with multiple inputs "
                "for memref operand");
      }
      buffer = cast.getInputs()[0];
    }
    assert(buffer.getDefiningOp<gpu::AllocOp>() &&
           "expected gpu dealloc to use a gpu alloc");
    // Preserve async token type and dependencies from the original dealloc
    SmallVector<Type> resultTypes;
    if (op.getAsyncToken())
      resultTypes.push_back(op.getAsyncToken().getType());
    rewriter.replaceOpWithNewOp<gpu::DeallocOp>(
        op, resultTypes, op.getAsyncDependencies(), buffer);
    return success();
  }
};

struct GpuAllocRewritePattern : OpRewritePattern<gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto memrefTy = dyn_cast<MemRefType>(allocOp.getMemref().getType());
    if (!memrefTy || !isTarget4Bit(memrefTy.getElementType()))
      return failure();

    ArrayRef<int64_t> shape = memrefTy.getShape();
    // Check if the last dimension is dynamic - we cannot handle this case
    // because we'd need to compute (dynamicSize + 1) / 2 at runtime
    if (shape.empty() || ShapedType::isDynamic(shape.back())) {
      return rewriter.notifyMatchFailure(
          allocOp, "cannot convert 4-bit alloc with dynamic last dimension");
    }

    auto i8Ty = rewriter.getI8Type();
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    // Pack two 4-bit values in 1 i8, using ceiling division
    newShape.back() = llvm::divideCeil(newShape.back(), 2);
    auto newMemRefTy = MemRefType::get(newShape, i8Ty, memrefTy.getLayout(),
                                       memrefTy.getMemorySpace());

    // Preserve async token type if present
    SmallVector<Type> newAllocResultTypes{newMemRefTy};
    if (allocOp.getAsyncToken())
      newAllocResultTypes.push_back(allocOp.getAsyncToken().getType());

    rewriter.setInsertionPoint(allocOp);
    auto newAlloc = rewriter.create<gpu::AllocOp>(
        allocOp.getLoc(), newAllocResultTypes,
        ValueRange{allocOp.getDynamicSizes()},
        ValueRange{allocOp.getAsyncDependencies()},
        ValueRange{allocOp.getSymbolOperands()}, allocOp.getHostShared());

    // Cast back: convert (new_i8_memref [, token]) to (old_i4_memref [, token])
    // Collect inputs for the cast (memref and optionally token)
    SmallVector<Value> castInputs{newAlloc.getMemref()};
    SmallVector<Type> castOutputTypes{memrefTy};
    if (allocOp.getAsyncToken()) {
      castInputs.push_back(newAlloc.getAsyncToken());
      castOutputTypes.push_back(allocOp.getAsyncToken().getType());
    }

    auto castBack = rewriter.create<UnrealizedConversionCastOp>(
        allocOp.getLoc(), castOutputTypes, castInputs);

    allocOp.replaceAllUsesWith(castBack.getResults());
    rewriter.eraseOp(allocOp);
    return success();
  }
};

class RockPack4BitGpuOpsTo8BitPass
    : public rock::impl::RockPack4BitGpuOpsTo8BitPassBase<
          RockPack4BitGpuOpsTo8BitPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<GpuAllocRewritePattern, GpuDeallocRewritePattern,
                 GpuMemcpyRewritePattern>(&getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
