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

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCONVERT4BITALLOCTO8BITPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {

static bool isTarget4Bit(Type t) {
  if (auto ft = dyn_cast<FloatType>(t))
    return ft.getWidth() == 4; // f4E2M2FN
  if (auto it = dyn_cast<IntegerType>(t))
    return it.getWidth() == 4; // i4
  return false;
}

struct GpuMemcpyRewritePattern : public OpRewritePattern<gpu::MemcpyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(gpu::MemcpyOp op, 
                  PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getDst();
    if(src.getDefiningOp<mlir::UnrealizedConversionCastOp>() == nullptr ||
       dst.getDefiningOp<mlir::UnrealizedConversionCastOp>() == nullptr) {
      // no casts, nothing to do
      return failure();
    }
    while(auto cast = 
              src.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      src = cast.getInputs()[0];
    }
    while(auto cast = dst.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      dst = cast.getInputs()[0];
    }
    if(dst.getType() != src.getType()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<gpu::MemcpyOp>(op, TypeRange{}, ValueRange{dst, src}); 
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
      buffer = cast.getInputs()[0];
    }
    assert(buffer.getDefiningOp<gpu::AllocOp>() &&
           "expected gpu dealloc to use a gpu alloc");
    gpu::DeallocOp newDealloc = rewriter.create<gpu::DeallocOp>(
        op.getLoc(), TypeRange{}, ValueRange{buffer});
    rewriter.replaceOp(op, newDealloc);
    return success();
  }
};

struct GpuAllocRewritePattern : OpRewritePattern<gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto memrefTy = dyn_cast<MemRefType>(allocOp.getResult(0).getType());
    if (!memrefTy || !isTarget4Bit(memrefTy.getElementType()))
      return failure();

    auto i8Ty = rewriter.getI8Type();
    ArrayRef<int64_t> shape = memrefTy.getShape(); 
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape.back() =
        (newShape.back() + 1) / 2; // pack two 4 bit values in 1 i8
    auto newMemRefTy = MemRefType::get(newShape, i8Ty,
                                       memrefTy.getLayout(),
                                       memrefTy.getMemorySpace());
    rewriter.setInsertionPoint(allocOp);
    auto newAlloc = rewriter.create<gpu::AllocOp>(
        allocOp.getLoc(), newMemRefTy, ValueRange{allocOp.getDynamicSizes()},
        ValueRange{allocOp.getAsyncDependencies()}, ValueRange{allocOp.getSymbolOperands()},
        allocOp.getHostShared());

    auto castBack = rewriter.create<UnrealizedConversionCastOp>(
        allocOp.getLoc(), TypeRange{memrefTy}, ValueRange{newAlloc.getResult(0)});

    allocOp.replaceAllUsesWith(castBack.getResults());
    rewriter.eraseOp(allocOp);
    return success();
  }
};

class RockConvert4BitAllocTo8BitPass
    : public rock::impl::RockConvert4BitAllocTo8BitPassBase<
          RockConvert4BitAllocTo8BitPass> {
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

