//===- SwapGemmOperands.cpp - Swap GEMM operands for vectorization --------===//
//
// Copyright 2025 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
//
// This pass swaps A and B operands of rock.gemm to compute
// C^T = B^T * A^T instead of C = A * B, enabling better store
// vectorization. After MFMA/WMMA, each thread holds values along
// the column dimension. By computing the transposed product, each
// thread holds row values instead.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-swap-gemm-operands"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSWAPGEMMOPERANDSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

// Returns true if the value traces back to a block argument without
// hitting any intermediate memref::AllocOp (which indicates a fusion).
static bool tracesToBlockArgDirectly(Value v) {
  return succeeded(rock::findBlockArgument(v));
}

// Returns true if the gemm op has no input or output fusions.
static bool hasNoFusions(GemmOp op) {
  return tracesToBlockArgDirectly(op.getA()) &&
         tracesToBlockArgDirectly(op.getB()) &&
         tracesToBlockArgDirectly(op.getC());
}

// Returns true if the swap optimization should be applied.
static bool shouldSwapOperands(GemmOp op) {
  if (!hasNoFusions(op)) {
    LLVM_DEBUG(llvm::dbgs() << "SwapGemmOperands: skipping due to fusions\n");
    return false;
  }
  if (op.getCTransposed()) {
    LLVM_DEBUG(llvm::dbgs()
               << "SwapGemmOperands: skipping, C already transposed\n");
    return false;
  }
  return true;
}

namespace {
struct SwapGemmOperandsPattern : public OpRewritePattern<GemmOp> {
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmOp op, PatternRewriter &b) const override {
    if (!shouldSwapOperands(op))
      return failure();

    // C = A * B  =>  C^T = B^T * A^T
    //
    // Original: A is [G] x M x K (or [G] x K x M if aTransposed)
    //           B is [G] x K x N (or [G] x N x K if bTransposed)
    //           C is [G] x M x N
    //
    // After swap:
    //   new A' = old B, with bTransposed inverted (B^T)
    //   new B' = old A, with aTransposed inverted (A^T)
    //   C gets cTransposed set (output is C^T = [G] x N x M view)
    Value newA = op.getB();
    Value newB = op.getA();
    Value newScaleA = op.getScaleB();
    Value newScaleB = op.getScaleA();

    // Invert transpose flags: if B was transposed (N x K), then B^T is
    // not transposed (K x N), and vice versa.
    UnitAttr newATransposed = op.getBTransposed() ? nullptr : b.getUnitAttr();
    UnitAttr newBTransposed = op.getATransposed() ? nullptr : b.getUnitAttr();

    // Swap scale transpose flags correspondingly
    UnitAttr newAScaleTransposed =
        op.getBScaleTransposed() ? nullptr : b.getUnitAttr();
    UnitAttr newBScaleTransposed =
        op.getAScaleTransposed() ? nullptr : b.getUnitAttr();

    // Set cTransposed so GemmToGridwise adds the transpose view on C
    UnitAttr newCTransposed = b.getUnitAttr();

    auto resultType = op.getResult() ? op.getResult().getType() : Type{};

    auto newGemm = GemmOp::create(
        b, op.getLoc(), resultType, newA, newB, op.getC(), newScaleA, newScaleB,
        newATransposed, newBTransposed, newCTransposed, newAScaleTransposed,
        newBScaleTransposed, op.getFeaturesAttr(), op.getStoreMethodAttr(),
        op.getDerivedBlockSizeAttr(), op.getGridSizeAttr(), op.getParamsAttr());

    if (op.getResult())
      b.replaceOp(op, newGemm->getResults());
    else
      b.eraseOp(op);

    return success();
  }
};

struct RockSwapGemmOperandsPass
    : public rock::impl::RockSwapGemmOperandsPassBase<
          RockSwapGemmOperandsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SwapGemmOperandsPattern>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace
