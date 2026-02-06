//===- SwapGemmOperands.cpp - Transform C = A * B to C^T = B^T * A^T ===//
//
// Copyright 2024 Advanced Micro Devices.
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
// This pass applies the matrix identity (AB)^T = B^T * A^T to transform
// the GEMM operation from C = A * B to C^T = B^T * A^T by swapping operands
// and adjusting transpose flags.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSWAPGEMMOPERANDSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-swap-gemm-operands"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// This pattern transforms C = A * B into C^T = B^T * A^T
/// by swapping A and B operands and flipping all transpose flags.
///
/// The mathematical identity used is: (AB)^T = B^T * A^T
///
/// For the original operation C = A * B:
///   - A is [G] x M x K (or K x M if aTransposed)
///   - B is [G] x K x N (or N x K if bTransposed)
///   - C is [G] x M x N (or N x M if cTransposed)
///
/// After transformation C^T = B^T * A^T:
///   - New A (old B) with flipped transpose: if B was K x N, B^T is N x K
///   - New B (old A) with flipped transpose: if A was M x K, A^T is K x M
///   - New C with flipped transpose
struct SwapGemmOperands : public OpRewritePattern<rock::GemmOp> {
  using OpRewritePattern<rock::GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::GemmOp op,
                                PatternRewriter &rw) const override {
    // Skip if already processed (we use a marker attribute to avoid
    // infinite loops)
    if (op->hasAttr("rock.operands_swapped"))
      return failure();

    Location loc = op.getLoc();

    // Get original operands
    Value origA = op.getA();
    Value origB = op.getB();
    Value origC = op.getC();
    Value origScaleA = op.getScaleA();
    Value origScaleB = op.getScaleB();

    // Flip transpose flags:
    // Original: C = A * B
    // New:      C^T = B^T * A^T
    // So new_aTransposed = !old_bTransposed
    //    new_bTransposed = !old_aTransposed
    //    new_cTransposed = !old_cTransposed
    bool newATransposed = !op.getBTransposed();
    bool newBTransposed = !op.getATransposed();
    bool newCTransposed = !op.getCTransposed();

    // For scales, flip them similarly
    bool newAScaleTransposed = !op.getBScaleTransposed();
    bool newBScaleTransposed = !op.getAScaleTransposed();

    // Create the new GemmOp with swapped operands:
    // - New A is old B
    // - New B is old A
    // - New scaleA is old scaleB
    // - New scaleB is old scaleA
    auto newGemm = rock::GemmOp::create(
        rw, loc, origC.getType(),
        /*a=*/origB,
        /*b=*/origA,
        /*c=*/origC,
        /*scaleA=*/origScaleB,
        /*scaleB=*/origScaleA,
        /*aTransposed=*/newATransposed,
        /*bTransposed=*/newBTransposed,
        /*cTransposed=*/newCTransposed,
        /*aScaleTransposed=*/newAScaleTransposed,
        /*bScaleTransposed=*/newBScaleTransposed,
        op.getFeaturesAttr(), op.getStoreMethod(), op.getDerivedBlockSizeAttr(),
        op.getGridSizeAttr(), op.getParamsAttr());

    // Mark the new op to prevent re-processing
    newGemm->setAttr("rock.operands_swapped", rw.getUnitAttr());

    // Copy over optional attributes like perf_config
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newGemm->setAttr("perf_config", attr);

    // Replace the original op
    rw.replaceOp(op, newGemm->getResults());

    return success();
  }
};

struct RockSwapGemmOperandsPass
    : public rock::impl::RockSwapGemmOperandsPassBase<RockSwapGemmOperandsPass> {
  void runOnOperation() override;
};
} // end namespace

void RockSwapGemmOperandsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    // disable for non-kernels
    return;
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<SwapGemmOperands>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
}
