//===- SortDimensionsMemoryLayoutPass.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass sort dimensions using the underlying memory layout.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SortDimensionsMemoryLayout/SortDimensionsMemoryLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SORTDIMENSIONSMEMORYLAYOUTPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;

template <typename OpT>
static SmallVector<Operation *> getOperations(func::FuncOp &func) {
  SmallVector<Operation *, 4> ops;
  func.walk([&ops](OpT operation) { ops.push_back(operation); });

  return ops;
}

namespace {
struct SortDimensionsMemoryLayoutPass
    : public impl::SortDimensionsMemoryLayoutPassBase<
          SortDimensionsMemoryLayoutPass> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    if (!func->hasAttr("kernel")) {
      return;
    }
    auto &ctx = getContext();
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    RewritePatternSet patternsConv(&ctx);
    mlir::populateSortConvRewritePatterns(&ctx, patternsConv);
    if (failed(applyOpPatternsGreedily(getOperations<rock::ConvOp>(func),
                                       std::move(patternsConv), config)))
      return signalPassFailure();

    RewritePatternSet patternsConvBwdData(&ctx);
    mlir::populateSortConvBwdDataRewritePatterns(&ctx, patternsConvBwdData);
    if (failed(applyOpPatternsGreedily(getOperations<rock::ConvBwdDataOp>(func),
                                       std::move(patternsConvBwdData), config)))
      return signalPassFailure();

    RewritePatternSet patternsConvBwdWeight(&ctx);
    mlir::populateSortConvBwdWeightRewritePatterns(&ctx, patternsConvBwdWeight);
    if (failed(
            applyOpPatternsGreedily(getOperations<rock::ConvBwdWeightOp>(func),
                                    std::move(patternsConvBwdWeight), config)))
      return signalPassFailure();

    RewritePatternSet patternsGemm(&ctx);
    mlir::populateSortGemmRewritePatterns(&ctx, patternsGemm);
    if (failed(applyOpPatternsGreedily(getOperations<rock::GemmOp>(func),
                                       std::move(patternsGemm), config)))
      return signalPassFailure();

    RewritePatternSet patternsAttention(&ctx);
    mlir::populateSortAttentionRewritePatterns(&ctx, patternsAttention);
    if (failed(applyOpPatternsGreedily(getOperations<rock::AttentionOp>(func),
                                       std::move(patternsAttention), config)))
      return signalPassFailure();
  }
};
} // namespace
