//===- LinalgToRockPass.cpp - Lowering Linalg to Rock Dialect ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Linalg to the Rock dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/LinalgToRock/LinalgToRock.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_LINALGTOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

namespace {
struct LinalgToRockPass : public impl::LinalgToRockPassBase<LinalgToRockPass> {
  void runOnOperation() override;
};
} // namespace

static void populateLinalgToRockDialectConversion(ConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect,
                         rock::RockDialect,
                         bufferization::BufferizationDialect, math::MathDialect>();

  // We only allow Linalg operations that are elementwise. Fusion is supported
  // via linalg.generic when it is an elementwise operation. Elementwise
  // operations would be converted into linalg.generic in later passes
  target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
      [=](Operation *op) -> std::optional<bool> {
        auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
        if (!linalgOp) {
          return std::nullopt;
        }
        return linalg::isElementwise(linalgOp) || isa<linalg::GenericOp>(op) ||
               isa<linalg::YieldOp>(op);
      });
}

static void populateLinalgGenericDialectConversion(ConversionTarget &target) {
  target.addIllegalDialect<linalg::LinalgDialect>();
  target.addLegalOp<linalg::GenericOp, linalg::YieldOp>();
}

void LinalgToRockPass::runOnOperation() {
  MLIRContext &ctx = getContext();
  func::FuncOp func = getOperation();
  if (!func->hasAttr("kernel")) {
    func->emitError("func op does not have the kernel attribute for "
                    "linalg-to-rock lowering");
    return signalPassFailure();
  }

  // Here we are doing two passes because applyPartialConversion doesn't
  // guarantee a ordering of the passes that is going performed. We want to
  // first try to convert all the named linalg operations first, and then
  // transform the remaining linalg operations into linalg.generic operations.
  ConversionTarget bodyConversionTarget(ctx);
  TypeConverter converter;
  RewritePatternSet bodyPatterns(&ctx);
  populateLinalgToRockDialectConversion(bodyConversionTarget);
  rock::populateLinalgToRockConversionPattern(bodyPatterns, &ctx);
  if (failed(applyPartialConversion(func, bodyConversionTarget,
                                    std::move(bodyPatterns)))) {
    return signalPassFailure();
  }

  // Converting the remaining linalg operations into linalg.generic
  ConversionTarget finalConversionTarget(ctx);
  RewritePatternSet genericPatternSet(&ctx);
  populateLinalgGenericDialectConversion(finalConversionTarget);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(genericPatternSet);
  if (failed(applyPartialConversion(func, finalConversionTarget,
                                    std::move(genericPatternSet)))) {
    return signalPassFailure();
  }
}
