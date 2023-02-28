//===- TosaToRockPass.cpp - Lowering Tosa to Rock Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the Rock dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToRock/TosaToRock.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_TOSATOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct TosaToRock : public impl::TosaToRockPassBase<TosaToRock> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    if (!func->hasAttr("kernel")) {
      return;
    }
    auto &ctx = getContext();
    // Split patterns into two stages by bufferization
    RewritePatternSet tensorPatterns(&ctx);
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);

    mlir::tosa::populateTosaToRockTensorConversionPatterns(&ctx,
                                                           tensorPatterns);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(tensorPatterns))))
      signalPassFailure();

    target.addLegalDialect<rock::RockDialect, tosa::TosaDialect,
                           tensor::TensorDialect,
                           bufferization::BufferizationDialect>();
    target.addIllegalOp<tosa::Conv2DOp, tosa::MatMulOp, tosa::ReduceSumOp>();

    mlir::tosa::populateTosaToRockConversionPatterns(func->getContext(),
                                                     patterns);
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::tosa::addTosaToRockPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createTosaToRockPass());
}
