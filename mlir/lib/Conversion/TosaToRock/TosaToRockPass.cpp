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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<rock::RockDialect, linalg::LinalgDialect,
                    bufferization::BufferizationDialect, func::FuncDialect>();
  }

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

    target.addLegalDialect<rock::RockDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, tosa::TosaDialect,
                           bufferization::BufferizationDialect,
                           mlir::func::FuncDialect>();
    target.addIllegalOp<tosa::Conv2DOp, tosa::MatMulOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    bufferization::BufferizeTypeConverter typeConverter;
    mlir::tosa::populateTosaToRockConversionPatterns(
        typeConverter, func->getContext(), patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::tosa::addTosaToRockPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createTosaToRockPass());
}
