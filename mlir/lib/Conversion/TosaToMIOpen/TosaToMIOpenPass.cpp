//===- TosaToMIOpenPass.cpp - Lowering Tosa to MIOpen Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the MIOpen dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/TosaToMIOpen/TosaToMIOpen.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct TosaToMIOpen : public TosaToMIOpenBase<TosaToMIOpen> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<miopen::MIOpenDialect, linalg::LinalgDialect,
                    bufferization::BufferizationDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (!func->hasAttr("kernel")) {
      return;
    }
    auto &ctx = getContext();
    // Split patterns into two stages by bufferization
    RewritePatternSet tensor_patterns(&ctx);
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);

    mlir::tosa::populateTosaToMIOpenTensorConversionPatterns(&ctx,
                                                             tensor_patterns);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(tensor_patterns))))
      signalPassFailure();

    target.addLegalDialect<miopen::MIOpenDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, tosa::TosaDialect,
                           bufferization::BufferizationDialect,
                           mlir::func::FuncDialect>();
    target.addIllegalOp<tosa::Conv2DOp, tosa::MatMulOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    bufferization::BufferizeTypeConverter typeConverter;
    mlir::tosa::populateTosaToMIOpenConversionPatterns(
        typeConverter, func->getContext(), patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToMIOpenPass() {
  return std::make_unique<TosaToMIOpen>();
}

void mlir::tosa::addTosaToMIOpenPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createTosaToMIOpenPass());
}
