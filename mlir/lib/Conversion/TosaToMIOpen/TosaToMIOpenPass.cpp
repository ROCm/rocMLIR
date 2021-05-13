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
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
struct TosaToMIOpenOnTensors
    : public TosaToMIOpenOnTensorsBase<TosaToMIOpenOnTensors> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<miopen::MIOpenDialect, linalg::LinalgDialect,
                    StandardOpsDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<miopen::MIOpenDialect, linalg::LinalgDialect,
                           StandardOpsDialect>();
    // target.addIllegalDialect<tosa::TosaDialect>();
    target.addIllegalOp<tosa::Conv2DOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FuncOp func = getFunction();
    mlir::tosa::populateTosaToMIOpenOnTensorsConversionPatterns(
        func.getContext(), &patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToMIOpenOnTensors() {
  return std::make_unique<TosaToMIOpenOnTensors>();
}

void mlir::tosa::addTosaToMIOpenOnTensorsPasses(OpPassManager &pm) {
  // pm.addNestedPass<FuncOp>(createTosaMakeBroadcastablePass());
  pm.addNestedPass<FuncOp>(createTosaToMIOpenOnTensors());
}
