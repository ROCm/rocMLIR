//===- TosaToMIGraphXPass.cpp - Lowering Tosa to MIGraphX Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes Tosa operations to the MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/TosaToMIGraphX/TosaToMIGraphX.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
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
struct TosaToMIGraphXRandom
    : public TosaToMIGraphXRandomBase<TosaToMIGraphXRandom> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<migraphx::MIGraphXDialect, StandardOpsDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<tosa::TosaDialect, migraphx::MIGraphXDialect, StandardOpsDialect>();
    target.addIllegalOp<tosa::ConstOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FuncOp func = getFunction();
    mlir::tosa::populateConstRandomPatterns(
        func.getContext(), &patterns);

//llvm::errs()<< "run on function!!\n"; 

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {

        llvm::errs()<< "fully conv failed!!\n"; 
      signalPassFailure();

    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaToMIGraphXRandom() {
  return std::make_unique<TosaToMIGraphXRandom>();
}

void mlir::tosa::addTosaToMIGraphXRandomPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createTosaToMIGraphXRandom());
}