//===- MIGraphXToTosaPass.cpp - Lowering MIGraphX to Tosa Dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes MIGraphX operations to the Tosa dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/MIGraphXToTosa/MIGraphXToTosa.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace {
// import tablegen'ed populate function
#include "MIGraphXToTosa.cpp.inc"

struct MIGraphXToTosa
    : public MIGraphXToTosaBase<MIGraphXToTosa> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect, migraphx::MIGraphXDialect, StandardOpsDialect>();
  }

  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<tosa::TosaDialect, migraphx::MIGraphXDialect, StandardOpsDialect>();
    target.addIllegalOp<migraphx::AddOp, migraphx::ConstantOp, migraphx::ConvolutionOp,
      migraphx::RsqrtOp, migraphx::ReluOp, migraphx::TransposeOp, migraphx::ReshapeOp
                        >();
    /*
    target.addIllegalOp<tosa::AddOp, tosa::SubOp, tosa::ReshapeOp, tosa::RsqrtOp,
                        tosa::MulOp, tosa::TransposeOp, tosa::PadOp, tosa::Conv2DOp,
                        tosa::ClampOp>();
                        */
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FuncOp func = getFunction();
    populateWithGenerated(func.getContext(), patterns);
   
    if (failed(applyFullConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::migraphx::createMIGraphXToTosa() {
  return std::make_unique<MIGraphXToTosa>();
}
void mlir::migraphx::addMIGraphXToTosaPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createMIGraphXToTosa());
}
