//===- MIGraphXToTosaPass.cpp - Lowering MIGraphX to Tosa Dialect
//-------------===//
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

#include "mlir/Conversion/MIGraphXToTosa/MIGraphXToTosa.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_MIGRAPHXTOTOSAPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
namespace {
// import tablegen'ed populate function
#include "MIGraphXToTosa.cpp.inc"

struct MIGraphXToTosa : public impl::MIGraphXToTosaPassBase<MIGraphXToTosa> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect, migraphx::MIGraphXDialect,
                    arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<tosa::TosaDialect, func::FuncDialect,
                           mhal::MHALDialect>();
    target.addIllegalDialect<migraphx::MIGraphXDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    func::FuncOp func = getOperation();
    populateWithGenerated(patterns);
    migraphx::populateMIGraphXToTosaConversionPatterns(func.getContext(),
                                                       patterns);

    if (failed(applyFullConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }

    OpPassManager cleanPM("func.func");
    cleanPM.addPass(createCSEPass());
    (void)runPipeline(cleanPM, func);
  }
};

} // namespace

void mlir::migraphx::addMIGraphXToTosaPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createMIGraphXToTosaPass());
}
