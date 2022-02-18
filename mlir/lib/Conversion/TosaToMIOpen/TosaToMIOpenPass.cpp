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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
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
struct TosaToMIOpen : public TosaToMIOpenBase<TosaToMIOpen> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<miopen::MIOpenDialect, linalg::LinalgDialect,
                    bufferization::BufferizationDialect, StandardOpsDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    // Split patterns into two stages by bufferization
    RewritePatternSet tensor_patterns(&ctx);
    RewritePatternSet patterns(&ctx);
    ConversionTarget tensor_target(ctx);
    ConversionTarget target(ctx);
    auto func = getOperation();

    tensor_target.addLegalDialect<miopen::MIOpenDialect, tosa::TosaDialect,
                                  memref::MemRefDialect, StandardOpsDialect,
                                  BuiltinDialect, arith::ArithmeticDialect>();
    tensor_target.addDynamicallyLegalOp<tosa::TransposeOp>(
        [&](tosa::TransposeOp op) {
          auto attrDeletable = op->getAttr("changing_layout_root");
          if (attrDeletable)
            // Only apply the pattern to the transpose at the bottom
            return !attrDeletable.dyn_cast<BoolAttr>().getValue();
          return true;
        });
    mlir::tosa::populateTosaToMIOpenTensorConversionPatterns(func.getContext(),
                                                             tensor_patterns);
    if (failed(applyFullConversion(func, tensor_target,
                                   std::move(tensor_patterns))))
      signalPassFailure();

    target.addLegalDialect<miopen::MIOpenDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, tosa::TosaDialect,
                           bufferization::BufferizationDialect,
                           StandardOpsDialect>();
    target.addIllegalOp<tosa::Conv2DOp>();
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
  pm.addNestedPass<FuncOp>(createTosaToMIOpenPass());
}
