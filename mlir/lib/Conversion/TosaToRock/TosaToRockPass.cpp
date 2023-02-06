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
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
    // expand tosa.reshape to tensor.collapse_shape and tensor.expand_shape
    // within kernel functions.
    OpPassManager pm("func.func");
    pm.addPass(mlir::tosa::createTosaToTensor());
    if (failed(runPipeline(pm, func))) {
      signalPassFailure();
      return;
    }

    auto &ctx = getContext();
    // Split patterns into two stages by bufferization
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);

    target.addLegalDialect<rock::RockDialect, tosa::TosaDialect,
                           tensor::TensorDialect,
                           bufferization::BufferizationDialect>();
    target.addIllegalOp<tosa::Conv2DOp, tosa::MatMulOp, tensor::CollapseShapeOp,
                        tensor::ExpandShapeOp, tosa::TransposeOp>();

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
