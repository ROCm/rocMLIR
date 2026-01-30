//===---- MIGraphXToLinalgPass.cpp - Lowering MIGrpahX to Linalg Dialect
//----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes MIGraphX operations to the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MIGraphXToLinalg/MIGraphXToLinalg.h"
#include "mlir/Conversion/MIGraphXToTosa/MIGraphXToTosa.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MIGRAPHXTOLINALGPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
namespace {
struct MIGraphXToLinalgPass
    : public impl::MIGraphXToLinalgPassBase<MIGraphXToLinalgPass> {
  void runOnOperation() override;
};
} // namespace

void MIGraphXToLinalgPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();

  ConversionTarget bodyConversionTarget(*ctx);
  linalg::MIXRShapedToTensorConverter typeConverter;
  RewritePatternSet bodyPatterns(ctx);
  bodyConversionTarget.addLegalDialect<
      linalg::LinalgDialect, arith::ArithDialect, tensor::TensorDialect>();
  linalg::populateMIGraphXToLinalgConversionPatterns(typeConverter,
                                                     bodyPatterns);

  if (failed(applyPartialConversion(func, bodyConversionTarget,
                                    std::move(bodyPatterns)))) {
    return signalPassFailure();
  }

  // trying to figure out the shape from materialization
  ConversionTarget boundaryConversionTarget(*ctx);
  boundaryConversionTarget.addLegalDialect<
      linalg::LinalgDialect, arith::ArithDialect, tensor::TensorDialect>();
  linalg::BoundaryTypeConverter boundaryTypeConverter;
  RewritePatternSet boundaryPattern(ctx);

  boundaryConversionTarget.addIllegalDialect<migraphx::MIGraphXDialect>();
  boundaryConversionTarget.addLegalDialect<
      linalg::LinalgDialect, arith::ArithDialect, tensor::TensorDialect>();
  boundaryConversionTarget.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType());
      });
  boundaryConversionTarget.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
  boundaryConversionTarget.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });

  populateAnyFunctionOpInterfaceTypeConversionPattern(boundaryPattern,
                                                      boundaryTypeConverter);
  linalg::populateMIGraphXFuncBoundaryToLinalgConversionPatterns(
      boundaryPattern, boundaryTypeConverter);
  if (failed(applyPartialConversion(func, boundaryConversionTarget,
                                    std::move(boundaryPattern)))) {
    return signalPassFailure();
  }
}
