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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "migraphx-to-tosa"

namespace mlir {
#define GEN_PASS_DEF_MIGRAPHXTOTOSAPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
namespace {
struct MIGraphXToTosa : public impl::MIGraphXToTosaPassBase<MIGraphXToTosa> {
public:
  void runOnOperation() override;
};
} // end namespace

void mlir::migraphx::populateMIGraphXToTosaDialectConversion(
    ConversionTarget &target, TypeConverter *typeConverter) {
  target.addIllegalDialect<migraphx::MIGraphXDialect>();
  target
      .addLegalOp<migraphx::AsLogicalShapeOp, migraphx::AsUnderlyingShapeOp>();
  target.addDynamicallyLegalDialect<tosa::TosaDialect, arith::ArithDialect>(
      [=](Operation *op) -> std::optional<bool> {
        return typeConverter->isLegal(op);
      });
  target.addDynamicallyLegalOp<linalg::GenericOp>(
      [=](linalg::GenericOp op) { return typeConverter->isLegal(op); });
  target.addLegalOp<tensor::EmptyOp, linalg::YieldOp, arith::UIToFPOp,
                    arith::ExtUIOp>();
}

void mlir::migraphx::populateMIGraphXFuncBoundaryToTosaDialectConversion(
    ConversionTarget &target, TypeConverter *typeConverter) {
  target.addIllegalDialect<migraphx::MIGraphXDialect>();
  target.addLegalDialect<tosa::TosaDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>(
      [=](func::FuncOp op) -> std::optional<bool> {
        return typeConverter->isSignatureLegal(op.getFunctionType());
      });
  target.addDynamicallyLegalOp<func::CallOp>(
      [=](func::CallOp op) -> std::optional<bool> {
        return typeConverter->isSignatureLegal(op.getCalleeType());
      });
  target.addDynamicallyLegalOp<mhal::LaunchOp>(
      [=](mhal::LaunchOp op) -> std::optional<bool> {
        return typeConverter->isLegal(op.getResultTypes()) &&
               typeConverter->isLegal(op.getOperandTypes());
      });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [=](func::ReturnOp op) -> std::optional<bool> {
        return typeConverter->isLegal(op);
      });
}

void MIGraphXToTosa::runOnOperation() {
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();

  migraphx::MIXRShapedToTensorConverter bodyTypeConverter;
  ConversionTarget bodyConversionTarget(*ctx);
  RewritePatternSet bodyPatterns(ctx);
  migraphx::populateMIGraphXToTosaDialectConversion(bodyConversionTarget,
                                                    &bodyTypeConverter);
  migraphx::populateMIGraphXToTosaConversionPatterns(bodyPatterns,
                                                     bodyTypeConverter);
  if (failed(applyPartialConversion(func, bodyConversionTarget,
                                    std::move(bodyPatterns))))
    return signalPassFailure();

  // We do this is a second stage because, while MIGraphX operations are
  // converted such that a shaped type gets translated to a tensor of the same
  // logical shape. However, the inputs to and outputs of the function will be
  // laid out according to the strides in the shape, which may not line up in
  // the logical shape. During the conversion of the body, these boundaries were
  // marked with migraphx.mlir.as_logical_shape and
  // migraphx.mlir.as_underlying_shape cast operations. This pass rewrites away
  // those operations, replacing them with tensor operations like
  // `tosa.transpose` which encode the views needed to make the data in memory
  // look like its logical shape.
  //
  // This translation doesn't necessarily need to happen here. For instance, we
  // could have done one round of this conversion in the tosa-to-rock pass and
  // then translated any remaining shape conversions to linalg.generic
  // operations for validation. However, since we're only handling simple cases
  // currently, we do the conversion here.
  migraphx::MIXRShapedToMemoryLayoutConverter boundaryTypeConverter;
  ConversionTarget boundaryConversionTarget(*ctx);
  RewritePatternSet boundaryPatterns(ctx);
  migraphx::populateMIGraphXFuncBoundaryToTosaDialectConversion(
      boundaryConversionTarget, &boundaryTypeConverter);
  migraphx::populateMIGraphXFuncBoundaryToTosaConversionPatterns(
      boundaryPatterns, boundaryTypeConverter);
  if (failed(applyPartialConversion(func, boundaryConversionTarget,
                                    std::move(boundaryPatterns))))
    return signalPassFailure();

  OpPassManager cleanPM("func.func");
  cleanPM.addPass(createCSEPass());
  (void)runPipeline(cleanPM, func);
}

void mlir::migraphx::addMIGraphXToTosaPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createMIGraphXToTosaPass());
}
