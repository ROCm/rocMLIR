//===--- MIGraphXToLinalgPass.cpp - Lowering MIGraphX to Linalg Dialect ---==//
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_MIGRAPHXTOLINALGPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

namespace {
struct MIGraphXToLinalgPass
    : public impl::MIGraphXToLinalgPassBase<MIGraphXToLinalgPass> {
  void runOnOperation() override;
};
} // namespace

void mlir::migraphx::populateMIGraphXToLinalgDialectConversion(
    ConversionTarget &target) {
  target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                         tensor::TensorDialect, math::MathDialect>();
  target.addIllegalDialect<migraphx::MIGraphXDialect>();
  target
      .addLegalOp<migraphx::AsLogicalShapeOp, migraphx::AsUnderlyingShapeOp>();
}

void mlir::migraphx::populateMIGraphXToLinalgBoundaryDialectConversion(
    ConversionTarget &target, TypeConverter &typeConverter) {
  target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                         tensor::TensorDialect, math::MathDialect>();
  target.addIllegalDialect<migraphx::MIGraphXDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
}

void MIGraphXToLinalgPass::runOnOperation() {
  // MIGraphX to Linalg conversion is performed in two passes:
  //
  // Pass 1: Convert MIGraphX operations to their Linalg equivalents.
  // The !migraphx.shaped type contains both shape and stride (memory layout)
  // information. During this pass, sourceMaterialization and
  // targetMaterialization insert temporary ops (migraphx.mlir.as_logical_shape
  // and migraphx.mlir.as_underlying_shape) to handle the type conversions.
  //
  // Pass 2: Convert the boundary/materialization operations to proper
  // memory layout representations using tensor operations, completing the
  // conversion from MIGraphX's shaped types to standard tensor types.
  MLIRContext *ctx = &getContext();
  func::FuncOp func = getOperation();

  ConversionTarget bodyConversionTarget(*ctx);
  migraphx::MIXRShapedToTensorConverter typeConverter;
  RewritePatternSet bodyPatterns(ctx);
  migraphx::populateMIGraphXToLinalgDialectConversion(bodyConversionTarget);
  migraphx::populateMIGraphXToLinalgConversionPatterns(typeConverter,
                                                       bodyPatterns);
  if (failed(applyPartialConversion(func, bodyConversionTarget,
                                    std::move(bodyPatterns)))) {
    return signalPassFailure();
  }

  ConversionTarget boundaryConversionTarget(*ctx);
  migraphx::MIXRShapedToMemoryLayoutConverter boundaryTypeConverter;
  RewritePatternSet boundaryPattern(ctx);
  migraphx::populateMIGraphXToLinalgBoundaryDialectConversion(
      boundaryConversionTarget, boundaryTypeConverter);
  migraphx::populateMIGraphXFuncBoundaryToLinalgConversionPatterns(
      boundaryPattern, boundaryTypeConverter);
  if (failed(applyPartialConversion(func, boundaryConversionTarget,
                                    std::move(boundaryPattern)))) {
    return signalPassFailure();
  }
}
