//===- RocmlirCustomTosaToLinalg.cpp - Lowering custom Tosa to Linalg --===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2024 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// This pass lowers custom Tosa ops with the "rocmlir" domain to Linalg ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RocmlirCustomTosaToLinalg/RocmlirCustomTosaToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_ROCMLIRCUSTOMTOSATOLINALGPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct RocmlirCustomLinalgToTosaPass
    : public impl::RocmlirCustomTosaToLinalgPassBase<
          RocmlirCustomLinalgToTosaPass> {
  void runOnOperation() override;
};

struct UnsignedCastLoweringPattern
    : public OpConversionPattern<tosa::CustomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end namespace

LogicalResult UnsignedCastLoweringPattern::matchAndRewrite(
    tosa::CustomOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (op.getDomainName() != "rocmlir")
    return rewriter.notifyMatchFailure(op, "domain isn't rocmlir");
  if (op.getOperatorName() != "unsigned_cast")
    return rewriter.notifyMatchFailure(op, "isn't an unsigned_cast");
  Location loc = op.getLoc();
  auto outType = cast<RankedTensorType>(op.getResults().front().getType());
  Type outElemType = outType.getElementType();
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, outType, /*dynamic_sizes=*/ValueRange{});

  SmallVector<AffineMap> iterationMaps(
      2, rewriter.getMultiDimIdentityMap(outType.getRank()));
  SmallVector<utils::IteratorType> iteratorKinds(outType.getRank(),
                                                 utils::IteratorType::parallel);
  auto cast = rewriter.create<linalg::GenericOp>(
      loc, outType, adaptor.getInputs(), emptyTensor, iterationMaps,
      iteratorKinds, [&](OpBuilder &b, Location loc, ValueRange inputs) {
        Value result;
        if (isa<FloatType>(outElemType))
          result = b.create<arith::UIToFPOp>(loc, outElemType, inputs[0]);
        else
          result = b.create<arith::ExtUIOp>(loc, outElemType, inputs[0]);
        b.create<linalg::YieldOp>(loc, result);
      });
  rewriter.replaceOp(op, cast);
  return success();
}

void mlir::rock::populateRocmlirCustomTosaToLinalgTarget(
    ConversionTarget &target) {
  target.addLegalOp<linalg::GenericOp, linalg::YieldOp, arith::ExtUIOp,
                    arith::UIToFPOp, tensor::EmptyOp>();
  target.addDynamicallyLegalOp<tosa::CustomOp>(
      [](tosa::CustomOp op) { return op.getDomainName() != "rocmlir"; });
}

void mlir::rock::populateRocmlirCustomTosaToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<UnsignedCastLoweringPattern>(patterns.getContext());
}

void RocmlirCustomLinalgToTosaPass::runOnOperation() {
  Operation *op = getOperation();

  ConversionTarget target(getContext());
  rock::populateRocmlirCustomTosaToLinalgTarget(target);

  RewritePatternSet patterns(&getContext());
  rock::populateRocmlirCustomTosaToLinalgConversionPatterns(patterns);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();
}
