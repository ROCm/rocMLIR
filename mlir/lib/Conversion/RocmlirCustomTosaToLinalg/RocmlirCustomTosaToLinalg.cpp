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
// This pass lowers custom TOSA ops with the "rocmlir" domain and TOSA maximum
// ops carrying rocMLIR-specific `nsz` semantics to Linalg ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RocmlirCustomTosaToLinalg/RocmlirCustomTosaToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/RockTosaCustomOps.h"
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

struct UnsignedOpLoweringPattern : public OpConversionPattern<tosa::CustomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct MaximumOpLoweringPattern : public OpConversionPattern<tosa::MaximumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::MaximumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end namespace

LogicalResult UnsignedOpLoweringPattern::matchAndRewrite(
    tosa::CustomOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME)
    return rewriter.notifyMatchFailure(op, "domain isn't rocmlir");
  if (op.getOperatorName() != ROCK_CUSTOMOP_UNSIGNED_CAST &&
      op.getOperatorName() != ROCK_CUSTOMOP_UNSIGNED_DIV &&
      op.getOperatorName() != ROCK_CUSTOMOP_UNSIGNED_MAX)
    return rewriter.notifyMatchFailure(
        op, "isn't an unsigned_cast, unsigned_div, or unsigned_max");

  Location loc = op.getLoc();
  auto outType = cast<RankedTensorType>(op.getResults().front().getType());
  Type inElemType = cast<RankedTensorType>(op.getInputList().front().getType())
                        .getElementType();
  Type outElemType = outType.getElementType();
  Value emptyTensor = tensor::EmptyOp::create(rewriter, loc, outType,
                                              /*dynamic_sizes=*/ValueRange{});

  SmallVector<AffineMap> iterationMaps(
      op.getInputList().size() + 1,
      rewriter.getMultiDimIdentityMap(outType.getRank()));
  SmallVector<utils::IteratorType> iteratorKinds(outType.getRank(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, outType, adaptor.getInputList(), emptyTensor,
      iterationMaps, iteratorKinds,
      [&](OpBuilder &b, Location loc, ValueRange inputs) {
        Value result;
        if (op.getOperatorName() == ROCK_CUSTOMOP_UNSIGNED_CAST) {
          assert(inputs.size() == 2);
          if (isa<IntegerType>(inElemType)) {
            if (isa<FloatType>(outElemType)) {
              result = arith::UIToFPOp::create(b, loc, outElemType, inputs[0]);
            } else if (outElemType.getIntOrFloatBitWidth() >
                       inElemType.getIntOrFloatBitWidth()) {
              result = arith::ExtUIOp::create(b, loc, outElemType, inputs[0]);
            } else {
              result = arith::TruncIOp::create(b, loc, outElemType, inputs[0]);
            }
          } else {
            assert(isa<FloatType>(inElemType));
            assert(isa<IntegerType>(outElemType));
            result = arith::FPToUIOp::create(b, loc, outElemType, inputs[0]);
          }
        } else if (op.getOperatorName() == ROCK_CUSTOMOP_UNSIGNED_DIV) {
          assert(isa<IntegerType>(outElemType));
          assert(isa<IntegerType>(inElemType));
          assert(inputs.size() == 3);
          result =
              arith::DivUIOp::create(b, loc, outElemType, inputs[0], inputs[1]);
        } else {
          assert(op.getOperatorName() == ROCK_CUSTOMOP_UNSIGNED_MAX);
          assert(isa<IntegerType>(outElemType));
          assert(isa<IntegerType>(inElemType));
          assert(inputs.size() == 3);
          result =
              arith::MaxUIOp::create(b, loc, outElemType, inputs[0], inputs[1]);
        }
        linalg::YieldOp::create(b, loc, result);
      });
  rewriter.replaceOp(op, genericOp);
  return success();
}

LogicalResult MaximumOpLoweringPattern::matchAndRewrite(
    tosa::MaximumOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!op->hasAttr(ROCK_ATTR_NO_SIGNED_ZEROS))
    return rewriter.notifyMatchFailure(op, "does not request nsz semantics");

  Location loc = op.getLoc();
  auto outType = cast<RankedTensorType>(op.getType());
  if (!llvm::all_of(adaptor.getOperands(),
                    [&](Value value) { return value.getType() == outType; }))
    return rewriter.notifyMatchFailure(
        op, "nsz lowering requires equal-shaped operands and result");

  Type outElemType = outType.getElementType();
  Value emptyTensor = tensor::EmptyOp::create(rewriter, loc, outType,
                                              /*dynamic_sizes=*/ValueRange{});
  SmallVector<AffineMap> iterationMaps(
      adaptor.getOperands().size() + 1,
      rewriter.getMultiDimIdentityMap(outType.getRank()));
  SmallVector<utils::IteratorType> iteratorKinds(outType.getRank(),
                                                 utils::IteratorType::parallel);
  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, outType, adaptor.getOperands(), emptyTensor, iterationMaps,
      iteratorKinds, [&](OpBuilder &b, Location bodyLoc, ValueRange inputs) {
        Value result;
        if (isa<FloatType>(outElemType)) {
          if (op.getNanMode() == tosa::NanPropagationMode::PROPAGATE) {
            result = arith::MaximumFOp::create(b, bodyLoc, inputs[0], inputs[1],
                                               arith::FastMathFlags::nsz);
          } else {
            result = arith::MaxNumFOp::create(b, bodyLoc, inputs[0], inputs[1],
                                              arith::FastMathFlags::nsz);
          }
        } else {
          assert(outElemType.isSignlessInteger());
          result = arith::MaxSIOp::create(b, bodyLoc, inputs[0], inputs[1]);
        }
        linalg::YieldOp::create(b, bodyLoc, result);
      });
  rewriter.replaceOp(op, genericOp);
  return success();
}

void mlir::rock::populateRocmlirCustomTosaToLinalgTarget(
    ConversionTarget &target) {
  target.addLegalOp<linalg::GenericOp, linalg::YieldOp, arith::ExtUIOp,
                    arith::TruncIOp, arith::DivUIOp, arith::MaximumFOp,
                    arith::MaxNumFOp, arith::MaxSIOp, arith::MaxUIOp,
                    arith::FPToUIOp, arith::UIToFPOp, tensor::EmptyOp>();
  target.addDynamicallyLegalOp<tosa::CustomOp>([](tosa::CustomOp op) {
    return op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME;
  });
  target.addDynamicallyLegalOp<tosa::MaximumOp>([](tosa::MaximumOp op) {
    return !op->hasAttr(ROCK_ATTR_NO_SIGNED_ZEROS);
  });
}

void mlir::rock::populateRocmlirCustomTosaToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MaximumOpLoweringPattern, UnsignedOpLoweringPattern>(
      patterns.getContext());
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
