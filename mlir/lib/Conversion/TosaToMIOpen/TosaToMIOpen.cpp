//===- TosaToMIOpen.cpp - Lowering Tosa to MIOpen Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the MIOpen dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToMIOpen/TosaToMIOpen.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;


namespace {

class ConvConverter final : public OpConversionPattern<tosa::Conv2DOp> {
public:
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::Conv2DOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto results = op->getResults();
    auto input_t = operands[0];
    auto filter_t = operands[1];
    auto bias_t = operands[2];

    assert(results.size() == 1);

    for (Value in : op->getOperands()) {
      auto inType = in.getType().template cast<ShapedType>();
      if (!inType.hasStaticShape())
        return rewriter.notifyMatchFailure(
                                           op,
                                           "tosa to miopen conversion expects statically shaped tensors");
    }

    for (auto result : results) {
      auto resultType = result.getType().template cast<ShapedType>();
      if (!resultType.hasStaticShape())
        return rewriter.notifyMatchFailure(
                                           op,
                                           "tosa to linalg conversion expects statically shaped tensors");
    }

    auto outputType = getTypeConverter<BufferizeTypeConverter>()->convertType(results[0].getType()).cast<MemRefType>();
    
    Value output_t = rewriter.create<AllocOp>(loc, outputType);

    ValueRange args({filter_t, input_t, output_t});

    TypeRange resTypes;
    rewriter.create<mlir::miopen::Conv2DOp>(loc, resTypes, args);

    rewriter.replaceOp(op, output_t);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToMIOpenOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns) {
  static BufferizeTypeConverter bufferizer;
  patterns->insert<ConvConverter>(bufferizer, context);
}

