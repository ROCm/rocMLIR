//===- LowerExpandStrides.cpp - Lower rock.expand_strides to transforms --===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// This pass lowers rock.expand_strides operations on memrefs to
// rock.transform + memref.copy, avoiding the generic memref.subview pattern.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKEXPANDSTRIDESLOWERINGPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

struct ExpandStridesLoweringPattern
    : public OpRewritePattern<rock::ExpandStridesOp> {
  using OpRewritePattern<rock::ExpandStridesOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ExpandStridesOp op,
                                PatternRewriter &rewriter) const override {
    // Only lower memref version (no result means post-bufferization)
    if (op.getResult())
      return rewriter.notifyMatchFailure(op, "tensor version, skip");

    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    
    auto inputType = cast<MemRefType>(input.getType());
    auto outputType = cast<MemRefType>(output.getType());
    
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t rank = inputType.getRank();

    // Build dimension names
    SmallVector<StringRef> lowerDims, upperDims;
    SmallVector<int64_t> begins, ends;
    for (int64_t i = 0; i < rank; ++i) {
      lowerDims.push_back(rewriter.getStringAttr("dim" + std::to_string(i)));
      upperDims.push_back(rewriter.getStringAttr("slice" + std::to_string(i)));
      begins.push_back(0);
      ends.push_back(inputShape[i]);
    }

    // Create the transform: output[4,48,24] -> view[4,24,24] using Slice
    // Start with the output dimensions
    BottomUpTMBuilder builder(rewriter, lowerDims, outputShape, loc);
    
    // Apply Slice to get the smaller view matching input shape
    builder.slice(upperDims, lowerDims, begins, ends);
    
    TransformMapAttr transform = builder.get();
    
    // Create transform view of output
    Value outputView = rock::TransformOp::create(
        rewriter, loc, output, transform);

    // Copy input into the transformed view
    memref::CopyOp::create(rewriter, loc, input, outputView);
    
    rewriter.eraseOp(op);
    return success();
  }
};

struct RockExpandStridesLoweringPass
    : public rock::impl::RockExpandStridesLoweringPassBase<
          RockExpandStridesLoweringPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ExpandStridesLoweringPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
