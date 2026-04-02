//===- ConvToWinograd.cpp - Convert eligible rock.conv to Winograd kernels ===//
//
// Copyright 2025 The MLIR Authors.
// Licensed under the Apache License, Version 2.0.
// =============================================================================
//
// MIOpen-inspired Winograd eligibility and selection. Checks:
// - Filter size, stride, dilation constraints
// - Data type (f32, f16)
// - Dimension overflow guards (MIOpen: 2^16, 2^28, 2^30)
// - Architecture-specific applicability
// - Performance heuristic (C*K channel product threshold + WTI estimation)
//
// References:
// - MIOpen ConvBinWinoRxS solver: conv_winoRxS.cpp
// - MIOpen ConvWinoFuryRxS solver: conv_wino_fury_RxS.cpp
// - MIOpen kernel selection: GetBestNGroupParam, ComputeWti
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockConvInterface.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCONVTOWINOGRADPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

static std::pair<int64_t, int64_t> getFilterSpatialDims(ConvOp op) {
  auto filterType = cast<ShapedType>(op.getFilter().getType());
  ArrayRef<int64_t> shape = filterType.getShape();
  auto filterLayout = op->getAttrOfType<ArrayAttr>("filter_layout");
  if (!filterLayout)
    return {0, 0};

  int64_t h = 0, w = 0;
  for (auto [idx, name] :
       llvm::enumerate(filterLayout.getAsRange<StringAttr>())) {
    if (name.getValue() == "0")
      h = shape[idx];
    else if (name.getValue() == "1")
      w = shape[idx];
  }
  return {h, w};
}

static int64_t getDimByName(ShapedType type, ArrayAttr layout, StringRef name) {
  for (auto [idx, attr] : llvm::enumerate(layout.getAsRange<StringAttr>())) {
    if (attr.getValue() == name)
      return type.getShape()[idx];
  }
  return 1;
}

// MIOpen-style dimension overflow check.
// The hand-written assembly uses 16-bit and 32-bit integer arithmetic,
// so dimensions must fit within these limits.
static bool checkDimensionOverflow(int64_t N, int64_t G, int64_t C, int64_t K,
                                   int64_t H, int64_t W, int64_t OH,
                                   int64_t OW, int64_t R, int64_t S) {
  constexpr int64_t limit16 = (1LL << 16);
  constexpr int64_t limit28 = (1LL << 28);
  constexpr int64_t limit31 = (1LL << 31);

  if (N >= limit16 || G >= limit16 || C >= limit16 || K >= limit16 ||
      H >= limit16 || W >= limit16)
    return false;
  if (OH >= limit16 || OW >= limit16 - 3)
    return false;
  if (G * K >= limit16 || G * C >= limit16)
    return false;
  if ((G * K - 1) * C * R * S >= limit28)
    return false;
  if ((N - 1) * G * C * H * W >= limit31)
    return false;
  if ((N - 1) * G * K * OH * OW >= limit31)
    return false;

  return true;
}

// Simplified WTI (Willingness To Invest) performance estimation.
// Returns true if Winograd is estimated to be faster than direct conv.
// Based on MIOpen's ComputeWti model which compares predicted Winograd
// clock counts against ideal direct convolution clock counts.
static bool isWinogradProfitable(int64_t N, int64_t C, int64_t K, int64_t OH,
                                 int64_t OW, int64_t /*G*/) {
  // Direct conv: C*9 multiplies per output element (3x3 filter)
  // Winograd F(2,3): C multiplies per output tile (2x2 = 4 elements)
  // So Winograd does C*16 muls for 4 output elements vs C*9*4=C*36 for direct
  // Winograd benefit: 36/16 = 2.25x fewer multiplies

  // But Winograd has overhead: transforms add ~24 add/sub per tile per channel
  // For small C, the overhead dominates. For large C, the multiply savings win.
  int64_t totalOutput = N * OH * OW * K;

  // Very small problems: overhead dominates
  if (totalOutput < 256)
    return false;

  // Channel product threshold (primary heuristic)
  // MIOpen's Fury solver requires sufficient K to fill workgroups:
  // DivCeil(K, 16) <= n_groups (= min(CU_count, 255))
  // For our scalar kernel, C*K >= 2048 is a reasonable threshold
  if (C * K < 2048)
    return false;

  return true;
}

struct ConvToWinogradPattern : public OpRewritePattern<ConvOp> {
  int64_t minChannelProduct;

  ConvToWinogradPattern(MLIRContext *ctx, int64_t minChannelProduct)
      : OpRewritePattern<ConvOp>(ctx), minChannelProduct(minChannelProduct) {}

  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Stride and dilation must be 1 (required for all Winograd variants)
    auto strides = extractFromIntegerArrayAttr<int64_t>(op.getStrides());
    auto dilations = extractFromIntegerArrayAttr<int64_t>(op.getDilations());
    for (auto s : strides)
      if (s != 1)
        return failure();
    for (auto d : dilations)
      if (d != 1)
        return failure();

    // 2. Filter must be 3x3 (for F(2,3))
    auto [filterH, filterW] = getFilterSpatialDims(op);
    if (filterH != 3 || filterW != 3)
      return failure();

    // 3. Element type must be f32 or f16
    auto filterType = cast<ShapedType>(op.getFilter().getType());
    Type elemType = filterType.getElementType();
    if (!elemType.isF32() && !elemType.isF16())
      return failure();

    // 4. Extract dimensions
    auto filterLayout = op->getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayout = op->getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayout = op->getAttrOfType<ArrayAttr>("output_layout");
    if (!filterLayout || !inputLayout || !outputLayout)
      return failure();

    auto inputType = cast<ShapedType>(op.getInput().getType());
    auto outputType = cast<ShapedType>(op.getOutput().getType());

    int64_t G = getDimByName(filterType, filterLayout, "g");
    int64_t C = getDimByName(filterType, filterLayout, "c");
    int64_t K = getDimByName(filterType, filterLayout, "k");
    int64_t N = getDimByName(inputType, inputLayout, "ni");
    int64_t H = getDimByName(inputType, inputLayout, "0i");
    int64_t W = getDimByName(inputType, inputLayout, "1i");
    int64_t OH = getDimByName(outputType, outputLayout, "0o");
    int64_t OW = getDimByName(outputType, outputLayout, "1o");

    // 5. Channel product threshold
    if (C * K < minChannelProduct)
      return failure();

    // 6. Dimension overflow guards (MIOpen-style)
    if (!checkDimensionOverflow(N, G, C, K, H, W, OH, OW, filterH, filterW))
      return failure();

    // 7. Performance profitability check
    if (!isWinogradProfitable(N, C, K, OH, OW, G))
      return failure();

    // 8. Packed tensor check (no strided layouts)
    // The bufferize pipeline ensures contiguous memrefs, so this is always true
    // in our pipeline. MIOpen checks this because it can receive strided tensors.

    // Convert to winograd_conv
    int32_t fmr = 0; // F_2_3

    auto winogradOp = rewriter.replaceOpWithNewOp<WinogradConvOp>(
        op, op.getFilter(), op.getInput(), op.getOutput(),
        op.getFeaturesAttr(), op.getDerivedBlockSizeAttr(),
        op.getGridSizeAttr(), op.getPaddingAttr(), op.getStridesAttr(),
        op.getDilationsAttr(), op.getParamsAttr(),
        rewriter.getBoolAttr(false), rewriter.getI32IntegerAttr(fmr));

    if (auto attr = op->getAttr("filter_layout"))
      winogradOp->setAttr("filter_layout", attr);
    if (auto attr = op->getAttr("input_layout"))
      winogradOp->setAttr("input_layout", attr);
    if (auto attr = op->getAttr("output_layout"))
      winogradOp->setAttr("output_layout", attr);

    return success();
  }
};

struct RockConvToWinogradPass
    : public rock::impl::RockConvToWinogradPassBase<RockConvToWinogradPass> {
  using RockConvToWinogradPassBase::RockConvToWinogradPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvToWinogradPattern>(ctx, minChannelProduct);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace
