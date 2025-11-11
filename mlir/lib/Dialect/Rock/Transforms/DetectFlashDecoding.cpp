//===- DetectFlashDecoding.cpp - Detect and fix flash decoding splitKV --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass detects flash decoding patterns (splitKV > 1) from rock.transform
// operations and updates the AttentionOp with the correct splitKV value.
// It also removes the splitKV broadcast from Q, K, V tensors since the
// attention implementation handles splitting internally.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKDETECTFLASHDECODINGPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-detect-flash-decoding"

using namespace mlir;
using namespace mlir::rock;

namespace {

// Helper to check if splitKV value is supported
// Supported values are powers of 2: 2, 4, 8, 16, 32, 64, 128
static bool isSupportedSplitKV(int64_t splitKV) {
  constexpr int64_t supportedValues[] = {2, 4, 8, 16, 32, 64, 128};
  return llvm::is_contained(supportedValues, splitKV);
}

// Detect splitKV from Q tensor by finding the Broadcast operation
// Q pattern: [B, H, 1, M, K] --Broadcast--> [B, H, SplitKV, M, K]
//                            --Merge--> [B*H*SplitKV, M, K]
// Returns the splitKV value if found, otherwise 1
static int64_t detectSplitKVFromQ(Value qTensor) {
  int64_t splitKV = 1;
  SmallVector<TransformMapAttr> transforms;
  rock::untransform(qTensor, transforms);

  if (transforms.empty())
    return splitKV;

  LLVM_DEBUG(llvm::dbgs() << "Analyzing Q tensor for splitKV:\n");

  // Look for a Broadcast operation at dimension 2
  for (TransformMapAttr transformMap : transforms) {
    ArrayRef<int64_t> upperBounds = transformMap.getUpperBounds();

    // We're looking for a 5D intermediate shape [B, H, 1, M, K]
    if (upperBounds.size() != 5)
      continue;

    for (rock::TransformAttr op : transformMap.getOps()) {
      // Look for Broadcast operation
      if (op.getType() != rock::TransformType::Broadcast)
        continue;

      ArrayRef<uint32_t> upperDims = op.getUpperDims();
      ArrayRef<int64_t> params = op.getParams();

      // Check if broadcasting dimension 2 (splitKV position) and that
      // params[0] is the size being broadcasted from (should be 1)
      if (!upperDims.empty() && upperDims[0] == 2 && !params.empty() &&
          params[0] == 1) {
        // The upper bound at dimension 2 is the splitKV value
        auto potentialSplitKV = upperBounds[2];

        if (isSupportedSplitKV(potentialSplitKV)) {
          splitKV = potentialSplitKV;
          LLVM_DEBUG(llvm::dbgs() << "\tQ: Found Broadcast at dim 2, "
                                  << "splitKV = " << splitKV << "\n");
          return splitKV;
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\tQ: No Broadcast pattern found\n");
  return splitKV;
}

// Detect splitKV from V tensor by finding the Unmerge with splitKV dimension
// V pattern: flat --Unmerge--> [B, D, SplitKV, N/SplitKV] ...
// Returns the splitKV value if found, otherwise 1
static int64_t detectSplitKVFromV(Value vTensor) {
  int64_t splitKV = 1;
  SmallVector<TransformMapAttr> transforms;
  rock::untransform(vTensor, transforms);

  if (transforms.empty())
    return splitKV;

  LLVM_DEBUG(llvm::dbgs() << "Analyzing V tensor for splitKV:\n");

  // Look for an Unmerge operation that creates a 5D shape with splitKV
  for (TransformMapAttr transformMap : transforms) {
    ArrayRef<int64_t> upperBounds = transformMap.getUpperBounds();

    // We're looking for a 5D intermediate shape where G is the splitKV
    // dimension at position 3
    if (upperBounds.size() != 5)
      continue;

    for (rock::TransformAttr op : transformMap.getOps()) {
      if (op.getType() != rock::TransformType::Unmerge)
        continue;

      ArrayRef<int64_t> params = op.getParams();
      if (params.size() != 4)
        continue;

      auto potentialSplitKV = params[2];

      // Check if the splitKV value we extracted from the unmerge parameters
      // match the actual dimension size at position 3
      if (upperBounds[3] == potentialSplitKV &&
          isSupportedSplitKV(potentialSplitKV)) {
        splitKV = potentialSplitKV;
        LLVM_DEBUG(llvm::dbgs()
                   << "\tV: Found Unmerge{" << params[0] << "," << params[1]
                   << "," << params[2] << "," << params[3]
                   << "}, splitKV = " << splitKV << "\n");
        return splitKV;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\tV: No Unmerge pattern found\n");
  return splitKV;
}

// Helper function that validates tensor shape and unmerges batch dimension
// Returns: (newBatch, intermediate 4D tensor with splitKV exposed)
static FailureOr<std::pair<int64_t, Value>>
unmergeBackForSplitKV(Value tensor, int64_t splitKV, PatternRewriter &rewriter,
                      Location loc, StringRef tensorName,
                      ArrayRef<StringRef> dimNames) {
  auto tensorType = cast<ShapedType>(tensor.getType());
  ArrayRef<int64_t> shape = tensorType.getShape();

  if (shape.size() != 3) {
    LLVM_DEBUG(llvm::dbgs() << tensorName << ": Expected 3D tensor, got "
                            << shape.size() << "D\n");
    return failure();
  }

  int64_t currentBatch = shape[0];
  if (currentBatch % splitKV != 0) {
    LLVM_DEBUG(llvm::dbgs() << tensorName << ": Batch dimension "
                            << currentBatch << " not divisible by splitKV "
                            << splitKV << "\n");
    return failure();
  }

  int64_t newBatch = currentBatch / splitKV;
  
  // Unmerge batch into [newBatch, splitKV] and PassThrough other dims
  SmallVector<int64_t> unmergeParams = {newBatch, splitKV};
  SmallVector<int64_t> intermediateShape = {newBatch, splitKV, shape[1], shape[2]};
  SmallVector<int64_t> inputShape(shape.begin(), shape.end());

  TransformAttr unmergeOp = TransformAttr::get(
      rewriter.getContext(), rock::TransformType::Unmerge, unmergeParams,
      {"batch", "splitKV"}, {0, 1}, {"batch_merged"}, {0});

  TransformAttr passThroughOp = TransformAttr::get(
      rewriter.getContext(), rock::TransformType::PassThrough, {},
      dimNames, {2, 3}, dimNames, {1, 2});

  TransformMapAttr step1Map = TransformMapAttr::get(
      {unmergeOp, passThroughOp}, intermediateShape, inputShape);
  Value intermediate = rewriter.create<rock::TransformOp>(loc, tensor, step1Map);

  return std::make_pair(newBatch, intermediate);
}

// Add a transform on top of Q tensor to remove splitKV from batch dimension
// Q: [B*H*SplitKV, M, K] -> [B*H, M, K]
static FailureOr<Value> removeSplitKVFromQ(Value qTensor, int64_t splitKV,
                                            PatternRewriter &rewriter,
                                            Location loc) {
  // Step 1: Validate and unmerge batch
  auto maybeUnmerged = unmergeBackForSplitKV(qTensor, splitKV, rewriter, loc,
                                             "Q", {"M", "K"});
  if (failed(maybeUnmerged))
    return failure();

  auto [newBatch, intermediate] = maybeUnmerged.value();
  auto shape = cast<ShapedType>(qTensor.getType()).getShape();
  int64_t M = shape[1];
  int64_t K = shape[2];

  // Get the intermediate shape from the unmerged tensor
  ArrayRef<int64_t> intermediateShape = 
      cast<ShapedType>(intermediate.getType()).getShape();

  // Step 2: Slice the splitKV dimension to take only index [0:1], then merge
  // away the size-1 dim. This goes from [newBatch, splitKV, M, K] to
  // [newBatch, M, K]
  SmallVector<int64_t> outputShape = {newBatch, M, K};

  // Use affine_map to directly drop dimension 1 by fixing it to 0
  // Output [d0, d1, d2] maps to Input [d0, 0, d1, d2]
  SmallVector<TransformAttr> step2Ops;

  // PassThrough for batch
  step2Ops.push_back(TransformAttr::get(rewriter.getContext(),
                                        rock::TransformType::PassThrough, {},
                                        {"batch"}, {0}, {"batch"}, {0}));

  // ConstDim to fix splitKV to 0 (no upper dim, just fixes lower dim 1 to
  // constant 0)
  SmallVector<int64_t> constParams = {0, splitKV};
  step2Ops.push_back(TransformAttr::get(rewriter.getContext(),
                                        rock::TransformType::ConstDim,
                                        constParams, {}, {}, {"splitKV"}, {1}));

  // PassThrough for M and K (upper dims 1,2 map to lower dims 2,3)
  step2Ops.push_back(TransformAttr::get(
      rewriter.getContext(), rock::TransformType::PassThrough, {}, {"M", "K"},
      {1, 2}, {"M", "K"}, {2, 3}));

  TransformMapAttr step2Map =
      TransformMapAttr::get(step2Ops, outputShape, intermediateShape);

  Value result =
      rewriter.create<rock::TransformOp>(loc, intermediate, step2Map);
  return result;
}

// Add a transform on top of K tensor to remove splitKV from batch and restore N
// dimension K: [B*H*splitKV, K, N/splitKV] -> [B*H, K, N]
static FailureOr<Value> removeSplitKVFromK(Value kTensor, int64_t splitKV,
                                            PatternRewriter &rewriter,
                                            Location loc) {
  // Step 1: Validate and unmerge batch
  auto maybeUnmerged = unmergeBackForSplitKV(kTensor, splitKV, rewriter, loc,
                                             "K", {"K", "seqChunk"});
  if (failed(maybeUnmerged))
    return failure();

  auto [newBatch, intermediate] = maybeUnmerged.value();
  auto shape = cast<ShapedType>(kTensor.getType()).getShape();
  int64_t K = shape[1];
  int64_t seqChunk = shape[2]; // N / splitKV
  int64_t fullSeq = seqChunk * splitKV; // Restored N

  // Get the intermediate shape from the unmerged tensor
  ArrayRef<int64_t> intermediateShape = 
      cast<ShapedType>(intermediate.getType()).getShape();

  // Step 2: Use Merge to reconstruct the full sequence dimension
  // Output: [newBatch, K, fullSeq]
  // Merge combines splitKV and seqChunk: fullSeq = splitKV * seqChunk
  SmallVector<int64_t> outputShape = {newBatch, K, fullSeq};
  SmallVector<TransformAttr> step2Ops;

  // PassThrough for batch
  step2Ops.push_back(TransformAttr::get(rewriter.getContext(),
                                        rock::TransformType::PassThrough, {},
                                        {"batch"}, {0}, {"batch"}, {0}));

  // PassThrough for K (upper dim 1 -> lower dim 2)
  step2Ops.push_back(TransformAttr::get(rewriter.getContext(),
                                        rock::TransformType::PassThrough, {},
                                        {"K"}, {1}, {"K"}, {2}));

  // Merge splitKV (lower dim 1) and seqChunk (lower dim 3) into fullSeq (upper
  // dim 2)
  SmallVector<int64_t> mergeParams = {splitKV, seqChunk};
  step2Ops.push_back(TransformAttr::get(
      rewriter.getContext(), rock::TransformType::Merge, mergeParams,
      {"fullSeq"}, {2}, {"splitKV", "seqChunk"}, {1, 3}));

  TransformMapAttr step2Map =
      TransformMapAttr::get(step2Ops, outputShape, intermediateShape);

  Value result =
      rewriter.create<rock::TransformOp>(loc, intermediate, step2Map);
  return result;
}

// Add a transform on top of V tensor to remove splitKV from batch and restore N
// dimension V: [B*H*splitKV, N/splitKV, D] -> [B*H, N, D]
static FailureOr<Value> removeSplitKVFromV(Value vTensor, int64_t splitKV,
                                            PatternRewriter &rewriter,
                                            Location loc) {
  // Step 1: Validate and unmerge batch
  auto maybeUnmerged = unmergeBackForSplitKV(vTensor, splitKV, rewriter, loc,
                                             "V", {"seqChunk", "D"});
  if (failed(maybeUnmerged))
    return failure();

  auto [newBatch, intermediate] = maybeUnmerged.value();
  auto shape = cast<ShapedType>(vTensor.getType()).getShape();
  int64_t seqChunk = shape[1]; // N / splitKV
  int64_t D = shape[2];
  int64_t fullSeq = seqChunk * splitKV; // Restored N

  // Get the intermediate shape from the unmerged tensor
  ArrayRef<int64_t> intermediateShape = 
      cast<ShapedType>(intermediate.getType()).getShape();

  // Step 2: Use Merge to reconstruct the full sequence dimension
  // Output: [newBatch, fullSeq, D]
  // Merge combines splitKV and seqChunk: fullSeq = splitKV * seqChunk +
  SmallVector<int64_t> outputShape = {newBatch, fullSeq, D};
  SmallVector<TransformAttr> step2Ops;

  // PassThrough for batch
  step2Ops.push_back(TransformAttr::get(rewriter.getContext(),
                                        rock::TransformType::PassThrough, {},
                                        {"batch"}, {0}, {"batch"}, {0}));

  // Merge splitKV (lower dim 1) and seqChunk (lower dim 2) into fullSeq (upper
  // dim 1)
  SmallVector<int64_t> mergeParams = {splitKV, seqChunk};
  step2Ops.push_back(TransformAttr::get(
      rewriter.getContext(), rock::TransformType::Merge, mergeParams,
      {"fullSeq"}, {1}, {"splitKV", "seqChunk"}, {1, 2}));

  // PassThrough for D (upper dim 2 -> lower dim 3)
  step2Ops.push_back(TransformAttr::get(rewriter.getContext(),
                                        rock::TransformType::PassThrough, {},
                                        {"D"}, {2}, {"D"}, {3}));

  TransformMapAttr step2Map =
      TransformMapAttr::get(step2Ops, outputShape, intermediateShape);

  Value result =
      rewriter.create<rock::TransformOp>(loc, intermediate, step2Map);
  return result;
}

struct DetectFlashDecodingPattern : public OpRewritePattern<AttentionOp> {
  using OpRewritePattern<AttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AttentionOp op,
                                PatternRewriter &rewriter) const override {
    // Only process if splitKV is currently 1 (not yet detected/set)
    if (op.getSplitKV() != 1)
      return failure();

    // Flash decoding requires LSE for correctness
    std::optional<Value> lse = op.getLse();
    if (!lse.has_value())
      return failure();

    Value queries = op.getQueries();
    Value values = op.getValues();

    // Try to detect splitKV from Q and V input tensors
    // Note: K's splitKV transformation is optimized away during MIGraphX->TOSA
    // conversion, so we rely on Q and V for detection
    int64_t splitKVFromQ = detectSplitKVFromQ(queries);
    int64_t splitKVFromV = detectSplitKVFromV(values);

    // Both Q and V should agree on splitKV value
    if (splitKVFromQ != splitKVFromV) {
      LLVM_DEBUG(llvm::dbgs() << "\tMismatch: Q and V have different splitKV "
                              << "values\n");
      return failure();
    }

    // No flash decoding detected
    if (splitKVFromQ == 1) {
      LLVM_DEBUG(llvm::dbgs() << "\tNo flash decoding detected\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "\tFlash decoding detected: splitKV = "
                            << splitKVFromQ << "\n");

    // Add transforms to remove splitKV from batch dimension of inputs
    Value keys = op.getKeys();

    auto maybeNewQueries =
        removeSplitKVFromQ(queries, splitKVFromQ, rewriter, op.getLoc());
    auto maybeNewKeys =
        removeSplitKVFromK(keys, splitKVFromQ, rewriter, op.getLoc());
    auto maybeNewValues =
        removeSplitKVFromV(values, splitKVFromQ, rewriter, op.getLoc());

    if (failed(maybeNewQueries) || failed(maybeNewKeys) ||
        failed(maybeNewValues)) {
      LLVM_DEBUG(llvm::dbgs() << "\tFailed to create new transforms\n");
      return failure();
    }

    Value newQueries = maybeNewQueries.value();
    Value newKeys = maybeNewKeys.value();
    Value newValues = maybeNewValues.value();

    // Extract result types - AttentionOp has optional result andlseOut. lseOut
    // was already verified to be non-null.
    Type resultType = op.getResult() ? op.getResult().getType() : Type();
    Type lseOutType = op.getLseOut().getType();

    auto newOp = rock::AttentionOp::create(
        rewriter, op->getLoc(), resultType, lseOutType, newQueries, newKeys,
        newValues, op.getPreSoftmaxElemWiseInputs(), op.getCurrentSeqLen(),
        op.getOut(), op.getLse(), op.getNumHeadsQAttr(), op.getNumHeadsKVAttr(),
        op.getQTransposedAttr(), op.getKTransposedAttr(),
        op.getVTransposedAttr(), op.getOTransposedAttr(), op.getCausalAttr(),
        rewriter.getI32IntegerAttr(splitKVFromQ), op.getFeaturesAttr(),
        op.getStoreMethodAttr(), op.getSoftmaxTypeAttr(), op.getParams0Attr(),
        op.getParams1Attr(), op.getFirstGemmIndicesAttr());

    // Copy the preSoftmax elementwise region if it exists
    if (!op.getPreSoftmaxBody().empty()) {
      rewriter.inlineRegionBefore(op.getPreSoftmaxBody(),
                                  newOp.getPreSoftmaxBody(),
                                  newOp.getPreSoftmaxBody().begin());
    }

    // Copy perf_config attribute if present
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newOp->setAttr("perf_config", attr);

    // Replace the old op with the new one
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct RockDetectFlashDecodingPass
    : public rock::impl::RockDetectFlashDecodingPassBase<
          RockDetectFlashDecodingPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<DetectFlashDecodingPattern>(ctx);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace
