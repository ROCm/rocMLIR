//===- DetectFlashDecoding.cpp - Detect and fix flash decoding splitKV --===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
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
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

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
// Supported values are powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, etc.
static bool isSupportedSplitKV(int64_t splitKV) {
  return splitKV > 0 && llvm::isPowerOf2_64(splitKV);
}

// Detect splitKV from Q tensor by finding the Broadcast operation
// Q pattern: [B, H, 1, M, K] --Broadcast--> [B, H, SplitKV, M, K]
// Returns (splitKV, dimensionality), where dimensionality is 4 or 5
// Returns (1, 0) if not found
// Note, this detection logic only works if we find the broadcast operation and
// it is has not been optimized away by one of the earlier passes.
static std::pair<int64_t, int64_t> detectSplitKVFromQ(Value qTensor) {
  SmallVector<TransformMapAttr> transforms;
  rock::untransform(qTensor, transforms);

  if (transforms.empty())
    return {1, 0};

  LLVM_DEBUG(llvm::dbgs() << "Analyzing Q tensor for splitKV:\n");

  // Look for a Broadcast operation at dimension 1 (4D) or 2 (5D)
  for (TransformMapAttr transformMap : transforms) {
    ArrayRef<int64_t> upperBounds = transformMap.getUpperBounds();

    for (rock::TransformAttr op : transformMap.getOps()) {
      // Look for Broadcast operation
      if (op.getType() != rock::TransformType::Broadcast)
        continue;

      ArrayRef<uint32_t> upperDims = op.getUpperDims();
      ArrayRef<int64_t> params = op.getParams();

      if (upperDims.empty() || params.empty() || params[0] != 1)
        continue;

      uint32_t broadcastDim = upperDims[0];

      // Helper lambda to check broadcast pattern and return splitKV if found
      auto checkBroadcastPattern = [&](uint32_t expectedDim,
                                       unsigned expectedDimensionality)
          -> std::optional<std::pair<int64_t, int64_t>> {
        if (broadcastDim == expectedDim &&
            upperBounds.size() == expectedDimensionality) {
          auto potentialSplitKV = upperBounds[expectedDim];
          if (isSupportedSplitKV(potentialSplitKV)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "\tQ: Found " << expectedDimensionality
                       << "D Broadcast at dim " << expectedDim << ", "
                       << "splitKV = " << potentialSplitKV << "\n");
            return std::make_pair(potentialSplitKV, expectedDimensionality);
          } else {
            LLVM_DEBUG(llvm::dbgs()
                       << "\tQ: Found " << expectedDimensionality
                       << "D Broadcast at dim " << expectedDim << ", but "
                       << "splitKV = " << potentialSplitKV
                       << " not supported (must be power of 2)\n");
          }
        }
        return std::nullopt;
      };

      // Check for 5D pattern: broadcast at dimension 2, shape
      // [B, H, splitKV, M, K]
      if (auto result = checkBroadcastPattern(2, 5))
        return *result;

      // Check for 4D pattern: broadcast at dimension 1, shape
      // [BH, splitKV, M, K]
      if (auto result = checkBroadcastPattern(1, 4))
        return *result;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\tQ: No Broadcast pattern found\n");
  return {1, 0};
}

// Detect splitKV from K or V tensor by finding the Unmerge with splitKV
// dimension. This works for both K and V since they have similar patterns.
// Returns (splitKV, dimensionality), where dimensionality is 4 or 5
// Returns (1, 0) if not found. E.g.,
// - 5D: flat --Unmerge--> [B, H, D, SplitKV, N/SplitKV] (splitKV at position 3)
// - 4D: flat --Unmerge--> [BH, D, SplitKV, N/SplitKV] (splitKV at position 2)
// - 4D: flat --Unmerge--> [B, SplitKV, D, N/SplitKV] (splitKV at position 1)
static std::pair<int64_t, int64_t> detectSplitKVFromKV(Value tensor,
                                                       StringRef tensorName) {
  SmallVector<TransformMapAttr> transforms;
  rock::untransform(tensor, transforms);

  if (transforms.empty())
    return {1, 0};

  LLVM_DEBUG(llvm::dbgs() << "Analyzing " << tensorName
                          << " tensor for splitKV:\n");

  // Look for an Unmerge operation that creates a 4D or 5D shape with splitKV
  for (TransformMapAttr transformMap : transforms) {
    ArrayRef<int64_t> upperBounds = transformMap.getUpperBounds();

    for (rock::TransformAttr op : transformMap.getOps()) {
      if (op.getType() != rock::TransformType::Unmerge)
        continue;

      ArrayRef<int64_t> params = op.getParams();

      // Helper lambda to check unmerge pattern and return splitKV if found
      auto checkUnmergePattern = [&](unsigned expectedDimensionality,
                                     unsigned splitKVPosition,
                                     int64_t potentialSplitKV)
          -> std::optional<std::pair<int64_t, int64_t>> {
        if (upperBounds.size() == expectedDimensionality &&
            upperBounds[splitKVPosition] == potentialSplitKV &&
            isSupportedSplitKV(potentialSplitKV)) {
          LLVM_DEBUG(llvm::dbgs() << "\t" << tensorName << ": Found "
                                  << expectedDimensionality << "D Unmerge{";
                     llvm::interleaveComma(params, llvm::dbgs());
                     llvm::dbgs()
                     << "}, splitKV = " << potentialSplitKV << " at position "
                     << splitKVPosition << "\n");
          return std::make_pair(potentialSplitKV, expectedDimensionality);
        }
        return std::nullopt;
      };

      // 5D case with AddDim: Unmerge has 4 params, AddDim adds the 5th.
      if (params.size() == 4 && upperBounds.size() == 5) {
        if (auto result = checkUnmergePattern(5, 3, params[2]))
          return *result;
        if (auto result = checkUnmergePattern(5, 4, params[3]))
          return *result;
      }

      // 5D case without AddDim: Unmerge has 5 params.
      if (params.size() == 5 && upperBounds.size() == 5) {
        if (auto result = checkUnmergePattern(5, 2, params[2]))
          return *result;
        if (auto result = checkUnmergePattern(5, 3, params[3]))
          return *result;
      }

      // 4D case with AddDim: Unmerge has 3 params, AddDim adds the 4th.
      if (params.size() == 3 && upperBounds.size() == 4) {
        if (auto result = checkUnmergePattern(4, 2, params[1]))
          return *result;
        if (auto result = checkUnmergePattern(4, 3, params[2]))
          return *result;
      }

      // 4D case without AddDim: Unmerge has 4 params.
      if (params.size() == 4 && upperBounds.size() == 4) {
        if (auto result = checkUnmergePattern(4, 1, params[1]))
          return *result;
        if (auto result = checkUnmergePattern(4, 2, params[2]))
          return *result;
      }

      // 3D case: Unmerge has 3 params, splitKV merged into batch.
      if (params.size() == 3 && upperBounds.size() == 3) {
        if (auto result = checkUnmergePattern(3, 1, params[1]))
          return *result;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\t" << tensorName
                          << ": No Unmerge pattern found\n");
  return {1, 0};
}

// Helper function that validates tensor shape and unmerges batch dimension
// Returns: (newBatch, intermediate 4D tensor with splitKV exposed)
static FailureOr<std::pair<int64_t, Value>>
unmergeBackForSplitKV(PatternRewriter &rewriter, Location loc, Value tensor,
                      int64_t splitKV, StringRef tensorName,
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
    LLVM_DEBUG(llvm::dbgs()
               << tensorName << ": Batch dimension " << currentBatch
               << " not divisible by splitKV " << splitKV << "\n");
    return failure();
  }

  int64_t newBatch = currentBatch / splitKV;

  // Start from [batch_merged, dim0, dim1] and unmerge to
  // [batch, splitKV, dim0, dim1]
  SmallVector<StringRef> lowerNames = {"batch_merged"};
  lowerNames.append(dimNames.begin(), dimNames.end());

  rock::BottomUpTMBuilder builder(rewriter, lowerNames, shape, loc);
  builder.unmerge({"batch", "splitKV"}, {0, 1}, "batch_merged",
                  {newBatch, splitKV});
  builder.passThrough(dimNames, {2, 3}, dimNames);

  TransformMapAttr transformMap = builder.get();
  Value intermediate =
      rock::TransformOp::create(rewriter, loc, tensor, transformMap);

  return std::make_pair(newBatch, intermediate);
}

// Common helper to slice away splitKV dimension by fixing it to index 0.
// Takes a tensor with splitKV in batch dimension and slices it out.
// Input: [batch*splitKV, dim0, dim1, ...] -> Output: [batch, dim0, dim1, ...]
// For 1D tensors: [batch*splitKV] -> [batch]
static FailureOr<Value>
sliceSplitKVFromBatch(PatternRewriter &rewriter, Location loc, Value tensor,
                      int64_t splitKV, ArrayRef<StringRef> trailingDimNames) {
  auto tensorType = cast<ShapedType>(tensor.getType());
  ArrayRef<int64_t> shape = tensorType.getShape();

  if (shape.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot process 0D tensor\n");
    return failure();
  }

  int64_t currentBatch = shape[0];
  if (currentBatch % splitKV != 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "Batch dimension " << currentBatch
               << " not divisible by splitKV " << splitKV << "\n");
    return failure();
  }

  int64_t newBatch = currentBatch / splitKV;

  // Step 1: Unmerge batch to [newBatch, splitKV, ...]
  SmallVector<StringRef> lowerNames = {"batch_merged"};
  lowerNames.append(trailingDimNames.begin(), trailingDimNames.end());

  rock::BottomUpTMBuilder builder(rewriter, lowerNames, shape, loc);
  builder.unmerge({"batch", "splitKV"}, {0, 1}, "batch_merged",
                  {newBatch, splitKV});

  // PassThrough for trailing dimensions (starting at index 2 in output)
  SmallVector<uint32_t> trailingUpperDims, trailingLowerDims;
  for (size_t i = 0; i < trailingDimNames.size(); ++i) {
    trailingUpperDims.push_back(2 + i);
    trailingLowerDims.push_back(1 + i);
  }
  if (!trailingDimNames.empty()) {
    builder.passThrough(trailingDimNames, trailingUpperDims, trailingDimNames);
  }

  TransformMapAttr unmergeMap = builder.get();
  Value intermediate =
      rock::TransformOp::create(rewriter, loc, tensor, unmergeMap);

  ArrayRef<int64_t> intermediateShape =
      cast<ShapedType>(intermediate.getType()).getShape();

  // Step 2: Slice splitKV to index 0 using ConstDim
  SmallVector<int64_t> outputShape = {newBatch};
  for (size_t i = 1; i < shape.size(); ++i)
    outputShape.push_back(shape[i]);

  SmallVector<TransformAttr> sliceOps;

  // PassThrough for batch
  sliceOps.push_back(TransformAttr::get(rewriter.getContext(),
                                        rock::TransformType::PassThrough, {},
                                        {"batch"}, {0}, {"batch"}, {0}));

  // ConstDim to fix splitKV to 0
  SmallVector<int64_t> constParams = {0, splitKV};
  sliceOps.push_back(TransformAttr::get(rewriter.getContext(),
                                        rock::TransformType::ConstDim,
                                        constParams, {}, {}, {"splitKV"}, {1}));

  // PassThrough for trailing dimensions
  if (!trailingDimNames.empty()) {
    SmallVector<uint32_t> upperDims, lowerDims;
    for (size_t i = 0; i < trailingDimNames.size(); ++i) {
      upperDims.push_back(1 + i);
      lowerDims.push_back(2 + i);
    }
    sliceOps.push_back(TransformAttr::get(
        rewriter.getContext(), rock::TransformType::PassThrough, {},
        trailingDimNames, upperDims, trailingDimNames, lowerDims));
  }

  TransformMapAttr sliceMap =
      TransformMapAttr::get(sliceOps, outputShape, intermediateShape);

  return rock::TransformOp::create(rewriter, loc, intermediate, sliceMap)
      .getResult();
}

// Helper function to remove splitKV from K or V tensors
// K: [B*H*splitKV, K, seq_k/splitKV] -> [B*H, K, seq_k]
// V: [B*H*splitKV, seq_k/splitKV, D] -> [B*H, seq_k, D]
static FailureOr<Value>
removeSplitKVWithMerge(PatternRewriter &rewriter, Location loc, Value tensor,
                       int64_t splitKV, StringRef tensorName,
                       StringRef featureDimName, bool featureFirst) {
  // Step 1: Validate and unmerge batch
  SmallVector<StringRef> dimNames =
      featureFirst ? SmallVector<StringRef>{featureDimName, "seq_k_chunk"}
                   : SmallVector<StringRef>{"seq_k_chunk", featureDimName};

  auto maybeUnmerged = unmergeBackForSplitKV(rewriter, loc, tensor, splitKV,
                                             tensorName, dimNames);
  if (failed(maybeUnmerged))
    return failure();

  auto [newBatch, intermediate] = maybeUnmerged.value();
  auto shape = cast<ShapedType>(tensor.getType()).getShape();
  int64_t featureDimSize = featureFirst ? shape[1] : shape[2];
  int64_t seqKChunk = featureFirst ? shape[2] : shape[1];
  int64_t seqK = seqKChunk * splitKV; // Restored seq_k

  // Get the intermediate shape from the unmerged tensor
  ArrayRef<int64_t> intermediateShape =
      cast<ShapedType>(intermediate.getType()).getShape();

  // Step 2: Use Merge to reconstruct the full sequence dimension
  SmallVector<int64_t> outputShape;
  SmallVector<StringRef> lowerDimNames;
  unsigned mergeDim;
  unsigned passThroughDim;

  if (featureFirst) {
    // K case: [batch, K, seq_k]
    outputShape = {newBatch, featureDimSize, seqK};
    lowerDimNames = {"batch", "splitKV", featureDimName, "seq_k_chunk"};
    mergeDim = 2;
    passThroughDim = 1;
  } else {
    outputShape = {newBatch, seqK, featureDimSize};
    lowerDimNames = {"batch", "splitKV", "seq_k_chunk", featureDimName};
    mergeDim = 1;
    passThroughDim = 2;
  }

  rock::BottomUpTMBuilder builder(rewriter, lowerDimNames, intermediateShape,
                                  loc);
  builder.passThrough({"batch"}, {0}, {"batch"});
  builder.merge("seq_k", mergeDim, {"splitKV", "seq_k_chunk"});
  builder.passThrough({featureDimName}, {passThroughDim}, {featureDimName});

  TransformMapAttr transformMap = builder.get();
  Value result =
      rock::TransformOp::create(rewriter, loc, intermediate, transformMap);

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

    // Flash decoding detection only supports non-transposed inputs
    if (op.getQTransposed() || op.getKTransposed() || op.getVTransposed()) {
      return failure();
    }

    Value queries = op.getQueries();
    Value keys = op.getKeys();
    Value values = op.getValues();

    // Try to detect splitKV from Q, K, and V input tensors
    auto [splitKVFromQ, qDim] = detectSplitKVFromQ(queries);
    auto [splitKVFromK, kDim] = detectSplitKVFromKV(keys, "K");
    auto [splitKVFromV, vDim] = detectSplitKVFromKV(values, "V");

    LLVM_DEBUG(llvm::dbgs()
               << "splitKV detection results:\n"
               << "  Q: splitKV=" << splitKVFromQ << ", dim=" << qDim << "\n"
               << "  K: splitKV=" << splitKVFromK << ", dim=" << kDim << "\n"
               << "  V: splitKV=" << splitKVFromV << ", dim=" << vDim << "\n");

    // No flash decoding detected
    if (splitKVFromQ == 1 || qDim == 0) {
      LLVM_DEBUG(llvm::dbgs() << "No flash decoding detected\n");
      return failure();
    }

    // Require one other tensor to agree on the splitKV value from Q for
    // reliable detection. K and V are occasionally optimized away by upstream
    // canonicalization passes which is why we don't require all three.
    int agreements =
        (splitKVFromK == splitKVFromQ) + (splitKVFromV == splitKVFromQ);

    if (agreements < 1) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Insufficient agreement on splitKV, no flash decoding detected\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Flash decoding detected: splitKV = " << splitKVFromQ
               << ", dimensionality = " << qDim << "D\n");

    // Add transforms to remove splitKV from batch dimension of inputs
    auto maybeNewQueries = sliceSplitKVFromBatch(rewriter, op.getLoc(), queries,
                                                 splitKVFromQ, {"M", "K"});
    auto maybeNewKeys =
        removeSplitKVWithMerge(rewriter, op.getLoc(), keys, splitKVFromQ, "K",
                               "K", /*featureFirst=*/true);
    auto maybeNewValues =
        removeSplitKVWithMerge(rewriter, op.getLoc(), values, splitKVFromQ, "V",
                               "D", /*featureFirst=*/false);

    if (failed(maybeNewQueries) || failed(maybeNewKeys) ||
        failed(maybeNewValues)) {
      op.emitError("Failed to create new transforms");
      return failure();
    }

    Value newQueries = maybeNewQueries.value();
    Value newKeys = maybeNewKeys.value();
    Value newValues = maybeNewValues.value();

    Type resultType = op.getResult().getType();
    Type lseOutType = op.getLseOut().getType();

    // Lambda to transform optional batch tensors (E.g., currentSeqLen,
    // prefixOffset).
    auto splitKVVal = splitKVFromQ;
    auto transformOptionalTensor = [&](Value tensor) -> FailureOr<Value> {
      if (!tensor)
        return Value(nullptr);
      auto maybeNew = sliceSplitKVFromBatch(rewriter, op.getLoc(), tensor,
                                            splitKVVal, {});
      if (failed(maybeNew)) {
        op.emitOpError("Failed to transform ") << tensor;
        return failure();
      }
      return maybeNew.value();
    };

    auto maybeNewCurrentSeqLen =
        transformOptionalTensor(op.getCurrentSeqLen());
    if (failed(maybeNewCurrentSeqLen))
      return failure();
    Value newCurrentSeqLen = maybeNewCurrentSeqLen.value();

    auto maybeNewPrefixOffset = transformOptionalTensor(op.getPrefixOffset());
    if (failed(maybeNewPrefixOffset))
      return failure();
    Value newPrefixOffset = maybeNewPrefixOffset.value();

    auto newOp = rock::AttentionOp::create(
        rewriter, op->getLoc(), resultType, lseOutType, newQueries, newKeys,
        newValues, op.getPreSoftmaxElemWiseInputs(), newCurrentSeqLen,
        newPrefixOffset, op.getOut(), op.getLse(), op.getNumHeadsQAttr(),
        op.getNumHeadsKVAttr(), op.getQTransposedAttr(),
        op.getKTransposedAttr(), op.getVTransposedAttr(),
        op.getOTransposedAttr(), op.getCausalAttr(),
        rewriter.getI32IntegerAttr(splitKVFromQ), op.getFeaturesAttr(),
        op.getStoreMethodAttr(), op.getSoftmaxTypeAttr(), op.getParams0Attr(),
        op.getParams1Attr(), op.getFirstGemmIndicesAttr(),
        /*preSoftmaxHasSplitKVTransforms=*/rewriter.getBoolAttr(true));

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

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace
