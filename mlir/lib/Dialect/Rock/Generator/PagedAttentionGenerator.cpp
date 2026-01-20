//===- PagedAttentionGenerator.cpp - Paged attention transform generation ===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Generator/PagedAttentionGenerator.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/IR/Rock.h"

using namespace mlir;
using namespace mlir::rock;

LogicalResult PagedAttentionConfig::validate() const {
  int64_t totalPageElements = numPages * pageSize;
  int64_t expectedElements = numHeadsKV * seqLenK * headDimQK;
  
  if (totalPageElements != expectedElements) {
    return failure();
  }
  
  return success();
}

LogicalResult PagedAttentionConfig::computeAndValidate() {
  // Total elements in paged cache
  int64_t totalElements = numPages * pageSize;
  
  // Compute seqLenK: totalElements = numHeadsKV * seqLenK * headDimQK
  // seqLenK = totalElements / (numHeadsKV * headDimQK)
  if (numHeadsKV <= 0 || headDimQK <= 0) {
    return failure();
  }
  
  int64_t denominator = numHeadsKV * headDimQK;
  if (totalElements % denominator != 0) {
    return failure();
  }
  
  seqLenK = totalElements / denominator;
  return validate();
}

MemRefType PagedAttentionGenerator::getDerefOutputType(OpBuilder &builder) const {
  return MemRefType::get({config.batch, config.numPages, config.pageSize},
                         config.elemType);
}

MemRefType PagedAttentionGenerator::getPageTableType(OpBuilder &builder) const {
  return MemRefType::get({config.batch, config.numPages, 1},
                         builder.getI64Type());
}

SmallVector<int64_t, 3> PagedAttentionGenerator::getKShape() const {
  // K shape matches regular attention: [groupSize * numHeadsKV, ...]
  int64_t G = config.batch * config.numHeadsKV;
  if (config.transposeK) {
    // Not transposed: [G, seqK, headDimQK]
    return {G, config.seqLenK, config.headDimQK};
  } else {
    // Transposed: [G, headDimQK, seqK]
    return {G, config.headDimQK, config.seqLenK};
  }
}

SmallVector<int64_t, 3> PagedAttentionGenerator::getVShape() const {
  // V shape matches regular attention: [groupSize * numHeadsKV, ...]
  int64_t G = config.batch * config.numHeadsKV;
  if (config.transposeV) {
    // Transposed: [G, headDimV, seqK]
    return {G, config.headDimV, config.seqLenK};
  } else {
    // Not transposed: [G, seqK, headDimV]
    return {G, config.seqLenK, config.headDimV};
  }
}

Value PagedAttentionGenerator::createDerefToKTransforms(
    OpBuilder &builder, Location loc, Value derefOutput) const {
  
  // Input: [batch, numPages, pageSize]
  // Target (transposeK=true): [G_kv, seqK, headDimQK] where G_kv = batch * numHeadsKV
  // Target (transposeK=false): [G_kv, headDimQK, seqK]
  
  SmallVector<StringRef> startNames = {"batch", "numPages", "pageSize"};
  ArrayRef<int64_t> inpShape = cast<MemRefType>(derefOutput.getType()).getShape();
  
  // Step 1: Merge [batch, numPages, pageSize] -> [batch, total]
  BottomUpTMBuilder mergeB(builder, startNames, inpShape);
  mergeB.passThrough({"batch"}, {0}, {"batch"});
  mergeB.merge("total", 1, {"numPages", "pageSize"});
  auto mergeAttr = mergeB.get();
  Value merged = TransformOp::create(builder, loc, derefOutput, mergeAttr);
  
  // Step 2: Unmerge [batch, total] -> [batch, numHeadsKV, seqK, headDimQK]
  auto unmergeB = BottomUpTMBuilder::above(mergeB, mergeAttr);
  unmergeB.passThrough({"batch"}, {0}, {"batch"});
  unmergeB.unmerge({"numHeadsKV", "seqK", "headDimQK"}, {1, 2, 3}, "total",
                   {config.numHeadsKV, config.seqLenK, config.headDimQK});
  auto unmergeAttr = unmergeB.get();
  Value unmerged = TransformOp::create(builder, loc, merged, unmergeAttr);
  
  // Step 3: Merge [batch, numHeadsKV] -> [G_kv] and handle transpose
  // Input after unmerge: [batch, numHeadsKV, seqK, headDimQK]
  // inNames determine which input dim maps to each output position
  auto finalB = BottomUpTMBuilder::above(unmergeB, unmergeAttr);
  finalB.merge("G", 0, {"batch", "numHeadsKV"});
  
  if (config.transposeK) {
    // transposeK=true means NOT transposed layout: [G, seqK, headDimQK]
    // Map: inNames[0]="seqK" -> outDims[0]=1, inNames[1]="headDimQK" -> outDims[1]=2
    finalB.passThrough({"seqK", "headDimQK"}, {1, 2}, {"seqK", "headDimQK"});
  } else {
    // transposeK=false means transposed layout: [G, headDimQK, seqK]
    // Map: inNames[0]="headDimQK" -> outDims[0]=1, inNames[1]="seqK" -> outDims[1]=2
    finalB.passThrough({"headDimQK", "seqK"}, {1, 2}, {"headDimQK", "seqK"});
  }
  auto finalAttr = finalB.get();
  return TransformOp::create(builder, loc, unmerged, finalAttr);
}

Value PagedAttentionGenerator::createDerefToVTransforms(
    OpBuilder &builder, Location loc, Value derefOutput) const {
  
  // Input: [batch, numPages, pageSize]
  // Target (transposeV=true): [G_kv, headDimV, seqK]
  // Target (transposeV=false): [G_kv, seqK, headDimV]
  
  // Note: For V, we use headDimV which may differ from headDimQK
  // The pageSize must accommodate: numHeadsKV * seqK * headDimV
  // For simplicity, we assume headDimV == headDimQK in validation
  // If they differ, the caller must ensure pageSize is correct for V
  
  SmallVector<StringRef> startNames = {"batch", "numPages", "pageSize"};
  ArrayRef<int64_t> inpShape = cast<MemRefType>(derefOutput.getType()).getShape();
  
  // Step 1: Merge [batch, numPages, pageSize] -> [batch, total]
  BottomUpTMBuilder mergeB(builder, startNames, inpShape);
  mergeB.passThrough({"batch"}, {0}, {"batch"});
  mergeB.merge("total", 1, {"numPages", "pageSize"});
  auto mergeAttr = mergeB.get();
  Value merged = TransformOp::create(builder, loc, derefOutput, mergeAttr);
  
  // Step 2: Unmerge [batch, total] -> [batch, numHeadsKV, seqK, headDimV]
  auto unmergeB = BottomUpTMBuilder::above(mergeB, mergeAttr);
  unmergeB.passThrough({"batch"}, {0}, {"batch"});
  unmergeB.unmerge({"numHeadsKV", "seqK", "headDimV"}, {1, 2, 3}, "total",
                   {config.numHeadsKV, config.seqLenK, config.headDimV});
  auto unmergeAttr = unmergeB.get();
  Value unmerged = TransformOp::create(builder, loc, merged, unmergeAttr);
  
  // Step 3: Merge [batch, numHeadsKV] -> [G_kv] and handle transpose
  // Input after unmerge: [batch, numHeadsKV, seqK, headDimV]
  // inNames determine which input dim maps to each output position
  auto finalB = BottomUpTMBuilder::above(unmergeB, unmergeAttr);
  finalB.merge("G", 0, {"batch", "numHeadsKV"});
  
  if (config.transposeV) {
    // transposeV=true means transposed layout: [G, headDimV, seqK]
    // Map: inNames[0]="headDimV" -> outDims[0]=1, inNames[1]="seqK" -> outDims[1]=2
    finalB.passThrough({"headDimV", "seqK"}, {1, 2}, {"headDimV", "seqK"});
  } else {
    // transposeV=false means NOT transposed layout: [G, seqK, headDimV]
    // Map: inNames[0]="seqK" -> outDims[0]=1, inNames[1]="headDimV" -> outDims[1]=2
    finalB.passThrough({"seqK", "headDimV"}, {1, 2}, {"seqK", "headDimV"});
  }
  auto finalAttr = finalB.get();
  return TransformOp::create(builder, loc, unmerged, finalAttr);
}
