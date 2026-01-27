//===- PagedAttentionGenerator.h - Paged attention transform generation --===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for generating paged attention IR, specifically
// the transform chains needed to convert rock.deref output shape to attention's
// expected K/V shapes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_PAGEDATTENTIONGENERATOR_H_
#define MLIR_DIALECT_ROCK_PAGEDATTENTIONGENERATOR_H_

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace rock {

// Configuration for paged attention generation.
struct PagedAttentionConfig {
  int64_t batch;        // Batch size (groupSize in attention terminology)
  int64_t numPages;     // Number of pages in the page table
  int64_t pageSize;     // Number of elements per page
  int64_t numHeadsKV;   // Number of KV heads
  int64_t numHeadsQ;    // Number of Q heads (for GQA, handled by attention op)
  int64_t seqLenK;      // Sequence length K (derived or set)
  int64_t headDimQK;    // Head dimension for Q and K
  int64_t headDimV;     // Head dimension for V (may differ from headDimQK)
  bool transposeK;      // K layout: true=[G,seqK,headDim], false=[G,headDim,seqK]
  bool transposeV;      // V layout: true=[G,headDimV,seqK], false=[G,seqK,headDimV]
  Type elemType;        // Element type (e.g., f16)

  // Validate that the configuration is consistent.
  LogicalResult validate() const;

  // Compute derived values and validate.
  LogicalResult computeAndValidate();
};

class PagedAttentionGenerator {
public:
  explicit PagedAttentionGenerator(const PagedAttentionConfig &config)
      : config(config) {}

  // Generate transforms from deref output to K matrix shape.
  Value createDerefToKTransforms(OpBuilder &builder, Location loc,
                                 Value derefOutput) const;

  // Generate transforms from deref output to V matrix shape.
  Value createDerefToVTransforms(OpBuilder &builder, Location loc,
                                 Value derefOutput) const;

  // Get the expected deref output type given page table input.
  MemRefType getDerefOutputType(OpBuilder &builder) const;

  // Get the expected page table type.
  MemRefType getPageTableType(OpBuilder &builder) const;

  // Get the K matrix shape after transforms (for attention op).
  SmallVector<int64_t, 3> getKShape() const;

  // Get the V matrix shape after transforms (for attention op).
  SmallVector<int64_t, 3> getVShape() const;

private:
  PagedAttentionConfig config;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PAGEDATTENTIONGENERATOR_H_
