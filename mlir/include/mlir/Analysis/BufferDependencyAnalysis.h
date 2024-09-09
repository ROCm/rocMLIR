//===- BufferDependencyAnalysis.h - a set of useful generic analyses
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ROCK_ANALYSIS_LOWERINGUTILS_H
#define ROCK_ANALYSIS_LOWERINGUTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
// Given an operation (e.g., func::FuncOp), the analysis finds readers
// and writers associated with memory created by each `allocOp` operation.
//
// Users can use `getReaders` and/or `getWriters` methods to look-up
// readers/writers associated with a specific `allocOp`.
//
// Note, the analysis is intended to be performed after bufferization.
struct BufferDependencyAnalysis {
  BufferDependencyAnalysis(Operation *op);

  /// Get all the OpOperands that read from the given buffer, potentially
  /// through any number of views.
  std::optional<llvm::SmallVector<OpOperand *>>
  getReaders(memref::AllocOp allocOp) const;
  /// Get all the OpOperands that write to the given buffer, potentially through
  /// any number of views.
  std::optional<llvm::SmallVector<OpOperand *>>
  getWriters(memref::AllocOp allocOp) const;

  /// Return, if one is known, the buffer that this OpOperand reads from.
  /// The argument must be operand of the memory-writing operation, not any
  /// intermediate view.
  std::optional<memref::AllocOp> getReadBuffer(OpOperand *use) const;
  /// Return, if one is known, the buffer that this OpOperand writes to.
  /// The argument must be operand of the memory-writing operation, not any
  /// intermediate view.
  std::optional<memref::AllocOp> getWrittenBuffer(OpOperand *use) const;

  const llvm::DenseMap<memref::AllocOp, llvm::SmallVector<OpOperand *>> &
  getReadersTable() const {
    return readersTable;
  }
  const llvm::DenseMap<memref::AllocOp, llvm::SmallVector<OpOperand *>> &
  getWritersTable() const {
    return writersTable;
  }
  const llvm::DenseMap<OpOperand *, memref::AllocOp> &
  getReaderToBufferTable() const {
    return readerToBufferTable;
  }
  const llvm::DenseMap<OpOperand *, memref::AllocOp> &
  getWriterToBufferTable() const {
    return writerToBufferTable;
  }

  // Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return op; }

  // Compute the analysis on `op`, overwriting existing analysis if necessary.
  // This is to be used after updating the graph.
  void analyze(memref::AllocOp allocOp);

private:
  Operation *op;
  llvm::DenseMap<memref::AllocOp, llvm::SmallVector<OpOperand *>> readersTable;
  llvm::DenseMap<memref::AllocOp, llvm::SmallVector<OpOperand *>> writersTable;
  llvm::DenseMap<OpOperand *, memref::AllocOp> readerToBufferTable;
  llvm::DenseMap<OpOperand *, memref::AllocOp> writerToBufferTable;
};
} // namespace mlir

#endif // ROCK_ANALYSIS_LOWERINGUTILS_H
