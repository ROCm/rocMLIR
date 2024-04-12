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

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

  std::optional<llvm::SmallVector<Operation *>>
  getReaders(memref::AllocOp allocOp);
  std::optional<llvm::SmallVector<Operation *>>
  getWriters(memref::AllocOp allocOp);

  const llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> &
  getReadersTable() {
    return readersTable;
  }
  const llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> &
  getWritersTable() {
    return writersTable;
  }

  // Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return op; }

private:
  Operation *op;
  llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> readersTable;
  llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> writersTable;
};
} // namespace mlir

#endif // ROCK_ANALYSIS_LOWERINGUTILS_H
