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

struct BufferDependencyAnalysis {
  BufferDependencyAnalysis(Operation *op);

  static std::optional<memref::AllocOp> getAllocation(Value value);
  static llvm::SmallVector<Operation *> getReaders(memref::AllocOp allocOp);
  static llvm::SmallVector<Operation *> getWriters(memref::AllocOp allocOp);

  struct Pair {
    llvm::SmallVector<Operation *> readers;
    llvm::SmallVector<Operation *> writers;
  };
  static Pair getReadersAndWriters(memref::AllocOp allocOps);

  LogicalResult run();
  llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>>
  getReadersTable() {
    return readersTable;
  }
  llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>>
  getWritersTable() {
    return writersTable;
  }
  func::FuncOp getFuncton() const { return func; }

private:
  // The function this analysis was constructed from.
  func::FuncOp func;

  llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> readersTable;
  llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> writersTable;
};

} // namespace mlir

#endif // ROCK_ANALYSIS_LOWERINGUTILS_H