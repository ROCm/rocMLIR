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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

struct BufferDependencyAnalysis {
  static std::optional<memref::AllocOp> getAllocation(Value value);
  static llvm::SmallVector<Operation *> getReaders(memref::AllocOp allocOp);
  static llvm::SmallVector<Operation *> getWriters(memref::AllocOp allocOp);

  struct LocalResult {
    llvm::SmallVector<Operation *> readers;
    llvm::SmallVector<Operation *> writers;
  };
  static LocalResult findReadersAndWriters(memref::AllocOp allocOps);

  struct AnalysisResults {
    llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> readers;
    llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> writers;
  };
  static std::optional<AnalysisResults> run(func::FuncOp func);
};

} // namespace mlir

#endif // ROCK_ANALYSIS_LOWERINGUTILS_H