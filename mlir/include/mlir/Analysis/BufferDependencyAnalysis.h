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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

struct BufferDependencyAnalysis {
  struct AnalysisResults {
    llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> readers;
    llvm::DenseMap<memref::AllocOp, llvm::SmallVector<Operation *>> writers;
  };

  struct LocalResult {
    llvm::SmallVector<Operation *> readers;
    llvm::SmallVector<Operation *> writers;
  };

  static LocalResult findReadersAndWriters(memref::AllocOp allocOps);
  static AnalysisResults run(llvm::SmallVectorImpl<memref::AllocOp> &allocOps);
};

} // namespace mlir

#endif // ROCK_ANALYSIS_LOWERINGUTILS_H