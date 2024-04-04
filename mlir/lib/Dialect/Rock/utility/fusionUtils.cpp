//===- fusionUtils.cpp - Rock utility for fusion -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::rock;

bool mlir::rock::testFusibility(ModuleOp mod) {
  WalkResult result = mod->walk([](func::FuncOp func) -> WalkResult {
    if (!testFusibility(func)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

bool mlir::rock::testFusibility(func::FuncOp func) {
  auto analysisResult = BufferDependencyAnalysis::run(func);
  if (!analysisResult.has_value()) {
    return true;
  }

  auto tables = analysisResult.value();
  WalkResult walkResult =
      func.walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
        auto gemmResult = gemmOp->getOperand(2);
        auto allocOp = BufferDependencyAnalysis::getAllocation(gemmResult);
        if (!allocOp.has_value()) {
          return WalkResult::advance();
        }

        // make sure that no `linalg::GenericOp` reads from a gemm output
        if (tables.readers.contains(allocOp.value())) {
          for (Operation *op : tables.readers[allocOp.value()]) {
            if (dyn_cast<linalg::GenericOp>(op)) {
              return WalkResult::interrupt();
            }
          }
        }

        // make sure that no `linalg::GenericOp` writes to a gemm output
        if (tables.writers.contains(allocOp.value())) {
          for (Operation *op : tables.writers[allocOp.value()]) {
            if (dyn_cast<linalg::GenericOp>(op)) {
              return WalkResult::interrupt();
            }
          }
        }

        return WalkResult::advance();
      });

  return !walkResult.wasInterrupted();
}
