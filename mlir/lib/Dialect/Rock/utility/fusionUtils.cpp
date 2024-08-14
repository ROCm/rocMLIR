//===- fusionUtils.cpp - Rock utility for fusion -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::rock;

LogicalResult mlir::rock::testFusionLegality(func::FuncOp func) {
  auto analysis = BufferDependencyAnalysis(func.getOperation());
  const auto &readersTable = analysis.getReadersTable();
  const auto &writersTable = analysis.getWritersTable();

  WalkResult walkResult =
      func.walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
        auto gemmResult = gemmOp.getOutArgument()->get();
        auto maybeAlloc = findMemrefAlloc(gemmResult);
        if (failed(maybeAlloc)) {
          return WalkResult::advance();
        }

        // make sure that no `linalg::GenericOp` reads from a gemm output
        if (readersTable.contains(*maybeAlloc)) {
          for (OpOperand *op : readersTable.at(*maybeAlloc)) {
            if (isa<linalg::GenericOp>(op->getOwner())) {
              return WalkResult::interrupt();
            }
          }
        }

        // make sure that no `linalg::GenericOp` writes to a gemm output
        if (writersTable.contains(maybeAlloc.value())) {
          for (OpOperand *op : writersTable.at(*maybeAlloc)) {
            if (isa<linalg::GenericOp>(op->getOwner())) {
              return WalkResult::interrupt();
            }
          }
        }

        return WalkResult::advance();
      });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegality(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  assert(std::distance(funcs.begin(), funcs.end()) &&
         "expected ModuleOp containing a single func::FuncOp");
  func::FuncOp func = *(funcs.begin());
  return testFusionLegality(func);
}
