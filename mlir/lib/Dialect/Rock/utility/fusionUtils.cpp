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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::rock;

std::optional<memref::AllocOp> mlir::rock::getAllocation(Value value) {
  while (auto viewOp = dyn_cast<ViewLikeOpInterface>(value.getDefiningOp())) {
    value = viewOp.getViewSource();
  }
  if (auto allocOp = dyn_cast<memref::AllocOp>(value.getDefiningOp())) {
    return allocOp;
  }
  return std::nullopt;
}

LogicalResult mlir::rock::testFusionLegality(func::FuncOp func) {
  auto analysis = BufferDependencyAnalysis(func.getOperation());
  const auto &readersTable = analysis.getReadersTable();
  const auto &writersTable = analysis.getWritersTable();

  WalkResult walkResult =
      func.walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
        auto gemmResult = gemmOp->getOperand(2);
        auto allocOp = getAllocation(gemmResult);
        if (!allocOp.has_value()) {
          return WalkResult::advance();
        }

        // make sure that no `linalg::GenericOp` reads from a gemm output
        if (readersTable.contains(allocOp.value())) {
          for (Operation *op : readersTable.at(allocOp.value())) {
            if (dyn_cast<linalg::GenericOp>(op)) {
              return WalkResult::interrupt();
            }
          }
        }

        // make sure that no `linalg::GenericOp` writes to a gemm output
        if (writersTable.contains(allocOp.value())) {
          for (Operation *op : writersTable.at(allocOp.value())) {
            if (dyn_cast<linalg::GenericOp>(op)) {
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
