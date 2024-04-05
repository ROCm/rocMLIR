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

struct TestFusibilityPass
    : public PassWrapper<TestFusibilityPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    BufferDependencyAnalysis &analysis =
        getAnalysis<BufferDependencyAnalysis>();
    if (failed(analysis.run())) {
      return;
    }

    const auto &readersTable = analysis.getReadersTable();
    const auto &writersTable = analysis.getWritersTable();

    func::FuncOp func = getOperation();
    WalkResult walkResult =
        func.walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
          auto gemmResult = gemmOp->getOperand(2);
          auto allocOp = BufferDependencyAnalysis::getAllocation(gemmResult);
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

    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

LogicalResult mlir::rock::testFusibility(ModuleOp mod) {
  PassManager pm = PassManager::on<ModuleOp>(mod->getContext(),
                                             PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<TestFusibilityPass>());
  return pm.run(mod);
}

LogicalResult mlir::rock::testFusibility(func::FuncOp func) {
  PassManager pm = PassManager::on<func::FuncOp>(
      func->getContext(), PassManager::Nesting::Implicit);
  pm.addPass(std::make_unique<TestFusibilityPass>());
  return pm.run(func);
}
