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

bool mlir::rock::testFusability(ModuleOp mod) {
  llvm::SmallVector<memref::AllocOp, 8> memAllocOps;
  mod.walk([&](memref::AllocOp allocOp) { memAllocOps.push_back(allocOp); });

  if (memAllocOps.empty()) {
    return true;
  }

  auto maps = BufferDependencyAnalysis::run(memAllocOps);

  WalkResult walkResult =
      mod.walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
        auto gemmResult = gemmOp->getOperand(2);
        while (auto viewOp =
                   dyn_cast<ViewLikeOpInterface>(gemmResult.getDefiningOp())) {
          gemmResult = viewOp.getViewSource();
        }
        Operation *gemmResultSrc = gemmResult.getDefiningOp();
        if (auto allocOp = dyn_cast<memref::AllocOp>(gemmResultSrc)) {
          if (maps.readers.contains(allocOp)) {
            for (Operation *op : maps.readers[allocOp]) {
              if (dyn_cast<linalg::GenericOp>(op)) {
                return WalkResult::interrupt();
              }
            }
          }
        }

        return WalkResult::advance();
      });

  return !walkResult.wasInterrupted();
}
