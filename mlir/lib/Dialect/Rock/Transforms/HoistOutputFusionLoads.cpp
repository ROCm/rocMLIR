//===- RockHoistOutputFusionLoadsPass - MLIR Rock ops lowering passes -----===//
//
// Copyright 2025 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
//
// This pass modifies hoist loads from output fusions
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/memoryUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKHOISTOUTPUTFUSIONLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-hoist-output-fusion-loads"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockHoistOutputFusionLoadsPass
    : public rock::impl::RockHoistOutputFusionLoadsPassBase<RockHoistOutputFusionLoadsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static LogicalResult getDependencies(Value value, llvm::SetVector<Operation*>& deps) {
  if(isa<BlockArgument>(value))
    return success();

  Operation* op = value.getDefiningOp();
  if(deps.contains(op))
    return success();

  if(!isa<ViewLikeOpInterface, memref::AllocOp, rock::GpuAllocOp>(op)) {
    return failure();
  }

  LogicalResult res = success();
  if (auto viewOp = dyn_cast<ViewLikeOpInterface>(op)) {
    res = getDependencies(viewOp.getViewSource(), deps);
  }
  deps.insert(op);
  return res;
}

static LogicalResult getDependenciesArith(Value value, llvm::SetVector<Operation*>& deps) {
  Operation* op = value.getDefiningOp();
  if(deps.contains(op))
    return success();

  if(!op || isa<rock::WorkitemIdOp, rock::WorkgroupIdOp>(op)) {
    deps.insert(op);
    return success();
  }

  if (op->getDialect()->getNamespace() != "arith") {
    return failure();
  }

  LogicalResult res = success();
  for(auto operand : op->getOperands()) {
    res = getDependenciesArith(operand, deps);
  }
  deps.insert(op);
  return res;
}

static bool shouldMove(ThreadwiseReadIntoOp op, Operation* mainLoop) {
  if(!hasGlobalMemoryAddressSpace(op.getSource().getType()) || isa_and_nonnull<scf::ForOp, affine::AffineForOp>(op->getParentOp()))
    return false;

  if(mainLoop->getBlock() != op->getBlock())
    return false;

  if(op->isBeforeInBlock(mainLoop))
    return false;

  return true;
}

static LogicalResult getOperationsToMove(ThreadwiseReadIntoOp op, llvm::SetVector<Operation*>& deps) {
  LLVM_DEBUG(llvm::dbgs() << "Found a ThreadwiseReadIntoOp to hoist: " << op << "\n");

  if(getDependencies(op.getSource(), deps).failed())
    return failure();
  if(getDependencies(op.getDest(), deps).failed())
    return failure();
  for(auto index : op.getExtraIndices()) {
    if(getDependenciesArith(index, deps).failed())
      return failure();
  }

  for(auto validities : op.getDynamicValidities()) {
    if(getDependencies(validities, deps).failed())
      return failure();
  }
    
  return success();
}

static FailureOr<Operation*> findPreviousToLastLoop(SmallVector<Operation*>& loops) {
  if (loops.size() == 1)
    return loops[0];

  // Ensure all loops are in the same block
  mlir::Block *block = loops.front()->getBlock();
  for (auto *loop : loops) {
    if (loop->getBlock() != block) {
      LLVM_DEBUG(llvm::dbgs() << "fail2\n");
      return failure();
    }
  }

  // Sort loops by their order in the block
  std::sort(loops.begin(), loops.end(), [](mlir::Operation *a, mlir::Operation *b) {
    return a->isBeforeInBlock(b);
  });

  // Return the previous-to-last loop
  return loops[loops.size() - 2];
  
}

static FailureOr<Operation*> getMainLoop(func::FuncOp func) {
  // find main loop
  // TODO: fix for attention
  SmallVector<Operation*> gemmOps;
  func.walk([&gemmOps](ThreadwiseAccelGemmOp op) {
    gemmOps.push_back(op);
  });
  if(gemmOps.empty()) {
    return failure();
  }

  SmallVector<Operation*> loops;
  for(auto *gemmOp : gemmOps) {
    Operation* current = gemmOp;

    while(current) {
      if(isa<scf::ForOp, affine::AffineForOp>(current)) {
        mlir::Operation *parent = current->getParentOp();
        if(!parent || !isa<scf::ForOp, affine::AffineForOp>(parent))
          loops.push_back(current);
      }
      current = current->getParentOp();
    }
  }
  if(loops.size() != gemmOps.size()) {
    return failure();
  }

  return findPreviousToLastLoop(loops);
}

void RockHoistOutputFusionLoadsPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  FailureOr<Operation*> maybeMainLoop = getMainLoop(func);
  if(failed(maybeMainLoop)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't find the main loop\n");
    return;
  }

  auto *mainLoop = maybeMainLoop.value();
  SmallVector<ThreadwiseReadIntoOp> readIntos;
  func.walk([&readIntos, &mainLoop](ThreadwiseReadIntoOp readInto) {
    if(shouldMove(readInto, mainLoop))
      readIntos.push_back(readInto);
  });

  if(!readIntos.empty()) {
    llvm::SetVector<Operation*> deps;
    for(auto readInto : readIntos) {
      if(getOperationsToMove(readInto, deps).failed())
        return signalPassFailure();
    }

    // Move each operation to the beginning of the block
    LLVM_DEBUG(llvm::dbgs() << "Hoisting " << deps.size()+readIntos.size() << " operations\n");
    Block *block = readIntos[0]->getBlock();
    Operation* prevOp = &block->front();
    for(Operation* opToMove : deps) {
      opToMove->moveAfter(prevOp);
      prevOp = opToMove;
    }
    
    for(ThreadwiseReadIntoOp readInto : readIntos) {
      readInto->moveAfter(mainLoop);
    }
  }
}
