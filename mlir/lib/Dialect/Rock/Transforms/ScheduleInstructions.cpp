//===- ScheduleInstructions - MLIR Rock ops lowering passes -----===//
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
// This pass reorders operations to do instruction scheduling optimizations such
// as ping-pong
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSCHEDULEINSTRUCTIONSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-schedule-instructions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockScheduleInstructionsPass
    : public rock::impl::RockScheduleInstructionsPassBase<
          RockScheduleInstructionsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static gpu::AddressSpace getAddressSpace(MemRefType type) {
  if (type.getMemorySpace()) {
    return cast<gpu::AddressSpaceAttr>(type.getMemorySpace()).getValue();
  }
  return gpu::AddressSpace::Global;
}

static bool isAccelOp(Operation *op) {
  return isa<amdgpu::WMMAOp, amdgpu::MFMAOp, amdgpu::ScaledMFMAOp>(op);
}

static bool isGlobalLoad(func::FuncOp &func, Operation *op) {
  auto funcArgs = func.getArguments();
  for (Value operand : op->getOperands()) {
    bool isRead = hasEffect<MemoryEffects::Read>(op, operand) &&
                  !hasEffect<MemoryEffects::Write>(op, operand);
    if (!isRead)
      continue;
    auto maybeBlockArg = rock::findBlockArgument(operand);
    if (failed(maybeBlockArg))
      continue;
    auto blockArg = maybeBlockArg.value();
    if (llvm::is_contained(funcArgs, blockArg))
      return true;
  }
  return false;
}

static bool isCopyRegisterOp(Operation *op) {
  bool readFromPrivate = false;
  for (Value operand : op->getOperands()) {
    bool isRead = hasEffect<MemoryEffects::Read>(op, operand) &&
                  !hasEffect<MemoryEffects::Write>(op, operand);
    if (!isRead)
      continue;

    if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
      if (getAddressSpace(memrefType) == gpu::AddressSpace::Private)
        readFromPrivate = true;
    }
  }
  // we expect a load from private, then store to private
  if (readFromPrivate && op->getNumResults() == 1) {
    Value res = op->getResult(0);
    if (res.getNumUses() == 1) {
      Operation *user = *res.getUsers().begin();
      for (Value operand : user->getOperands()) {
        bool isWrite = !hasEffect<MemoryEffects::Read>(user, operand) &&
                       hasEffect<MemoryEffects::Write>(user, operand);
        if (!isWrite)
          continue;

        if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
          if (getAddressSpace(memrefType) == gpu::AddressSpace::Private)
            return true;
        }
      }
    }
  }
  return false;
}

static bool isLDSLoad(Operation *op) {
  for (Value operand : op->getOperands()) {
    bool isRead = hasEffect<MemoryEffects::Read>(op, operand) &&
                  !hasEffect<MemoryEffects::Write>(op, operand);
    if (!isRead)
      continue;

    if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
      if (getAddressSpace(memrefType) == gpu::AddressSpace::Workgroup)
        return true;
    }
  }
  return false;
}

static bool isLDSStore(Operation *op) {
  for (Value operand : op->getOperands()) {
    bool isWrite = hasEffect<MemoryEffects::Write>(op, operand) &&
                   !hasEffect<MemoryEffects::Read>(op, operand);
    if (!isWrite)
      continue;
    if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
      if (getAddressSpace(memrefType) == gpu::AddressSpace::Workgroup)
        return true;
    }
  }
  return false;
}

static Operation *moveOpAndDepedencies(Operation *op, Operation *lastInsertedOp,
                                       scf::ForOp loop) {
  // Track visited operations to avoid redundant moves
  llvm::SmallPtrSet<Operation *, 16> visited;

  // Get the loop body block
  Block *loopBody = loop.getBody();

  // Recursive lambda to move dependencies
  std::function<void(Operation *)> moveRecursive = [&](Operation *currentOp) {
    if (!currentOp || visited.contains(currentOp))
      return;

    visited.insert(currentOp);

    // Only proceed if the operation is inside the loop body
    if (currentOp->getBlock() != loopBody)
      return;

    // Move dependencies first
    for (Value operand : currentOp->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        moveRecursive(defOp);
      }
    }

    // If the operation has regions, process their blocks recursively
    for (Region &region : currentOp->getRegions()) {
      for (Block &block : region) {
        for (Operation &nestedOp : block) {
          for (Value operand : nestedOp.getOperands()) {
            if (Operation *defOp = operand.getDefiningOp()) {
              moveRecursive(defOp);
            }
          }
        }
      }
    }

    // Only move if currentOp is not already before lastInsertedOp
    if (currentOp->isBeforeInBlock(lastInsertedOp))
      return;

    // Move the current operation after the last inserted one
    currentOp->moveAfter(lastInsertedOp);
    lastInsertedOp = currentOp;
  };

  moveRecursive(op);

  // Move the instructions that use the output of 'op'
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      // Only consider users in the same block
      if (user->getBlock() != loop.getBody())
        continue;

      // Skip if already visited
      if (visited.contains(user))
        continue;

      // Move only if it's after lastInsertedOp
      if (user->isBeforeInBlock(lastInsertedOp))
        continue;

      moveRecursive(user);
    }
  }
  return lastInsertedOp;
}

static Operation *addClusterBarrier(OpBuilder &builder,
                                    Operation *lastInsertedOp) {
  Location loc = lastInsertedOp->getLoc();
  builder.setInsertionPointAfter(lastInsertedOp);
  builder.create<gpu::BarrierOp>(loc);
  return builder.create<ROCDL::SchedBarrier>(loc, 0);
}

// for 8 waves, split into 4 "stages"
// for 8 waves, schedule 16 mfmas each time, 64 in total
// for 8 waves, schedule 2 global loads each time, 8 in total
// for 8 waves, schedule 24 LDS loads each time, 6 in total (if LDS load stage
// is not part of MMA stage, otherwise, just do what needs to be loaded for the
// mfma) for 8 waves, schedule 2 LDS stores each time, 8 in total (0 if direct
// to LDS)
static LogicalResult
scheduleInstruction8waves(OpBuilder &builder, scf::ForOp loop,
                          const ArrayRef<Operation *> accelOps,
                          const ArrayRef<Operation *> globalLoads,
                          const ArrayRef<Operation *> copyRegistersOps,
                          const ArrayRef<Operation *> ldsLoads,
                          const ArrayRef<Operation *> ldsStores) {
  // Remove all existing barriers
  loop->walk([&](rock::LDSBarrierOp barrier) { barrier->erase(); });

  int lowPriority = 0;
  int highPriority = 1;
  size_t numClusters = 4;
  size_t numGlobalLoads = llvm::divideCeil(globalLoads.size(), numClusters / 2);
  size_t numLDSLoads = llvm::divideCeil(ldsLoads.size(), numClusters-1);
  size_t numAccelOps = llvm::divideCeil(accelOps.size(), numClusters);
  // size_t numLDSLoads[3] = {6, 6, 12};
  // numLDSLoads = ldsLoads.size();

  Operation *lastInsertedOp = &loop.getBody()->front();

  size_t globalLoadIdx = 0;
  size_t ldsLoadIdx = 0;
  for (size_t cluster = 0; cluster < numClusters; cluster++) {
    llvm::errs() << "cluster=" << cluster << "\n";
    // 1. memory cluster
    if (cluster == numClusters - 1) {
      // store cluster
      int n = 0;
      // all copy register ops happen in the last memory cluster
      for (size_t idx = 0; idx < copyRegistersOps.size(); idx++) {
        if (idx < copyRegistersOps.size()) {
          lastInsertedOp =
              moveOpAndDepedencies(copyRegistersOps[idx], lastInsertedOp, loop);
          n++;
        }
      }
      llvm::errs() << "moved " << n << " copy register ops\n";
      builder.setInsertionPointAfter(lastInsertedOp);
      // TODO: only needed for single-buffer
      // lastInsertedOp =
      // builder.create<rock::LDSBarrierOp>(lastInsertedOp->getLoc());
      n = 0;
      // all stores happen in the last memory cluster
      for (size_t idx = 0; idx < ldsStores.size(); idx++) {
        if (idx < ldsStores.size()) {
          lastInsertedOp =
              moveOpAndDepedencies(ldsStores[idx], lastInsertedOp, loop);
          n++;
        }
      }
      llvm::errs() << "moved " << n << " LDS stores\n";
    } else {
      if (cluster == 0) {
        // backward barrier
        builder.setInsertionPointAfter(lastInsertedOp);
        lastInsertedOp =
            rock::LDSBarrierOp::create(builder, lastInsertedOp->getLoc());
      }
      // LDS loads
      int n = 0;
      for (size_t idx = 0; idx < ldsLoads.size(); idx++) {
        size_t clusterIdx = idx + ldsLoadIdx;
        if (clusterIdx < ldsLoads.size()) {
          lastInsertedOp =
              moveOpAndDepedencies(ldsLoads[clusterIdx], lastInsertedOp, loop);
          n++;
        }
      }
      ldsLoadIdx += n;
      llvm::errs() << "moved " << n << " LDS loads\n";

      // global loads
      n = 0;
      // llvm::errs() << "cluster % 2 == 0 = " << (cluster % 2 == 0) << "\n";
      // if(cluster % 2 == 0) {
      for (size_t idx = 0; idx < numGlobalLoads; idx++) {
        size_t clusterIdx = idx + globalLoadIdx;
        if (clusterIdx < globalLoads.size()) {
          lastInsertedOp = moveOpAndDepedencies(globalLoads[clusterIdx],
                                                lastInsertedOp, loop);
          n++;
        }
      }
      globalLoadIdx += n;
      // }
      llvm::errs() << "moved " << n << " global loads\n";
    }

    lastInsertedOp = addClusterBarrier(builder, lastInsertedOp);

    // 2. compute cluster
    builder.setInsertionPointAfter(lastInsertedOp);
    lastInsertedOp = builder.create<ROCDL::SetPrioOp>(lastInsertedOp->getLoc(),
                                                      highPriority);
    int n = 0;
    for (size_t idx = 0; idx < numAccelOps; idx++) {
      size_t clusterIdx = idx + cluster * numAccelOps;
      if (clusterIdx < accelOps.size()) {
        lastInsertedOp =
            moveOpAndDepedencies(accelOps[clusterIdx], lastInsertedOp, loop);
        n++;
      }
    }
    builder.setInsertionPointAfter(lastInsertedOp);
    lastInsertedOp =
        builder.create<ROCDL::SetPrioOp>(lastInsertedOp->getLoc(), lowPriority);
    llvm::errs() << "moved " << n << " accel ops\n";
    if (cluster != numClusters - 2)
      lastInsertedOp = addClusterBarrier(builder, lastInsertedOp);
  }
  return success();
}

static Operation *extractOp(Operation *op, scf::ForOp parentLoop) {
  // Get the region of the parent loop
  Block *loopBody = parentLoop.getBody();

  // Traverse up the parent hierarchy
  Operation *currentOp = op;
  while (currentOp) {
    // Check if the operation is inside the loop body
    if (currentOp->getBlock() == loopBody &&
        currentOp != parentLoop.getOperation()) {
      return currentOp;
    }
    currentOp = currentOp->getParentOp();
  }

  // If no enclosing operation is found within the loop, return nullptr
  return nullptr;
}

static void condBarrier(OpBuilder &builder, Location loc, CmpIOp cond) {
  scf::IfOp ifb = scf::IfOp::create(builder, loc, cond,
                                    /*withElseRegion=*/false);
  {
    OpBuilder thenb = ifb.getThenBodyBuilder();
    ROCDL::SBarrierOp::create(thenb, loc);
  }
}

static void addAsymmetricSyncToLoop(OpBuilder &builder, scf::ForOp loop) {
  Location loc = loop->getLoc();
  builder.setInsertionPoint(loop);
  // Set barrier before starting the loop. This resolves any remaining required
  // synchronization before beginning the specialized asymmetric
  // synchronization.
  auto preBarrier = builder.create<gpu::BarrierOp>(loc);
  builder.setInsertionPointAfter(preBarrier);

  // Insert condbarrier::second_half before starting the loop
  auto i32ty = builder.getIntegerType(32);
  auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
  auto constZero = builder.create<arith::ConstantIntOp>(loc, i32ty, 0);
  // assuming block_size=512
  auto constWarpSize = builder.create<arith::ConstantIntOp>(loc, i32ty, 256);
  auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWarpSize);
  auto warpLow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               warpIDX, constZero);
  auto warpHigh = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                warpIDX, constZero);
  condBarrier(builder, loc, warpHigh);

  // Insert condbarrier::first_half loop the end of the loop
  builder.setInsertionPointAfter(loop);
  condBarrier(builder, loc, warpLow);
}

static LogicalResult
scheduleInstructions(OpBuilder &builder, func::FuncOp &func, scf::ForOp forOp) {
  // 1. unroll accel affine loops
  SmallVector<affine::AffineForOp> loopsToUnroll;
  // Since this is a post-order walk through a loop nest, the
  // first loop we see is innermost and we want to unroll it first.
  forOp.walk<WalkOrder::PostOrder>([&loopsToUnroll](affine::AffineForOp loop) {
    auto foundAccel = loop.walk([](Operation *accelOp) -> WalkResult {
      if (isAccelOp(accelOp))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    // found an accel op, unroll the loop
    if (foundAccel.wasInterrupted())
      loopsToUnroll.push_back(loop);
  });

  for (auto loop : loopsToUnroll) {
    if (failed(mlir::affine::loopUnrollFull(loop))) {
      LLVM_DEBUG(DBGS() << "Failed to unroll loop =" << loop << "\n");
      return failure();
    }
  }

  SmallVector<Operation *> accelOps;
  SmallVector<Operation *> globalLoads;
  SmallVector<Operation *> copyRegistersOps;
  // TODO: find ldsLoads as dependency of accelOps!
  SmallVector<Operation *> ldsLoads;
  SmallVector<Operation *> ldsStores;
  // 2. get loads, stores and accel ops inside the loop
  forOp->walk([&](Operation *op) {
    if (isGlobalLoad(func, op))
      globalLoads.push_back(extractOp(op, forOp));
    if (isCopyRegisterOp(op))
      copyRegistersOps.push_back(extractOp(op, forOp));
    else if (isLDSLoad(op))
      ldsLoads.push_back(extractOp(op, forOp));
    else if (isLDSStore(op))
      ldsStores.push_back(extractOp(op, forOp));
    else if (isAccelOp(op))
      accelOps.push_back(op);
  });

  if (accelOps.empty() || globalLoads.empty()) {
    LLVM_DEBUG(DBGS() << "Expected the loop to have at least some accel ops "
                         "and global loads\n");
    return failure();
  }
  if (llvm::any_of(globalLoads, [](Operation *op) { return op == nullptr; })) {
    LLVM_DEBUG(DBGS() << "Found nullptr in global load list\n");
    return failure();
  }
  if (llvm::any_of(copyRegistersOps,
                   [](Operation *op) { return op == nullptr; })) {
    LLVM_DEBUG(DBGS() << "Found nullptr in copy register op list\n");
    return failure();
  }
  if (llvm::any_of(accelOps, [](Operation *op) { return op == nullptr; })) {
    LLVM_DEBUG(DBGS() << "Found nullptr in accel op list\n");
    return failure();
  }
  if (llvm::any_of(ldsLoads, [](Operation *op) { return op == nullptr; })) {
    LLVM_DEBUG(DBGS() << "Found nullptr in LDS load list\n");
    return failure();
  }
  if (llvm::any_of(ldsStores, [](Operation *op) { return op == nullptr; })) {
    LLVM_DEBUG(DBGS() << "Found nullptr in LDS store list\n");
    return failure();
  }

  LLVM_DEBUG(DBGS() << "Found " << accelOps.size() << " accel ops:\n");
  for (Operation *op : accelOps)
    LLVM_DEBUG(DBGS() << *op << "\n");
  LLVM_DEBUG(DBGS() << "Found " << globalLoads.size() << " global load ops:\n");
  for (Operation *op : globalLoads)
    LLVM_DEBUG(DBGS() << *op << "\n");
  LLVM_DEBUG(DBGS() << "Found " << copyRegistersOps.size()
                    << " copy register ops:\n");
  for (Operation *op : copyRegistersOps)
    LLVM_DEBUG(DBGS() << *op << "\n");
  LLVM_DEBUG(DBGS() << "Found " << ldsLoads.size() << " LDS load ops:\n");
  for (Operation *op : ldsLoads)
    LLVM_DEBUG(DBGS() << *op << "\n");
  LLVM_DEBUG(DBGS() << "Found " << ldsStores.size() << " LDS store ops:\n");
  for (Operation *op : ldsStores)
    LLVM_DEBUG(DBGS() << *op << "\n");

  auto maybeBlocksize = rock::getBlockSize(forOp);
  if (failed(maybeBlocksize)) {
    LLVM_DEBUG(DBGS() << "Could not find block size\n");
    return failure();
  }
  int64_t blockSize = maybeBlocksize.value().getValue().getSExtValue();
  int64_t waveSize = rock::lookupArchInfo(rock::getArchValue(forOp)).waveSize;
  int64_t numWaves = blockSize / waveSize;

  // 3. reorder them
  if (numWaves == 8) {
    LLVM_DEBUG(DBGS() << "8 waves\n");
    if (failed(scheduleInstruction8waves(builder, forOp, accelOps, globalLoads,
                                         copyRegistersOps, ldsLoads,
                                         ldsStores)))
      return failure();
  }

  // if(failed(dummyMover(builder, forOp, accelOps, globalLoads,
  // copyRegistersOps,
  //                                        ldsLoads, ldsStores))) {
  //     return failure();
  // }

  addAsymmetricSyncToLoop(builder, forOp);
  return success();
}

static LogicalResult scheduleInstructions(func::FuncOp &func) {
  OpBuilder builder(func->getContext());

  // schedule instructions for all scf.for loops
  auto walkRes = func.walk([&func, &builder](scf::ForOp forOp) -> WalkResult {
    if (failed(scheduleInstructions(builder, func, forOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (walkRes.wasInterrupted())
    return failure();

  return success();
}

void RockScheduleInstructionsPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  if (failed(scheduleInstructions(func))) {
    return signalPassFailure();
  }
}
