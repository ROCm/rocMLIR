//===- AddAsyncWait - MLIR Rock ops lowering passes -----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass adds async wait operations for LDS memory
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKADDASYNCWAITPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-add-async-wait"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockAddAsyncWaitPass
    : public rock::impl::RockAddAsyncWaitPassBase<RockAddAsyncWaitPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

/// Check if a ThreadwiseReadIntoOp reads from global memory (no GPU address
/// space).
static bool isGlobalLoad(ThreadwiseReadIntoOp readOp) {
  auto sourceMemSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(
      readOp.getSource().getType().getMemorySpace());
  return !sourceMemSpace;
}

static bool hasLDSAddressSpace(gpu::AddressSpaceAttr spaceAttr) {
  return spaceAttr &&
         spaceAttr.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// Check if a ThreadwiseReadIntoOp reads from LDS (workgroup address space).
static bool isLDSLoad(ThreadwiseReadIntoOp readOp) {
  auto sourceMemSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(
      readOp.getSource().getType().getMemorySpace());
  return hasLDSAddressSpace(sourceMemSpace);
}

// Check if a ThreadwiseReadIntoOp writes to LDS (workgroup address space).
static bool isLDSWrite(ThreadwiseReadIntoOp readOp) {
  auto destMemSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(
      readOp.getDest().getType().getMemorySpace());
  return hasLDSAddressSpace(destMemSpace);
}

/// Check if an operation should be skipped when looking for reads after
/// startOp.
static bool
shouldSkipOperation(Operation *op, Operation *startOp,
                    const llvm::DenseSet<Operation *> insertionPoints) {
  // Skip if it's thesame as startOp, or already has insertion point
  if (op == startOp || insertionPoints.contains(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping op because it's already been used: "
                            << *op << "\n");
    return true;
  }
  // If in the same block, skip if before startOp
  if (op->getBlock() == startOp->getBlock() && op->isBeforeInBlock(startOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping op because it's before startOp: " << *op << "\n");
    return true;
  }
  return false;
}

/// Compare two operations to determine which comes first in program order.
/// Returns true if op1 comes before op2, false otherwise.
static bool comesBeforeInProgramOrder(Operation *op1, Operation *op2) {
  Block *block1 = op1->getBlock();
  Block *block2 = op2->getBlock();

  // Same block: use isBeforeInBlock for ordering
  if (block1 == block2) {
    return op1->isBeforeInBlock(op2);
  }

  // Different blocks: check if one is nested inside the other
  Operation *parent1 = block1->getParentOp();
  Operation *parent2 = block2->getParentOp();

  // Check if block1 is nested inside block2's region
  if (parent1 && parent1->getBlock() == block2) {
    return parent1->isBeforeInBlock(op2);
  }

  // Check if block2 is nested inside block1's region
  if (parent2 && parent2->getBlock() == block1) {
    return op1->isBeforeInBlock(parent2);
  }

  // If neither is nested in the other, we can't determine ordering
  return false;
}

/// Traverse forward from a value to find ThreadwiseReadIntoOps that read from
/// LDS. Returns the first one found in program order.
static ThreadwiseReadIntoOp
findFirstReadFromLDS(Value value, Operation *startOp,
                     const llvm::DenseSet<Operation *> insertionPoints) {
  ThreadwiseReadIntoOp firstRead = nullptr;

  // Worklist to traverse forward through views/extract_multibuffer/transform
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;

  auto addToWorklist = [&](Value v) {
    if (visited.insert(v).second) {
      worklist.push_back(v);
    }
  };

  addToWorklist(value);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();

    LLVM_DEBUG(llvm::dbgs() << "Checking current value: " << current << "\n");

    if (current.getUsers().empty()) {
      continue;
    }

    // Check all users of this value
    for (Operation *user : current.getUsers()) {
      if (auto readOp = dyn_cast<ThreadwiseReadIntoOp>(user)) {
        // Only consider reads from LDS.
        if (!isLDSLoad(readOp)) {
          continue;
        }

        // We may want to skip an op if its after the startOp, or if it has
        // already been used before.
        if (shouldSkipOperation(user, startOp, insertionPoints)) {
          continue;
        }

        // Found a read after startOp, track the first one in program order
        if (!firstRead ||
            comesBeforeInProgramOrder(user, firstRead.getOperation())) {
          firstRead = readOp;
        }
      }
      // If this is a view-like operation, follow its result
      else if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user)) {
        addToWorklist(viewOp->getResult(0));
      }
      // If this is an extract_multibuffer, follow its result
      else if (auto extractOp = dyn_cast<rock::ExtractMultiBufferOp>(user)) {
        addToWorklist(extractOp.getResult());
      }
    }
  }

  return firstRead;
}

/// Find the first use after a ThreadwiseReadIntoOp that writes to LDS.
/// The function traces back the dest value to the alloc, then finds the first
/// ThreadwiseReadIntoOp that reads from that alloc.
static FailureOr<Operation *>
findFirstUseAfter(ThreadwiseReadIntoOp writeOp,
                  const llvm::DenseSet<Operation *> insertionPoints) {
  // Get the destination value (what is written to)
  Value dest = writeOp.getDest();

  // Trace back to find all possible allocs
  SmallVector<rock::GpuAllocOp> allocs = rock::findAllGpuAllocs(dest);
  if (allocs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to trace dest to alloc\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << allocs.size() << " alloc(s)\n");

  // Find the first read from LDS that uses any of these allocs
  ThreadwiseReadIntoOp firstRead = nullptr;
  Operation *firstReadOp = nullptr;

  for (auto alloc : allocs) {
    LLVM_DEBUG(llvm::dbgs() << "Checking alloc: " << alloc << "\n");
    ThreadwiseReadIntoOp read = findFirstReadFromLDS(
        alloc.getResult(), writeOp.getOperation(), insertionPoints);

    if (read) {
      Operation *readOp = read.getOperation();
      // Track the first one in program order
      if (!firstReadOp || readOp->isBeforeInBlock(firstReadOp)) {
        firstRead = read;
        firstReadOp = readOp;
      }
    }
  }

  if (!firstRead) {
    LLVM_DEBUG(llvm::dbgs() << "No read found after writeOp\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "After writeOp: " << *writeOp.getOperation()
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Found first read after writeOp: " << firstRead
                          << "\n");
  return firstRead.getOperation();
}

/// Count global loads (ThreadwiseReadIntoOp from global to LDS) between two
/// operations in a block. Counts operations from startOp (exclusive) to endOp
/// (exclusive). If countFromStart is true, counts from the beginning of the
/// block, ignoring startOp (which would be nullptr).
static int countGlobalLoadsBetween(Operation *startOp, Operation *endOp,
                                   Block *block, bool countFromStart) {
  int count = 0;
  bool counting = countFromStart;

  for (Operation &op : *block) {
    if (!countFromStart && startOp && &op == startOp) {
      counting = true;
      continue; // Start counting after startOp
    }
    if (endOp && &op == endOp) {
      break; // Stop counting at endOp
    }
    if (!counting) {
      continue;
    }
    if (auto readOp = dyn_cast<ThreadwiseReadIntoOp>(&op)) {
      // Check if it reads from global memory
      if (isGlobalLoad(readOp) && isLDSWrite(readOp)) {
        auto maybeLoopCount = rock::predictThreadwiseReadIntoLoopCount(readOp);
        if (failed(maybeLoopCount)) {
          // If we failed to predict the loop count, we don't know how many
          // global loads are between the two operations, so we will be
          // conservative and assume it will lower to just one global load.
          LLVM_DEBUG(
              llvm::dbgs()
              << "Failed to predict loop count for ThreadwiseReadIntoOp: "
              << *readOp << "\n");
        }
        count += maybeLoopCount.value_or(1);
      }
    }
  }

  return count;
}

/// The localLoadOp is the load that triggers the dependency, and the
/// globalLoadOp is the load that is dependent on the localLoadOp.
std::pair<int, bool> getWaitCount(Operation *localLoadOp,
                                  Operation *globalLoadOp) {
  if (!localLoadOp || !globalLoadOp)
    return {-1, false};

  auto localReadOp = dyn_cast<ThreadwiseReadIntoOp>(localLoadOp);
  auto globalReadOp = dyn_cast<ThreadwiseReadIntoOp>(globalLoadOp);
  if (!localReadOp || !globalReadOp)
    return {-1, false};

  // Get parent blocks and check for loops
  Block *globalBlock = globalLoadOp->getBlock();
  Block *localBlock = localLoadOp->getBlock();
  assert(globalBlock && "Expected global load op to be in a block");
  assert(localBlock && "Expected local load op to be in a block");

  LoopLikeOpInterface localLoop =
      localLoadOp->getParentOfType<LoopLikeOpInterface>();
  LoopLikeOpInterface globalLoop =
      globalLoadOp->getParentOfType<LoopLikeOpInterface>();

  int waitCount = 0;
  bool pipeliningEnabled = false;

  // Case 1: Both have the same parent block (function) - prologue
  if (!localLoop && !globalLoop) {
    assert(globalBlock == localBlock &&
           "Expected global and local load ops to be in the same block");
    LLVM_DEBUG(llvm::dbgs() << "Case 1: Both in function (prologue)\n");
    // Count global loads between globalLoadOp and localLoadOp in the same block
    waitCount = countGlobalLoadsBetween(globalLoadOp, localLoadOp, globalBlock,
                                        /*countFromStart=*/false);
    pipeliningEnabled = true;
  }
  // Case 2: localLoad in loop, globalLoad in function - body
  else if (localLoop && !globalLoop) {
    LLVM_DEBUG(llvm::dbgs()
               << "Case 2: localLoad in loop, globalLoad in function (body)\n");
    // Count from globalLoadOp to the loop operation (which marks the start of
    // the loop block)
    Block *loopOpBlock = localLoop->getBlock();
    if (loopOpBlock == globalBlock) {
      waitCount += countGlobalLoadsBetween(
          globalLoadOp, localLoop.getOperation(), globalBlock,
          /*countFromStart=*/false);
    }

    // Count from the start of the loop body block to localLoadOp
    assert(localLoop.getLoopRegions().size() == 1 &&
           "Expected local loop to have exactly one region");
    Block *loopBody = &localLoop.getLoopRegions().front()->front();
    waitCount += countGlobalLoadsBetween(nullptr, localLoadOp, loopBody,
                                         /*countFromStart=*/true);
    pipeliningEnabled = true;
  }
  // Case 3: localLoad in function, globalLoad in loop - epilogue
  else if (!localLoop && globalLoop) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Case 3: localLoad in function, globalLoad in loop (epilogue)\n");
    // Always return 0 for epilogue
    return {0, true};
  }
  // Case 4: no pipelining.
  else {
    LLVM_DEBUG(llvm::dbgs() << "Case 4: both in loops (no pipelining)\n");
    return {0, false};
  }

  LLVM_DEBUG(llvm::dbgs() << "  waitCount: " << waitCount << "\n");

  return {waitCount >= 0 ? waitCount : 0, pipeliningEnabled};
}

static Operation *getInsertionPointForAsyncWait(IRRewriter &rewriter,
                                                Operation *op,
                                                bool pipeliningEnabled) {
  if (pipeliningEnabled) {
    // If pipelining is enabled, we always insert the AsyncWaitOp just before
    // the local load.
    return op;
  } else {
    // If pipelining is disabled, we have to insert the AsyncWaitOp just after
    // the LDSBarrierOp. Find the first LDSBarrierOp with
    // barrier_stage="backward" in the function
    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    if (!func) {
      LLVM_DEBUG(llvm::dbgs() << "No function found, inserting before op\n");
      return nullptr;
    }

    rock::LDSBarrierOp forwardBarrier = nullptr;
    func.walk([&](rock::LDSBarrierOp barrier) {
      if (!forwardBarrier) {
        auto barrierStageAttr = barrier.getBarrierStageAttr();
        if (barrierStageAttr &&
            barrierStageAttr.getValue() == rock::BarrierStage::Forward) {
          forwardBarrier = barrier;
        }
      }
    });

    if (forwardBarrier) {
      llvm::dbgs() << "Found forward barrier: " << forwardBarrier << "\n";
      return forwardBarrier;
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "No forward barrier found, inserting before op\n");
      return op;
    }
  }
}

/// Add AsyncWaitOps to the function. This function has 3 main steps:
/// 1. Find all ThreadwiseReadIntoOp operations that write into LDS memory.
///    Then, for each of them, call findFirstUseAfter, which will find the first
///    op that uses the result of the ThreadwiseReadIntoOp. In essence, this
///    gives us the pair of global memory read and the LDS load that depends
///    on it.
/// 2. For each pair of global and local reads, call getWaitCount, which will
///    return the number of AsyncWaitOps to insert.
/// 3. Once we know the wait count and the pair of reads that depends on each
///    other, figure out where to insert the waitcount, which mainly depends on
///    whether we are running pipelined or not.
static LogicalResult addAsyncWait(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());

  // Find all ThreadwiseReadIntoOp operations
  SmallVector<rock::ThreadwiseReadIntoOp> readOps;
  func.walk([&](rock::ThreadwiseReadIntoOp op) {
    // Only add reads that write into LDS memory
    if (isLDSWrite(op)) {
      readOps.push_back(op);
    }
  });

  // Track insertion points to avoid inserting multiple AsyncWaitOps at the same
  // location
  llvm::DenseSet<Operation *> insertionPoints;
  for (auto readOp : readOps) {
    auto firstUseOr = findFirstUseAfter(readOp, insertionPoints);

    if (failed(firstUseOr)) {
      // If pattern doesn't match or no use found, fail.
      return emitError(readOp.getLoc(),
                       "No use found for ThreadwiseReadIntoOp. Is there a "
                       "ThreadwiseReadIntoOp that writes to LDS but none is "
                       "reading from it?");
    }

    Operation *firstUse = *firstUseOr;

    auto [waitCount, pipeliningEnabled] = getWaitCount(firstUse, readOp);
    if (waitCount == -1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get wait count for ThreadwiseReadIntoOp\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Looking for insertion point for AsyncWaitOp after "
               << firstUse << "\n");

    Operation *insertionPoint =
        getInsertionPointForAsyncWait(rewriter, firstUse, pipeliningEnabled);

    // Only insert one AsyncWaitOp per insertion point
    if (insertionPoints.contains(insertionPoint)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping insertion point because it already "
                                 "has an AsyncWaitOp\n");
      continue;
    }
    insertionPoints.insert(insertionPoint);
    if (pipeliningEnabled)
      rewriter.setInsertionPoint(insertionPoint);
    else
      rewriter.setInsertionPointAfter(insertionPoint);

    rock::AsyncWaitOp::create(rewriter, insertionPoint->getLoc(), waitCount);

    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  return success();
}

void RockAddAsyncWaitPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel")) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping RockAddAsyncWaitPass on func with "
                               "no kernel attribute\n");
    return;
  }

  if (failed(addAsyncWait(func))) {
    return signalPassFailure();
  }
}
