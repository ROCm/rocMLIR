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
    : public rock::impl::RockAddAsyncWaitPassBase<
          RockAddAsyncWaitPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

/// Trace back a value to find all GpuAllocOps it originates from.
/// Handles views, extract_multibuffer, and transform operations.
/// Returns all allocs that could be the source (for extract_multibuffer with multiple buffers).
static SmallVector<rock::GpuAllocOp> traceToAllocs(Value value) {
  SmallVector<rock::GpuAllocOp> allocs;
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
    auto *curOp = current.getDefiningOp();
    
    if (!curOp) {
      // Value doesn't have a defining op (e.g., block argument), skip it
      continue;
    }
    
    if (auto allocOp = dyn_cast<rock::GpuAllocOp>(curOp)) {
      allocs.push_back(allocOp);
      continue;
    }

    // Keep going until the operation that defines the value is a
    // view-like operation
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(curOp)) {
      addToWorklist(viewOp.getViewSource());
    } else if (auto extractMultiBufferOp =
                   dyn_cast<rock::ExtractMultiBufferOp>(curOp)) {
      // For extract_multibuffer, check all buffers since reads might use any of them
      auto buffers = extractMultiBufferOp.getBuffers();
      for (auto buffer : buffers) {
        addToWorklist(buffer);
      }
    } else if (auto transformOp = dyn_cast<rock::TransformOp>(curOp)) {
      addToWorklist(transformOp.getInput());
    }
  }

  return allocs;
}

/// Traverse forward from a value to find ThreadwiseReadIntoOps that read from LDS.
/// Returns the first one found in program order.
static ThreadwiseReadIntoOp findFirstReadFromLDS(Value value, Operation *startOp, llvm::DenseSet<Operation*> &insertionPoints) {
  ThreadwiseReadIntoOp firstRead = nullptr;
  Operation *firstReadOp = nullptr;

  // Worklist to traverse forward through views/extract_multibuffer/transform
  SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  
  auto addToWorklist = [&](Value v) {
    LLVM_DEBUG(llvm::dbgs() << "  -> Checking: " << *v.getDefiningOp() << "\n");
    if (visited.insert(v).second) {
      worklist.push_back(v);
      LLVM_DEBUG(llvm::dbgs() << "  -> Added to worklist!\n");
    }
  };

  addToWorklist(value);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();

    LLVM_DEBUG(llvm::dbgs() << "  -> Checking current: " << *current.getDefiningOp() << "\n");

    if (current.getUsers().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "  -> No users found for current: " << *current.getDefiningOp() << "\n");
      continue;
    }

    // Check all users of this value
    for (Operation *user : current.getUsers()) {
      // If this is a ThreadwiseReadIntoOp, check if it reads FROM LDS
      if (auto readOp = dyn_cast<ThreadwiseReadIntoOp>(user)) {
        // Check if source is LDS (workgroup address space)
        auto sourceMemSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(
            readOp.getSource().getType().getMemorySpace());
        if (sourceMemSpace && 
            sourceMemSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace()) {
          // This reads FROM LDS, check if it's after startOp
          // Skip if: no startOp, same as startOp, already has insertion point, or in same block and before startOp
          if (!startOp || user == startOp || insertionPoints.contains(user)) {
            LLVM_DEBUG(llvm::dbgs() << "Skipping readOp because it's already been used: " << *readOp << "\n");
            continue;
          }
          // If in the same block, skip if before startOp
          if (user->getBlock() == startOp->getBlock() && user->isBeforeInBlock(startOp)) {
            LLVM_DEBUG(llvm::dbgs() << "Skipping readOp because it's before startOp: " << *readOp << "\n");
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << "Found something!\n");
          // Found a read after startOp, track the first one in program order
          if (!firstReadOp) {
            firstRead = readOp;
            firstReadOp = user;
          } else if (user->getBlock() == firstReadOp->getBlock()) {
            // Same block: use isBeforeInBlock for ordering
            if (user->isBeforeInBlock(firstReadOp)) {
              firstRead = readOp;
              firstReadOp = user;
            }
          } 
          else {
            // Different blocks: check if one is nested inside the other
            Block *userBlock = user->getBlock();
            Block *firstReadBlock = firstReadOp->getBlock();
            
            // Check if firstReadBlock is nested inside userBlock's region
            Operation *firstReadParentOp = firstReadBlock->getParentOp();
            if (firstReadParentOp && firstReadParentOp->getBlock() == userBlock) {
              // firstReadBlock is nested in userBlock, compare firstReadParentOp with user
              if (firstReadParentOp->isBeforeInBlock(user)) {
                // firstReadParentOp comes before user, so firstReadOp comes first
                // Don't update, keep current firstRead
              } else {
                // user comes before firstReadParentOp, so user comes first
                firstRead = readOp;
                firstReadOp = user;
              }
            }
            // Check if userBlock is nested inside firstReadBlock's region
            else if (Operation *userParentOp = userBlock->getParentOp()) {
              if (userParentOp->getBlock() == firstReadBlock) {
                // userBlock is nested in firstReadBlock, compare userParentOp with firstReadOp
                if (userParentOp->isBeforeInBlock(firstReadOp)) {
                  // userParentOp comes before firstReadOp, so user comes first
                  firstRead = readOp;
                  firstReadOp = user;
                }
                // Otherwise firstReadOp comes first, don't update
              }
            }
            // If neither is nested in the other, don't update firstRead
          }
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
      // If this is a transform operation, follow its result
      else if (auto transformOp = dyn_cast<rock::TransformOp>(user)) {
        addToWorklist(transformOp.getResult());
      }
    }
  }

  return firstRead;
}

/// Find the first use after a ThreadwiseReadIntoOp that writes to LDS.
/// The function traces back the dest value to the alloc, then finds the first
/// ThreadwiseReadIntoOp that reads from that alloc.
static Operation* findFirstUseAfter(ThreadwiseReadIntoOp writeOp, llvm::DenseSet<Operation*> &insertionPoints) {
  // Get the destination value (what is written to)
  Value dest = writeOp.getDest();

  // Trace back to find all possible allocs
  SmallVector<rock::GpuAllocOp> allocs = traceToAllocs(dest);
  if (allocs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to trace dest to alloc\n");
    return nullptr;
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << allocs.size() << " alloc(s)\n");

  // Find the first read from LDS that uses any of these allocs
  ThreadwiseReadIntoOp firstRead = nullptr;
  Operation *firstReadOp = nullptr;

  for (auto alloc : allocs) {
    LLVM_DEBUG(llvm::dbgs() << "Checking alloc: " << alloc << "\n");
    ThreadwiseReadIntoOp read = findFirstReadFromLDS(alloc.getResult(), writeOp.getOperation(), insertionPoints);
    
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
    return nullptr;
  }

  LLVM_DEBUG(llvm::dbgs() << "After writeOp: " << *writeOp.getOperation() << "\n");
  LLVM_DEBUG(llvm::dbgs() << "->-> Found first read: " << firstRead << "\n");
  return firstRead.getOperation();
}

/// Check if a ThreadwiseReadIntoOp reads from global memory (no GPU address space).
static bool isGlobalLoad(ThreadwiseReadIntoOp readOp) {
  auto sourceMemSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(
      readOp.getSource().getType().getMemorySpace());
  // Global memory has no address space attribute (or it's null)
  return !sourceMemSpace;
}

/// Count global loads (ThreadwiseReadIntoOp from global to LDS) between two operations in a block.
/// Counts operations from startOp (exclusive) to endOp (exclusive).
/// If startOp is nullptr, counts from the beginning of the block.
static int countGlobalLoadsBetween(Operation *startOp, Operation *endOp, Block *block) {
  int count = 0;
  bool counting = (startOp == nullptr); // If startOp is nullptr, start counting immediately
  
  for (Operation &op : *block) {
    if (startOp && &op == startOp) {
      counting = true;
      continue; // Start counting after startOp
    }
    if (endOp && &op == endOp) {
      break; // Stop counting at endOp
    }
    if (counting) {
      if (auto readOp = dyn_cast<ThreadwiseReadIntoOp>(&op)) {
        // Check if it reads from global memory
        if (!isGlobalLoad(readOp))
          continue;
        // Check if it writes to LDS
        auto destMemSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(
            readOp.getDest().getType().getMemorySpace());
        if (destMemSpace && destMemSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace()) {
          auto maybeLoopCount = rock::predictThreadwiseReadIntoLoopCount(readOp);
          if (failed(maybeLoopCount)) {
            LLVM_DEBUG(llvm::dbgs() << "Failed to predict loop count for ThreadwiseReadIntoOp: " << *readOp << "\n");
            count++;
          }
          else {
            count += maybeLoopCount.value();
          }
        }
      }
    }
  }
  
  return count;
}

/// The localLoadOp is the load that triggers the dependency, and the
/// globalLoadOp is the load that is dependent on the localLoadOp.
std::pair<int, bool> getWaitCount(Operation *localLoadOp, Operation *globalLoadOp) {  
  if (!localLoadOp || !globalLoadOp)
    return {-1, false};

  auto localReadOp = dyn_cast<ThreadwiseReadIntoOp>(localLoadOp);
  auto globalReadOp = dyn_cast<ThreadwiseReadIntoOp>(globalLoadOp);
  if (!localReadOp || !globalReadOp)
    return {-1, false};

  // Get parent blocks and check for loops
  Block *globalBlock = globalLoadOp->getBlock();
  
  scf::ForOp localLoop = localLoadOp->getParentOfType<scf::ForOp>();
  scf::ForOp globalLoop = globalLoadOp->getParentOfType<scf::ForOp>();
  
  int waitCount = 0;
  bool pipeliningEnabled = false;

  // Case 1: Both have the same parent block (function) - prologue
  if (!localLoop && !globalLoop) {
    LLVM_DEBUG(llvm::dbgs() << "Case 1: Both in function (prologue)\n");
    // Count global loads between globalLoadOp and localLoadOp in the same block
    waitCount = countGlobalLoadsBetween(globalLoadOp, localLoadOp, globalBlock);
    pipeliningEnabled = true;
  }
  // Case 2: localLoad in loop, globalLoad in function - body
  else if (localLoop && !globalLoop) {
    LLVM_DEBUG(llvm::dbgs() << "Case 2: localLoad in loop, globalLoad in function (body)\n");
    // Count from globalLoadOp to the loop operation (which marks the start of the loop block)
    Block *loopOpBlock = localLoop->getBlock();
    if (loopOpBlock == globalBlock) {
      waitCount += countGlobalLoadsBetween(globalLoadOp, localLoop.getOperation(), globalBlock);
    }
    // Note: If loopOpBlock != globalBlock, we'd need to handle cross-block counting,
    // but in typical cases they should be in the same block
    
    // Count from the start of the loop body block to localLoadOp
    Block *loopBody = localLoop.getBody();
    waitCount += countGlobalLoadsBetween(nullptr, localLoadOp, loopBody);
    pipeliningEnabled = true;
  }
  // Case 3: localLoad in function, globalLoad in loop - epilogue
  else if (!localLoop && globalLoop) {
    LLVM_DEBUG(llvm::dbgs() << "Case 3: localLoad in function, globalLoad in loop (epilogue)\n");
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

static Operation* getInsertionPointForAsyncWait(IRRewriter &rewriter, Operation *op, bool pipeliningEnabled) {
  if (pipeliningEnabled) {
    // If pipelining is enabled, we always insert the AsyncWaitOp just before the local load.
    return op;
  } else {
    // If pipelining is disabled, we have to insert the AsyncWaitOp just after the LDSBarrierOp.
    // Find the first LDSBarrierOp with barrier_stage="backward" in the function
    func::FuncOp func = op->getParentOfType<func::FuncOp>();
    if (!func) {
      LLVM_DEBUG(llvm::dbgs() << "No function found, inserting before op\n");
      return nullptr;
    }
    
    rock::LDSBarrierOp forwardBarrier = nullptr;
    func.walk([&](rock::LDSBarrierOp barrier) {
      if (!forwardBarrier) {
        auto barrierStageAttr = barrier.getBarrierStageAttr();
        if (barrierStageAttr && barrierStageAttr.getValue() == rock::BarrierStage::Forward) {
          forwardBarrier = barrier;
        }
      }
    });
    
    if (forwardBarrier) {
      llvm::dbgs() << "Found forward barrier: " << forwardBarrier << "\n";
      return forwardBarrier;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "No forward barrier found, inserting before op\n");
      return op;
    }
  }
}

/// Add AsyncWaitOps to the function. This function has two main steps:
/// 1. Find all ThreadwiseReadIntoOp operations that write into LDS memory.
///    Then, for each of them, call findFirstUseAfter, which will find the first op
///    that uses the result of the ThreadwiseReadIntoOp.
/// 2. Once we know where to insert the AsyncWaitOp, call getWaitCount, which will
///    return the number of AsyncWaitOps to insert.
static LogicalResult addAsyncWait(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());

  // Find all ThreadwiseReadIntoOp operations
  SmallVector<rock::ThreadwiseReadIntoOp> readOps;
  func.walk([&](rock::ThreadwiseReadIntoOp op) {
    // Only add reads that write into LDS memory
    auto memSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(op.getDest().getType().getMemorySpace());
    if (memSpace && memSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      readOps.push_back(op);
    }    
  });

  // Track insertion points to avoid inserting multiple AsyncWaitOps at the same location
  llvm::DenseSet<Operation*> insertionPoints;
  for (auto readOp : readOps) {
    Operation* firstUse = findFirstUseAfter(readOp, insertionPoints);
    
    if (!firstUse) {
      // If pattern doesn't match or no use found, skip this readOp
      LLVM_DEBUG(llvm::dbgs() << "Pattern doesn't match or no use found for ThreadwiseReadIntoOp\n");
      continue;
    }
    
    auto [waitCount, pipeliningEnabled] = getWaitCount(firstUse, readOp);
    if (waitCount == -1) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to get wait count for ThreadwiseReadIntoOp\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Looking for insertion point for AsyncWaitOp after " << firstUse << "\n");

    Operation* insertionPoint = getInsertionPointForAsyncWait(rewriter, firstUse, pipeliningEnabled);      
    
    // Only insert one AsyncWaitOp per insertion point
    if (insertionPoints.contains(insertionPoint)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping insertion point because it already has an AsyncWaitOp\n");
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
