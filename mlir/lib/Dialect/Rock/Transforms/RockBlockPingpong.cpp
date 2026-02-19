//===- RockBlockPingpong.cpp - Block ping-pong scheduling ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Block Ping-Pong Scheduling Pass
// ================================
//
// This pass applies block ping-pong scheduling to improve compute/memory
// overlap in GEMM kernels. Enable via rock.use_block_pingpong func attr or
// ROCMLIR_ENABLE_BLOCK_PINGPONG=1.
//
// IMPLEMENTATION MODES:
//
// 1. FULL PING-PONG MODE (with triple buffering):
//    When rock.triple_buffered attr is present, applies:
//    - M-loop slicing: splits M-loop into 2 halves with s_barrier between
//    - Phase shift: cond_barrier(warpHigh) before loop, cond_barrier(warpLow)
//      after loop
//    - Cluster boundaries: sched_barrier around existing barriers
//    - SetPrio: setprio(1) before MFMA, setprio(0) after
//
// 2. SCHEDULING HINTS ONLY (with double buffering):
//    When only rock.double_buffered attr is present:
//    - sched_barrier around existing LDS barriers (cluster boundaries)
//    - setprio(1) before MFMA clusters, setprio(0) after
//
// IMPORTANT: TRUE COMPUTE/MEMORY OVERLAP IS NOT ACHIEVED
//
// Despite the structural elements (phase shift, M-loop slicing, triple
// buffering), the trace shows a WATERFALL pattern, not true ping-pong.
//
// ROOT CAUSE: M-loop slicing creates 2 COMPUTE (MFMA) clusters. When barriers
// release, both wave groups compete for the same MFMA hardware units and
// serialize.
//
// TRUE PING-PONG REQUIRES: Interleaved COMPUTE + MEMORY clusters (different
// hardware). Triton achieves this by alternating dot(MFMA) and memory
// clusters. rocMLIR would need to restructure RockPipeline.cpp to produce:
//   [MFMA ops] → s_barrier → [Memory ops (DSR, DSW, GL)]
//
// See BLOCK_PINGPONG_DESIGN.md for detailed analysis and recommendations.
//
// WHY TRIPLE BUFFERING IS REQUIRED FOR PHASE SHIFT:
//
// With phase shift, waves are at different main loop iterations:
//   - Group 0 at iter N: reads buffer[N%B]
//   - Group 1 at iter N-1: reads buffer[(N-1)%B]
//
// With B=2 (double buffering): buffer[N%2] == buffer[(N-2)%2] → CONFLICT
// With B=3 (triple buffering): buffer[N%3] ≠ buffer[(N-1)%3] → SAFE
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Debug.h"

#include <cstdlib>

#define DEBUG_TYPE "rock-block-pingpong"

namespace mlir {
namespace rock {

#define GEN_PASS_DEF_ROCKBLOCKPINGPONGPASS
#include "mlir/Dialect/Rock/Passes.h.inc"

} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

//===----------------------------------------------------------------------===//
// Eligibility and Configuration
//===----------------------------------------------------------------------===//

/// Returns true if block ping-pong is enabled for this function (attr or env).
static bool isBlockPingpongEnabled(func::FuncOp func) {
  if (func->getAttr("rock.use_block_pingpong"))
    return true;
  const char *env = std::getenv("ROCMLIR_ENABLE_BLOCK_PINGPONG");
  return env && (std::atoi(env) != 0);
}

/// Returns block_size from func attr, or -1 if not set.
static int64_t getBlockSize(func::FuncOp func) {
  auto attr = func->getAttrOfType<IntegerAttr>("block_size");
  return attr ? attr.getInt() : -1;
}

/// Returns grid_size from func attr, or -1 if not set.
static int64_t getGridSize(func::FuncOp func) {
  auto attr = func->getAttrOfType<IntegerAttr>("grid_size");
  return attr ? attr.getInt() : -1;
}

//===----------------------------------------------------------------------===//
// Loop Detection
//===----------------------------------------------------------------------===//

/// Finds the first (outermost) scf.for in the func that contains an
/// rock.lds_barrier (pipelined kernel loop).
static scf::ForOp findPipelinedLoop(func::FuncOp func) {
  scf::ForOp found;
  func.walk([&](scf::ForOp forOp) {
    if (found)
      return WalkResult::skip();
    bool hasBarrier = false;
    for (Operation &op : forOp.getBody()->getOperations()) {
      if (isa<rock::LDSBarrierOp>(op)) {
        hasBarrier = true;
        break;
      }
      if (op.getNumRegions() > 0) {
        op.walk([&](rock::LDSBarrierOp) { hasBarrier = true; });
        if (hasBarrier)
          break;
      }
    }
    if (hasBarrier) {
      found = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Cluster Boundaries (sched_barrier around existing barriers)
//===----------------------------------------------------------------------===//

/// Adds sched_barrier around existing barriers without replacing them.
/// Creates cluster boundaries for better instruction scheduling.
static void insertClusterBoundariesAtExistingBarriers(scf::ForOp forOp) {
  OpBuilder b(forOp.getContext());
  SmallVector<rock::LDSBarrierOp> barriers;

  // Collect all existing barriers first to avoid iterator invalidation.
  forOp.getBody()->walk(
      [&](rock::LDSBarrierOp barrier) { barriers.push_back(barrier); });

  // Insert sched_barrier BEFORE and AFTER each existing LDS barrier.
  // Before: prevents scheduler from moving vmcnt before MFMAs
  // After: creates boundary between barrier and next compute
  for (auto barrier : barriers) {
    // Insert sched_barrier BEFORE the LDS barrier.
    // This creates a boundary: [compute cluster] | sched_barrier | lds_barrier
    b.setInsertionPoint(barrier);
    amdgpu::SchedBarrierOp::create(b, barrier.getLoc(),
                                   amdgpu::sched_barrier_opt_enum::none);

    // Insert sched_barrier AFTER the LDS barrier.
    // This creates a boundary: lds_barrier | sched_barrier | [memory cluster]
    b.setInsertionPointAfter(barrier);
    amdgpu::SchedBarrierOp::create(b, barrier.getLoc(),
                                   amdgpu::sched_barrier_opt_enum::none);
  }
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: inserted "
                          << (barriers.size() * 2)
                          << " cluster boundaries (sched_barrier)\n");
}

//===----------------------------------------------------------------------===//
// SetPrio (optional, for dot clusters)
//===----------------------------------------------------------------------===//

/// Returns true if the operation is an MFMA-like compute operation.
static bool isMFMAOp(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName.contains("accel") || opName.contains("mfma") ||
         opName.contains("wmma");
}

/// Returns true if the operation or any of its nested ops contains an MFMA.
static bool containsMFMA(Operation *op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    if (isMFMAOp(nested)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Finds the outermost affine.for (or similar loop) that contains MFMA ops.
/// We want to wrap the entire loop nest with setprio, not individual MFMAs,
/// so that after loop unrolling, all MFMAs in the cluster are covered.
static SmallVector<Operation *> findMFMAContainingLoops(Operation *parentOp) {
  SmallVector<Operation *> result;

  // Look for top-level operations in the parent that contain MFMAs.
  // These could be affine.for loops or direct MFMA ops.
  for (Region &region : parentOp->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (isMFMAOp(&op)) {
          // Direct MFMA op
          result.push_back(&op);
        } else if (isa<affine::AffineForOp>(&op) && containsMFMA(&op)) {
          // affine.for containing MFMAs - wrap the whole loop
          result.push_back(&op);
        }
      }
    }
  }

  return result;
}

/// Wraps MFMA-containing operations (either affine.for loops containing MFMAs,
/// or direct MFMA ops) with setprio(1) before and setprio(0) after.
/// The key insight is to wrap at the LOOP level, not the individual MFMA level,
/// so that after loop unrolling, the entire cluster is covered by one setprio
/// pair.
static void insertSetPrioAroundMFMA(scf::ForOp forOp) {
  OpBuilder b(forOp.getContext());
  int mfmaContainerCount = 0;

  // Find MFMA-containing ops in the scf.for loop body.
  auto mfmaContainers = findMFMAContainingLoops(forOp);

  for (Operation *container : mfmaContainers) {
    Location loc = container->getLoc();

    // Insert setprio(1) before the container (loop or MFMA).
    b.setInsertionPoint(container);
    ROCDL::SetPrioOp::create(b, loc, static_cast<int16_t>(1));

    // Insert setprio(0) after the container.
    b.setInsertionPointAfter(container);
    ROCDL::SetPrioOp::create(b, loc, static_cast<int16_t>(0));

    mfmaContainerCount++;
  }

  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE "]: wrapped " << mfmaContainerCount
                   << " MFMA-containing ops in main loop with setprio\n");
}

/// Wraps epilogue MFMA operations (after the main pipelined loop) with setprio.
/// The epilogue contains the final MFMA clusters that drain the pipeline.
static void insertSetPrioAroundEpilogue(scf::ForOp forOp) {
  OpBuilder b(forOp.getContext());
  int epilogueCount = 0;

  // Get the block containing the scf.for.
  Block *block = forOp->getBlock();

  // Iterate over operations after the scf.for loop.
  bool afterLoop = false;
  SmallVector<Operation *> epilogueContainers;

  for (Operation &op : *block) {
    if (&op == forOp.getOperation()) {
      afterLoop = true;
      continue;
    }
    if (!afterLoop)
      continue;

    // Check if this is an MFMA-containing op or loop.
    if (isMFMAOp(&op)) {
      epilogueContainers.push_back(&op);
    } else if (isa<affine::AffineForOp>(&op) && containsMFMA(&op)) {
      epilogueContainers.push_back(&op);
    }
  }

  for (Operation *container : epilogueContainers) {
    Location loc = container->getLoc();

    // Insert setprio(1) before the container.
    b.setInsertionPoint(container);
    ROCDL::SetPrioOp::create(b, loc, static_cast<int16_t>(1));

    // Insert setprio(0) after the container.
    b.setInsertionPointAfter(container);
    ROCDL::SetPrioOp::create(b, loc, static_cast<int16_t>(0));

    epilogueCount++;
  }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: wrapped " << epilogueCount
                          << " epilogue MFMA-containing ops with setprio\n");
}

//===----------------------------------------------------------------------===//
// Phase Shift (Triton-style: cond_barrier before/after loop only)
//===----------------------------------------------------------------------===//
//
// NOTE: These functions are IMPLEMENTED but NOT CURRENTLY USED.
//
// The phase shift approach (Option A) does not work with rocMLIR's current
// pipelining structure because:
// 1. rocMLIR peels prologue/epilogue stages OUTSIDE the scf.for loop
// 2. Triton keeps all stages INSIDE the loop
// 3. Adding cond_barriers at scf.for boundaries causes wave desync
//
// The functions are kept here for future use when we implement Option A'
// (restructured pipelining) or Option B (single-barrier compute-first).
//
// See BLOCK_PINGPONG_DESIGN.md for detailed analysis.
//===----------------------------------------------------------------------===//

/// Creates warp group predicates for 8-wave ping-pong following Triton's
/// approach. We split the 8 waves into 2 groups of 4 waves each:
///   - Group 0: threads 0-255 (waves 0-3)   → warpGroupID = 0 (warpLow)
///   - Group 1: threads 256-511 (waves 4-7) → warpGroupID = 1 (warpHigh)
///
/// Returns (warpHigh, warpLow) predicates.
static std::pair<Value, Value>
createWaveGroupPredicates(OpBuilder &b, Location loc, int64_t blockSize) {
  // For 512 threads (8 waves), divide by 256 to get 2 warp groups
  int64_t halfBlockSize = blockSize / 2;

  Value tid = rock::WorkitemIdOp::create(b, loc, b.getIndexType());
  Value tidI32 = arith::IndexCastOp::create(b, loc, b.getI32Type(), tid);
  Value halfBlockVal = arith::ConstantOp::create(
      b, loc, b.getI32Type(), b.getI32IntegerAttr(halfBlockSize));
  Value warpGroupID = arith::DivUIOp::create(b, loc, tidI32, halfBlockVal);
  Value cst0 =
      arith::ConstantOp::create(b, loc, b.getI32Type(), b.getI32IntegerAttr(0));

  // warpLow = (warpGroupID == 0) - threads 0-255, waves 0-3
  Value warpLow = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                        warpGroupID, cst0);

  // warpHigh = (warpGroupID != 0) - threads 256-511, waves 4-7
  Value warpHigh = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                         warpGroupID, cst0);

  return {warpHigh, warpLow};
}

/// Implements Triton-style dot slicing by splitting the mRepeats loop.
///
/// Triton achieves ping-pong by slicing the dot operation along K, creating
/// 2 compute clusters per iteration with barriers between them. In rocMLIR,
/// the equivalent is to split the mRepeats loop into 2 halves:
///
/// Original (mRepeats=4):
///   affine.for m = 0 to 4:
///     affine.for n = 0 to 2:
///       affine.for k = 0 to 1:
///         threadwise_gemm_accel
///
/// After dot slicing:
///   // Cluster 0: first half of M repeats
///   affine.for m = 0 to 2:
///     affine.for n = 0 to 2:
///       affine.for k = 0 to 1:
///         threadwise_gemm_accel
///   s_barrier  ← NEW: cluster boundary
///   // Cluster 1: second half of M repeats
///   affine.for m = 2 to 4:
///     affine.for n = 0 to 2:
///       affine.for k = 0 to 1:
///         threadwise_gemm_accel
///
/// With 2 barriers per iteration, phase shift now works:
/// - Group 0 at Barrier0 (iter N) + Group 1 at Barrier1 (iter N-1) = 8 waves
/// - Group 0 does cluster0 (first M half), Group 1 does cluster1 (second M
/// half)
/// - Different data = no conflict!
///
/// Returns true if slicing was applied successfully.
static bool applyDotSlicing(scf::ForOp forOp) {
  // Find affine.for loops with threadwise_gemm_accel inside the main loop
  SmallVector<affine::AffineForOp> mLoops;

  forOp.walk([&](affine::AffineForOp affineFor) {
    // Check if this loop directly contains another affine.for (making it the
    // outer M loop)
    bool containsMFMA = false;
    affineFor.walk([&](rock::ThreadwiseGemmAccelOp) { containsMFMA = true; });

    if (!containsMFMA)
      return;

    // Check if this is the outermost affine.for containing MFMA
    // (it should have a parent that is NOT an affine.for)
    Operation *parent = affineFor->getParentOp();
    if (!isa<affine::AffineForOp>(parent)) {
      mLoops.push_back(affineFor);
    }
  });

  if (mLoops.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "]: no M loops found for dot slicing\n");
    return false;
  }

  bool anySliced = false;
  OpBuilder b(forOp.getContext());

  for (affine::AffineForOp mLoop : mLoops) {
    // Get loop bounds
    int64_t lb = mLoop.getConstantLowerBound();
    int64_t ub = mLoop.getConstantUpperBound();
    int64_t step = mLoop.getStepAsInt();

    int64_t numIters = (ub - lb) / step;

    // Need at least 2 iterations to split
    if (numIters < 2) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: mLoop has only " << numIters
                              << " iterations, cannot slice\n");
      continue;
    }

    int64_t midpoint = lb + (numIters / 2) * step;

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "]: slicing mLoop: [" << lb << ", " << ub
               << ") step " << step << " at " << midpoint << "\n");

    // Clone the loop for the second half
    b.setInsertionPointAfter(mLoop);

    // Create second half loop: [midpoint, ub)
    auto secondHalf = cast<affine::AffineForOp>(b.clone(*mLoop));
    secondHalf.setConstantLowerBound(midpoint);

    // Insert barrier between the two halves
    b.setInsertionPoint(secondHalf);
    rock::SBarrierOp::create(b, mLoop.getLoc());

    // Modify original loop to be first half: [lb, midpoint)
    mLoop.setConstantUpperBound(midpoint);

    anySliced = true;
  }

  if (anySliced) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: applied dot slicing to "
                            << mLoops.size() << " M loops\n");
  }

  return anySliced;
}

/// Replaces rock.lds_barrier inside the loop with rock.s_barrier.
///
/// For ping-pong scheduling to work, the barriers INSIDE the loop must be
/// control-only (s_barrier) without the LDS wait (lgkmcnt). This is because:
///
/// 1. With phase shift, waves are at different iterations concurrently
/// 2. Waves at iteration N are writing to buffer[(N+offset)%2]
/// 3. Waves at iteration N-1 are reading from a different buffer
/// 4. If we wait for LDS (lgkmcnt=0), we block the pipelining
///
/// The LDS wait is NOT needed inside the loop because:
/// - The double-buffering ensures different iterations use different buffers
/// - The barrier synchronization ensures all waves in an iteration complete
///   before the next iteration reads from that buffer
///
/// The LDS wait IS still needed at:
/// - The last barrier before reading from LDS in the prologue
/// - The epilogue barriers
static void replaceLoopBarriersWithSBarrier(scf::ForOp forOp) {
  SmallVector<rock::LDSBarrierOp> toReplace;

  // Collect all lds_barrier ops inside the loop
  forOp.walk([&](rock::LDSBarrierOp op) { toReplace.push_back(op); });

  // Replace each lds_barrier with s_barrier
  for (rock::LDSBarrierOp op : toReplace) {
    OpBuilder b(op);
    rock::SBarrierOp::create(b, op.getLoc());
    op.erase();
  }

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: replaced " << toReplace.size()
                          << " lds_barrier with s_barrier inside loop\n");
}

/// Applies Triton-style phase shift: cond_barrier ONLY before/after loop.
///
/// This approach uses control-only barriers (s_barrier) inside the loop, with
/// phase shift created by cond_barriers at loop boundaries:
///
///   1. Pre-loop: full barrier, then cond_barrier(warpHigh)
///      - All 8 waves sync at full barrier
///      - Waves 4-7 (warpHigh) hit cond_barrier and wait
///      - Waves 0-3 (warpLow) skip cond_barrier and enter loop
///   2. Loop: s_barrier (no LDS wait) pairs waves at different iterations
///      - When waves 0-3 hit first barrier in loop iteration 0,
///        waves 4-7 exit cond_barrier (4+4=8 waves, barrier releases)
///      - Now waves 4-7 are at iteration 0, waves 0-3 at iteration 1
///   3. Post-loop: cond_barrier(warpLow) to reconverge
///      - Waves 0-3 wait while waves 4-7 finish their last iteration
///
/// CRITICAL: The barriers inside the loop MUST be s_barrier (not lds_barrier)
/// because with phase shift, waves are at different iterations and we don't
/// want to wait for LDS operations that haven't completed yet.
///
/// With double buffering, waves at different iterations access different
/// buffers (iter % 2), so there's no read-write conflict.
///
/// PREREQUISITE: The loop must have 2+ barriers per iteration BEFORE calling
/// this function. This can be achieved via:
///   - Dot slicing (applyDotSlicing): splits M loop, inserts s_barrier between
///   - Loop unrolling: creates 2 copies of loop body with barriers
///
/// With 2 barriers per iteration, the phase shift creates true ping-pong.
static void applyTritonStylePhaseShift(OpBuilder &b, Location loc,
                                       scf::ForOp forOp, int64_t blockSize) {
  // CRITICAL: Replace lds_barrier inside loop with s_barrier FIRST
  // This ensures the loop barriers don't wait for LDS operations
  replaceLoopBarriersWithSBarrier(forOp);

  // Insert predicates and pre-loop barriers BEFORE the loop
  b.setInsertionPoint(forOp);

  auto [warpHigh, warpLow] = createWaveGroupPredicates(b, loc, blockSize);

  // Pre-loop: full barrier (with LDS wait) to sync all waves before phase shift
  rock::LDSBarrierOp::create(b, loc);

  // Pre-loop: cond_barrier(warpHigh) - waves 4-7 wait here
  // When waves 0-3 hit the first barrier inside the loop, both groups
  // (4 from cond_barrier + 4 from loop barrier = 8) release together.
  // This creates the phase shift: waves 0-3 proceed into loop, waves 4-7
  // follow.
  rock::CondBarrierOp::create(b, loc, warpHigh);

  // Post-loop: cond_barrier(warpLow) - waves 0-3 wait for waves 4-7 to finish
  b.setInsertionPointAfter(forOp);
  rock::CondBarrierOp::create(b, loc, warpLow);

  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE
                      "]: applied Triton-style phase shift (cond_barriers "
                      "before/after loop, s_barrier inside loop)\n");
}

//===----------------------------------------------------------------------===//
// Main Pass
//===----------------------------------------------------------------------===//

struct RockBlockPingpong
    : public rock::impl::RockBlockPingpongPassBase<RockBlockPingpong> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!isBlockPingpongEnabled(func)) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: disabled (no attr/env)\n");
      return;
    }

    StringAttr archAttr = rock::getArchValue(func);
    if (!archAttr) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: no arch on func, skip\n");
      return;
    }
    rock::AmdArchInfo archInfo = rock::lookupArchInfo(archAttr.getValue());
    int64_t waveSize = archInfo.waveSize;
    int64_t minNumCU = archInfo.minNumCU;

    int64_t blockSize = getBlockSize(func);
    if (blockSize <= 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "]: skip (no block_size attr)\n");
      return;
    }
    int64_t wavesPerBlock = blockSize / waveSize;

    scf::ForOp forOp = findPipelinedLoop(func);
    if (!forOp) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: no pipelined loop found\n");
      return;
    }

    // Check buffering mode
    bool isDoubleBuffered = func->hasAttr("rock.double_buffered");

    // OpBuilder for phase shift insertion
    OpBuilder b(func.getContext());

    // Mode selection based on waves per block.
    if (wavesPerBlock == 8) {
      // 8-wave mode with double-buffering: Apply Triton-style phase shift.
      //
      // Triton-style approach:
      //   - Keep FULL barriers inside the loop (all 8 waves sync)
      //   - Add cond_barrier ONLY before/after loop for phase shift
      //   - Full barriers inside loop "pair up" waves at different iterations
      //   - With double buffering (2 LDS buffers), waves at different
      //   iterations
      //     access different buffers (iter % 2), avoiding conflicts
      //
      // This works because:
      //   - At each full barrier: 4 waves from iter N + 4 waves from iter N+1 =
      //   8
      //   - s_barrier counter hits 8, all waves release
      //   - Both groups proceed, maintaining the 1-iteration offset
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "]: applying 8-wave mode"
                 << (isDoubleBuffered ? " with phase shift (double-buffered)"
                                      : " (scheduling hints only)")
                 << "\n");

      // Check if we have triple buffering for full ping-pong support
      bool isTripleBuffered = func->hasAttr("rock.triple_buffered");

      if (isTripleBuffered) {
        // FULL PING-PONG MODE with triple buffering
        //
        // With triple buffering, the iteration offset conflict is resolved:
        // - Group 0 at iter N: reads buffer[N%3]
        // - Group 1 at iter N-1: reads buffer[(N-1)%3]
        // - buffer[N%3] != buffer[(N-1)%3] (always different with 3 buffers)
        //
        // Implementation:
        // 1. Apply dot slicing: split M-loop into 2 halves with s_barrier
        // 2. Apply phase shift: cond_barrier before/after loop

        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]: triple-buffered mode, applying "
                      "full ping-pong with dot slicing + phase shift\n");

        // Step 1: Apply dot slicing - creates 2 compute clusters per iteration
        bool sliced = applyDotSlicing(forOp);
        if (!sliced) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[" DEBUG_TYPE
                        "]: dot slicing failed, falling back to hints only\n");
          // Fall through to scheduling hints
        } else {
          // Step 2: Apply phase shift with cond_barriers
          Location loc = forOp.getLoc();
          applyTritonStylePhaseShift(b, loc, forOp, blockSize);

          // Step 3: Cluster boundaries around the newly inserted barriers
          insertClusterBoundariesAtExistingBarriers(forOp);

          // Step 4: SetPrio around MFMA operations
          insertSetPrioAroundMFMA(forOp);
          insertSetPrioAroundEpilogue(forOp);

          LLVM_DEBUG(llvm::dbgs()
                     << "[" DEBUG_TYPE "]: applied full ping-pong mode\n");
          return;
        }
      }

      // SCHEDULING HINTS ONLY (double-buffered or fallback)
      //
      // Without triple buffering, we can't do phase shift safely because:
      // - Group 0 at K-iter N: reads LDS buffer[N%2], writes buffer[(N+1)%2]
      // - Group 1 at K-iter N-1: reads buffer[(N-1)%2], writes buffer[N%2]
      // - Result: One group reads what the other writes = CONFLICT
      //
      // Use scheduling hints for modest improvements instead.

      // Cluster boundaries: sched_barrier around existing LDS barriers.
      insertClusterBoundariesAtExistingBarriers(forOp);

      // SetPrio around MFMA operations in main loop.
      insertSetPrioAroundMFMA(forOp);

      // SetPrio around epilogue MFMA operations.
      insertSetPrioAroundEpilogue(forOp);

      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                                 "]: applied 8-wave mode (scheduling hints)\n");
    } else if (wavesPerBlock == 4) {
      // 4-wave mode: cluster-only (no phase shift).
      // Overlap comes from 2 workgroups per CU interleaving.
      int64_t gridSize = getGridSize(func);
      bool assume2WgPerCU = func->hasAttr("rock.assume_2_wg_per_cu");

      // Check if we expect 2+ workgroups per CU.
      if (!assume2WgPerCU && gridSize > 0 && minNumCU > 0 &&
          gridSize < 2 * minNumCU) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "]: 4-wave mode but gridSize=" << gridSize
                   << " < 2*minNumCU=" << (2 * minNumCU)
                   << ", insufficient occupancy, skip\n");
        return;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "]: applying 4-wave cluster-only mode\n");

      // No phase shift for 4-wave mode - workgroups cannot sync with each
      // other. Overlap happens naturally when GPU schedules 2 blocks on same
      // CU.

      // Cluster boundaries: sched_barrier at existing LDS barriers.
      insertClusterBoundariesAtExistingBarriers(forOp);

      // SetPrio around MFMA operations in main loop.
      insertSetPrioAroundMFMA(forOp);

      // SetPrio around epilogue MFMA operations.
      insertSetPrioAroundEpilogue(forOp);

      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "]: applied 4-wave cluster-only mode\n");

    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "]: waves_per_block=" << wavesPerBlock
                 << " (only 4 or 8 supported), skip\n");
      return;
    }
  }
};

} // namespace
