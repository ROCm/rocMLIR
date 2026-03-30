//===- AddSchedGroupBarriers.cpp - Add scheduling group barriers ----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass analyzes scf.for loops and inserts scheduling group barriers
// (ROCDL::SchedGroupBarrier) to optimize instruction scheduling on AMD GPUs.
//
// The pass skips functions that contain nested scf.for loops.
// For the remaining single-loop functions, barriers are only inserted if:
// - The loop uses double buffering (LDS reads/writes use arith.select).
// - The loop does not use direct-to-LDS loads (amdgpu.gather_to_lds or
//   amdgpu.async_load_to_lds).
// - The loop has at most one rock.lds_barrier (excludes attention kernels).
// - The loop contains at least one global load and one matrix multiply op.
// - The number of matrix multiply ops per iteration does not exceed 25.
// - The loop body has no scf.if (mutually exclusive branches make
//   instruction counts unreliable).
// - Scheduling barriers are not already present in the loop body.
//
// The counts factor in affine.for loop trip counts.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "rock-add-sched-group-barriers"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKADDSCHEDGROUPBARRIERSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Check if a memref type has workgroup (LDS) address space
static bool hasWorkgroupAddressSpace(MemRefType memrefType) {
  auto addrSpace = memrefType.getMemorySpace();
  if (!addrSpace)
    return false;

  // Check for gpu.address_space<workgroup>
  if (auto gpuAddrSpace = dyn_cast<gpu::AddressSpaceAttr>(addrSpace)) {
    return gpuAddrSpace.getValue() == gpu::AddressSpace::Workgroup;
  }
  return false;
}

/// Check if a memref type has global address space
static bool hasGlobalAddressSpace(MemRefType memrefType) {
  auto addrSpace = memrefType.getMemorySpace();
  // No address space means global by default
  if (!addrSpace)
    return true;

  // Check for gpu.address_space<global>
  if (auto gpuAddrSpace = dyn_cast<gpu::AddressSpaceAttr>(addrSpace)) {
    return gpuAddrSpace.getValue() == gpu::AddressSpace::Global;
  }
  return false;
}

/// Check if a value is defined by an arith.select operation, which indicates
/// double buffering (selecting between two different LDS buffers)
static bool isDefinedBySelect(Value val) {
  return val.getDefiningOp<arith::SelectOp>() != nullptr;
}

/// Get the trip count of an affine.for loop, returns 1 if unknown
static uint64_t getAffineForTripCount(affine::AffineForOp affineFor) {
  std::optional<uint64_t> tripCount = affine::getConstantTripCount(affineFor);
  if (tripCount.has_value()) {
    return tripCount.value();
  }
  // If we can't determine the trip count, return 1 (conservative estimate)
  return 1;
}

/// Compute the multiplier for an operation based on enclosing affine.for loops
/// within the scf.for boundary
static uint64_t computeAffineLoopMultiplier(Operation *op,
                                            scf::ForOp boundary) {
  uint64_t multiplier = 1;
  Operation *parent = op->getParentOp();

  while (parent && parent != boundary.getOperation()) {
    if (auto affineFor = dyn_cast<affine::AffineForOp>(parent)) {
      multiplier *= getAffineForTripCount(affineFor);
    }
    parent = parent->getParentOp();
  }

  return multiplier;
}

struct ScfForAnalysisResult {
  uint64_t globalLoads = 0;
  uint64_t ldsReads = 0;
  uint64_t ldsWrites = 0;
  uint64_t matrixMultiplyOps = 0;
  /// Direct loads from global memory to LDS (amdgpu.gather_to_lds or
  /// amdgpu.async_load_to_lds)
  uint64_t directLoadsToLDS = 0;
  /// Whether the loop body contains scf.if ops (mutually exclusive branches
  /// make instruction counts unreliable for scheduling)
  bool hasConditionalCode = false;
  /// Indicates if the loop uses double buffering (LDS reads/writes use
  /// arith.select to choose between two buffers)
  bool isDoubleBuffered = false;
};

/// Analyze a single scf.for operation
static ScfForAnalysisResult analyzeScfFor(scf::ForOp forOp) {
  ScfForAnalysisResult result;

  forOp.walk([&](Operation *op) {
    uint64_t multiplier = computeAffineLoopMultiplier(op, forOp);

    // Count amdgpu.raw_buffer_load (global loads)
    if (isa<amdgpu::RawBufferLoadOp>(op)) {
      result.globalLoads += multiplier;
      return;
    }

    // Count vector.load from global memory or workgroup memory (LDS)
    if (auto vectorLoad = dyn_cast<vector::LoadOp>(op)) {
      if (auto memrefType =
              dyn_cast<MemRefType>(vectorLoad.getBase().getType())) {
        if (hasGlobalAddressSpace(memrefType)) {
          result.globalLoads += multiplier;
        } else if (hasWorkgroupAddressSpace(memrefType)) {
          result.ldsReads += multiplier;
          // Check for double buffering: if the memref is selected via
          // arith.select, it indicates alternating between two LDS buffers
          if (isDefinedBySelect(vectorLoad.getBase())) {
            result.isDoubleBuffered = true;
          }
        }
      }
      return;
    }

    // Count vector.transfer_read from global memory or workgroup memory (LDS)
    if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
      if (auto memrefType =
              dyn_cast<MemRefType>(transferRead.getBase().getType())) {
        if (hasGlobalAddressSpace(memrefType)) {
          result.globalLoads += multiplier;
        } else if (hasWorkgroupAddressSpace(memrefType)) {
          result.ldsReads += multiplier;
          // Check for double buffering: if the memref is selected via
          // arith.select, it indicates alternating between two LDS buffers
          if (isDefinedBySelect(transferRead.getBase())) {
            result.isDoubleBuffered = true;
          }
        }
      }
      return;
    }

    // Count memref.load from workgroup memory (LDS reads) or global memory
    if (auto memrefLoad = dyn_cast<memref::LoadOp>(op)) {
      if (auto memrefType =
              dyn_cast<MemRefType>(memrefLoad.getMemRef().getType())) {
        if (hasWorkgroupAddressSpace(memrefType)) {
          result.ldsReads += multiplier;
          if (isDefinedBySelect(memrefLoad.getMemRef())) {
            result.isDoubleBuffered = true;
          }
        } else if (hasGlobalAddressSpace(memrefType)) {
          result.globalLoads += multiplier;
        }
      }
      return;
    }

    // Count memref.store to workgroup memory (LDS writes)
    if (auto memrefStore = dyn_cast<memref::StoreOp>(op)) {
      if (auto memrefType =
              dyn_cast<MemRefType>(memrefStore.getMemRef().getType())) {
        if (hasWorkgroupAddressSpace(memrefType)) {
          result.ldsWrites += multiplier;
          // Check for double buffering: if the memref is selected via
          // arith.select, it indicates alternating between two LDS buffers
          if (isDefinedBySelect(memrefStore.getMemRef())) {
            result.isDoubleBuffered = true;
          }
        }
      }
      return;
    }

    // Count vector.transfer_write to workgroup memory (LDS writes)
    if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
      if (auto memrefType =
              dyn_cast<MemRefType>(transferWrite.getBase().getType())) {
        if (hasWorkgroupAddressSpace(memrefType)) {
          result.ldsWrites += multiplier;
          // Check for double buffering: if the memref is selected via
          // arith.select, it indicates alternating between two LDS buffers
          if (isDefinedBySelect(transferWrite.getBase())) {
            result.isDoubleBuffered = true;
          }
        }
      }
      return;
    }

    // Count Matrix multiply operations
    if (isa<amdgpu::MFMAOp>(op) || isa<amdgpu::ScaledMFMAOp>(op) ||
        isa<amdgpu::WMMAOp>(op)) {
      result.matrixMultiplyOps += multiplier;
      return;
    }

    // Count direct loads from global memory to LDS
    if (isa<amdgpu::GatherToLDSOp>(op) || isa<amdgpu::AsyncLoadToLDSOp>(op)) {
      result.directLoadsToLDS += multiplier;
      return;
    }

    // Detect conditional/branching ops (scf.if, scf.index_switch, etc.)
    // by checking for RegionBranchOpInterface without LoopLikeOpInterface.
    // Loop ops (scf.for, affine.for) are handled separately above.
    if (isa<RegionBranchOpInterface>(op) && !isa<LoopLikeOpInterface>(op)) {
      result.hasConditionalCode = true;
      return;
    }
  });

  return result;
}

/// Rewrite pattern to insert scheduling group barriers in double-buffered
/// scf.for loops that have no direct-to-LDS loads, at most 25 matrix multiply
/// operations per iteration, no conditional code (scf.if), and at most one
/// LDS barrier (skipping complex multi-phase loops like attention).
struct InsertSchedGroupBarrierPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rw) const override {
    mlir::Region &region = op.getRegion();
    Block &block = region.front();

    // Skip loops with multiple LDS barriers (e.g. attention kernels have
    // a multi-phase loop body with many barriers for softmax, rescaling, etc.)
    unsigned ldsBarrierCount = 0;
    block.walk([&](LDSBarrierOp) { ++ldsBarrierCount; });
    if (ldsBarrierCount > 1)
      return failure();

    // Check if SchedBarrierOp already exists (to avoid duplicates)
    WalkResult result = block.walk([&](Operation *innerOp) {
      if (isa<amdgpu::SchedBarrierOp>(innerOp)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    // Analyze the scf.for loop to get operation counts
    ScfForAnalysisResult analysis = analyzeScfFor(op);
    bool isDirectToLDS = analysis.directLoadsToLDS > 0;
    if (isDirectToLDS || !analysis.isDoubleBuffered)
      return failure();

    // Skip if no meaningful operations found
    if (analysis.globalLoads == 0 || analysis.matrixMultiplyOps == 0)
      return failure();

    // Skip loops with scf.if: mutually exclusive branches make the flat
    // instruction count unreliable for interleaving decisions.
    if (analysis.hasConditionalCode)
      return failure();

    // Print analysis results for debugging
    LLVM_DEBUG({
      llvm::dbgs() << "=== scf.for Analysis ===\n";
      llvm::dbgs() << "Location: " << op.getLoc() << "\n";
      llvm::dbgs() << "Global memory loads per iteration: "
                   << analysis.globalLoads << "\n";
      llvm::dbgs() << "Direct loads to LDS per iteration: "
                   << analysis.directLoadsToLDS << "\n";
      llvm::dbgs() << "LDS reads per iteration: " << analysis.ldsReads << "\n";
      llvm::dbgs() << "LDS writes per iteration: " << analysis.ldsWrites
                   << "\n";
      llvm::dbgs() << "Matrix multiply operations per iteration: "
                   << analysis.matrixMultiplyOps << "\n";
      llvm::dbgs() << "Double buffering detected: "
                   << (analysis.isDoubleBuffered ? "yes" : "no") << "\n";
      llvm::dbgs() << "Conditional code (scf.if): "
                   << (analysis.hasConditionalCode ? "yes" : "no") << "\n";
      llvm::dbgs() << "========================\n\n";
    });

    uint64_t numBufferLoads = analysis.globalLoads;
    uint64_t numDSReads = analysis.ldsReads;
    uint64_t numDSWrites = analysis.ldsWrites;
    uint64_t numMatrixMultiplyOps = analysis.matrixMultiplyOps;

    // Large numbers of MFMAs produce excessive sched_group_barrier
    // instructions that significantly increase backend compile time.
    if (numMatrixMultiplyOps > 25)
      return failure();

    // Insert sched_barrier at the start of the block
    rw.setInsertionPointToStart(&block);
    amdgpu::SchedBarrierOp::create(
        rw, op.getLoc(),
        amdgpu::sched_barrier_opt_enumAttr::get(
            rw.getContext(), amdgpu::sched_barrier_opt_enum::none));

    // Insert sched group barriers before the terminator
    auto *lastOp = block.getTerminator()->getPrevNode();
    rw.setInsertionPointAfter(lastOp);
    uint64_t dsReadsPerMFMA =
        llvm::divideCeil(numDSReads, numMatrixMultiplyOps);
    uint64_t dsWritesPerMFMA =
        llvm::divideCeil(numDSWrites, numMatrixMultiplyOps);
    uint64_t bufferLoadsPerMFMA =
        llvm::divideCeil(numBufferLoads, numMatrixMultiplyOps);
    // Insert sched group barriers based on the analysis.
    // Each iteration emits one MFMA group plus the proportional share of
    // DS writes, VMEM loads, and DS reads. Use std::min to avoid requesting
    // more instructions than remain in the final iterations.
    for (uint64_t i = 0; i < numMatrixMultiplyOps; i++) {
      ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x008, 1,
                                       0); // MFMA
      if (numDSWrites > 0) {
        uint64_t count = std::min(dsWritesPerMFMA, numDSWrites);
        ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x200, count,
                                         0); // DS Writes
        numDSWrites -= count;
      }
      if (numBufferLoads > 0) {
        uint64_t count = std::min(bufferLoadsPerMFMA, numBufferLoads);
        ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x020, count,
                                         0); // VMEM
        numBufferLoads -= count;
      }
      if (numDSReads > 0) {
        uint64_t count = std::min(dsReadsPerMFMA, numDSReads);
        ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x100, count,
                                         0); // DS Reads
        numDSReads -= count;
      }
    }

    // Insert sched_barrier at the end
    amdgpu::SchedBarrierOp::create(
        rw, op.getLoc(),
        amdgpu::sched_barrier_opt_enumAttr::get(
            rw.getContext(), amdgpu::sched_barrier_opt_enum::none));

    return success();
  }
};

struct RockAddSchedGroupBarriersPass final
    : rock::impl::RockAddSchedGroupBarriersPassBase<
          RockAddSchedGroupBarriersPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    // Skip the entire function if it contains nested scf.for loops
    bool hasNestedFor = false;
    funcOp.walk([&](scf::ForOp forOp) {
      if (forOp->getParentOfType<scf::ForOp>())
        hasNestedFor = true;
    });
    if (hasNestedFor)
      return;

    RewritePatternSet patterns(ctx);
    patterns.add<InsertSchedGroupBarrierPattern>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end namespace
