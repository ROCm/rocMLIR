//===- AddSchedGroupBarriers.cpp - Add scheduling group barriers ----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass analyzes scf.for loops to count memory operations and MFMA
// instructions per iteration, then inserts scheduling group barriers:
// - Global memory loads (amdgpu.raw_buffer_load, vector.load from global)
// - LDS/workgroup memory reads (memref.load from workgroup address space)
// - LDS/workgroup memory writes (memref.store to workgroup address space)
// - MFMA instructions (amdgpu.mfma)
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
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
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
  /// Direct loads from global memory to LDS (amdgpu.gather_to_lds)
  uint64_t directLoadsToLDS = 0;
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

    // Count memref.load from workgroup memory (LDS reads)
    if (auto memrefLoad = dyn_cast<memref::LoadOp>(op)) {
      if (auto memrefType =
              dyn_cast<MemRefType>(memrefLoad.getMemRef().getType())) {
        if (hasWorkgroupAddressSpace(memrefType)) {
          result.ldsReads += multiplier;
          // Check for double buffering: if the memref is selected via
          // arith.select, it indicates alternating between two LDS buffers
          if (isDefinedBySelect(memrefLoad.getMemRef())) {
            result.isDoubleBuffered = true;
          }
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

    // Count direct loads from global memory to LDS (amdgpu.gather_to_lds)
    if (isa<amdgpu::GatherToLDSOp>(op)) {
      result.directLoadsToLDS += multiplier;
      return;
    }
  });

  return result;
}

/// Rewrite pattern to insert scheduling group barriers in scf.for loops
struct InsertSchedGroupBarrierPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rw) const override {
    mlir::Region &region = op.getRegion();
    Block &block = region.front();

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

    // Skip if no meaningful operations found
    if (analysis.globalLoads == 0 && analysis.matrixMultiplyOps == 0 &&
        analysis.directLoadsToLDS == 0)
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
      llvm::dbgs() << "========================\n\n";
    });

    uint64_t numBufferLoads = analysis.globalLoads + analysis.directLoadsToLDS;
    uint64_t numDSReads = analysis.ldsReads;
    uint64_t numDSWrites = analysis.ldsWrites;
    uint64_t numMatrixMultiplyOps = analysis.matrixMultiplyOps;

    // Insert sched_barrier at the start of the block
    rw.setInsertionPointToStart(&block);
    amdgpu::SchedBarrierOp::create(
        rw, op.getLoc(),
        amdgpu::sched_barrier_opt_enumAttr::get(
            rw.getContext(), amdgpu::sched_barrier_opt_enum::none));

    // Insert sched group barriers before the terminator
    auto *lastOp = block.getTerminator()->getPrevNode();
    rw.setInsertionPointAfter(lastOp);

    // Insert sched group barriers based on the analysis
    if (numBufferLoads > 0 && numMatrixMultiplyOps > 0) {
      for (uint64_t i = 0; i < numBufferLoads; i++) {
        uint64_t dsReadsPerLoad = llvm::divideCeil(numDSReads, numBufferLoads);
        uint64_t dsWritesPerLoad =
            llvm::divideCeil(numDSWrites, numBufferLoads);
        uint64_t matrixMultiplyPerLoad =
            llvm::divideCeil(numMatrixMultiplyOps, numBufferLoads);
        if (analysis.isDoubleBuffered) {
          uint64_t dsWritesPerMFMA =
              llvm::divideCeil(dsWritesPerLoad, matrixMultiplyPerLoad);
          while (dsWritesPerLoad > 0 && matrixMultiplyPerLoad > 0) {
            ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x200,
                                             dsWritesPerMFMA, 0); // DS Writes
            ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x008, 1,
                                             0); // MFMA
            matrixMultiplyPerLoad--;
            dsWritesPerLoad -= dsWritesPerMFMA;
          }
          ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x020, 1,
                                           0); // VMEM
          if (matrixMultiplyPerLoad > 0) {
            ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x008,
                                             matrixMultiplyPerLoad,
                                             0); // MFMA
          }
          if (dsReadsPerLoad > 0) {
            ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x100,
                                             dsReadsPerLoad,
                                             0); // DS Reads
          }
        } else {
          ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x020, 1,
                                           0); // VMEM
          if (dsReadsPerLoad > 0) {
            ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x100,
                                             dsReadsPerLoad,
                                             0); // DS Reads
          }
          uint64_t dsWritesPerMFMA =
              llvm::divideCeil(dsWritesPerLoad, matrixMultiplyPerLoad);
          while (dsWritesPerLoad > 0 && matrixMultiplyPerLoad > 0) {
            ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x200,
                                             dsWritesPerMFMA,
                                             0); // DS Writes
            ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x008, 1,
                                             0); // MFMA
            matrixMultiplyPerLoad--;
            dsWritesPerLoad -= dsWritesPerMFMA;
          }
          if (matrixMultiplyPerLoad > 0) {
            ROCDL::SchedGroupBarrier::create(rw, op.getLoc(), 0x008,
                                             matrixMultiplyPerLoad,
                                             0); // MFMA
          }
        }
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

    RewritePatternSet patterns(ctx);
    patterns.add<InsertSchedGroupBarrierPattern>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end namespace
