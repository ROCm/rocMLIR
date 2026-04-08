//===- AddSchedGroupBarriers.cpp - Add IGLP opt scheduling hints ----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass analyzes scf.for loops and inserts rocdl.iglp.opt scheduling hints
// to optimize instruction scheduling on AMD GPUs.
//
// The iglp_opt intrinsic provides predefined instruction scheduling strategies
// in the LLVM AMDGPU backend:
//   variant 0: Interleave DS and MFMA for small GEMM kernels
//
// The pass skips functions that contain nested scf.for loops (e.g. attention).
// For the remaining single-loop functions, iglp_opt is only inserted if the
// loop contains matrix multiply ops (MFMA/WMMA), does not use direct-to-LDS
// loads, and has no conditional code. Works with both single and double
// buffered pipelines.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/Debug.h"

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

/// Check if a loop is eligible for iglp_opt insertion.
/// Returns true if the loop has MFMA/WMMA ops and a clean scheduling region.
static bool isEligibleForIglpOpt(scf::ForOp forOp) {
  bool hasMatrixOp = false;
  bool hasDirectToLDS = false;
  bool hasConditionalCode = false;

  forOp.walk([&](Operation *op) {
    if (isa<amdgpu::MFMAOp, amdgpu::ScaledMFMAOp, amdgpu::WMMAOp>(op))
      hasMatrixOp = true;
    else if (isa<amdgpu::GatherToLDSOp, amdgpu::AsyncLoadToLDSOp>(op))
      hasDirectToLDS = true;
    else if (isa<RegionBranchOpInterface>(op) && !isa<LoopLikeOpInterface>(op))
      hasConditionalCode = true;
  });

  return hasMatrixOp && !hasDirectToLDS && !hasConditionalCode;
}

/// Try to insert an iglp_opt scheduling hint for a single scf.for loop.
static bool tryInsertIglpOpt(scf::ForOp forOp) {
  Block &block = forOp.getRegion().front();

  bool hasExistingHint = false;
  block.walk([&](ROCDL::IglpOpt) { hasExistingHint = true; });
  if (hasExistingHint)
    return false;

  if (!isEligibleForIglpOpt(forOp))
    return false;

  LLVM_DEBUG(llvm::dbgs() << "Inserting iglp_opt variant 0 at "
                          << forOp.getLoc() << "\n");

  OpBuilder builder(forOp.getContext());
  builder.setInsertionPointToStart(&block);
  ROCDL::IglpOpt::create(builder, forOp.getLoc(), /*variant=*/0);

  return true;
}

struct RockAddSchedGroupBarriersPass final
    : rock::impl::RockAddSchedGroupBarriersPassBase<
          RockAddSchedGroupBarriersPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    bool hasNestedFor = false;
    funcOp.walk([&](scf::ForOp forOp) {
      if (forOp->getParentOfType<scf::ForOp>())
        hasNestedFor = true;
    });
    if (hasNestedFor)
      return;

    funcOp.walk([&](scf::ForOp forOp) { tryInsertIglpOpt(forOp); });
  }
};

} // end namespace
