//===- SubgroupReduceToDPP.cpp - Lower SubgroupReduceOp to DPP -===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass lowers gpu.subgroup_reduce operations to AMD DPP instructions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {

#define GEN_PASS_DEF_ROCKSUBGROUPREDUCETODPPPASS
#include "mlir/Dialect/Rock/Passes.h.inc"

struct RockSubgroupReduceToDPPPass
    : public impl::RockSubgroupReduceToDPPPassBase<
          RockSubgroupReduceToDPPPass> {

  RockSubgroupReduceToDPPPass() = default;
  RockSubgroupReduceToDPPPass(const RockSubgroupReduceToDPPPassOptions &options)
      : RockSubgroupReduceToDPPPassBase(options) {}

  void runOnOperation() override {
    auto maybeChipset = amdgpu::Chipset::parse(chip);
    if (failed(maybeChipset)) {
      getOperation()->emitError() << "Invalid chipset: " << chip;
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    unsigned subgroupSize = rock::lookupArchInfo(chip).waveSize;

    populateGpuBreakDownSubgroupReducePatterns(
        patterns, /*maxShuffleBitwidth=*/32, PatternBenefit(3));

    populateGpuLowerSubgroupReduceToDPPPatterns(
        patterns, subgroupSize, *maybeChipset, PatternBenefit(2));
    populateGpuLowerClusteredSubgroupReduceToDPPPatterns(
        patterns, subgroupSize, *maybeChipset, PatternBenefit(2));

    // Shuffle-based fallback (lower priority) ensures any gpu.subgroup_reduce
    // that DPP patterns cannot handle is still lowered, rather than surviving
    // into convert-gpu-to-rocdl where it would be illegal.
    populateGpuLowerSubgroupReduceToShufflePatterns(
        patterns, subgroupSize, /*shuffleBitwidth=*/32, PatternBenefit(1));
    populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
        patterns, subgroupSize, /*shuffleBitwidth=*/32, PatternBenefit(1));

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace rock
} // namespace mlir
