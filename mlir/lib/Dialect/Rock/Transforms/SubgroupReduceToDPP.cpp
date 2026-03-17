//===- SubgroupReduceToDPP.cpp - Lower SubgroupReduceOp to DPP --*- C++ -*-===//
//
// Part of the rocMLIR Project.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers gpu.subgroup_reduce operations to AMD DPP instructions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {

#define GEN_PASS_DEF_ROCKSUBGROUPREDUCETODPPPASS
#include "mlir/Dialect/Rock/Passes.h.inc"

struct RockSubgroupReduceToDPPPass
    : public impl::RockSubgroupReduceToDPPPassBase<RockSubgroupReduceToDPPPass> {

  RockSubgroupReduceToDPPPass() = default;
  RockSubgroupReduceToDPPPass(const RockSubgroupReduceToDPPPassOptions &options)
      : RockSubgroupReduceToDPPPassBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<amdgpu::AMDGPUDialect, gpu::GPUDialect,
                    ROCDL::ROCDLDialect>();
  }

  void runOnOperation() override {
    auto maybeChipset = amdgpu::Chipset::parse(chip);
    if (failed(maybeChipset)) {
      getOperation()->emitError() << "Invalid chipset: " << chip;
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    unsigned subgroupSize = 64;
    if (maybeChipset->majorVersion >= 10) {
      subgroupSize = 32;
    }

    populateGpuBreakDownSubgroupReducePatterns(
        patterns, /*maxShuffleBitwidth=*/32, PatternBenefit(3));

    populateGpuLowerSubgroupReduceToDPPPatterns(patterns, subgroupSize,
                                                *maybeChipset, PatternBenefit(2));
    populateGpuLowerClusteredSubgroupReduceToDPPPatterns(
        patterns, subgroupSize, *maybeChipset, PatternBenefit(2));

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace rock
} // namespace mlir
