//===- PreparePipeline - MLIR Rock ops lowering passes -----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This pass prepares ops for pipelining
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPREPAREPIPELINEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-prepare-pipeline"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockPreparePipelinePass
    : public rock::impl::RockPreparePipelinePassBase<RockPreparePipelinePass> {
  void runOnOperation() override;
};

} // end anonymous namespace

static LogicalResult
mergeStages(ArrayRef<StageOp> stages, StringRef newStageName,
            std::optional<int64_t> barrierBefore = std::nullopt) {
  if (stages.size() <= 1)
    return failure();

  StageOp firstStage = stages[0];
  firstStage.setName(newStageName);

  if (barrierBefore.has_value()) {
    if (barrierBefore == 0) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "It doesn't make sense to add a barrier before the first stage\n");
      return failure();
    }
    if (barrierBefore >= stages.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "barrierBefore is bigger than stages.size()!\n");
      return failure();
    }
  }

  // skip first stage
  int64_t idx = 1;
  for (StageOp stage : stages.drop_front(1)) {
    Region &region = stage.getRegion();

    // Empty region, just remove the op
    if (region.empty())
      stage.erase();

    Block &block = region.front();
    SmallVector<Operation *> opsToMove;

    for (Operation &op : block.getOperations()) {
      if (!op.hasTrait<OpTrait::IsTerminator>())
        opsToMove.push_back(&op);
    }

    // Add barrier
    if (barrierBefore.has_value() && barrierBefore == idx) {
      OpBuilder builder(&firstStage.getRegion().front().back());
      LDSBarrierOp::create(builder, firstStage->getLoc());
    }

    // Move operations
    for (Operation *op : opsToMove) {
      // insertion point: last operation of the first stage
      Operation *insertionPoint = &firstStage.getRegion().front().back();
      op->moveAfter(insertionPoint);
    }

    stage.erase();
    idx++;
  }

  // move rock.yield to be the last op
  Operation *terminator = nullptr;
  for (Operation &op : firstStage.getRegion().front().getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      terminator = &op;
  }
  if (!terminator)
    return failure();

  // insertion point: last operation of the stage
  Operation *insertionPoint = &firstStage.getRegion().front().back();
  terminator->moveAfter(insertionPoint);

  return success();
}

static void inlineStageOp(StageOp stageOp) {
  Region &region = stageOp.getRegion();

  // Empty region, just remove the op
  if (region.empty()) {
    stageOp.erase();
    return;
  }

  Block &block = region.front();
  SmallVector<Operation *> opsToMove;

  for (Operation &op : block.getOperations()) {
    if (!op.hasTrait<OpTrait::IsTerminator>())
      opsToMove.push_back(&op);
  }
  Operation *insertionPoint = stageOp;

  // Move operations
  for (Operation *op : opsToMove)
    op->moveBefore(insertionPoint);

  stageOp.erase();
}

static FailureOr<rock::StageOp> getStage(scf::ForOp loop, StringRef stageName) {
  FailureOr<rock::StageOp> maybeStage = failure();
  loop->walk([&](rock::StageOp op) {
    if (op.getName() == stageName)
      maybeStage = op;
  });

  return maybeStage;
}

static LogicalResult regroupStagesForOuterLoop(scf::ForOp outerLoop) {
  // Create a stage of InitGemm0 + MMAGemm0 + PostProcessGemm0 + LDSWriteGemm1
  FailureOr<rock::StageOp> initGemm0 = getStage(outerLoop, "InitGemm0");
  FailureOr<rock::StageOp> mmaGemm0 = getStage(outerLoop, "MMAGemm0");
  FailureOr<rock::StageOp> postprocessGemm0 =
      getStage(outerLoop, "PostProcessGemm0");
  FailureOr<rock::StageOp> ldsWriteGemm1 = getStage(outerLoop, "LDSWriteGemm1");
  if (failed(initGemm0) || failed(mmaGemm0) || failed(postprocessGemm0) ||
      failed(ldsWriteGemm1)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't find expected stages\n");
    return failure();
  }

  if (failed(mergeStages({initGemm0.value(), mmaGemm0.value(),
                          postprocessGemm0.value(), ldsWriteGemm1.value()},
                         "MMAG0+PPG0+LWG1")))
    return failure();

  // Create a stage of LDSWriteGemm0 + LDSReadGemm0 + GlobalReadGemm1
  FailureOr<rock::StageOp> ldsWriteGemm0 = getStage(outerLoop, "LDSWriteGemm0");
  FailureOr<rock::StageOp> ldsReadGemm0 = getStage(outerLoop, "LDSReadGemm0");
  FailureOr<rock::StageOp> globalReadGemm1 =
      getStage(outerLoop, "GlobalReadGemm1");
  if (failed(ldsWriteGemm0) || failed(ldsReadGemm0) ||
      failed(globalReadGemm1)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't find expected stages\n");
    return failure();
  }

  if (failed(mergeStages({ldsWriteGemm0.value(), ldsReadGemm0.value(),
                          globalReadGemm1.value()},
                         "LWG0+LRG0+GRG1", /*barrierBefore=*/1)))
    return failure();

  // Create a stage of InitGemm1 + LDSReadGemm1 + MMAGemm1 + PostProcessGemm1
  FailureOr<rock::StageOp> initGemm1 = getStage(outerLoop, "InitGemm1");
  FailureOr<rock::StageOp> ldsReadGemm1 = getStage(outerLoop, "LDSReadGemm1");
  FailureOr<rock::StageOp> mmaGemm1 = getStage(outerLoop, "MMAGemm1");
  FailureOr<rock::StageOp> postProcessGemm1 =
      getStage(outerLoop, "PostProcessGemm1");
  if (failed(initGemm1) || failed(ldsReadGemm1) || failed(mmaGemm1) ||
      failed(postProcessGemm1)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't find expected stages\n");
    return failure();
  }

  if (failed(mergeStages({initGemm1.value(), ldsReadGemm1.value(),
                          mmaGemm1.value(), postProcessGemm1.value()},
                         "LRG1+MMAG1+PPG1", /*barrierBefore=*/2)))
    return failure();

  FailureOr<rock::StageOp> globalReadGemm0 =
      getStage(outerLoop, "GlobalReadGemm0");
  FailureOr<rock::StageOp> secondStage = getStage(outerLoop, "LWG0+LRG0+GRG1");
  FailureOr<rock::StageOp> thirdStage = getStage(outerLoop, "MMAG0+PPG0+LWG1");

  if (failed(globalReadGemm0) || failed(secondStage) || failed(thirdStage)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't find expected stages\n");
    return failure();
  }
  globalReadGemm0.value()->moveBefore(thirdStage.value());
  secondStage.value()->moveBefore(thirdStage.value());

  return success();
}

static bool heuristicPipelineOuterLoop(scf::ForOp outerLoop,
                                       ArrayRef<scf::ForOp> innerLoops) {
  std::optional<APInt> outerLoopIters = outerLoop.getStaticTripCount();
  // if the outer loop is dynamic, we do not pipeline it
  // TODO: add support for dynamic loop pipelining
  bool pipelineOuterLoop = outerLoopIters.has_value();

  // we pipeline the outer loop only if all inner loops have one iteration
  // TODO: investigate a better heuristic
  for (scf::ForOp innerLoop : innerLoops) {
    std::optional<APInt> loopIters = innerLoop.getStaticTripCount();
    pipelineOuterLoop &=
        loopIters.has_value() && loopIters.value().getSExtValue() == 1;
  }

  // We only pipeline the outer loop for schedule v2
  assert(outerLoop->hasAttr(PipelineAttr::getMnemonic()));
  int64_t ii =
      cast<rock::PipelineAttr>(outerLoop->getAttr(PipelineAttr::getMnemonic()))
          .getInitiationInterval();
  pipelineOuterLoop &= ii == 1;

  return pipelineOuterLoop;
}

static void prepareGemmPipeline(scf::ForOp outerLoop) {
  // Move allocs outside of stages outside of the loop
  SmallVector<rock::GpuAllocOp> allocs;
  outerLoop->walk([&allocs](rock::GpuAllocOp op) {
    if (!isa<rock::StageOp>(op->getParentOp()))
      allocs.push_back(op);
  });
  LLVM_DEBUG(llvm::dbgs() << "Moving " << allocs.size()
                          << " rock.alloc ops to before the outer loop\n");

  for (rock::GpuAllocOp alloc : allocs)
    alloc->moveBefore(outerLoop);
}

static LogicalResult prepareGemmGemmPipeline(scf::ForOp outerLoop,
                                             ArrayRef<scf::ForOp> innerLoops) {
  // The pipeline pass can't do nested loop pipelining (where both inner and
  // outer have pipelining stages). So, here, we either decide to pipeline the
  // inner loops or unroll the inner loops and pipeline the outer loop. We use a
  // heuristic to decide which one is best to do.
  bool pipelineOuterLoop = heuristicPipelineOuterLoop(outerLoop, innerLoops);

  // Move allocs outside of stages outside of the loop
  prepareGemmPipeline(outerLoop);

  if (pipelineOuterLoop) {
    // We use a 4-stage pipeline for the outer loop
    // The four stages are:
    // 1. GlobalReadGemm0
    // 2. GlobalReadGemm1 + LDSWriteGemm0 + LDSReadGemm0
    // 3. PostProcessGemm0 + MMAGemm0 + LDSWriteGemm1 + Softmax (if attention)
    // 4. InitGemm1 + LDSReadGemm1 + MMAGemm1 + PostProcessGemm1
    LLVM_DEBUG(llvm::dbgs() << "Preparing to pipeline outer loop\n");
    // 1. Change name of stages inside inner loops
    for (auto [gemmNum, innerLoop] : llvm::enumerate(innerLoops)) {
      SmallVector<rock::StageOp> stages;
      innerLoop->walk([&stages](rock::StageOp op) { stages.push_back(op); });
      for (rock::StageOp stage : stages) {
        StringRef name = stage.getName();
        StringAttr newName = StringAttr::get(
            outerLoop->getContext(), Twine(name) + "Gemm" + Twine(gemmNum));
        stage.setName(newName);
      }
    }

    // 2. Unroll the inner loops
    for (scf::ForOp innerLoop : innerLoops) {
      LogicalResult res = loopUnrollFull(innerLoop);
      if (failed(res)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to unroll loop: " << innerLoop << "\n");
        return failure();
      }
    }

    // 3. Regroup stages
    if (failed(regroupStagesForOuterLoop(outerLoop)))
      return failure();
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Preparing to pipeline inner loops\n");

    // 1. Remove outer loop stages
    SmallVector<rock::StageOp> stages;
    outerLoop.walk([&](rock::StageOp op) {
      if (op->getParentOp() == outerLoop)
        stages.push_back(op);
    });
    for (rock::StageOp stage : stages)
      inlineStageOp(stage);

    // 2. Merge MMA and PostProcess of the second loop
    // we do this to avoid extra multibuffers
    FailureOr<rock::StageOp> mmaGemm1 = getStage(innerLoops[1], "MMA");
    FailureOr<rock::StageOp> postprocessGemm1 =
        getStage(innerLoops[1], "PostProcess");
    if (failed(mmaGemm1) || failed(postprocessGemm1)) {
      LLVM_DEBUG(llvm::dbgs() << "Couldn't find expected stages\n");
      return failure();
    }
    if (failed(mergeStages({mmaGemm1.value(), postprocessGemm1.value()},
                           "MMA+PostProcess")))
      return failure();

    // 3. Remove pipeline attribute from outer loop
    outerLoop->removeAttr(PipelineAttr::getMnemonic());
  }

  return success();
}

void RockPreparePipelinePass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  // get inner loops and outer loop
  SmallVector<scf::ForOp> innerLoops;
  SmallVector<scf::ForOp> outerLoops;
  func.walk([&](scf::ForOp op) {
    if (isa<scf::ForOp>(op->getParentOp()))
      innerLoops.push_back(op);
    else
      outerLoops.push_back(op);
  });

  if (innerLoops.size() == 0 && outerLoops.size() == 1) {
    LLVM_DEBUG(llvm::dbgs() << "This looks like a gemm-like kernel\n");
    prepareGemmPipeline(outerLoops[0]);
  } else if (innerLoops.size() == 2 && outerLoops.size() == 1) {

    LLVM_DEBUG(llvm::dbgs() << "This looks like a gemm+gemm like kernel\n");
    if (failed(prepareGemmGemmPipeline(outerLoops[0], innerLoops)))
      return signalPassFailure();
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "This doesn't look like a gemm or gemm+gemm kernel\n");
  }
}
