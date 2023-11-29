//===- Pipeline.cpp   ---===//
//
// Copyright 2022 AMD
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
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Transforms/RockMultibuffer.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"

#include <map>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPIPELINEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using mlir::gpu::AddressSpace;

enum class MemoryAccessType : uint32_t { READ = 1, WRITE = 2, UNKNOWN = 3 };

using DependencyType = std::pair<MemoryAccessType, MemoryAccessType>;
constexpr DependencyType RAR{MemoryAccessType::READ, MemoryAccessType::READ};

using ScheduleType = std::vector<std::pair<Operation *, unsigned>>;
using DagType =
    DenseMap<rock::StageOp,
             DenseMap<rock::StageOp,
                      DenseSet<std::pair<rock::GpuAllocOp, DependencyType>>>>;

namespace llvm {
template <> struct DenseMapInfo<MemoryAccessType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline MemoryAccessType getEmptyKey() {
    return static_cast<MemoryAccessType>(StorageInfo::getEmptyKey());
  }

  static inline MemoryAccessType getTombstoneKey() {
    return static_cast<MemoryAccessType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const MemoryAccessType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const MemoryAccessType &lhs,
                      const MemoryAccessType &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

namespace {

AddressSpace getAddressSpace(rock::GpuAllocOp alloc) {
  if (alloc.getType().getMemorySpace()) {
    return alloc.getType()
        .getMemorySpace()
        .cast<gpu::AddressSpaceAttr>()
        .getValue();
  }
  return gpu::AddressSpace::Global;
}

// Simple rewrite pass to remove the stages
struct RemoveStagesRewritePattern : public OpRewritePattern<rock::StageOp> {
  using OpRewritePattern<rock::StageOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::StageOp op,
                                PatternRewriter &rw) const override {
    Block *sourceBlock = &op.getRegion().front();
    rw.eraseOp(sourceBlock->getTerminator());
    if (!sourceBlock->empty()) {
      rw.inlineBlockBefore(sourceBlock, op);
    }
    rw.eraseOp(op);
    return failure();
  }
};

struct RemoveBackToBackBarriersRewritePattern
    : public OpRewritePattern<rock::LDSBarrierOp> {
  using OpRewritePattern<rock::LDSBarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::LDSBarrierOp op,
                                PatternRewriter &rw) const override {
    if (dyn_cast<rock::LDSBarrierOp>(op->getNextNode())) {
      op->getNextNode()->erase();
      return success();
    }
    return failure();
  }
};

// Given an operation and its operand, find out what kind of access (if any)
// the operation does on the operand
MemoryAccessType getOperandAccessType(Operation *op, Value operand) {
  if (hasEffect<MemoryEffects::Write>(op, operand)) {
    return MemoryAccessType::WRITE;
  } else if (hasEffect<MemoryEffects::Read>(op, operand)) {
    return MemoryAccessType::READ;
  } else {
    return MemoryAccessType::UNKNOWN;
  }
}

// Create a dependency graph of the given set of stages. The
// idea is to represent the dependencies through a DAG with
// the set of shared resources on the the edges
DagType createDependencyGraph(ArrayRef<rock::StageOp> stages,
                              const DenseSet<rock::GpuAllocOp> &allocs) {
  // Mapping resource->[stages using the given resource]
  DenseMap<rock::StageOp, DenseMap<rock::GpuAllocOp, MemoryAccessType>>
      resourceMap;

  // Mapping stages->[resources used by the given stage]
  DenseMap<rock::GpuAllocOp, DenseMap<rock::StageOp, MemoryAccessType>>
      resourceMapR;

  // For each of the stages walk through the resources they are using. For now
  // the only type of resource is memory
  for (auto stage : stages) {
    stage.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        MemoryAccessType accessType = getOperandAccessType(op, operand);
        auto maybeAlloc = rock::findAlloc(operand);
        if (accessType != MemoryAccessType::UNKNOWN && succeeded(maybeAlloc) &&
            allocs.contains(*maybeAlloc)) {
          resourceMap[stage][*maybeAlloc] = accessType;
          resourceMapR[*maybeAlloc][stage] = accessType;
        }
      }
    });
  }

  DagType dag;
  DenseSet<rock::StageOp> pastStages;
  for (auto source : stages) {
    for (auto [resource, typeSource] : resourceMap[source]) {
      for (auto [sink, typeSink] : resourceMapR[resource]) {
        if (pastStages.contains(sink))
          continue;
        std::pair<MemoryAccessType, MemoryAccessType> dependencyType{typeSource,
                                                                     typeSink};
        if (source != sink && dependencyType != RAR) {
          dag[source][sink].insert({resource, dependencyType});
        }
      }
    }
    pastStages.insert(source);
  }
  return dag;
}

DenseSet<std::pair<rock::GpuAllocOp, DependencyType>>
getDependencies(rock::StageOp stage0, rock::StageOp stage1, DagType &dag) {
  DenseSet<std::pair<rock::GpuAllocOp, DependencyType>> dependencies;
  if (dag.contains(stage0)) {
    if (dag[stage0].contains(stage1)) {
      for (auto dep : dag[stage0][stage1]) {
        dependencies.insert(dep);
      }
    }
  }
  return dependencies;
}

// Function to create the schedule of the current set of stages
void createSchedule(SmallVector<rock::StageOp> &stages,
                    const DenseSet<rock::GpuAllocOp> &resources, int64_t ii,
                    ScheduleType &schedule,
                    DenseMap<rock::GpuAllocOp, int> &multiBuffers) {
  // Create the dependency graph
  DagType dag = createDependencyGraph(stages, resources);

  // Start building the schedules
  //
  // Since we accept the stages from the user, we don't need to do any
  // analysis to determine what goes in each stage. We only have to group things
  // in set of stages of length II.
  //  For instance, consider the following unpipelined schedule. The column `t`
  //  represents
  // the time slot, and the subsequent columns represents the iterations.
  //  +t\i+=== 0 ===+
  //  + 0 +== S0  ==+
  //  +===+=========+
  //  + 1 +== S1  ==+
  //  +===+=========+
  //  + 2 +== S2  ==+
  //  +===+=========+
  //  + 3 +== S3  ==+
  //  +===+=========+
  // When the II == 3 it means that S0/S3 will run in parallel while S2 and S3
  // will run sequentially. Please note that S0 and S3 belong to two different
  // iterations (0 and 1, respectively). This is the resulting schedule:
  //  +t\i+=== 0 ===++=== 1 ===+
  //  + 0 +== S0  ==++== S3  ==+
  //  +===+=========++=========+
  //  + 1 +== S1  ==+
  //  +===+=========+
  //  + 2 +== S2  ==+
  //  +===+=========+
  // In this case, we reduced the time slots to 3, and we have 2 set of stages
  // runnning in parallel. Please note that conflicts can only happen between S0
  // and S3. If we increase II, we generate the following pipeline:
  //  +t\i+=== 0 ===++=== 1 ===+
  //  + 0 +== S0  ==++== S2  ==+
  //  +===+=========++=========+
  //  + 1 +== S1  ==++== S3  ==+
  //  +===+=========++=========+
  // Now we have only two time slots and 2 iterations. conflicts
  // can happen between S0 and S2 and between S1 and S3. This is all captured in
  // the following algorithm. `t` is the time slot, i.e., the flowing of the
  // time, and goes from 0 to II-1. `i` is the iteration that is starting at
  // time `t`
  for (int t = 0; t < ii; t++) {
    int iteration = 0;

    // The following stages need to run in parallel
    SmallVector<rock::StageOp> parallelStages;
    for (size_t j = t; j < stages.size(); j += ii) {
      schedule.push_back({stages[j], iteration++});

      // This is the set of multi-buffers needed at this time slot
      DenseMap<rock::GpuAllocOp, int> thisMultiBuffers = multiBuffers;
      for (auto [alloc, factor] : thisMultiBuffers) {
        thisMultiBuffers[alloc] = 1;
      }
      // The only resource that can conflict btween different stages is memory
      // If there are memory conflicts we can sort them via multibuffers. I.e.,
      // we can (logically) provide a different buffer for different cycles
      for (auto otherStage : parallelStages) {
        auto dependencies = getDependencies(otherStage, stages[j], dag);
        for (auto [res, type] : dependencies) {
          thisMultiBuffers[res]++;
        }
      }
      parallelStages.push_back(stages[j]);

      // Update the global multibuffers by merging in the factors needed for
      // the current time slot
      for (auto [buffer, factor] : thisMultiBuffers)
        if (factor > multiBuffers[buffer])
          multiBuffers[buffer] = factor;
    }
  }
}

// Prune a dependency graph taking into account multi-buffers. Since
// multi-buffers are logically different for each iteration, if the dependency
// on a multi-buffer spans multiple iteration then it can be pruned
DagType pruneGraph(DagType dag) {
  DagType prunedGraph;
  // Multibuffers have the logical property of being unique for each iteration
  // of the loop Hence, if we know we are dealing with a multi-buffer and the
  // dependency concerns two different iteration. In other words, if stageA
  // accesses LDS in iteration i and stageB accesses LDS in iteration j stageA
  // and stageB have no dependencies as long as i!=j
  for (auto [sink, edges] : dag) {
    for (auto [source, deps] : edges) {
      DenseSet<std::pair<rock::GpuAllocOp, DependencyType>> newDeps;
      for (auto [alloc, type] : deps) {
        if (getAddressSpace(alloc) != gpu::AddressSpace::Workgroup)
          continue;
        newDeps.insert({alloc, type});
      }
      if (!newDeps.empty())
        prunedGraph[sink][source] = newDeps;
    }
  }
  return prunedGraph;
}

// Utility function to place an empty stage before or after another `stage`. The
// empty stage will contain an `lds_barrier` if `isBarrier` is set to true
rock::StageOp placeEmptyStage(IRRewriter &rewriter, Location loc,
                              rock::StageOp stage, bool isBarrier,
                              bool isBefore) {
  PatternRewriter::InsertionGuard guard(rewriter);
  if (isBefore)
    rewriter.setInsertionPoint(stage);
  else
    rewriter.setInsertionPointAfter(stage);
  auto barrierStage = rewriter.create<rock::StageOp>(loc, "barrier");
  rewriter.setInsertionPointToStart(&barrierStage.getRegion().emplaceBlock());
  if (isBarrier) {
    rewriter.create<rock::LDSBarrierOp>(loc);
  }
  rewriter.create<rock::YieldOp>(loc);
  return barrierStage;
}

// Barrier placement after the pipeline pass.
// We add a dummy stage between each pair of stages. This makes
// the process of pipelining easier, because we can use a
// initiation interval twice as big and pipeline as usual. This function
// takes also care to update the initiation interval, so that the caller
// does not have to know how `placeBarrier` internally works.
void placeBarriers(IRRewriter &rewriter, Location loc, scf::ForOp forOp,
                   ArrayRef<rock::StageOp> stages,
                   DenseSet<rock::GpuAllocOp> &allocs,
                   SmallVector<rock::StageOp> &extendedStages,
                   int64_t &initiationInterval) {
  DagType dag = createDependencyGraph(stages, allocs);
  dag = pruneGraph(dag);

  extendedStages.push_back(stages[0]);
  for (size_t i = 0; i < stages.size() - 1; i++) {
    bool placeBarrier = dag[stages[i]].contains(stages[i + 1]);
    bool before = true;
    auto barrierStage =
        placeEmptyStage(rewriter, loc, stages[i + 1], placeBarrier, before);
    extendedStages.push_back(barrierStage);
    extendedStages.push_back(stages[i + 1]);
  }
  // Place a barrier after the last stage. This barrier is necessary to take
  // into consideration the loop carried dependencies when pipelining. However,
  // this might introduce an unnecessary barrier as the last operation of the
  // epilogue. We will take care of removing this barrier at the end of the pass
  auto maybeNumIterations =
      rock::computeConstDiff(forOp.getLowerBound(), forOp.getUpperBound());
  if (!maybeNumIterations.has_value() ||
      (maybeNumIterations.has_value() && maybeNumIterations.value() > 1)) {
    const bool placeBarrier = true;
    const bool after = false;
    extendedStages.push_back(
        placeEmptyStage(rewriter, loc, stages.back(), placeBarrier, after));
  }
  initiationInterval *= 2;
}

struct RockPipeline : public rock::impl::RockPipelinePassBase<RockPipeline> {
  using rock::impl::RockPipelinePassBase<RockPipeline>::RockPipelinePassBase;
  void runOnOperation() override;
};

} // end namespace

void RockPipeline::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = func->getContext();
  Location loc = func->getLoc();
  IRRewriter rewriter(ctx);

  std::map<Operation *, SmallVector<rock::GpuAllocOp>> ldsAllocMap;
  llvm::DenseSet<rock::GpuAllocOp> allocs;

  // Collect the global resources (i.e., the memory allocations)
  // Note: we can only have two kind of memory:
  // - Registers
  // - LDS
  func.walk([&](rock::GpuAllocOp alloc) { allocs.insert(alloc); });

  DenseMap<rock::GpuAllocOp, int> multiBufferFactors;
  llvm::DenseMap<scf::ForOp, ScheduleType> scheduleMap;
  llvm::DenseMap<scf::ForOp, Operation *> nextOpMap;
  for (auto res : allocs)
    multiBufferFactors[res] = 1;

  auto rockPipelineAttrName = rock::PipelineAttr::getMnemonic();
  bool isNestedPipelining = false;
  func.walk([&](scf::ForOp forOp) -> WalkResult {
    SmallVector<rock::StageOp> stages;

    if (!forOp->hasAttrOfType<rock::PipelineAttr>(rockPipelineAttrName))
      return WalkResult::advance();

    forOp.getBody()->walk([&](scf::ForOp nestedFor) {
      if (nestedFor->hasAttr(rockPipelineAttrName))
        isNestedPipelining = true;
    });

    if (isNestedPipelining)
      return WalkResult::interrupt();

    // Get the initiation interval (II)
    int64_t ii = forOp->removeAttr(rockPipelineAttrName)
                     .dyn_cast<rock::PipelineAttr>()
                     .getInitiationInterval();

    forOp.walk([&](rock::StageOp stageOp) { stages.push_back(stageOp); });

    forOp.walk([](rock::LDSBarrierOp barrier) {
      if (!barrier->getParentOfType<rock::StageOp>())
        barrier->erase();
    });

    if (stages.empty())
      WalkResult::advance();

    LLVM_DEBUG(DBGS() << "Number of stages: " << stages.size() << "\n");
    LLVM_DEBUG(DBGS() << "Initiation Interval: " << ii << "\n");

    // Insert the barriers as new stages
    SmallVector<rock::StageOp> extendedStages;
    placeBarriers(rewriter, loc, forOp, stages, allocs, extendedStages, ii);

    ScheduleType schedule;
    createSchedule(extendedStages, allocs, ii, schedule, multiBufferFactors);

    scheduleMap[forOp] = schedule;
    // Annotate the operation that defines the boundary of the `forOp`. This
    // is because, at the end of the pass, we will remove any barrier at the
    // boundary (because they are not useful)
    nextOpMap[forOp] = forOp->getNextNode();

    return WalkResult::advance();
  });

  if (isNestedPipelining)
    return signalPassFailure();

  // Multi-buffer(if needed)
  bool isMultiBufferingFailed = false;
  for (auto [alloc, factor] : multiBufferFactors) {
    if (factor > 1) {
      if (failed(rock::multiBuffer(rewriter, alloc, factor, true))) {
        isMultiBufferingFailed = true;
        break;
      }
    }
  }
  if (isMultiBufferingFailed)
    return signalPassFailure();

  // Check we didn't push memory too far
  DenseMap<AddressSpace, size_t> gpuMemoryBytes;
  func.walk([&](rock::GpuAllocOp alloc) {
    auto addressSpace = getAddressSpace(alloc);
    gpuMemoryBytes[addressSpace] += alloc.getType().getShape().back();
  });

  if (gpuMemoryBytes[AddressSpace::Workgroup] > size_t(64 * 1024))
    return signalPassFailure();

  // Pipeline the loops
  {
    RewritePatternSet patterns(&getContext());
    mlir::scf::PipeliningOption options;
    options.getScheduleFn = [&](scf::ForOp op, ScheduleType &sched) {
      sched = scheduleMap[op];
    };
    scf::populateSCFLoopPipeliningPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  // Cleanup the stages
  {
    if (removeStages) {
      RewritePatternSet patterns(&getContext());
      patterns.add<RemoveStagesRewritePattern,
                   RemoveBackToBackBarriersRewritePattern>(&getContext());
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
  }

  // Cleanup unwanted barriers
  for (auto [forOp, nextOp] : nextOpMap) {
    if (dyn_cast<rock::LDSBarrierOp>(nextOp->getPrevNode()))
      nextOp->getPrevNode()->erase();
  }
}
