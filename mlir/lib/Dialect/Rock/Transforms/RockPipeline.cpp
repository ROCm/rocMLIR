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

#include <algorithm>
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
constexpr DependencyType RAW{MemoryAccessType::WRITE, MemoryAccessType::READ};
constexpr DependencyType WAR{MemoryAccessType::READ, MemoryAccessType::WRITE};

using ScheduleType = std::vector<std::pair<Operation *, unsigned>>;
using DagType =
    DenseMap<rock::StageOp,
             DenseMap<rock::StageOp,
                      DenseSet<std::pair<rock::GpuAllocOp, DependencyType>>>>;

namespace llvm {
template <>
struct DenseMapInfo<MemoryAccessType> {
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

template <typename MemrefTypedValue>
AddressSpace getAddressSpace(MemrefTypedValue val) {
  if (val.getType().getMemorySpace()) {
    return val.getType()
        .getMemorySpace()
        .template cast<gpu::AddressSpaceAttr>()
        .getValue();
  }
  return gpu::AddressSpace::Global;
}

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

// Simple rewrite pass to remove the stages and backward barriers in the
// prologue and in the Epilogue
struct RemoveStagesRewritePattern : public OpRewritePattern<rock::StageOp> {
  using OpRewritePattern<rock::StageOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::StageOp op,
                                PatternRewriter &rw) const override {
    Block *sourceBlock = &op.getRegion().front();
    rw.eraseOp(sourceBlock->getTerminator());
    bool isRemovableBarrier = (op.getName() == "__bwd_barrier__" &&
                               !dyn_cast<scf::ForOp>(op->getParentOp()));
    if (!sourceBlock->empty() && !isRemovableBarrier) {
      rw.inlineBlockBefore(sourceBlock, op);
    }
    rw.eraseOp(op);
    return failure();
  }
};

// Simple rewrite pass to remove back-to-back barriers
struct RemoveBackToBackBarriersRewritePattern
    : public OpRewritePattern<rock::LDSBarrierOp> {
  using OpRewritePattern<rock::LDSBarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::LDSBarrierOp op,
                                PatternRewriter &rw) const override {
    if (dyn_cast_or_null<rock::LDSBarrierOp>(op->getNextNode())) {
      op->getNextNode()->erase();
      return success();
    }
    return failure();
  }
};

// Simple rewrite pass to hoist operations that do not
// access LDS before the barriers
struct PushBarrierDownRewritePattern
    : public OpRewritePattern<rock::LDSBarrierOp> {
  using OpRewritePattern<rock::LDSBarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::LDSBarrierOp op,
                                PatternRewriter &rw) const override {
    Operation *nextOp = op->getNextNode();

    // Make sure that there is a nextOp
    if (!nextOp)
      return failure();

    // Don't go over the terminator
    if (!nextOp->getNextNode())
      return failure();

    // We assume that operations that have a body may modify LDS
    if (nextOp->getNumRegions() > 0)
      return failure();

    bool moveDown = true;
    // Make sure that the "nextOp" doesn't modify LDS
    for (Value operand : nextOp->getOperands()) {
      auto maybeAlloc = rock::findAlloc(operand);
      if (succeeded(maybeAlloc) &&
          getAddressSpace(*maybeAlloc) == AddressSpace::Workgroup)
        moveDown = false;
    }

    if (moveDown) {
      rw.setInsertionPointAfter(nextOp);
      rw.create<rock::LDSBarrierOp>(nextOp->getLoc());
      rw.eraseOp(op);
      return success();
    }
    return failure();
  }
};

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

    DenseMap<rock::StageOp, int> stageIter;

    // The following stages will run in parallel, but each
    // stage needs to start at the right iteration
    SmallVector<rock::StageOp> parallelStages;
    for (size_t j = t; j < stages.size(); j += ii) {
      stageIter[stages[j]] = iteration++;
      parallelStages.push_back(stages[j]);
    }

    // This is the set of multi-buffers needed at this time slot
    // to ensure that the stage can run in parallel without messing
    // each other's buffers
    DenseMap<rock::GpuAllocOp, int> thisMultiBuffers = multiBuffers;
    for (auto [alloc, factor] : thisMultiBuffers) {
      thisMultiBuffers[alloc] = 1;
    }

    // Optimization: if there is a RAW register dependency (addrspace(5)) swap
    // the stages. In this way, we don't need multibuffers (i.e., we read the
    // buffer first and then we write into it). From the point of view of the
    // stages, they don't care because they belong to different iterations. In
    // theory this could be applied to any buffer, but for proper memory, this
    // can be more expensive (i.e., you need barriers)
    DenseMap<int, SmallVector<int>> swapCandidates;
    DenseMap<int, SmallVector<int>> swapCandidatesR;

    // Go through the stages and take note of the possible swap candidates
    for (size_t i = 0; i < parallelStages.size(); i++) {
      for (size_t j = i + 1; j < parallelStages.size(); j++) {
        auto dependencies =
            getDependencies(parallelStages[i], parallelStages[j], dag);
        // Select all register dependencies
        SmallVector<DependencyType> privateDependencyTypes;
        for (auto [res, type] : dependencies)
          if (getAddressSpace(res) == AddressSpace::Private)
            privateDependencyTypes.push_back(type);
        // If there are no register dependencies, don't bother
        if (privateDependencyTypes.empty())
          continue;
        // See if they are all swappable
        bool canSwap = llvm::all_of(privateDependencyTypes,
                                    [&](auto type) { return (type == RAW); });
        // Add to the list of swap candidates
        if (canSwap) {
          swapCandidates[i].push_back(j);
          swapCandidatesR[j].push_back(i);
        }
      }
    }

    // Swap only pairs. If there are more intricate dependency
    // patterns just use multibuffers, since it is safer.
    for (auto [source, sinks] : swapCandidates) {
      bool singleSink = (sinks.size() == 1);
      bool singleSource = swapCandidatesR[sinks.back()].size() == 1;
      // Found a pair, now swap it
      if (singleSink && singleSource) {
        int sink = sinks.back();
        auto t = parallelStages[source];
        parallelStages[source] = parallelStages[sink];
        parallelStages[sink] = t;
      }
    }

    // Whatever resource is shared, we need to select among multiple buffers.
    for (size_t i = 0; i < parallelStages.size(); i++) {
      // The only resource that can conflict btween different stages is memory
      // If there are memory conflicts we can sort them via multibuffers. I.e.,
      // we can (logically) provide a different buffer for different cycles
      for (size_t j = i + 1; j < parallelStages.size(); j++) {
        auto dependencies =
            getDependencies(parallelStages[i], parallelStages[j], dag);
        for (auto [res, type] : dependencies) {
          if (type == WAR && getAddressSpace(res) == AddressSpace::Private)
            continue;
          thisMultiBuffers[res]++;
        }
      }
    }

    // Update the global multibuffers by merging in the factors needed for
    // the current time slot
    for (auto [buffer, factor] : thisMultiBuffers)
      if (factor > multiBuffers[buffer])
        multiBuffers[buffer] = factor;

    // Add the parallel stages
    for (auto stage : parallelStages)
      schedule.push_back({stage, stageIter[stage]});
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
                              StringRef name) {
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(stage);
  auto barrierStage = rewriter.create<rock::StageOp>(loc, name);
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

  auto maybeNumIterations =
      rock::computeConstDiff(forOp.getLowerBound(), forOp.getUpperBound());

  // If there is a loop, we probably need a backward barrier, i.e.,
  // an LDS barrier that takes the loop dependency into account
  const bool addBackwardBarrier =
      (!maybeNumIterations.has_value() ||
       (maybeNumIterations.has_value() && maybeNumIterations.value() > 1));

  DenseMap<rock::StageOp, int> timeSlotMap;
  int timeSlot = 0;
  for (auto stage : stages) {
    timeSlotMap[stage] = (timeSlot % initiationInterval);
    timeSlot++;
  }

  // Algorithm for barrier placment:
  // a. Add forward barriers to address the dependency in the basic block
  // b. Add backward barriers to account for loop carried dependency
  // c. Add empty stages to make the pipeline balanced, so that we can double up
  //    the initiation interval and let the pipeline transformation automaticall
  //    do the work for us
  DenseSet<rock::StageOp> forwardStages;

  // a. Place forward barriers
  for (auto [source, edges] : dag) {
    for (auto [sink, deps] : edges) {
      if (!forwardStages.contains(sink)) {
        forwardStages.insert(sink);
      }
    }
  }

  // b. If necessary, place a single backward barrier
  rock::StageOp backwardStage;
  if (addBackwardBarrier) {
    // b.1 find the last sink of a dependendency
    rock::StageOp lastSink;
    for (auto stage : llvm::reverse(stages)) {
      if (forwardStages.contains(stage)) {
        lastSink = stage;
        break;
      }
    }

    // b.2 find the first stage not in the same timeslot. This will be
    // the placement for the backward barrier.
    for (auto stage : stages) {
      if (timeSlotMap[stage] != timeSlotMap[lastSink]) {
        backwardStage = stage;
        break;
      }
    }
  }

  // c. Insert fwd/bwd barriers or empty stages
  for (auto stage : stages) {
    rock::StageOp additionalStage;
    if (forwardStages.contains(stage)) {
      additionalStage = placeEmptyStage(rewriter, loc, stage,
                                        /**isBarrier=*/true, "__fwd_barrier__");
    } else if (backwardStage == stage) {
      additionalStage = placeEmptyStage(rewriter, loc, stage,
                                        /**isBarrier=*/true, "__bwd_barrier__");
    } else {
      additionalStage = placeEmptyStage(
          rewriter, loc, stage, /**isBarrier=*/false, "__empty_stage__");
    }
    extendedStages.push_back(additionalStage);
    extendedStages.push_back(stage);
  }

  // d. Update the initiation interval
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
    return WalkResult::advance();
  });

  if (isNestedPipelining) {
    emitError(loc, "Nested pipelining is not supported yet!\n");
    return signalPassFailure();
  }

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
  if (isMultiBufferingFailed) {
    emitError(loc, "Multi buffering failed to apply!\n");
    return signalPassFailure();
  }

  // Check we didn't push memory too far
  DenseMap<AddressSpace, size_t> gpuMemoryBytes;
  func.walk([&](rock::GpuAllocOp alloc) {
    auto addressSpace = getAddressSpace(alloc);
    gpuMemoryBytes[addressSpace] += alloc.getType().getShape().back();
  });

  if (gpuMemoryBytes[AddressSpace::Workgroup] > size_t(64 * 1024)) {
    emitError(loc, "LDS consumption is more than 64K!\n");
    return signalPassFailure();
  }

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
      patterns.add<RemoveStagesRewritePattern, PushBarrierDownRewritePattern>(
          &getContext());
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
  }
}
