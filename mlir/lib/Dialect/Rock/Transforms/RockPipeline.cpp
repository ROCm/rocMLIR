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
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
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

// Simple rewrite pass to remove the stages
struct RemoveStagesRewritePattern : public OpRewritePattern<rock::StageOp> {
  using OpRewritePattern<rock::StageOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::StageOp op,
                                PatternRewriter &rw) const override {
    Block *sourceBlock = &op.getRegion().front();
    rw.eraseOp(sourceBlock->getTerminator());
    rw.inlineBlockBefore(sourceBlock, op);
    rw.eraseOp(op);
    return failure();
  }
};

// Given an operation and its operand, find out what kind of access (if any)
// the operation does on the operand
MemoryAccessType getOperandAccessType(Operation *op, Value operand) {
  if (hasEffect<MemoryEffects::Read>(op, operand)) {
    return MemoryAccessType::READ;
  } else if (hasEffect<MemoryEffects::Write>(op, operand)) {
    return MemoryAccessType::WRITE;
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
  // Get the resource map (stage->allocs) and its inverse (alloc->stages)

  // Create the dependency graph
  DagType dag = createDependencyGraph(stages, resources);

  for (auto res : resources)
    multiBuffers[res] = 1;

  // Start building the schedules
  //
  // Since we accept the stages from the user, we don't need to do any
  // analysis to determine what goes in each stage. We only have to group things
  // in set of stages of length II.
  //  For instance, if this is the unpipelined schedule decided by the user
  //  +t\s+=== 0 ===+
  //  + 0 +== S0  ==+
  //  +===+=========+
  //  + 1 +== S1  ==+
  //  +===+=========+
  //  + 2 +== S2  ==+
  //  +===+=========+
  //  + 3 +== S3  ==+
  //  +===+=========+
  // When the II == 3 it means that S0/S3 will run in parallel while S2 and S3
  // will run sequentially
  //  +t\s+=== 0 ===++=== 1 ===+
  //  + 0 +== S0  ==++== S3  ==+
  //  +===+=========++=========+
  //  + 1 +== S1  ==+
  //  +===+=========+
  //  + 2 +== S2  ==+
  //  +===+=========+
  // In this case, we reduced the time slots to 3, and we have 2 set of stages
  // runnning in parallel. Please note that conflicts can only happen between S0
  // and S3. If we increase II, we get to :
  //  +t\s+=== 0 ===++=== 1 ===+
  //  + 0 +== S0  ==++== S2  ==+
  //  +===+=========++=========+
  //  + 1 +== S1  ==++== S3  ==+
  //  +===+=========++=========+
  // Now we have only two time slots and 2 sets of parallel stages. conflicts
  // can happen between S0 and S2 and between S1 and S3. The above is capture in
  // the following algorithm. `t` is the time slot, i.e., the flowing of the
  // time, and goes from 0 to II. `stageSet` is the set of stages that run in
  // parallel.
  for (int t = 0; t < ii; t++) {
    int iteration = 0;

    // The following stages need to run in parallel
    SmallVector<rock::StageOp> parallelStages;
    for (size_t j = t; j < stages.size(); j += ii) {
      schedule.push_back({stages[j], iteration++});
      // The only resource that can conflict btween different stages is memory
      // If there are memory conflicts we can sort them via multibuffers. I.e.,
      // we can (logically) provide a different buffer for different cycles
      for (auto otherStage : parallelStages) {
        auto dependencies = getDependencies(otherStage, stages[j], dag);
        for (auto [res, type] : dependencies) {
          multiBuffers[res]++;
        }
      }
      parallelStages.push_back(stages[j]);
    }
  }
}

// Prune a dependency graph taking into account multi-buffers. Since
// multi-buffers are logically different for each iteration, if the dependency
// on a multi-buffer spans multiple iteration then it can be pruned
DagType pruneGraph(DagType dag, DenseMap<rock::StageOp, int> &iterationMap,
                   DenseMap<rock::GpuAllocOp, int> &factors) {
  DagType prunedGraph;
  // Multibuffers have the logical property of being unique for each iteration
  // of the loop Hence, if we know we are dealing with a multi-buffer and the
  // dependency concerns two different iteration. In other woeds, if stageA
  // accesses LDS in iteration i and stageB accesses LDS in iteration j stageA
  // and stageB have no dependencies as long as i!=j
  for (auto [sink, edges] : dag) {
    for (auto [source, deps] : edges) {
      DenseSet<std::pair<rock::GpuAllocOp, DependencyType>> newDeps;
      for (auto [alloc, type] : deps) {
        // Add the dependency only if it's not over a multiBuffer (factor==1) or
        // if it's over a multibuffer, if sink and source share the same
        // iteration
        if (factors[alloc] == 1 ||
            (factors[alloc] > 1 &&
             ((iterationMap[sink] % 2) == (iterationMap[source] % 2)))) {
          newDeps.insert({alloc, type});
        }
      }
      if (!newDeps.empty()) {
        prunedGraph[sink][source] = newDeps;
      }
    }
  }
  return prunedGraph;
}

// Barrier placement after the pipeline pass
void placeBarriers(IRRewriter &rewriter, Location loc,
                   DenseMap<rock::GpuAllocOp, int> &factors,
                   func::FuncOp func) {

  llvm::DenseSet<rock::GpuAllocOp> allocs;
  SmallVector<rock::StageOp> stages;

  DenseMap<StringRef, int> iterationCount;
  DenseMap<rock::StageOp, int> iterationMap;
  func.walk([&](rock::StageOp stageOp) {
    iterationMap[stageOp] = iterationCount[stageOp.getName()]++;
    stages.push_back(stageOp);
  });

  func.walk([&](rock::GpuAllocOp alloc) { allocs.insert(alloc); });

  DagType dag = createDependencyGraph(stages, allocs);

  // Given that we might be using multi-buffers, the graph
  // has to be pruned. I.e., if there is a [source]->alloc->[sink]
  // dependency, but alloc is a multi-buffer and [source] and
  // [sink] don't live in the same buffer, then we should not add
  // a barrier
  dag = pruneGraph(dag, iterationMap, factors);

  DenseSet<rock::StageOp> markedStages;
  for (auto [source, dependencies] : dag) {
    for (auto [sink, resources] : dependencies) {
      for (auto [alloc, type] : resources) {
        // We now have the [source] -> (resources) -> [sink]
        // dependency edge. We need to filter out register dependencies
        // (don't need barriers) and edges we are already considered (don't
        // insert barriers multiple times)
        auto addressSpace = alloc.getType()
                                .getMemorySpace()
                                .cast<gpu::AddressSpaceAttr>()
                                .getValue();
        if (addressSpace != gpu::AddressSpace::Workgroup ||
            markedStages.contains(sink))
          continue;

        // If you have any dependency that involves modifying LDS
        // we need to add a barrier
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(sink);
        rewriter.create<rock::LDSBarrierOp>(loc);
        markedStages.insert(sink);
      }
    }
  }
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

  // Start analysis. We need a schedule for each loop that we have to schedule
  // and we also need the multi-buffer factors for the different buffers in the
  // IR
  DenseMap<rock::GpuAllocOp, int> multiBufferFactors;
  llvm::DenseMap<scf::ForOp, ScheduleType> scheduleMap;

  func.walk([&](scf::ForOp forOp) -> WalkResult {
    SmallVector<rock::StageOp> stages;

    if (!forOp->hasAttrOfType<IntegerAttr>(rock::kInitiationIntervalAttrName))
      return WalkResult::advance();

    bool isNestedPipelining = false;
    forOp.getBody()->walk([&](scf::ForOp nestedFor) {
      if (nestedFor->hasAttrOfType<IntegerAttr>(
              rock::kInitiationIntervalAttrName))
        isNestedPipelining = true;
    });

    if (isNestedPipelining)
      return WalkResult::advance();

    // Get the initiation interval (II)
    int64_t ii = forOp->removeAttr(rock::kInitiationIntervalAttrName)
                     .dyn_cast<IntegerAttr>()
                     .getInt();

    forOp.walk([&](rock::StageOp stageOp) { stages.push_back(stageOp); });

    if (stages.empty())
      WalkResult::advance();

    LLVM_DEBUG(DBGS() << "Number of stages: " << stages.size() << "\n");
    LLVM_DEBUG(DBGS() << "Initiation Interval: " << ii << "\n");

    ScheduleType schedule;
    createSchedule(stages, allocs, ii, schedule, multiBufferFactors);

    scheduleMap[forOp] = schedule;

    return WalkResult::advance();
  });

  // Done with analysis: we have the schedule and the multiBuffers needed

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

  if (!isMultiBufferingFailed) {
    // Remove the barriers that don't belong to stages, if any
    for (auto [forOp, sched] : scheduleMap) {
      forOp.walk([](rock::LDSBarrierOp barrier) {
        if (!barrier->getParentOfType<rock::StageOp>())
          barrier->erase();
      });
    }

    // Actual pipeline
    {
      RewritePatternSet patterns(&getContext());
      mlir::scf::PipeliningOption options;
      options.getScheduleFn = [&](scf::ForOp op, ScheduleType &sched) {
        sched = scheduleMap[op];
      };
      scf::populateSCFLoopPipeliningPatterns(patterns, options);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    // Place the barriers back
    placeBarriers(rewriter, loc, multiBufferFactors, func);
  }

  // Remove stages
  if (removeStages) {
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveStagesRewritePattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
}
