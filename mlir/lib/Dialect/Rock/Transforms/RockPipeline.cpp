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

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Transforms/RockMultibuffer.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <map>
#include <set>

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
    return cast<gpu::AddressSpaceAttr>(val.getType().getMemorySpace())
        .getValue();
  }
  return gpu::AddressSpace::Global;
}

// Given an operation and its operand, find out what kind of access (if any)
// the operation does on the operand
MemoryAccessType getOperandAccessType(Operation *op, Value operand) {
  if (hasEffect<MemoryEffects::Write>(op, operand)) {
    return MemoryAccessType::WRITE;
  }
  if (hasEffect<MemoryEffects::Read>(op, operand)) {
    return MemoryAccessType::READ;
  }
  return MemoryAccessType::UNKNOWN;
}

// Returns true if compute-first scheduling should be used.
// Conditions: kernel function, double buffering (II == 1), blockSize == 8 *
// waveSize
static bool shouldUseComputeFirstSchedule(func::FuncOp func, int64_t ii) {
  // Condition 0: Must be a kernel function
  if (!func->hasAttr("kernel"))
    return false;

  // Condition 1: Double buffering (II == 1)
  if (ii != 1)
    return false;

  // Condition 2: blockSize == 8 * waveSize
  auto maybeBlockSize = rock::getBlockSize(func);
  if (failed(maybeBlockSize))
    return false;
  int64_t blockSize = maybeBlockSize.value().getValue().getSExtValue();

  StringRef arch = rock::getArchValue(func);
  if (arch.empty())
    return false;

  auto archInfo = rock::lookupArchInfo(arch);
  if (archInfo.waveSize == 0)
    return false;

  return blockSize == 8 * archInfo.waveSize;
}

// Reorder stages from memory-first [GlobalRead, LDSWrite, LDSRead, MMA]
// to compute-first order: [MMA, LDSRead, LDSWrite, GlobalRead]
static void reorderToComputeFirst(SmallVector<rock::StageOp> &stages) {
  SmallVector<rock::StageOp> mmaStages, ldsReadStages, ldsWriteStages,
      globalReadStages, otherStages;

  for (auto stage : stages) {
    StringRef name = stage.getName();
    if (name == "MMA")
      mmaStages.push_back(stage);
    else if (name == "LDSRead")
      ldsReadStages.push_back(stage);
    else if (name == "LDSWrite")
      ldsWriteStages.push_back(stage);
    else if (name == "GlobalRead")
      globalReadStages.push_back(stage);
    else
      otherStages.push_back(stage);
  }

  // Compute-first order: MMA, LDSRead, LDSWrite, GlobalRead
  stages.clear();
  stages.append(mmaStages);
  stages.append(ldsReadStages);
  stages.append(ldsWriteStages);
  stages.append(globalReadStages);
  stages.append(otherStages);
}

// Forward declaration of placeEmptyStage (defined later in file)
static rock::StageOp placeEmptyStage(IRRewriter &rewriter, Location loc,
                                     rock::StageOp stage, bool isBarrier,
                                     StringRef name);

// Insert scheduling hints at start and end of MMA stage body.
// This prevents the instruction scheduler from moving MFMA instructions
// outside of the compute cluster, enabling proper ping-pong scheduling.
// Structure:
//   sched_barrier(none) - prevent instruction movement into MMA
//   setPrio(1)          - prioritize MFMA execution
//   ... MFMA ops ...
//   setPrio(0)          - reset priority
//   sched_barrier(none) - prevent instruction movement out of MMA
static void insertSchedBarriersInMMAStage(IRRewriter &rewriter,
                                          rock::StageOp mmaStage) {
  Block &body = mmaStage.getRegion().front();
  Location loc = mmaStage.getLoc();

  // Insert at the start of the MMA stage:
  // 1. sched_barrier(none)
  // 2. setPrio(1)
  rewriter.setInsertionPointToStart(&body);
  amdgpu::SchedBarrierOp::create(rewriter, loc,
                                 amdgpu::sched_barrier_opt_enum::none);
  ROCDL::SetPrioOp::create(rewriter, loc, rewriter.getI16IntegerAttr(1));

  // Insert before the yield (at the end of the MMA stage):
  // 1. setPrio(0)
  // 2. sched_barrier(none)
  Operation *yieldOp = body.getTerminator();
  rewriter.setInsertionPoint(yieldOp);
  ROCDL::SetPrioOp::create(rewriter, loc, rewriter.getI16IntegerAttr(0));
  amdgpu::SchedBarrierOp::create(rewriter, loc,
                                 amdgpu::sched_barrier_opt_enum::none);
}

// Create compute-first stages with barriers for ping-pong scheduling.
// Uses 4 stages: MMA, LDSRead, LDSWrite, GlobalRead
// But assigns offsets to create a shallow prologue (1 iteration):
//   GlobalRead=0, LDSWrite=0 (same offset → both in prologue iter 0)
//   LDSRead=1, MMA=1 (same offset → both start in main loop)
// This minimizes operations between cond_barrier and first MFMA.
static void createComputeFirstStages(IRRewriter &rewriter, Location loc,
                                     SmallVector<rock::StageOp> &stages,
                                     SmallVector<rock::StageOp> &extendedStages,
                                     int64_t &initiationInterval) {
  reorderToComputeFirst(stages);

  rock::StageOp lastMMAStage = nullptr;
  rock::StageOp lastStage = nullptr;
  for (auto stage : stages) {
    if (stage.getName() == "MMA")
      lastMMAStage = stage;
    lastStage = stage;
  }

  for (auto stage : stages) {
    if (stage.getName() == "MMA")
      insertSchedBarriersInMMAStage(rewriter, stage);
  }

  // Build extended stages: MMA, LDSBarrier, LDSRead, LDSWrite, GlobalRead,
  // LDSBarrier
  for (auto stage : stages) {
    auto emptyStage = placeEmptyStage(rewriter, loc, stage,
                                      /*isBarrier=*/false, "__empty_stage__");
    extendedStages.push_back(emptyStage);
    extendedStages.push_back(stage);

    if (lastMMAStage && stage == lastMMAStage) {
      auto barrier = placeEmptyStage(rewriter, loc, stage, /*isBarrier=*/true,
                                     "__fwd_barrier__");
      extendedStages.push_back(barrier);
    }

    if (lastStage && stage == lastStage) {
      auto endBarrier = placeEmptyStage(rewriter, loc, stage,
                                        /*isBarrier=*/true, "__end_barrier__");
      extendedStages.push_back(endBarrier);
    }
  }

  initiationInterval *= 2;
}

// Create compute-first schedule with inverted stage offsets.
// This ensures that memory operations (GlobalRead, LDSWrite, LDSRead) appear
// in the prologue BEFORE MMA, but MMA executes FIRST in the main loop.
//
// For compute-first:
// - Execution order in main loop: MMA, LDSRead, LDSWrite, GlobalRead
// - Stage offsets (inverted): MMA=3, LDSRead=2, LDSWrite=1, GlobalRead=0
//
// Forward declarations for functions defined later in this file
DagType createDependencyGraph(ArrayRef<rock::StageOp> stages,
                              const SetVector<rock::GpuAllocOp> &resources);
DenseSet<std::pair<rock::GpuAllocOp, DependencyType>>
getDependencies(rock::StageOp stage0, rock::StageOp stage1, DagType &dag);

// This produces:
// - Prologue: GlobalRead(0), LDSWrite(0)+GlobalRead(1),
// LDSRead(0)+LDSWrite(1)+GlobalRead(2)
// - Main loop: MMA(I), LDSRead(I+1), LDSWrite(I+2), GlobalRead(I+3)
// - MMA now executes first in main loop but data is ready from prologue
static void
createComputeFirstSchedule(SmallVector<rock::StageOp> &extendedStages,
                           const SetVector<rock::GpuAllocOp> &resources,
                           int64_t ii, ScheduleType &schedule,
                           DenseMap<rock::GpuAllocOp, int> &multiBuffers) {
  // Apply RAW private-register swap: reorder parallel stages within each
  // time slot so readers execute before writers, avoiding multi-buffering.
  DagType dag = createDependencyGraph(extendedStages, resources);

  for (int t = 0; t < ii; t++) {
    SmallVector<rock::StageOp> parallelStages;
    for (size_t j = t; j < extendedStages.size(); j += ii)
      parallelStages.push_back(extendedStages[j]);

    DenseMap<unsigned, SmallVector<unsigned>> swapCandidates;
    DenseMap<unsigned, SmallVector<unsigned>> swapCandidatesR;

    for (size_t i = 0; i < parallelStages.size(); i++) {
      for (size_t j = i + 1; j < parallelStages.size(); j++) {
        auto dependencies =
            getDependencies(parallelStages[i], parallelStages[j], dag);
        SmallVector<DependencyType> privateDependencyTypes;
        for (auto [res, type] : dependencies)
          if (getAddressSpace(res) == AddressSpace::Private)
            privateDependencyTypes.push_back(type);
        if (privateDependencyTypes.empty())
          continue;
        bool canSwap = llvm::all_of(privateDependencyTypes,
                                    [&](auto type) { return (type == RAW); });
        if (canSwap) {
          swapCandidates[i].push_back(j);
          swapCandidatesR[j].push_back(i);
        }
      }
    }

    DenseMap<unsigned, SmallVector<unsigned>> mustPrecede;
    SmallVector<unsigned> inDegrees(parallelStages.size(), 0);
    bool hasConstraints = false;

    for (auto &[source, sinks] : swapCandidates) {
      bool singleSink = (sinks.size() == 1);
      bool singleSource = swapCandidatesR[sinks.back()].size() == 1;
      if (singleSink && singleSource) {
        unsigned sink = sinks.back();
        mustPrecede[sink].push_back(source);
        inDegrees[source]++;
        hasConstraints = true;
      }
    }

    if (hasConstraints) {
      std::set<unsigned> ready;
      for (unsigned i = 0; i < parallelStages.size(); i++) {
        if (inDegrees[i] == 0)
          ready.insert(i);
      }

      SmallVector<unsigned> order;
      while (!ready.empty()) {
        unsigned cur = *ready.begin();
        ready.erase(ready.begin());
        order.push_back(cur);
        for (unsigned next : mustPrecede[cur]) {
          if (--inDegrees[next] == 0)
            ready.insert(next);
        }
      }
      assert(order.size() == parallelStages.size() &&
             "cycle in private-memory RAW constraints");

      SmallVector<rock::StageOp> reordered;
      for (unsigned idx : order)
        reordered.push_back(parallelStages[idx]);

      // Write back ONLY the reordered stages at this time slot
      size_t k = 0;
      for (size_t j = t; j < extendedStages.size(); j += ii) {
        extendedStages[j] = reordered[k++];
      }

      LLVM_DEBUG({
        DBGS() << "Applied RAW swap at time slot " << t << ":";
        for (auto s : reordered)
          DBGS() << " " << s.getName();
        DBGS() << "\n";
      });
    }
  }

  // Collect actual computation stages (not barriers/empty stages) in order,
  // along with their positions in extendedStages
  SmallVector<std::pair<rock::StageOp, size_t>> computeStagesWithPos;
  for (size_t idx = 0; idx < extendedStages.size(); idx++) {
    StringRef name = extendedStages[idx].getName();
    if (name != "__empty_stage__" && name != "__fwd_barrier__" &&
        name != "__bwd_barrier__" && name != "__end_barrier__") {
      computeStagesWithPos.push_back({extendedStages[idx], idx});
    }
  }

  size_t numComputeStages = computeStagesWithPos.size();
  if (numComputeStages == 0)
    return;

  unsigned maxOffset = numComputeStages - 1;

  LLVM_DEBUG(DBGS() << "Creating compute-first schedule with "
                    << numComputeStages
                    << " compute stages, max offset = " << maxOffset << "\n");

  // Build schedule with inverted offsets
  // Order in schedule determines execution order in main loop
  // Offset determines when stage first appears in prologue (higher = later)
  for (size_t stageIdx = 0; stageIdx < extendedStages.size(); stageIdx++) {
    auto stage = extendedStages[stageIdx];
    StringRef name = stage.getName();
    unsigned offset = 0;

    if (name == "__empty_stage__" || name == "__fwd_barrier__" ||
        name == "__bwd_barrier__" || name == "__end_barrier__") {
      bool found = false;
      for (size_t i = 0; i < computeStagesWithPos.size(); i++) {
        if (stageIdx < computeStagesWithPos[i].second) {
          offset = maxOffset - i;
          found = true;
          break;
        }
      }
      if (!found) {
        offset = 0;
      }
    } else {
      // Standard dependency-based offsets (same as default schedule):
      // GlobalRead=0, LDSWrite=1, LDSRead=2, MMA=3
      // The prologue fills the pipeline in dependency order.
      // The stage ORDER in extendedStages (MMA, LDSRead, LDSWrite,
      // GlobalRead) determines execution order in the main loop body.
      if (name == "GlobalRead") {
        offset = 0;
      } else if (name == "LDSWrite") {
        offset = 1;
      } else if (name == "LDSRead") {
        offset = 2;
      } else if (name == "MMA") {
        offset = maxOffset;
      } else {
        for (size_t i = 0; i < computeStagesWithPos.size(); i++) {
          if (computeStagesWithPos[i].first == stage) {
            offset = maxOffset - i;
            break;
          }
        }
      }
    }

    schedule.emplace_back(stage.getOperation(), offset);
    LLVM_DEBUG(DBGS() << "  Stage '" << name << "' -> offset " << offset
                      << "\n");
  }

  // For compute-first, we typically need double buffering for LDS
  for (auto [alloc, factor] : multiBuffers) {
    if (factor < 2)
      multiBuffers[alloc] = 2;
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
    if (nextOp->getNumRegions() > 0 && !dyn_cast<linalg::GenericOp>(nextOp))
      return failure();

    bool moveDown = true;
    // Make sure that the "nextOp" doesn't modify LDS
    for (Value operand : nextOp->getOperands()) {
      auto maybeAlloc = rock::findGpuAlloc(operand);
      if (succeeded(maybeAlloc) &&
          getAddressSpace(*maybeAlloc) == AddressSpace::Workgroup)
        moveDown = false;
    }

    if (moveDown) {
      rw.setInsertionPointAfter(nextOp);
      rock::LDSBarrierOp::create(rw, nextOp->getLoc());
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
                              const SetVector<rock::GpuAllocOp> &allocs) {
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
        auto maybeAlloc = rock::findGpuAlloc(operand);
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
                    const SetVector<rock::GpuAllocOp> &resources, int64_t ii,
                    ScheduleType &schedule,
                    DenseMap<rock::GpuAllocOp, int> &multiBuffers) {
  // Create the dependency graph
  DagType dag = createDependencyGraph(stages, resources);

  // Definition of initiation interval (II)
  // Initiation interval is defined by number of cycles in each iteration of a
  // loop. Only one cycle is counted for parallel stages. Assume each stage
  // executes in one cycle.
  // Start building the schedules. Since we accept the stages from the user, we
  // don't need to do any analysis to determine what goes in each stage. Each
  // `II` number of stages will execute in sequence. All groups of `II`
  // stages execute in parallel.
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
  // and S3. If we decrease II, we generate the following pipeline:
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
    // theory this could be applied to any buffer, but for LDS memory this
    // can be more expensive (i.e., you need barriers)
    DenseMap<unsigned, SmallVector<unsigned>> swapCandidates;
    DenseMap<unsigned, SmallVector<unsigned>> swapCandidatesR;

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

    // Build a constraint graph from the swap candidates and use a
    // topological sort to determine the final stage execution order.
    // Each candidate pair (source=writer, sink=reader) becomes a
    // directed edge: sink -> source, meaning "reader before writer".
    // Multiple pairs can chain when they share a stage index, e.g.:
    //   LDSRead[2] writes %47, MMA[3] reads %47  -> MMA before LDSRead
    //   MMA[3]     writes %49, PP[4]  reads %49  -> PP  before MMA
    // Combined: PP < MMA < LDSRead (3-element rotation).
    DenseMap<unsigned, SmallVector<unsigned>> mustPrecede;
    SmallVector<unsigned> inDegrees(parallelStages.size(), 0);
    bool hasConstraints = false;

    for (auto [source, sinks] : swapCandidates) {
      bool singleSink = (sinks.size() == 1);
      bool singleSource = swapCandidatesR[sinks.back()].size() == 1;
      if (singleSink && singleSource) {
        unsigned sink = sinks.back();
        // Edge: sink (reader) -> source (writer).
        mustPrecede[sink].push_back(source);
        inDegrees[source]++;
        hasConstraints = true;
      }
    }

    if (hasConstraints) {
      // Kahn's algorithm: repeatedly emit the smallest-index node
      // whose in-degree is zero, then decrement its successors'
      // in-degrees. Using smallest-index as the tie-breaker keeps
      // unconstrained stages in their original relative order.
      std::set<unsigned> ready;
      for (unsigned i = 0; i < parallelStages.size(); i++) {
        if (inDegrees[i] == 0)
          ready.insert(i);
      }

      SmallVector<unsigned> order;
      while (!ready.empty()) {
        unsigned cur = *ready.begin();
        ready.erase(ready.begin());
        order.push_back(cur);
        for (unsigned next : mustPrecede[cur]) {
          if (--inDegrees[next] == 0)
            ready.insert(next);
        }
      }
      assert(order.size() == parallelStages.size() &&
             "cycle in private-memory RAW constraints; "
             "this indicates an unexpected circular dependency "
             "between pipeline stages");

      SmallVector<rock::StageOp> reordered;
      for (unsigned idx : order)
        reordered.push_back(parallelStages[idx]);
      parallelStages = reordered;
    }

    // Whatever resource is shared, we need to select among multiple buffers.
    for (size_t i = 0; i < parallelStages.size(); i++) {
      // The only resource that can conflict between different stages is memory
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
    for (auto stage : parallelStages) {
      schedule.emplace_back(stage, stageIter[stage]);
    }
  }
}

// Prune a dependency graph taking into account multi-buffers. Since
// multi-buffers are logically different for each iteration, if the dependency
// on a multi-buffer spans multiple iteration then it can be pruned
DagType pruneGraph(const DagType &dag) {
  DagType prunedGraph;
  // Multibuffers have the logical property of being unique for each iteration
  // of the loop Hence, if we know we are dealing with a multi-buffer and the
  // dependency concerns two different iteration. In other words, if stageA
  // accesses LDS in iteration i and stageB accesses LDS in iteration j stageA
  // and stageB have no dependencies as long as i!=j
  for (const auto &[source, edges] : dag) {
    for (const auto &[sink, deps] : edges) {
      DenseSet<std::pair<rock::GpuAllocOp, DependencyType>> newDeps;
      for (auto [alloc, type] : deps) {
        if (getAddressSpace(alloc) != gpu::AddressSpace::Workgroup)
          continue;
        newDeps.insert({alloc, type});
      }
      if (!newDeps.empty())
        prunedGraph[source][sink] = newDeps;
    }
  }
  return prunedGraph;
}

// Utility function to place an empty stage before or after another `stage`. The
// empty stage will contain an `lds_barrier` if `isBarrier` is set to true
static rock::StageOp placeEmptyStage(IRRewriter &rewriter, Location loc,
                                     rock::StageOp stage, bool isBarrier,
                                     StringRef name) {
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(stage);
  auto barrierStage = rock::StageOp::create(rewriter, loc, name);
  rewriter.setInsertionPointToStart(&barrierStage.getRegion().emplaceBlock());
  if (isBarrier) {
    rock::LDSBarrierOp::create(rewriter, loc);
  }
  rock::YieldOp::create(rewriter, loc);
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
                   SetVector<rock::GpuAllocOp> &allocs,
                   SmallVector<rock::StageOp> &extendedStages,
                   int64_t &initiationInterval, int64_t numIterations) {
  DagType dag = createDependencyGraph(stages, allocs);
  // prune non-LDS dependencies
  dag = pruneGraph(dag);

  // If there is a loop, we probably need a backward barrier, i.e.,
  // an LDS barrier that takes the loop dependency into account
  const bool addBackwardBarrier = numIterations > 1;

  DenseMap<rock::StageOp, int> timeSlotMap;
  int timeSlot = 0;
  for (auto stage : stages) {
    timeSlotMap[stage] = (timeSlot % initiationInterval);
    timeSlot++;
  }

  // Algorithm for barrier placement:
  // a. Add forward barriers to address the dependency in the basic block
  // b. Add backward barriers to account for loop carried dependency
  // c. Add empty stages to make the pipeline balanced, so that we can double up
  //    the initiation interval and let the pipeline transformation
  //    automatically do the work for us
  DenseSet<rock::StageOp> forwardStages;

  // a. Place forward barriers
  for (const auto &[source, edges] : dag) {
    for (const auto &[sink, deps] : edges) {
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

bool checkIfPipeliningSupported(scf::ForOp forOp) {
  auto rockPipelineAttrName = rock::PipelineAttr::getMnemonic();
  while (scf::ForOp parentLoop = forOp->getParentOfType<scf::ForOp>()) {
    if (parentLoop->hasAttr(rockPipelineAttrName)) {
      return true;
    }
    forOp = parentLoop;
  }
  return false;
}

// Return a list of the loops in the function `func` that represents
// in level order in a list.
SmallVector<scf::ForOp> collectLoopLevels(mlir::func::FuncOp func) {
  SmallVector<scf::ForOp> loops;

  unsigned curLevelLen = 0;
  func.walk([&](scf::ForOp forOp) {
    // A loop is top-level if there is no enclosing outer loop.
    // Traverse backwards through the parent chain until we reach the function.
    // If we encounter a LoopLikeOp along the way, this is not a top-level loop.
    bool isTopLevel = true;
    Operation *parentOp = forOp->getParentOp();
    while (parentOp && parentOp != func) {
      if (isa<LoopLikeOpInterface>(parentOp)) {
        isTopLevel = false;
        break;
      }
      parentOp = parentOp->getParentOp();
    }

    if (isTopLevel) {
      loops.push_back(forOp);
      curLevelLen++;
    }
  });

  unsigned curLevelPos = 0;
  while (curLevelLen) {
    unsigned nextLevelLen = 0;
    for (unsigned i = 0; i < curLevelLen; i++) {
      scf::ForOp currParent = loops[curLevelPos + i];
      currParent.getBody()->walk([&](scf::ForOp forOp) {
        if (forOp->getParentOp() == currParent) {
          loops.push_back(forOp);
          nextLevelLen++;
        }
      });
    }
    curLevelPos += curLevelLen;
    curLevelLen = nextLevelLen;
  }

  return loops;
}

void adjustInitiationInterval(int64_t numIterations, size_t numStages,
                              int64_t &ii) {
  int64_t numParallelStages = llvm::divideCeil(numStages, ii);
  // calculate number of prologue executions
  int64_t numPrologues = numParallelStages - 1;
  // if number of iterations are less than number of prologues that are going
  // to be emitted, it will not result in correct output therefore increase II
  // until that condition becomes false. This can help achieve maximum loop
  // pipelining
  while (numIterations < numPrologues) {
    ii++;
    LLVM_DEBUG(DBGS() << "Adjusted II to  " << ii << "\n");
    numParallelStages = llvm::divideCeil(numStages, ii);
    numPrologues = numParallelStages - 1;
  }
  LLVM_DEBUG(DBGS() << "Number of parallel stages: " << numParallelStages
                    << "\n");
  LLVM_DEBUG(DBGS() << "Number of Prologues: " << numPrologues << "\n");
  // num of prologues == number of epilogues
  LLVM_DEBUG(DBGS() << "Number of Epilogues: " << numPrologues << "\n");
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

  // Track loops that use compute-first scheduling for later barrier insertion
  SmallVector<scf::ForOp> computeFirstLoops;

  auto rockPipelineAttrName = rock::PipelineAttr::getMnemonic();

  // Maybe this might be a bit too much for now, but we are a compiler
  // after all. So let's try to be generic. We collect all loops
  // in a level traversal order of the loop nests
  SmallVector<scf::ForOp> loops = collectLoopLevels(func);

  // Filter out loops that don't need pipelining
  // and check for nested-pipelining for loops that do need to
  // be pipelened. We pipeline from the innermost to the outermost loop,
  // hence traverse the list in a reverse order (from the bottom levels to
  // the top levels)
  SmallVector<scf::ForOp> loopsToPipeline;
  for (auto forOp : llvm::reverse(loops)) {
    if (forOp->hasAttr(rockPipelineAttrName)) {
      if (checkIfPipeliningSupported(forOp)) {
        forOp.emitError("Nested pipelining is not supported yet");
        return signalPassFailure();
      }
      loopsToPipeline.push_back(forOp);
    }
  }

  if (loopsToPipeline.empty()) {
    LLVM_DEBUG(DBGS() << "No loops to pipeline\n");

    if (removeStages) {
      // Remove all stages
      RewritePatternSet patterns(&getContext());
      patterns.add<RemoveStagesRewritePattern>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
    }
  } else {
    LLVM_DEBUG(DBGS() << "Found " << loopsToPipeline.size()
                      << " loops to pipeline\n");

    // Allocs before we transform them into multibuffers
    llvm::SetVector<rock::GpuAllocOp> singleAllocs;
    func.walk([&](rock::GpuAllocOp alloc) { singleAllocs.insert(alloc); });

    // Always (try to) multi-buffer by one and store the new
    // allocs in a set
    // Store multibuffers in "multiAllocs" and store all buffers
    // including private and global in "resources"
    llvm::SetVector<rock::GpuAllocOp> multiAllocs;
    llvm::SetVector<rock::GpuAllocOp> resources;
    for (auto alloc : singleAllocs) {
      SmallVector<rock::GpuAllocOp> newAllocs;
      if (succeeded(rock::multiBuffer(rewriter, alloc, newAllocs, 1, true))) {
        multiAllocs.insert(newAllocs.back());
        resources.insert(newAllocs.back());
      } else {
        resources.insert(alloc);
      }
    }

    // Collect the global resources (i.e., the memory allocations)
    // Note: we can only have two kind of memory:
    // - Registers
    // - LDS
    DenseMap<rock::GpuAllocOp, int> multiBufferFactors;
    for (auto res : multiAllocs)
      multiBufferFactors[res] = 1;

    for (auto forOp : loopsToPipeline) {
      SmallVector<rock::StageOp> stages;

      // Get the initiation interval (II)
      int64_t ii =
          dyn_cast<rock::PipelineAttr>(forOp->removeAttr(rockPipelineAttrName))
              .getInitiationInterval();

      forOp.walk([&](rock::StageOp stageOp) { stages.push_back(stageOp); });

      forOp.walk([](rock::LDSBarrierOp barrier) {
        if (!barrier->getParentOfType<rock::StageOp>())
          barrier->erase();
      });

      if (stages.empty())
        continue;

      // Check if compute-first scheduling should be used
      bool useComputeFirst = shouldUseComputeFirstSchedule(func, ii);

      LLVM_DEBUG(DBGS() << "Number of stages: " << stages.size() << "\n");
      LLVM_DEBUG(DBGS() << "Initiation Interval: " << ii << "\n");
      LLVM_DEBUG(if (useComputeFirst) DBGS()
                 << "Using compute-first scheduling\n");
      size_t numStages = stages.size();
      auto maybeNumIterations = forOp.getStaticTripCount();
      if (!maybeNumIterations.has_value()) {
        forOp.emitError(
            "Number of iterations are unknown while doing rock-pipeline");
        return signalPassFailure();
      }
      int64_t numIterations = maybeNumIterations.value().getSExtValue();
      if (!isConstantIntValue(forOp.getStep(), 1)) {
        forOp.emitError(
            "Step size other one is not permitted in rock-pipeline");
        return signalPassFailure();
      }
      adjustInitiationInterval(numIterations, numStages, ii);

      // Insert the barriers as new stages
      SmallVector<rock::StageOp> extendedStages;
      ScheduleType schedule;
      if (useComputeFirst) {
        // Triton-style ping-pong: reorder stages to MMA-first but use
        // STANDARD dependency-based offsets so the prologue is normal.
        createComputeFirstStages(rewriter, loc, stages, extendedStages, ii);
        createComputeFirstSchedule(extendedStages, resources, ii, schedule,
                                   multiBufferFactors);
      } else {
        placeBarriers(rewriter, loc, forOp, stages, multiAllocs, extendedStages,
                      ii, numIterations);
        createSchedule(extendedStages, resources, ii, schedule,
                       multiBufferFactors);
      }

      // Remember the parent block and position before pipelining
      Block *parentBlock = forOp->getBlock();
      Operation *opBeforeLoop = forOp->getPrevNode();

      RewritePatternSet patterns(&getContext());
      mlir::scf::PipeliningOption options;
      options.getScheduleFn = [&](scf::ForOp curFurOp, ScheduleType &sched) {
        if (curFurOp == forOp)
          sched = schedule;
      };

      scf::populateSCFLoopPipeliningPatterns(patterns, options);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();

      // For compute-first scheduling, insert barriers around the pipelined loop
      // Structure: LDSBarrier, cond_barrier, scf.for{...}, cond_barrier,
      // LDSBarrier
      if (useComputeFirst) {
        // Find the main pipelined loop (the one with the most/dynamic
        // iterations)
        scf::ForOp mainForOp = nullptr;
        int64_t maxTripCount = -1;

        Operation *searchStart =
            opBeforeLoop ? opBeforeLoop->getNextNode() : &parentBlock->front();

        for (Operation *op = searchStart; op != nullptr;
             op = op->getNextNode()) {
          if (auto candidateFor = dyn_cast<scf::ForOp>(op)) {
            std::optional<llvm::APInt> tripCountOpt =
                candidateFor.getStaticTripCount();
            if (tripCountOpt.has_value()) {
              int64_t tripCount = tripCountOpt->getSExtValue();
              if (tripCount > maxTripCount) {
                maxTripCount = tripCount;
                mainForOp = candidateFor;
              }
            } else {
              // Dynamic trip count - assume this is the main loop
              mainForOp = candidateFor;
              break;
            }
          }
        }

        if (mainForOp) {
          StringRef arch = rock::getArchValue(func);
          auto archInfo = rock::lookupArchInfo(arch);
          int64_t waveSize = archInfo.waveSize;

          // Insert cond_barrier right before the main loop.
          // All prologue ops are before this point (executed by all threads).
          rewriter.setInsertionPoint(mainForOp);

          Value workitemId = rock::WorkitemIdOp::create(
              rewriter, loc, rewriter.getIndexType());
          Value threshold =
              arith::ConstantIndexOp::create(rewriter, loc, 4 * waveSize);
          Value isWaveGroup1 = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::uge, workitemId, threshold);

          amdgpu::SchedBarrierOp::create(rewriter, loc,
                                         amdgpu::sched_barrier_opt_enum::none);
          rock::CondBarrierOp::create(rewriter, loc, isWaveGroup1);
          amdgpu::SchedBarrierOp::create(rewriter, loc,
                                         amdgpu::sched_barrier_opt_enum::none);

          LLVM_DEBUG(DBGS()
                     << "Inserted sched_barrier + cond_barrier + sched_barrier "
                        "before main loop\n");

          // Insert after the main loop: cond_barrier, LDSBarrier
          rewriter.setInsertionPointAfter(mainForOp);
          Value isWaveGroup0 = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::ult, workitemId, threshold);
          rock::CondBarrierOp::create(rewriter, loc, isWaveGroup0);
          rock::LDSBarrierOp::create(rewriter, loc);

          LLVM_DEBUG(DBGS()
                     << "Inserted cond_barrier + LDSBarrier after main loop\n");

          // Disable LICM on the loop
          auto trueAttr = rewriter.getBoolAttr(true);
          auto licmAttr =
              LLVM::LoopLICMAttr::get(ctx, /*disable=*/trueAttr,
                                      /*versioningDisable=*/trueAttr);
          auto loopAnnotation = LLVM::LoopAnnotationAttr::get(
              ctx, /*disableNonforced=*/{}, /*vectorize=*/{},
              /*interleave=*/{}, /*unroll=*/{}, /*unrollAndJam=*/{},
              /*licm=*/licmAttr, /*distribute=*/{}, /*pipeline=*/{},
              /*peeled=*/{}, /*unswitch=*/{}, /*mustProgress=*/{},
              /*isVectorized=*/{}, /*startLoc=*/{}, /*endLoc=*/{},
              /*parallelAccesses=*/{});
          mainForOp->setAttr("llvm.loop_annotation", loopAnnotation);

          computeFirstLoops.push_back(mainForOp);
        }
      }
    }

    // Remulti-buffer(if needed). Now we know what all the loops need, hence
    // we can safely allocate the right amount of resources in the function
    for (auto [alloc, factor] : multiBufferFactors) {
      SmallVector<rock::GpuAllocOp> newAllocs;
      if (factor > 1) {
        if (failed(rock::updateMultiBuffer(rewriter, loc, {alloc}, newAllocs,
                                           factor))) {

          alloc.emitError()
              << "Failed to update multibuffer with factor " << factor << "\n";
          return signalPassFailure();
        }
      }
    }

    // Cleanup the stages
    {
      if (removeStages) {
        RewritePatternSet patternsPushBarrier(&getContext());
        // run PushBarrierDownRewritePattern before RemoveStagesRewritePattern,
        // because the latter will remove the stages and their terminators
        patternsPushBarrier.add<PushBarrierDownRewritePattern>(ctx);
        if (failed(applyPatternsGreedily(func, std::move(patternsPushBarrier))))
          return signalPassFailure();

        // run RemoveStagesRewritePattern before
        // RemoveBackToBackBarriersRewritePattern, because the latter expects to
        // find no stages
        RewritePatternSet patternsRemoveStages(&getContext());
        patternsRemoveStages.add<RemoveStagesRewritePattern>(ctx);
        if (failed(
                applyPatternsGreedily(func, std::move(patternsRemoveStages))))
          return signalPassFailure();

        RewritePatternSet patternsBackToBack(&getContext());
        patternsBackToBack.add<RemoveBackToBackBarriersRewritePattern>(ctx);
        if (failed(applyPatternsGreedily(func, std::move(patternsBackToBack))))
          return signalPassFailure();
      }
    }
  }

  // Pair each lds_barrier inside compute-first loops with a sched_barrier.
  for (auto mainForOp : computeFirstLoops) {
    if (!mainForOp)
      continue;
    mainForOp.getBody()->walk([&](rock::LDSBarrierOp barrierOp) {
      rewriter.setInsertionPointAfter(barrierOp);
      amdgpu::SchedBarrierOp::create(rewriter, loc,
                                     amdgpu::sched_barrier_opt_enum::none);
    });
  }
}
