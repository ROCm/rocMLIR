//===- ReuseLDS - MLIR Rock ops lowering passes -----===//
//
// Copyright 2024 The MLIR Authors.
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
// ============================================================
//
// This pass re-uses LDS memory by using the lifetime annotations (rock.live_in,
// rock.live_out)
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREUSELDSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-reuse-lds"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockReuseLDSPass
    : public rock::impl::RockReuseLDSPassBase<RockReuseLDSPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

struct InterferenceData {
  SmallVector<GpuAllocOp> allocs;
  SmallVector<LiveInOp> liveIns;
  SmallVector<LiveOutOp> liveOuts;
  llvm::SmallDenseMap<GpuAllocOp, llvm::SetVector<GpuAllocOp>>
      interferenceGraph;
};

struct GraphColoringInfo {
  llvm::MapVector<int64_t, int64_t> colorSizes;
  SmallVector<std::tuple<GpuAllocOp, int64_t, int64_t>> allocOffsets;
  llvm::SmallDenseMap<GpuAllocOp, SetVector<int64_t>> colorAssignment;
};

// Check wheter we use more than the LDS size the arch provides
static LogicalResult checkLDSSize(Operation *op, int64_t ldsBytes) {
  // Check for arch limitations exceeded
  FailureOr<StringAttr> maybeArch = getArch(op);
  if (succeeded(maybeArch)) {
    return checkLDSSize(maybeArch.value(), ldsBytes);
  }
  return success();
}

static void assignColors(
    GpuAllocOp alloc, llvm::SetVector<int64_t> &usedColors,
    llvm::MapVector<int64_t, int64_t> &colorMemSize,
    llvm::SmallDenseMap<GpuAllocOp, SetVector<int64_t>> &colorAssignment) {
  const std::optional<int64_t> maybeRequiredSize =
      getWorkgroupMemorySize(alloc.getOutput().getType());
  assert(maybeRequiredSize.has_value());
  const int64_t requiredSize = maybeRequiredSize.value();
  assert(requiredSize > 0);
  int64_t color = 0;
  int64_t allocatedSize = 0;

  while (allocatedSize < requiredSize) {
    if (usedColors.contains(color)) {
      // found a used color, all the assigned colors must be consecutive
      // let's start again
      allocatedSize = 0;
      colorAssignment[alloc].clear();
      // Find the first available color
      while (usedColors.contains(color)) {
        color++;
      }
    }
    colorAssignment[alloc].insert(color);
    // New color
    if (!colorMemSize.contains(color)) {
      // Make this aligned to 128 bits
      colorMemSize[color] = llvm::alignTo(requiredSize - allocatedSize, 16);
    }
    allocatedSize += colorMemSize[color];
    color++;
  }
}

/// This is a greedy graph coloring algorithm.
/// There are some changes to make it work for LDS, the main one:
/// each alloc can be assigned more than one color, this is because
/// in graph coloring all vertex are assumed to be the same size
/// (for example, register allocation).
/// Example: A=GpuAllocOp(1kB), B=GpuAllocOp(1kB), C=GpuAllocOp(2kB)
///           A <--> B     C (A and B have an edge, C disjoint)
/// In this case, we can assign colors: A -> {0}, B -> {1}, and C -> {0, 1}.
/// Colors 0 and 1 are 1kB each.
/// Note: If an alloc has more than one color assigned, they have to be
///       consecutive.
static GraphColoringInfo
graphColoring(const InterferenceData &interferenceData) {
  GraphColoringInfo graphColoringInfo;
  llvm::MapVector<int64_t, int64_t> colorMemSize;

  SmallVector<GpuAllocOp> sortedAllocs(interferenceData.allocs);

  // Sort by alloc size
  llvm::sort(sortedAllocs, [](GpuAllocOp &a, GpuAllocOp &b) {
    auto aSize = getWorkgroupMemorySize(a.getOutput().getType());
    auto bSize = getWorkgroupMemorySize(b.getOutput().getType());
    assert(aSize.has_value() && bSize.has_value());
    return aSize.value() < bSize.value();
  });
  // Assign colors using greedy algorithm
  for (GpuAllocOp alloc : sortedAllocs) {
    llvm::SetVector<int64_t> usedColors;
    assert(interferenceData.interferenceGraph.contains(alloc));
    for (const GpuAllocOp neighbor :
         interferenceData.interferenceGraph.at(alloc)) {
      if (graphColoringInfo.colorAssignment.contains(neighbor)) {
        for (int64_t color : graphColoringInfo.colorAssignment[neighbor]) {
          usedColors.insert(color);
        }
      }
    }

    // Assign a set of colors
    assignColors(alloc, usedColors, colorMemSize,
                 graphColoringInfo.colorAssignment);
  }

  // If we replace all GpuAllocOps with a single one, we run into
  // aliasing issues that cause performance regressions.
  // In order to avoid that, we first merge colors so that each
  // GpuAlloc is assigned a single color and an offset.
  // Then, we can generate more than one GpuAllocOp and improve
  // aliasing issues.
  // This might be removed in the future if aliasing issues are solved.
  llvm::SmallDenseMap<int64_t, SetVector<int64_t>> colorsToMerge;
  llvm::SmallDenseMap<int64_t, int64_t> oldColorToNew;
  for (GpuAllocOp alloc : sortedAllocs) {
    int64_t firstColor = graphColoringInfo.colorAssignment[alloc][0];
    // if the color has been replaced already
    if (oldColorToNew.contains(firstColor)) {
      int64_t newColor = oldColorToNew[firstColor];
      // assign all the colors of the current 'alloc' to newColor
      for (int64_t color : graphColoringInfo.colorAssignment[alloc]) {
        oldColorToNew[color] = newColor;
        colorsToMerge[newColor].insert(color);
      }
    } else {
      // the color has not been replaced yet
      for (auto [i, color] :
           llvm::enumerate(graphColoringInfo.colorAssignment[alloc])) {
        // merge all non-first colors with the first one
        if (i > 0) {
          oldColorToNew[color] = firstColor;
          colorsToMerge[firstColor].insert(color);
          // if the current 'color' has merged some colors,
          // merge its colors to 'firstColor'
          if (colorsToMerge.contains(color)) {
            for (int64_t otherColor : colorsToMerge[color]) {
              oldColorToNew[otherColor] = firstColor;
              colorsToMerge[firstColor].insert(otherColor);
            }
            colorsToMerge.erase(color);
          }
        }
      }
    }
  }

  // Compute offsets and new sizes (after merging)
  llvm::MapVector<int64_t, int64_t> mergedColorMemSize(colorMemSize);
  llvm::MapVector<int64_t, int64_t> colorOffset;
  for (auto [color, _] : colorMemSize) {
    colorOffset[color] = 0;
  }
  for (const auto &[color, oldColors] : colorsToMerge) {
    for (int64_t oldColor : oldColors) {
      assert(oldColor > color);
      colorOffset[oldColor] = mergedColorMemSize[color];
      mergedColorMemSize[color] += mergedColorMemSize[oldColor];
      mergedColorMemSize.erase(oldColor);
    }
  }

  // Compute information per GpuAllocOp
  llvm::SetVector<int64_t> usedColors;
  for (const GpuAllocOp alloc : interferenceData.allocs) {
    assert(graphColoringInfo.colorAssignment.contains(alloc));
    int64_t oldColor = graphColoringInfo.colorAssignment[alloc][0];
    assert(colorOffset.contains(oldColor));
    int64_t offset = colorOffset[oldColor];
    int64_t newColor =
        (oldColorToNew.contains(oldColor)) ? oldColorToNew[oldColor] : oldColor;
    graphColoringInfo.allocOffsets.push_back(
        std::tuple(alloc, newColor, offset));
  }

  graphColoringInfo.colorSizes = mergedColorMemSize;
  return graphColoringInfo;
}

// Create the interference graph.
// Note that we compute the interference graph at gpu alloc level. So, if
// two allocs have interference sometimes and sometimes not, that is not taken
// into account. This is something that might need to get improved if we
// generate IR like this:
// clang-format off

// %a = alloc(), %b = alloc() 
// Note there's no interference here: 
// live_in(%a) 
// use(%a) 
// live_out(%a) 
// live_in(%b) 
// use(%b)
// live_out(%b)

// Later on, on the IR we see:
// live_in(%a)
// use(%a)
// live_in(%b)
// use(%b)
// live_out(%b)
// live_out(%a)

// clang-format on
// There's interference in this case. But the current code, will determine
// interference for allocs %a and %b. And will not be able to reuse LDS for
// the first case.
static FailureOr<InterferenceData> createInterferenceGraph(func::FuncOp &func) {
  InterferenceData interferenceData;
  SetVector<GpuAllocOp> currentAllocs;
  llvm::SmallDenseMap<Value, GpuAllocOp> memrefToAlloc;

  WalkResult walkResult = func.walk<WalkOrder::PreOrder>([&](Operation *op)
                                                             -> WalkResult {
    if (auto gpuAlloc = dyn_cast<GpuAllocOp>(op)) {
      auto type = gpuAlloc.getOutput().getType();

      std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
      if (maybeSize.has_value()) {
        int64_t size = maybeSize.value();
        LLVM_DEBUG(llvm::dbgs() << "Found rock.alloc of " << size << " bytes "
                                << gpuAlloc << "\n");

        memrefToAlloc[gpuAlloc.getOutput()] = gpuAlloc;
        interferenceData.allocs.push_back(gpuAlloc);
        interferenceData.interferenceGraph[gpuAlloc] = {};
      }
    } else if (auto liveIn = dyn_cast<LiveInOp>(op)) {
      auto type = liveIn.getMemref().getType();

      if (!memrefToAlloc.contains(liveIn.getMemref())) {
        LLVM_DEBUG(llvm::dbgs() << "Called rock.live_in multiple times?\n");
        return WalkResult::interrupt();
      }

      auto gpuAlloc = memrefToAlloc[liveIn.getMemref()];
      std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
      if (maybeSize.has_value()) {
        int64_t size = maybeSize.value();
        LLVM_DEBUG(llvm::dbgs() << "Found rock.live_in of " << size
                                << " bytes, " << liveIn << "\n");

        if (currentAllocs.contains(gpuAlloc)) {
          LLVM_DEBUG(llvm::dbgs() << "Called rock.live_in multiple times?\n");
          return WalkResult::interrupt();
        }

        // add vertex and connections
        for (auto alloc : currentAllocs) {
          interferenceData.interferenceGraph[alloc].insert(gpuAlloc);
          interferenceData.interferenceGraph[gpuAlloc].insert(alloc);
        }
        currentAllocs.insert(gpuAlloc);
        interferenceData.liveIns.push_back(liveIn);
      }
    } else if (auto liveOut = dyn_cast<LiveOutOp>(op)) {
      auto type = liveOut.getMemref().getType();
      std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
      if (maybeSize.has_value()) {
        int64_t size = maybeSize.value();
        LLVM_DEBUG(llvm::dbgs() << "Found rock.live_out of " << size
                                << " bytes, " << liveOut << "\n");

        if (!memrefToAlloc.contains(liveOut.getMemref())) {
          LLVM_DEBUG(llvm::dbgs() << "LiveOut before GpuAlloc?\n");
          return WalkResult::interrupt();
        }
        bool erased = currentAllocs.remove(memrefToAlloc[liveOut.getMemref()]);
        if (!erased) {
          LLVM_DEBUG(llvm::dbgs() << "Called rock.live_out multiple times?\n");
          return WalkResult::interrupt();
        }
        interferenceData.liveOuts.push_back(liveOut);
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    LLVM_DEBUG(llvm::dbgs() << "Walk interrupted\n");
    return failure();
  }

  // same number of rock.live_in and rock.live_out
  if (interferenceData.liveOuts.size() != interferenceData.liveIns.size() ||
      !currentAllocs.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "There should be an equal number of rock.live_in "
                  "and rock.live_out (for LDS)\n");
    return failure();
  }

  if (interferenceData.allocs.size() !=
      interferenceData.interferenceGraph.size()) {
    LLVM_DEBUG(llvm::dbgs() << "Some allocs have no liveness annotations\n");
    return failure();
  }

  return interferenceData;
}

// Add LDS barriers when we reuse LDS for different buffers.
// If we reuse LDS for buffers A and B, this is to avoid that we start writing
// to B, while some threads are still loading from A.
static LogicalResult addLDSBarriers(IRRewriter &rewriter, func::FuncOp &func,
                                    const GraphColoringInfo &coloringInfo) {
  // we keep track of used colors during the walk
  SetVector<int64_t> usedColors;
  // SetVector<int64_t> usedColorsSinceLastBarrier;
  llvm::SmallDenseMap<int64_t, GpuAllocOp> usedColorsSinceLastBarrier;
  SmallVector<LiveInOp> addBarrier;

  auto filterUsedColorsSinceLastBarrier =
      [&](GpuAllocOp alloc) -> llvm::SetVector<int64_t> {
    llvm::SetVector<int64_t> newUsedColors;
    for (auto pair : usedColorsSinceLastBarrier) {
      if (pair.getSecond() != alloc) {
        newUsedColors.insert(pair.getFirst());
      }
    }
    return newUsedColors;
  };

  auto updateUsedColorsSinceLastBarrier =
      [&]() -> llvm::SmallDenseMap<int64_t, GpuAllocOp> {
    // update 'usedColorsSinceLastBarrier': remove colors not found in
    // 'usedColors'
    llvm::SmallDenseMap<int64_t, GpuAllocOp> newUsedColors;
    for (auto color : usedColorsSinceLastBarrier.keys()) {
      if (llvm::is_contained(usedColors, color))
        newUsedColors[color] = usedColorsSinceLastBarrier[color];
    }

    return newUsedColors;
  };

  WalkResult walkResult = func.walk<WalkOrder::PreOrder>([&](Operation *op)
                                                             -> WalkResult {
    if (auto liveIn = dyn_cast<LiveInOp>(op)) {
      auto maybeGpuAlloc = findGpuAlloc(liveIn.getMemref());
      if (failed(maybeGpuAlloc)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Can't trace back to rock.alloc from rock.live_in\n");
        return WalkResult::interrupt();
      }
      auto gpuAlloc = maybeGpuAlloc.value();
      if (!coloringInfo.colorAssignment.contains(gpuAlloc)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "No color assignment information for this alloc="
                   << gpuAlloc << "\n");
        return WalkResult::interrupt();
      }
      SetVector<int64_t> colorsAlloc =
          coloringInfo.colorAssignment.at(gpuAlloc);
      bool useLDSBarrier = false;
      auto filteredUsedColors = filterUsedColorsSinceLastBarrier(gpuAlloc);
      for (auto color : colorsAlloc)
        useLDSBarrier |= llvm::is_contained(filteredUsedColors, color);

      // add barrier
      if (useLDSBarrier) {
        // update 'usedColorsSinceLastBarrier': remove colors not found in
        // 'usedColors'
        usedColorsSinceLastBarrier = updateUsedColorsSinceLastBarrier();

        LLVM_DEBUG(llvm::dbgs()
                   << "Adding a LDS barrier for " << liveIn << "\n");
        addBarrier.push_back(liveIn);
      }

      // add used colors
      for (auto color : colorsAlloc) {
        if (usedColors.contains(color)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Color " << color
                     << " was already allocated, trying to use it for alloc="
                     << gpuAlloc << "\n");
          return WalkResult::interrupt();
        }
        usedColors.insert(color);
        // if it's been used since the last barrier, it must be the same alloc
        if (usedColorsSinceLastBarrier.contains(color))
          assert(usedColorsSinceLastBarrier[color] == gpuAlloc);

        usedColorsSinceLastBarrier[color] = gpuAlloc;
      }
    } else if (auto liveOut = dyn_cast<LiveOutOp>(op)) {
      auto maybeGpuAlloc = findGpuAlloc(liveOut.getMemref());
      if (failed(maybeGpuAlloc)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Can't trace back to rock.alloc from rock.live_out\n");
        return WalkResult::interrupt();
      }
      if (!coloringInfo.colorAssignment.contains(maybeGpuAlloc.value())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "No color assignment information for this alloc="
                   << maybeGpuAlloc.value() << "\n");
        return WalkResult::interrupt();
      }
      SetVector<int64_t> colorsAlloc =
          coloringInfo.colorAssignment.at(maybeGpuAlloc.value());
      // remove used colors
      for (auto color : colorsAlloc) {
        bool removed = usedColors.remove(color);
        if (!removed) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Trying to remove a color that does not exist, color="
                     << color << "\n");
          return WalkResult::interrupt();
        }
      }
    } else if (auto barrierOp = dyn_cast<LDSBarrierOp>(op)) {
      // update 'usedColorsSinceLastBarrier': remove colors not found in
      // 'usedColors'
      usedColorsSinceLastBarrier = updateUsedColorsSinceLastBarrier();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    LLVM_DEBUG(llvm::dbgs() << "Walk interrupted\n");
    return failure();
  }

  // Add barriers
  for (LiveInOp liveIn : addBarrier) {
    rewriter.setInsertionPoint(liveIn);
    LDSBarrierOp::create(rewriter, liveIn->getLoc());
  }

  return success();
}

// Remove liveness annotations
static void removeLivenessAnnotation(IRRewriter &rewriter,
                                     InterferenceData &interferenceData) {
  // Remove ops
  for (auto op : interferenceData.liveIns)
    rewriter.eraseOp(op);

  for (auto op : interferenceData.liveOuts)
    rewriter.eraseOp(op);
}

static LogicalResult reuseLDS(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());

  auto maybeInterferenceData = createInterferenceGraph(func);
  if (failed(maybeInterferenceData))
    return failure();

  InterferenceData &interferenceData = maybeInterferenceData.value();

  // nothing to do if there is only one (or none) LDS allocation
  if (interferenceData.interferenceGraph.size() < 2) {
    LLVM_DEBUG(llvm::dbgs() << "Not enough LDS allocations, skipping pass\n");

    // Remove rock.live_in and rock.live_out
    removeLivenessAnnotation(rewriter, interferenceData);
    return success();
  }

  GraphColoringInfo coloringInfo = graphColoring(interferenceData);

  int64_t requiredMemory = 0;
  for (auto [_, size] : coloringInfo.colorSizes) {
    requiredMemory += size;
  }

  // not enough LDS memory
  if (failed(checkLDSSize(func, requiredMemory))) {
    LLVM_DEBUG(llvm::dbgs() << "ReuseLDS requires too much LDS memory: "
                            << requiredMemory << " bytes\n");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Total LDS memory needed: " << requiredMemory
                          << " bytes\n");

  // Add LDS barriers
  if (failed(addLDSBarriers(rewriter, func, coloringInfo))) {
    LLVM_DEBUG(llvm::dbgs() << "Failed while adding LDS barriers\n");
    return failure();
  }

  // Remove rock.live_in and rock.live_out
  removeLivenessAnnotation(rewriter, interferenceData);

  // write the new GpuAllocOp to the start
  rewriter.setInsertionPointToStart(&func.getBody().front());

  // New allocations
  llvm::MapVector<int64_t, GpuAllocOp> colorAllocs;
  auto workgroupMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getWorkgroupAddressSpace());
  for (auto [color, size] : coloringInfo.colorSizes) {
    auto ldsMemRefBufferType = MemRefType::get(
        {size}, rewriter.getI8Type(), AffineMap{}, workgroupMemoryAddressSpace);

    Location loc = func.getLoc();
    GpuAllocOp colorAlloc =
        GpuAllocOp::create(rewriter, loc, ldsMemRefBufferType);
    colorAllocs[color] = colorAlloc;
    LLVM_DEBUG(llvm::dbgs()
               << "allocating " << size << " bytes, color: " << color << "\n");
  }

  // Replace all GpuAllocs as ViewOps
  for (auto allocTuple : coloringInfo.allocOffsets) {
    GpuAllocOp alloc;
    int64_t color, offset;
    std::tie(alloc, color, offset) = allocTuple;
    int64_t colorSize = coloringInfo.colorSizes[color];
    GpuAllocOp newAlloc = colorAllocs[color];

    if (offset < 0) {
      LLVM_DEBUG(llvm::dbgs() << "Negative offset\n");
      return failure();
    }
    if (offset >= colorSize) {
      LLVM_DEBUG(llvm::dbgs() << "Offset is too big\n");
      return failure();
    }
    auto bufferType = alloc.getOutput().getType();
    auto numElements = bufferType.getNumElements();
    auto elementBytes = bufferType.getElementType().getIntOrFloatBitWidth() / 8;
    auto rank = bufferType.getRank();
    if (elementBytes != 1) {
      LLVM_DEBUG(llvm::dbgs() << "GpuAllocOp allocates a type of "
                              << elementBytes << " bytes, expected 1 byte\n");
      return failure();
    }
    if (rank != 1) {
      LLVM_DEBUG(llvm::dbgs() << "Rank should be one, it's " << rank << "\n");
      return failure();
    }

    rewriter.setInsertionPointAfter(alloc);
    Location loc = alloc->getLoc();

    LLVM_DEBUG(llvm::dbgs() << "color: " << color << " offset: " << offset
                            << " numElements: " << numElements << "\n");
    Value byteOffset =
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, offset);

    auto newViewType =
        MemRefType::get({numElements}, rewriter.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    rewriter.replaceOpWithNewOp<memref::ViewOp>(alloc, newViewType, newAlloc,
                                                byteOffset, ValueRange{});
  }

  return success();
}

void RockReuseLDSPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  if (failed(reuseLDS(func))) {
    return signalPassFailure();
  }
}
