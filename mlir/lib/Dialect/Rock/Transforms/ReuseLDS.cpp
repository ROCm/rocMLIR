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
// This pass re-uses LDS memory by using the lifetime annotations (rock.dealloc)
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

static std::optional<int64_t> getWorkgroupMemorySize(MemRefType type) {
  auto memSpaceValue =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace()).getValue();
  if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
    return type.getNumElements() * getByteWidth(type.getElementType());
  }
  return std::nullopt;
}

static LogicalResult checkLDSSize(Operation *op, int64_t ldsBytes) {
  // Check for arch limitations exceeded
  FailureOr<StringAttr> maybeArch = getArch(op);
  if (succeeded(maybeArch)) {
    StringAttr arch = maybeArch.value();
    const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
    return success(ldsBytes <= ldsSize);
  }
  return success();
}

static void assignColors(
    GpuAllocOp alloc, llvm::SetVector<int64_t> &usedColors,
    llvm::SmallDenseMap<int64_t, int64_t> &colorMemSize,
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

static std::tuple<int64_t, SmallVector<std::tuple<GpuAllocOp, int64_t, bool>>>
graphColoring(
    SmallVector<GpuAllocOp> &gpuAllocs,
    llvm::SmallDenseMap<GpuAllocOp, SetVector<GpuAllocOp>> &interferenceGraph,
    llvm::SmallDenseMap<GpuAllocOp, SetVector<GpuAllocOp>> &deallocBefore) {
  llvm::SmallDenseMap<GpuAllocOp, SetVector<int64_t>> colorAssignment;
  llvm::SmallDenseMap<int64_t, int64_t> colorMemSize;

  SmallVector<GpuAllocOp> sortedAllocs(gpuAllocs);

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
    for (GpuAllocOp neighbor : interferenceGraph[alloc]) {
      if (colorAssignment.contains(neighbor)) {
        for (int64_t color : colorAssignment[neighbor]) {
          usedColors.insert(color);
        }
      }
    }

    // Assign a set of colors
    assignColors(alloc, usedColors, colorMemSize, colorAssignment);
  }

  // Compute required memory and offsets per color
  SmallVector<int64_t> colorOffsets;
  int64_t offset = 0;
  for (int64_t color = 0; color < colorMemSize.size(); color++) {
    colorOffsets.push_back(offset);
    offset += colorMemSize[color];
  }

  // Compute offsets per GpuAllocOp
  SmallVector<std::tuple<GpuAllocOp, int64_t, bool>> offsets;
  llvm::SetVector<int64_t> usedColors;
  for (const GpuAllocOp alloc : gpuAllocs) {
    // use the offset of the first color only
    int64_t firstColor = colorAssignment[alloc][0];
    int64_t offset = colorOffsets[firstColor];

    // if the color has been used, we are "reusing" memory,
    // we need a LDS barrier
    bool useLDSBarrier = false;
    for (int64_t color : colorAssignment[alloc]) {
      useLDSBarrier |= usedColors.contains(color);
      usedColors.insert(color);
    }

    if (useLDSBarrier) {
      for (GpuAllocOp deadAlloc : deallocBefore[alloc]) {
        for (int64_t color : colorAssignment[deadAlloc]) {
          if (!colorAssignment[alloc].contains(color)) {
            usedColors.remove(color);
          }
        }
      }
    }

    offsets.push_back(std::tuple(alloc, offset, useLDSBarrier));
  }

  return std::tuple(offset, offsets);
}

static LogicalResult reuseLDS(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());

  SmallVector<GpuAllocOp> allocs;
  SmallVector<GpuDeallocOp> deallocs;
  SetVector<GpuAllocOp> currentAllocs;
  llvm::SmallDenseMap<GpuAllocOp, llvm::SetVector<GpuAllocOp>>
      interferenceGraph;
  llvm::SmallDenseMap<Value, GpuAllocOp> memrefToAlloc;
  llvm::SmallDenseMap<GpuAllocOp, llvm::SetVector<GpuAllocOp>> deallocBefore;
  llvm::SetVector<GpuAllocOp> deallocsUpToNow;

  // Create the interference graph and save all allocs and deallocs (LDS)
  WalkResult walkResult = func.walk([&](Operation *op) -> WalkResult {
    if (auto gpuAlloc = dyn_cast<GpuAllocOp>(op)) {
      auto type = gpuAlloc.getOutput().getType();

      std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
      if (maybeSize.has_value()) {
        int64_t size = maybeSize.value();
        LLVM_DEBUG(llvm::dbgs()
                   << "Found rock.alloc of " << size << " bytes\n");

        // save deallocs up to this point
        deallocBefore[gpuAlloc] = SetVector<GpuAllocOp>(deallocsUpToNow.begin(),
                                                        deallocsUpToNow.end());
        deallocsUpToNow.clear();

        // add vertex and connections
        for (auto alloc : currentAllocs) {
          interferenceGraph[alloc].insert(gpuAlloc);
          interferenceGraph[gpuAlloc].insert(alloc);
        }
        // if it has no neighbors, we still want to add a vertex
        if (currentAllocs.empty()) {
          interferenceGraph[gpuAlloc] = {};
        }
        currentAllocs.insert(gpuAlloc);
        memrefToAlloc[gpuAlloc.getOutput()] = gpuAlloc;
        allocs.push_back(gpuAlloc);
      }
    } else if (auto gpuDealloc = dyn_cast<GpuDeallocOp>(op)) {
      auto type = gpuDealloc.getMemref().getType();
      std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
      if (maybeSize.has_value()) {
        int64_t size = maybeSize.value();
        LLVM_DEBUG(llvm::dbgs()
                   << "Found rock.dealloc of " << size << " bytes\n");

        if (memrefToAlloc.find(gpuDealloc.getMemref()) == memrefToAlloc.end()) {
          LLVM_DEBUG(llvm::dbgs() << "Called rock.dealloc multiple times?\n");
          return WalkResult::interrupt();
        }
        bool erased =
            currentAllocs.remove(memrefToAlloc[gpuDealloc.getMemref()]);
        deallocsUpToNow.insert(memrefToAlloc[gpuDealloc.getMemref()]);
        if (!erased) {
          LLVM_DEBUG(llvm::dbgs() << "Called rock.dealloc multiple times?\n");
          return WalkResult::interrupt();
        }
        deallocs.push_back(gpuDealloc);
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    LLVM_DEBUG(llvm::dbgs() << "Walk interrupted\n");
    return failure();
  }

  // same number of rock.alloc and rock.dealloc
  if (deallocs.size() != allocs.size() ||
      allocs.size() != interferenceGraph.size() || !currentAllocs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "There should be an equal number of rock.alloc "
                               "and rock.dealloc (for LDS)\n");
    return failure();
  }

  // nothing to do if there is only one (or none) LDS allocation
  if (interferenceGraph.size() < 2) {
    LLVM_DEBUG(llvm::dbgs() << "Not enough LDS allocations, skipping pass\n");
    return success();
  }

  int64_t requiredMemory;
  SmallVector<std::tuple<GpuAllocOp, int64_t, bool>> allocOffsets;
  std::tie(requiredMemory, allocOffsets) =
      graphColoring(allocs, interferenceGraph, deallocBefore);

  // not enough LDS memory
  if (failed(checkLDSSize(func, requiredMemory))) {
    LLVM_DEBUG(llvm::dbgs() << "ReuseLDS requires too much LDS memory: "
                            << requiredMemory << " bytes\n");
    return failure();
  }

  // New allocation
  auto workgroupMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getWorkgroupAddressSpace());
  auto ldsMemRefBufferType =
      MemRefType::get({requiredMemory}, rewriter.getI8Type(), AffineMap{},
                      workgroupMemoryAddressSpace);

  // write the big GpuAllocOp to the start
  rewriter.setInsertionPointToStart(&func.getBody().front());
  Location loc = func.getLoc();
  auto ldsBigPool = rewriter.create<GpuAllocOp>(loc, ldsMemRefBufferType);
  LLVM_DEBUG(llvm::dbgs() << "allocating " << requiredMemory << " bytes\n");

  // Replace all GpuAllocs as ViewOps
  for (auto allocTuple : allocOffsets) {
    GpuAllocOp alloc;
    int64_t offset;
    bool useLDSBarrier;
    std::tie(alloc, offset, useLDSBarrier) = allocTuple;

    if (offset < 0) {
      LLVM_DEBUG(llvm::dbgs() << "Negative offset\n");
      return failure();
    }
    if (offset >= requiredMemory) {
      LLVM_DEBUG(llvm::dbgs() << "Offset is too big\n");
      return failure();
    }
    auto bufferType = alloc.getOutput().getType();
    auto numElements = bufferType.getNumElements();
    auto elementBytes = getByteWidth(bufferType.getElementType());
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

    LLVM_DEBUG(llvm::dbgs() << "offset: " << offset
                            << " numElements: " << numElements << "\n");
    Value byteOffset =
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, offset);

    auto newViewType =
        MemRefType::get({numElements}, rewriter.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    rewriter.replaceOpWithNewOp<memref::ViewOp>(alloc, newViewType, ldsBigPool,
                                                byteOffset, ValueRange{});

    // add barrier if needed
    if (useLDSBarrier) {
      rewriter.create<LDSBarrierOp>(loc);
    }
  }

  // Remove all GpuDeallocOps but the last one
  for (auto [i, dealloc] : llvm::enumerate(deallocs)) {
    rewriter.setInsertionPointAfter(dealloc);
    if (i == deallocs.size() - 1) {
      rewriter.replaceOpWithNewOp<GpuDeallocOp>(dealloc, ldsBigPool);
    } else {
      rewriter.eraseOp(dealloc);
    }
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
