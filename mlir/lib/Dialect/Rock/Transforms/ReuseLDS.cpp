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

// Function to round up size to the nearest multiple of 16 bytes
static int64_t roundUpToMultipleOf16(int64_t size) { return (size + 15) & ~15; }

static int64_t getWorkgroupMemorySize(MemRefType type) {
  auto memSpaceValue =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace()).getValue();
  int64_t size = -1;
  if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
    size = type.getNumElements() * getByteWidth(type.getElementType());
  }
  return size;
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

static int64_t
assignColors(GpuAllocOp alloc, llvm::SetVector<int64_t> &usedColors,
             llvm::MapVector<int64_t, int64_t> &colorMemSize,
             llvm::MapVector<GpuAllocOp, SetVector<int64_t>> &colorAssignment) {
  const int64_t requiredSize =
      getWorkgroupMemorySize(alloc.getOutput().getType());
  assert(requiredSize > 0);
  int64_t color = 0;
  int64_t allocatedSize = 0;
  int64_t maxColor = 0;

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
    maxColor = std::max(maxColor, color);
    colorAssignment[alloc].insert(color);
    bool existingColor = colorMemSize.contains(color);
    // New color
    if (!existingColor) {
      // Make this aligned to 128 bits
      colorMemSize[color] = roundUpToMultipleOf16(requiredSize - allocatedSize);
    }
    allocatedSize += colorMemSize[color];
    color++;
  }
  return maxColor;
}

static std::tuple<int64_t, SmallVector<std::tuple<GpuAllocOp, int64_t, bool>>>
graphColoring(
    SmallVector<GpuAllocOp> &gpuAllocs,
    llvm::MapVector<GpuAllocOp, SetVector<GpuAllocOp>> &interferenceGraph) {
  llvm::MapVector<GpuAllocOp, SetVector<int64_t>> colorAssignment;
  llvm::MapVector<int64_t, int64_t> colorMemSize;
  int64_t maxColor = 0;

  // Assign colors using greedy algorithm
  for (GpuAllocOp alloc : gpuAllocs) {
    llvm::SetVector<int64_t> usedColors;
    for (GpuAllocOp neighbor : interferenceGraph[alloc]) {
      if (colorAssignment.find(neighbor) != colorAssignment.end()) {
        for (int64_t color : colorAssignment[neighbor]) {
          usedColors.insert(color);
        }
      }
    }

    // Assign a set of colors
    int64_t localMaxColor =
        assignColors(alloc, usedColors, colorMemSize, colorAssignment);
    maxColor = std::max(maxColor, localMaxColor);
  }

  // Compute required memory and offsets per color
  SmallVector<int64_t> colorOffsets;
  int64_t offset = 0;
  for (int64_t color = 0; color < maxColor + 1; color++) {
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

    offsets.push_back(std::tuple(alloc, offset, useLDSBarrier));
  }

  return std::tuple(offset, offsets);
}

static LogicalResult reuseLDS(func::FuncOp &func,
                              SmallVector<Operation *> &deallocs) {
  IRRewriter rewriter(func->getContext());

  SmallVector<GpuAllocOp> allocs;
  SetVector<GpuAllocOp> currentAllocs;
  llvm::MapVector<GpuAllocOp, llvm::SetVector<GpuAllocOp>> interferenceGraph;
  llvm::MapVector<Value, GpuAllocOp> memrefToAlloc;

  // Create the interference graph and save all allocs and deallocs (LDS)
  WalkResult walkResult = func.walk([&](Operation *op) -> WalkResult {
    if (auto gpuAlloc = dyn_cast<GpuAllocOp>(op)) {
      auto type = gpuAlloc.getOutput().getType();

      int64_t size = getWorkgroupMemorySize(type);
      if (size != -1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Found rock.alloc of " << size << " bytes\n");
        assert(size > 0);

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
      int64_t size = getWorkgroupMemorySize(type);
      if (size != -1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Found rock.dealloc of " << size << " bytes\n");
        assert(size > 0);

        if (memrefToAlloc.find(gpuDealloc.getMemref()) == memrefToAlloc.end()) {
          LLVM_DEBUG(llvm::dbgs() << "Called rock.dealloc multiple times?\n");
          return WalkResult::interrupt();
        }
        bool erased =
            currentAllocs.remove(memrefToAlloc[gpuDealloc.getMemref()]);
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
    LLVM_DEBUG(llvm::dbgs()
               << "1: " << deallocs.size() << " != " << allocs.size() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "2: " << allocs.size()
                            << " != " << interferenceGraph.size() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "3: " << currentAllocs.size() << "\n");
    return failure();
  }

  // nothing to do if there is only one LDS allocation or none
  if (interferenceGraph.size() < 2) {
    LLVM_DEBUG(llvm::dbgs() << "Not enough LDS allocations, skipping pass\n");
    return success();
  }

  int64_t requiredMemory;
  SmallVector<std::tuple<GpuAllocOp, int64_t, bool>> allocOffsets;
  std::tie(requiredMemory, allocOffsets) =
      graphColoring(allocs, interferenceGraph);

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

  rewriter.setInsertionPoint(std::get<0>(allocOffsets[0]));
  Location loc = std::get<0>(allocOffsets[0])->getLoc();
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
  return success();
}

struct GpuDeallocOpRewritePattern : public OpRewritePattern<GpuDeallocOp> {
  using OpRewritePattern<GpuDeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GpuDeallocOp op,
                                PatternRewriter &b) const override {

    b.eraseOp(op);
    return mlir::success();
  }
};

void RockReuseLDSPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  SmallVector<Operation *> deallocs;
  if (failed(reuseLDS(func, deallocs))) {
    return signalPassFailure();
  }

  // Remove all GpuDeallocOps
  RewritePatternSet patterns(&getContext());
  patterns.add<GpuDeallocOpRewritePattern>(&getContext());

  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  if (failed(applyOpPatternsAndFold(deallocs, std::move(patterns), config))) {
    return signalPassFailure();
  }
}
