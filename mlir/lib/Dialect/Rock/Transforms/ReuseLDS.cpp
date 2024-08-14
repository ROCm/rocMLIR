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
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
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

static std::tuple<int64_t, SmallVector<std::tuple<GpuAllocOp, int64_t, bool>>>
graphColoring(
    SmallVector<GpuAllocOp> &gpuAllocs,
    llvm::MapVector<GpuAllocOp, SetVector<GpuAllocOp>> &interferenceGraph) {
  llvm::MapVector<GpuAllocOp, int64_t> colorAssignment;

  // Assign colors using greedy algorithm
  for (const GpuAllocOp alloc : gpuAllocs) {
    llvm::SetVector<int64_t> usedColors;
    for (GpuAllocOp neighbor : interferenceGraph[alloc]) {
      if (colorAssignment.find(neighbor) != colorAssignment.end()) {
        usedColors.insert(colorAssignment[neighbor]);
      }
    }

    // Find the first available color
    int64_t color = 0;
    while (usedColors.contains(color)) {
      color++;
    }
    colorAssignment[alloc] = color;
  }

  // Compute maximum memory needed per color
  llvm::MapVector<int64_t, int64_t> colorMemSize;
  int64_t maxColor = 0;
  for (const auto &tuple : colorAssignment) {
    GpuAllocOp alloc;
    int64_t color;
    std::tie(alloc, color) = tuple;
    maxColor = std::max(maxColor, color);
    int64_t allocSize = getWorkgroupMemorySize(alloc.getOutput().getType());
    if (colorMemSize.find(color) == colorMemSize.end()) {
      colorMemSize[color] = allocSize;
    } else {
      colorMemSize[color] = std::max(colorMemSize[color], allocSize);
    }
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
    int64_t color = colorAssignment[alloc];
    int64_t offset = colorOffsets[color];

    // if the color has been used, we are "reusing" memory,
    // we need a LDS barrier
    bool useLDSBarrier = usedColors.contains(color);
    usedColors.insert(color);
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
