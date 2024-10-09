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
#include "mlir/Dialect/Rock/utility/memoryUtils.h"

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

static LogicalResult reuseLDS(func::FuncOp &func) {
  FailureOr<LDSInfo> maybeLdsInfo = createInterferenceGraph(func);
  if (failed(maybeLdsInfo)) {
    return failure();
  }
  LDSInfo ldsInfo = maybeLdsInfo.value();

  // add debug information
  for (GpuAllocOp alloc : ldsInfo.allocs) {
    auto type = alloc.getOutput().getType();
    std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
    assert(maybeSize.has_value());
    int64_t size = maybeSize.value();
    LLVM_DEBUG(llvm::dbgs() << "Found rock.alloc of " << size << " bytes\n");
  }
  for (GpuDeallocOp dealloc : ldsInfo.deallocs) {
    auto type = dealloc.getMemref().getType();
    std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
    assert(maybeSize.has_value());
    int64_t size = maybeSize.value();
    LLVM_DEBUG(llvm::dbgs() << "Found rock.dealloc of " << size << " bytes\n");
  }

  // nothing to do if there is only one (or none) LDS allocation
  if (ldsInfo.interferenceGraph.size() < 2) {
    LLVM_DEBUG(llvm::dbgs() << "Not enough LDS allocations, skipping pass\n");
    return success();
  }

  llvm::MapVector<int64_t, int64_t> colorSizes;
  SmallVector<std::tuple<GpuAllocOp, int64_t, int64_t, bool>> allocOffsets;
  std::tie(colorSizes, allocOffsets) = graphColoring(ldsInfo);

  // write the new GpuAllocOp to the start
  IRRewriter rewriter(func->getContext());
  rewriter.setInsertionPointToStart(&func.getBody().front());

  // New allocations
  llvm::MapVector<int64_t, GpuAllocOp> colorAllocs;
  auto workgroupMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getWorkgroupAddressSpace());
  for (auto [color, size] : colorSizes) {
    auto ldsMemRefBufferType = MemRefType::get(
        {size}, rewriter.getI8Type(), AffineMap{}, workgroupMemoryAddressSpace);

    Location loc = func.getLoc();
    GpuAllocOp colorAlloc =
        rewriter.create<GpuAllocOp>(loc, ldsMemRefBufferType);
    colorAllocs[color] = colorAlloc;
    LLVM_DEBUG(llvm::dbgs()
               << "allocating " << size << " bytes, color: " << color << "\n");
  }

  // Replace all GpuAllocs as ViewOps
  for (auto allocTuple : allocOffsets) {
    GpuAllocOp alloc;
    int64_t color, offset;
    bool useLDSBarrier;
    std::tie(alloc, color, offset, useLDSBarrier) = allocTuple;
    int64_t colorSize = colorSizes[color];
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

    LLVM_DEBUG(llvm::dbgs() << "color: " << color << " offset: " << offset
                            << " numElements: " << numElements << "\n");
    Value byteOffset =
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, offset);

    auto newViewType =
        MemRefType::get({numElements}, rewriter.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    rewriter.replaceOpWithNewOp<memref::ViewOp>(alloc, newViewType, newAlloc,
                                                byteOffset, ValueRange{});

    // add barrier if needed
    if (useLDSBarrier) {
      rewriter.create<LDSBarrierOp>(loc);
    }
  }

  // Remove all GpuDeallocOps but the last one and add a new alloc/dealloc pair
  // for each buffer
  for (auto [i, dealloc] : llvm::enumerate(ldsInfo.deallocs)) {
    rewriter.setInsertionPointAfter(dealloc);
    if (i == ldsInfo.deallocs.size() - 1) {
      GpuDeallocOp prevDealloc = rewriter.replaceOpWithNewOp<GpuDeallocOp>(
          dealloc, std::get<1>(colorAllocs.front()));
      for (auto [i, pair] : llvm::enumerate(colorAllocs)) {
        // skip first, already done
        if (i > 0) {
          GpuAllocOp alloc = std::get<1>(pair);
          rewriter.setInsertionPointAfter(prevDealloc);
          Location loc = prevDealloc.getLoc();
          prevDealloc = rewriter.create<GpuDeallocOp>(loc, alloc);
        }
      }
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
