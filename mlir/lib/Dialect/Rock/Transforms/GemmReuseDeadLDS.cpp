//===- GemmOutputSwizzle - MLIR Rock ops lowering passes -----===//
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
// This pass changes GPUAllocOp to reuse LDS memory of previous allocations
// which are assumed dead
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGEMMREUSEDEADLDSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-reuse-dead-lds"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using mlir::gpu::AddressSpace;

namespace {
struct RockGemmReuseDeadLDSPass
    : public rock::impl::RockGemmReuseDeadLDSPassBase<
          RockGemmReuseDeadLDSPass> {
  void runOnOperation() override;
};
} // end anonymous namespace


void RockGemmReuseDeadLDSPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = func->getContext();
  IRRewriter rewriter(ctx);

    // get all GpuAlloc operations to LDS and
    // compute space taken by all expect the last GpuAlloc,
    // and compute space taken by last GpuAlloc
    SmallVector<GpuAllocOp> allocs;
    int64_t totalSize = 0;
    int64_t lastSize = 0;
    func.walk([&](GpuAllocOp gpuAlloc) {
        auto type = gpuAlloc.getOutput().getType();
        auto memSpaceValue =
            dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace())
                .getValue();
        if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
            lastSize = type.getNumElements()*getByteWidth(type.getElementType());
            totalSize += lastSize;
            LLVM_DEBUG(llvm::dbgs() << "Size: " << lastSize << "\n");
            allocs.push_back(gpuAlloc);
        }
    });
    assert(allocs.size() > 1 && "There should be at least two LDS allocations");
    LLVM_DEBUG(llvm::dbgs() << "number of GpuAllocOp ops to rewrite: " << allocs.size() << "\n");
    int64_t deadSize = totalSize - lastSize;

    // we assume when the last GpuAlloc takes place the previous are dead and
    // re-use their memory.
    
    // New allocation
    int64_t requiredMemory = std::max(deadSize, lastSize);
    LLVM_DEBUG(llvm::dbgs() << "totalSize: " << totalSize << "\n");
    LLVM_DEBUG(llvm::dbgs() << "deadSize: " << deadSize << "\n");
    LLVM_DEBUG(llvm::dbgs() << "lastSize: " << lastSize << "\n");
    LLVM_DEBUG(llvm::dbgs() << "requiredMemory: " << requiredMemory << "\n");

    auto workgroupMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefBufferType =
        MemRefType::get({requiredMemory}, rewriter.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    
    rewriter.setInsertionPoint(allocs[0]);
    Location loc = allocs[0]->getLoc();
    auto ldsByteBuffer = rewriter.create<GpuAllocOp>(loc, ldsMemRefBufferType);

    // Rewrite all GpuAllocs as SubViewOp
    int64_t offset = 0;
    size_t i = 0;
    for(GpuAllocOp alloc : allocs) {
        auto bufferType = alloc.getOutput().getType();
        auto numElements = bufferType.getNumElements();
        auto elementBytes = getByteWidth(bufferType.getElementType());
        auto rank = bufferType.getRank();
        assert(elementBytes == 1 && "All GpuAllocOp should allocate bytes");
        assert(rank == 1 && "Rank should be one");

        rewriter.setInsertionPoint(alloc);
        Location loc = alloc->getLoc();
        
        if(i == allocs.size()-1) {
            offset = 0;
        }
        LLVM_DEBUG(llvm::dbgs() << "i: " << i << " offset: " << offset << " elementBytes: " << elementBytes << " rank: " << rank << " numElements: " << numElements << "\n");
        auto subView = rewriter.create<memref::SubViewOp>(
            loc, bufferType, ldsByteBuffer, 
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, offset)},
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, numElements)},
        ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, 1)});
        offset += numElements;
        rewriter.replaceOp(alloc, subView);
        i++;
    }
}
