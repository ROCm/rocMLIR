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
// This pass changes ThreadwiseWriteAllOp to swizzle on LDS before writing to 
// GPU memory
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

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

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGEMMOUTPUTSWIZZLEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gemm-output-swizzle"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using mlir::gpu::AddressSpace;

namespace {
struct RockGemmOutputSwizzlePass
    : public rock::impl::RockGemmOutputSwizzlePassBase<
          RockGemmOutputSwizzlePass> {
  void runOnOperation() override;
};

bool hasWorkgroupMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuAttr.getValue() == AddressSpace::Workgroup;
  return false;
}

struct ThreadwiseWriteAllRewritePattern
    : public OpRewritePattern<ThreadwiseWriteAllOp> {
  using OpRewritePattern<ThreadwiseWriteAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseWriteAllOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Prepare some useful constants.
    Value convertedC = op.getSource();
    Value matC = op.getDest();
    Type destType = op.getDest().getType().getElementType();
    MemRefType destMemRefType = cast<MemRefType>(op.getDest().getType());
    // if ThreadwiseWriteAllOp is saving to LDS, skip pass
    if(hasWorkgroupMemoryAddressSpace(destMemRefType)) {
        return success();
    }

    // Convert from reg -> memory transform to reg -> block
    ArrayAttr srcTransform = op.getExtraViewsAttr();
    Operation::operand_range extraIndices = op.getExtraIndices();
    SetVector<StringRef> dimensionsToRemove;
    dimensionsToRemove.insert("g_block");
    dimensionsToRemove.insert("m_block");
    dimensionsToRemove.insert("n_block");
    FailureOr<ArrayAttr> maybeIdToLDS = removeUpperDims(b, srcTransform, dimensionsToRemove);
    if (failed(maybeIdToLDS)) {
      return failure();
    }
    ArrayAttr idToLDS = maybeIdToLDS.value();

    // TODO: decide if we want to skip this pass 
    // if writes are already coalesced and 128b

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> cShape;
    cShape = op.getDest().getType().getShape();
    int64_t G = cShape[0];
    int64_t M = cShape[1];
    int64_t N = cShape[2];
    FailureOr<IntegerAttr> maybeBlockSize = getBlockSize(op);
    if (failed(maybeBlockSize)) {
        return failure();
    }
    int64_t blockSize = maybeBlockSize.value().getValue().getSExtValue();
    int64_t mPerBlock = getLowerShape(idToLDS)[0];
    int64_t nPerBlock = getLowerShape(idToLDS)[1];
    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;
    bool useIndexDiffs = true;
    bool forceUnroll = true;
    
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};

    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    // Allocate LDS for output.
    // TODO: reuse LDS memory
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefOutputType =
        MemRefType::get({mPerBlock, nPerBlock}, destType, AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsBufferOutput = b.create<GpuAllocOp>(loc, ldsMemRefOutputType);

    // Emit blockwise stores.
    b.create<ThreadwiseWriteAllOp>(loc, convertedC, ldsBufferOutput,
                                    /*extraViews=*/idToLDS,
                                    /*extraIndices=*/ValueRange{tid},
                                    op.getFeatures(), StoreMethod::Set,
                                    /*forceUnroll=*/forceUnroll,
                                    /*useIndexDiffs=*/useIndexDiffs);

    // Decide register vectorization.
    constexpr int64_t dimensionM = 1;
    constexpr int64_t dimensionN = 2;
    int64_t dataPerThread = (nPerBlock * mPerBlock) / blockSize;
    VectorizationResult mVectorRes = getMaxVectorization(
        matC, dimensionM, /*inputDimLen=*/
        dataPerThread,
        matC.getDefiningOp());
    int64_t mVectorLen = mVectorRes.max;
    VectorizationResult nVectorRes = getMaxVectorization(
        matC, dimensionN, /*inputDimLen=*/
        dataPerThread,
        matC.getDefiningOp());
    int64_t nVectorLen = nVectorRes.max;

    int64_t dim;
    if (mVectorLen > nVectorLen) {
        dim = dimensionM;
    } else {
        dim = dimensionN;
    }
    
    // Load from LDS.
    int64_t elementsWrittenPerThread = 128 / destType.getIntOrFloatBitWidth();
    int64_t iter = dataPerThread / elementsWrittenPerThread;
    Value finalC = gpuAlloc(b, loc, dataPerThread, destType,
                                AddressSpace::Private);
    
    TopDownTMBuilder tidIterMerge(b, {"tid", "iter"},
                            {blockSize, dataPerThread});
    SmallVector<StringRef, 5> throughDims{"tid"};
    tidIterMerge.passThrough(throughDims);
    tidIterMerge.merge(
        {"iter", "numElements"}, {1, 2}, "iter",
        {iter, elementsWrittenPerThread});
    auto tidIterMergeAttr = tidIterMerge.get();

    auto tidIterFlatten = TopDownTMBuilder::below(tidIterMerge, tidIterMergeAttr);
    tidIterFlatten.unmerge("flattenBlock", 0, {"iter", "tid", "numElements"},
                        {iter, blockSize, elementsWrittenPerThread});
    auto tidIterFlattenAttr = tidIterFlatten.get();

    auto flattenToLDS = TopDownTMBuilder::below(tidIterFlatten, tidIterFlattenAttr);
    if(dim == dimensionN) {
        flattenToLDS.merge(
            {"gemmM", "gemmN"}, {0, 1}, "flattenBlock",
            {mPerBlock, nPerBlock});
    } else {
        flattenToLDS.merge(
            {"gemmN", "gemmM"}, {0, 1}, "flattenBlock",
            {mPerBlock, nPerBlock});
    }
    auto flattenToLDSAttr = flattenToLDS.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(tidIterMergeAttr);
    transformAttrs.push_back(tidIterFlattenAttr);
    transformAttrs.push_back(flattenToLDSAttr);
    if(dim == dimensionM) {
        auto switchMandN = TopDownTMBuilder::below(flattenToLDS, flattenToLDSAttr);
        switchMandN.passThrough({0, 1}, {1, 0});
        auto switchMandNAttr = switchMandN.get();
        transformAttrs.push_back(switchMandNAttr);
    }
    
    ArrayAttr ldsRead = b.getArrayAttr(transformAttrs);
    auto ldsBufferForLoad = transform(b, ldsBufferOutput, ldsRead);

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);
    
    b.create<ThreadwiseReadIntoOp>(
        loc, ldsBufferForLoad, finalC, b.getArrayAttr({}),
        ValueRange{tid}, forceUnroll, useIndexDiffs);

    // Save to memory
    // Create views as gridwise sub-tile of C
    TopDownTMBuilder tidIterMerge2(
        b, {"g_block", "m_block", "n_block", "tid", "iter"},
        {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], blockSize,
        dataPerThread},
        loc);
    tidIterMerge2.passThrough({"g_block", "m_block", "n_block", "tid"});
    tidIterMerge2.merge(
        {"iter", "numElements"}, {4, 5}, "iter",
        {iter, elementsWrittenPerThread});
    auto tidIterMerge2Attr = tidIterMerge2.get();

    TopDownTMBuilder tidIterFlatten2 = TopDownTMBuilder::below(tidIterMerge2, tidIterMerge2Attr);
    tidIterFlatten2.passThrough({"g_block", "m_block", "n_block"});
    tidIterFlatten2.unmerge("flattenBlock", 3, {"iter", "tid", "numElements"},
                        {iter, blockSize, elementsWrittenPerThread});
    auto tidIterFlatten2Attr = tidIterFlatten2.get();
                        
    auto flattenToBlockCoord = TopDownTMBuilder::below(tidIterFlatten2, tidIterFlatten2Attr);
    flattenToBlockCoord.passThrough({"g_block", "m_block", "n_block"});
    if(dim == dimensionN) {
        flattenToBlockCoord.merge(
            {"block_m", "block_n"}, {3, 4}, "flattenBlock",
            {mPerBlock, nPerBlock});
    } else {
        flattenToBlockCoord.merge(
            {"block_n", "block_m"}, {3, 4}, "flattenBlock",
            {nPerBlock, mPerBlock});
    }
    TransformMapAttr flattenToBlockCoordAttr = flattenToBlockCoord.get();

    auto toMatrixC = TopDownTMBuilder::below(flattenToBlockCoord, flattenToBlockCoordAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {"g_block"});
    toMatrixC.unmerge("gemmM", 1, {"m_block", "block_m"}, {bidGridLengths[1], mPerBlock});
    toMatrixC.unmerge("gemmN", 2, {"n_block", "block_n"}, {bidGridLengths[2], nPerBlock});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();

    SmallVector<Attribute> transformAttrsStore{tidIterMerge2Attr,
                                        tidIterFlatten2Attr,
                                        flattenToBlockCoordAttr,
                                        toMatrixCAttr};
    ArrayAttr idToMatrixCMaps = b.getArrayAttr(transformAttrsStore);

    b.create<ThreadwiseWriteAllOp>(
        loc, finalC, matC, idToMatrixCMaps,
        /*extraIndices=*/
        extraIndices,
        op.getFeatures(), op.getStoreMethod(), forceUnroll, useIndexDiffs);

    b.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void RockGemmOutputSwizzlePass::runOnOperation() {
    func::FuncOp func = getOperation();
    //Location loc = func->getLoc();

    // Rewrite 
    RewritePatternSet patterns(&getContext());
    patterns.add<ThreadwiseWriteAllRewritePattern>(
        &getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
