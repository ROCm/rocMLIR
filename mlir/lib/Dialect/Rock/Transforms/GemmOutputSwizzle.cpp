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
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

bool hasPrivateMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)) {

    return gpuAttr.getValue() == AddressSpace::Private;
  }
  return false;
}

bool hasWorkgroupMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)) {

    return gpuAttr.getValue() == AddressSpace::Workgroup;
  }
  return false;
}

bool hasGlobalMemoryAddressSpace(MemRefType type) {
  return !hasWorkgroupMemoryAddressSpace(type) &&
         !hasPrivateMemoryAddressSpace(type);
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

// Function to create the schedule of the current set of stages
void reuseDeadLDS(func::FuncOp &func) {
  // Retrieve all GpuAlloc operations targeting LDS and
  // calculate the total memory space required for all operations
  // except the last one. Additionally, compute the memory space
  // required by the last GpuAlloc operation.
  SmallVector<GpuAllocOp> allocs;
  int64_t totalSize = 0;
  int64_t lastSize = 0;
  func.walk([&](GpuAllocOp gpuAlloc) {
    auto type = gpuAlloc.getOutput().getType();
    auto memSpaceValue =
        dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace())
            .getValue();
    if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      lastSize = type.getNumElements() * getByteWidth(type.getElementType());
      totalSize += lastSize;
      allocs.push_back(gpuAlloc);
    }
  });
  // There should be at least two LDS allocations
  if (allocs.size() < 2) {
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "number of GpuAllocOp ops to rewrite: "
                          << allocs.size() << "\n");
  int64_t deadSize = totalSize - lastSize;

  // we assume the last GpuAllocOp can reuse previous
  // LDS allocations because they are no longer needed

  // New allocation
  MLIRContext *ctx = func->getContext();
  IRRewriter rewriter(ctx);
  int64_t requiredMemory = std::max(deadSize, lastSize);
  LLVM_DEBUG(llvm::dbgs() << "deadSize: " << deadSize
                          << " lastSize: " << lastSize
                          << " requiredMemory: " << requiredMemory << "\n");
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
  for (GpuAllocOp alloc : allocs) {
    auto bufferType = alloc.getOutput().getType();
    auto numElements = bufferType.getNumElements();
    auto elementBytes = getByteWidth(bufferType.getElementType());
    auto rank = bufferType.getRank();
    assert(elementBytes == 1 && "All GpuAllocOp should allocate bytes");
    assert(rank == 1 && "Rank should be one");

    rewriter.setInsertionPointAfter(alloc);
    Location loc = alloc->getLoc();

    if (i == allocs.size() - 1) {
      offset = 0;
    }
    LLVM_DEBUG(llvm::dbgs() << "offset: " << offset
                            << " numElements: " << numElements << "\n");

    bool replaceWithSubView = offset == 0;
    if (!replaceWithSubView) {
      size_t numViews = 0;
      for (OpOperand &use : alloc->getUses()) {
        Operation *owner = use.getOwner();
        if (auto view = dyn_cast<memref::ViewOp>(owner)) {
          Value byteOffset =
              rewriter.createOrFold<arith::ConstantIndexOp>(loc, offset);

          auto newView = rewriter.create<memref::ViewOp>(
              loc, view.getType(), ldsByteBuffer, byteOffset,
              view.getDynamicSizes());
          // rewriter.replaceAllOpUsesWith(view, newView);
          rewriter.replaceAllUsesWith(view->getResults(),
                                      newView->getResults());
          rewriter.eraseOp(view);
          ++numViews;
        } else {
          llvm_unreachable("All uses should be views");
        }
      }
      if (numViews > 0) {
        rewriter.eraseOp(alloc);
      } else {
        replaceWithSubView = true;
      }
    }

    if (replaceWithSubView) {
      auto subView = rewriter.create<memref::SubViewOp>(
          loc, ldsByteBuffer, ArrayRef<int64_t>{offset},
          ArrayRef<int64_t>{numElements}, ArrayRef<int64_t>{1});
      rewriter.replaceOp(alloc, subView);
    }

    offset += numElements;
    i++;
  }
}

static FailureOr<std::tuple<int64_t, int64_t, ArrayAttr>>
getIdToLDS(ThreadwiseWriteAllOp &op, OpBuilder &b) {

  ArrayAttr srcTransform = op.getExtraViewsAttr();
  SetVector<StringRef> dimensionsToRemove;
  dimensionsToRemove.insert("g_block");
  dimensionsToRemove.insert("m_block");
  dimensionsToRemove.insert("n_block");
  FailureOr<ArrayAttr> maybeIdToLDS =
      removeUpperDims(b, srcTransform, dimensionsToRemove);
  if (failed(maybeIdToLDS)) {
    return failure();
  }
  ArrayAttr idToLDS = maybeIdToLDS.value();

  int64_t mPerBlock = getLowerShape(idToLDS)[0];
  int64_t nPerBlock = getLowerShape(idToLDS)[1];

  return std::make_tuple(mPerBlock, nPerBlock, idToLDS);
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

    // Convert from reg -> memory transform to reg -> block
    int64_t mPerBlock, nPerBlock;
    ArrayAttr idToLDS;
    FailureOr<std::tuple<int64_t, int64_t, ArrayAttr>> maybeTuple =
        getIdToLDS(op, b);
    if (failed(maybeTuple)) {
      return failure();
    }
    std::tie(mPerBlock, nPerBlock, idToLDS) = maybeTuple.value();

    // TODO: decide if we want to skip this pass
    // if writes are already coalesced and 128b

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> cShape;
    cShape = op.getDest().getType().getShape();
    int64_t G = cShape[0];
    int64_t M = cShape[1];
    int64_t N = cShape[2];
    LLVM_DEBUG(llvm::dbgs() << "G: " << G << " N: " << N << " M: " << M
                            << " nPerBlock: " << nPerBlock
                            << " mPerBlock: " << mPerBlock << "\n");
    FailureOr<IntegerAttr> maybeBlockSize = getBlockSize(op);
    if (failed(maybeBlockSize)) {
      return failure();
    }
    int64_t blockSize = maybeBlockSize.value().getValue().getSExtValue();
    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;
    bool useIndexDiffs = true;
    bool forceUnroll = true;
    int64_t ldsRequiredBytes = mPerBlock * nPerBlock * getByteWidth(destType);

    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};

    // Decide register vectorization.
    constexpr int64_t dimensionM = 1;
    constexpr int64_t dimensionN = 2;
    int64_t dataPerThread = (nPerBlock * mPerBlock) / blockSize;
    VectorizationResult mVectorRes =
        getMaxVectorization(matC, dimensionM, /*inputDimLen=*/
                            dataPerThread, matC.getDefiningOp());
    int64_t mVectorLen = mVectorRes.max;
    VectorizationResult nVectorRes =
        getMaxVectorization(matC, dimensionN, /*inputDimLen=*/
                            dataPerThread, matC.getDefiningOp());
    int64_t nVectorLen = nVectorRes.max;

    int64_t dim;
    if (mVectorLen > nVectorLen) {
      dim = dimensionM;
    } else {
      dim = dimensionN;
    }

    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    // Allocate LDS for output.
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefOutputType =
        MemRefType::get({ldsRequiredBytes}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsBufferOutput = b.create<GpuAllocOp>(loc, ldsMemRefOutputType);
    auto typedBuffer = viewBufferAs(b, ldsBufferOutput, destType);

    // Convert from raw -> mPerBlock, nPerBlock
    TopDownTMBuilder mnToRaw(b, {"gemmM", "gemmN"}, {mPerBlock, nPerBlock});
    if (dim == dimensionN) {
      mnToRaw.unmerge("flatten", 0, {"gemmM", "gemmN"}, {mPerBlock, nPerBlock});
    } else {
      mnToRaw.unmerge("flatten", 0, {"gemmN", "gemmM"}, {nPerBlock, mPerBlock});
    }
    auto mnToRawAttr = mnToRaw.get();

    SmallVector<Attribute> transformMNToRawAttrs;
    transformMNToRawAttrs.push_back(mnToRawAttr);
    ArrayAttr transformMNToRaw = b.getArrayAttr(transformMNToRawAttrs);

    auto ldsBufferMNToRaw = transform(b, typedBuffer, transformMNToRaw);

    // Store C results to LDS.
    b.create<ThreadwiseWriteAllOp>(loc, convertedC, ldsBufferMNToRaw,
                                   /*extraViews=*/idToLDS,
                                   /*extraIndices=*/ValueRange{tid},
                                   op.getFeatures(), StoreMethod::Set,
                                   /*forceUnroll=*/forceUnroll,
                                   /*useIndexDiffs=*/useIndexDiffs);

    // Load from LDS to registers.
    int64_t elementsWrittenPerThread = 128 / destType.getIntOrFloatBitWidth();
    elementsWrittenPerThread =
        math_util::gcd(dataPerThread, elementsWrittenPerThread);
    int64_t iter = dataPerThread / elementsWrittenPerThread;
    LLVM_DEBUG(llvm::dbgs()
               << "blockSize: " << blockSize
               << " dataPerThread: " << dataPerThread
               << " elementsWrittenPerThread: " << elementsWrittenPerThread
               << " iter: " << iter << "\n");
    float row;
    if (dim == dimensionM) {
      LLVM_DEBUG(llvm::dbgs() << "dim = M\n");
      row = mPerBlock;
    } else {
      row = nPerBlock;
      LLVM_DEBUG(llvm::dbgs() << "dim = N\n");
    }
    float rowsPerWave = (elementsWrittenPerThread * 64) / row;
    LLVM_DEBUG(llvm::dbgs()
               << "row: " << row << " rowsPerWave: " << rowsPerWave << "\n");

    Value finalC =
        gpuAlloc(b, loc, dataPerThread, destType, AddressSpace::Private);

    TopDownTMBuilder tidIterMerge(b, {"tid", "iter"},
                                  {blockSize, dataPerThread});
    SmallVector<StringRef, 5> throughDims{"tid"};
    tidIterMerge.passThrough(throughDims);
    tidIterMerge.merge({"iter", "numElements"}, {1, 2}, "iter",
                       {iter, elementsWrittenPerThread});
    auto tidIterMergeAttr = tidIterMerge.get();

    auto tidIterFlatten =
        TopDownTMBuilder::below(tidIterMerge, tidIterMergeAttr);
    tidIterFlatten.unmerge("flattenBlock", 0, {"iter", "tid", "numElements"},
                           {iter, blockSize, elementsWrittenPerThread});
    auto tidIterFlattenAttr = tidIterFlatten.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(tidIterMergeAttr);
    transformAttrs.push_back(tidIterFlattenAttr);

    ArrayAttr ldsRead = b.getArrayAttr(transformAttrs);
    auto ldsBufferForLoad = transform(b, typedBuffer, ldsRead);

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    b.create<ThreadwiseReadIntoOp>(loc, ldsBufferForLoad, finalC,
                                   b.getArrayAttr({}), ValueRange{tid},
                                   forceUnroll, useIndexDiffs);

    // Save to memory
    // Create views as gridwise sub-tile of C
    TopDownTMBuilder tidIterMergeMem(
        b, {"g_block", "m_block", "n_block", "tid", "iter"},
        {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], blockSize,
         dataPerThread},
        loc);
    tidIterMergeMem.passThrough({"g_block", "m_block", "n_block", "tid"});
    tidIterMergeMem.merge({"iter", "numElements"}, {4, 5}, "iter",
                          {iter, elementsWrittenPerThread});
    auto tidIterMergeMemAttr = tidIterMergeMem.get();

    TopDownTMBuilder tidIterFlattenMem =
        TopDownTMBuilder::below(tidIterMergeMem, tidIterMergeMemAttr);
    tidIterFlattenMem.passThrough({"g_block", "m_block", "n_block"});
    tidIterFlattenMem.unmerge("flattenBlock", 3, {"iter", "tid", "numElements"},
                              {iter, blockSize, elementsWrittenPerThread});
    auto tidIterFlattenMemAttr = tidIterFlattenMem.get();

    auto flattenToBlockCoord =
        TopDownTMBuilder::below(tidIterFlattenMem, tidIterFlattenMemAttr);
    flattenToBlockCoord.passThrough({"g_block", "m_block", "n_block"});
    if (dim == dimensionN) {
      flattenToBlockCoord.merge({"block_m", "block_n"}, {3, 4}, "flattenBlock",
                                {mPerBlock, nPerBlock});
    } else {
      flattenToBlockCoord.merge({"block_n", "block_m"}, {3, 4}, "flattenBlock",
                                {nPerBlock, mPerBlock});
    }
    TransformMapAttr flattenToBlockCoordAttr = flattenToBlockCoord.get();

    auto toMatrixC =
        TopDownTMBuilder::below(flattenToBlockCoord, flattenToBlockCoordAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {"g_block"});
    toMatrixC.unmerge("gemmM", 1, {"m_block", "block_m"},
                      {bidGridLengths[1], mPerBlock});
    toMatrixC.unmerge("gemmN", 2, {"n_block", "block_n"},
                      {bidGridLengths[2], nPerBlock});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();

    SmallVector<Attribute> transformAttrsStore{
        tidIterMergeMemAttr, tidIterFlattenMemAttr, flattenToBlockCoordAttr,
        toMatrixCAttr};
    ArrayAttr idToMatrixCMaps = b.getArrayAttr(transformAttrsStore);

    b.create<ThreadwiseWriteAllOp>(loc, finalC, matC, idToMatrixCMaps,
                                   /*extraIndices=*/
                                   op.getExtraIndices(), op.getFeatures(),
                                   op.getStoreMethod(), forceUnroll,
                                   useIndexDiffs);

    b.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void RockGemmOutputSwizzlePass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = func->getContext();
  IRRewriter rewriter(ctx);

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  SmallVector<ThreadwiseWriteAllOp> writes;
  func.walk([&](ThreadwiseWriteAllOp threadwiseWriteAll) {
    MemRefType destMemRefType =
        cast<MemRefType>(threadwiseWriteAll.getDest().getType());
    // keep ThreadwiseWriteAllOp that save to global memory
    if (hasGlobalMemoryAddressSpace(destMemRefType)) {
      // Compute LDS size

      int64_t mPerBlock, nPerBlock;
      ArrayAttr idToLDS;
      FailureOr<std::tuple<int64_t, int64_t, ArrayAttr>> maybeTuple =
          getIdToLDS(threadwiseWriteAll, rewriter);
      if (failed(maybeTuple)) {
        signalPassFailure();
      }
      std::tie(mPerBlock, nPerBlock, idToLDS) = maybeTuple.value();
      Type destType = threadwiseWriteAll.getDest().getType().getElementType();
      int64_t ldsRequiredBytes = mPerBlock * nPerBlock * getByteWidth(destType);

      if (failed(checkLDSSize(threadwiseWriteAll, ldsRequiredBytes))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "GemmOutputSwizzle requires too much LDS memory: "
                   << ldsRequiredBytes << " bytes, skipping pass\n");
        return;
      }
      writes.push_back(threadwiseWriteAll);
    }
  });
  if (writes.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "More than one ThreadwiseWriteAllOp writes to "
                               "global memory, skipping pass\n");
  } else if (writes.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No ThreadwiseWriteAllOp writes to "
                               "global memory, skipping pass\n");
  } else {
    // Rewrite
    RewritePatternSet patterns(&getContext());
    patterns.add<ThreadwiseWriteAllRewritePattern>(&getContext());

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    for (Operation *op : writes) {
      if (failed(applyOpPatternsAndFold(op, std::move(patterns), config))) {
        signalPassFailure();
      }
    }

    // Reuse LDS, we assume the last GpuAllocOp can reuse previous
    // LDS allocations because they are no longer needed
    // Note: this is a temporary trick that will be solved here:
    // https://github.com/ROCm/rocMLIR-internal/issues/1487
    reuseDeadLDS(func);
  }
}
