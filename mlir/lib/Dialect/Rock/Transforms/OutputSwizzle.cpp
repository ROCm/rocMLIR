//===- OutputSwizzle - MLIR Rock ops lowering passes -----===//
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
#define GEN_PASS_DEF_ROCKOUTPUTSWIZZLEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-output-swizzle"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using mlir::gpu::AddressSpace;

namespace {
struct RockOutputSwizzlePass
    : public rock::impl::RockOutputSwizzlePassBase<RockOutputSwizzlePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static bool hasPrivateMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace)) {

    return gpuAttr.getValue() == AddressSpace::Private;
  }
  return false;
}

static bool hasGlobalMemoryAddressSpace(MemRefType type) {
  return !gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(type) &&
         !hasPrivateMemoryAddressSpace(type);
}

static int64_t getLDSTotalSize(func::FuncOp &func) {
  int64_t totalSize = 0;
  func.walk([&](GpuAllocOp gpuAlloc) {
    mlir::MemRefType type = gpuAlloc.getOutput().getType();
    auto memSpaceValue =
        dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace())
            .getValue();
    if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      totalSize += type.getNumElements() * getByteWidth(type.getElementType());
    }
  });
  return totalSize;
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
static LogicalResult reuseDeadLDS(func::FuncOp &func) {
  // Retrieve all GpuAlloc operations targeting LDS and
  // calculate the total memory space required for all operations
  // except the last one. Additionally, compute the memory space
  // required by the last GpuAlloc operation.
  SmallVector<std::tuple<GpuAllocOp, int64_t>> allocs;
  int64_t totalSize = 0;
  int64_t lastSize = 0;
  func.walk([&](GpuAllocOp gpuAlloc) {
    auto type = gpuAlloc.getOutput().getType();
    auto memSpaceValue =
        dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace())
            .getValue();
    if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
      allocs.push_back({gpuAlloc, totalSize});
      lastSize = type.getNumElements() * getByteWidth(type.getElementType());
      totalSize += lastSize;
    }
  });
  // There should be at least two LDS allocations
  if (allocs.size() < 2) {
    LLVM_DEBUG(llvm::dbgs() << "Expected at least two GpuAllocOp, found "
                            << allocs.size() << "\n");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "number of GpuAllocOp ops to rewrite: "
                          << allocs.size() << "\n");
  int64_t deadSize = totalSize - lastSize;

  // we assume the last GpuAllocOp can reuse previous
  // LDS allocations because they are no longer needed

  // New allocation
  IRRewriter rewriter(func->getContext());
  int64_t requiredMemory = std::max(deadSize, lastSize);
  LLVM_DEBUG(llvm::dbgs() << "deadSize: " << deadSize
                          << " lastSize: " << lastSize
                          << " requiredMemory: " << requiredMemory << "\n");
  auto workgroupMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getWorkgroupAddressSpace());
  auto ldsMemRefBufferType =
      MemRefType::get({requiredMemory}, rewriter.getI8Type(), AffineMap{},
                      workgroupMemoryAddressSpace);

  rewriter.setInsertionPoint(std::get<0>(allocs[0]));
  Location loc = std::get<0>(allocs[0])->getLoc();
  auto ldsByteBuffer = rewriter.create<GpuAllocOp>(loc, ldsMemRefBufferType);

  // Rewrite all GpuAllocs as ViewOp
  for (auto [i, allocTuple] : llvm::enumerate(allocs)) {
    GpuAllocOp alloc;
    int64_t offset;
    std::tie(alloc, offset) = allocTuple;
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

    // last GpuAllocOp reuses LDS memory
    if (i == allocs.size() - 1) {
      offset = 0;
    }
    LLVM_DEBUG(llvm::dbgs() << "offset: " << offset
                            << " numElements: " << numElements << "\n");
    Value byteOffset =
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, offset);

    auto newViewType =
        MemRefType::get({numElements}, rewriter.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    rewriter.replaceOpWithNewOp<memref::ViewOp>(
        alloc, newViewType, ldsByteBuffer, byteOffset, ValueRange{});
  }
  return success();
}

static std::optional<std::tuple<int64_t, int64_t, ArrayAttr>>
getIdToLDS(ThreadwiseWriteAllOp &op, OpBuilder &b) {
  ArrayAttr srcTransform = op.getExtraViewsAttr();
  SetVector<StringRef> dimensionsToRemove;
  dimensionsToRemove.insert("g_block");
  dimensionsToRemove.insert("m_block");
  dimensionsToRemove.insert("n_block");
  FailureOr<ArrayAttr> maybeIdToLDS =
      removeUpperDims(b, srcTransform, dimensionsToRemove);
  if (failed(maybeIdToLDS)) {
    LLVM_DEBUG(llvm::dbgs() << "getIdToLDS failed\n");
    return std::nullopt;
  }
  ArrayAttr idToLDS = maybeIdToLDS.value();

  ArrayRef<int64_t> shape = getLowerShape(idToLDS);
  if (shape.size() != 2) {
    LLVM_DEBUG(llvm::dbgs()
               << "Currently, this pass works only for two dimensions\n");
    return std::nullopt;
  }
  int64_t dim0PerBlock = shape[0];
  int64_t dim1PerBlock = shape[1];

  return std::make_tuple(dim0PerBlock, dim1PerBlock, idToLDS);
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
    int64_t dim0PerBlock, dim1PerBlock;
    ArrayAttr idToLDS;
    std::optional<std::tuple<int64_t, int64_t, ArrayAttr>> maybeBlockInfo =
        getIdToLDS(op, b);
    if (!maybeBlockInfo.has_value()) {
      return failure();
    }
    std::tie(dim0PerBlock, dim1PerBlock, idToLDS) = maybeBlockInfo.value();

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> cShape;
    cShape = op.getDest().getType().getShape();
    if (cShape.size() != 3) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << " dim1PerBlock: " << dim1PerBlock
                            << " dim0PerBlock: " << dim0PerBlock << "\n");
    FailureOr<IntegerAttr> maybeGridSize = getGridSize(op);
    if (failed(maybeGridSize)) {
      return failure();
    }
    FailureOr<IntegerAttr> maybeBlockSize = getBlockSize(op);
    if (failed(maybeBlockSize)) {
      return failure();
    }
    int64_t blockSize = maybeBlockSize.value().getValue().getSExtValue();
    bool useIndexDiffs = true;
    bool forceUnroll = true;
    int64_t ldsRequiredBytes =
        dim0PerBlock * dim1PerBlock * getByteWidth(destType);

    // Decide register vectorization.
    constexpr int64_t dimensionM = 1;
    constexpr int64_t dimensionN = 2;
    int64_t dataPerThread = (dim1PerBlock * dim0PerBlock) / blockSize;
    if ((dim1PerBlock * dim0PerBlock) % blockSize != 0) {
      return failure();
    }

    VectorizationResult mVectorRes =
        getMaxVectorization(matC, dimensionM, /*inputDimLen=*/
                            std::nullopt, matC.getDefiningOp());
    int64_t mVectorLen = mVectorRes.max;
    VectorizationResult nVectorRes =
        getMaxVectorization(matC, dimensionN, /*inputDimLen=*/
                            std::nullopt, matC.getDefiningOp());
    int64_t nVectorLen = nVectorRes.max;
    int64_t dim = (mVectorLen > nVectorLen) ? dimensionM : dimensionN;
    int64_t vectorLen = std::max(mVectorLen, nVectorLen);

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

    // Convert from raw -> dim0PerBlock, dim1PerBlock
    TopDownTMBuilder mnToRaw(b, {"gemmM", "gemmN"},
                             {dim0PerBlock, dim1PerBlock});
    if (dim == dimensionN) {
      mnToRaw.unmerge("flatten", 0, {"gemmM", "gemmN"},
                      {dim0PerBlock, dim1PerBlock});
    } else {
      mnToRaw.unmerge("flatten", 0, {"gemmN", "gemmM"},
                      {dim1PerBlock, dim0PerBlock});
    }
    auto mnToRawAttr = mnToRaw.get();

    SmallVector<Attribute> transformMNToRawAttrs;
    transformMNToRawAttrs.push_back(mnToRawAttr);
    ArrayAttr transformMNToRaw = b.getArrayAttr(transformMNToRawAttrs);

    auto ldsBufferMNToRaw = transform(b, typedBuffer, transformMNToRaw);

    // LDS barrier.
    // This barrier is needed because we reuse A and B buffers
    b.create<LDSBarrierOp>(loc);

    // Store C results to LDS.
    b.create<ThreadwiseWriteAllOp>(loc, convertedC, ldsBufferMNToRaw,
                                   /*extraViews=*/idToLDS,
                                   /*extraIndices=*/ValueRange{tid},
                                   op.getFeatures(), StoreMethod::Set,
                                   /*forceUnroll=*/forceUnroll,
                                   /*useIndexDiffs=*/useIndexDiffs);

    // Load from LDS to registers.
    int64_t elementsWrittenPerThread = math_util::gcd(dataPerThread, vectorLen);
    int64_t iter = dataPerThread / elementsWrittenPerThread;
    LLVM_DEBUG(llvm::dbgs()
               << "blockSize: " << blockSize
               << " dataPerThread: " << dataPerThread
               << " elementsWrittenPerThread: " << elementsWrittenPerThread
               << " iter: " << iter << "\n");
    if (dim == dimensionM) {
      LLVM_DEBUG(llvm::dbgs() << "dim = M\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "dim = N\n");
    }

    Value finalC =
        gpuAlloc(b, loc, dataPerThread, destType, AddressSpace::Private);

    TopDownTMBuilder tidIterMerge(b, {"tid", "iter"},
                                  {blockSize, dataPerThread});
    tidIterMerge.passThrough(ArrayRef<StringRef>{"tid"});
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

    SmallVector<int64_t, 5> bidGridLengths;
    SmallVector<StringRef, 5> bidGridOrder;

    llvm::SmallVector<uint32_t> passThrough, passThroughWOTid;
    // drop last dimension (tid)
    passThroughWOTid.reserve(op.getExtraIndices().size() - 1);
    passThrough.reserve(op.getExtraIndices().size());
    for (size_t dim = 0; dim < op.getExtraIndices().size() - 1; ++dim) {
      passThroughWOTid.push_back(dim);
      passThrough.push_back(dim);
    }
    passThrough.push_back(op.getExtraIndices().size() - 1);
    uint32_t idx = op.getExtraIndices().size();
    if (idx < 3) {
      return failure();
    }

    bool isAttention = idx == 3;
    LLVM_DEBUG(llvm::dbgs() << "isAttention: " << isAttention << "\n");

    // Save to memory
    ArrayAttr srcTransform = op.getExtraViewsAttr();
    const auto upperTr = cast<TransformMapAttr>(srcTransform[0]);
    ArrayRef<int64_t> startShape = upperTr.getUpperBounds().asArrayRef();
    llvm::SmallVector<StringRef> startNames;
    startNames.reserve(startShape.size());
    for (auto tr : upperTr.getOps()) {
      ArrayRef<StringRef> upperNames = tr.getUpperNames();
      startNames.append(upperNames.begin(), upperNames.end());
    }
    assert(startNames.size() == startShape.size());

    // get dimension names
    StringRef gDimName = startNames[0];
    StringRef nDimName = startNames[startNames.size() - 3];
    StringRef tidDimName = startNames[startNames.size() - 2];
    StringRef itemDimName = startNames[startNames.size() - 1];

    // get mBlocks and nBlocks
    int64_t nBlocks = startShape[startShape.size() - 3];
    // only valid for !isAttention
    int64_t mBlocks = startShape[startShape.size() - 4];

    TopDownTMBuilder tidIterMergeMem(b, startNames, startShape, loc);
    tidIterMergeMem.passThrough(passThrough, passThrough);
    tidIterMergeMem.merge({"iter", "numElements"}, {idx, idx + 1}, itemDimName,
                          {iter, elementsWrittenPerThread});
    auto tidIterMergeMemAttr = tidIterMergeMem.get();

    TopDownTMBuilder tidIterFlattenMem =
        TopDownTMBuilder::below(tidIterMergeMem, tidIterMergeMemAttr);
    tidIterFlattenMem.passThrough(passThroughWOTid, passThroughWOTid);
    tidIterFlattenMem.unmerge("flattenBlock", idx - 1,
                              {"iter", tidDimName, "numElements"},
                              {iter, blockSize, elementsWrittenPerThread});
    auto tidIterFlattenMemAttr = tidIterFlattenMem.get();

    auto flattenToBlockCoord =
        TopDownTMBuilder::below(tidIterFlattenMem, tidIterFlattenMemAttr);
    flattenToBlockCoord.passThrough(passThroughWOTid, passThroughWOTid);
    if (dim == dimensionN) {
      flattenToBlockCoord.merge({"block_m", "block_n"}, {idx - 1, idx},
                                "flattenBlock", {dim0PerBlock, dim1PerBlock});
    } else {
      flattenToBlockCoord.merge({"block_n", "block_m"}, {idx - 1, idx},
                                "flattenBlock", {dim1PerBlock, dim0PerBlock});
    }
    TransformMapAttr flattenToBlockCoordAttr = flattenToBlockCoord.get();

    auto toMatrixC =
        TopDownTMBuilder::below(flattenToBlockCoord, flattenToBlockCoordAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {gDimName});
    if (isAttention) {
      toMatrixC.passThrough({"gemmM"}, {1}, {"block_m"});
    } else {
      toMatrixC.unmerge("gemmM", 1, {startNames[1], "block_m"},
                        {mBlocks, dim0PerBlock});
    }
    toMatrixC.unmerge("gemmN", 2, {nDimName, "block_n"},
                      {nBlocks, dim1PerBlock});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();

    SmallVector<Attribute> transformAttrsStore{
        tidIterMergeMemAttr, tidIterFlattenMemAttr, flattenToBlockCoordAttr,
        toMatrixCAttr};
    ArrayAttr idToMatrixCMaps = b.getArrayAttr(transformAttrsStore);

    b.replaceOpWithNewOp<ThreadwiseWriteAllOp>(
        op, finalC, matC, idToMatrixCMaps,
        /*extraIndices=*/
        op.getExtraIndices(), op.getFeatures(), op.getStoreMethod(),
        forceUnroll, useIndexDiffs);
    return success();
  }
};

void RockOutputSwizzlePass::runOnOperation() {
  func::FuncOp func = getOperation();
  IRRewriter rewriter(func->getContext());

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  // Get total LDS memory allocated
  int64_t ldsAllocated = getLDSTotalSize(func);

  SmallVector<Operation *, 4> writes;
  func.walk([&writes, &rewriter,
             ldsAllocated](ThreadwiseWriteAllOp threadwiseWriteAll) {
    MemRefType destMemRefType =
        cast<MemRefType>(threadwiseWriteAll.getDest().getType());

    // process ThreadwiseWriteAllOp that save to global memory
    if (hasGlobalMemoryAddressSpace(destMemRefType)) {
      int64_t dim0PerBlock, dim1PerBlock;
      ArrayAttr idToLDS;
      rewriter.setInsertionPoint(threadwiseWriteAll);
      std::optional<std::tuple<int64_t, int64_t, ArrayAttr>> maybeBlockInfo =
          getIdToLDS(threadwiseWriteAll, rewriter);
      if (!maybeBlockInfo.has_value()) {
        return;
      }
      std::tie(dim0PerBlock, dim1PerBlock, idToLDS) = maybeBlockInfo.value();

      Type destType = threadwiseWriteAll.getDest().getType().getElementType();
      int64_t ldsRequiredBytes =
          dim0PerBlock * dim1PerBlock * getByteWidth(destType);

      // not enough LDS memory
      if (failed(checkLDSSize(threadwiseWriteAll, ldsRequiredBytes))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "OutputSwizzle requires too much LDS memory: "
                   << ldsRequiredBytes << " bytes, skipping pass\n");
        return;
      }
      // heuristic: if we need more LDS, skip this pass
      if (ldsRequiredBytes > ldsAllocated) {
        LLVM_DEBUG(llvm::dbgs()
                   << "OutputSwizzle requires more LDS memory, current usage: "
                   << ldsAllocated << " bytes, required: " << ldsRequiredBytes
                   << " bytes, skipping pass\n");
        return;
      }
      // heuristic: check vectorization of iter in the original map
      ArrayAttr srcTransform = threadwiseWriteAll.getExtraViews();
      Value matC = threadwiseWriteAll.getDest();
      Value destView = transform(rewriter, matC, srcTransform);
      auto destViewType = cast<MemRefType>(destView.getType());
      auto destElemType = destViewType.getElementType();
      if (auto elemVecType = dyn_cast<VectorType>(destElemType)) {
        LLVM_DEBUG(llvm::dbgs() << "ThreadwiseWriteAllOp saves a vector type"
                                << ", skipping swizzle\n");
      } else {
        size_t extraIdxCount = threadwiseWriteAll.getExtraIndices().size();
        VectorizationResult vectorRes =
            getMaxVectorization(destView, extraIdxCount);
        int64_t vectorLenBits =
            vectorRes.max * destElemType.getIntOrFloatBitWidth();

        // Ensure a 128-bit store where possible, even if the data is already
        // somawhat vectorized from the matrix instruction. This is important
        // because GCN-based chips only form clauses with 128-bit memory
        // operations, and we may need to revisit this heuristic (as opposed to,
        // say, checking for at least N elements in the vectorization output) on
        // future hardware.
        if (vectorLenBits == 128) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Vectorization of 'iter' is " << vectorLenBits
                     << " bits, skipping swizzle\n");
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Vectorization of 'iter' is " << vectorLenBits
                     << " bits, performing swizzle\n");
          writes.push_back(threadwiseWriteAll);
        }
      }
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

    // TODO: this pass works for one or two dimensions only at the block level
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    if (failed(applyOpPatternsAndFold(writes, std::move(patterns), config))) {
      return signalPassFailure();
    }

    // Reuse LDS, we assume the last GpuAllocOp can reuse previous
    // LDS allocations because they are no longer needed
    // Note: this is a temporary trick that will be solved here:
    // https://github.com/ROCm/rocMLIR-internal/issues/1487
    if (failed(reuseDeadLDS(func))) {
      return signalPassFailure();
    }
  }
}
