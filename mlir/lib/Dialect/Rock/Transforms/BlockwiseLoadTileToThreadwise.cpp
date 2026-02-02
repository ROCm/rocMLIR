//===- BlockwiseLoadTileToThreadwise.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers `rock.blockwise_load_tile` to rock.threadwise_* ops.
// The pass will create a number of stages depending on the load type:
// - Default: Creates two stages, (1) load from memory, (2) write to LDS.
// - BypassLDS: Bypasses LDS and loads from device memory to registers directly
// (only one stage).
// - DoubleBuffer: Creates three stages, (1) load from memory, (2) write to LDS,
// (3) load to registers.
//
//===----------------------------------------------------------------------===//

#include "GridLayoutEmitter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/LdsTransposeLoad.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKBLOCKWISELOADTILETOTHREADWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-blockwise-load-tile-to-threadwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using mlir::gpu::AddressSpace;

namespace {
struct RockBlockwiseLoadTileToThreadwisePass
    : public rock::impl::RockBlockwiseLoadTileToThreadwisePassBase<
          RockBlockwiseLoadTileToThreadwisePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

class LoweringBlockwiseLoadTileOp final
    : public OpConversionPattern<rock::BlockwiseLoadTileOp> {
  using OpConversionPattern<rock::BlockwiseLoadTileOp>::OpConversionPattern;

  // Generate the Read loop from LDS.  So we read A[0:mRepeats,
  // 0:kBasePerThread] and B[0:nRepeats, 0:kBasePerThread] before entering the
  // MMA loop
  void generateReadLoop(
      Location loc, PatternRewriter &b,
      const std::unique_ptr<rock::accel::AccelEmitter> &accelEmitterPtr,
      Value tid, StringRef dName, Value ldsView, Value regs, int64_t blockSize,
      bool forceUnroll, const BlockwiseMatrixParamsAttr &matrixParams,
      LDSTransposeConfigAttr transposeAttr = nullptr,
      bool useLdsTransposeLoad = false) const {

    // wrapLDSBufferForLoad is reading a single set of Ks into private memory
    // A/B[m/n, 0:kBasePerThread]
    Value ldsViewForLoad = accelEmitterPtr->wrapLDSBufferForLoad(
        b, loc, ldsView, matrixParams, blockSize, dName, useLdsTransposeLoad);

    // We enhance the transformation from wrapLDSBufferForLoad using a builder
    // that, given a single index, splits it into "m"("n") and "k" and lets
    // tid pass through. We can give those indices to wrapLDSBufferForLoad which
    // should compute the right transform

    StringRef dkName = (dName == "m") ? "mk" : "nk";

    // Read from LDS buffer
    ArrayRef<int64_t> ldsShape =
        cast<ShapedType>(ldsViewForLoad.getType()).getShape();
    assert(ldsShape.size() == 3);
    assert(ldsShape[0] == blockSize);
    TopDownTMBuilder mkBuilder(b, {"tid", dkName},
                               {blockSize, ldsShape[1] * ldsShape[2]}, loc);
    mkBuilder.passThrough("tid");
    mkBuilder.merge({dName, "k"}, {1, 2}, dkName, {ldsShape[1], ldsShape[2]});
    ldsViewForLoad =
        rock::transform(b, ldsViewForLoad, b.getArrayAttr({mkBuilder.get()}));

    ArrayRef<int64_t> regShape = cast<ShapedType>(regs.getType()).getShape();
    assert(regShape.size() == 2 || regShape.size() == 1);
    if (regShape.size() == 2) {
      TopDownTMBuilder mkRegBuilder(b, {dkName}, {regShape[0] * regShape[1]},
                                    loc);
      mkRegBuilder.merge({dName, "k"}, {0, 1}, dkName,
                         {regShape[0], regShape[1]});
      regs = rock::transform(b, regs, b.getArrayAttr({mkRegBuilder.get()}));
    }

    ThreadwiseReadIntoOp::create(b, loc, ldsViewForLoad, regs,
                                 b.getArrayAttr({}), ValueRange{tid},
                                 /*forceUnroll=*/forceUnroll,
                                 /*useIndexDiffs=*/true,
                                 /*ldsTransposeConfig=*/transposeAttr);
  }

  std::pair<StageOp, bool> createOrGetStage(PatternRewriter &b, Location loc,
                                            StringRef name,
                                            Operation *parentOp) const {
    StageOp stageOp = nullptr;
    parentOp->walk([&](StageOp op) {
      if (op.getName() == name)
        stageOp = op;
    });
    bool isNew = !stageOp;
    if (isNew) {
      stageOp = StageOp::create(b, loc, name);
      stageOp.getRegion().emplaceBlock();
    }

    return std::make_pair(stageOp, isNew);
  }

  // Compute firstPageIdx by evaluating the composed transforms.
  FailureOr<Value> computeFirstPageIdx(PatternRewriter &b, Location loc,
                                       Value source, ValueRange indices,
                                       ArrayAttr gridSubTile,
                                       int64_t pageSize) const {
    Value wrappedSourceForFlat = transform(b, source, gridSubTile);

    // indices = [kIter, g_block, m_block, n_block, tid]
    // Replace tid with 0 to compute the tile's starting position (origin).
    // This gives us the flat offset of the first element in the tile,
    // which we then divide by pageSize to get the first page index.
    SmallVector<Value> tileOriginIndices(indices.begin(), indices.end());
    Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
    if (!tileOriginIndices.empty())
      tileOriginIndices.back() = zero;

    FailureOr<Value> maybeTileStartFlat =
        computeFlatPosition(b, loc, wrappedSourceForFlat, tileOriginIndices);
    if (failed(maybeTileStartFlat))
      return failure();

    Value pageSizeVal = b.createOrFold<arith::ConstantIndexOp>(loc, pageSize);
    return arith::DivUIOp::create(b, loc, *maybeTileStartFlat, pageSizeVal)
        .getResult();
  }

  // Emit the page pointer loading logic to LDS.
  void emitPagePointerLoads(PatternRewriter &b, Location loc, Value pageTable,
                            Value ldsPagePtrs, Value firstPageIdx,
                            int64_t numPages, int64_t numPagesPerBatch) const {
    Value tid = WorkitemIdOp::create(b, loc, b.getIndexType());
    Value numPagesVal = b.createOrFold<arith::ConstantIndexOp>(loc, numPages);
    Value numPagesPerBatchVal =
        b.createOrFold<arith::ConstantIndexOp>(loc, numPagesPerBatch);

    // Get number of batches from page table shape for bounds checking
    auto pageTableType = cast<MemRefType>(pageTable.getType());
    int64_t numBatches = pageTableType.getShape()[0];
    Value numBatchesVal =
        b.createOrFold<arith::ConstantIndexOp>(loc, numBatches);

    // Only threads with tid < numPagesForTile participate in loading.
    // Each such thread either loads from page table or stores 0 to its LDS
    // slot.
    Value withinTileBound = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, tid, numPagesVal);

    scf::IfOp::create(
        b, loc, withinTileBound,
        [&](OpBuilder &outerThenBuilder, Location outerThenLoc) {
          // This thread is responsible for LDS slot [tid]
          // globalPageIdx is across all batches
          Value globalPageIdx = arith::AddIOp::create(
              outerThenBuilder, outerThenLoc, firstPageIdx, tid);

          // Split into batch and local page indices:
          // batchIdx = globalPageIdx / numPagesPerBatch
          // localPageIdx = globalPageIdx % numPagesPerBatch
          Value batchIdx =
              arith::DivUIOp::create(outerThenBuilder, outerThenLoc,
                                     globalPageIdx, numPagesPerBatchVal);
          Value localPageIdx =
              arith::RemUIOp::create(outerThenBuilder, outerThenLoc,
                                     globalPageIdx, numPagesPerBatchVal);

          // Check that batch index is within bounds.
          Value withinTableBound = arith::CmpIOp::create(
              outerThenBuilder, outerThenLoc, arith::CmpIPredicate::ult,
              batchIdx, numBatchesVal);

          // Select the pointer value: load from page table if in bounds, else 0
          scf::IfOp ptrIfOp = scf::IfOp::create(
              outerThenBuilder, outerThenLoc, withinTableBound,
              [&](OpBuilder &thenBuilder, Location thenLoc) {
                // Load page pointer from page table
                Value zeroIdx =
                    thenBuilder.createOrFold<arith::ConstantIndexOp>(thenLoc,
                                                                     0);
                SmallVector<Value> pageTableIndices = {batchIdx, localPageIdx,
                                                       zeroIdx};
                Value ptr = memref::LoadOp::create(thenBuilder, thenLoc,
                                                   pageTable, pageTableIndices);
                scf::YieldOp::create(thenBuilder, thenLoc, ptr);
              },
              [&](OpBuilder &elseBuilder, Location elseLoc) {
                Value nullPtr = elseBuilder.createOrFold<arith::ConstantIntOp>(
                    elseLoc, 0, 64);
                scf::YieldOp::create(elseBuilder, elseLoc, nullPtr);
              });

          // Store the selected pointer to LDS
          Value ptrToStore = ptrIfOp.getResult(0);
          memref::StoreOp::create(outerThenBuilder, outerThenLoc, ptrToStore,
                                  ldsPagePtrs, tid);
          scf::YieldOp::create(outerThenBuilder, outerThenLoc);
        });
  }

  LogicalResult matchAndRewrite(rock::BlockwiseLoadTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final {
    Location loc = op.getLoc();
    ValueRange indices = op.getSourceIndices();
    Value source = op.getSource();
    Value ldsByteBuffer = op.getDestLDS();
    Value destRegisters = op.getDestRegisters();
    BlockwiseMatrixParamsAttr matrixParamsA = op.getMatrixParamsA();
    BlockwiseMatrixParamsAttr matrixParamsB = op.getMatrixParamsB();
    BlockwiseMatrixParamsAttr matrixParams =
        op.getIsA() ? matrixParamsA : matrixParamsB;

    StringRef arch = rock::getArchValue(op);
    rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
    GemmFeatures features = archInfo.defaultFeatures;
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    uint32_t blockSize = op.getBlockSize();

    int64_t G = matrixParamsA.getG();
    int64_t M = matrixParamsA.getD();
    int64_t N = matrixParamsB.getD();
    bool isA = op.getIsA();
    StringRef dName = isA ? "m" : "n";

    // Check if LDS transpose is enabled for this operand
    bool ldsTransposeEnabled = matrixParams.getLdsTransposeEnabled();

    bool doRotateWithK = matrixParams.getRotateDWithK();
    bool doSwapThreadIterSubDims = matrixParams.getSwapThreadIterSubDims();
    bool ldsLayoutDxK = matrixParams.getLDSLayoutDxK();
    LDSLayoutConfigDim ldsLayoutConfig{doRotateWithK, doSwapThreadIterSubDims,
                                       ldsLayoutDxK};

    Type elementTypeA = matrixParamsA.getElementType();
    Type elementTypeB = matrixParamsB.getElementType();
    Type elementTypeLoad = op.getElementLoadType();
    Type elementType = op.getElementType();

    auto accelEmitterPtr = accel::AccelEmitter::select(
        features, elementTypeA, elementTypeB, arch, tuningParams);

    if (!accelEmitterPtr)
      return op.emitOpError("Unable to emit accelerator code.");

    int64_t kpack = tuningParams.getKpack();
    int64_t kpacksPerBlock = tuningParams.getKpackPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();
    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;
    bool forceUnroll = tuningParams.getForceUnroll();
    int64_t kPerBlock = kpacksPerBlock * kpack;
    int64_t kGlobal = cast<MemRefType>(source.getType()).getShape()[1];
    int64_t kIters = kGlobal / kPerBlock;
    GemmLoadTileType loadType = op.getLoadType();

    int64_t dPerBlock = isA ? mPerBlock : nPerBlock;
    int64_t copyPerThread = (kPerBlock * dPerBlock) / blockSize;

    bool directToLDS = loadType == GemmLoadTileType::DirectToLDSDefault ||
                       loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;
    bool doubleBuffer = loadType == GemmLoadTileType::DoubleBuffer ||
                        loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;

    // Build LDS transpose config attribute if enabled
    // The decision was already made in GridwiseGemmToBlockwise pass
    LDSTransposeConfigAttr transposeAttr = nullptr;
    if (ldsTransposeEnabled) {
      // Get accelerator dimensions from matrix params and tuning params
      // accelDDim = AccelDDim (for MFMA instructions with blocksMfma=1)
      // accelKDim = accelKDim from BlockwiseMatrixParamsAttr
      int64_t accelDDim = matrixParams.getAccelDDim();
      int64_t accelKDim = matrixParams.getAccelKDim();
      assert(accelDDim > 0 && accelKDim > 0 &&
             "ldsTranspose=true requires valid accel geometry in params");

      // Build transpose config attribute using helper
      transposeAttr = hwtranspose::buildTransposeAttrFromParams(
          b, accelDDim, accelKDim, mPerBlock, nPerBlock, kPerBlock,
          tuningParams.getMPerWave(), tuningParams.getNPerWave(), doubleBuffer,
          isA);

      LLVM_DEBUG(llvm::dbgs()
                 << "[lds_transpose] Built transpose config for operand "
                 << (isA ? "A" : "B") << "\n");
    }

    FailureOr<VectorDimInfo> maybeVecDimInfo =
        getVectorDim(loc, source, elementTypeLoad, blockSize, kPerBlock,
                     dPerBlock, kpack, directToLDS);
    if (failed(maybeVecDimInfo)) {
      return failure();
    }
    VectorDimInfo vecDimInfo = maybeVecDimInfo.value();
    bool isKContiguousDim = vecDimInfo.vectorDim == GemmDimension::K;

    // we want to insert all allocs and transforms before the loop
    Operation *parentOp = op->getParentOp();
    assert(parentOp && "BlockwiseLoadTileOp must have a parent op");
    if (isa<LoopLikeOpInterface>(parentOp))
      b.setInsertionPoint(parentOp);
    else
      b.setInsertionPoint(op);

    Value loadBuffer, storeBuffer;
    if (loadType == GemmLoadTileType::BypassLDS) {
      auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
          gpu::GPUDialect::getPrivateAddressSpace());
      auto accelParams = accelEmitterPtr->getParams();
      int64_t dRepeats = (isA ? accelParams.mRepeats : accelParams.nRepeats);
      // We allocate a buffer of size nRepeats * kpackPerThread * kpack
      auto loadBufferType =
          MemRefType::get({dRepeats * accelParams.kpackPerThread * kpack},
                          elementType, AffineMap{}, privateMemoryAddressSpace);
      loadBuffer = GpuAllocOp::create(b, loc, loadBufferType);
    } else if (directToLDS) {
      loadBuffer = viewBufferAs(b, ldsByteBuffer, elementType);
    } else {
      loadBuffer =
          gpuAlloc(b, loc, copyPerThread, elementType, AddressSpace::Private);
      storeBuffer =
          gpuAlloc(b, loc, copyPerThread, elementType, AddressSpace::Private);
    }
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};

    // Create buffer views early so we can use them for paged attention
    // calculations
    FailureOr<RegsAsMatrixSubTiles> maybeBufferViews;
    if (loadType == GemmLoadTileType::BypassLDS) {
      maybeBufferViews = accelEmitterPtr->createAccelGemmOperandTransforms(
          b, loc, kIters, bidGridLengths, blockSize, vecDimInfo.inDPerThread,
          dName, isKContiguousDim, false);
    } else {
      maybeBufferViews = getLoadRegsAsTileViews(
          b, loc, source, dName, bidGridOrder, bidGridLengths, blockSize,
          kPerBlock, dPerBlock, vecDimInfo.inKPerThread,
          vecDimInfo.inDPerThread, isKContiguousDim, directToLDS);
    }
    if (failed(maybeBufferViews))
      return failure();

    // Handle paged attention setup
    Value pageTable = op.getPageTable();
    std::optional<int64_t> maybePageSize;
    if (auto pageSizeAttr = op.getPageSizeAttr())
      maybePageSize = pageSizeAttr.getInt();
    bool isPagedLoad = pageTable != nullptr && maybePageSize.has_value();
    Value ldsPagePtrs;
    Value firstPageIdx;
    int64_t pageSize = isPagedLoad ? *maybePageSize : 0;
    int64_t numPagesForTile = 0;
    int64_t numPagesPerBatch = 0;

    if (isPagedLoad) {
      // Compute maximum number of pages this tile can span
      int64_t span = (dPerBlock - 1) * kGlobal + (kPerBlock - 1);
      numPagesForTile = (pageSize - 1 + span) / pageSize + 1;

      // Get batch count and pages per batch from page table shape
      // [batch, numPages, 1]
      auto pageTableType = cast<MemRefType>(pageTable.getType());
      numPagesPerBatch = pageTableType.getShape()[1];

      // Allocate LDS for page pointers as i8 byte buffer (required by
      // ReuseLDS), then view as i64. Each i64 pointer is 8 bytes.
      auto ldsMemorySpace =
          b.getAttr<gpu::AddressSpaceAttr>(gpu::AddressSpace::Workgroup);
      int64_t ldsBytesForPagePtrs = numPagesForTile * 8; // sizeof(i64) = 8
      auto ldsPagePtrsByteType = MemRefType::get(
          {ldsBytesForPagePtrs}, b.getI8Type(), AffineMap{}, ldsMemorySpace);
      Value ldsPagePtrsBytes = GpuAllocOp::create(b, loc, ldsPagePtrsByteType);

      // Create i64 view of the byte buffer
      auto ldsPagePtrsViewType = MemRefType::get(
          {numPagesForTile}, b.getI64Type(), AffineMap{}, ldsMemorySpace);
      Value zeroOffset = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
      ldsPagePtrs =
          memref::ViewOp::create(b, loc, ldsPagePtrsViewType, ldsPagePtrsBytes,
                                 zeroOffset, ValueRange{});
    }

    // Set insertion point for stage creation (inside the loop)
    if (isa<LoopLikeOpInterface>(parentOp))
      b.setInsertionPoint(op);

    // For paged loads, we split into two stages:
    // 1. PagePtrLoad: Load page pointers from page table to LDS
    // 2. GlobalRead: Use page pointers to load actual data to LDS (or regs)
    StageOp existingGlobalRead = nullptr;
    if (isPagedLoad) {
      parentOp->walk([&](StageOp op) {
        if (op.getName() == "GlobalRead")
          existingGlobalRead = op;
      });

      // If GlobalRead already exists, set insertion point before it
      if (existingGlobalRead) {
        b.setInsertionPoint(existingGlobalRead);
      }

      // Stage 1: PagePtrLoad - loads page pointers to LDS
      auto [stagePagePtrLoad, stagePagePtrLoadNew] =
          createOrGetStage(b, loc, "PagePtrLoad", parentOp);
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stagePagePtrLoad.getRegion().back());

        // Compute firstPageIdx
        FailureOr<Value> maybeFirstPageIdx = computeFirstPageIdx(
            b, loc, source, indices, maybeBufferViews->gridSubTile, pageSize);
        if (failed(maybeFirstPageIdx))
          return failure();
        Value pagePtrFirstPageIdx = *maybeFirstPageIdx;

        // Load page pointers to LDS
        emitPagePointerLoads(b, loc, pageTable, ldsPagePtrs,
                             pagePtrFirstPageIdx, numPagesForTile,
                             numPagesPerBatch);

        if (stagePagePtrLoadNew) {
          LDSBarrierOp::create(b, loc);
          rock::YieldOp::create(b, loc);
        }
      }

      // Restore insertion point after the existing GlobalRead for subsequent
      // stages
      if (existingGlobalRead) {
        b.setInsertionPointAfter(existingGlobalRead);
      }
    }

    auto [stageGlobalRead, stageGlobalReadNew] =
        createOrGetStage(b, loc, "GlobalRead", parentOp);
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(&stageGlobalRead.getRegion().back());

      FailureOr<RegsAsMatrixSubTiles> maybeBufferViews;
      if (loadType == GemmLoadTileType::BypassLDS) {
        // Check if the other operand uses LDS transpose load
        bool otherOperandUsesLdsTranspose =
            isA ? matrixParamsB.getLdsTransposeEnabled()
                : matrixParamsA.getLdsTransposeEnabled();
        maybeBufferViews = accelEmitterPtr->createAccelGemmOperandTransforms(
            b, loc, kIters, bidGridLengths, blockSize, vecDimInfo.inDPerThread,
            dName, isKContiguousDim, false,
            /*doSplitKAcrossThreadsFirst=*/false, otherOperandUsesLdsTranspose);
      } else {
        maybeBufferViews = getLoadRegsAsTileViews(
            b, loc, source, dName, bidGridOrder, bidGridLengths, blockSize,
            kPerBlock, dPerBlock, vecDimInfo.inKPerThread,
            vecDimInfo.inDPerThread, isKContiguousDim, directToLDS);
      }

      // For paged loads, recompute firstPageIdx in this stage
      // (the page pointers are already in ldsPagePtrs from PagePtrLoad stage)
      if (isPagedLoad) {
        FailureOr<Value> maybeFirstPageIdx = computeFirstPageIdx(
            b, loc, source, indices, maybeBufferViews->gridSubTile, pageSize);
        if (failed(maybeFirstPageIdx))
          return failure();
        firstPageIdx = *maybeFirstPageIdx;
      }

      Value wrappedSource = transform(b, source, maybeBufferViews->gridSubTile);

      if (isPagedLoad) {
        // Create ThreadwiseReadIntoOp with paging attributes
        ThreadwiseReadIntoOp::create(
            b, loc, vectorOfBoolShapedLike(loadBuffer), wrappedSource,
            loadBuffer,
            /*dynamicValidities=*/ValueRange{},
            /*extraViews=*/b.getArrayAttr({}),
            /*extraIndices=*/indices, forceUnroll, /*useIndexDiffs=*/true,
            /*ldsTransposeConfig=*/nullptr,
            /*ldsPagePtrs=*/ldsPagePtrs,
            /*firstPageIndex=*/firstPageIdx,
            /*pageSize=*/b.getIndexAttr(pageSize),
            /*numPagesPerBatch=*/b.getIndexAttr(numPagesPerBatch));
      } else {
        // Standard non-paged path
        ThreadwiseReadIntoOp::create(b, loc, vectorOfBoolShapedLike(loadBuffer),
                                     wrappedSource, loadBuffer,
                                     /*dynamicValidities=*/ValueRange{},
                                     /*extraViews=*/b.getArrayAttr({}),
                                     /*extraIndices=*/indices, forceUnroll,
                                     true,
                                     /*ldsTransposeConfig=*/nullptr);
      }

      // Skip prefetch for paged loads - would need next tile's page pointers
      if (rock::isGlobalPrefetchSupported(arch) && !isPagedLoad) {
        // add one to k_loop to prefetch next iteration
        SmallVector<Value> indicesNext(indices.begin(), indices.end());
        Value one = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
        indicesNext[0] =
            arith::AddIOp::create(b, loc, indicesNext[0], one).getResult();

        // it's acceptable if the indices are out of bounds because we use
        // GLOBAL_PREFETCH_B8 with Speculative Prefetch. See llvm.prefetch
        // documentation in AMDGPUUsage.rst
        rock::ThreadwisePrefetchOp::create(b, loc, wrappedSource,
                                           /*extraViews=*/b.getArrayAttr({}),
                                           /*extraIndices=*/indicesNext,
                                           forceUnroll, true);
      }
      if (stageGlobalReadNew)
        rock::YieldOp::create(b, loc);
    }

    if (loadType == GemmLoadTileType::BypassLDS) {
      auto [stageRegTranspose, stageRegTransposeNew] =
          createOrGetStage(b, loc, "RegTranspose", parentOp);
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stageRegTranspose.getRegion().back());

        accel::AccelEmitterParams accelEmitterParams =
            accelEmitterPtr->getParams();
        int64_t dRepeats = (dName == "m" ? accelEmitterParams.mRepeats
                                         : accelEmitterParams.nRepeats);
        affine::AffineForOp dRepeatsLoop =
            affine::AffineForOp::create(b, loc, 0, dRepeats, 1);
        {
          PatternRewriter::InsertionGuard guard(b);
          b.setInsertionPointToStart(dRepeatsLoop.getBody());
          Value di = dRepeatsLoop.getInductionVar();
          Value subview = destRegisters;
          if (dRepeats > 1) {
            subview = createSliceOfFirstDim(b, loc, subview, di);
          }

          // Check if the other operand uses LDS transpose load
          bool otherOperandUsesLdsTranspose =
              isA ? matrixParamsB.getLdsTransposeEnabled()
                  : matrixParamsA.getLdsTransposeEnabled();
          FailureOr<RegsAsMatrixSubTiles> maybeBufferViews =
              accelEmitterPtr->createAccelGemmOperandTransforms(
                  b, loc, kIters, bidGridLengths, blockSize,
                  vecDimInfo.inDPerThread, dName, isKContiguousDim, false,
                  /*doSplitKAcrossThreadsFirst=*/false,
                  otherOperandUsesLdsTranspose);
          if (failed(maybeBufferViews))
            return failure();
          // InBufferViews provide --> K x D subtile views.
          // Since we are iterating on D dimension, we need to transpose it.
          RegsAsMatrixSubTiles inBufferViewsTr =
              transposeSubTileViews(b, loc, maybeBufferViews.value());
          FailureOr<ArrayAttr> maybeInBufferViewsTrAttr =
              invertTransforms(b, loc, inBufferViewsTr.threadSubTile);
          if (failed(maybeInBufferViewsTrAttr)) {
            return op.emitError("cannot invert inBufferViewsTr.threadSubTile");
          }
          Value viewLoadedBuffer =
              transform(b, loadBuffer, maybeInBufferViewsTrAttr.value());
          ThreadwiseReadIntoOp::create(b, loc, viewLoadedBuffer, subview,
                                       b.getArrayAttr({}), ValueRange{di},
                                       forceUnroll, true);
        }

        if (stageRegTransposeNew)
          rock::YieldOp::create(b, loc);
      }
    } else {
      if (!directToLDS) {
        auto [stageLDSWrite, stageLDSWriteNew] =
            createOrGetStage(b, loc, "LDSWrite", parentOp);
        {
          PatternRewriter::InsertionGuard guard(b);
          b.setInsertionPointToStart(&stageLDSWrite.getRegion().back());

          // Get current workitem ID.
          auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

          assert(directToLDS == false);
          FailureOr<RegsAsMatrixSubTiles> maybeBufferViews =
              getLoadRegsAsTileViews(
                  b, loc, source, dName, bidGridOrder, bidGridLengths,
                  blockSize, kPerBlock, dPerBlock, vecDimInfo.inKPerThread,
                  vecDimInfo.inDPerThread, isKContiguousDim, directToLDS);
          if (failed(maybeBufferViews))
            return failure();
          // We invert the transforms that are iter --> K x D slice of the
          // tensor so that we can view loadBuffer as a K x D tensor
          FailureOr<ArrayAttr> maybeLoadBufferViews =
              invertTransforms(b, loc, maybeBufferViews->threadSubTile);
          if (failed(maybeLoadBufferViews)) {
            return op.emitError(
                "cannot invert maybeBufferViews->threadSubTile");
          }
          Value viewLoadBuffer =
              transform(b, loadBuffer, maybeLoadBufferViews.value());

          FailureOr<RegsAsMatrixSubTiles> maybeLdsStoreViews =
              getPackedRegsAsTileViews(
                  b, loc, source, dName, bidGridOrder, bidGridLengths,
                  blockSize, kPerBlock, dPerBlock, vecDimInfo.inKPerThread,
                  vecDimInfo.inDPerThread, kpack, isKContiguousDim,
                  ldsLayoutConfig.doSwapThreadIterSubDims);
          if (failed(maybeLdsStoreViews))
            return failure();

          FailureOr<ArrayAttr> maybeStoreBufferViews =
              invertTransforms(b, loc, maybeLdsStoreViews->threadSubTile);
          if (failed(maybeStoreBufferViews)) {
            return op.emitError(
                "cannot invert maybeLdsStoreViews->threadSubTile");
          }
          Value viewStoreBuffer =
              transform(b, storeBuffer, maybeStoreBufferViews.value());

          Type ldsReadType = vectorTypeOrSelf(elementType, kpack);
          FailureOr<Value> maybeWrappedLds = wrapLDSBufferForStore(
              b, loc, ldsByteBuffer, ldsReadType, kpacksPerBlock, dName,
              dPerBlock, vecDimInfo.inKPerThread, vecDimInfo.inDPerThread,
              ldsLayoutConfig.doRotateWithK);
          if (failed(maybeWrappedLds))
            return maybeWrappedLds;
          // This is KxD view of the flat LDS buffer
          Value wrappedLds = maybeWrappedLds.value();
          // This will produce a (tid, iter) --> flat LDS view
          wrappedLds =
              transform(b, wrappedLds, maybeLdsStoreViews->blockSubTile);

          // Emit potentially-transposing copies to store buffer. This is here
          // both to enable code motion for fusions and to prevent the accesses
          // to the memory from breaking software pipelining.
          ThreadwiseCopyOp::create(b, loc, viewLoadBuffer, ValueRange{},
                                   viewStoreBuffer, ValueRange{}, false, false);
          // Emit blockwise stores
          ThreadwiseWriteAllOp::create(b, loc, storeBuffer, wrappedLds,
                                       /*extraViews=*/b.getArrayAttr({}),
                                       /*extraIndices=*/ValueRange{tid},
                                       StoreMethod::Set,
                                       /*forceUnroll=*/forceUnroll,
                                       /*useIndexDiffs=*/true);
          if (stageLDSWriteNew)
            rock::YieldOp::create(b, loc);
        }
      }

      if (doubleBuffer) {
        // Pipeline pass will remove this if the loop uses pipelining
        LDSBarrierOp::create(b, loc);

        // If we are running double-buffered pipelines, it makes sense to also
        // parallelize the LDSRead/MMA stages. We do this here, by splitting the
        // MMA loop in two separate stages
        auto [stageLDSRead, stageLDSReadNew] =
            createOrGetStage(b, loc, "LDSRead", parentOp);
        {
          // Read from LDS into registers
          PatternRewriter::InsertionGuard guard(b);
          b.setInsertionPointToStart(&stageLDSRead.getRegion().back());

          // Get current workitem ID.
          auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

          Value ldsViewForGemm;
          if (directToLDS) {
            ldsViewForGemm = viewBufferAs(b, ldsByteBuffer, elementType);
          } else {
            Type ldsReadType = vectorTypeOrSelf(elementType, kpack);
            ldsViewForGemm = viewBufferAs(b, ldsByteBuffer, ldsReadType);
          }

          // Determine if the other operand uses LDS transpose load
          // If we're loading A, check if B uses transpose; if loading B, check
          // A
          bool useLdsTransposeLoad =
              isA ? matrixParamsB.getLdsTransposeEnabled()
                  : matrixParamsA.getLdsTransposeEnabled();
          generateReadLoop(loc, b, accelEmitterPtr, tid, dName, ldsViewForGemm,
                           destRegisters, blockSize, forceUnroll, matrixParams,
                           transposeAttr, useLdsTransposeLoad);
          if (stageLDSReadNew)
            rock::YieldOp::create(b, loc);
        }
      }
    }
    b.eraseOp(op);

    return success();
  }
};

void RockBlockwiseLoadTileToThreadwisePass::runOnOperation() {
  auto &ctx = getContext();
  ConversionTarget target(ctx);

  target.addLegalDialect<rock::RockDialect, affine::AffineDialect,
                         arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect>();
  target.addIllegalOp<rock::BlockwiseLoadTileOp>();
  auto func = getOperation();

  RewritePatternSet patterns(&ctx);
  patterns.add<LoweringBlockwiseLoadTileOp>(&ctx);
  if (failed(applyPartialConversion(func, target, std::move(patterns))))
    signalPassFailure();
}
