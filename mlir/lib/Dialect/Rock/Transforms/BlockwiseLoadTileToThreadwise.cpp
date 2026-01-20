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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
      LDSTransposeConfigAttr transposeAttr = nullptr) const {

    // wrapLDSBufferForLoad is reading a single set of Ks into private memory
    // A/B[m/n, 0:kBasePerThread]
    Value ldsViewForLoad = accelEmitterPtr->wrapLDSBufferForLoad(
        b, loc, ldsView, matrixParams, blockSize, dName);

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

  // Generate paged attention GlobalRead stage logic
  // The page address (i64) is already loaded in GridwiseGemmToBlockwise,
  // so we just use it directly for the memory access.
  //
  // Uses ThreadwiseReadIntoOp with the passed page address. The lowering of
  // ThreadwiseReadIntoOp handles the indirect memory access and vectorization.
  //
  // Thread distribution: Each thread loads (pageSize / blockSize) elements
  // with interleaved access pattern for coalesced memory access.
  LogicalResult
  emitPagedGlobalRead(rock::BlockwiseLoadTileOp op, PatternRewriter &b,
                      Location loc, Value tid, int64_t blockSize) const {
    Value pageAddress = op.getPageAddress();
    Value ldsByteBuffer = op.getDestLDS();

    std::optional<int64_t> pageSizeOpt = std::nullopt;
    if (op.getPageSize())
      pageSizeOpt = op.getPageSize()->getSExtValue();

    if (!ldsByteBuffer) {
      return op.emitOpError("Paged attention requires destLDS buffer");
    }

    if (!pageSizeOpt) {
      return op.emitOpError("Paged attention requires pageSize");
    }

    Type elemType = op.getElementType();
    int64_t pageSize = *pageSizeOpt;

    // View the LDS buffer as element type for the destination (1D)
    Value ldsBufferView = viewBufferAs(b, ldsByteBuffer, elemType);

    // Compute elements per thread for cooperative loading
    // Each thread loads pageSize/blockSize elements with interleaved access
    int64_t elemsPerThread = pageSize / blockSize;
    if (pageSize % blockSize != 0) {
      return op.emitOpError("pageSize must be divisible by blockSize for "
                            "cooperative paged loading");
    }

    // Create a dummy source buffer for iteration pattern.
    // The actual loads go through pageAddress, not this buffer.
    // Shape [blockSize, elemsPerThread] matches thread distribution pattern.
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    auto dummySrcType = MemRefType::get({blockSize, elemsPerThread}, elemType,
                                        AffineMap{}, privateMemoryAddressSpace);
    Value dummySrc = GpuAllocOp::create(b, loc, dummySrcType);

    // Create transforms that map [tid, iter] -> linear offset in the page
    // Linear offset = tid + iter * blockSize (interleaved access for coalescing)
    TopDownTMBuilder srcBuilder(b, {"tid", "iter"},
                                {blockSize, elemsPerThread}, loc);
    srcBuilder.embed("offset", 0, pageSize, {"tid", "iter"},
                     {1, static_cast<int64_t>(blockSize)});
    Value wrappedSource =
        transform(b, dummySrc, b.getArrayAttr({srcBuilder.get()}));

    // Use ThreadwiseReadIntoOp with the already-loaded page address
    // The lowering handles:
    // 1. Using the already-loaded page base address
    // 2. Computing byte offsets from the linear index (tid + iter * blockSize)
    // 3. Indirect memory access via inttoptr/load
    // 4. Vectorization for better memory bandwidth
    ThreadwiseReadIntoOp::create(b, loc, wrappedSource, ldsBufferView,
                                 /*extraViews=*/b.getArrayAttr({}),
                                 /*extraIndices=*/ValueRange{tid},
                                 /*forceUnroll=*/true,
                                 /*useIndexDiffs=*/true,
                                 /*pageAddress=*/pageAddress);

    // NOTE: Barrier is placed by the caller between GlobalRead_Paged and
    // LDSRead_Paged stages to ensure all threads complete the page load
    // before any thread reads from LDS.
    //
    // TODO: For proper page data caching optimization, the global read should
    // be conditional (only when page changes) while the LDS read should be
    // unconditional. This requires restructuring in GridwiseGemmToBlockwise
    // to wrap the BlockwiseLoadTileOp in a conditional that checks if the
    // page index changed, similar to how page address caching works.

    return success();
  }

  LogicalResult matchAndRewrite(rock::BlockwiseLoadTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final {
    Location loc = op.getLoc();
    ValueRange indices = op.getSourceIndices();
    Value source = op.getSource();
    Value ldsByteBuffer = op.getDestLDS();
    Value destRegisters = op.getDestRegisters();

    // Check if this is paged attention
    Value pageAddress = op.getPageAddress();
    bool isPagedAttention = pageAddress != nullptr;

    auto features = rock::getFeatures(op);
    StringRef arch = rock::getArchValue(op);
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    uint32_t blockSize = op.getBlockSize();

    BlockwiseMatrixParamsAttr matrixParamsA = op.getMatrixParamsA();
    BlockwiseMatrixParamsAttr matrixParamsB = op.getMatrixParamsB();
    BlockwiseMatrixParamsAttr matrixParams =
        op.getIsA() ? matrixParamsA : matrixParamsB;
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
      // accelDDim = mnPerXdl (for MFMA instructions with blocksMfma=1)
      // accelKDim = accelKDim from BlockwiseMatrixParamsAttr
      int64_t accelDDim = tuningParams.getMnPerXdl();
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

    // Create the stages for the blockwise load tile op
    if (isa<LoopLikeOpInterface>(parentOp))
      b.setInsertionPoint(op);

    // For paged attention, create a separate stage at the current op's position.
    // This is necessary because:
    // 1. The paged op's operands (neededPageIdx, etc.) are defined after any
    //    previous blockwise_load_tile ops, so they don't dominate a shared stage
    // 2. We still need a stage to satisfy pipeline pass requirements
    if (isPagedAttention) {
      // Create a new stage at current position for paged global read
      auto stagePagedGlobalRead =
          StageOp::create(b, loc, "GlobalRead_Paged");
      stagePagedGlobalRead.getRegion().emplaceBlock();
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stagePagedGlobalRead.getRegion().back());
        auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());
        if (failed(emitPagedGlobalRead(op, b, loc, tid, blockSize)))
          return failure();
        rock::YieldOp::create(b, loc);
      }
    } else {
      auto [stageGlobalRead, stageGlobalReadNew] =
          createOrGetStage(b, loc, "GlobalRead", parentOp);
      {
        PatternRewriter::InsertionGuard guard(b);
        Block &globalReadBlock = stageGlobalRead.getRegion().back();
        // For new stages, insert at start. For existing stages, insert before
        // terminator to maintain dominance.
        if (stageGlobalReadNew || globalReadBlock.empty())
          b.setInsertionPointToStart(&globalReadBlock);
        else
          b.setInsertionPoint(globalReadBlock.getTerminator());
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

        Value wrappedSource = transform(b, source, maybeBufferViews->gridSubTile);

        ThreadwiseReadIntoOp::create(b, loc, vectorOfBoolShapedLike(loadBuffer),
                                     wrappedSource, loadBuffer,
                                     /*dynamicValidities=*/ValueRange{},
                                     /*extraViews=*/b.getArrayAttr({}),
                                     /*extraIndices=*/indices, forceUnroll, true,
                                     /*ldsTransposeConfig=*/nullptr);

        if (rock::isGlobalPrefetchSupported(arch)) {
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
    }

    if (loadType == GemmLoadTileType::BypassLDS) {
      auto [stageRegTranspose, stageRegTransposeNew] =
          createOrGetStage(b, loc, "RegTranspose", parentOp);
      {
        PatternRewriter::InsertionGuard guard(b);
        Block &regTransposeBlock = stageRegTranspose.getRegion().back();
        if (stageRegTransposeNew || regTransposeBlock.empty())
          b.setInsertionPointToStart(&regTransposeBlock);
        else
          b.setInsertionPoint(regTransposeBlock.getTerminator());

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

          FailureOr<RegsAsMatrixSubTiles> maybeBufferViews =
              accelEmitterPtr->createAccelGemmOperandTransforms(
                  b, loc, kIters, bidGridLengths, blockSize,
                  vecDimInfo.inDPerThread, dName, isKContiguousDim, false);
          if (failed(maybeBufferViews))
            return failure();
          // InBufferViews provide --> K x D subtile views.
          // Since we are iterating on D dimension, we need to transpose it.
          RegsAsMatrixSubTiles inBufferViewsTr =
              transposeSubTileViews(b, loc, maybeBufferViews.value());
          Value viewLoadedBuffer = transform(
              b, loadBuffer,
              invertTransforms(b, loc, inBufferViewsTr.threadSubTile));
          ThreadwiseReadIntoOp::create(b, loc, viewLoadedBuffer, subview,
                                       b.getArrayAttr({}), ValueRange{di},
                                       forceUnroll, true);
        }

        if (stageRegTransposeNew)
          rock::YieldOp::create(b, loc);
      }
    } else {
      if (!directToLDS && !isPagedAttention) {
        auto [stageLDSWrite, stageLDSWriteNew] =
            createOrGetStage(b, loc, "LDSWrite", parentOp);
        {
          PatternRewriter::InsertionGuard guard(b);
          Block &ldsWriteBlock = stageLDSWrite.getRegion().back();
          if (stageLDSWriteNew || ldsWriteBlock.empty())
            b.setInsertionPointToStart(&ldsWriteBlock);
          else
            b.setInsertionPoint(ldsWriteBlock.getTerminator());

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
          ArrayAttr loadBufferViews =
              invertTransforms(b, loc, maybeBufferViews->threadSubTile);
          Value viewLoadBuffer = transform(b, loadBuffer, loadBufferViews);

          FailureOr<RegsAsMatrixSubTiles> maybeLdsStoreViews =
              getPackedRegsAsTileViews(
                  b, loc, source, dName, bidGridOrder, bidGridLengths,
                  blockSize, kPerBlock, dPerBlock, vecDimInfo.inKPerThread,
                  vecDimInfo.inDPerThread, kpack, isKContiguousDim,
                  ldsLayoutConfig.doSwapThreadIterSubDims);
          if (failed(maybeLdsStoreViews))
            return failure();

          ArrayAttr storeBufferViews =
              invertTransforms(b, loc, maybeLdsStoreViews->threadSubTile);
          Value viewStoreBuffer = transform(b, storeBuffer, storeBufferViews);

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

      // LDSRead stage: needed for doubleBuffer OR paged attention
      // For paged attention, we've loaded page data to LDS in GlobalRead,
      // now we use generateReadLoop to read tile data from LDS to registers
      if (isPagedAttention) {
        // Barrier between GlobalRead_Paged and LDSRead_Paged stages.
        // This ensures all threads complete the page load before any thread
        // reads from LDS. The RockPipelinePass will handle this barrier
        // appropriately when pipelining is enabled.
        LDSBarrierOp::create(b, loc);

        // Create a new stage at current position for paged LDS read.
        // This avoids dominance issues with tileOffsetInPage operand while
        // still satisfying pipeline pass requirements.
        auto stagePagedLDSRead = StageOp::create(b, loc, "LDSRead_Paged");
        stagePagedLDSRead.getRegion().emplaceBlock();
        {
          PatternRewriter::InsertionGuard guard(b);
          b.setInsertionPointToStart(&stagePagedLDSRead.getRegion().back());

          auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

          Value tileOffsetInPage = op.getTileOffsetInPage();

          // View the LDS buffer with the appropriate type for GEMM reads.
          // For paged attention, we always use Default loadType (not directToLDS),
          // so we need kpack-packed view for the accelerator.
          // NOTE: The page data is stored as scalars, but kpack is 1 for paged
          // attention since we're loading contiguous page data. If kpack > 1
          // is needed in the future, the global read should store packed too.
          Type ldsReadType = vectorTypeOrSelf(elementType, kpack);
          Value ldsViewForGemm = viewBufferAs(b, ldsByteBuffer, ldsReadType);

          // Create a subview at the tile offset within the page
          if (tileOffsetInPage) {
            // The page is laid out as [pageSize / kpack] kpack-vectors
            // The tile starts at tileOffsetInPage (in scalar elements)
            // We need to read dPerBlock * kPerBlock elements = tileSize/kpack vectors
            int64_t tileSize = dPerBlock * kPerBlock;

            // For kpack-packed view, the offset needs to be in terms of vectors
            // tileOffsetInPage is in elements, divide by kpack for vector offset
            Value kpackVal = b.createOrFold<arith::ConstantIndexOp>(loc, kpack);
            Value vectorOffset =
                arith::DivUIOp::create(b, loc, tileOffsetInPage, kpackVal);

            int64_t vectorTileSize = tileSize / kpack;
            SmallVector<OpFoldResult> offsets = {vectorOffset};
            SmallVector<OpFoldResult> sizes = {b.getIndexAttr(vectorTileSize)};
            SmallVector<OpFoldResult> strides = {b.getIndexAttr(1)};

            ldsViewForGemm = memref::SubViewOp::create(b, loc, ldsViewForGemm,
                                                       offsets, sizes, strides);
          }

          generateReadLoop(loc, b, accelEmitterPtr, tid, dName, ldsViewForGemm,
                           destRegisters, blockSize, forceUnroll, matrixParams,
                           transposeAttr);
          rock::YieldOp::create(b, loc);
        }
      } else if (doubleBuffer) {
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
          Block &ldsReadBlock = stageLDSRead.getRegion().back();
          if (stageLDSReadNew || ldsReadBlock.empty())
            b.setInsertionPointToStart(&ldsReadBlock);
          else
            b.setInsertionPoint(ldsReadBlock.getTerminator());

          // Get current workitem ID.
          auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

          Value ldsViewForGemm;
          if (directToLDS) {
            ldsViewForGemm = viewBufferAs(b, ldsByteBuffer, elementType);
          } else {
            Type ldsReadType = vectorTypeOrSelf(elementType, kpack);
            ldsViewForGemm = viewBufferAs(b, ldsByteBuffer, ldsReadType);
          }

          generateReadLoop(loc, b, accelEmitterPtr, tid, dName, ldsViewForGemm,
                           destRegisters, blockSize, forceUnroll, matrixParams,
                           transposeAttr);
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
