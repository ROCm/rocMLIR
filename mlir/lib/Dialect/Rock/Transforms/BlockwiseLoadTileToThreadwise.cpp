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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
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
      int64_t inDPerThread, bool rotateDWithK, bool forceUnroll,
      bool directToLDS, bool ldsLayoutDxK) const {

    // wrapLDSBufferForLoad is reading a single set of Ks into private memory
    // A/B[m/n, 0:kBasePerThread]
    Value ldsViewForLoad = accelEmitterPtr->wrapLDSBufferForLoad(
        b, loc, ldsView, blockSize, inDPerThread, dName, rotateDWithK,
        directToLDS, ldsLayoutDxK);

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
                                 /*useIndexDiffs=*/true);
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

  LogicalResult matchAndRewrite(rock::BlockwiseLoadTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final {
    Location loc = op.getLoc();
    ValueRange indices = op.getSourceIndices();
    Value source = op.getSource();
    Value ldsByteBuffer = op.getDestLDS();
    Value destRegisters = op.getDestRegisters();

    auto features = rock::getFeatures(op);
    StringRef arch = rock::getArchValue(op);
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    uint32_t blockSize = op.getBlockSize();

    int64_t G = op.getG();
    int64_t M = op.getM();
    int64_t N = op.getN();
    bool isA = op.getIsA();
    StringRef dName = isA ? "m" : "n";

    bool doRotateWithK = op.getRotateWithK();
    bool doSwapThreadIterSubDims = op.getSwapThreadIterSubDims();
    bool ldsLayoutDxK = op.getLDSLayoutDxK();
    LDSLayoutConfigDim ldsLayoutConfig{doRotateWithK, doSwapThreadIterSubDims,
                                       ldsLayoutDxK};

    Type elementTypeA = op.getElementTypeA();
    Type elementTypeB = op.getElementTypeB();
    Type elementTypeLoad =
        isA ? op.getElementTypeALoad() : op.getElementTypeBLoad();
    Type elementType = isA ? elementTypeA : elementTypeB;

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

    auto [stageGlobalRead, stageGlobalReadNew] =
        createOrGetStage(b, loc, "GlobalRead", parentOp);
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(&stageGlobalRead.getRegion().back());

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
                                   /*extraIndices=*/indices, forceUnroll, true);
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

          auto copyDPerThread = vecDimInfo.inDPerThread;
          generateReadLoop(loc, b, accelEmitterPtr, tid, dName, ldsViewForGemm,
                           destRegisters, blockSize, copyDPerThread,
                           ldsLayoutConfig.doRotateWithK, forceUnroll,
                           directToLDS, ldsLayoutConfig.ldsLayoutDxK);
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
                         arith::ArithDialect, memref::MemRefDialect>();
  target.addIllegalOp<rock::BlockwiseLoadTileOp>();
  auto func = getOperation();

  RewritePatternSet patterns(&ctx);
  patterns.add<LoweringBlockwiseLoadTileOp>(&ctx);
  if (failed(applyPartialConversion(func, target, std::move(patterns))))
    signalPassFailure();
}
