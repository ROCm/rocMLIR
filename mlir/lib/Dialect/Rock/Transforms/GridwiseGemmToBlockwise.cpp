//===- GridwiseGemmToBlockwise - MLIR Rock ops lowering passes -----===//
//
// Copyright 2020 The MLIR Authors.
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
// This pass converts rock.gridwise_gemm_accel and rock.gridwise_attention_accel
// into block- and threadwise ops
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "GridLayoutEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <optional>
#include <tuple>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGRIDWISEGEMMTOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gridwise-to-blockwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockGridwiseGemmToBlockwisePass
    : public rock::impl::RockGridwiseGemmToBlockwisePassBase<
          RockGridwiseGemmToBlockwisePass> {
  void runOnOperation() override;
};

} // end anonymous namespace

static void blockwiseGemmAccel(PatternRewriter &rewriter, Location loc,
                               Value bufferA, Value bufferB, Value matrixC,
                               Value bufferScaleA, Value bufferScaleB) {
  BlockwiseGemmAccelOp::create(rewriter, loc, bufferA, bufferB, matrixC,
                               bufferScaleA, bufferScaleB);
}

static scf::ForOp createMainLoop(PatternRewriter &rewriter, Location loc,
                                 Value end) {
  Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
  Value start = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
  scf::ForOp loopOp = scf::ForOp::create(rewriter, loc, start, end, one);
  return loopOp;
}

// This function will process a tile of gemm input into LDS (or register)
// buffer in a way it could be fed to blockwise_gemm_accel op
static void loadAndStoreGemmInputTile(PatternRewriter &rewriter, Location loc,
                                      Value in, Value kIter, Value tid,
                                      rock::layout::GridCoordinates gridCoords,
                                      Value destRegs) {
  // FailureOr<RegsAsMatrixSubTiles> maybeBufferViews;
  // if (loadType == GemmLoadTileType::BypassLDS) {
  //   maybeBufferViews = accelEmitterPtr->createAccelGemmOperandTransforms(
  //       b, loc, kIters, bidGridLengths, blockSize, vecDimInfo.inDPerThread,
  //       dName, isKContiguousDim, false);
  // } else {
  //   maybeBufferViews = getLoadRegsAsTileViews(
  //       b, loc, source, dName, bidGridOrder, bidGridLengths, blockSize,
  //       kPerBlock, dPerBlock, vecDimInfo.inKPerThread,
  //       vecDimInfo.inDPerThread, isKContiguousDim, directToLDS);
  // }
  FailureOr<RegsAsMatrixSubTiles> maybeBufferViews = getLoadRegsAsTileViews(
      rewriter, loc, in, dName, bidGridOrder, bidGridLengths, blockSize,
      kPerBlock, dPerBlock, vecDimInfo.inKPerThread, vecDimInfo.inDPerThread,
      isKContiguousDim, directToLDS);
  Value wrappedSource = transform(rewriter, in, maybeBufferViews->gridSubTile);

  // Load from global memory to LDS or register buffer.
  BlockwiseLoadTileOp::create(rewriter, loc, in, destRegs,
                              ValueRange{kIter, gridCoords.g_block,
                                         gridCoords.m_block, gridCoords.n_block,
                                         tid});
}

static Value createLDSByteBuffer(PatternRewriter &rewriter, Location loc,
                                 int64_t numElements, Type elementType) {
  int64_t ldsBlockSize = getPackedByteSize(numElements, elementType);
  auto workgroupMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getWorkgroupAddressSpace());
  auto ldsMemRefType =
      MemRefType::get({ldsBlockSize}, rewriter.getI8Type(), AffineMap{},
                      workgroupMemoryAddressSpace);
  Value ldsByteBuffer = GpuAllocOp::create(rewriter, loc, ldsMemRefType);
  return ldsByteBuffer;
}

// This fuction creates interrim register buffers to store data in once
// loaded from the LDS before accelerator intrinsics are called
static Value createRegInterrimBufferForAccel(PatternRewriter &rewriter,
                                             Location loc, Type argType,
                                             int64_t dPerBlock,
                                             int64_t kPerBlock) {
  Value array;
  SmallVector<int64_t> shape{dPerBlock, kPerBlock};
  auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getPrivateAddressSpace());

  auto arrayType =
      MemRefType::get(shape, argType, AffineMap{}, privateMemoryAddressSpace);
  array = GpuAllocOp::create(rewriter, loc, arrayType);
  return array;
}

// This function creates the accumulator register buffer
static Value createBufferForAccelGemmOut(PatternRewriter &rewriter,
                                         Location loc, int64_t mPerBlock,
                                         int64_t nPerBlock, Type accType) {
  auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getPrivateAddressSpace());
  MemRefType regCAllocType =
      MemRefType::get({mPerBlock, nPerBlock}, accType, AffineMap{},
                      /*memorySpace=*/privateMemoryAddressSpace);
  Value regCAllocOp = GpuAllocOp::create(rewriter, loc, regCAllocType);
  return regCAllocOp;
}

static void zeroAccBuffer(PatternRewriter &rewriter, Location loc,
                          Value accBuffer) {
  MemRefType accBufferType = cast<MemRefType>(accBuffer.getType());
  Value zeroConstantCOp =
      createZeroConstantOp(rewriter, loc, accBufferType.getElementType());
  FillOp::create(rewriter, loc, accBuffer, zeroConstantCOp);
}

static LogicalResult checkLDSSize(Operation *op, int64_t aBufferBytes,
                                  int64_t bBufferBytes,
                                  int64_t aBufferScaleBytes = 0,
                                  int64_t bBufferScaleBytes = 0) {
  int64_t ldsBytes =
      aBufferBytes + bBufferBytes + aBufferScaleBytes + bBufferScaleBytes;
  // Check for arch limitations exceeded
  FailureOr<StringAttr> maybeArch = getArch(op);
  if (succeeded(maybeArch)) {
    StringAttr arch = maybeArch.value();
    const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
    return success(ldsBytes <= ldsSize);
  }
  return success();
}

static LDSLayoutConfigDim getLDSLayoutConfigDim(Type elementType, int64_t kpack,
                                                const VectorDimInfo &vecDimInfo,
                                                bool directToLDS) {
  LDSLayoutConfigDim cfg;
  int64_t maxVlen = 128 / elementType.getIntOrFloatBitWidth();
  int64_t copyDPerThread = vecDimInfo.inDPerThread;
  bool isKContiguousDim = vecDimInfo.vectorDim == GemmDimension::K;
  // If kpack is less than the hardware max vector length, and we are
  // writing more contiguous kpack elements, there is a possibility to
  // vectorize that we want to preserve (i.e., we favour vectorization over
  // bank conflicts resolution)
  bool isPossibleToVectorizeD = (kpack < maxVlen && copyDPerThread > 1);
  cfg.doRotateWithK = isKContiguousDim && !isPossibleToVectorizeD;
  cfg.doSwapThreadIterSubDims = !isKContiguousDim && !isPossibleToVectorizeD;
  cfg.ldsLayoutDxK = false;

  // For direct to LDS, we can't use rotateWithK or swapThreadIterSubDims
  // because we there's no LDS write instruction.
  // Also, we use the same memory layout as the global memory layout (KxD or
  // DxK).
  if (directToLDS) {
    cfg.doRotateWithK = false;
    cfg.doSwapThreadIterSubDims = false;
    cfg.ldsLayoutDxK = isKContiguousDim;
  }
  LLVM_DEBUG(llvm::dbgs() << "rotateWithK: " << cfg.doRotateWithK << "\n"
                          << "doSwapThreadIterSubDimsForM: "
                          << cfg.doSwapThreadIterSubDims << "\n"
                          << "ldsLayoutDxK: " << cfg.ldsLayoutDxK << "\n");
  return cfg;
}

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// GridwiseGemmAccel lowering.
//===----------------------------------------------------------------------===//
struct GridwiseGemmAccelRewritePattern
    : public OpRewritePattern<GridwiseGemmAccelOp> {
  using OpRewritePattern<GridwiseGemmAccelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GridwiseGemmAccelOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain data types of inputs.
    auto elementTypeA = op.getA().getType().getElementType();
    auto maybeElementTypeALoad = getInputFusionElementType(op.getA());
    auto elementTypeALoad = failed(maybeElementTypeALoad)
                                ? elementTypeA
                                : maybeElementTypeALoad.value();

    auto elementTypeB = op.getB().getType().getElementType();
    auto maybeElementTypeBLoad = getInputFusionElementType(op.getB());
    auto elementTypeBLoad = failed(maybeElementTypeBLoad)
                                ? elementTypeB
                                : maybeElementTypeBLoad.value();
    auto destType = op.getC().getType().getElementType();
    auto scaleA = op.getScaleA();
    auto scaleB = op.getScaleB();
    bool hasScaleA = scaleA != nullptr;
    bool hasScaleB = scaleB != nullptr;
    bool isScaledGemm = hasScaleA && hasScaleB;
    auto elementTypeScaleA =
        isScaledGemm ? scaleA.getType().getElementType() : nullptr;
    auto elementTypeScaleB =
        isScaledGemm ? scaleB.getType().getElementType() : nullptr;

    // Get 'features' from the op
    auto features = rock::getFeatures(op);
    auto featuresAttr = op.getFeaturesAttr();

    // Prepare some useful constants.
    Value matA = op.getA();
    Value matB = op.getB();

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> aShape, bShape;
    aShape = op.getA().getType().getShape();
    bShape = op.getB().getType().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t K = aShape[1];
    int64_t M = aShape[2];
    int64_t N = bShape[2];

    // Obtain critical tuning parameters.
    StringRef arch = rock::getArchValue(op);
    uint32_t blockSize = op.getBlockSize();
    uint32_t gridSize = op.getGridSize();
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    int64_t kpack = tuningParams.getKpack();
    // TODO: kPerBlock, as defined in parameter selection etc,
    // is in units of kPack, not individual k. This should be changed
    // at some future point, but it'll be worked around for now.
    int64_t kpacksPerBlock = tuningParams.getKpackPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();
    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;
    bool forceUnroll = tuningParams.getForceUnroll();
    int64_t kPerBlock = kpacksPerBlock * kpack;

    if (!isValidBlockSize(blockSize, kPerBlock, mPerBlock, nPerBlock)) {
      return emitError(loc) << "Block size too large, rejecting as invalid.\n";
    }

    LLVM_DEBUG(llvm::dbgs() << "gridSize: " << gridSize << "\n"
                            << "blockSize: " << blockSize << "\n"
                            << "elementTypeALoad: " << elementTypeALoad << "\n"
                            << "elementTypeBLoad: " << elementTypeBLoad << "\n"
                            << "\n"
                            << "kPerBlock: " << kPerBlock << "\n"
                            << "mPerBlock: " << mPerBlock << "\n"
                            << "nPerBlock: " << nPerBlock << "\n");
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};

    // Get current workgroup ID.
    auto bid = WorkgroupIdOp::create(b, loc, b.getIndexType());
    // Get current workitem ID.
    // auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

    // Compute grid coordinates
    int64_t gridGroupSize = tuningParams.getGridGroupSize();
    auto gridCoords = layout::makeGroupedGridLayout(
        b, loc, bid,
        {G, mBlocks, nBlocks, rock::getNumCUValue(op),
         rock::getNumChipletsValue(op), elementTypeA, destType, gridGroupSize},
        arch);

    // wavesPerEU is needed in RockToGPU pass and OutputSwizzle for the
    // OutputSwizzle pass. We add them as func attributes.
    IntegerAttr wavesPerEUAttr =
        b.getI64IntegerAttr(tuningParams.getWavesPerEU());
    func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
    funcOp->setAttr(rock::WavesPerEUAttr::getMnemonic(), wavesPerEUAttr);

    // Obtain Accelerator-related attributes.
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();

    bool useIndexDiffs = true;

    LLVM_DEBUG(llvm::dbgs() << "M: " << M << "\n"
                            << "N: " << N << "\n"
                            << "K: " << K << "\n"
                            << "G: " << G << "\n"
                            << "mPerBlock: " << mPerBlock << "\n"
                            << "nPerBlock: " << nPerBlock << "\n"
                            << "kPerBlock: " << kPerBlock << "\n"
                            << "kpack: " << kpack << "\n"
                            << "mBlocks = M / mPerBlock: " << mBlocks << "\n"
                            << "nBlocks = N / nPerBlock: " << nBlocks << "\n"
                            << "mPerWave: " << mPerWave << "\n"
                            << "nPerWave: " << nPerWave << "\n");

    auto arrayA = createRegInterrimBufferForAccel(b, loc, elementTypeA,
                                                  mPerBlock, kPerBlock);
    auto arrayB = createRegInterrimBufferForAccel(b, loc, elementTypeB,
                                                  nPerBlock, kPerBlock);

    // TODO(roctriton): f32 if float, i32 if int
    Type accType = b.getF32Type();
    Value regCAllocOp =
        createBufferForAccelGemmOut(b, loc, mPerBlock, nPerBlock, accType);
    zeroAccBuffer(b, loc, regCAllocOp);
    Value arrayScaleA, arrayScaleB;
    if (isScaledGemm) {
      arrayScaleA = createRegInterrimBufferForAccel(b, loc, elementTypeScaleA,
                                                    mPerBlock, kPerBlock);
      arrayScaleB = createRegInterrimBufferForAccel(b, loc, elementTypeScaleB,
                                                    nPerBlock, kPerBlock);
    }

    // Emit loop.
    int64_t kIterations = K / kPerBlock;
    Value nIterations = ConstantIndexOp::create(b, loc, kIterations);

    scf::ForOp loopOp = createMainLoop(b, loc, nIterations);
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());
      Value iv = loopOp.getInductionVar();

      // Load from global memory to LDS
      loadAndStoreGemmInputTile(b, loc, matB, /*kiter=*/iv, tid, gridCoords,
                                arrayB);
      loadAndStoreGemmInputTile(b, loc, matA, /*kiter=*/iv, tid, gridCoords,
                                arrayA);
      if (isScaledGemm) {
        loadAndStoreGemmInputTile(b, loc, scaleB, /*kiter=*/iv, tid, gridCoords,
                                  arrayScaleB);
        loadAndStoreGemmInputTile(b, loc, scaleA, /*kiter=*/iv, tid, gridCoords,
                                  arrayScaleA);
      }

      // Emit blockwise GEMM. This will load data from LDS (or registers) and
      // compute the MMA at the same time
      blockwiseGemmAccel(b, loc, arrayA, arrayB, regCAllocOp,
                         /*bufferScaleA=*/arrayScaleA,
                         /*bufferScaleB=*/arrayScaleB);
    }

    FailureOr<RegsAsMatrixSubTiles> maybeIdToMatrixCMaps =
        computeOutputTransforms(b, loc, M, N, blockSize, bidGridLengths,
                                maybeVecDimInfoA->inDPerThread,
                                maybeVecDimInfoB->inDPerThread,
                                ldsLayoutConfigA.doSwapThreadIterSubDims,
                                ldsLayoutConfigB.doSwapThreadIterSubDims);
    if (failed(maybeIdToMatrixCMaps)) {
      return failure();
    }
    ArrayAttr idToMatrixCMaps = maybeIdToMatrixCMaps.value().gridSubTile;

    BlockwiseStoreTileOp::create(
        b, loc, regCAllocOp, op.getC(), idToMatrixCMaps,
        /*extraIndices=*/
        ValueRange{gridCoords.g_block, gridCoords.m_block, gridCoords.n_block,
                   tid},
        op.getStoreMethod(), forceUnroll, useIndexDiffs);
    b.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void RockGridwiseGemmToBlockwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::GridwiseGemmAccelOp, GridwiseAttentionAccelOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         memref::MemRefDialect, affine::AffineDialect,
                         vector::VectorDialect, linalg::LinalgDialect,
                         scf::SCFDialect, math::MathDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmAccelRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
