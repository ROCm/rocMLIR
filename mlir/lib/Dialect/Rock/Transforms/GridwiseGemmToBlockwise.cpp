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
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
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
#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "mlir/Dialect/Rock/utility/LdsTransposeLoad.h"
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

static void blockwiseGemmAccel(
    PatternRewriter &rewriter, Location loc, GemmLoadTileType loadTypeA,
    GemmLoadTileType loadTypeB, Value bufferA, Value bufferB, Value matrixC,
    const BlockwiseMatrixParamsAttr &matrixParamsA,
    const BlockwiseMatrixParamsAttr &matrixParamsB, Value matrixAInput,
    Value matrixBInput, Value scaleA, Value scaleB, Value bufferScaleA,
    Value bufferScaleB, GemmFeaturesAttr &features, IntegerAttr blockSize,
    const RockAccelTuningParamAttrInterface &params) {
  // only pass LDS if BlockwiseGemmAccelOp will load from LDS
  Value matrixA = nullptr;
  if (loadTypeA == GemmLoadTileType::Default ||
      loadTypeA == GemmLoadTileType::DirectToLDSDefault) {
    matrixA = matrixAInput;
    assert(matrixA != nullptr);
  }
  Value matrixB = nullptr;
  if (loadTypeB == GemmLoadTileType::Default ||
      loadTypeB == GemmLoadTileType::DirectToLDSDefault) {
    matrixB = matrixBInput;
    assert(matrixB != nullptr);
  }

  BlockwiseGemmAccelOp::create(rewriter, loc, bufferA, bufferB, matrixC,
                               matrixParamsA, matrixParamsB, matrixA, matrixB,
                               scaleA, scaleB, bufferScaleA, bufferScaleB,
                               features, blockSize, params);
}

static scf::ForOp createMainLoop(PatternRewriter &rewriter, Location loc,
                                 Value end, GemmLoadTileType loadType) {
  bool doubleBuffering = loadType == GemmLoadTileType::DoubleBuffer ||
                         loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;

  // TODO: add an heuristic to decide if the it should use scheduleV1 or V2.
  // Logic to setup buffers for blockwise_gemm_accel.
  int64_t initiationInterval = doubleBuffering ? 1 : 2;

  Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
  Value start = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
  scf::ForOp loopOp = scf::ForOp::create(rewriter, loc, start, end, one);
  loopOp->setAttr(
      PipelineAttr::getMnemonic(),
      rock::PipelineAttr::get(rewriter.getContext(), initiationInterval));
  return loopOp;
}

// This function will process a tile of gemm input into LDS (or register)
// buffer in a way it could be fed to blockwise_gemm_accel op
static void loadAndStoreGemmInputTile(
    PatternRewriter &rewriter, Location loc, Value in, Value kIter, Value tid,
    rock::layout::GridCoordinates gridCoords, Value destLDS, Value destRegs,
    GemmLoadTileType loadType, StringRef nonKDimName, uint32_t blockSize,
    Type elementType, Type elementLoadType,
    const RockAccelTuningParamAttrInterface &gemmTuningParams,
    const GemmFeaturesAttr &featuresAttr,
    const BlockwiseMatrixParamsAttr &matrixParamsA,
    const BlockwiseMatrixParamsAttr &matrixParamsB) {
  UnitAttr isA = nonKDimName == "m" ? rewriter.getUnitAttr() : nullptr;
  auto loadTypeAttr =
      GemmLoadTileTypeAttr::get(rewriter.getContext(), loadType);

  // Load from global memory to LDS or register buffer.
  BlockwiseLoadTileOp::create(
      rewriter, loc, in, destLDS, destRegs, loadTypeAttr,
      TypeAttr::get(elementType), TypeAttr::get(elementLoadType), matrixParamsA,
      matrixParamsB, isA,
      ValueRange{kIter, gridCoords.g_block, gridCoords.m_block,
                 gridCoords.n_block, tid},
      featuresAttr, rewriter.getI32IntegerAttr(blockSize), gemmTuningParams);
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
static std::pair<Value, Value>
createRegInterrimBufferForAccel(PatternRewriter &rewriter, Location loc,
                                Type argType, int64_t kBasePerThread,
                                int64_t repeats, bool directToLDS) {
  Value array;
  Value arrayForLoad;
  SmallVector<int64_t> shape{kBasePerThread};
  if (repeats > 1) {
    shape.insert(shape.begin(), repeats);
  }
  if (directToLDS) {
    int64_t length = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                     std::multiplies<>());

    array = gpuAlloc(rewriter, loc, getPackedByteSize(length, argType),
                     rewriter.getI8Type(), gpu::AddressSpace::Private);

    SmallVector<int64_t> shapeForLoad(shape);
    if (auto vectorType = dyn_cast<VectorType>(argType)) {
      assert(vectorType.hasRank() == 1 && "Expected rank 1");
      shapeForLoad.back() = vectorType.getDimSize(0) * shapeForLoad.back();
    }
    arrayForLoad = viewBufferAs(rewriter, array, getElementTypeOrSelf(argType),
                                shapeForLoad);
  } else {
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());

    auto arrayType =
        MemRefType::get(shape, argType, AffineMap{}, privateMemoryAddressSpace);
    array = GpuAllocOp::create(rewriter, loc, arrayType);
    arrayForLoad = array;
  }
  return {arrayForLoad, array};
}

// This function creates the accumulator register buffer
static Value createBufferForAccelGemmOut(Location loc,
                                         rock::accel::AccelEmitterParams params,
                                         PatternRewriter &rewriter,
                                         int64_t numBuffers = 1) {
  auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getPrivateAddressSpace());
  int64_t nResultVectors = params.nResultVectors;
  int64_t mRepeats = params.mRepeats;
  int64_t nRepeats = params.nRepeats;
  VectorType accVectorType = params.accVectorType;
  int64_t nOutputVectors = nResultVectors * mRepeats * nRepeats;
  MemRefType regCAllocType;
  if (numBuffers > 1) {
    regCAllocType = MemRefType::get({numBuffers, nOutputVectors}, accVectorType,
                                    AffineMap{},
                                    /*memorySpace=*/privateMemoryAddressSpace);
  } else {
    regCAllocType = MemRefType::get(nOutputVectors, accVectorType, AffineMap{},
                                    /*memorySpace=*/privateMemoryAddressSpace);
  }
  Value regCAllocOp = GpuAllocOp::create(rewriter, loc, regCAllocType);
  return regCAllocOp;
}

// This function creates a simple scalar reg buffer (i.e. without vectors)
static Value createBufferForGemmOut(Location loc, Type gemmOutElemType,
                                    rock::accel::AccelEmitterParams params,
                                    PatternRewriter &rewriter,
                                    int64_t numBuffers = 1) {
  auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
      gpu::GPUDialect::getPrivateAddressSpace());
  int64_t numOutputElements = params.numOutputVectorElements();
  MemRefType gemmOutScalarBufferType;
  if (numBuffers > 1) {
    gemmOutScalarBufferType = MemRefType::get(
        {numBuffers, numOutputElements}, gemmOutElemType, AffineMap{},
        /*memorySpace=*/privateMemoryAddressSpace);
  } else {
    gemmOutScalarBufferType =
        MemRefType::get({numOutputElements}, gemmOutElemType, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
  }
  Value gemmOutScalarBuffer =
      GpuAllocOp::create(rewriter, loc, gemmOutScalarBufferType);
  return gemmOutScalarBuffer;
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
  StringAttr arch = getArchValue(op);
  const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
  return success(ldsBytes <= ldsSize);
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
struct GridwiseGemmRewritePattern : public OpRewritePattern<GridwiseGemmOp> {
  using OpRewritePattern<GridwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GridwiseGemmOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain data type.
    Type elementTypeA = op.getA().getType().getElementType();
    Type elementTypeB = op.getB().getType().getElementType();
    Type destType = op.getC().getType().getElementType();

    // Prepare some useful constants.
    Value zeroConstantFloatOp = createZeroConstantOp(b, loc, destType);

    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = op.getA().getType().getShape();
    bShape = op.getB().getType().getShape();
    cShape = op.getC().getType().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t K = aShape[1];
    int64_t M = aShape[2];
    int64_t N = bShape[2];

    if (bShape[0] != G || cShape[0] != G) {
      return op.emitOpError("Mismatched G dimensions in matrix multiply;")
             << " A[0] = " << G << " b[0] = " << bShape[0]
             << " C[0] = " << cShape[0];
    }
    if (cShape[1] != M) {
      return op.emitOpError("Mismatched M dimensions in matrix multiply:")
             << " A[2] = " << M << " C[1] = " << cShape[1];
    }
    if (bShape[1] != K) {
      return op.emitOpError("Mismatched K dimensions in matrix multiply:")
             << " A[1] = " << K << " B[1] = " << bShape[1];
    }

    if (cShape[2] != N) {
      return op.emitOpError("Mismatched N dimensions in matrix multiply:")
             << " B[2] = " << N << " C[2] = " << cShape[2];
    }

    // Obtain critical tuning parameters.
    uint32_t gridSize = op.getGridSize();
    GeneralGemmParamsAttr tuningParams = op.getParams();
    int64_t kpack = tuningParams.getKpack();
    // TODO: kPerBlock, as defined in parameter selection etc,
    // is in units of kPack, not individual k. This should be changed
    // at some future point, but it'll be worked around for now.
    uint32_t blockSize = tuningParams.getBlockSize();
    int64_t kpacksPerBlock = tuningParams.getKPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();
    int64_t mPerThread = tuningParams.getMPerThread();
    int64_t nPerThread = tuningParams.getNPerThread();

    GeneralGemmBlockStructure blockStructure =
        *deriveGeneralGemmBlockStructure(blockSize);
    int64_t mThreadsPerCuwave = blockStructure.mThreadsPerCuwave;
    int64_t nThreadsPerCuwave = blockStructure.nThreadsPerCuwave;
    int64_t mCuwavesPerBlock = blockStructure.mCuwavesPerBlock;
    int64_t nCuwavesPerBlock = blockStructure.nCuwavesPerBlock;

    int64_t kPerBlock = kpacksPerBlock * kpack;

    bool useIndexDiffs = true;

    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;

    LLVM_DEBUG(llvm::dbgs() << "\ngridwise_gemm op:\n");
    LLVM_DEBUG(op.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << M << "\n"
               << "N: " << N << "\n"
               << "K: " << K << "\n"
               << "G: " << G << "\n"
               << "blockSize: " << blockSize << "\n"
               << "mPerBlock: " << mPerBlock << "\n"
               << "mBlocks = M / mPerBlock: " << mBlocks << "\n"
               << "nPerBlock: " << nPerBlock << "\n"
               << "nBlocks = N / nPerBlock: " << nBlocks << "\n"
               << "kPerBlock: " << kPerBlock << "\n"
               << "kpack: " << kpack << "\n"
               << "mPerThread: " << mPerThread << "\n"
               << "nPerThread: " << nPerThread << "\n"
               << "mThreadsPerCuwave: " << mThreadsPerCuwave << "\n"
               << "mCuwavesPerBlock: " << mCuwavesPerBlock << "\n"
               << "nThreadsPerCuwave: " << nThreadsPerCuwave << "\n"
               << "nCuwavesPerBlock: " << nCuwavesPerBlock << "\n");

    // Compute required LDS sizes.
    int64_t ldsBlockASize =
        getPackedByteSize(kpacksPerBlock * mPerBlock * kpack, elementTypeA);
    int64_t ldsBlockBSize =
        getPackedByteSize(kpacksPerBlock * nPerBlock * kpack, elementTypeB);
    LLVM_DEBUG(llvm::dbgs() << "LDS block size (in bytes):" << ldsBlockASize
                            << " " << ldsBlockBSize << "\n");
    if (failed(checkLDSSize(op, ldsBlockASize, ldsBlockBSize)))
      return op.emitOpError("requires too much LDS");

    // Allocate LDS.
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefAType =
        MemRefType::get({ldsBlockASize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsByteBufferA = GpuAllocOp::create(b, loc, ldsMemRefAType);
    auto ldsMemRefBType =
        MemRefType::get({ldsBlockBSize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsByteBufferB = GpuAllocOp::create(b, loc, ldsMemRefBType);

    // Alloc for Matrix C on registers.
    // Compute register size from attributes.

    int64_t gemmMRepeat =
        mPerBlock / (mPerThread * mThreadsPerCuwave * mCuwavesPerBlock);
    int64_t gemmNRepeat =
        nPerBlock / (nPerThread * nThreadsPerCuwave * nCuwavesPerBlock);

    LLVM_DEBUG(llvm::dbgs() << "GemmMRepeat: " << gemmMRepeat << "\n");
    LLVM_DEBUG(llvm::dbgs() << "GemmNRepeat: " << gemmNRepeat << "\n");

    int64_t threadCNumM = gemmMRepeat * mPerThread;
    int64_t threadCNumN = gemmNRepeat * nPerThread;
    int64_t threadCNumRegisters = threadCNumM * threadCNumN;
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    auto threadCRegisterMemRefType =
        MemRefType::get({threadCNumRegisters}, destType, AffineMap{},
                        privateMemoryAddressSpace);
    Value registerMatrixCAllocOp =
        GpuAllocOp::create(b, loc, threadCRegisterMemRefType);
    Value registerMatrixCViewOp = reshapeBuffer(
        b, loc, registerMatrixCAllocOp, {"m", "n"}, {threadCNumM, threadCNumN});

    // Zero init Matrix C on registers.
    FillOp::create(b, loc, registerMatrixCAllocOp, zeroConstantFloatOp);

    // Get current workgroup ID.
    auto bid = WorkgroupIdOp::create(b, loc, b.getIndexType());
    // Get current workitem ID.
    auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

    if (!isValidBlockSize(blockSize, kPerBlock, mPerBlock, nPerBlock)) {
      return emitError(loc) << "Block size too large, rejecting as invalid.\n";
    }

    int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
    int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;

    // direct to LDS not supported for non-accel GEMM
    bool directToLDS = false;

    FailureOr<VectorDimInfo> maybeVecDimInfoA =
        getVectorDim(loc, op.getA(), elementTypeA, blockSize, kPerBlock,
                     mPerBlock, kpack, directToLDS);
    if (failed(maybeVecDimInfoA)) {
      return failure();
    }
    FailureOr<VectorDimInfo> maybeVecDimInfoB =
        getVectorDim(loc, op.getB(), elementTypeB, blockSize, kPerBlock,
                     nPerBlock, kpack, directToLDS);
    if (failed(maybeVecDimInfoB)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "aCopyPerThread: " << aCopyPerThread << "\n"
               << "bCopyPerThread: " << bCopyPerThread << "\n"
               << "aVectorDim: " << maybeVecDimInfoA->vectorDim << "\n"
               << "aVectorLen: " << maybeVecDimInfoA->vectorLen << "\n"
               << "bVectorDim: " << maybeVecDimInfoB->vectorDim << "\n"
               << "bVectorLen: " << maybeVecDimInfoB->vectorLen << "\n"
               << "vectorTiebreaker: " << maybeVecDimInfoA->vectorTiebreaker
               << "\n");
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    FailureOr<RegsAsMatrixSubTiles> maybeABufferViews = getLoadRegsAsTileViews(
        b, loc, op.getA(), "m", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, mPerBlock, maybeVecDimInfoA->inKPerThread,
        maybeVecDimInfoA->inDPerThread,
        maybeVecDimInfoA->vectorDim == GemmDimension::K, directToLDS);
    if (failed(maybeABufferViews)) {
      return failure();
    }
    Value wrappedA = transform(b, op.getA(), maybeABufferViews->gridSubTile);
    FailureOr<RegsAsMatrixSubTiles> maybeBBufferViews = getLoadRegsAsTileViews(
        b, loc, op.getB(), "n", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, nPerBlock, maybeVecDimInfoB->inKPerThread,
        maybeVecDimInfoB->inDPerThread,
        maybeVecDimInfoB->vectorDim == GemmDimension::K, directToLDS);
    if (failed(maybeBBufferViews)) {
      return failure();
    }
    Value wrappedB = transform(b, op.getB(), maybeBBufferViews->gridSubTile);

    auto makeRegs = [&](int64_t len, Type elementType) -> GpuAllocOp {
      Type allocType = MemRefType::get({len}, elementType, AffineMap{},
                                       privateMemoryAddressSpace);
      return GpuAllocOp::create(b, loc, allocType);
    };
    GpuAllocOp loadBufferA = makeRegs(aCopyPerThread, elementTypeA);
    GpuAllocOp loadBufferB = makeRegs(bCopyPerThread, elementTypeB);

    // Compute grid coordinates
    StringAttr arch = getArchValue(op);
    // always use heuristic for non-accel path
    int64_t gridGroupSize = 0;
    auto gridCoords = layout::makeGroupedGridLayout(
        b, loc, bid,
        {G, mBlocks, nBlocks, rock::getNumCUValue(op),
         rock::getNumChipletsValue(op), elementTypeA, destType, gridGroupSize},
        arch);

    Value storeBufferA = GpuAllocOp::create(b, loc, loadBufferA.getType());
    Value storeBufferB = GpuAllocOp::create(b, loc, loadBufferB.getType());

    LDSLayoutConfigDim ldsLayoutConfigA = getLDSLayoutConfigDim(
        elementTypeA, kpack, maybeVecDimInfoA.value(), directToLDS);
    LDSLayoutConfigDim ldsLayoutConfigB = getLDSLayoutConfigDim(
        elementTypeB, kpack, maybeVecDimInfoB.value(), directToLDS);

    // We invert the transforms that are iter --> K x D slice of the tensor
    // so that we can view loadBuffer as a K x D tensor
    FailureOr<ArrayAttr> maybeLoadBufferAViews =
        invertTransforms(b, loc, maybeABufferViews->threadSubTile);
    if (failed(maybeLoadBufferAViews)) {
      return op.emitError("cannot invert maybeABufferViews->threadSubTile");
    }
    Value viewLoadBufferA =
        transform(b, loadBufferA, maybeLoadBufferAViews.value());
    // Prior to LDS store, we need re-arrange register buffer to maxmize LDS
    // vectorization Hence, creating the view w.r.t global that correspond to
    // such re-arranged register buffer
    FailureOr<RegsAsMatrixSubTiles> maybeALdsStoreViews =
        getPackedRegsAsTileViews(
            b, loc, op.getA(), "m", bidGridOrder, bidGridLengths, blockSize,
            kPerBlock, mPerBlock, maybeVecDimInfoA->inKPerThread,
            maybeVecDimInfoA->inDPerThread, kpack,
            maybeVecDimInfoA->vectorDim == GemmDimension::K,
            ldsLayoutConfigA.doSwapThreadIterSubDims);
    if (failed(maybeALdsStoreViews)) {
      return failure();
    }
    FailureOr<ArrayAttr> maybeStoreBufferAViews =
        invertTransforms(b, loc, maybeALdsStoreViews->threadSubTile);
    FailureOr<ArrayAttr> maybeLoadBufferBViews =
        invertTransforms(b, loc, maybeBBufferViews->threadSubTile);
    if (failed(maybeStoreBufferAViews) || failed(maybeLoadBufferBViews)) {
      return op.emitError("cannot invert store and load buffer");
    }
    Value viewStoreBufferA =
        transform(b, storeBufferA, maybeStoreBufferAViews.value());
    Value viewLoadBufferB =
        transform(b, loadBufferB, maybeLoadBufferBViews.value());
    // Prior to LDS store, we need re-arrange register buffer to maxmize LDS
    // vectorization Hence, creating the view w.r.t global that correspond to
    // such re-arranged register buffer
    FailureOr<RegsAsMatrixSubTiles> maybeBLdsStoreViews =
        getPackedRegsAsTileViews(
            b, loc, op.getB(), "n", bidGridOrder, bidGridLengths, blockSize,
            kPerBlock, nPerBlock, maybeVecDimInfoB->inKPerThread,
            maybeVecDimInfoB->inDPerThread, kpack,
            maybeVecDimInfoB->vectorDim == GemmDimension::K,
            ldsLayoutConfigB.doSwapThreadIterSubDims);
    if (failed(maybeBLdsStoreViews)) {
      return failure();
    }

    FailureOr<ArrayAttr> maybeStoreBufferBViews =
        invertTransforms(b, loc, maybeBLdsStoreViews->threadSubTile);
    if (failed(maybeStoreBufferBViews)) {
      return op.emitError("cannot invert store buffer");
    }
    Value viewStoreBufferB =
        transform(b, storeBufferB, maybeStoreBufferBViews.value());

    Type ldsReadTypeA = vectorTypeOrSelf(elementTypeA, kpack);
    FailureOr<Value> maybeWrappedLdsA = wrapLDSBufferForStore(
        b, loc, ldsByteBufferA, ldsReadTypeA, kpacksPerBlock, "m", mPerBlock,
        maybeVecDimInfoA->inKPerThread, maybeVecDimInfoA->inDPerThread,
        ldsLayoutConfigA.doRotateWithK);
    if (failed(maybeWrappedLdsA))
      return maybeWrappedLdsA;
    // This is KxD view of the flat LDS buffer
    Value wrappedLdsA = std::move(*maybeWrappedLdsA);
    // This will produce a (tid, iter) --> flat LDS view
    wrappedLdsA = transform(b, wrappedLdsA, maybeALdsStoreViews->blockSubTile);

    Type ldsReadTypeB = vectorTypeOrSelf(elementTypeB, kpack);
    FailureOr<Value> maybeWrappedLdsB = wrapLDSBufferForStore(
        b, loc, ldsByteBufferB, ldsReadTypeB, kpacksPerBlock, "n", nPerBlock,
        maybeVecDimInfoB->inKPerThread, maybeVecDimInfoB->inDPerThread,
        ldsLayoutConfigB.doRotateWithK);
    if (failed(maybeWrappedLdsB))
      return maybeWrappedLdsB;
    // This is KxD view of the flat LDS buffer
    Value wrappedLdsB = std::move(*maybeWrappedLdsB);
    // This will produce a (tid, iter) --> flat LDS view
    wrappedLdsB = transform(b, wrappedLdsB, maybeBLdsStoreViews->blockSubTile);

    // The blockwise gemm isn't set up for vector-of-kpack loads and so expects
    // a scalar kpacksPerBlock x dPerBlock x kpack x T buffer unconditionally.
    Value ldsMatrixA = viewBufferAs(b, ldsByteBufferA, elementTypeA);
    ldsMatrixA = reshapeBuffer(b, loc, ldsMatrixA, {"k", "m", "kpack"},
                               {kpacksPerBlock, mPerBlock, kpack});
    Value ldsMatrixB = viewBufferAs(b, ldsByteBufferB, elementTypeB);
    ldsMatrixB = reshapeBuffer(b, loc, ldsMatrixB, {"k", "n", "kpack"},
                               {kpacksPerBlock, nPerBlock, kpack});

    // Emit loop.
    Value nIterations = ConstantIndexOp::create(b, loc, K / kPerBlock);

    // double buffer not implemented for non-accel
    scf::ForOp loopOp =
        createMainLoop(b, loc, nIterations, GemmLoadTileType::Default);
    {
      // inside the loop.
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());

      Value iv = loopOp.getInductionVar();

      auto stage0 = StageOp::create(b, loc, "GlobalRead");
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stage0.getRegion().emplaceBlock());

        ThreadwiseReadIntoOp::create(
            b, loc, vectorOfBoolShapedLike(loadBufferA), wrappedA, loadBufferA,
            /*dynamicValidities=*/ValueRange{},
            /*extraViews=*/b.getArrayAttr({}),
            /*extraIndices=*/
            ValueRange{/*kIter=*/iv, gridCoords.g_block, gridCoords.m_block,
                       gridCoords.n_block, tid},
            true, true);
        ThreadwiseReadIntoOp::create(
            b, loc, vectorOfBoolShapedLike(loadBufferB), wrappedB, loadBufferB,
            /*dynamicValidities=*/ValueRange{},
            /*extraViews=*/b.getArrayAttr({}),
            /*extraIndices=*/
            ValueRange{/*kIter=*/iv, gridCoords.g_block, gridCoords.m_block,
                       gridCoords.n_block, tid},
            true, true);
        rock::YieldOp::create(b, loc);
      }

      auto stage1 = StageOp::create(b, loc, "LDSWrite");
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stage1.getRegion().emplaceBlock());

        ThreadwiseCopyOp::create(b, loc, viewLoadBufferA, ValueRange{},
                                 viewStoreBufferA, ValueRange{}, useIndexDiffs,
                                 true);
        ThreadwiseCopyOp::create(b, loc, viewLoadBufferB, ValueRange{},
                                 viewStoreBufferB, ValueRange{}, useIndexDiffs,
                                 true);

        ThreadwiseWriteAllOp::create(b, loc, storeBufferA, wrappedLdsA,
                                     /*extraViews=*/b.getArrayAttr({}),
                                     /*extraIndices=*/ValueRange{tid},
                                     StoreMethod::Set,
                                     /*forceUnroll=*/true,
                                     /*useIndexDiffs=*/true);
        ThreadwiseWriteAllOp::create(b, loc, storeBufferB, wrappedLdsB,
                                     /*extraViews=*/b.getArrayAttr({}),
                                     /*extraIndices=*/ValueRange{tid},
                                     StoreMethod::Set,
                                     /*forceUnroll=*/true,
                                     /*useIndexDiffs=*/true);

        rock::YieldOp::create(b, loc);
      }

      auto stage2 = StageOp::create(b, loc, "MMA");
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stage2.getRegion().emplaceBlock());

        // Emit blockwise GEMM.
        BlockwiseGemmOp::create(
            b, loc, ldsMatrixA, ldsMatrixB, registerMatrixCViewOp,
            b.getI32IntegerAttr(maybeVecDimInfoA->inDPerThread),
            b.getI32IntegerAttr(maybeVecDimInfoB->inDPerThread),
            ldsLayoutConfigA.doRotateWithK ? b.getUnitAttr() : nullptr,
            ldsLayoutConfigB.doRotateWithK ? b.getUnitAttr() : nullptr,
            op.getParamsAttr());
        rock::YieldOp::create(b, loc);
      }
    }

    SmallVector<Attribute> transformAttrs;

    // Threadwise copy from register (naive tensor) to global (generic tensor).
    TopDownTMBuilder splitMemoryCoords(
        b, {"g_block", "m_block", "n_block", "tid", "iter"},
        {gridSize, mBlocks, nBlocks, blockSize, threadCNumRegisters}, loc);
    splitMemoryCoords.passThrough({"g_block", "m_block", "n_block"});
    splitMemoryCoords.merge({"m_cuwaves", "n_cuwaves", "m_cuwave", "n_cuwave"},
                            {3, 4, 5, 6}, "tid",
                            {mCuwavesPerBlock, nCuwavesPerBlock,
                             mThreadsPerCuwave, nThreadsPerCuwave});
    splitMemoryCoords.merge({"m_repeat", "m_thread", "n_repeat", "n_thread"},
                            {7, 8, 9, 10}, "iter",
                            {gemmMRepeat, mPerThread, gemmNRepeat, nPerThread});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();
    transformAttrs.push_back(splitMemoryCoordsAttr);

    auto toMatrixC =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    toMatrixC.passThrough({"g_block", "m_block", "n_block"});
    toMatrixC.unmerge(
        "gemmBlockM", 3, {"m_repeat", "m_cuwaves", "m_cuwave", "m_thread"},
        {gemmMRepeat, mCuwavesPerBlock, mThreadsPerCuwave, mPerThread});
    toMatrixC.unmerge(
        "gemmBlockN", 4, {"n_repeat", "n_cuwaves", "n_cuwave", "n_thread"},
        {gemmNRepeat, nCuwavesPerBlock, nThreadsPerCuwave, nPerThread});

    FailureOr<TopDownTMBuilder> swapRes = swapThreadIdAndIteration(
        toMatrixC, /*mBlocks=*/bidGridLengths[1],
        /*nBlocks=*/bidGridLengths[2], maybeVecDimInfoA->inDPerThread,
        maybeVecDimInfoB->inDPerThread, mPerBlock, nPerBlock,
        ldsLayoutConfigA.doSwapThreadIterSubDims,
        ldsLayoutConfigB.doSwapThreadIterSubDims,
        /*isBlockwise=*/false, transformAttrs);
    if (failed(swapRes))
      return failure();

    Value registerC = registerMatrixCAllocOp;
    ArrayAttr idToMatrixCMaps = b.getArrayAttr(transformAttrs);
    ThreadwiseWriteAllOp::create(b, loc, registerC, op.getC(), idToMatrixCMaps,
                                 /*extraIndices=*/
                                 ValueRange{gridCoords.g_block,
                                            gridCoords.m_block,
                                            gridCoords.n_block, tid},
                                 op.getStoreMethod(),
                                 /*forceUnroll=*/true, useIndexDiffs);
    b.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// GridwiseAttentionAccel lowering.
//===----------------------------------------------------------------------===//
struct ElementwiseMultOp {
  using Float = arith::MulFOp;
  using Int = arith::MulIOp;
};

struct ElementwiseAddOp {
  using Float = arith::AddFOp;
  using Int = arith::AddIOp;
};

struct GridwiseAttentionAccelRewritePattern
    : public OpRewritePattern<GridwiseAttentionAccelOp> {
  using OpRewritePattern<GridwiseAttentionAccelOp>::OpRewritePattern;

  LogicalResult storeGemmInputTile(
      PatternRewriter &rewriter, Location loc, int64_t kpack, Value regBuffer,
      RegsAsMatrixSubTiles toLDSViews, Value storeBuffer,
      Value ldsTileByteBuffer, int64_t kpacksPerBlock, StringRef nonKDimName,
      int64_t kPerBlock, int64_t dPerBlock, int64_t copyKPerThread,
      int64_t copyDPerThread, bool forceUnroll, bool rotateDWithK) const {
    Type elemType = cast<MemRefType>(regBuffer.getType()).getElementType();
    FailureOr<ArrayAttr> maybeStoreBufferViews =
        invertTransforms(rewriter, loc, toLDSViews.threadSubTile);
    if (failed(maybeStoreBufferViews)) {
      return failure();
    }
    Value viewStoreBuffer =
        transform(rewriter, storeBuffer, maybeStoreBufferViews.value());
    // The following is fine for software pipelining optimization as it could be
    // considered "compute". In future, consider refactoring the following loop
    // to be a single reg->reg op avoid verbose IR at this level.
    ThreadwiseCopyOp::create(rewriter, loc, regBuffer, ValueRange{},
                             viewStoreBuffer, ValueRange{}, false, false);
    Type ldsReadType = vectorTypeOrSelf(elemType, kpack);
    FailureOr<Value> maybeWrappedLds = wrapLDSBufferForStore(
        rewriter, loc, ldsTileByteBuffer, ldsReadType, kpacksPerBlock,
        nonKDimName, dPerBlock, copyKPerThread, copyDPerThread, rotateDWithK);
    if (failed(maybeWrappedLds)) {
      return failure();
    }
    // This is KxD view of the flat LDS buffer
    Value wrappedLds = maybeWrappedLds.value();
    // This will produce a (tid, iter) --> flat LDS view
    wrappedLds = transform(rewriter, wrappedLds, toLDSViews.blockSubTile);
    auto tid = WorkitemIdOp::create(rewriter, loc, rewriter.getIndexType());

    ThreadwiseWriteAllOp::create(rewriter, loc, storeBuffer, wrappedLds,
                                 /*extraViews=*/rewriter.getArrayAttr({}),
                                 /*extraIndices=*/ValueRange{tid},
                                 StoreMethod::Set, forceUnroll, true);
    return success();
  }

  // This function computes exp(gemm0 - rowmax_j)
  void expSubstractMaxFromGemm0(PatternRewriter &rewriter, Location loc,
                                Value gemm0OutThreadwiseView,
                                Value gemm0OutExpThreadwiseView,
                                Value gemm0OutBufferMaxView,
                                Value maxRowBuffer) const {
    Value gemm0OutBufferMax, gemm0OutExp, gemm0Out;
    ArrayAttr gemm0OutBufferMaxTrs, gemm0OutExpTrs, gemm0OutTrs;
    std::tie(gemm0OutBufferMax, gemm0OutBufferMaxTrs, std::ignore) =
        untransform(rewriter, gemm0OutBufferMaxView);
    std::tie(gemm0OutExp, gemm0OutExpTrs, std::ignore) =
        untransform(rewriter, gemm0OutExpThreadwiseView);
    std::tie(gemm0Out, gemm0OutTrs, std::ignore) =
        untransform(rewriter, gemm0OutThreadwiseView);

    MemRefType gemm0OutViewType =
        cast<MemRefType>(gemm0OutThreadwiseView.getType());
    int64_t g0Mpt = gemm0OutViewType.getShape()[0];
    int64_t g0Npt = gemm0OutViewType.getShape()[1];

    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    auto loop = TransformingForOp::create(
        rewriter, loc,
        ArrayRef<ValueRange>{
            {zero, zero}, {zero, zero}, {zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), gemm0OutBufferMaxTrs,
                            gemm0OutExpTrs, gemm0OutTrs},
        /*bounds=*/ArrayRef<int64_t>{g0Mpt, g0Npt},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Block::BlockArgListType upperCoords = loop.getLowerCoords(0);
      Block::BlockArgListType gemm0OutBufferMaxCoords = loop.getLowerCoords(1);
      Block::BlockArgListType gemm0OutExpCoords = loop.getLowerCoords(2);
      Block::BlockArgListType gemm0OutCoords = loop.getLowerCoords(3);

      // maxRowBufferNew = max(maxRowBuffer, gemm0OutBufferMaxView[:,0])
      Type maxRowBufferElemType = getElementTypeOrSelf(maxRowBuffer.getType());
      Value ldMaxRowBuffer =
          InBoundsLoadOp::create(rewriter, loc, maxRowBufferElemType,
                                 maxRowBuffer, ValueRange{upperCoords[0]});
      Value ldgemm0OutBufferMax =
          InBoundsLoadOp::create(rewriter, loc, maxRowBufferElemType,
                                 gemm0OutBufferMax, gemm0OutBufferMaxCoords);
      // Use MaxNumFOp to avoid NaN-propagation through the row max.
      Value maxRowBufferNew = arith::MaxNumFOp::create(
          rewriter, loc, ldMaxRowBuffer, ldgemm0OutBufferMax);

      // ldGemm0OutSubMaxExp = exp(gemm0Out  -maxRowBufferNew)
      Type ldGemm0OutElemType = getElementTypeOrSelf(gemm0Out.getType());
      Value ldGemm0Out = InBoundsLoadOp::create(
          rewriter, loc, ldGemm0OutElemType, gemm0Out, gemm0OutCoords);
      Value ldGemm0OutSubMax =
          arith::SubFOp::create(rewriter, loc, ldGemm0Out, maxRowBufferNew);
      Value ldGemm0OutSubMaxExp =
          math::Exp2Op::create(rewriter, loc, ldGemm0OutSubMax);

      // Store back to gemm0Out
      InBoundsStoreOp::create(rewriter, loc, ldGemm0OutSubMaxExp, gemm0OutExp,
                              gemm0OutExpCoords);
    }
  }

  // This updates the row sum according to the following
  // formula:
  // li = exp(m_{j-1} - m_{j}) * l_{j-1} + rowsum(Pij)
  // where
  // l is the rowsum accumulator
  // m is the rowmax accmulator
  // P is exp(gemm0 - rowmax_j)
  void updateRowSum(PatternRewriter &rewriter, Location loc,
                    Value gemm0OutBufferSumView, Value gemm0OutBufferMaxView,
                    Value sumRowBuffer, Value maxRowBuffer,
                    Value expMaxDiffRowBuffer) const {
    Value gemm0OutBufferSum, gemm0OutBufferMax;
    ArrayAttr gemm0OutBufferSumTrs, gemm0OutBufferMaxTrs;
    std::tie(gemm0OutBufferMax, gemm0OutBufferMaxTrs, std::ignore) =
        untransform(rewriter, gemm0OutBufferMaxView);
    std::tie(gemm0OutBufferSum, gemm0OutBufferSumTrs, std::ignore) =
        untransform(rewriter, gemm0OutBufferSumView);

    MemRefType gemm0OutViewType =
        cast<MemRefType>(gemm0OutBufferSumView.getType());
    int64_t g0Npt = gemm0OutViewType.getShape()[0];
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    auto loop = TransformingForOp::create(
        rewriter, loc,
        ArrayRef<ValueRange>{{zero, zero}, {zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), gemm0OutBufferSumTrs,
                            gemm0OutBufferMaxTrs},
        /*bounds=*/ArrayRef<int64_t>{g0Npt, 1},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Block::BlockArgListType upperCoords = loop.getLowerCoords(0);
      Block::BlockArgListType gemm0OutBufferSumCoords = loop.getLowerCoords(1);
      Block::BlockArgListType gemm0OutBufferMaxCoords = loop.getLowerCoords(2);
      // sumRowBufferNew = exp(maxRowBuffer - maxRowBufferNew) * sumRowBuffer +
      // exp(gemm0OutBufferMaxView[:,0] - maxRowBufferNew) *
      // gemm0OutBufferSumView[:,0]
      Type sumRowBufferElemType = getElementTypeOrSelf(sumRowBuffer.getType());
      Value ldSumRowBuffer =
          InBoundsLoadOp::create(rewriter, loc, sumRowBufferElemType,
                                 sumRowBuffer, ValueRange{upperCoords[0]});
      Value ldgemm0OutBufferSum =
          InBoundsLoadOp::create(rewriter, loc, sumRowBufferElemType,
                                 gemm0OutBufferSum, gemm0OutBufferSumCoords);
      // sumRowBufferNew0 = exp(maxRowBuffer - maxRowBufferNew) * sumRowBuffer
      Type maxRowBufferElemType = getElementTypeOrSelf(maxRowBuffer.getType());
      Value ldMaxRowBuffer =
          InBoundsLoadOp::create(rewriter, loc, maxRowBufferElemType,
                                 maxRowBuffer, ValueRange{upperCoords[0]});
      Value ldgemm0OutBufferMax =
          InBoundsLoadOp::create(rewriter, loc, maxRowBufferElemType,
                                 gemm0OutBufferMax, gemm0OutBufferMaxCoords);
      // Use MaxNumFOp (not MaximumFOp) so that NaN does not propagate
      // through the max reduction. MaximumFOp would let a single NaN
      // poison the row max, causing every exp(score - max) to produce
      // NaN and corrupting the entire softmax output. With MaxNumFOp
      // only the originally-NaN elements yield NaN in exp; the final
      // result is identical because the sum reduction over exp results
      // will still include that NaN.
      Value maxRowBufferNew = arith::MaxNumFOp::create(
          rewriter, loc, ldMaxRowBuffer, ldgemm0OutBufferMax);
      Value maxRowDiff =
          arith::SubFOp::create(rewriter, loc, ldMaxRowBuffer, maxRowBufferNew);
      Value maxRowDiffExp = math::Exp2Op::create(rewriter, loc, maxRowDiff);
      InBoundsStoreOp::create(rewriter, loc, maxRowDiffExp, expMaxDiffRowBuffer,
                              ValueRange{upperCoords[0]});
      Value sumRowBufferNew = maxRowDiffExp;
      sumRowBufferNew =
          arith::MulFOp::create(rewriter, loc, sumRowBufferNew, ldSumRowBuffer);
      sumRowBufferNew = arith::AddFOp::create(rewriter, loc, sumRowBufferNew,
                                              ldgemm0OutBufferSum);
      InBoundsStoreOp::create(rewriter, loc, sumRowBufferNew, sumRowBuffer,
                              ValueRange{upperCoords[0]});
      InBoundsStoreOp::create(rewriter, loc, maxRowBufferNew, maxRowBuffer,
                              ValueRange{upperCoords[0]});
    }
  }

  // This computes LSE (log-sum-exp)
  // Note that this happens at the end of the kernel, so m and l are not running
  // sum/max anymore. They are the final values.
  // input = gemm0 output
  // x = input/log(2) -> we divide by log(2) to be able to use exp2()
  // m = max x
  // l = sum exp2(x-m)
  // We want to compute log(sum e^x), therefore we do:
  // log(l*exp2(m)) = (log2(l) + m)*log(2) -> we use exp2() for "m", because we
  // need to use the same exp function used for "l"
  void computeLse(PatternRewriter &rewriter, Location loc, Value lseBufferView,
                  Value sumRowBuffer, Value maxRowBuffer) const {
    MemRefType memrefType = cast<MemRefType>(sumRowBuffer.getType());
    assert(maxRowBuffer.getType() == sumRowBuffer.getType());

    Type inputElemType = memrefType.getElementType();

    Value lseBuffer;
    ArrayAttr lseBufferTrs;
    std::tie(lseBuffer, lseBufferTrs, std::ignore) =
        untransform(rewriter, lseBufferView);
    MemRefType lseBufferViewType = cast<MemRefType>(lseBufferView.getType());
    Type outElemType = lseBufferViewType.getElementType();
    int64_t g1Npt = lseBufferViewType.getShape()[0];
    int64_t g1Mpt = lseBufferViewType.getShape()[1];
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    Value ln2Const = createConstantFloatOp(
        rewriter, loc, outElemType, outElemType, 0.69314718f,
        outElemType.getIntOrFloatBitWidth() >= 32 ? APFloat::opOK
                                                  : APFloat::opInexact);
    auto loop = TransformingForOp::create(
        rewriter, loc, ArrayRef<ValueRange>{{zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), lseBufferTrs},
        /*bounds=*/ArrayRef<int64_t>{g1Npt, g1Mpt},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      // lower = upper because the transform is empty
      Block::BlockArgListType upperCoords = loop.getLowerCoords(0);
      Block::BlockArgListType lseBufferCoords = loop.getLowerCoords(1);

      Value ldMaxRowBuffer =
          InBoundsLoadOp::create(rewriter, loc, inputElemType, maxRowBuffer,
                                 ValueRange{upperCoords[0]});
      Value ldSumRowBuffer =
          InBoundsLoadOp::create(rewriter, loc, inputElemType, sumRowBuffer,
                                 ValueRange{upperCoords[0]});

      // convert to LSE type
      ldMaxRowBuffer =
          createTypeConversionOp(rewriter, loc, ldMaxRowBuffer, outElemType);
      ldSumRowBuffer =
          createTypeConversionOp(rewriter, loc, ldSumRowBuffer, outElemType);
      // lse_i = (log2(l_i) + m_i)*log(2)
      // Migraphx expects LSE to be log
      Value log2Li = math::Log2Op::create(rewriter, loc, ldSumRowBuffer);
      Value log2Mi = ldMaxRowBuffer;
      Value lseLog2 = arith::AddFOp::create(rewriter, loc, log2Li, log2Mi);
      Value lseOut = arith::MulFOp::create(rewriter, loc, lseLog2, ln2Const);
      InBoundsStoreOp::create(rewriter, loc, lseOut, lseBuffer,
                              lseBufferCoords);
    }
  }

  // This is the out of loop scaling of attention output
  // where its divided by the accumulated rowsum
  void scaleFinalOutput(PatternRewriter &rewriter, Location loc,
                        Value attentionOutAccBufferView,
                        Value sumRowBuffer) const {
    Value attentionOutAccBuffer;
    ArrayAttr attentionOutAccTrs;
    std::tie(attentionOutAccBuffer, attentionOutAccTrs, std::ignore) =
        untransform(rewriter, attentionOutAccBufferView);
    MemRefType attentionOutAccViewType =
        cast<MemRefType>(attentionOutAccBufferView.getType());
    Type outElemType = attentionOutAccViewType.getElementType();
    int64_t g1Npt = attentionOutAccViewType.getShape()[0];
    int64_t g1Mpt = attentionOutAccViewType.getShape()[1];
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    auto loop = TransformingForOp::create(
        rewriter, loc, ArrayRef<ValueRange>{{zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), attentionOutAccTrs},
        /*bounds=*/ArrayRef<int64_t>{g1Npt, g1Mpt},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Block::BlockArgListType upperCoords = loop.getLowerCoords(0);
      Block::BlockArgListType attentionOutAccBufferCoords =
          loop.getLowerCoords(1);
      Value ldAttentionOutAccBuffer = InBoundsLoadOp::create(
          rewriter, loc, outElemType, attentionOutAccBuffer,
          attentionOutAccBufferCoords);
      Type sumRowBufferElemType = getElementTypeOrSelf(sumRowBuffer.getType());
      Value ldSumRowBuffer =
          InBoundsLoadOp::create(rewriter, loc, sumRowBufferElemType,
                                 sumRowBuffer, ValueRange{upperCoords[0]});
      // Use arcp (allow reciprocal) fast-math flag to generate
      // v_rcp_f32 + v_mul_f32 instead of the full division sequence.
      Value stAttentionOutAccBuffer =
          arith::DivFOp::create(rewriter, loc, ldAttentionOutAccBuffer,
                                ldSumRowBuffer, arith::FastMathFlags::arcp);
      InBoundsStoreOp::create(rewriter, loc, stAttentionOutAccBuffer,
                              attentionOutAccBuffer,
                              attentionOutAccBufferCoords);
    }
  }

  // This function does the corrections to row-based tiled reductions
  // according to flash attention 2 algorithm :
  // https://arxiv.org/pdf/2205.14135.pdf
  //
  // The shapes expected by the functions:
  // gemm0OutBufferMaxView.shape = [g0.Mpt, g0.Npt]
  // gemm1OutThreadwiseView.shape = [g1.Mpt=g0.Mpt, g1.Npt]
  // attentionOutAccBuffer.shape = [g1.Mpt=g0.Mpt, g1.Npt]
  //
  // This function will do the following logic :
  //
  // maxRowBufferNew = max(maxRowBuffer, gemm0OutBufferMaxView[:,0])
  // expMaxDiff = exp(maxRowBuffer - maxRowBufferNew)
  // attentionOutAccBufferMaxScaled = if not first iter ? attentionOutAccBuffer
  // / expMaxDiff : attentionOutAccBuffer attentionOutAccBufferMaxScaled +=
  // gemm1OutThreadwiseView [STORE] attentionOutAccBuffer =
  // attentionOutAccBufferMaxScaled
  void createAttentionRowStateCorrections(PatternRewriter &rewriter,
                                          Location loc,
                                          Value gemm1OutThreadwiseView,
                                          Value attentionOutAccBufferView,
                                          Value expMaxDiffRowBuffer) const {
    Value gemm1Out, attentionOutAccBuffer;
    ArrayAttr gemm1OutTrs, attentionOutAccBufferTrs;
    std::tie(gemm1Out, gemm1OutTrs, std::ignore) =
        untransform(rewriter, gemm1OutThreadwiseView);
    std::tie(attentionOutAccBuffer, attentionOutAccBufferTrs, std::ignore) =
        untransform(rewriter, attentionOutAccBufferView);

    MemRefType attentionOutAccBufferType =
        cast<MemRefType>(attentionOutAccBufferView.getType());
    Type outElemType = attentionOutAccBufferType.getElementType();
    int64_t g1Npt = attentionOutAccBufferType.getShape()[0];
    int64_t g1Mpt = attentionOutAccBufferType.getShape()[1];

    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);

    auto loop = TransformingForOp::create(
        rewriter, loc,
        ArrayRef<ValueRange>{{zero, zero}, {zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), gemm1OutTrs,
                            attentionOutAccBufferTrs},
        /*bounds=*/ArrayRef<int64_t>{g1Npt, g1Mpt},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());

      Block::BlockArgListType upperCoords = loop.getLowerCoords(0);
      Block::BlockArgListType gemm1OutCoords = loop.getLowerCoords(1);
      Block::BlockArgListType attentionOutAccBufferCoords =
          loop.getLowerCoords(2);

      Type expMaxDiffRowBufferElemType =
          getElementTypeOrSelf(expMaxDiffRowBuffer.getType());
      Value maxRowDiffExp = InBoundsLoadOp::create(
          rewriter, loc, expMaxDiffRowBufferElemType, expMaxDiffRowBuffer,
          ValueRange{upperCoords[0]});
      Value ldAttentionOutAccBuffer = InBoundsLoadOp::create(
          rewriter, loc, outElemType, attentionOutAccBuffer,
          attentionOutAccBufferCoords);
      Value scaledldAttentionOutAccBuffer = arith::MulFOp::create(
          rewriter, loc, ldAttentionOutAccBuffer, maxRowDiffExp);

      Value ldGemm1Out = InBoundsLoadOp::create(rewriter, loc, outElemType,
                                                gemm1Out, gemm1OutCoords);
      Value stAttentionOutAccBuffer = arith::AddFOp::create(
          rewriter, loc, scaledldAttentionOutAccBuffer, ldGemm1Out);
      InBoundsStoreOp::create(rewriter, loc, stAttentionOutAccBuffer,
                              attentionOutAccBuffer,
                              attentionOutAccBufferCoords);
    }
  }

  // This function will take a view stack that has lower view as m x n.
  // Then append a view to make it : m x n --> m --> m x constDim(0, n).
  // This is used to get corresponding 0th col idx in between two matrices
  // that have same number of rows.
  ArrayAttr createNZeroBroadcastView(PatternRewriter &rewriter, Location loc,
                                     ArrayAttr subTileView,
                                     int64_t zeroNDimSize) const {
    ArrayRef<int64_t> lowerShape = getLowerShape(subTileView);
    bool hasGDim = lowerShape.size() == 3;
    SmallVector<StringRef> topNames{"m", "n"};
    int nDimIdx = 1;
    if (hasGDim) {
      topNames.insert(topNames.begin(), "g");
      nDimIdx = 2;
    }
    TopDownTMBuilder dropNTop(rewriter, topNames, lowerShape, loc);
    if (hasGDim) {
      dropNTop.passThrough("g");
    }
    dropNTop.passThrough("m");
    dropNTop.constDim("nzero", nDimIdx, 0, zeroNDimSize);
    TransformMapAttr mOnlyViewMap = dropNTop.get();
    return prependUpperViews(rewriter, subTileView,
                             rewriter.getArrayAttr({mOnlyViewMap}));
  }

  // This function will call makeNZeroSubTile on subtile views of registers
  // across grid, block and thread levels.
  RegsAsMatrixSubTiles makeNZeroSubTile(PatternRewriter &rewriter, Location loc,
                                        RegsAsMatrixSubTiles subTileViews,
                                        int64_t nLen, int64_t nPerBlock,
                                        int64_t nPerThread) const {
    RegsAsMatrixSubTiles ret;
    ret.gridSubTile =
        createNZeroBroadcastView(rewriter, loc, subTileViews.gridSubTile, nLen);
    ret.blockSubTile = createNZeroBroadcastView(
        rewriter, loc, subTileViews.blockSubTile, nPerBlock);
    ret.threadSubTile = createNZeroBroadcastView(
        rewriter, loc, subTileViews.threadSubTile, nPerThread);
    return ret;
  }

  // This function will create a grid subtile view that has the unpadded
  // coordinates if there were any padding involved in the gemm operands.
  RegsAsMatrixSubTiles unpadGridSubTileView(PatternRewriter &rewriter,
                                            Location loc,
                                            RegsAsMatrixSubTiles subtileViews,
                                            int64_t prePadDim1,
                                            int64_t prePadDim2) const {
    ArrayRef<int64_t> paddedShape = getLowerShape(subtileViews.gridSubTile);
    TopDownTMBuilder viewBuilder{
        rewriter, {"g", "paddedDim1", "paddedDim2"}, paddedShape, loc};
    viewBuilder.passThrough("g");
    // paddedShape is G x M x N
    viewBuilder.pad(
        {"paddedDim1", "paddedDim2"},
        {0, paddedShape[1] - prePadDim1, 0, paddedShape[2] - prePadDim2});
    TransformMapAttr padMap = viewBuilder.get();

    subtileViews.gridSubTile = prependUpperViews(
        rewriter, subtileViews.gridSubTile, rewriter.getArrayAttr({padMap}));
    return subtileViews;
  }

  // If padding is used in the kernel, this means the first gemm
  // will be done in a larger matrix. In typical, gemm kernels
  // the padded region in the output will just contain zeros. However,
  // attention kernel will perform softmax normalization on rows.
  // Therefore, having zeros -- zero not being the minimum representable
  // value in the element type -- going to affect all the values
  // post normalization. Therefore, this function creates a transforming
  // for loop that overwrites out of bounds values of first gemm output
  // to be negative infinity.
  void createFirstGemmNegInfPadding(
      PatternRewriter &rewriter, Location loc,
      layout::GridCoordinates gridCoords, Value gemm0OutBuffer,
      RegsAsMatrixSubTiles gemm0OutSubTileViews) const {
    MemRefType gemm0OutBufferType = cast<MemRefType>(gemm0OutBuffer.getType());
    auto negInfTyped = createConstantFloatOp(
        rewriter, loc, gemm0OutBufferType.getElementType(),
        gemm0OutBufferType.getElementType(),
        -std::numeric_limits<float>::infinity(), APFloat::opOK);
    // Get current workitem ID.
    auto tid = WorkitemIdOp::create(rewriter, loc, rewriter.getIndexType());
    int64_t elementsInThreadBuffer = gemm0OutBufferType.getNumElements();
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);

    auto loop = TransformingForOp::create(
        rewriter, loc,
        ArrayRef<ValueRange>{{gridCoords.g_block, gridCoords.m_block,
                              gridCoords.n_block, tid, zero},
                             {zero, zero, zero, zero, zero}},
        ArrayRef<Attribute>{gemm0OutSubTileViews.gridSubTile,
                            rewriter.getArrayAttr({})},
        /*bounds=*/ArrayRef<int64_t>{1, 1, 1, 1, elementsInThreadBuffer},
        /*strides=*/ArrayRef<int64_t>{1, 1, 1, 1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());

      Block::BlockArgListType upperCoords = loop.getLowerCoords(1);
      TypedValue<IntegerType> isValid = loop.getValidity(0);
      Value zeroBit = createConstantIntOp(rewriter, loc, isValid.getType(),
                                          isValid.getType(), 0);
      auto isInvalid = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, isValid, zeroBit);
      scf::IfOp ifb = scf::IfOp::create(rewriter, loc, isInvalid,
                                        /*withElseRegion=*/false);
      {
        OpBuilder thenb = ifb.getThenBodyBuilder();
        InBoundsStoreOp::create(thenb, loc, negInfTyped, gemm0OutBuffer,
                                ValueRange{upperCoords[4]});
      }
    }
  }

  enum class OutOfScopeType { KVCache, Causal, PrefixCausal, SlidingWindow };

  void setGemm0OutputOutOfScope(
      PatternRewriter &rewriter, Location loc, OutOfScopeType outOfScopeType,
      layout::GridCoordinates gridCoords, Value gemm0OutBuffer,
      RegsAsMatrixSubTiles gemm0OutSubTileViews, bool enabled, Value mLoopIV,
      Value gemm0MBlocksLastIter, Value currentSeqLen, Value prefixOffset,
      IntegerAttr numRepeatsGQA, Value slidingWindowLowerBound,
      Value firstCausalMaskIter = nullptr) const {
    if (enabled) {
      // Use a lambda to generate the masking logic.
      auto generateMaskingLogic = [&](OpBuilder &b) {
        Value constNumRepeatsGQA = nullptr;
        if (numRepeatsGQA)
          constNumRepeatsGQA = b.createOrFold<arith::ConstantIndexOp>(
              loc, numRepeatsGQA.getInt());

        MemRefType gemm0OutBufferType =
            cast<MemRefType>(gemm0OutBuffer.getType());
        auto negInfTyped = createConstantFloatOp(
            b, loc, gemm0OutBufferType.getElementType(),
            gemm0OutBufferType.getElementType(),
            -std::numeric_limits<float>::infinity(), APFloat::opOK);
        // Get current workitem ID.
        auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());
        int64_t elementsInThreadBuffer = gemm0OutBufferType.getNumElements();
        Value zero = b.createOrFold<ConstantIndexOp>(loc, 0);
        auto loop = TransformingForOp::create(
            b, loc,
            ArrayRef<ValueRange>{{gridCoords.g_block, gridCoords.m_block,
                                  gridCoords.n_block, tid, zero},
                                 {zero, zero, zero, zero, zero}},
            ArrayRef<Attribute>{gemm0OutSubTileViews.gridSubTile,
                                b.getArrayAttr({})},
            /*bounds=*/ArrayRef<int64_t>{1, 1, 1, 1, elementsInThreadBuffer},
            /*strides=*/ArrayRef<int64_t>{1, 1, 1, 1, 1},
            /*forceUnroll=*/true, /*useIndexDiffs=*/true);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(loop.getBody());

          Block::BlockArgListType lowerCoords = loop.getLowerCoords(0);
          Block::BlockArgListType upperCoords = loop.getLowerCoords(1);
          Value isInvalid;
          Value mIndex = lowerCoords[2];
          switch (outOfScopeType) {
          case OutOfScopeType::KVCache:
            assert(currentSeqLen != nullptr);
            isInvalid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                              mIndex, currentSeqLen);
            break;
          case OutOfScopeType::Causal: {
            Value nIndex = lowerCoords[1];
            if (constNumRepeatsGQA)
              nIndex = b.createOrFold<arith::DivUIOp>(loc, nIndex,
                                                      constNumRepeatsGQA);

            isInvalid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                              mIndex, nIndex);
            break;
          }
          case OutOfScopeType::PrefixCausal: {
            // Prefix causal: mask when key_pos > (query_pos + prefix_offset).
            // This is used for prefix attention where:
            // - A prefix of tokens (0..prefix_offset) is always visible
            // - Anything after the prefix, standard causal masking applies
            assert(prefixOffset != nullptr);
            Value nIndex = lowerCoords[1];
            if (constNumRepeatsGQA)
              nIndex = b.createOrFold<arith::DivUIOp>(loc, nIndex,
                                                      constNumRepeatsGQA);

            // Compute query_pos + prefix_offset
            Value threshold =
                arith::AddIOp::create(b, loc, nIndex, prefixOffset);
            isInvalid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                              mIndex, threshold);
            break;
          }
          case OutOfScopeType::SlidingWindow: {
            // Sliding window: mask when key_pos < max(0, currentSeqLen -
            // windowSize). slidingWindowLowerBound is precomputed as
            // max(0, currentSeqLen - windowSize).
            assert(slidingWindowLowerBound != nullptr);
            isInvalid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                              mIndex, slidingWindowLowerBound);
            break;
          }
          }

          scf::IfOp ifOp = scf::IfOp::create(b, loc, isInvalid,
                                             /*withElseRegion=*/false);
          {
            OpBuilder thenBody = ifOp.getThenBodyBuilder();
            InBoundsStoreOp::create(thenBody, loc, negInfTyped, gemm0OutBuffer,
                                    ValueRange{upperCoords[4]});
          }
        }
      };

      if (outOfScopeType == OutOfScopeType::KVCache) {
        // For KVCache, we only need to mask on the last iteration (the
        // boundary block where K positions may exceed currentSeqLen).
        auto isLastIteration =
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                  mLoopIV, gemm0MBlocksLastIter);
        scf::IfOp ifb = scf::IfOp::create(rewriter, loc, isLastIteration,
                                          /*withElseRegion=*/false);
        {
          OpBuilder thenb = ifb.getThenBodyBuilder();
          generateMaskingLogic(thenb);
        }
      } else if (firstCausalMaskIter) {
        // For causal / prefix-causal masking, only iterations at or beyond
        // the "diagonal" (where K positions can exceed Q positions) need
        // element-wise masking. Iterations before firstCausalMaskIter have
        // all K positions <= all Q positions, so no masking is needed.
        auto needsMasking =
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                                  mLoopIV, firstCausalMaskIter);
        scf::IfOp ifb = scf::IfOp::create(rewriter, loc, needsMasking,
                                          /*withElseRegion=*/false);
        {
          OpBuilder thenb = ifb.getThenBodyBuilder();
          generateMaskingLogic(thenb);
        }
      } else {
        // Fallback: apply masking on every iteration
        generateMaskingLogic(rewriter);
      }
    }
  }

  template <typename ElementwiseOpType>
  void postProcessFirstGemmSplat(PatternRewriter &rewriter, Location loc,
                                 layout::GridCoordinates gridCoords,
                                 Value gemm0OutBuffer,
                                 RegsAsMatrixSubTiles gemm0OutViews,
                                 TypedAttr splatVal) const {
    MemRefType bufType = cast<MemRefType>(gemm0OutBuffer.getType());
    SmallVector<AffineMap, 2> indexingMaps{
        2, rewriter.getMultiDimIdentityMap(bufType.getRank())};
    SmallVector<utils::IteratorType> iteratorTypes(
        bufType.getRank(), utils::IteratorType::parallel);
    linalg::GenericOp::create(
        rewriter, loc, ValueRange(gemm0OutBuffer), ValueRange(gemm0OutBuffer),
        indexingMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value splatScalarConst = arith::ConstantOp::create(
              nestedBuilder, loc, bufType.getElementType(), splatVal);
          Value elementwiseOp;
          if (bufType.getElementType().isIntOrIndex()) {
            elementwiseOp = ElementwiseOpType::Int::create(
                nestedBuilder, loc, args[0], splatScalarConst);
          } else {
            elementwiseOp = ElementwiseOpType::Float::create(
                nestedBuilder, loc, args[0], splatScalarConst);
          }
          linalg::YieldOp::create(nestedBuilder, nestedLoc, elementwiseOp);
        });
  }

  /// Undo GQA transforms for tensors of the fusion between first gemm and
  /// second gemm
  ArrayAttr undoGQATransforms(PatternRewriter &rewriter, Location loc,
                              GridwiseAttentionAccelOp op,
                              ArrayRef<int64_t> unpaddedShape) const {
    ArrayAttr gqaTransform = nullptr;
    if (op.getNumRepeatsGQAAttr()) {
      SmallVector<StringRef> startNames = {"gemmG", "seqLenQ", "seqLenKV"};
      int64_t numRepeats = op.getNumRepeatsGQAAttr().getInt();

      assert(unpaddedShape.size() == 3);
      int64_t gemmG = unpaddedShape[0];
      int64_t seqLenQ = unpaddedShape[1];
      int64_t seqLenKV = unpaddedShape[2];
      assert(seqLenQ % numRepeats == 0);

      // (gemmG, seqLenQ*numRepeats, seqLenKV) -> (gemmG, numRepeats, seqLenQ,
      // seqLenKV)
      rock::TopDownTMBuilder unmerge(rewriter, startNames,
                                     {gemmG, seqLenQ, seqLenKV});
      unmerge.merge({"seqLenQ", "numRepeats"}, {2, 1}, "seqLenQ",
                    {seqLenQ / numRepeats, numRepeats});
      unmerge.passThrough({"gemmG", "seqLenKV"}, {0, 3}, {"gemmG", "seqLenKV"});
      auto unmergeAttr = unmerge.get();

      // (gemmG, numRepeats, seqLenQ, seqLenKV) -> (gemmG*numRepeats, seqLenQ,
      // seqLenKV)
      auto merger = rock::TopDownTMBuilder::below(unmerge, unmergeAttr);
      merger.unmerge("gemmG", 0, {"gemmG", "numRepeats"}, {gemmG, numRepeats});
      merger.passThrough({"seqLenQ", "seqLenKV"}, {1, 2},
                         {"seqLenQ", "seqLenKV"});
      auto mergerAttr = merger.get();

      SmallVector<Attribute> transformAttrs{unmergeAttr, mergerAttr};
      gqaTransform = rewriter.getArrayAttr(transformAttrs);
    }
    return gqaTransform;
  }

  // Transform GEMM0 output buffer for splitKV > 1 to match preSoftmaxBody
  // expectations. The preSoftmaxBody was created with splitKV baked into the
  // shapes, but GEMM0 computes without splitKV. This transform expands the
  // shapes at the fusion boundary.
  static ArrayAttr
  createSplitKVTransformsForGemm0Out(OpBuilder &builder, Location loc,
                                     ArrayRef<int64_t> gemm0OutShape,
                                     int64_t splitKV) {
    if (splitKV == 1)
      return nullptr;

    // GEMM0 output is [B*H, SeqQ, SeqK]
    // Need to transform to [B*H*splitKV, SeqQ, SeqK/splitKV] for fusion
    assert(gemm0OutShape.size() == 3 && "GEMM0 output must be 3D");
    assert(gemm0OutShape[2] % splitKV == 0 &&
           "SeqK must be divisible by splitKV");

    int64_t seqK = gemm0OutShape[2];
    int64_t seqKChunk = seqK / splitKV;

    // Step 1: Unmerge seqK: [B*H, SeqQ, SeqK] -> [B*H, SeqQ, splitKV,
    // SeqK/splitKV]
    rock::BottomUpTMBuilder unmergeSeqK(builder, {"batch", "seqQ", "seqK"},
                                        gemm0OutShape, loc);
    unmergeSeqK.unmerge({"splitKV", "seqK_chunk"}, {2, 3}, "seqK",
                        {splitKV, seqKChunk});
    unmergeSeqK.passThrough({"batch", "seqQ"}, {0, 1}, {"batch", "seqQ"});
    auto unmergeSeqKAttr = unmergeSeqK.get();

    // Step 2: Merge batch+splitKV: [B*H, SeqQ, splitKV, SeqK/splitKV] ->
    // [B*H*splitKV, SeqQ, SeqK/splitKV]
    auto merge = rock::BottomUpTMBuilder::above(unmergeSeqK, unmergeSeqKAttr);
    merge.merge("batch", 0, {"batch", "splitKV"});
    merge.passThrough({"seqQ", "seqK_chunk"}, {1, 2}, {"seqQ", "seqK_chunk"});
    auto mergeAttr = merge.get();

    return builder.getArrayAttr({mergeAttr, unmergeSeqKAttr});
  }

  FailureOr<Value> postProcessFirstGemm(
      PatternRewriter &rewriter, Location loc, GridwiseAttentionAccelOp op,
      layout::GridCoordinates gridCoords, Value srcGemm0OutBuffer,
      Value destGemm0OutBuffer, RegsAsMatrixSubTiles gemm0OutViews) const {
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    int64_t linalgOpIndex = -1;
    MemRefType srcBufType = cast<MemRefType>(srcGemm0OutBuffer.getType());
    MemRefType destBufType = cast<MemRefType>(destGemm0OutBuffer.getType());
    Value prevGemm0OutBuffer = srcGemm0OutBuffer;
    ArrayAttr linalgGridSubTileMaps = gemm0OutViews.gridSubTile;

    // Get grid-level GEMM0 output shape from attention op inputs
    ArrayRef<int64_t> qShape =
        cast<MemRefType>(op.getQueries().getType()).getShape();
    ArrayRef<int64_t> kShape =
        cast<MemRefType>(op.getKeys().getType()).getShape();
    SmallVector<int64_t, 3> gridGemm0OutShape = {qShape[0], qShape[1],
                                                 kShape[1]};
    if (op.getPreSoftmaxBody().getBlocks().empty()) {
      // nothing to process
      return prevGemm0OutBuffer;
    }

    int64_t firstGemmBlockArgNum = -1;
    Block &preSoftMaxBodyBlock = op.getPreSoftmaxBody().getBlocks().front();
    WalkResult res = op.getPreSoftmaxBody().walk([&](linalg::GenericOp genOp) {
      linalgOpIndex++;
      auto tid = WorkitemIdOp::create(rewriter, loc, rewriter.getIndexType());
      SmallVector<Value> inputTileBuffers;

      // Pull non-identiy index maps to rock transforms
      LogicalResult linalgIdentityRes =
          makeLinalgGenericWithIdentityAffMaps(rewriter, genOp);
      if (failed(linalgIdentityRes)) {
        genOp.emitError(
            "Failed to make linalg generic with identity affine maps");
        return WalkResult::interrupt();
      }

      // Obtain transform stack from gemmOutput to linalg generic input.
      ArrayAttr linalgToGemmOutMaps;
      Value gemm0BasedArg =
          genOp.getInputs()[op.getFirstGemmIndices()[linalgOpIndex]];
      Value mayBeFirstGemmBlockArg;
      std::tie(mayBeFirstGemmBlockArg, linalgToGemmOutMaps, std::ignore) =
          untransform(rewriter, gemm0BasedArg);

      // If the gemm0BasedArg is a block argument, we need to get its
      // blockArgNum
      if (auto firstGemmBlockArg =
              dyn_cast<BlockArgument>(mayBeFirstGemmBlockArg)) {
        assert(firstGemmBlockArgNum == -1 &&
               "firstGemmBlockArgNum should be set only once");
        // trace it back to block input
        if (firstGemmBlockArg.getOwner() == &preSoftMaxBodyBlock) {
          firstGemmBlockArgNum = firstGemmBlockArg.getArgNumber();
        } else {
          llvm::report_fatal_error("first gemm block argument does not belong "
                                   "to block of preSoftBody\n");
        }
      }
      // The obtained transforms will be linalg generic being the upperview
      // leading to gemmOutput being the lowerview. However, we need to
      // construct
      //  the following sequence :
      //  (bid, tid, iter) > ... > [gemmOutput: k x d]
      //                         > invertTr(linalg input to gemmOutput maps)
      //                         > (linalgOtherInput to op arg maps)
      FailureOr<ArrayAttr> maybeGemmOutToLinalgMaps =
          invertTransforms(rewriter, loc, linalgToGemmOutMaps);
      if (failed(maybeGemmOutToLinalgMaps)) {
        genOp.emitError("We can't invert linalg input to gemmOutput maps");
        return WalkResult::interrupt();
      }
      ArrayAttr gemmOutToLinalgMaps = maybeGemmOutToLinalgMaps.value();
      if (!gemmOutToLinalgMaps.empty()) {
        linalgGridSubTileMaps = prependUpperViews(
            rewriter, linalgGridSubTileMaps, gemmOutToLinalgMaps);
      }

      for (auto [idx, genOpInput] : llvm::enumerate(genOp.getInputs())) {
        if (idx ==
            static_cast<unsigned long>(op.getFirstGemmIndices()[linalgOpIndex]))
          continue;

        Value otherInput;
        ArrayAttr linalgToOtherInputMaps;
        std::tie(otherInput, linalgToOtherInputMaps, std::ignore) =
            untransform(rewriter, genOpInput);

        MemRefType otherInputBufType = cast<MemRefType>(otherInput.getType());
        MemRefType tileBufType = MemRefType::get(
            srcBufType.getShape(), otherInputBufType.getElementType(),
            AffineMap{}, privateMemoryAddressSpace);
        auto tileBuffer = rock::GpuAllocOp::create(rewriter, loc, tileBufType);

        ArrayAttr gemmOutToOtherInputMaps = linalgGridSubTileMaps;
        if (!linalgToOtherInputMaps.empty()) {
          gemmOutToOtherInputMaps = prependUpperViews(
              rewriter, linalgGridSubTileMaps, linalgToOtherInputMaps);
        }
        // If other input is a block argument of the attention op fusion
        if (auto blockArg = dyn_cast<BlockArgument>(otherInput)) {
          // trace it back to block input
          if (blockArg.getOwner() == &preSoftMaxBodyBlock) {
            int64_t blockArgNum = blockArg.getArgNumber();
            // we are processing other inputs. Block Argument number shouldn't
            // be the same as gemm input to first linalg generic op
            assert(firstGemmBlockArgNum != -1 &&
                   "firstGemmBlockArgNum should be set before processing other "
                   "inputs");
            assert(blockArgNum != firstGemmBlockArgNum);

            // if the gemm index is smaller, we need to substract one from the
            // index as `getPreSoftmaxElemWiseInputs()` doesn't contain
            // gemm0 output explictly
            if (blockArgNum > firstGemmBlockArgNum) {
              --blockArgNum;
            }
            otherInput = op.getPreSoftmaxElemWiseInputs()[blockArgNum];
          } else {
            llvm::report_fatal_error("Found blockArgument that does not belong "
                                     "to block of preSoftBody\n");
          }
        }
        ThreadwiseReadIntoOp::create(
            rewriter, loc, otherInput, tileBuffer, gemmOutToOtherInputMaps,
            ValueRange{gridCoords.g_block, gridCoords.m_block,
                       gridCoords.n_block, tid},
            true, true);
        inputTileBuffers.push_back(tileBuffer);
      }
      // Insert the first gemm output buffer according to which input
      // it was to the linalg generic
      inputTileBuffers.insert(inputTileBuffers.begin() +
                                  op.getFirstGemmIndices()[linalgOpIndex],
                              prevGemm0OutBuffer);
      Type outputType = genOp.getOutputs().back().getType();
      if (outputType != destGemm0OutBuffer.getType()) {
        MemRefType genOpOutMemrefType = cast<MemRefType>(outputType);
        MemRefType outTileBufType = MemRefType::get(
            destBufType.getShape(), genOpOutMemrefType.getElementType(),
            AffineMap{}, privateMemoryAddressSpace);
        auto outTileBuffer =
            rock::GpuAllocOp::create(rewriter, loc, outTileBufType);
        inputTileBuffers.push_back(outTileBuffer);
      } else {
        // reuse the same dest buffer if types match
        inputTileBuffers.push_back(destGemm0OutBuffer);
      }
      linalg::GenericOp newLinalgOp;

      mlir::IRMapping mapper;
      for (auto [operand, tilebuffer] :
           llvm::zip(genOp->getOperands(), inputTileBuffers)) {
        mapper.map(operand, tilebuffer);
      }
      newLinalgOp = cast<linalg::GenericOp>(rewriter.clone(*genOp, mapper));
      SmallVector<AffineMap> indexingMaps;
      for (size_t i = 0; i < inputTileBuffers.size(); i++) {
        indexingMaps.push_back(rewriter.getMultiDimIdentityMap(1));
      }
      newLinalgOp.setIndexingMapsAttr(
          rewriter.getAffineMapArrayAttr(indexingMaps));
      SmallVector<Attribute, 5> iteratorTypes;
      iteratorTypes.resize(
          1, linalg::IteratorTypeAttr::get(rewriter.getContext(),
                                           utils::IteratorType::parallel));
      newLinalgOp.setIteratorTypesAttr(rewriter.getArrayAttr(iteratorTypes));
      // set previous source buffer for the next linalg generic
      prevGemm0OutBuffer = inputTileBuffers.back();
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) {
      return op.emitError("pre softmax linalg regularization failed.\n");
    }
    // if not linalg generic was found, we just return the srcBuffer
    if (linalgOpIndex == -1) {
      return srcGemm0OutBuffer;
    }
    assert(prevGemm0OutBuffer.getType() == destGemm0OutBuffer.getType() &&
           "after the regularization final output buffer type should match "
           "previously allocated fusion buffer type");
    assert(static_cast<size_t>(linalgOpIndex + 1) ==
               op.getFirstGemmIndices().size() &&
           "number of linalg generic ops and number of firstGemmIndices must "
           "match");
    return prevGemm0OutBuffer;
  }

  Value transposeAttnOperand(PatternRewriter &rewriter, Location loc,
                             TypedValue<MemRefType> operand) const {
    BottomUpTMBuilder viewBuilder(rewriter, operand.getType().getShape(), loc);
    viewBuilder.passThrough({0, 1, 2}, {0, 2, 1});
    TransformMapAttr trMap = viewBuilder.get();
    return TransformOp::create(rewriter, loc, operand, trMap);
  }

  /// Check whether the op can bypass LDS-based swizzling
  /// for the B operand of the second gemm.
  bool canBypassLDSForSecondGemm(GridwiseAttentionAccelOp op) const {
    Type elemTypeQ =
        cast<MemRefType>(op.getQueries().getType()).getElementType();
    Type elemTypeK = cast<MemRefType>(op.getKeys().getType()).getElementType();
    StringRef arch = rock::getArchValue(op);
    rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
    GemmFeatures features = archInfo.defaultFeatures;
    RockAccelTuningParamAttrInterface gemm0TuningParams = op.getParams0();
    auto accelEmitterPtrGemm0 = accel::AccelEmitter::select(
        features, elemTypeQ, elemTypeK, arch, gemm0TuningParams);
    if (auto mfmaEmitter =
            dyn_cast<accel::MfmaEmitter>(accelEmitterPtrGemm0.get())) {
      if (!mfmaEmitter->isKReduction()) {
        return false;
      }
      int64_t mWaves =
          gemm0TuningParams.getMPerBlock() / gemm0TuningParams.getMPerWave();
      if (mWaves != 1) {
        return false;
      }
      // TODO: explore if this could be relaxed
      // Right now, the way we load thins from
      // LDS for the other operand distributes
      // kPack set of values from K dim. Therefore
      // to match with the MFMA output the Kpack
      // has to match rowGroupSize if we are to
      // avoid LDS for the current operand.
      if (gemm0TuningParams.getKpack() != mfmaEmitter->getRowGroupSize()) {
        return false;
      }
      return true;
    }
    return false;
  }

  /// check whether the op can bypass LDS when loading
  /// Q tiles to accel_gemm layouts
  bool canBypassLDSForQ(GridwiseAttentionAccelOp op) const {
    ArrayRef<int64_t> qShape =
        cast<MemRefType>(op.getQueries().getType()).getShape();
    int64_t gemm0K = qShape[1];
    RockAccelTuningParamAttrInterface gemm0TuningParams = op.getParams0();
    int64_t gemm0kpack = gemm0TuningParams.getKpack();
    int64_t gemm0KpacksPerBlock = gemm0TuningParams.getKpackPerBlock();
    int64_t gemm0KPerBlock = gemm0kpack * gemm0KpacksPerBlock;
    bool enableQLDSBypass = !op.getDisableQBypassLDS();
    return enableQLDSBypass && (gemm0K == gemm0KPerBlock);
  }

  TransformMapAttr getFlatToMiterMap(PatternRewriter &rewriter, int64_t gBlocks,
                                     int64_t mIterLen, int64_t nBlocks,
                                     int64_t blockSize,
                                     int64_t numElements) const {
    TopDownTMBuilder viewBuilder(rewriter,
                                 {"g_block", "n_block", "tid", "flatiter"},
                                 {gBlocks, nBlocks, blockSize, numElements});
    viewBuilder.passThrough({"g_block", "n_block", "tid"}, {0, 2, 3},
                            {"g_block", "n_block", "tid"});
    viewBuilder.merge({"mIter", "iter"}, {1, 4}, "flatiter",
                      {mIterLen, numElements / mIterLen});
    return viewBuilder.get();
  }

  std::tuple<Value, Value, Value, Value, Value, Value, Value>
  getMLoopInfo(PatternRewriter &rewriter, Location loc,
               layout::AttnGridCoordinates gridCoordsGemm0,
               Value currentSeqLenTensor, Value prefixOffsetTensor,
               int64_t gemm0M, int64_t gemm0N, int64_t gemm0MPerBlock,
               int64_t gemm0NPerBlock, int64_t splitKV, bool isCausal,
               bool isKVCache, bool isPrefixCausal, int64_t slidingWindowSize,
               IntegerAttr numRepeatsGQA = nullptr) const {
    Value gemm0MBlocksLastIter;
    Value firstCausalMaskIter;
    Value currentSeqLen;
    Value prefixOffset;
    Value slidingWindowLowerBound;
    Value effectiveSeqLen;
    Value start, end;

    // Lambda to load a 1D tensor value (used for currentSeqLen and
    // prefixOffset)
    auto loadTensorValue = [&](Value tensor) -> Value {
      assert(tensor && "tensor must be non-null");
      Value zero = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
      // add dim 1 for thread_read_into (registers)
      ArrayRef<int64_t> inpShape =
          cast<ShapedType>(tensor.getType()).getShape();
      SmallVector<StringRef> startNames = {"gemmG"};
      rock::BottomUpTMBuilder addDim(rewriter, startNames, inpShape);
      addDim.addDim("dummy", 1, 1);
      addDim.passThrough(ArrayRef<uint32_t>{0}, ArrayRef<uint32_t>{0});
      auto addDimAttr = addDim.get();
      Value tensorAddDim =
          rock::TransformOp::create(rewriter, loc, tensor, addDimAttr);
      Type elemType = getElementTypeOrSelf(tensorAddDim.getType());

      // create registers
      auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
          gpu::GPUDialect::getPrivateAddressSpace());
      auto memrefType = MemRefType::get({1}, elemType, AffineMap{},
                                        privateMemoryAddressSpace);
      auto loadAlloc = GpuAllocOp::create(rewriter, loc, memrefType);

      // load from memory to registers
      ThreadwiseReadIntoOp::create(
          rewriter, loc, vectorOfBoolShapedLike(loadAlloc), tensorAddDim,
          loadAlloc,
          /*dynamicValidities=*/ValueRange{},
          /*extraViews=*/rewriter.getArrayAttr({}),
          /*extraIndices=*/
          ValueRange{gridCoordsGemm0.g_block}, true, true);

      // load from registers
      Value loadedValue = InBoundsLoadOp::create(rewriter, loc, elemType,
                                                 loadAlloc, ValueRange{zero});
      return rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), loadedValue);
    };

    // This is needed for KV Cache/Causal/Prefix Causal/Sliding Window masking
    if (isCausal || isKVCache || isPrefixCausal || slidingWindowSize > 0) {
      if (isKVCache) {
        currentSeqLen = loadTensorValue(currentSeqLenTensor);
        effectiveSeqLen = currentSeqLen;
      }

      // Compute sliding window lower bound: max(0, currentSeqLen - windowSize)
      if (slidingWindowSize > 0) {
        assert(currentSeqLen != nullptr &&
               "sliding window requires currentSeqLen (KV-cache)");
        Value constWindowSize = rewriter.createOrFold<arith::ConstantIndexOp>(
            loc, slidingWindowSize);
        Value zero = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
        Value lowerBound = arith::SubIOp::create(rewriter, loc, currentSeqLen,
                                                 constWindowSize);
        slidingWindowLowerBound =
            arith::MaxSIOp::create(rewriter, loc, lowerBound, zero);
      }

      if (isCausal || isPrefixCausal) {
        // Compute the last Q position in the block.
        // (nIndex + 1) * NPerBlock - 1.
        Value nIndex = gridCoordsGemm0.n_block;
        Value constGemm0NPerBlock =
            rewriter.createOrFold<arith::ConstantIndexOp>(loc, gemm0NPerBlock);
        Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
        Value nIndexPlusOne = arith::AddIOp::create(rewriter, loc, nIndex, one);
        Value nextBlockStart = arith::MulIOp::create(
            rewriter, loc, nIndexPlusOne, constGemm0NPerBlock);
        Value maxRowOfBlock =
            arith::SubIOp::create(rewriter, loc, nextBlockStart, one);
        if (numRepeatsGQA) {
          Value constNumRepeatsGQA =
              rewriter.createOrFold<arith::ConstantIndexOp>(
                  loc, numRepeatsGQA.getInt());
          maxRowOfBlock = rewriter.createOrFold<arith::DivUIOp>(
              loc, maxRowOfBlock, constNumRepeatsGQA);
        }

        if (isPrefixCausal) {
          assert(isCausal && "isPrefixCausal requires isCausal");
          // For prefix causal: effective seq len = maxRowOfBlock + offset
          // This determines how many M-blocks we need to process
          prefixOffset = loadTensorValue(prefixOffsetTensor);
          maxRowOfBlock =
              arith::AddIOp::create(rewriter, loc, maxRowOfBlock, prefixOffset);
        }

        if (effectiveSeqLen) {
          // if effectiveSeqLen is set, it means KV Cache is enabled,
          // so we need to take the minimum of currentSeqLen and maxRowOfBlock
          maxRowOfBlock = arith::MinUIOp::create(rewriter, loc, currentSeqLen,
                                                 maxRowOfBlock);
        }

        // For prefix causal, adding prefix_offset can push maxRowOfBlock beyond
        // gemm0M. Similarly, when gemm0N > gemm0M, the last query position can
        // exceed the key sequence length. In both cases, bound by gemm0M - 1.
        if (gemm0N > gemm0M || isPrefixCausal) {
          // Bound by actual K dimension (key sequence length)
          Value gemm0MMinusOne =
              rewriter.createOrFold<arith::ConstantIndexOp>(loc, gemm0M - 1);
          maxRowOfBlock = arith::MinUIOp::create(rewriter, loc, maxRowOfBlock,
                                                 gemm0MMinusOne);
        }

        effectiveSeqLen = maxRowOfBlock;
      }

      // compute end index
      Value constGemm0MPerBlock =
          rewriter.createOrFold<arith::ConstantIndexOp>(loc, gemm0MPerBlock);
      Value numerator = arith::AddIOp::create(rewriter, loc, effectiveSeqLen,
                                              constGemm0MPerBlock);
      end = rewriter.createOrFold<arith::DivUIOp>(loc, numerator,
                                                  constGemm0MPerBlock);

      // Compute the first M-loop iteration that requires causal masking.
      // Only needed for causal / prefix-causal masking; KV-cache masking
      // does not use firstCausalMaskIter.
      //
      // For a given Q block (n_block), the block covers Q positions
      // [n_block * NPerBlock, (n_block + 1) * NPerBlock). Iterations of
      // the M-loop where all K positions are <= the minimum effective Q
      // position don't need causal masking. Only iterations at or beyond
      // the "diagonal" need element-wise masking.
      //
      // Block i is fully unmasked when:
      //   (i+1) * MPerBlock - 1 <= minQEffective
      // So firstCausalMaskIter = (minQEffective + 1) / MPerBlock
      if (isCausal || isPrefixCausal) {
        Value nIndex = gridCoordsGemm0.n_block;
        Value constNPerBlock =
            rewriter.createOrFold<arith::ConstantIndexOp>(loc, gemm0NPerBlock);
        Value minQEffective =
            arith::MulIOp::create(rewriter, loc, nIndex, constNPerBlock);
        if (numRepeatsGQA) {
          Value constGQA = rewriter.createOrFold<arith::ConstantIndexOp>(
              loc, numRepeatsGQA.getInt());
          minQEffective = rewriter.createOrFold<arith::DivUIOp>(
              loc, minQEffective, constGQA);
        }
        if (isPrefixCausal) {
          minQEffective =
              arith::AddIOp::create(rewriter, loc, minQEffective, prefixOffset);
        }
        Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
        Value minQPlusOne =
            arith::AddIOp::create(rewriter, loc, minQEffective, one);
        firstCausalMaskIter = rewriter.createOrFold<arith::DivUIOp>(
            loc, minQPlusOne, constGemm0MPerBlock);
      }
      Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
      Value zero = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);

      // start index is zero unless split-kv is enabled
      start = zero;
      if (splitKV != 1) {
        // here, "end" now means number of iterations in total, we need to split
        // those iterations into split-kv blocks.
        // see runEarlyExit() for details about early exit.
        Value constSplitKV =
            rewriter.createOrFold<arith::ConstantIndexOp>(loc, splitKV);
        Value constSplitKVM1 =
            rewriter.createOrFold<arith::ConstantIndexOp>(loc, splitKV - 1);
        Value numerator =
            arith::AddIOp::create(rewriter, loc, end, constSplitKVM1);
        Value gemm0MIterations =
            rewriter.createOrFold<arith::DivUIOp>(loc, numerator, constSplitKV);

        // if split-kv is enabled, we need to compute the start and end indices.
        start = arith::MulIOp::create(
            rewriter, loc, gridCoordsGemm0.split_block, gemm0MIterations);
        Value splitPlusOne = arith::AddIOp::create(
            rewriter, loc, gridCoordsGemm0.split_block, one);
        Value endSplitKV = arith::MulIOp::create(rewriter, loc, splitPlusOne,
                                                 gemm0MIterations);
        end = arith::MinUIOp::create(rewriter, loc, end, endSplitKV);
      }

      // Adjust start for sliding window: skip M-blocks that are entirely
      // below the window. All positions in those blocks would be masked to
      // -inf anyway, so we can avoid the loads and GEMMs altogether.
      if (slidingWindowSize > 0) {
        Value slidingWindowStart = rewriter.createOrFold<arith::DivUIOp>(
            loc, slidingWindowLowerBound, constGemm0MPerBlock);
        start =
            arith::MaxSIOp::create(rewriter, loc, start, slidingWindowStart);
      }

      // compute last iteration of the block, this will be used later in
      // setGemm0OutputOutOfScope()
      gemm0MBlocksLastIter =
          rewriter.createOrFold<arith::SubIOp>(loc, end, one);
    } else if (splitKV != 1) {
      // if split-kv is enabled, we need to compute the start and end indices.
      // this is the code for the case where kv-cache and causal are not
      // enabled. the logic is easier, but note that some blocks will early
      // exit, see runEarlyExit() for details.
      Value gemm0MIterations = rewriter.createOrFold<arith::ConstantIndexOp>(
          loc, gemm0M / (gemm0MPerBlock * splitKV));
      Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
      start = arith::MulIOp::create(rewriter, loc, gridCoordsGemm0.split_block,
                                    gemm0MIterations);
      Value splitPlusOne = arith::AddIOp::create(
          rewriter, loc, gridCoordsGemm0.split_block, one);
      end =
          arith::MulIOp::create(rewriter, loc, splitPlusOne, gemm0MIterations);
    } else {
      start = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
      int64_t gemm0MBlocks = gemm0M / gemm0MPerBlock;
      end = rewriter.createOrFold<arith::ConstantIndexOp>(loc, gemm0MBlocks);
    }
    return std::make_tuple(start, end, gemm0MBlocksLastIter, currentSeqLen,
                           prefixOffset, slidingWindowLowerBound,
                           firstCausalMaskIter);
  }

  // Helper function to determine if early exit optimization is possible.
  // Early exit requires splitKV > 1 and at least one of: padding in gemm0M,
  // causal masking, or KV cache.
  static bool isEarlyExitPossible(int64_t splitKV, int64_t gemm0MPerBlock,
                                  std::optional<APInt> prePadG0M, bool isCausal,
                                  bool isKVCache) {
    // We have no work to do if (1) and (2 || 3) conditions are true:
    // 1. split-kv > 1
    // 2. there's padding in gemm0M && (at least) the last block in split-kv
    // dimension has nothing to do
    // 3. (kvcache || causal) && (end <= start)
    // - Note, causal could be set true here for prefix causal, or just
    //   regular causal.
    if (splitKV == 1)
      return false;

    bool earlyExitDueToPadding =
        prePadG0M.has_value() &&
        (prePadG0M.value().getSExtValue() >= gemm0MPerBlock);
    bool earlyExitDueToCausalOrKVCache = isCausal || isKVCache;

    return earlyExitDueToPadding || earlyExitDueToCausalOrKVCache;
  }

  // Helper function to compute the 'someWorkToDo' condition used for early
  // exit optimization.
  FailureOr<Value> computeIfWorkToDo(PatternRewriter &rewriter, Location loc,
                                     Value start, Value end, int64_t splitKV,
                                     int64_t gemm0MPerBlock,
                                     std::optional<APInt> prePadG0M,
                                     bool isCausal, bool isKVCache) const {
    if (!isEarlyExitPossible(splitKV, gemm0MPerBlock, prePadG0M, isCausal,
                             isKVCache))
      return failure();

    // Determine which condition applies to generate the appropriate runtime
    // check
    bool earlyExitDueToPadding =
        prePadG0M.has_value() &&
        (prePadG0M.value().getSExtValue() >= gemm0MPerBlock);
    bool earlyExitDueToCausalOrKVCache = isCausal || isKVCache;

    Value someWorkToDo;
    // For dynamic kernels, no need to check padding condition. start/end
    // checks can handle padding as well.
    if (earlyExitDueToCausalOrKVCache) {
      // If end is less than (or equal) start, then we can early exit the
      // split KV loop.
      someWorkToDo = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ugt, end, start);
    } else if (earlyExitDueToPadding) {
      Value constGemm0MPerBlock =
          rewriter.createOrFold<arith::ConstantIndexOp>(loc, gemm0MPerBlock);
      Value prePadMValue = rewriter.createOrFold<arith::ConstantIndexOp>(
          loc, prePadG0M.value().getSExtValue());
      Value startIteration =
          arith::MulIOp::create(rewriter, loc, start, constGemm0MPerBlock);

      // If startIteration is less than prePadMValue, then there is work to do
      someWorkToDo =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                startIteration, prePadMValue);
    }

    return someWorkToDo;
  }

  std::optional<scf::IfOp> runEarlyExit(PatternRewriter &rewriter, Location loc,
                                        Value start, Value end, int64_t splitKV,
                                        int64_t gemm0MPerBlock,
                                        std::optional<APInt> prePadG0M,
                                        bool isCausal, bool isKVCache) const {
    // Compute the work condition using the extracted helper function
    FailureOr<Value> maybeSomeWorkToDo =
        computeIfWorkToDo(rewriter, loc, start, end, splitKV, gemm0MPerBlock,
                          prePadG0M, isCausal, isKVCache);

    if (failed(maybeSomeWorkToDo))
      return std::nullopt;

    scf::IfOp ifb = scf::IfOp::create(rewriter, loc, maybeSomeWorkToDo.value(),
                                      /*withElseRegion=*/false);
    rewriter.setInsertionPointToStart(&ifb.getThenRegion().front());

    // Return the IfOp so caller can close it later (by setting insertion point
    // after it) to ensure output writes happen unconditionally
    return ifb;
  }

  LogicalResult matchAndRewrite(GridwiseAttentionAccelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef arch = rock::getArchValue(op);
    uint32_t blockSize = op.getBlockSize();
    uint32_t gridSize = op.getGridSize();

    TypedValue<MemRefType> inQ = op.getQueries();
    ArrayRef<int64_t> qShape = cast<MemRefType>(inQ.getType()).getShape();
    Type elemTypeQ = cast<MemRefType>(inQ.getType()).getElementType();
    FailureOr<Type> maybeElemTypeQLoad = getInputFusionElementType(inQ);
    Type elemTypeQLoad =
        failed(maybeElemTypeQLoad) ? elemTypeQ : maybeElemTypeQLoad.value();

    TypedValue<MemRefType> inK = op.getKeys();
    ArrayRef<int64_t> kShape = cast<MemRefType>(inK.getType()).getShape();
    Type elemTypeK = cast<MemRefType>(inK.getType()).getElementType();
    FailureOr<Type> maybeElemTypeKLoad = getInputFusionElementType(inK);
    Type elemTypeKLoad =
        failed(maybeElemTypeKLoad) ? elemTypeK : maybeElemTypeKLoad.value();

    // Get 'features' from arch
    rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
    auto features = archInfo.defaultFeatures;
    auto featuresAttr = op.getFeaturesAttr();

    TypedValue<MemRefType> inV = op.getValues();
    Type elemTypeV = inV.getType().getElementType();
    FailureOr<Type> maybeElemTypeVLoad = getInputFusionElementType(inV);
    Type elemTypeVLoad =
        failed(maybeElemTypeVLoad) ? elemTypeV : maybeElemTypeVLoad.value();

    TypedValue<MemRefType> out = op.getOut();
    Value trOut = transposeAttnOperand(rewriter, loc, out);
    ArrayRef<int64_t> outShape = cast<MemRefType>(trOut.getType()).getShape();
    Type elemTypeOut = cast<MemRefType>(trOut.getType()).getElementType();

    Value lse = op.getLse();

    TypedValue<MemRefType> currentSeqLenTensor = op.getCurrentSeqLen();
    TypedValue<MemRefType> prefixOffsetTensor = op.getPrefixOffset();
    bool isKVCache = currentSeqLenTensor != nullptr;
    bool isCausal = op.getCausal();
    bool isPrefixCausal = isCausal && prefixOffsetTensor;
    int64_t slidingWindowSize =
        static_cast<int64_t>(op.getSlidingWindowSize().value_or(0));
    int64_t splitKV = op.getSplitKV();

    // Gemm0 out is casted to be softmaxType (if null, it's casted to elemTypeV)
    Type elemTypeSoftmax = op.getSoftmaxType().value_or(elemTypeV);

    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());

    int64_t gemm0G = qShape[0];
    int64_t gemm0K = qShape[1];
    int64_t gemm0N = qShape[2];
    int64_t gemm0M = kShape[2];

    int64_t gemm1M = outShape[1];
    int64_t gemm1N = outShape[2];

    RockAccelTuningParamAttrInterface gemm0TuningParams = op.getParams0();
    RockAccelTuningParamAttrInterface gemm1TuningParams = op.getParams1();
    int64_t gemm0kpack = gemm0TuningParams.getKpack();
    int64_t gemm0KpacksPerBlock = gemm0TuningParams.getKpackPerBlock();
    int64_t gemm0MPerBlock = gemm0TuningParams.getMPerBlock();
    int64_t gemm0NPerBlock = gemm0TuningParams.getNPerBlock();
    bool forceUnroll = gemm0TuningParams.getForceUnroll();
    int64_t gemm0MBlocks = gemm0M / gemm0MPerBlock;
    int64_t gemm0NBlocks = gemm0N / gemm0NPerBlock;
    int64_t gemm1kpack = gemm1TuningParams.getKpack();

    // Compute whether early exit is possible
    bool earlyExitPossible = isEarlyExitPossible(
        splitKV, gemm0MPerBlock, op.getPrePadG0M(), isCausal, isKVCache);

    int64_t scheduleVersion = gemm0TuningParams.getScheduleVersion();
    int64_t scheduleVersionG1 = gemm1TuningParams.getScheduleVersion();
    assert(scheduleVersion == scheduleVersionG1);

    // Check if the schedule version is supported by the hardware
    SmallVector<Type> types = {elemTypeQ, elemTypeK};
    if (failed(
            isScheduleVersionSupported(scheduleVersion, archInfo, types, arch)))
      return op.emitOpError("schedule version not supported");

    std::optional<GemmLoadTileType> maybeLoadType =
        symbolizeGemmLoadTileType(scheduleVersion);
    if (!maybeLoadType.has_value())
      return op.emitOpError("schedule version value is incorrect");

    GemmLoadTileType loadType = maybeLoadType.value();
    bool directToLDS = loadType == GemmLoadTileType::DirectToLDSDefault ||
                       loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;

    auto accelEmitterPtrGemm0 = accel::AccelEmitter::select(
        features, elemTypeQ, elemTypeK, arch, gemm0TuningParams);
    if (!accelEmitterPtrGemm0)
      return op.emitOpError("Unable to emit accelerator code.");
    bool doBypassLDSSecondGemm = canBypassLDSForSecondGemm(op);
    bool doBypassLDSForQ = canBypassLDSForQ(op);
    rock::accel::AccelEmitterParams accelParamsGemm0 =
        accelEmitterPtrGemm0->getParams();
    auto accelEmitterPtrGemm1 = accel::AccelEmitter::select(
        features, elemTypeV, elemTypeV, arch, gemm1TuningParams);
    if (!accelEmitterPtrGemm1)
      return op.emitOpError("Unable to emit accelerator code.");
    rock::accel::AccelEmitterParams accelParamsGemm1 =
        accelEmitterPtrGemm1->getParams();

    // wavesPerEU is needed in RockToGPU pass and OutputSwizzle for the
    // OutputSwizzle pass. We add them as func attributes.
    assert(gemm0TuningParams.getWavesPerEU() ==
           gemm1TuningParams.getWavesPerEU());
    assert(gemm0TuningParams.getOutputSwizzle() ==
           gemm1TuningParams.getOutputSwizzle());
    IntegerAttr wavesPerEUAttr =
        rewriter.getI64IntegerAttr(gemm0TuningParams.getWavesPerEU());
    IntegerAttr outputSwizzleAttr =
        rewriter.getI64IntegerAttr(gemm0TuningParams.getOutputSwizzle());
    func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
    funcOp->setAttr(rock::WavesPerEUAttr::getMnemonic(), wavesPerEUAttr);
    funcOp->setAttr(rock::OutputSwizzleAttr::getMnemonic(), outputSwizzleAttr);

    // Get current workgroup ID.
    auto bid = WorkgroupIdOp::create(rewriter, loc, rewriter.getIndexType());
    // Get current workitem ID.
    auto tid = WorkitemIdOp::create(rewriter, loc, rewriter.getIndexType());

    // Calculate different size derivations
    int64_t gemm0KPerBlock = gemm0kpack * gemm0KpacksPerBlock;
    int64_t gemm1KPerBlock = gemm0MPerBlock;
    int64_t gemm1MPerBlock = gemm1TuningParams.getMPerBlock();
    int64_t gemm1NPerBlock = gemm1TuningParams.getNPerBlock();

    // params related to how we load Q
    bool prefetchQTile = gemm0K == gemm0KPerBlock;
    auto loadTypeQ = (prefetchQTile && doBypassLDSForQ)
                         ? GemmLoadTileType::BypassLDS
                         : loadType;
    bool directToLDSQ = loadTypeQ == GemmLoadTileType::DirectToLDSDefault ||
                        loadTypeQ == GemmLoadTileType::DirectToLDSDoubleBuffer;

    // Determine if Q loads from LDS (for LDS transpose decision)
    // Q bypasses LDS only when loadTypeQ is BypassLDS
    bool qLoadsFromLDS = loadTypeQ != GemmLoadTileType::BypassLDS;

    // Note that kPerBlock for Gemm1B is mPerBlock of Gemm0 out
    // Note that mPerBlock for Gemm1A is mPerBlock of Gemm0 out
    // Note that nPerBlock for Gemm1B is nPerBlock of Gemm0 out
    int64_t gemm1MBlocks = gemm1M / gemm1MPerBlock;
    int64_t gemm1NBlocks = gemm1N / gemm1NPerBlock;
    assert(gemm0NPerBlock % gemm0kpack == 0 &&
           "nPerBlock should be divisible by kpack");
    int64_t gemm1KpacksPerBlock = gemm1KPerBlock / gemm1kpack;
    SmallVector<int64_t, 3> gemm0BidGridLengths = {gemm0G, gemm0MBlocks,
                                                   gemm0NBlocks};
    FailureOr<VectorDimInfo> maybeVectorDimInfoQ =
        getVectorDim(loc, inQ, elemTypeQLoad, blockSize, gemm0KPerBlock,
                     gemm0NPerBlock, gemm0kpack, directToLDSQ);
    if (failed(maybeVectorDimInfoQ)) {
      return failure();
    }
    LDSLayoutConfigDim ldsLayoutCfgNG0 = getLDSLayoutConfigDim(
        elemTypeQ, gemm0kpack, maybeVectorDimInfoQ.value(), directToLDSQ);
    if (doBypassLDSForQ) {
      ldsLayoutCfgNG0.doSwapThreadIterSubDims = false;
    }
    if (op.getEnableSoftmax()) {
      // TODO: Workaround for issue
      // https://github.com/ROCm/rocMLIR-internal/issues/1802 If sumRowBuffer
      // and expMaxDiffRowBuffer are filled with doSwapThreadIterSubDims=true,
      // it does not match with the second GEMM N dimension. Find a good
      // solution to this.
      ldsLayoutCfgNG0.doSwapThreadIterSubDims = false;
    }
    FailureOr<VectorDimInfo> maybeVectorDimInfoK =
        getVectorDim(loc, inK, elemTypeKLoad, blockSize, gemm0KPerBlock,
                     gemm0MPerBlock, gemm0kpack, directToLDS);
    if (failed(maybeVectorDimInfoK)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "elemTypeQLoad: " << elemTypeQLoad << "\n"
               << "elemTypeKLoad: " << elemTypeKLoad << "\n"
               << "elemTypeVLoad: " << elemTypeVLoad << "\n"
               << "qVectorDim: " << maybeVectorDimInfoQ->vectorDim << "\n"
               << "qVectorLen: " << maybeVectorDimInfoQ->vectorLen << "\n"
               << "kVectorDim: " << maybeVectorDimInfoK->vectorDim << "\n"
               << "kVectorLen: " << maybeVectorDimInfoK->vectorLen << "\n");
    LDSLayoutConfigDim ldsLayoutCfgMG0 = getLDSLayoutConfigDim(
        elemTypeK, gemm0kpack, maybeVectorDimInfoK.value(), directToLDS);
    ldsLayoutCfgMG0.doRotateWithK = false;
    if (doBypassLDSSecondGemm) {
      ldsLayoutCfgMG0.doSwapThreadIterSubDims = false;
    }
    int64_t gemm0InMPerThread = maybeVectorDimInfoK->inDPerThread;
    int64_t gemm0InNPerThread = maybeVectorDimInfoQ->inDPerThread;
    FailureOr<RegsAsMatrixSubTiles> maybeGemm0OutSubTileViews =
        accelEmitterPtrGemm0->computeOutputTransforms(
            rewriter, loc, gemm0M, gemm0N, blockSize, gemm0BidGridLengths,
            gemm0InMPerThread, gemm0InNPerThread,
            ldsLayoutCfgMG0.doSwapThreadIterSubDims,
            ldsLayoutCfgNG0.doSwapThreadIterSubDims);
    if (failed(maybeGemm0OutSubTileViews)) {
      return failure();
    }
    auto gemm0OutSubTileViews = maybeGemm0OutSubTileViews.value();
    RegsAsMatrixSubTiles gemm0OutSubTileViewsTr =
        transposeSubTileViews(rewriter, loc, gemm0OutSubTileViews);
    int64_t gemm0MPerThread =
        getLowerShape(gemm0OutSubTileViews.threadSubTile)[0];
    int64_t gemm0NPerThread =
        getLowerShape(gemm0OutSubTileViews.threadSubTile)[1];
    int64_t gemm1InNPerThread = gemm0NPerThread;

    // Create shared buffers accross gemms and reductions
    int64_t ldsByteBufferQSize = gemm0KPerBlock * gemm0NPerBlock;
    if (doBypassLDSForQ) {
      ldsByteBufferQSize = 0;
    }
    int64_t reductionWorkspaceSize =
        (gemm0MPerBlock / gemm0MPerThread) * gemm0NPerBlock;
    int64_t gemm1LDSByteBufferBSize = gemm1KPerBlock * gemm1NPerBlock;
    if (doBypassLDSSecondGemm) {
      gemm1LDSByteBufferBSize = 0;
    }

    bool doubleBuffering =
        loadType == GemmLoadTileType::DoubleBuffer ||
        loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;

    bool doubleBufferingQ =
        loadTypeQ == GemmLoadTileType::DoubleBuffer ||
        loadTypeQ == GemmLoadTileType::DirectToLDSDoubleBuffer;

    // Note that we dont provide nRepeats because we dont want
    // nRepeats times reg buffer to be created for B of gemm0
    // because we wont be prefetching that.
    auto [preAccelRegBufferKForLoad, preAccelRegBufferK] =
        createRegInterrimBufferForAccel(
            rewriter, loc, accelParamsGemm0.argTypeA,
            accelParamsGemm0.kBasePerThread,
            doubleBuffering ? accelParamsGemm0.mRepeats : 1, directToLDS);

    auto [preAccelRegBuffersQForLoad, preAccelRegBuffersQ] =
        createRegInterrimBufferForAccel(
            rewriter, loc, accelParamsGemm0.argTypeB,
            accelParamsGemm0.kBasePerThread,
            (prefetchQTile || doubleBufferingQ) ? accelParamsGemm0.nRepeats : 1,
            directToLDSQ);
    Value accRegBufferGemm0 =
        createBufferForAccelGemmOut(loc, accelParamsGemm0, rewriter);
    // Currently, there is a working assumption that this kernel is meant
    // support fp32/fp16/bf16. This should be guaranteed by op verifiers.
    Type gemmOutElemType = elemTypeV;
    if (elemTypeQ == rewriter.getI8Type()) {
      gemmOutElemType = rewriter.getI32Type();
    }

    // Walk the preSoftmax body to determine element types:
    // - gemmOutElemType: from the first generic's gemm0-based input
    // - fusionOutElemType: from the last generic's output
    Type fusionOutElemType = elemTypeV;
    bool isFirstGeneric = true;
    op.getPreSoftmaxBody().walk([&](linalg::GenericOp genOp) {
      if (isFirstGeneric) {
        if (op.getFirstGemmIndices().empty()) {
          op.emitOpError("firstGemmIndices is empty");
          return WalkResult::interrupt();
        }

        int64_t gemm0InputIdx = op.getFirstGemmIndices()[0];
        if (gemm0InputIdx >= static_cast<int64_t>(genOp.getInputs().size())) {
          op.emitOpError("firstGemmIndices[0] out of bounds for first "
                         "linalg.generic");
          return WalkResult::interrupt();
        }

        Value gemm0Input = genOp.getInputs()[gemm0InputIdx];
        gemmOutElemType = getElementTypeOrSelf(gemm0Input.getType());
        isFirstGeneric = false;
      }

      // Keep visiting to get fusionOutElemType from the last generic's output
      fusionOutElemType =
          cast<ShapedType>(genOp.getOutputs()[0].getType()).getElementType();
      return WalkResult::advance();
    });

    Value gemm0OutBuffer = createBufferForGemmOut(loc, gemmOutElemType,
                                                  accelParamsGemm0, rewriter);
    Value softmaxInputBuffer;
    if (fusionOutElemType != elemTypeSoftmax) {
      softmaxInputBuffer = createBufferForGemmOut(loc, elemTypeSoftmax,
                                                  accelParamsGemm0, rewriter);
    }
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};

    Value fusionOutBuffer = createBufferForGemmOut(loc, fusionOutElemType,
                                                   accelParamsGemm0, rewriter);
    // Buffers for reductions and softmax input
    Value softmaxBufferMax, softmaxBufferExp, softmaxBufferSum;
    if (op.getEnableSoftmax()) {
      softmaxBufferMax = createBufferForGemmOut(loc, elemTypeSoftmax,
                                                accelParamsGemm0, rewriter);
      softmaxBufferExp = createBufferForGemmOut(loc, elemTypeSoftmax,
                                                accelParamsGemm0, rewriter);
      softmaxBufferSum = createBufferForGemmOut(loc, elemTypeSoftmax,
                                                accelParamsGemm0, rewriter);
    }
    // Buffers for gemm 1
    Value gemm1RegBufferB;
    if (elemTypeV != elemTypeSoftmax) {
      gemm1RegBufferB =
          createBufferForGemmOut(loc, elemTypeV, accelParamsGemm0, rewriter);
    }
    Value gemm0ExpOutBufferToLDS =
        createBufferForGemmOut(loc, elemTypeV, accelParamsGemm0, rewriter);

    auto [preAccelRegBufferVForLoad, preAccelRegBufferV] =
        createRegInterrimBufferForAccel(
            rewriter, loc, accelParamsGemm1.argTypeA,
            accelParamsGemm1.kBasePerThread,
            doubleBuffering ? accelParamsGemm1.mRepeats : 1, directToLDS);
    auto [preAccelRegBufferQxKForLoad, preAccelRegBufferQxK] =
        createRegInterrimBufferForAccel(
            rewriter, loc, accelParamsGemm1.argTypeB,
            accelParamsGemm1.kBasePerThread,
            doBypassLDSSecondGemm ? accelParamsGemm1.nRepeats : 1, false);

    Value accRegBufferGemm1;
    Value gemm1OutBuffer;
    if (op.getEnableSoftmax()) {
      accRegBufferGemm1 =
          createBufferForAccelGemmOut(loc, accelParamsGemm1, rewriter);
      gemm1OutBuffer = createBufferForGemmOut(loc, elemTypeSoftmax,
                                              accelParamsGemm1, rewriter);
    } else {
      accRegBufferGemm1 = createBufferForAccelGemmOut(loc, accelParamsGemm1,
                                                      rewriter, gemm1MBlocks);
      gemm1OutBuffer = createBufferForGemmOut(
          loc, elemTypeSoftmax, accelParamsGemm1, rewriter, gemm1MBlocks);
    }

    SmallVector<int64_t, 3> gemm1BidGridLengths = {gemm0G, gemm1MBlocks,
                                                   gemm1NBlocks};
    FailureOr<VectorDimInfo> maybeVectorDimInfoV =
        getVectorDim(loc, inV, elemTypeVLoad, blockSize, gemm1KPerBlock,
                     gemm1MPerBlock, gemm1kpack, directToLDS);
    if (failed(maybeVectorDimInfoV)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "vVectorDim: " << maybeVectorDimInfoV->vectorDim << "\n"
               << "vVectorLen: " << maybeVectorDimInfoV->vectorLen << "\n");
    LDSLayoutConfigDim ldsLayoutCfgMG1 = getLDSLayoutConfigDim(
        elemTypeV, gemm1kpack, maybeVectorDimInfoV.value(), directToLDS);
    int64_t gemm1InMPerThread = maybeVectorDimInfoV->inDPerThread;
    FailureOr<RegsAsMatrixSubTiles> maybeGemm1OutSubTileViews =
        accelEmitterPtrGemm1->computeOutputTransforms(
            rewriter, loc, gemm1M, gemm1N, blockSize, gemm1BidGridLengths,
            gemm1InMPerThread, gemm1InNPerThread,
            ldsLayoutCfgMG1.doSwapThreadIterSubDims);
    if (failed(maybeGemm1OutSubTileViews)) {
      return failure();
    }
    auto gemm1OutSubTileViews = maybeGemm1OutSubTileViews.value();
    RegsAsMatrixSubTiles gemm1OutSubTileViewsTr =
        transposeSubTileViews(rewriter, loc, gemm1OutSubTileViews);
    int64_t gemm1MPerThread =
        getLowerShape(gemm1OutSubTileViewsTr.threadSubTile)[0];

    // Buffers for running row state

    // o buffer; this is exactly same as gemm1OutBuffer;
    // we just need another buffer to do the special accumulation
    Value attentionOutAccBuffer, outAccBufferOutTyped, sumRowBuffer,
        maxRowBuffer, expMaxDiffRowBuffer, lseBuffer;
    FailureOr<ArrayAttr> maybeAttentionOutAccBufferThreadSubTileViewMaps =
        failure();
    if (op.getEnableSoftmax()) {
      attentionOutAccBuffer = createBufferForGemmOut(
          loc, elemTypeSoftmax, accelParamsGemm1, rewriter, gemm1MBlocks);
      outAccBufferOutTyped = attentionOutAccBuffer;
      if (elemTypeSoftmax != elemTypeOut) {
        outAccBufferOutTyped = createBufferForGemmOut(
            loc, elemTypeOut, accelParamsGemm1, rewriter, gemm1MBlocks);
      }
      maybeAttentionOutAccBufferThreadSubTileViewMaps =
          invertTransforms(rewriter, loc, gemm1OutSubTileViewsTr.threadSubTile);
      if (failed(maybeAttentionOutAccBufferThreadSubTileViewMaps)) {
        return op.emitError("cannot invert attention buffer");
      }
      // m buffer; this only contains a reduced single value per row
      auto reducedBufferType =
          MemRefType::get({gemm1MPerThread}, elemTypeSoftmax, AffineMap{},
                          /*memorySpace=*/privateMemoryAddressSpace);
      auto negInfSumTyped = createConstantFloatOp(
          rewriter, loc, reducedBufferType.getElementType(),
          reducedBufferType.getElementType(),
          -std::numeric_limits<float>::infinity(), APFloat::opOK);
      maxRowBuffer = rock::GpuAllocOp::create(rewriter, loc, reducedBufferType);
      expMaxDiffRowBuffer =
          rock::GpuAllocOp::create(rewriter, loc, reducedBufferType);
      FillOp::create(rewriter, loc, maxRowBuffer, negInfSumTyped);
      // l buffer; this only contains a reduced single value per row
      sumRowBuffer = rock::GpuAllocOp::create(rewriter, loc, reducedBufferType);
      FillOp::create(rewriter, loc, sumRowBuffer,
                     createZeroConstantOp(rewriter, loc, elemTypeSoftmax));
      if (lse) {
        Type elemTypeLse = cast<MemRefType>(lse.getType()).getElementType();
        lseBuffer = createBufferForGemmOut(loc, elemTypeLse, accelParamsGemm1,
                                           rewriter);
        // Initialize lseBuffer to -infinity only when early exit is possible.
        if (earlyExitPossible) {
          auto negInfLse = createConstantFloatOp(
              rewriter, loc, elemTypeLse, elemTypeLse,
              -std::numeric_limits<float>::infinity(), APFloat::opOK);
          FillOp::create(rewriter, loc, lseBuffer, negInfLse);
        }
      }

      // Only zero the output-typed buffer when early exit is possible and it's
      // a different type (e.g., f16 vs f32). When early exit happens, the type
      // conversion from attentionOutAccBuffer to outAccBufferOutTyped is
      // skipped, so we need outAccBufferOutTyped pre-initialized to zeros.
      if (earlyExitPossible && outAccBufferOutTyped != attentionOutAccBuffer)
        zeroAccBuffer(rewriter, loc, outAccBufferOutTyped);
      zeroAccBuffer(rewriter, loc, attentionOutAccBuffer);
    } else {
      outAccBufferOutTyped = gemm1OutBuffer;
      if (elemTypeSoftmax != elemTypeOut) {
        outAccBufferOutTyped = createBufferForGemmOut(
            loc, elemTypeOut, accelParamsGemm1, rewriter, gemm1MBlocks);
      }
      zeroAccBuffer(rewriter, loc, accRegBufferGemm1);
    }

    int64_t numChiplets = rock::getNumChipletsValue(op);
    // if splitKV == 1, we define nullptr, and makeGxNGridLayout() will use
    // fewer instructions
    Value splitKVConst =
        (splitKV > 1) ? rewriter.createOrFold<ConstantIndexOp>(loc, splitKV)
                      : nullptr;
    auto gridCoordsGemm0mIter0 = layout::makeGxNGridLayout(
        rewriter, loc, bid,
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0), gemm0NBlocks,
        gridSize, arch, numChiplets, splitKVConst);

    Value gemm0MBlocksLastIter;
    Value currentSeqLen;
    Value prefixOffset;
    Value slidingWindowLowerBound;
    Value firstCausalMaskIter;
    Value start, end;
    // get mLoop
    std::tie(start, end, gemm0MBlocksLastIter, currentSeqLen, prefixOffset,
             slidingWindowLowerBound, firstCausalMaskIter) =
        getMLoopInfo(rewriter, loc, gridCoordsGemm0mIter0, currentSeqLenTensor,
                     prefixOffsetTensor, gemm0M, gemm0N, gemm0MPerBlock,
                     gemm0NPerBlock, splitKV, isCausal, isKVCache,
                     isPrefixCausal, slidingWindowSize,
                     op.getNumRepeatsGQAAttr());

    // Early exit: Skip all computation when there's no work but always write
    // output.
    // This wraps Q loads, M/K loops, GEMMs, softmax, etc. in a conditional.
    std::optional<scf::IfOp> earlyExitIf =
        runEarlyExit(rewriter, loc, start, end, splitKV, gemm0MPerBlock,
                     op.getPrePadG0M(), isCausal, isKVCache);

    // LDS Transpose Decision for GEMM0 (K x Q)
    // Pass qLoadsFromLDS to disable LDS transpose for Q when it's prefetched
    hwtranspose::LDSTransposeDecision ldsDecisionGemm0 =
        hwtranspose::decideLDSTransposeForOperands(
            accelEmitterPtrGemm0.get(), arch, elemTypeK, elemTypeQ, directToLDS,
            ldsLayoutCfgMG0, ldsLayoutCfgNG0, gemm0MPerBlock, gemm0NPerBlock,
            gemm0KPerBlock, gemm0TuningParams.getMPerWave(),
            gemm0TuningParams.getNPerWave(), gemm0kpack,
            /*doubleBuffering=*/false, /*bLoadsFromLDS=*/qLoadsFromLDS);

    // Disable LDS transpose for large head dimensions (HeadDimQK >= 512)
    // Note: gemm0N = qShape[2] = head_dim_qk
    if (gemm0N >= 512) {
      ldsDecisionGemm0.enableA = false;
      ldsDecisionGemm0.enableB = false;
    }

    // create matrix params
    BlockwiseMatrixParamsAttr matrixParamsK = BlockwiseMatrixParamsAttr::get(
        rewriter.getContext(), elemTypeK, elemTypeKLoad,
        ldsLayoutCfgMG0.doRotateWithK, ldsLayoutCfgMG0.doSwapThreadIterSubDims,
        ldsLayoutCfgMG0.ldsLayoutDxK, directToLDS,
        /*splitKAcrossThreadsFirst=*/false, gemm0G, gemm0M, gemm0InMPerThread,
        /*ldsTransposeEnabled=*/ldsDecisionGemm0.enableA,
        /*accelDDim=*/ldsDecisionGemm0.mfmaDDim,
        /*accelKDim=*/ldsDecisionGemm0.mfmaKDim);

    BlockwiseMatrixParamsAttr matrixParamsQ = BlockwiseMatrixParamsAttr::get(
        rewriter.getContext(), elemTypeQ, elemTypeQLoad,
        ldsLayoutCfgNG0.doRotateWithK, ldsLayoutCfgNG0.doSwapThreadIterSubDims,
        ldsLayoutCfgNG0.ldsLayoutDxK, directToLDSQ,
        /*splitKAcrossThreadsFirst=*/false, gemm0G, gemm0N, gemm0InNPerThread,
        /*ldsTransposeEnabled=*/ldsDecisionGemm0.enableB,
        /*accelDDim=*/ldsDecisionGemm0.mfmaDDim,
        /*accelKDim=*/ldsDecisionGemm0.mfmaKDim);

    // LDS Transpose Decision for GEMM1 (V x P)
    // Note: LDS transpose for V is ONLY enabled when P is prefetched
    // (doBypassLDSSecondGemm = true).
    hwtranspose::LDSTransposeDecision ldsDecisionGemm1 =
        hwtranspose::decideLDSTransposeForOperands(
            accelEmitterPtrGemm1.get(), arch, elemTypeV, elemTypeV, directToLDS,
            ldsLayoutCfgMG1, ldsLayoutCfgMG1, gemm1MPerBlock, gemm1NPerBlock,
            gemm1KPerBlock, gemm1TuningParams.getMPerWave(),
            gemm1TuningParams.getNPerWave(), gemm1kpack,
            /*doubleBuffering=*/false,
            /*bLoadsFromLDS=*/!doBypassLDSSecondGemm);

    // Enable LDS transpose for V only when P is prefetched
    bool enableLdsTransposeForV =
        doBypassLDSSecondGemm && ldsDecisionGemm1.enableA;

    BlockwiseMatrixParamsAttr matrixParamsV = BlockwiseMatrixParamsAttr::get(
        rewriter.getContext(), elemTypeV, elemTypeVLoad,
        ldsLayoutCfgMG1.doRotateWithK, ldsLayoutCfgMG1.doSwapThreadIterSubDims,
        ldsLayoutCfgMG1.ldsLayoutDxK, directToLDS, doBypassLDSSecondGemm,
        gemm0G, gemm1M, gemm1InMPerThread,
        /*ldsTransposeEnabled=*/enableLdsTransposeForV,
        /*accelDDim=*/ldsDecisionGemm1.mfmaDDim,
        /*accelKDim=*/ldsDecisionGemm1.mfmaKDim);

    // P matrix (operand B) - when prefetched, uses LDS transpose compatible
    // K formula via otherOperandUsesLdsTranspose in
    // createAccelGemmOperandTransforms
    BlockwiseMatrixParamsAttr matrixParamsKxQ = BlockwiseMatrixParamsAttr::get(
        rewriter.getContext(), elemTypeV, elemTypeVLoad, /*rotateDWithK=*/false,
        /*swapThreadIterSubDims=*/false, /*LDSLayoutDxK=*/false,
        /*directToLDS=*/false, /*splitKAcrossThreadsFirst=*/false, gemm0G,
        gemm1N, gemm1InMPerThread,
        /*ldsTransposeEnabled=*/false,
        /*accelDDim=*/ldsDecisionGemm1.mfmaDDim,
        /*accelKDim=*/ldsDecisionGemm1.mfmaKDim);

    // If gemm0K is equal to gemm0KPerBlock that means
    // effectively there is no K loop. Therefore, we
    // can prefetch the Q tile into regs outside of the
    // loop.
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    if (prefetchQTile) {
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: gemm0K is equal to gemm0KPerBlock\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: Prefetching Q tile into regs...\n");

      // it is fine m iteration to be zero as it irrelevant to Q tensor
      // as the first gemm is Kt x Qt.
      auto gridCoordsGemm0LoadQ =
          layout::makeGxNGridLayout(rewriter, loc, bid, zero, gemm0NBlocks,
                                    gridSize, arch, numChiplets, splitKVConst);

      Value ldsByteBufferQ = nullptr;
      if (!doBypassLDSForQ)
        ldsByteBufferQ =
            createLDSByteBuffer(rewriter, loc, ldsByteBufferQSize, elemTypeQ);

      loadAndStoreGemmInputTile(
          rewriter, loc, inQ, /*kiter=*/zero, tid, gridCoordsGemm0LoadQ,
          ldsByteBufferQ, preAccelRegBuffersQForLoad, loadTypeQ, "n", blockSize,
          elemTypeQ, elemTypeQLoad, gemm0TuningParams, featuresAttr,
          matrixParamsK, matrixParamsQ);
    }

    bool dynamicMLoop = splitKV != 1 || isCausal || isKVCache;

    Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
    scf::ForOp mLoopOp = scf::ForOp::create(rewriter, loc, start, end, one);
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mLoopOp.getBody());
      int64_t kIterationsGemm0 = gemm0K / gemm0KPerBlock;
      Value mLoopIV = mLoopOp.getInductionVar();
      zeroAccBuffer(rewriter, loc, accRegBufferGemm0);
      layout::GridCoordinates gridCoordsGemm0 =
          layout::makeGxNGridLayout(rewriter, loc, bid, mLoopIV, gemm0NBlocks,
                                    gridSize, arch, numChiplets, splitKVConst);

      // LDS buffers
      Value ldsByteBufferQ;
      if (!prefetchQTile) {
        ldsByteBufferQ =
            createLDSByteBuffer(rewriter, loc, ldsByteBufferQSize, elemTypeQ);
      }

      Value ldsByteBufferK = createLDSByteBuffer(
          rewriter, loc, gemm0KPerBlock * gemm0MPerBlock, elemTypeK);

      // LDS Barrier (issue 1811): some threads might be loading from LDS
      // while others are in the next iteration (here), writing to LDS. This
      // barrier prevents that.
      std::optional<APInt> mLoopIters = std::nullopt;
      // mLoopOp can be a dynamic loop if we are using KV Cache or Causal
      // masking. If that's the case, we can't know the number of iterations
      // at compile time.
      if (!dynamicMLoop)
        mLoopIters = mLoopOp.getStaticTripCount();

      bool mIterOneIter =
          mLoopIters.has_value() && mLoopIters.value().getSExtValue() == 1;
      if (!mIterOneIter) {
        LLVM_DEBUG(llvm::dbgs() << "adding a barrier in the first gemm loop\n");
        LDSBarrierOp::create(rewriter, loc);
      }

      Value endKLoop =
          rewriter.createOrFold<arith::ConstantIndexOp>(loc, kIterationsGemm0);
      scf::ForOp kLoopOp = createMainLoop(rewriter, loc, endKLoop, loadType);
      {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(kLoopOp.getBody());
        Value kLoopIV = kLoopOp.getInductionVar();

        // if gemm0K is equal to gemm0KPerBlock, the Q tile
        // is already prefetched into regs. See above.
        if (!prefetchQTile) {
          loadAndStoreGemmInputTile(
              rewriter, loc, inQ, kLoopIV, tid, gridCoordsGemm0, ldsByteBufferQ,
              preAccelRegBuffersQForLoad, loadTypeQ, "n", blockSize, elemTypeQ,
              elemTypeQLoad, gemm0TuningParams, featuresAttr, matrixParamsK,
              matrixParamsQ);
        }

        loadAndStoreGemmInputTile(
            rewriter, loc, inK, kLoopIV, tid, gridCoordsGemm0, ldsByteBufferK,
            preAccelRegBufferKForLoad, loadType, "m", blockSize, elemTypeK,
            elemTypeKLoad, gemm0TuningParams, featuresAttr, matrixParamsK,
            matrixParamsQ);

        // Conservative barrier: Ensure all LDS writes complete
        // before MMA stage reads from LDS. RockPipelinePass will remove this
        // and add optimized barriers when pipelining.
        LDSBarrierOp::create(rewriter, loc);

        auto computeStage = StageOp::create(rewriter, loc, "MMA");
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(
              &computeStage.getRegion().emplaceBlock());

          // Emit lowered blockwise GEMM 0.
          TypedValue<MemRefType> ldsTileBufferK;
          if (directToLDS) {
            ldsTileBufferK = viewBufferAs(rewriter, ldsByteBufferK, elemTypeK);
          } else {
            ldsTileBufferK =
                viewBufferAs(rewriter, ldsByteBufferK,
                             vectorTypeOrSelf(elemTypeK, gemm0kpack));
          }

          TypedValue<MemRefType> ldsTileBufferQ = nullptr;
          if (loadTypeQ != GemmLoadTileType::BypassLDS) {
            if (directToLDSQ) {
              ldsTileBufferQ =
                  viewBufferAs(rewriter, ldsByteBufferQ, elemTypeQ);
            } else {
              ldsTileBufferQ =
                  viewBufferAs(rewriter, ldsByteBufferQ,
                               vectorTypeOrSelf(elemTypeQ, gemm0kpack));
            }
          }

          blockwiseGemmAccel(
              rewriter, loc, loadType, loadTypeQ, preAccelRegBufferK,
              preAccelRegBuffersQ, accRegBufferGemm0, matrixParamsK,
              matrixParamsQ, ldsTileBufferK, ldsTileBufferQ,
              /*scaleA=*/nullptr, /*scaleB=*/nullptr,
              /*bufferScaleA=*/nullptr, /*bufferScaleB=*/nullptr, featuresAttr,
              op.getBlockSizeAttr(), gemm0TuningParams);

          rock::YieldOp::create(rewriter, loc);
        }

        // Conservative barrier: Ensure all LDS reads complete before the next
        // iteration writes to LDS. RockPipelinePass will remove this and add
        // optimized barriers when pipelining.
        LDSBarrierOp::create(rewriter, loc);
      }
      accelEmitterPtrGemm0->computeOutputConversion(
          rewriter, loc, accRegBufferGemm0, gemm0OutBuffer, forceUnroll);

      // ================================================================
      // V PREFETCH: Issue global reads for V tile 0 before softmax.
      // ================================================================
      // By issuing V global reads here (before softmax computation),
      // we overlap the ~120+ instructions of softmax work with the
      // global memory access latency for V, matching CK's approach.
      //
      // The flow is:
      //   1. Issue V global reads -> register buffer  [HERE, before softmax]
      //   2. Softmax computation                      [hides load latency]
      //   3. Write V from registers -> LDS            [after softmax]
      //   4. GEMM1 first iteration uses V from LDS    [peeled iteration]
      //   5. Remaining GEMM1 iters: normal load+MMA   [pipelineable loop]
      //
      // The split is implemented using two new GemmLoadTileType values:
      //   - GlobalReadOnly: emits only the GlobalRead stage
      //     (ThreadwiseReadIntoOp: global -> register buffer, no LDS write)
      //   - LDSWriteFromRegs: emits only the LDSWrite stage
      //     (ThreadwiseCopyOp + ThreadwiseWriteAllOp: regs -> LDS,
      //      no global read)
      // Both phases share the same flat register buffer (vPrefetchRegs).
      Value ldsByteBufferV;
      Value vPrefetchRegs;
      layout::GridCoordinates gridCoordsGemm1;
      bool prefetchFirstVTile = op.getEnableSoftmax() && !directToLDS;
      if (prefetchFirstVTile) {
        ldsByteBufferV = createLDSByteBuffer(
            rewriter, loc, gemm1KPerBlock * gemm1MPerBlock, elemTypeV);
        gridCoordsGemm1 = layout::makeGxNGridLayout(
            rewriter, loc, bid, zero, gemm1NBlocks, gridSize, arch,
            numChiplets, splitKVConst);
        gridCoordsGemm1.m_block = zero; // First V tile (block index 0)

        // Allocate a flat register buffer shared between the GlobalReadOnly
        // and LDSWriteFromRegs phases. Size must match what the lowering
        // computes: copyPerThread = (kPerBlock * dPerBlock) / blockSize.
        int64_t vCopyPerThread =
            (gemm1KPerBlock * gemm1MPerBlock) / blockSize;
        vPrefetchRegs = gpuAlloc(rewriter, loc, vCopyPerThread, elemTypeV,
                                 gpu::AddressSpace::Private);

        // Phase 1: Issue global reads for V tile 0 into register buffer.
        // Only the GlobalRead stage is emitted; LDS write is deferred.
        loadAndStoreGemmInputTile(
            rewriter, loc, inV,
            /*kIter=*/mLoopIV, tid, gridCoordsGemm1, ldsByteBufferV,
            vPrefetchRegs, GemmLoadTileType::GlobalReadOnly, "m", blockSize,
            elemTypeV, elemTypeVLoad, gemm1TuningParams, featuresAttr,
            matrixParamsV, matrixParamsKxQ);

        // Insert a scheduling barrier to prevent the LLVM backend scheduler
        // from sinking the V global loads past the softmax computation.
        // Without this barrier, the scheduler moves the V loads to after
        // softmax, defeating the latency hiding optimization.
        // mask = none (0x0): full barrier, no instructions may cross.
        amdgpu::SchedBarrierOp::create(
            rewriter, loc, amdgpu::sched_barrier_opt_enum::none);
      }

      int64_t prePadG0M = gemm0M;
      if (op.getPrePadG0M().has_value()) {
        prePadG0M = op.getPrePadG0M().value().getSExtValue();
      }
      int64_t prePadG0N = gemm0N;
      if (op.getPrePadG0N().has_value()) {
        prePadG0N = op.getPrePadG0N().value().getSExtValue();
      }
      RegsAsMatrixSubTiles gemm0OutSubTileViewsTrUnPadded =
          unpadGridSubTileView(rewriter, loc, gemm0OutSubTileViewsTr, prePadG0N,
                               prePadG0M);

      // undo Grouped-Query Attention (GQA) transforms
      // This is needed because the preSoftmaxElementWise inputs (if any), don't
      // have the GQA transformed applied to them. So, we undo the transform to
      // the output of the first GEMM. See postProcessFirstGemm() to understand
      // the transforms done to preSoftmaxElementWise inputs.
      ArrayRef<int64_t> unpaddedShape =
          getLowerShape(gemm0OutSubTileViewsTrUnPadded.gridSubTile);
      ArrayAttr undoGQA = undoGQATransforms(rewriter, loc, op, unpaddedShape);

      // undo the GQA transforms for postProcessFirstGemm()
      if (undoGQA) {
        ArrayAttr linalgGridSubTileMaps =
            gemm0OutSubTileViewsTrUnPadded.gridSubTile;
        linalgGridSubTileMaps =
            prependUpperViews(rewriter, linalgGridSubTileMaps, undoGQA);
        gemm0OutSubTileViewsTrUnPadded.gridSubTile = linalgGridSubTileMaps;
      }

      // Apply splitKV transforms if needed
      // This transforms the GEMM0 output from [B*H, SeqQ, SeqK] to
      // [B*H*splitKV, SeqQ, SeqK/splitKV] to match the preSoftmax inputs.
      int64_t splitKV = op.getSplitKV();
      if (splitKV > 1 && op.getPreSoftmaxHasSplitKVTransforms()) {
        ArrayAttr splitKVTransforms = createSplitKVTransformsForGemm0Out(
            rewriter, loc, unpaddedShape, splitKV);
        assert(splitKVTransforms && "splitKV transforms should be non-null");
        ArrayAttr linalgGridSubTileMaps =
            gemm0OutSubTileViewsTrUnPadded.gridSubTile;
        linalgGridSubTileMaps = prependUpperViews(
            rewriter, linalgGridSubTileMaps, splitKVTransforms);
        gemm0OutSubTileViewsTrUnPadded.gridSubTile = linalgGridSubTileMaps;
      }

      // Align the preSoftmaxElementWise (if any) linalg.generic to
      // be performed on the output of the first gemm.
      FailureOr<Value> maybeFusionOutBuffer = postProcessFirstGemm(
          rewriter, loc, op, gridCoordsGemm0, gemm0OutBuffer, fusionOutBuffer,
          gemm0OutSubTileViewsTrUnPadded);
      if (failed(maybeFusionOutBuffer)) {
        return op.emitError("post processing first gemm failed.\n");
      }
      gemm0OutBuffer = maybeFusionOutBuffer.value();
      if (fusionOutElemType == elemTypeSoftmax)
        softmaxInputBuffer = gemm0OutBuffer;

      // Softmax
      if (op.getEnableSoftmax()) {
        // convert gemm0OutBuffer to elemTypeSoftmax
        if (fusionOutElemType != elemTypeSoftmax) {
          createTypeConversionFlatAndStore(rewriter, loc, gemm0OutBuffer,
                                           softmaxInputBuffer);
        }
        // Scale gemm0 output by (1/ln2)
        // So that we can use exp2 instead of exp.
        Value ln2Recip = createConstantFloatOp(
            rewriter, loc, elemTypeSoftmax, elemTypeSoftmax, 1.44269504f,
            elemTypeSoftmax.getIntOrFloatBitWidth() >= 32 ? APFloat::opOK
                                                          : APFloat::opInexact);
        postProcessFirstGemmSplat<ElementwiseMultOp>(
            rewriter, loc, gridCoordsGemm0, softmaxInputBuffer,
            gemm0OutSubTileViews,
            ln2Recip.getDefiningOp<arith::ConstantOp>().getValue());

        // Handle padding
        bool hasPadding =
            op.getPrePadG0M().has_value() || op.getPrePadG0N().has_value();
        if (hasPadding) {
          createFirstGemmNegInfPadding(rewriter, loc, gridCoordsGemm0,
                                       softmaxInputBuffer,
                                       gemm0OutSubTileViewsTrUnPadded);
        }
        // Negative Infinite for extra values based on masking type
        // KV cache masking is independent of causal masking - it masks out
        // positions beyond currentSeqLen (padding). Apply it whenever KV
        // cache is enabled, regardless of causal/prefix-causal mode.
        setGemm0OutputOutOfScope(rewriter, loc, OutOfScopeType::KVCache,
                                 gridCoordsGemm0, softmaxInputBuffer,
                                 gemm0OutSubTileViewsTr, isKVCache, mLoopIV,
                                 gemm0MBlocksLastIter, currentSeqLen,
                                 /*prefixOffset=*/nullptr,
                                 /*numRepeatsGQA=*/nullptr,
                                 /*slidingWindowLowerBound=*/nullptr);

        if (isPrefixCausal) {
          // Prefix causal: mask when key > (query + offset).
          // This combines causal masking with a prefix offset
          setGemm0OutputOutOfScope(
              rewriter, loc, OutOfScopeType::PrefixCausal, gridCoordsGemm0,
              softmaxInputBuffer, gemm0OutSubTileViewsTr, isPrefixCausal,
              mLoopIV, gemm0MBlocksLastIter,
              /*currentSeqLen=*/nullptr, prefixOffset,
              op.getNumRepeatsGQAAttr(),
              /*slidingWindowLowerBound=*/nullptr, firstCausalMaskIter);
        } else if (isCausal) {
          // Standard causal masking: mask when key > query
          setGemm0OutputOutOfScope(
              rewriter, loc, OutOfScopeType::Causal, gridCoordsGemm0,
              softmaxInputBuffer, gemm0OutSubTileViewsTr, isCausal, mLoopIV,
              gemm0MBlocksLastIter,
              /*currentSeqLen=*/nullptr,
              /*prefixOffset=*/nullptr, op.getNumRepeatsGQAAttr(),
              /*slidingWindowLowerBound=*/nullptr, firstCausalMaskIter);
        }

        // Sliding window masking: mask when key_pos < max(0, currentSeqLen -
        // windowSize). This is independent of causal masking and applies
        // alongside KV-cache masking.
        setGemm0OutputOutOfScope(
            rewriter, loc, OutOfScopeType::SlidingWindow, gridCoordsGemm0,
            softmaxInputBuffer, gemm0OutSubTileViewsTr, slidingWindowSize > 0,
            mLoopIV, gemm0MBlocksLastIter,
            /*currentSeqLen=*/nullptr,
            /*prefixOffset=*/nullptr, /*numRepeatsGQA=*/nullptr,
            slidingWindowLowerBound);

        APInt reductionAxis = APInt(64, 1);
        // Softmax max reduction
        Value ldsReductionWorkspaceByteBuffer = createLDSByteBuffer(
            rewriter, loc, reductionWorkspaceSize, elemTypeSoftmax);
        TypedValue<MemRefType> ldsReductionWorkspaceBuffer = viewBufferAs(
            rewriter, ldsReductionWorkspaceByteBuffer, elemTypeSoftmax);
        BlockwiseBroadcastReduceOp::create(
            rewriter, loc, softmaxInputBuffer, ldsReductionWorkspaceBuffer,
            softmaxBufferMax,
            /*extraOut=*/nullptr, reductionAxis, rock::ReduceMethod::Max,
            gemm0OutSubTileViewsTr.blockSubTile,
            gemm0OutSubTileViewsTr.blockSubTileTidSlice.value(),
            gemm0OutSubTileViewsTr.threadSubTile, /*extraViews=*/nullptr,
            blockSize);

        FailureOr<ArrayAttr> maybeGemm0ThreadSubTileInvert = invertTransforms(
            rewriter, loc, gemm0OutSubTileViewsTr.threadSubTile);
        if (failed(maybeGemm0ThreadSubTileInvert)) {
          return op.emitError(
              "cannot invert gemm0OutSubTileViewsTr.threadSubTile");
        }

        // softmax normalization.
        Value gemm0MNThreadwiseView =
            transform(rewriter, softmaxInputBuffer,
                      maybeGemm0ThreadSubTileInvert.value());
        Value gemm0MNExpThreadwiseView = transform(
            rewriter, softmaxBufferExp, maybeGemm0ThreadSubTileInvert.value());
        Value gemm0MNMaxThreadwiseView = transform(
            rewriter, softmaxBufferMax, maybeGemm0ThreadSubTileInvert.value());
        expSubstractMaxFromGemm0(rewriter, loc, gemm0MNThreadwiseView,
                                 gemm0MNExpThreadwiseView,
                                 gemm0MNMaxThreadwiseView, maxRowBuffer);

        // Softmax sum reduction
        Value ldsReductionWorkspaceByteSecondBuffer = createLDSByteBuffer(
            rewriter, loc, reductionWorkspaceSize, elemTypeSoftmax);
        TypedValue<MemRefType> ldsReductionWorkspaceSecondBuffer = viewBufferAs(
            rewriter, ldsReductionWorkspaceByteSecondBuffer, elemTypeSoftmax);
        BlockwiseBroadcastReduceOp::create(
            rewriter, loc, softmaxBufferExp, ldsReductionWorkspaceSecondBuffer,
            softmaxBufferSum, /*extraOut=*/nullptr, reductionAxis,
            rock::ReduceMethod::Sum, gemm0OutSubTileViewsTr.blockSubTile,
            gemm0OutSubTileViewsTr.blockSubTileTidSlice.value(),
            gemm0OutSubTileViewsTr.threadSubTile,
            /*extraViews=*/nullptr, blockSize);
        FailureOr<ArrayAttr> maybeThreadSubTileAttr = invertTransforms(
            rewriter, loc, gemm0OutSubTileViewsTr.threadSubTile);
        if (failed(maybeThreadSubTileAttr)) {
          return op.emitError(
              "cannot invert gemm0OutSubTileViewsTr.threadSubTile");
        }
        Value gemm0SumThreadwiseView = transform(
            rewriter, softmaxBufferSum, maybeThreadSubTileAttr.value());
        Value gemm0MaxThreadwiseView = transform(
            rewriter, softmaxBufferMax, maybeThreadSubTileAttr.value());
        updateRowSum(rewriter, loc, gemm0SumThreadwiseView,
                     gemm0MaxThreadwiseView, sumRowBuffer, maxRowBuffer,
                     expMaxDiffRowBuffer);
      }

      // ================================================================
      // V PREFETCH: Complete LDS write for first V tile after softmax.
      // ================================================================
      // The global reads issued before softmax should have completed
      // (or be very close to completing) by now, since ~120+ instructions
      // of softmax computation have executed in between. Write the V
      // data from the register buffer to LDS so GEMM1 can consume it.
      if (prefetchFirstVTile) {
        // No scheduling barrier here — we intentionally let the scheduler
        // move the V LDS writes (and the preceding s_waitcnt) earlier into
        // the tail of softmax. The V global loads were issued before softmax
        // and should have completed by this point, so the s_waitcnt is
        // essentially free and the ds_writes overlap with remaining softmax
        // work, giving us even more latency hiding.

        // Phase 2: Write V data from register buffer to LDS.
        // Only the LDSWrite stage is emitted; global read was already done
        // before softmax in Phase 1 (GlobalReadOnly).
        loadAndStoreGemmInputTile(
            rewriter, loc, inV,
            /*kIter=*/mLoopIV, tid, gridCoordsGemm1, ldsByteBufferV,
            vPrefetchRegs, GemmLoadTileType::LDSWriteFromRegs, "m",
            blockSize, elemTypeV, elemTypeVLoad, gemm1TuningParams,
            featuresAttr, matrixParamsV, matrixParamsKxQ);
        LDSBarrierOp::create(rewriter, loc);
      }

      // Emit blockwise GEMM 1.
      {
        auto gemm0Out =
            op.getEnableSoftmax() ? softmaxBufferExp : softmaxInputBuffer;
        if (elemTypeV != elemTypeSoftmax) {
          createTypeConversionFlatAndStore(rewriter, loc, gemm0Out,
                                           gemm1RegBufferB);
        } else {
          gemm1RegBufferB = gemm0Out;
        }
        Value gemm1LDSByteBufferB;
        // TODO: extend BlockwiseLoadTileOp to support loading from register
        // buffer (as below)
        if (!doBypassLDSSecondGemm) {
          // The output RegsAsSubTile views are N x M where N is reduction dim
          RegsAsMatrixSubTiles gemm0OutSubTileNxMViews = gemm0OutSubTileViews;
          FailureOr<ArrayAttr> gemm0ThreadwiseSubtileViewNxMMaps =
              invertTransforms(rewriter, loc,
                               gemm0OutSubTileNxMViews.threadSubTile);
          if (failed(gemm0ThreadwiseSubtileViewNxMMaps)) {
            return op.emitError(
                "cannot invert gemm0OutSubTileNxMViews.threadSubTile");
          }
          Value gemm0ExpNMThreadwiseView =
              transform(rewriter, gemm1RegBufferB,
                        gemm0ThreadwiseSubtileViewNxMMaps.value());
          // TODO: Correct the below toLDSViews to be max LDS vectorizable
          // (For now just hacked in the existing view)
          // Copy copyKPerThread is set to 1 because
          // K is not packed as kpack vectors. Therefore, setting
          // copyKPerThread to be 1 will always make the LDS write
          // to be scalars -- which makes the following layout agnostic.
          // We should get rid of storing to LDS altogether with
          // the transposed layout for this gemm.
          gemm1LDSByteBufferB = createLDSByteBuffer(
              rewriter, loc, gemm1LDSByteBufferBSize, elemTypeV);

          LogicalResult storeGemm1ATileStatus = storeGemmInputTile(
              rewriter, loc, gemm1kpack, gemm0ExpNMThreadwiseView,
              gemm0OutSubTileNxMViews, gemm0ExpOutBufferToLDS,
              gemm1LDSByteBufferB, gemm1KpacksPerBlock, "n", gemm1KPerBlock,
              gemm1NPerBlock, /*copyKPerThread=*/1, gemm1InNPerThread,
              forceUnroll, false);
          if (failed(storeGemm1ATileStatus)) {
            return failure();
          }
        }

        // ================================================================
        // V load + GEMM1 loop: Two paths depending on V prefetch.
        // ================================================================
        // For non-prefetch path: allocate V LDS buffer and grid coords
        // (prefetch path already did this before softmax).
        if (!prefetchFirstVTile) {
          ldsByteBufferV = createLDSByteBuffer(
              rewriter, loc, gemm1KPerBlock * gemm1MPerBlock, elemTypeV);
          gridCoordsGemm1 = layout::makeGxNGridLayout(
              rewriter, loc, bid, zero, gemm1NBlocks, gridSize, arch,
              numChiplets, splitKVConst);
        }

        // ----------------------------------------------------------------
        // Helper lambda: Emit GEMM1 MMA + PostProcess for a single V tile.
        // Parameterized by V block index (g1MBlockIdx) to support both
        // the peeled first iteration and the remaining loop iterations.
        // This avoids duplicating ~100 lines of MMA + PostProcess code.
        // ----------------------------------------------------------------
        auto emitGemm1Compute =
            [&](Value g1MBlockIdx, GemmLoadTileType vLoadType,
                Value vRegBuf) -> LogicalResult {
          // Emit GEMM 1 MMA.
          auto computeStage = StageOp::create(rewriter, loc, "MMA");
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(
                &computeStage.getRegion().emplaceBlock());

            Value matrixC = accRegBufferGemm1;
            if (op.getEnableSoftmax()) {
              zeroAccBuffer(rewriter, loc, matrixC);
            } else {
              if (gemm1MBlocks > 1) {
                matrixC = createSliceOfFirstDim(rewriter, loc, matrixC,
                                                g1MBlockIdx);
              }
            }

            if (doBypassLDSSecondGemm) {
              FailureOr<ArrayAttr> gemm1ThreadwiseSubtileViewDxKMaps =
                  invertTransforms(rewriter, loc,
                                   gemm0OutSubTileViewsTr.threadSubTile);
              if (failed(gemm1ThreadwiseSubtileViewDxKMaps)) {
                return op.emitError(
                    "cannot invert gemm0OutSubTileViewsTr.threadSubTile");
              }
              Value gemm1BDxKThreadwiseView =
                  transform(rewriter, gemm1RegBufferB,
                            gemm1ThreadwiseSubtileViewDxKMaps.value());
              affine::AffineForOp nRepeatsLoop = affine::AffineForOp::create(
                  rewriter, loc, 0, accelParamsGemm1.nRepeats, 1);
              {
                PatternRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(nRepeatsLoop.getBody());
                Value ni = nRepeatsLoop.getInductionVar();
                Value subview = preAccelRegBufferQxK;
                if (accelParamsGemm1.nRepeats > 1) {
                  subview = createSliceOfFirstDim(rewriter, loc,
                                                  preAccelRegBufferQxK, ni);
                }
                ThreadwiseReadIntoOp::create(
                    rewriter, loc, gemm1BDxKThreadwiseView, subview,
                    rewriter.getArrayAttr({}), ValueRange{ni}, true, true);
              }
            }

            TypedValue<MemRefType> ldsTileBufferV;
            if (directToLDS) {
              ldsTileBufferV =
                  viewBufferAs(rewriter, ldsByteBufferV, elemTypeV);
            } else {
              ldsTileBufferV =
                  viewBufferAs(rewriter, ldsByteBufferV,
                               vectorTypeOrSelf(elemTypeV, gemm1kpack));
            }
            TypedValue<MemRefType> gemm1LDSBufferB = nullptr;
            if (!doBypassLDSSecondGemm)
              gemm1LDSBufferB =
                  viewBufferAs(rewriter, gemm1LDSByteBufferB,
                               vectorTypeOrSelf(elemTypeV, gemm1kpack));

            auto loadTypeKxD = doBypassLDSSecondGemm
                                   ? GemmLoadTileType::BypassLDS
                                   : GemmLoadTileType::Default;
            blockwiseGemmAccel(
                rewriter, loc, vLoadType, loadTypeKxD, vRegBuf,
                preAccelRegBufferQxK, matrixC, matrixParamsV, matrixParamsKxQ,
                ldsTileBufferV, gemm1LDSBufferB,
                /*scaleA=*/nullptr, /*scaleB=*/nullptr,
                /*bufferScaleA=*/nullptr, /*bufferScaleB=*/nullptr,
                featuresAttr, op.getBlockSizeAttr(), gemm1TuningParams);

            rock::YieldOp::create(rewriter, loc);
          }

          // Emit GEMM 1 PostProcess.
          auto postProcessStage =
              StageOp::create(rewriter, loc, "PostProcess");
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(
                &postProcessStage.getRegion().emplaceBlock());

            // There is no second k-loop
            // Therefore can get the output straight away
            Value gemm1OutBufferPerG1MBlock = gemm1OutBuffer;
            Value matrixC = accRegBufferGemm1;
            if (!op.getEnableSoftmax() && gemm1MBlocks > 1) {
              gemm1OutBufferPerG1MBlock = createSliceOfFirstDim(
                  rewriter, loc, gemm1OutBuffer, g1MBlockIdx);
              matrixC = createSliceOfFirstDim(rewriter, loc, matrixC,
                                              g1MBlockIdx);
            }

            accelEmitterPtrGemm1->computeOutputConversion(
                rewriter, loc, matrixC, gemm1OutBufferPerG1MBlock, forceUnroll);
            if (op.getEnableSoftmax()) {
              Value attentionOutAccBufferPerG1MBlock = attentionOutAccBuffer;
              if (gemm1MBlocks > 1) {
                attentionOutAccBufferPerG1MBlock = createSliceOfFirstDim(
                    rewriter, loc, attentionOutAccBuffer, g1MBlockIdx);
              }
              FailureOr<ArrayAttr> maybeInvertedGemm1threadSubTileMaps =
                  invertTransforms(rewriter, loc,
                                   gemm1OutSubTileViewsTr.threadSubTile);
              if (failed(maybeInvertedGemm1threadSubTileMaps)) {
                return op.emitError(
                    "cannot invert gemm1OutSubTileViewsTr.threadSubTile");
              }
              Value gemm1MNThreadwiseView =
                  transform(rewriter, gemm1OutBufferPerG1MBlock,
                            maybeInvertedGemm1threadSubTileMaps.value());
              if (failed(maybeAttentionOutAccBufferThreadSubTileViewMaps)) {
                return op.emitError("cannot invert attention buffer");
              }
              // Rescale/correct output, rowMax and rowSums
              Value attentionOutAccBufferView = transform(
                  rewriter, attentionOutAccBufferPerG1MBlock,
                  maybeAttentionOutAccBufferThreadSubTileViewMaps.value());
              createAttentionRowStateCorrections(
                  rewriter, loc, gemm1MNThreadwiseView,
                  attentionOutAccBufferView, expMaxDiffRowBuffer);
            }

            rock::YieldOp::create(rewriter, loc);
          }

          return success();
        }; // end emitGemm1Compute lambda

        if (prefetchFirstVTile) {
          // ============================================================
          // PREFETCH PATH: First V tile already loaded into LDS.
          // ============================================================
          // V data for tile 0 was prefetched before softmax (global read)
          // and written to LDS after softmax (LDS write + barrier).
          // The first GEMM1 iteration is peeled out of the loop so the
          // remaining iterations form a clean, pipelineable loop.

          // --- Peeled first iteration (g1m = 0) ---
          gridCoordsGemm1.m_block = zero;
          // Use Default load type for the peeled iteration because the V
          // data was written to LDS by the LDSWriteFromRegs phase. There is
          // no BlockwiseLoadTileOp here to create an LDSRead stage, so the
          // GEMM must read V directly from LDS.
          //
          // When double-buffering is active, preAccelRegBufferV is rank-2
          // (e.g. memref<3x2xvector<4xf16>>) because it was allocated with
          // repeats=mRepeats. However, the Default load path in
          // BlockwiseGemmAccelOp reads from LDS into the buffer WITHOUT
          // slicing by the m-repeat loop variable. The downstream
          // generateThreadwiseViewBufferA then creates a rank-1 view,
          // leading to a memref.load rank mismatch. Fix: create a separate
          // rank-1 register buffer for the peeled iteration.
          Value peeledVRegBuf = preAccelRegBufferV;
          if (doubleBuffering) {
            auto [peeledVForLoad, peeledVBuf] =
                createRegInterrimBufferForAccel(
                    rewriter, loc, accelParamsGemm1.argTypeA,
                    accelParamsGemm1.kBasePerThread,
                    /*repeats=*/1, directToLDS);
            peeledVRegBuf = peeledVBuf;
          }
          if (failed(emitGemm1Compute(zero, GemmLoadTileType::Default,
                                      peeledVRegBuf)))
            return failure();

          // --- Remaining iterations (g1m = 1..gemm1MBlocks-1) ---
          // These form a standard pipelineable loop with V loads.
          if (gemm1MBlocks > 1) {
            LDSBarrierOp::create(rewriter, loc);

            Value startG1M =
                rewriter.createOrFold<ConstantIndexOp>(loc, 1);
            Value endG1MLoop =
                rewriter.createOrFold<ConstantIndexOp>(loc, gemm1MBlocks);
            Value oneVal =
                rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
            scf::ForOp g1MLoopOp = scf::ForOp::create(
                rewriter, loc, startG1M, endG1MLoop, oneVal);
            // Mark loop for pipelining
            bool g1DoubleBuffering =
                loadType == GemmLoadTileType::DoubleBuffer ||
                loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;
            int64_t g1InitiationInterval = g1DoubleBuffering ? 1 : 2;
            g1MLoopOp->setAttr(
                PipelineAttr::getMnemonic(),
                rock::PipelineAttr::get(rewriter.getContext(),
                                        g1InitiationInterval));
            {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(g1MLoopOp.getBody());
              Value g1MLoopIndVar = g1MLoopOp.getInductionVar();

              gridCoordsGemm1.m_block = g1MLoopIndVar;

              // Normal V tile load (global -> regs -> LDS)
              loadAndStoreGemmInputTile(
                  rewriter, loc, inV,
                  /*kIter=*/mLoopIV, tid, gridCoordsGemm1, ldsByteBufferV,
                  preAccelRegBufferVForLoad, loadType, "m", blockSize,
                  elemTypeV, elemTypeVLoad, gemm1TuningParams, featuresAttr,
                  matrixParamsV, matrixParamsKxQ);

              // Conservative barrier before MMA
              LDSBarrierOp::create(rewriter, loc);

              if (failed(emitGemm1Compute(g1MLoopIndVar, loadType,
                                          preAccelRegBufferV)))
                return failure();

              // Conservative barrier before next iteration's LDS writes
              LDSBarrierOp::create(rewriter, loc);
            }
          }
        } else {
          // ============================================================
          // ORIGINAL PATH: No V prefetch (softmax disabled).
          // ============================================================
          Value endG1MLoop =
              rewriter.createOrFold<ConstantIndexOp>(loc, gemm1MBlocks);
          scf::ForOp g1MLoopOp =
              createMainLoop(rewriter, loc, endG1MLoop, loadType);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(g1MLoopOp.getBody());
            Value g1MLoopIndVar = g1MLoopOp.getInductionVar();

            gridCoordsGemm1.m_block = g1MLoopIndVar;

            loadAndStoreGemmInputTile(
                rewriter, loc, inV,
                /*kIter=*/mLoopIV, tid, gridCoordsGemm1, ldsByteBufferV,
                preAccelRegBufferVForLoad, loadType, "m", blockSize, elemTypeV,
                elemTypeVLoad, gemm1TuningParams, featuresAttr, matrixParamsV,
                matrixParamsKxQ);

            // Conservative barrier before MMA
            LDSBarrierOp::create(rewriter, loc);

            if (failed(emitGemm1Compute(g1MLoopIndVar, loadType,
                                        preAccelRegBufferV)))
              return failure();

            // Conservative barrier before next iteration's LDS writes
            LDSBarrierOp::create(rewriter, loc);
          }
        }
      }
    }

    if (op.getEnableSoftmax()) {
      affine::AffineForOp g1MLoopOp =
          affine::AffineForOp::create(rewriter, loc, 0, gemm1MBlocks, 1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(g1MLoopOp.getBody());
        Value g1MLoopIndVar = g1MLoopOp.getInductionVar();
        Value attentionOutAccBufferPerG1MBlock = attentionOutAccBuffer;
        if (gemm1MBlocks > 1) {
          attentionOutAccBufferPerG1MBlock = createSliceOfFirstDim(
              rewriter, loc, attentionOutAccBuffer, g1MLoopIndVar);
        }
        if (failed(maybeAttentionOutAccBufferThreadSubTileViewMaps)) {
          return op.emitError("invertTransforms failed attention buffer");
        }
        Value attentionOutAccBufferView =
            transform(rewriter, attentionOutAccBufferPerG1MBlock,
                      maybeAttentionOutAccBufferThreadSubTileViewMaps.value());
        scaleFinalOutput(rewriter, loc, attentionOutAccBufferView,
                         sumRowBuffer);
      }
    }
    Value outAccBuffer =
        op.getEnableSoftmax() ? attentionOutAccBuffer : gemm1OutBuffer;
    if (elemTypeSoftmax != elemTypeOut) {
      // We flatten output buffer in case gemm1MBlocks > 1
      // where those are iterated.
      createTypeConversionFlatAndStore(rewriter, loc, outAccBuffer,
                                       outAccBufferOutTyped);
    }
    if (lse) {
      // it must be guaranteed by the verifier
      assert(op.getEnableSoftmax());
      assert(lseBuffer);
      Value lseBufferView =
          transform(rewriter, lseBuffer,
                    maybeAttentionOutAccBufferThreadSubTileViewMaps.value());
      computeLse(rewriter, loc, lseBufferView, sumRowBuffer, maxRowBuffer);
    }

    // Close the early exit if block here. Everything above this point is
    // conditional (only runs when there's work to do). Everything below
    // (output writes) always executes, writing zeros when there's no work.
    if (earlyExitIf.has_value()) {
      rewriter.setInsertionPointAfter(*earlyExitIf);
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: early exit enabled - "
                 << "output writes will execute unconditionally\n");
    }

    MemRefType outAccBufferOutType =
        cast<MemRefType>(outAccBufferOutTyped.getType());
    int64_t numElementsAttnOut = outAccBufferOutType.getNumElements();
    // This map will create an upper view [gblock, nblock, flatiter] -> [gblock,
    // miter, nblock, iter]
    TransformMapAttr flatToMiterMap =
        getFlatToMiterMap(rewriter, gemm0G, gemm1MBlocks, gemm1NBlocks,
                          blockSize, numElementsAttnOut);
    ArrayAttr outGridSubTile =
        prependUpperViews(rewriter, rewriter.getArrayAttr({flatToMiterMap}),
                          gemm1OutSubTileViews.gridSubTile);

    // Note that we don't use splitKV here because that dimension belongs to the
    // batch size already for output tensors
    auto gridCoordsGemm1 = layout::makeGxNGridLayout(
        rewriter, loc, bid, zero, gemm1NBlocks, gridSize, arch, numChiplets);
    Value outAccBufferOutTypedFlat =
        getFlattenedMemref(rewriter, outAccBufferOutTyped);
    ThreadwiseWriteAllOp::create(
        rewriter, loc, outAccBufferOutTypedFlat, trOut, outGridSubTile,
        /*extraIndices=*/
        ValueRange{gridCoordsGemm1.g_block, gridCoordsGemm1.n_block, tid},
        op.getStoreMethod(), forceUnroll,
        /*useIndexDiffs=*/true);

    // store LSE to device memory
    if (lse) {
      // drop gemmM dimension
      TopDownTMBuilder viewBuilder(rewriter, {"gemmG", "gemmM", "gemmN"},
                                   {gemm0G, gemm1M, gemm1N});
      viewBuilder.passThrough({"gemmG", "gemmN"}, {0, 1}, {"gemmG", "gemmN"});
      viewBuilder.ignore("gemmM");
      auto dropM = rewriter.getArrayAttr({viewBuilder.get()});

      MemRefType lseBufferOutType = cast<MemRefType>(lseBuffer.getType());
      int64_t numElementsLseOut = lseBufferOutType.getNumElements();
      auto flatToMiterMapAttr = getFlatToMiterMap(
          rewriter, gemm0G, 1, gemm1NBlocks, blockSize, numElementsLseOut);
      // slice mIter
      BottomUpTMBuilder sliceBuilder(
          rewriter, {"g_block", "mIter", "n_block", "tid", "iter"},
          {gemm0G, gemm1MBlocks, gemm1NBlocks, blockSize, numElementsLseOut},
          loc);
      sliceBuilder.passThrough({"g_block", "n_block", "tid", "iter"},
                               {0, 2, 3, 4},
                               {"g_block", "n_block", "tid", "iter"});
      sliceBuilder.slice({"mIter"}, {"mIter"}, {0}, {1});
      auto sliceAttr = sliceBuilder.get();

      ArrayAttr flatToMiterSlice = prependUpperViews(
          rewriter, rewriter.getArrayAttr({flatToMiterMapAttr}),
          rewriter.getArrayAttr({sliceAttr}));
      ArrayAttr outGridSubTile = prependUpperViews(
          rewriter, flatToMiterSlice, gemm1OutSubTileViews.gridSubTile);
      ArrayAttr lseMap = prependUpperViews(rewriter, outGridSubTile, dropM);
      ThreadwiseWriteAllOp::create(
          rewriter, loc, lseBuffer, lse, lseMap,
          /*extraIndices=*/
          ValueRange{gridCoordsGemm1.g_block, gridCoordsGemm1.n_block, tid},
          rock::StoreMethod::Set, forceUnroll,
          /*useIndexDiffs=*/true);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

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

    // Get 'features' from arch
    StringRef arch = rock::getArchValue(op);
    rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
    auto features = archInfo.defaultFeatures;
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

    int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
    int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;

    int64_t aCopyKpacksPerThread =
        math_util::integer_divide_ceil(aCopyPerThread, kpack);
    int64_t bCopyKpacksPerThread =
        math_util::integer_divide_ceil(bCopyPerThread, kpack);

    int64_t scheduleVersion = tuningParams.getScheduleVersion();

    // Check if the schedule version is supported by the hardware
    SmallVector<Type> types = {elementTypeA, elementTypeB};
    if (failed(
            isScheduleVersionSupported(scheduleVersion, archInfo, types, arch)))
      return op.emitOpError("schedule version not supported");

    std::optional<GemmLoadTileType> maybeLoadType =
        symbolizeGemmLoadTileType(scheduleVersion);
    if (!maybeLoadType.has_value())
      return op.emitOpError("schedule version value is incorrect");

    auto loadType = maybeLoadType.value();
    bool directToLDS = loadType == GemmLoadTileType::DirectToLDSDefault ||
                       loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;

    // Get the vector copy layout for A and B
    FailureOr<VectorDimInfo> maybeVecDimInfoA =
        getVectorDim(loc, matA, elementTypeALoad, blockSize, kPerBlock,
                     mPerBlock, kpack, directToLDS);
    if (failed(maybeVecDimInfoA)) {
      return failure();
    }
    FailureOr<VectorDimInfo> maybeVecDimInfoB =
        getVectorDim(loc, matB, elementTypeBLoad, blockSize, kPerBlock,
                     nPerBlock, kpack, directToLDS);
    if (failed(maybeVecDimInfoB)) {
      return failure();
    }
    auto copyMPerThread = maybeVecDimInfoA->inDPerThread;
    auto copyNPerThread = maybeVecDimInfoB->inDPerThread;
    LLVM_DEBUG(llvm::dbgs()
               << "gridSize: " << gridSize << "\n"
               << "blockSize: " << blockSize << "\n"
               << "elementTypeALoad: " << elementTypeALoad << "\n"
               << "elementTypeBLoad: " << elementTypeBLoad << "\n"
               << "aCopyPerThread: " << aCopyPerThread << "\n"
               << "bCopyPerThread: " << bCopyPerThread << "\n"
               << "aCopyKpacksPerThread: " << aCopyKpacksPerThread << "\n"
               << "bCopyKpacksPerThread: " << bCopyKpacksPerThread << "\n"
               << "aVectorDim: " << maybeVecDimInfoA->vectorDim << "\n"
               << "aVectorLen: " << maybeVecDimInfoA->vectorLen << "\n"
               << "bVectorDim: " << maybeVecDimInfoB->vectorDim << "\n"
               << "bVectorLen: " << maybeVecDimInfoB->vectorLen << "\n"
               << "vectorTiebreaker: " << maybeVecDimInfoA->vectorTiebreaker
               << "\n"
               << "kPerBlock: " << kPerBlock << "\n"
               << "mPerBlock: " << mPerBlock << "\n"
               << "nPerBlock: " << nPerBlock << "\n"
               << "aCopyKPerThread: " << maybeVecDimInfoA->inKPerThread << "\n"
               << "bCopyKPerThread: " << maybeVecDimInfoB->inKPerThread << "\n"
               << "copyMPerThread: " << copyMPerThread << "\n"
               << "copyNPerThread: " << copyNPerThread << "\n"
               << "directToLDS: " << directToLDS << "\n");
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};

    // Get current workgroup ID.
    auto bid = WorkgroupIdOp::create(b, loc, b.getIndexType());
    // Get current workitem ID.
    auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

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
    IntegerAttr outputSwizzleAttr =
        b.getI64IntegerAttr(tuningParams.getOutputSwizzle());
    func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
    funcOp->setAttr(rock::WavesPerEUAttr::getMnemonic(), wavesPerEUAttr);
    funcOp->setAttr(rock::OutputSwizzleAttr::getMnemonic(), outputSwizzleAttr);

    LDSLayoutConfigDim ldsLayoutConfigA = getLDSLayoutConfigDim(
        elementTypeA, kpack, maybeVecDimInfoA.value(), directToLDS);
    LDSLayoutConfigDim ldsLayoutConfigB = getLDSLayoutConfigDim(
        elementTypeB, kpack, maybeVecDimInfoB.value(), directToLDS);

    // Obtain Accelerator-related attributes.
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();

    auto accelEmitterPtr = accel::AccelEmitter::select(
        features, elementTypeA, elementTypeB, arch, tuningParams);

    if (!accelEmitterPtr)
      return op.emitOpError("Unable to emit accelerator code.");

    // TODO: add an heuristic to decide if the it should use scheduleV1 or V2.
    bool doubleBuffering =
        loadType == GemmLoadTileType::DoubleBuffer ||
        loadType == GemmLoadTileType::DirectToLDSDoubleBuffer;

    // Extract relevant accelerator parameters
    rock::accel::AccelEmitterParams params = accelEmitterPtr->getParams();
    bool useIndexDiffs = true;

    // ============================================================
    // LDS TRANSPOSE DECISION MAKING
    // ============================================================
    hwtranspose::LDSTransposeDecision ldsDecision =
        hwtranspose::decideLDSTransposeForOperands(
            accelEmitterPtr.get(), arch, elementTypeA, elementTypeB,
            directToLDS, ldsLayoutConfigA, ldsLayoutConfigB, mPerBlock,
            nPerBlock, kPerBlock, mPerWave, nPerWave, kpack, doubleBuffering);

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << M << "\n"
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
               << "nPerWave: " << nPerWave << "\n"
               << "aVectorLen: " << maybeVecDimInfoA->vectorLen << "\n"
               << "bVectorLen: " << maybeVecDimInfoB->vectorLen << "\n"
               << "aVectorDim: " << maybeVecDimInfoA->vectorDim << "\n"
               << "bVectorDim: " << maybeVecDimInfoB->vectorDim << "\n");

    // Alocate LDS and create subviews.

    // Compute required LDS sizes.
    int64_t ldsBlockASize =
        getPackedByteSize(kpacksPerBlock * mPerBlock * kpack, elementTypeA);
    int64_t ldsBlockBSize =
        getPackedByteSize(kpacksPerBlock * nPerBlock * kpack, elementTypeB);
    int64_t ldsBlockScaleASize =
        hasScaleA ? getPackedByteSize(kpacksPerBlock * mPerBlock * kpack,
                                      elementTypeScaleA)
                  : 0;
    int64_t ldsBlockScaleBSize =
        hasScaleB ? getPackedByteSize(kpacksPerBlock * nPerBlock * kpack,
                                      elementTypeScaleB)
                  : 0;
    LLVM_DEBUG(llvm::dbgs() << "LDS block sizes (bytes): " << ldsBlockASize
                            << " " << ldsBlockBSize << " " << ldsBlockScaleASize
                            << " " << ldsBlockScaleBSize << "\n");
    if (failed(checkLDSSize(op, ldsBlockASize, ldsBlockBSize,
                            ldsBlockScaleASize, ldsBlockScaleBSize)))
      return op.emitOpError("requires too much LDS");

    // create matrix params (with LDS transpose flag and accel K dimension)
    BlockwiseMatrixParamsAttr matrixParamsA = BlockwiseMatrixParamsAttr::get(
        b.getContext(), elementTypeA, elementTypeALoad,
        ldsLayoutConfigA.doRotateWithK,
        ldsLayoutConfigA.doSwapThreadIterSubDims, ldsLayoutConfigA.ldsLayoutDxK,
        directToLDS, /*splitKAcrossThreadsFirst=*/false, G, M, copyMPerThread,
        /*ldsTranspose=*/ldsDecision.enableA,
        /*accelDDim=*/ldsDecision.mfmaDDim,
        /*accelKDim=*/ldsDecision.mfmaKDim);

    BlockwiseMatrixParamsAttr matrixParamsB = BlockwiseMatrixParamsAttr::get(
        b.getContext(), elementTypeB, elementTypeBLoad,
        ldsLayoutConfigB.doRotateWithK,
        ldsLayoutConfigB.doSwapThreadIterSubDims, ldsLayoutConfigB.ldsLayoutDxK,
        directToLDS, /*splitKAcrossThreadsFirst=*/false, G, N, copyNPerThread,
        /*ldsTranspose=*/ldsDecision.enableB,
        /*accelDDim=*/ldsDecision.mfmaDDim,
        /*accelKDim=*/ldsDecision.mfmaKDim);

    // Allocate LDS.
    Value ldsByteBufferA = createLDSByteBuffer(
        b, loc, kpacksPerBlock * mPerBlock * kpack, elementTypeA);
    Value ldsByteBufferB = createLDSByteBuffer(
        b, loc, kpacksPerBlock * nPerBlock * kpack, elementTypeB);
    Value ldsByteBufferScaleA =
        hasScaleA
            ? createLDSByteBuffer(b, loc, kpacksPerBlock * mPerBlock * kpack,
                                  elementTypeScaleA)
            : nullptr;
    Value ldsByteBufferScaleB =
        hasScaleB
            ? createLDSByteBuffer(b, loc, kpacksPerBlock * nPerBlock * kpack,
                                  elementTypeScaleB)
            : nullptr;
    Type ldsReadTypeA = vectorTypeOrSelf(elementTypeA, kpack);
    Type ldsReadTypeB = vectorTypeOrSelf(elementTypeB, kpack);
    Value ldsViewForGemmA, ldsViewForGemmB, ldsViewForGemmScaleA,
        ldsViewForGemmScaleB;
    if (directToLDS) {
      ldsViewForGemmA = viewBufferAs(b, ldsByteBufferA, elementTypeA);
      ldsViewForGemmB = viewBufferAs(b, ldsByteBufferB, elementTypeB);
      if (isScaledGemm) {
        op->emitOpError("Direct to LDS scaled GEMM is not supported yet.");
        return failure();
      }
    } else {
      ldsViewForGemmA = viewBufferAs(b, ldsByteBufferA, ldsReadTypeA);
      ldsViewForGemmB = viewBufferAs(b, ldsByteBufferB, ldsReadTypeB);
      if (isScaledGemm) {
        Type ldsReadTypeScaleA = vectorTypeOrSelf(elementTypeScaleA, kpack);
        Type ldsReadTypeScaleB = vectorTypeOrSelf(elementTypeScaleB, kpack);
        ldsViewForGemmScaleA =
            viewBufferAs(b, ldsByteBufferScaleA, ldsReadTypeScaleA);
        ldsViewForGemmScaleB =
            viewBufferAs(b, ldsByteBufferScaleB, ldsReadTypeScaleB);
      }
    }

    auto [arrayAForLoad, arrayA] = createRegInterrimBufferForAccel(
        b, loc, params.argTypeA, params.kBasePerThread,
        doubleBuffering ? params.mRepeats : 1, directToLDS);
    auto [arrayBForLoad, arrayB] = createRegInterrimBufferForAccel(
        b, loc, params.argTypeB, params.kBasePerThread,
        doubleBuffering ? params.nRepeats : 1, directToLDS);
    Value regCAllocOp = createBufferForAccelGemmOut(loc, params, b);
    zeroAccBuffer(b, loc, regCAllocOp);
    Value arrayScaleA, arrayScaleB, arrayScaleAForLoad, arrayScaleBForLoad;
    if (isScaledGemm) {
      Type argTypeScaleA = elementTypeScaleA;
      Type argTypeScaleB = elementTypeScaleB;
      if (VectorType argAVector = dyn_cast<VectorType>(params.argTypeA)) {
        argTypeScaleA =
            VectorType::get(argAVector.getNumElements(), elementTypeScaleA);
      }
      if (VectorType argBVector = dyn_cast<VectorType>(params.argTypeB)) {
        argTypeScaleB =
            VectorType::get(argBVector.getNumElements(), elementTypeScaleB);
      }

      std::tie(arrayScaleAForLoad, arrayScaleA) =
          createRegInterrimBufferForAccel(
              b, loc, argTypeScaleA, params.kBasePerThread,
              doubleBuffering ? params.mRepeats : 1, directToLDS);
      std::tie(arrayScaleBForLoad, arrayScaleB) =
          createRegInterrimBufferForAccel(
              b, loc, argTypeScaleB, params.kBasePerThread,
              doubleBuffering ? params.nRepeats : 1, directToLDS);
    }

    // Emit loop.
    int64_t kIterations = K / kPerBlock;
    Value nIterations = ConstantIndexOp::create(b, loc, kIterations);

    scf::ForOp loopOp = createMainLoop(b, loc, nIterations, loadType);
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());
      Value iv = loopOp.getInductionVar();

      // Load from global memory to LDS
      loadAndStoreGemmInputTile(b, loc, matB, /*kiter=*/iv, tid, gridCoords,
                                ldsByteBufferB, arrayBForLoad, loadType, "n",
                                blockSize, elementTypeB, elementTypeBLoad,
                                tuningParams, featuresAttr, matrixParamsA,
                                matrixParamsB);
      loadAndStoreGemmInputTile(b, loc, matA, /*kiter=*/iv, tid, gridCoords,
                                ldsByteBufferA, arrayAForLoad, loadType, "m",
                                blockSize, elementTypeA, elementTypeALoad,
                                tuningParams, featuresAttr, matrixParamsA,
                                matrixParamsB);
      if (isScaledGemm) {
        loadAndStoreGemmInputTile(b, loc, scaleB, /*kiter=*/iv, tid, gridCoords,
                                  ldsByteBufferScaleB, arrayScaleBForLoad,
                                  loadType, "n", blockSize, elementTypeScaleB,
                                  elementTypeBLoad, tuningParams, featuresAttr,
                                  matrixParamsA, matrixParamsB);
        loadAndStoreGemmInputTile(b, loc, scaleA, /*kiter=*/iv, tid, gridCoords,
                                  ldsByteBufferScaleA, arrayScaleAForLoad,
                                  loadType, "m", blockSize, elementTypeScaleA,
                                  elementTypeALoad, tuningParams, featuresAttr,
                                  matrixParamsA, matrixParamsB);
      }

      // Conservative barrier: Ensure all LDS writes complete
      // before MMA stage reads from LDS. RockPipelinePass will remove this
      // and add optimized barriers when pipelining.
      LDSBarrierOp::create(b, loc);

      // Emit blockwise GEMM. This will load data from LDS (or registers) and
      // compute the MMA at the same time
      auto stage2 = StageOp::create(b, loc, "MMA");
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stage2.getRegion().emplaceBlock());

        blockwiseGemmAccel(
            b, loc, loadType, loadType, arrayA, arrayB, regCAllocOp,
            matrixParamsA, matrixParamsB, ldsViewForGemmA, ldsViewForGemmB,
            /*scaleA=*/ldsViewForGemmScaleA, /*scaleB=*/ldsViewForGemmScaleB,
            /*bufferScaleA=*/arrayScaleA, /*bufferScaleB=*/arrayScaleB,
            featuresAttr, op.getBlockSizeAttr(), tuningParams);
        YieldOp::create(b, loc);
      }

      // Conservative barrier: Ensure all LDS reads complete before the next
      // iteration writes to LDS. RockPipelinePass will remove this and add
      // optimized barriers when pipelining.
      LDSBarrierOp::create(b, loc);
    }

    // Matrix C write out logic.
    Value convertedC = createBufferForGemmOut(loc, destType, params, b);

    FailureOr<RegsAsMatrixSubTiles> maybeIdToMatrixCMaps =
        accelEmitterPtr->computeOutputTransforms(
            b, loc, M, N, blockSize, bidGridLengths,
            maybeVecDimInfoA->inDPerThread, maybeVecDimInfoB->inDPerThread,
            ldsLayoutConfigA.doSwapThreadIterSubDims,
            ldsLayoutConfigB.doSwapThreadIterSubDims);
    if (failed(maybeIdToMatrixCMaps)) {
      return failure();
    }
    ArrayAttr idToMatrixCMaps = maybeIdToMatrixCMaps.value().gridSubTile;

    accelEmitterPtr->computeOutputConversion(b, loc, regCAllocOp, convertedC,
                                             forceUnroll);

    ThreadwiseWriteAllOp::create(
        b, loc, convertedC, op.getC(), idToMatrixCMaps,
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
  target.addIllegalOp<rock::GridwiseGemmOp, rock::GridwiseGemmAccelOp,
                      GridwiseAttentionAccelOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         memref::MemRefDialect, affine::AffineDialect,
                         vector::VectorDialect, linalg::LinalgDialect,
                         scf::SCFDialect, math::MathDialect,
                         amdgpu::AMDGPUDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmRewritePattern, GridwiseGemmAccelRewritePattern,
               GridwiseAttentionAccelRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
