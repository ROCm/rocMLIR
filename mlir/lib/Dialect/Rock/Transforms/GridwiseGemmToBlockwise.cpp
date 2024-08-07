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
// This pass converts rock.gridwise_gemm[_v2] into block- and threadwise ops
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "GridLayoutEmitter.h"
#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

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
using mlir::gpu::AddressSpace;

namespace {
struct RockGridwiseGemmToBlockwisePass
    : public rock::impl::RockGridwiseGemmToBlockwisePassBase<
          RockGridwiseGemmToBlockwisePass> {
  void runOnOperation() override;
};

} // end anonymous namespace

/// Given a copy layout <copyDPerThread, copyKPerThread>, come up with the best
/// vectorization strategy for the layout. For instance, if the layout is <D,K>
/// = <2,16> and K is contiguous, we will vectorize by 16 along K and we will
/// loop over the other dimension
static std::pair<GemmDimension, int64_t>
bestGlobalVectorization(OpBuilder &b, Value matrix, int64_t copyDPerThread,
                        int64_t copyKPerThread, GemmDimension tiebreaker,
                        int64_t kPerBlock, int64_t dPerBlock) {
  // A future commit will account for the underlying buffer's vectorization
  // here.
  VectorizationResult kVectorRes = getMaxVectorization(
      matrix, static_cast<uint32_t>(GemmDimension::K), /*inputDimLen=*/
      math_util::gcd(copyKPerThread * copyDPerThread, kPerBlock),
      matrix.getDefiningOp());
  int64_t kVectorLen = kVectorRes.max;
  VectorizationResult dVectorRes = getMaxVectorization(
      matrix, static_cast<uint32_t>(GemmDimension::MorN), /*inputDimLen=*/
      math_util::gcd(copyDPerThread * copyKPerThread, dPerBlock),
      matrix.getDefiningOp());
  int64_t dVectorLen = dVectorRes.max;

  if (kVectorLen > dVectorLen) {
    kVectorLen = math_util::gcd(kVectorLen, copyKPerThread);
    return {GemmDimension::K, kVectorLen};
  }

  if (dVectorLen > kVectorLen) {
    dVectorLen = math_util::gcd(dVectorLen, copyDPerThread);
    return {GemmDimension::MorN, dVectorLen};
  }

  return {tiebreaker, kVectorLen};
}

/// Compute a thread copy layout, i.e., how many elements a single thread (or
/// workitem) reads along K and M (independently on how we vectorize the reads)
static FailureOr<std::pair<int64_t, int64_t>>
computeCopyPerThread(Type elementType, int64_t copyPerThread, int64_t kPerBlock,
                     int64_t dPerBlock, int64_t kpack, Location loc) {

  // By default, we try to maximize the LDS store vectorization. So we will try
  // to read as many elements as possible along the contiguous dimension in LDS
  // and `copyPerThread/elements` in the other dimension
  int64_t maxVlen = 128 / elementType.getIntOrFloatBitWidth();
  int64_t copyKPerThread = 0;
  int64_t copyDPerThread = 0;

  if (kpack == 1) {
    copyDPerThread = math_util::gcd(maxVlen, copyPerThread);
    copyKPerThread = copyPerThread / copyDPerThread;
  } else {
    copyKPerThread =
        math_util::gcd(maxVlen, math_util::gcd(kpack, copyPerThread));
    copyDPerThread = copyPerThread / copyKPerThread;
  }

  if (copyKPerThread == 0 || copyDPerThread == 0) {
    return emitError(loc) << "gemmA copy size too small,"
                          << " copyKPerThread: " << copyKPerThread
                          << " copyDPerThread: " << copyDPerThread << "\n";
  }
  if (kPerBlock < copyKPerThread || dPerBlock < copyDPerThread) {
    return mlir::emitError(loc)
           << "gemmA per thread copy smaller than per"
           << " block copy, incohereant tuning parameters\n";
  }
  return std::make_pair(copyKPerThread, copyDPerThread);
}

/// Wraps the LDS buffer "buffer", which is <kOuter * d * kpack *
/// sizeof(T) x i8> into a tid x iter view, where `iter` iterates over nominal
/// scalar indices into a buffer of type T. `buffer` will be reinterpreted as a
/// buffer with element type vector<kpackPerThread x T> (with kpackPerThread ==
/// 1 meaning just T). The resulting view must be iterated over with a stride of
/// no less than min(kPerThread, kpack). Also note that the `d` dimension
/// might be rotated to minimize bank conflicts (i.e., depending on
/// `rotateDWithK`
// we can apply a transformation similar to `d=(d+kOuter)%D`)
static FailureOr<Value> wrapLDSBufferForStore(OpBuilder &b, Location loc,
                                              Value buffer, Type ldsReadType,
                                              int64_t kOuter, StringRef dName,
                                              int64_t d, int64_t kPerThread,
                                              int64_t dPerThread,
                                              bool rotateDWithK = false) {
  MemRefType bufferType = cast<MemRefType>(buffer.getType());
  ArrayRef<int64_t> bufferShape = bufferType.getShape();
  Type dataType = ldsReadType;
  if (bufferShape.size() != 1)
    return emitError(loc, "Expected a flat buffer");
  int64_t kpack = 1;
  if (auto vectorDataType = dyn_cast<VectorType>(dataType)) {
    kpack = vectorDataType.getNumElements();
    dataType = vectorDataType.getElementType();
  }

  if (bufferShape[0] != kOuter * d * kpack * getByteWidth(dataType)) {
    return emitError(loc, "LDS buffer should have ")
           << kOuter * d * kpack * getByteWidth(dataType)
           << " elements but has " << bufferShape[0];
  }
  int64_t kpackPerThread = std::min(kPerThread, kpack);
  int64_t threadsPerKpack = kpack / kpackPerThread;

  Type ldsWriteType = vectorTypeOrSelf(dataType, kpackPerThread);
  auto typedBuffer = viewBufferAs(b, buffer, ldsWriteType);

  TopDownTMBuilder mergeKpack{
      b, {"k", "d"}, {kOuter * threadsPerKpack * kpackPerThread, d}};
  mergeKpack.merge({"k_outer", "kpack_idx", "kpack_vec"}, {0, 2, 3}, "k",
                   {kOuter, threadsPerKpack, kpackPerThread});
  mergeKpack.merge({dName}, {1}, "d", {d});

  TransformMapAttr mergeKpackAttr = mergeKpack.get();
  SmallVector<Attribute> transformAttrs{mergeKpackAttr};

  // Rotate the buffer if necessary to minimize bank conflicts. Rotating the
  // buffer has the benefit of minimizing bank conflicts when we are transposing
  // the matrix from global to LDS. I.e., instead of storing different items in
  // position (0,0), (1,0), (2,0), ... we store it in (0,0), (1,1), (2, 2), ...
  int64_t stride = (kpack == 1 ? dPerThread : 1);
  TopDownTMBuilder reshapeBuf = rotateIf(
      rotateDWithK, mergeKpack, mergeKpackAttr, stride, dName, d, 1, "k_outer",
      kOuter, {"k_outer"}, {"kpack_idx", "kpack_vec"}, transformAttrs);

  reshapeBuf.unmerge("raw", 0, {"k_outer", dName, "kpack_idx"},
                     {kOuter, d, threadsPerKpack});
  reshapeBuf.ignore("kpack_vec");
  TransformMapAttr reshapeBufAttr = reshapeBuf.get();
  transformAttrs.push_back(reshapeBufAttr);

  ArrayAttr asMatrix = b.getArrayAttr(transformAttrs);
  return transform(b, typedBuffer, asMatrix);
}

static LogicalResult checkLDSSize(Operation *op, int64_t aBufferBytes,
                                  int64_t bBufferBytes) {
  int64_t ldsBytes = aBufferBytes + bBufferBytes;
  // Check for arch limitations exceeded
  FailureOr<StringAttr> maybeArch = getArch(op);
  if (succeeded(maybeArch)) {
    StringAttr arch = maybeArch.value();
    const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
    return success(ldsBytes <= ldsSize);
  }
  return success();
}

// Following structures holds knobs to tweak the
// the LDS layout for gemms/attention ops.
struct LDSLayoutConfigDim {
  bool doRotateWithK;
  bool doSwapThreadIterSubDims;
};

// This is helper struct to aggregate
// derived information w.r.t load vectorization
struct VectorDimInfo {
  GemmDimension vectorDim;
  int64_t vectorLen;
  int64_t inKPerThread;
  int64_t inDPerThread;
  GemmDimension vectorTiebreaker;
};

static FailureOr<VectorDimInfo> getVectorDim(PatternRewriter &rewriter,
                                             Location loc, Value matrix,
                                             Type elemType, int64_t blockSize,
                                             int64_t kPerBlock,
                                             int64_t dPerBlock, int64_t kpack) {
  int64_t copyPerThread = (kPerBlock * dPerBlock) / blockSize;
  auto maybeCopyDPerThread = computeCopyPerThread(
      elemType, copyPerThread, kPerBlock, dPerBlock, kpack, loc);
  if (failed(maybeCopyDPerThread))
    return failure();

  int64_t copyKPerThread = (*maybeCopyDPerThread).first;
  int64_t copyDPerThread = (*maybeCopyDPerThread).second;
  // Find the best way of vectorizing the layout
  GemmDimension vectorTiebreaker =
      (kpack > 1) ? GemmDimension::K : GemmDimension::MorN;
  int64_t vectorLen;
  GemmDimension vectorDim;
  std::tie(vectorDim, vectorLen) =
      bestGlobalVectorization(rewriter, matrix, copyDPerThread, copyKPerThread,
                              vectorTiebreaker, kPerBlock, dPerBlock);
  return VectorDimInfo{vectorDim, vectorLen, copyKPerThread, copyDPerThread,
                       vectorTiebreaker};
}

static LDSLayoutConfigDim
getLDSLayoutConfigDim(Type elementType, int64_t kpack,
                      const VectorDimInfo &vecDimInfo) {
  LDSLayoutConfigDim cfg;
  int64_t maxVlen = 128 / elementType.getIntOrFloatBitWidth();
  int64_t copyDPerThread = vecDimInfo.inDPerThread;
  bool isKContigousDim = vecDimInfo.vectorDim == GemmDimension::K;
  // If kpack is less than the hardware max vector length, and we are
  // writing more contiguous kpack elements, there is a possibility to
  // vectorize that we want to preserve (i.e., we favour vectorization over
  // bank conflicts resolution)
  bool isPossibleToVectorizeD = (kpack < maxVlen && copyDPerThread > 1);
  cfg.doRotateWithK = isKContigousDim && !isPossibleToVectorizeD;
  cfg.doSwapThreadIterSubDims = !isKContigousDim && !isPossibleToVectorizeD;
  LLVM_DEBUG(llvm::dbgs() << "rotateWithK: " << cfg.doRotateWithK << "\n"
                          << "doSwapThreadIterSubDimsForM: "
                          << cfg.doSwapThreadIterSubDims << "\n");
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
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

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
        kpacksPerBlock * mPerBlock * kpack * getByteWidth(elementTypeA);
    int64_t ldsBlockBSize =
        kpacksPerBlock * nPerBlock * kpack * getByteWidth(elementTypeB);
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
    auto ldsByteBufferA = b.create<GpuAllocOp>(loc, ldsMemRefAType);
    auto ldsMemRefBType =
        MemRefType::get({ldsBlockBSize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsByteBufferB = b.create<GpuAllocOp>(loc, ldsMemRefBType);

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
        b.create<GpuAllocOp>(loc, threadCRegisterMemRefType);
    Value registerMatrixCViewOp = reshapeBuffer(
        b, loc, registerMatrixCAllocOp, {"m", "n"}, {threadCNumM, threadCNumN});

    // Zero init Matrix C on registers.
    b.create<FillOp>(loc, registerMatrixCAllocOp, zeroConstantFloatOp);

    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());
    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    if (!isValidBlockSize(blockSize, kPerBlock, mPerBlock, nPerBlock)) {
      return emitError(loc) << "Block size too large, rejecting as invalid.\n";
    }

    int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
    int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;

    FailureOr<VectorDimInfo> maybeVecDimInfoA =
        getVectorDim(b, loc, op.getA(), elementTypeA, blockSize, kPerBlock,
                     mPerBlock, kpack);
    if (failed(maybeVecDimInfoA)) {
      return failure();
    }
    FailureOr<VectorDimInfo> maybeVecDimInfoB =
        getVectorDim(b, loc, op.getB(), elementTypeB, blockSize, kPerBlock,
                     nPerBlock, kpack);
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
        maybeVecDimInfoA->vectorDim == GemmDimension::K);
    if (failed(maybeABufferViews)) {
      return failure();
    }
    Value wrappedA = transform(b, op.getA(), maybeABufferViews->gridSubTile);
    FailureOr<RegsAsMatrixSubTiles> maybeBBufferViews = getLoadRegsAsTileViews(
        b, loc, op.getB(), "n", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, nPerBlock, maybeVecDimInfoB->inKPerThread,
        maybeVecDimInfoB->inDPerThread,
        maybeVecDimInfoB->vectorDim == GemmDimension::K);
    if (failed(maybeBBufferViews)) {
      return failure();
    }
    Value wrappedB = transform(b, op.getB(), maybeBBufferViews->gridSubTile);

    Type loadBufferAType, loadBufferBType;
    loadBufferAType = MemRefType::get({aCopyPerThread}, elementTypeA,
                                      AffineMap{}, privateMemoryAddressSpace);
    loadBufferBType = MemRefType::get({bCopyPerThread}, elementTypeB,
                                      AffineMap{}, privateMemoryAddressSpace);

    auto loadBufferA = b.create<GpuAllocOp>(loc, loadBufferAType);
    auto loadBufferB = b.create<GpuAllocOp>(loc, loadBufferBType);

    // Compute grid coordinates
    FailureOr<mlir::StringAttr> maybeArch = getArch(op);
    if (failed(maybeArch)) {
      return op.emitError("arch needs to be set.");
    }
    auto gridCoords = layout::makeGroupedGridLayout(
        b, loc, bid,
        {G, mBlocks, nBlocks, op.getNumCU(), elementTypeA, destType},
        maybeArch->getValue());
    b.create<ThreadwiseReadIntoOp>(
        loc, wrappedA, loadBufferA, /*extraViews=*/b.getArrayAttr({}),
        /*extraIndices=*/
        ValueRange{/*kIter=*/zeroConstantOp, gridCoords.g_block,
                   gridCoords.m_block, gridCoords.n_block, tid},
        true, true);
    b.create<ThreadwiseReadIntoOp>(
        loc, wrappedB, loadBufferB, /*extraViews=*/b.getArrayAttr({}),
        /*extraIndices=*/
        ValueRange{/*kIter=*/zeroConstantOp, gridCoords.g_block,
                   gridCoords.m_block, gridCoords.n_block, tid},
        true, true);

    Value storeBufferA = b.create<GpuAllocOp>(loc, loadBufferA.getType());
    Value storeBufferB = b.create<GpuAllocOp>(loc, loadBufferB.getType());

    LDSLayoutConfigDim ldsLayoutConfigA =
        getLDSLayoutConfigDim(elementTypeA, kpack, maybeVecDimInfoA.value());
    LDSLayoutConfigDim ldsLayoutConfigB =
        getLDSLayoutConfigDim(elementTypeB, kpack, maybeVecDimInfoB.value());

    // We invert the transforms that are iter --> K x D slice of the tensor
    // so that we can view loadBuffer as a K x D tensor
    ArrayAttr loadBufferAViews =
        invertTransforms(b, loc, maybeABufferViews->threadSubTile);
    Value viewLoadBufferA = transform(b, loadBufferA, loadBufferAViews);
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
    ArrayAttr storeBufferAViews =
        invertTransforms(b, loc, maybeALdsStoreViews->threadSubTile);
    Value viewStoreBufferA = transform(b, storeBufferA, storeBufferAViews);
    auto packALoop = b.create<ThreadwiseCopyOp>(
        loc, viewLoadBufferA, ValueRange{}, viewStoreBufferA, ValueRange{},
        useIndexDiffs, true);
    ArrayAttr loadBufferBViews =
        invertTransforms(b, loc, maybeBBufferViews->threadSubTile);
    Value viewLoadBufferB = transform(b, loadBufferB, loadBufferBViews);
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
    ArrayAttr storeBufferBViews =
        invertTransforms(b, loc, maybeBLdsStoreViews->threadSubTile);
    Value viewStoreBufferB = transform(b, storeBufferB, storeBufferBViews);
    auto packBLoop = b.create<ThreadwiseCopyOp>(
        loc, viewLoadBufferB, ValueRange{}, viewStoreBufferB, ValueRange{},
        useIndexDiffs, true);

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

    ThreadwiseWriteAllOp blockwiseStoreA = b.create<ThreadwiseWriteAllOp>(
        loc, storeBufferA, wrappedLdsA,
        /*extraViews=*/b.getArrayAttr({}),
        /*extraIndices=*/ValueRange{tid}, op.getFeatures(), StoreMethod::Set,
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    ThreadwiseWriteAllOp blockwiseStoreB = b.create<ThreadwiseWriteAllOp>(
        loc, storeBufferB, wrappedLdsB,
        /*extraViews=*/b.getArrayAttr({}),
        /*extraIndices=*/ValueRange{tid}, op.getFeatures(), StoreMethod::Set,
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);

    // The blockwise gemm isn't set up for vector-of-kpack loads and so expects
    // a scalar kpacksPerBlock x dPerBlock x kpack x T buffer unconditionally.
    Value ldsMatrixA = viewBufferAs(b, ldsByteBufferA, elementTypeA);
    ldsMatrixA = reshapeBuffer(b, loc, ldsMatrixA, {"k", "m", "kpack"},
                               {kpacksPerBlock, mPerBlock, kpack});
    Value ldsMatrixB = viewBufferAs(b, ldsByteBufferB, elementTypeB);
    ldsMatrixB = reshapeBuffer(b, loc, ldsMatrixB, {"k", "n", "kpack"},
                               {kpacksPerBlock, nPerBlock, kpack});

    // Emit loop.
    int64_t nIterations = K / kPerBlock;
    BlockwiseGemmOp blockwiseGemmOp;
    // Start at 1 to make it clearer we have performed software pipelining.
    auto loopOp = b.create<affine::AffineForOp>(loc, 1, nIterations, 1);
    {
      // inside the loop.
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());

      Value iv = loopOp.getInductionVar();
      b.create<ThreadwiseReadIntoOp>(
          loc, wrappedA, loadBufferA, /*extraViews=*/b.getArrayAttr({}),
          /*extraIndices=*/
          ValueRange{/*kIter=*/iv, gridCoords.g_block, gridCoords.m_block,
                     gridCoords.n_block, tid},
          true, true);
      b.create<ThreadwiseReadIntoOp>(
          loc, wrappedB, loadBufferB, /*extraViews=*/b.getArrayAttr({}),
          /*extraIndices=*/
          ValueRange{/*kIter=*/iv, gridCoords.g_block, gridCoords.m_block,
                     gridCoords.n_block, tid},
          true, true);

      // LDS barrier.
      b.create<LDSBarrierOp>(loc);

      // Emit blockwise GEMM.
      blockwiseGemmOp = b.create<BlockwiseGemmOp>(
          loc, ldsMatrixA, ldsMatrixB, registerMatrixCViewOp,
          b.getI32IntegerAttr(maybeVecDimInfoA->inDPerThread),
          b.getI32IntegerAttr(maybeVecDimInfoB->inDPerThread),
          ldsLayoutConfigA.doRotateWithK ? b.getUnitAttr() : nullptr,
          ldsLayoutConfigB.doRotateWithK ? b.getUnitAttr() : nullptr,
          op.getParamsAttr());

      // LDS barrier.
      // This barrier prevents halo part of outputs having weird values.
      b.create<LDSBarrierOp>(loc);

      // Packing step
      b.clone(*packALoop.getOperation());
      b.clone(*packBLoop.getOperation());

      // Emit blockwise stores
      b.clone(*blockwiseStoreA.getOperation());
      b.clone(*blockwiseStoreB.getOperation());
    }
    // outside the loop.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    IRMapping tailGemmCloneMap;
    b.clone(*blockwiseGemmOp, tailGemmCloneMap);

    // Apparently, the canonicalizer doesn't get rid of empty loops without
    // results properly, remove them ourselves.
    if (nIterations <= 1)
      b.eraseOp(loopOp);

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

    swapThreadIdAndIteration(
        toMatrixC, /*mBlocks=*/bidGridLengths[1],
        /*nBlocks=*/bidGridLengths[2], maybeVecDimInfoA->inDPerThread,
        maybeVecDimInfoB->inDPerThread, mPerBlock, nPerBlock,
        ldsLayoutConfigA.doSwapThreadIterSubDims,
        ldsLayoutConfigB.doSwapThreadIterSubDims,
        /*isBlockwise=*/false, transformAttrs);

    Value registerC = registerMatrixCAllocOp;
    ArrayAttr idToMatrixCMaps = b.getArrayAttr(transformAttrs);
    b.create<ThreadwiseWriteAllOp>(loc, registerC, op.getC(), idToMatrixCMaps,
                                   /*extraIndices=*/
                                   ValueRange{gridCoords.g_block,
                                              gridCoords.m_block,
                                              gridCoords.n_block, tid},
                                   op.getFeatures(), op.getStoreMethod(),
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
    ArrayAttr storeBufferViews =
        invertTransforms(rewriter, loc, toLDSViews.threadSubTile);
    Value viewStoreBuffer = transform(rewriter, storeBuffer, storeBufferViews);
    // The following is fine for software pipelining optimization as it could be
    // considered "compute". In future, consider refactoring the following loop
    // to be a single reg->reg op avoid verbose IR at this level.
    rewriter.create<ThreadwiseCopyOp>(loc, regBuffer, ValueRange{},
                                      viewStoreBuffer, ValueRange{}, false,
                                      false);
    Type ldsReadType = vectorTypeOrSelf(elemType, kpack);
    FailureOr<Value> maybeWrappedLds = wrapLDSBufferForStore(
        rewriter, loc, ldsTileByteBuffer, ldsReadType, kpacksPerBlock,
        nonKDimName, dPerBlock, copyKPerThread, copyDPerThread, rotateDWithK);
    if (failed(maybeWrappedLds)) {
      return failure();
    }
    // This is KxD view of the flat LDS buffer
    Value wrappedLds = std::move(*maybeWrappedLds);
    // This will produce a (tid, iter) --> flat LDS view
    wrappedLds = transform(rewriter, wrappedLds, toLDSViews.blockSubTile);
    auto tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
    rewriter.create<ThreadwiseWriteAllOp>(
        loc, storeBuffer, wrappedLds, /*extraViews=*/rewriter.getArrayAttr({}),
        /*extraIndices=*/ValueRange{tid}, GemmFeatures::none, StoreMethod::Set,
        forceUnroll, true);
    return success();
  }

  // This function will process a tile of gemm input into LDS buffer
  // in a way it could be fed to blockwise_gemm_accel op
  LogicalResult loadAndStoreGemmInputTile(
      Location loc, Value in, Value kIter,
      rock::layout::GridCoordinates gridCoords, Value fromGlobalRegBuffer,
      Value toLDSRegBuffer, Value destBuffer, StringRef nonKDimName,
      int64_t kpack, int64_t kpacksPerBlock, int64_t dPerBlock,
      uint32_t blockSize, uint32_t gridSize, ArrayRef<StringRef> bidGridOrder,
      ArrayRef<int64_t> bidGridLengths, bool forceUnroll,
      PatternRewriter &rewriter, const accel::AccelEmitter &accelEmitter,
      LDSLayoutConfigDim ldsLayoutCfg) const {

    MemRefType destBufferType = cast<MemRefType>(destBuffer.getType());
    mlir::gpu::AddressSpace destBufferAddrSpace =
        cast<gpu::AddressSpaceAttr>(destBufferType.getMemorySpace()).getValue();
    bool isDestBufferLDS = destBufferAddrSpace == gpu::AddressSpace::Workgroup;
    if (!isDestBufferLDS && destBufferAddrSpace != gpu::AddressSpace::Private) {
      return emitError(loc) << "the destination buffer to load global input "
                               "tile should either be LDS or Regs.\n";
    }

    int64_t kPerBlock = kpacksPerBlock * kpack;
    int64_t copyPerThread = (kPerBlock * dPerBlock) / blockSize;
    int64_t kGlobal = cast<MemRefType>(in.getType()).getShape()[1];
    int64_t kIters = kGlobal / kPerBlock;
    Type elemType = cast<MemRefType>(in.getType()).getElementType();
    if (copyPerThread == 0) {
      return emitError(loc) << "Block size too large, rejecting as invalid.\n";
    }
    FailureOr<VectorDimInfo> maybeVectorDimInfo = getVectorDim(
        rewriter, loc, in, elemType, blockSize, kPerBlock, dPerBlock, kpack);
    if (failed(maybeVectorDimInfo)) {
      return failure();
    }
    int64_t vectorLen = maybeVectorDimInfo->vectorLen;
    GemmDimension vectorDim = maybeVectorDimInfo->vectorDim;
    FailureOr<RegsAsMatrixSubTiles> maybeInBufferViews;
    if (!isDestBufferLDS) {
      maybeInBufferViews = accelEmitter.createAccelGemmOperandTransforms(
          rewriter, loc, kIters, bidGridLengths, blockSize,
          maybeVectorDimInfo->inDPerThread, nonKDimName,
          vectorDim == GemmDimension::K, false);
    } else {
      maybeInBufferViews = getLoadRegsAsTileViews(
          rewriter, loc, in, nonKDimName, bidGridOrder, bidGridLengths,
          blockSize, kPerBlock, dPerBlock, maybeVectorDimInfo->inKPerThread,
          maybeVectorDimInfo->inDPerThread, vectorDim == GemmDimension::K);
    }
    if (failed(maybeInBufferViews)) {
      return failure();
    }
    Value viewIn = transform(rewriter, in, maybeInBufferViews->gridSubTile);
    auto tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
    rewriter.create<ThreadwiseReadIntoOp>(
        loc, viewIn, fromGlobalRegBuffer,
        /*extraViews=*/rewriter.getArrayAttr({}),
        ValueRange{kIter, gridCoords.g_block, gridCoords.m_block,
                   gridCoords.n_block, tid},
        forceUnroll, true);
    if (isDestBufferLDS) {
      // threadwiseView is iter --> K,D
      // Hence we invert to create the reg buffer to be viewed
      // as K x D memref
      ArrayAttr loadBufferViews =
          invertTransforms(rewriter, loc, maybeInBufferViews->threadSubTile);
      Value viewLoadBuffer =
          transform(rewriter, fromGlobalRegBuffer, loadBufferViews);

      FailureOr<RegsAsMatrixSubTiles> maybeLdsStoreViews =
          getPackedRegsAsTileViews(rewriter, loc, in, nonKDimName, bidGridOrder,
                                   bidGridLengths, blockSize, kPerBlock,
                                   dPerBlock, maybeVectorDimInfo->inKPerThread,
                                   maybeVectorDimInfo->inDPerThread, kpack,
                                   vectorDim == GemmDimension::K,
                                   ldsLayoutCfg.doSwapThreadIterSubDims);
      if (failed(maybeLdsStoreViews)) {
        return failure();
      }
      LogicalResult storeGemmTileStatus = storeGemmInputTile(
          rewriter, loc, kpack, viewLoadBuffer, maybeLdsStoreViews.value(),
          toLDSRegBuffer, destBuffer, kpacksPerBlock, nonKDimName, kPerBlock,
          dPerBlock, maybeVectorDimInfo->inKPerThread,
          maybeVectorDimInfo->inDPerThread, forceUnroll,
          ldsLayoutCfg.doRotateWithK);
      if (failed(storeGemmTileStatus)) {
        return failure();
      }
    } else {
      accel::AccelEmitterParams accelEmitterParams = accelEmitter.getParams();
      int64_t dRepeats = (nonKDimName == "m" ? accelEmitterParams.mRepeats
                                             : accelEmitterParams.nRepeats);
      affine::AffineForOp dRepeatsLoop =
          rewriter.create<affine::AffineForOp>(loc, 0, dRepeats, 1);
      {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(dRepeatsLoop.getBody());
        Value di = dRepeatsLoop.getInductionVar();
        Value subview = destBuffer;
        if (dRepeats > 1) {
          subview = createSliceOfFirstDim(rewriter, loc, destBuffer, di);
        }
        // InBufferViews provide --> K x D subtile views.
        // Since we are iterating on D dimension, we need to transpose it.
        RegsAsMatrixSubTiles inBufferViewsTr =
            transposeSubTileViews(rewriter, loc, maybeInBufferViews.value());
        Value viewLoadedBuffer = transform(
            rewriter, fromGlobalRegBuffer,
            invertTransforms(rewriter, loc, inBufferViewsTr.threadSubTile));
        rewriter.create<ThreadwiseReadIntoOp>(loc, viewLoadedBuffer, subview,
                                              rewriter.getArrayAttr({}),
                                              ValueRange{di}, true, true);
      }
    }
    return success();
  }

  Value createLDSByteBuffer(PatternRewriter &rewriter, Location loc,
                            int64_t numElements, Type elemType) const {
    auto workgroupMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    int64_t ldsBlockSize = numElements * getByteWidth(elemType);
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, rewriter.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    Value ldsByteBuffer = rewriter.create<GpuAllocOp>(loc, ldsMemRefType);
    return ldsByteBuffer;
  }

  std::tuple<SmallVector<Value>, int64_t>
  createSharedLDSByteBufferRefs(PatternRewriter &rewriter, Location loc,
                                ArrayRef<int64_t> numElementsArr,
                                ArrayRef<Type> elemTypeArr) const {
    auto workgroupMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    int64_t maxSizeBytes = 0;
    SmallVector<int64_t, 4> byteSizes;
    for (auto [numElements, elemType] :
         llvm::zip(numElementsArr, elemTypeArr)) {
      int64_t sizeBytes = numElements * getByteWidth(elemType);
      if (sizeBytes > maxSizeBytes) {
        maxSizeBytes = sizeBytes;
      }
      byteSizes.push_back(sizeBytes);
    }
    auto ldsMemRefType =
        MemRefType::get({maxSizeBytes}, rewriter.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    Value ldsByteBuffer = rewriter.create<GpuAllocOp>(loc, ldsMemRefType);

    SmallVector<Value> ret;
    for (int64_t byteSize : byteSizes) {
      if (byteSize == 0) {
        ret.push_back(Value());
        continue;
      }
      if (byteSize == maxSizeBytes) {
        ret.push_back(ldsByteBuffer);
        continue;
      }
      auto byteBufferType =
          MemRefType::get({byteSize}, rewriter.getI8Type(), AffineMap{},
                          workgroupMemoryAddressSpace);
      Value ldsByteBufferSubView = rewriter.create<memref::SubViewOp>(
          loc, byteBufferType, ldsByteBuffer, ArrayRef<int64_t>{0},
          ArrayRef<int64_t>{byteSize}, ArrayRef<int64_t>{1});
      ret.push_back(ldsByteBufferSubView);
    }
    return {ret, maxSizeBytes};
  }

  // This function will create fromGlobalRegsBuffer, toLDSRegBuffer and
  // ldsTileBuffer for a gemm input
  std::tuple<Value, Value>
  createRegBuffersForGemmIn(Location loc, int64_t kPerBlock, int64_t blockSize,
                            Type elemType, int64_t dPerBlock,
                            PatternRewriter &rewriter) const {
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    int64_t copyPerThread = (kPerBlock * dPerBlock) / blockSize;
    Type loadBufferType = MemRefType::get(
        {copyPerThread}, elemType, AffineMap{}, privateMemoryAddressSpace);
    Value fromGlobalRegBuffer =
        rewriter.create<GpuAllocOp>(loc, loadBufferType);
    Value toLDSRegBuffer = rewriter.create<GpuAllocOp>(loc, loadBufferType);
    return {fromGlobalRegBuffer, toLDSRegBuffer};
  }

  void zeroAccBuffer(PatternRewriter &rewriter, Location loc,
                     Value accBuffer) const {
    MemRefType accBufferType = cast<MemRefType>(accBuffer.getType());
    Value zeroConstantCOp =
        createZeroConstantOp(rewriter, loc, accBufferType.getElementType());
    rewriter.create<FillOp>(loc, accBuffer, zeroConstantCOp);
  }

  // This function creates the accumulator register buffer
  Value createBufferForAccelGemmOut(Location loc,
                                    rock::accel::AccelEmitterParams params,
                                    PatternRewriter &rewriter,
                                    int64_t numBuffers = 1) const {
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    int64_t nResultVectors = params.nResultVectors;
    int64_t mRepeats = params.mRepeats;
    int64_t nRepeats = params.nRepeats;
    VectorType accVectorType = params.accVectorType;
    int64_t nOutputVectors = nResultVectors * mRepeats * nRepeats;
    MemRefType regCAllocType;
    if (numBuffers > 1) {
      regCAllocType = MemRefType::get(
          {numBuffers, nOutputVectors}, accVectorType, AffineMap{},
          /*memorySpace=*/privateMemoryAddressSpace);
    } else {
      regCAllocType =
          MemRefType::get(nOutputVectors, accVectorType, AffineMap{},
                          /*memorySpace=*/privateMemoryAddressSpace);
    }
    Value regCAllocOp = rewriter.create<rock::GpuAllocOp>(loc, regCAllocType);
    return regCAllocOp;
  }

  // This function creates a simple scalar reg buffer (i.e. without vectors)
  Value createBufferForGemmOut(Location loc, Type gemmOutElemType,
                               rock::accel::AccelEmitterParams params,
                               PatternRewriter &rewriter,
                               int64_t numBuffers = 1) const {
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
        rewriter.create<rock::GpuAllocOp>(loc, gemmOutScalarBufferType);
    return gemmOutScalarBuffer;
  }

  // This fuction creates interrim register buffers to store data in once
  // loaded from the LDS before accelerator intrinsics are called
  std::tuple<Value, Value> createRegInterrimBufferForAccel(
      Location loc, rock::accel::AccelEmitterParams params,
      PatternRewriter &rewriter, int64_t mRepeats = 1,
      int64_t nRepeats = 1) const {
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    int64_t kBasePerThread = params.kBasePerThread;

    SmallVector<Value> arrayARegs;
    Type argTypeA = params.argTypeA;
    SmallVector<int64_t, 2> aShape{kBasePerThread};
    if (mRepeats > 1) {
      aShape.insert(aShape.begin(), mRepeats);
    }
    auto arrayAType = MemRefType::get(aShape, argTypeA, AffineMap{},
                                      privateMemoryAddressSpace);
    auto arrayA = rewriter.create<GpuAllocOp>(loc, arrayAType);

    SmallVector<Value> arrayBRegs;
    Type argTypeB = params.argTypeB;
    SmallVector<int64_t, 2> bShape{kBasePerThread};
    if (nRepeats > 1) {
      bShape.insert(bShape.begin(), nRepeats);
    }
    auto arrayBType = MemRefType::get(bShape, argTypeB, AffineMap{},
                                      privateMemoryAddressSpace);
    auto arrayB = rewriter.create<GpuAllocOp>(loc, arrayBType);
    return {arrayA, arrayB};
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
    auto loop = rewriter.create<TransformingForOp>(
        loc,
        ArrayRef<ValueRange>{
            {zero, zero}, {zero, zero}, {zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), gemm0OutBufferMaxTrs,
                            gemm0OutExpTrs, gemm0OutTrs},
        /*bounds=*/ArrayRef<int64_t>{g0Mpt, g0Npt},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*useIndexDiffs=*/true, /*forceUnroll=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Block::BlockArgListType upperCoords = loop.getLowerCoords(0);
      Block::BlockArgListType gemm0OutBufferMaxCoords = loop.getLowerCoords(1);
      Block::BlockArgListType gemm0OutExpCoords = loop.getLowerCoords(2);
      Block::BlockArgListType gemm0OutCoords = loop.getLowerCoords(3);

      // maxRowBufferNew = max(maxRowBuffer, gemm0OutBufferMaxView[:,0])
      Type maxRowBufferElemType = getElementTypeOrSelf(maxRowBuffer.getType());
      Value ldMaxRowBuffer = rewriter.create<InBoundsLoadOp>(
          loc, maxRowBufferElemType, maxRowBuffer, ValueRange{upperCoords[0]});
      Value ldgemm0OutBufferMax = rewriter.create<InBoundsLoadOp>(
          loc, maxRowBufferElemType, gemm0OutBufferMax,
          gemm0OutBufferMaxCoords);
      Value maxRowBufferNew = rewriter.create<arith::MaximumFOp>(
          loc, ldMaxRowBuffer, ldgemm0OutBufferMax);

      // ldGemm0OutSubMaxExp = exp(gemm0Out  -maxRowBufferNew)
      Type ldGemm0OutElemType = getElementTypeOrSelf(gemm0Out.getType());
      Value ldGemm0Out = rewriter.create<InBoundsLoadOp>(
          loc, ldGemm0OutElemType, gemm0Out, gemm0OutCoords);
      Value ldGemm0OutSubMax =
          rewriter.create<arith::SubFOp>(loc, ldGemm0Out, maxRowBufferNew);
      Value ldGemm0OutSubMaxExp =
          rewriter.create<math::Exp2Op>(loc, ldGemm0OutSubMax);

      // Store back to gemm0Out
      rewriter.create<InBoundsStoreOp>(loc, ldGemm0OutSubMaxExp, gemm0OutExp,
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
    int64_t g0Mpt = gemm0OutViewType.getShape()[0];
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    auto loop = rewriter.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{{zero, zero}, {zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), gemm0OutBufferSumTrs,
                            gemm0OutBufferMaxTrs},
        /*bounds=*/ArrayRef<int64_t>{g0Mpt, 1},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*useIndexDiffs=*/true, /*forceUnroll=*/true);
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
      Value ldSumRowBuffer = rewriter.create<InBoundsLoadOp>(
          loc, sumRowBufferElemType, sumRowBuffer, ValueRange{upperCoords[0]});
      Value ldgemm0OutBufferSum = rewriter.create<InBoundsLoadOp>(
          loc, sumRowBufferElemType, gemm0OutBufferSum,
          gemm0OutBufferSumCoords);
      // sumRowBufferNew0 = exp(maxRowBuffer - maxRowBufferNew) * sumRowBuffer
      Type maxRowBufferElemType = getElementTypeOrSelf(maxRowBuffer.getType());
      Value ldMaxRowBuffer = rewriter.create<InBoundsLoadOp>(
          loc, maxRowBufferElemType, maxRowBuffer, ValueRange{upperCoords[0]});
      Value ldgemm0OutBufferMax = rewriter.create<InBoundsLoadOp>(
          loc, maxRowBufferElemType, gemm0OutBufferMax,
          gemm0OutBufferMaxCoords);
      Value maxRowBufferNew = rewriter.create<arith::MaximumFOp>(
          loc, ldMaxRowBuffer, ldgemm0OutBufferMax);
      Value maxRowDiff =
          rewriter.create<arith::SubFOp>(loc, ldMaxRowBuffer, maxRowBufferNew);
      Value maxRowDiffExp = rewriter.create<math::Exp2Op>(loc, maxRowDiff);
      rewriter.create<InBoundsStoreOp>(loc, maxRowDiffExp, expMaxDiffRowBuffer,
                                       ValueRange{upperCoords[0]});
      Value sumRowBufferNew = maxRowDiffExp;
      sumRowBufferNew =
          rewriter.create<arith::MulFOp>(loc, sumRowBufferNew, ldSumRowBuffer);
      sumRowBufferNew = rewriter.create<arith::AddFOp>(loc, sumRowBufferNew,
                                                       ldgemm0OutBufferSum);
      rewriter.create<InBoundsStoreOp>(loc, sumRowBufferNew, sumRowBuffer,
                                       ValueRange{upperCoords[0]});
      rewriter.create<InBoundsStoreOp>(loc, maxRowBufferNew, maxRowBuffer,
                                       ValueRange{upperCoords[0]});
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
    int64_t g1Mpt = attentionOutAccViewType.getShape()[0];
    int64_t g1Npt = attentionOutAccViewType.getShape()[1];
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    auto loop = rewriter.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{{zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), attentionOutAccTrs},
        /*bounds=*/ArrayRef<int64_t>{g1Mpt, g1Npt},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*useIndexDiffs=*/true, /*forceUnroll=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Block::BlockArgListType upperCoords = loop.getLowerCoords(0);
      Block::BlockArgListType attentionOutAccBufferCoords =
          loop.getLowerCoords(1);
      Value ldAttentionOutAccBuffer = rewriter.create<InBoundsLoadOp>(
          loc, outElemType, attentionOutAccBuffer, attentionOutAccBufferCoords);
      Type sumRowBufferElemType = getElementTypeOrSelf(sumRowBuffer.getType());
      Value ldSumRowBuffer = rewriter.create<InBoundsLoadOp>(
          loc, sumRowBufferElemType, sumRowBuffer, ValueRange{upperCoords[0]});
      Value stAttentionOutAccBuffer = rewriter.create<arith::DivFOp>(
          loc, ldAttentionOutAccBuffer, ldSumRowBuffer);
      rewriter.create<InBoundsStoreOp>(loc, stAttentionOutAccBuffer,
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
    int64_t g1Mpt = attentionOutAccBufferType.getShape()[0];
    int64_t g1Npt = attentionOutAccBufferType.getShape()[1];

    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);

    auto loop = rewriter.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{{zero, zero}, {zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{rewriter.getArrayAttr({}), gemm1OutTrs,
                            attentionOutAccBufferTrs},
        /*bounds=*/ArrayRef<int64_t>{g1Mpt, g1Npt},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*useIndexDiffs=*/true, /*forceUnroll=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());

      Block::BlockArgListType upperCoords = loop.getLowerCoords(0);
      Block::BlockArgListType gemm1OutCoords = loop.getLowerCoords(1);
      Block::BlockArgListType attentionOutAccBufferCoords =
          loop.getLowerCoords(2);

      Type expMaxDiffRowBufferElemType =
          getElementTypeOrSelf(expMaxDiffRowBuffer.getType());
      Value maxRowDiffExp = rewriter.create<InBoundsLoadOp>(
          loc, expMaxDiffRowBufferElemType, expMaxDiffRowBuffer,
          ValueRange{upperCoords[0]});
      Value ldAttentionOutAccBuffer = rewriter.create<InBoundsLoadOp>(
          loc, outElemType, attentionOutAccBuffer, attentionOutAccBufferCoords);
      Value scaledldAttentionOutAccBuffer = rewriter.create<arith::MulFOp>(
          loc, ldAttentionOutAccBuffer, maxRowDiffExp);

      Value ldGemm1Out = rewriter.create<InBoundsLoadOp>(
          loc, outElemType, gemm1Out, gemm1OutCoords);
      Value stAttentionOutAccBuffer = rewriter.create<arith::AddFOp>(
          loc, scaledldAttentionOutAccBuffer, ldGemm1Out);
      rewriter.create<InBoundsStoreOp>(loc, stAttentionOutAccBuffer,
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
  // post normalization. Therefore, this function creates a trasnforming
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
        -std::numeric_limits<float>::infinity());
    // Get current workitem ID.
    auto tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
    int64_t elementsInThreadBuffer = gemm0OutBufferType.getNumElements();
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    auto loop = rewriter.create<TransformingForOp>(
        loc,
        ArrayRef<ValueRange>{{gridCoords.g_block, gridCoords.m_block,
                              gridCoords.n_block, tid, zero},
                             {zero, zero, zero, zero, zero}},
        ArrayRef<Attribute>{gemm0OutSubTileViews.gridSubTile,
                            rewriter.getArrayAttr({})},
        /*bounds=*/ArrayRef<int64_t>{1, 1, 1, 1, elementsInThreadBuffer},
        /*strides=*/ArrayRef<int64_t>{1, 1, 1, 1, 1},
        /*useIndexDiffs=*/true, /*forceUnroll=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());

      Block::BlockArgListType upperCoords = loop.getLowerCoords(1);
      TypedValue<IntegerType> isValid = loop.getValidity(0);
      Value zeroBit = createConstantIntOp(rewriter, loc, isValid.getType(),
                                          isValid.getType(), 0);
      auto isInvalid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, isValid, zeroBit);
      scf::IfOp ifb = rewriter.create<scf::IfOp>(loc, isInvalid,
                                                 /*withElseRegion=*/false);
      {
        OpBuilder thenb = ifb.getThenBodyBuilder();
        thenb.create<InBoundsStoreOp>(loc, negInfTyped, gemm0OutBuffer,
                                      ValueRange{upperCoords[4]});
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
    rewriter.create<linalg::GenericOp>(
        loc, ValueRange(gemm0OutBuffer), ValueRange(gemm0OutBuffer),
        indexingMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value splatScalarConst = nestedBuilder.create<arith::ConstantOp>(
              loc, bufType.getElementType(), splatVal);
          Value elementwiseOp;
          if (bufType.getElementType().isIntOrIndex()) {
            elementwiseOp =
                nestedBuilder.create<typename ElementwiseOpType::Int>(
                    loc, args[0], splatScalarConst);
          } else {
            elementwiseOp =
                nestedBuilder.create<typename ElementwiseOpType::Float>(
                    loc, args[0], splatScalarConst);
          }
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, elementwiseOp);
        });
  }

  FailureOr<Value> postProcessFirstGemm(
      PatternRewriter &rewriter, Location loc, GridwiseAttentionAccelOp op,
      layout::GridCoordinates gridCoords, Value srcGemm0OutBuffer,
      Value destGemm0OutBuffer, RegsAsMatrixSubTiles gemm0OutViews) const {
    LogicalResult res = success();
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    bool linalgOpFound = false;
    op.getPreSoftmaxBody().walk([&](linalg::GenericOp genOp) {
      linalgOpFound = true;
      auto tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
      SmallVector<Value> inputTileBuffers;
      inputTileBuffers.push_back(srcGemm0OutBuffer);
      MemRefType srcBufType = cast<MemRefType>(srcGemm0OutBuffer.getType());

      // Pull non-identiy index maps to rock transforms
      res = makeLinalgGenericWithIdentityAffMaps(rewriter, genOp);

      // Obtain transform stack from gemmOutput to linalg generic input.
      ArrayAttr linalgToGemmOutMaps;
      std::tie(std::ignore, linalgToGemmOutMaps, std::ignore) =
          untransform(rewriter, genOp.getInputs()[0]);
      // The obtained transforms will be linalg generic being the upperview
      // leading to gemmOutput being the lowerview. However, we need to
      // construct
      //  the following sequence :
      //  (bid, tid, iter) > ... > [gemmOutput: k x d]
      //                         > invertTr(linalg input to gemmOutput maps)
      //                         > (linalgOtherInput to op arg maps)
      ArrayAttr linalgGridSubTileMaps = gemm0OutViews.gridSubTile;
      ArrayAttr GemmOutToLinalgMaps =
          invertTransforms(rewriter, loc, linalgToGemmOutMaps);
      if (!GemmOutToLinalgMaps.empty()) {
        linalgGridSubTileMaps = prependUpperViews(
            rewriter, linalgGridSubTileMaps, GemmOutToLinalgMaps);
      }

      for (auto [idx, otherInput] :
           llvm::enumerate(op.getPreSoftmaxElemWiseInputs())) {
        MemRefType otherInputBufType = cast<MemRefType>(otherInput.getType());
        MemRefType tileBufType = MemRefType::get(
            srcBufType.getShape(), otherInputBufType.getElementType(),
            AffineMap{}, privateMemoryAddressSpace);
        auto tileBuffer = rewriter.create<rock::GpuAllocOp>(loc, tileBufType);
        auto genOpInput = genOp.getInputs()[idx + 1];
        ArrayAttr linalgToOtherInputMaps;
        std::tie(std::ignore, linalgToOtherInputMaps, std::ignore) =
            untransform(rewriter, genOpInput);
        ArrayAttr GemmOutToOtherInputMaps = linalgGridSubTileMaps;
        if (!linalgToOtherInputMaps.empty()) {
          GemmOutToOtherInputMaps = prependUpperViews(
              rewriter, linalgGridSubTileMaps, linalgToOtherInputMaps);
        }
        rewriter.create<ThreadwiseReadIntoOp>(
            loc, otherInput, tileBuffer, GemmOutToOtherInputMaps,
            ValueRange{gridCoords.g_block, gridCoords.m_block,
                       gridCoords.n_block, tid},
            true, true);
        inputTileBuffers.push_back(tileBuffer);
      }
      // Output is overwriting the same input buffer
      inputTileBuffers.push_back(destGemm0OutBuffer);
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
    });
    if (failed(res)) {
      return op.emitError("pre softmax linalg regularization failed.\n");
    }
    if (!linalgOpFound) {
      return srcGemm0OutBuffer;
    }
    return destGemm0OutBuffer;
  }

  void loadGemmOperandsFromLDSToRegs(PatternRewriter &rewriter, Location loc,
                                     Value ldsTileBuffer,
                                     Value preAccelRegBuffer, StringRef dName,
                                     int64_t blockSize, int64_t inDPerThread,
                                     const accel::AccelEmitter &accelEmitterPtr,
                                     bool rotateDWithK) const {
    // Get current workitem ID.
    auto tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
    rock::accel::AccelEmitterParams accelParams = accelEmitterPtr.getParams();
    Value wrappedLDSBufferForLoad = accelEmitterPtr.wrapLDSBufferForLoad(
        rewriter, loc, ldsTileBuffer, blockSize, inDPerThread, dName,
        rotateDWithK);
    int64_t repeats =
        dName == "m" ? accelParams.mRepeats : accelParams.nRepeats;
    affine::AffineForOp mRepeatsLoop =
        rewriter.create<affine::AffineForOp>(loc, 0, repeats, 1);
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mRepeatsLoop.getBody());
      Value mi = mRepeatsLoop.getInductionVar();
      Value subview = preAccelRegBuffer;
      if (repeats > 1) {
        subview = createSliceOfFirstDim(rewriter, loc, preAccelRegBuffer, mi);
      }
      rewriter.create<ThreadwiseReadIntoOp>(loc, wrappedLDSBufferForLoad,
                                            subview, rewriter.getArrayAttr({}),
                                            ValueRange{tid, mi}, true, true);
    }
  }

  Value transposeAttnOperand(PatternRewriter &rewriter, Location loc,
                             TypedValue<MemRefType> operand) const {
    BottomUpTMBuilder viewBuilder(rewriter, operand.getType().getShape(), loc);
    viewBuilder.passThrough({0, 1, 2}, {0, 2, 1});
    TransformMapAttr trMap = viewBuilder.get();
    return rewriter.create<TransformOp>(loc, operand, trMap);
  }

  /// Check whether the op can bypass LDS-based swizzling
  /// for the B operand of the second gemm.
  bool canBypassLDSForSecondGemm(GridwiseAttentionAccelOp op) const {
    Type elemTypeQ =
        cast<MemRefType>(op.getQueries().getType()).getElementType();
    Type elemTypeK = cast<MemRefType>(op.getKeys().getType()).getElementType();
    StringRef arch = op.getArch();
    RockAccelTuningParamAttrInterface gemm0TuningParams = op.getParams0();
    auto accelEmitterPtrGemm0 = accel::AccelEmitter::select(
        op.getFeatures(), elemTypeQ, elemTypeK, arch, gemm0TuningParams);
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

  LogicalResult matchAndRewrite(GridwiseAttentionAccelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    StringRef arch = op.getArch();
    uint32_t blockSize = op.getBlockSize();
    uint32_t gridSize = op.getGridSize();

    TypedValue<MemRefType> inQ = op.getQueries();
    ArrayRef<int64_t> qShape = cast<MemRefType>(inQ.getType()).getShape();
    Type elemTypeQ = cast<MemRefType>(inQ.getType()).getElementType();

    TypedValue<MemRefType> inK = op.getKeys();
    ArrayRef<int64_t> kShape = cast<MemRefType>(inK.getType()).getShape();
    Type elemTypeK = cast<MemRefType>(inK.getType()).getElementType();

    TypedValue<MemRefType> inV = op.getValues();
    Type elemTypeV = inV.getType().getElementType();

    TypedValue<MemRefType> out = op.getOut();
    Value trOut = transposeAttnOperand(rewriter, loc, out);
    ArrayRef<int64_t> outShape = cast<MemRefType>(trOut.getType()).getShape();
    Type elemTypeOut = cast<MemRefType>(trOut.getType()).getElementType();

    // Gemm0 out is casted to be elemTypeV
    Type elemTypeQxK = elemTypeV;

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

    auto accelEmitterPtrGemm0 = accel::AccelEmitter::select(
        op.getFeatures(), elemTypeQ, elemTypeK, arch, gemm0TuningParams);
    if (!accelEmitterPtrGemm0)
      return op.emitOpError("Unable to emit accelerator code.");
    bool doBypassLDSSecondGemm = canBypassLDSForSecondGemm(op);
    bool doBypassLDSForQ = canBypassLDSForQ(op);
    rock::accel::AccelEmitterParams accelParamsGemm0 =
        accelEmitterPtrGemm0->getParams();
    auto accelEmitterPtrGemm1 = accel::AccelEmitter::select(
        op.getFeatures(), elemTypeV, elemTypeV, arch, gemm1TuningParams);
    if (!accelEmitterPtrGemm1)
      return op.emitOpError("Unable to emit accelerator code.");
    rock::accel::AccelEmitterParams accelParamsGemm1 =
        accelEmitterPtrGemm1->getParams();

    // Get current workgroup ID.
    auto bid = rewriter.create<WorkgroupIdOp>(loc, rewriter.getIndexType());
    // Get current workitem ID.
    auto tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());

    // Calculate different size derivations
    int64_t gemm0KPerBlock = gemm0kpack * gemm0KpacksPerBlock;
    int64_t gemm1KPerBlock = gemm0MPerBlock;
    int64_t gemm1MPerBlock = gemm1TuningParams.getMPerBlock();
    int64_t gemm1NPerBlock = gemm1TuningParams.getNPerBlock();
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
        getVectorDim(rewriter, loc, inQ, elemTypeQ, blockSize, gemm0KPerBlock,
                     gemm0NPerBlock, gemm0kpack);
    if (failed(maybeVectorDimInfoQ)) {
      return failure();
    }
    LDSLayoutConfigDim ldsLayoutCfgNG0 = getLDSLayoutConfigDim(
        elemTypeQ, gemm0kpack, maybeVectorDimInfoQ.value());
    FailureOr<VectorDimInfo> maybeVectorDimInfoK =
        getVectorDim(rewriter, loc, inK, elemTypeK, blockSize, gemm0KPerBlock,
                     gemm0MPerBlock, gemm0kpack);
    if (failed(maybeVectorDimInfoK)) {
      return failure();
    }
    LDSLayoutConfigDim ldsLayoutCfgMG0 = getLDSLayoutConfigDim(
        elemTypeK, gemm0kpack, maybeVectorDimInfoK.value());
    ldsLayoutCfgMG0.doRotateWithK = false;
    if (doBypassLDSSecondGemm) {
      ldsLayoutCfgMG0.doSwapThreadIterSubDims = false;
    }
    int64_t gemm0InMPerThread = maybeVectorDimInfoK->inDPerThread;
    int64_t gemm0InNPerThread = maybeVectorDimInfoQ->inDPerThread;
    RegsAsMatrixSubTiles gemm0OutSubTileViews =
        accelEmitterPtrGemm0->computeOutputTransforms(
            rewriter, loc, gemm0M, gemm0N, blockSize, gemm0BidGridLengths,
            gemm0InMPerThread, gemm0InNPerThread,
            ldsLayoutCfgMG0.doSwapThreadIterSubDims,
            ldsLayoutCfgNG0.doSwapThreadIterSubDims);
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
    auto [sharedBuffersGemmsB, ldsSizeB] = createSharedLDSByteBufferRefs(
        rewriter, loc,
        {ldsByteBufferQSize, reductionWorkspaceSize, gemm1LDSByteBufferBSize},
        {elemTypeQ, elemTypeQxK, elemTypeV});
    Value ldsByteBufferQ = sharedBuffersGemmsB[0];
    Value ldsReductionWorkspaceByteBuffer = sharedBuffersGemmsB[1];
    Value gemm1LDSByteBufferB = sharedBuffersGemmsB[2];
    auto [sharedBuffersGemmsA, ldsSizeA] = createSharedLDSByteBufferRefs(
        rewriter, loc,
        {gemm0KPerBlock * gemm0MPerBlock, gemm1KPerBlock * gemm1MPerBlock},
        {elemTypeK, elemTypeV});
    Value ldsByteBufferK = sharedBuffersGemmsA[0];
    Value ldsByteBufferV = sharedBuffersGemmsA[1];
    const int64_t maxLdsSize =
        rock::lookupArchInfo(op.getArch()).maxSharedMemPerWG;
    if (ldsSizeB + ldsSizeA > maxLdsSize) {
      return op.emitError() << "totalLDSSize (" << ldsSizeB + ldsSizeA
                            << ") exceeds " << maxLdsSize << "KB\n";
    }

    // Bufers for Gemm0
    Value fromGlobalRegBufferQ;
    Value toLDSRegBufferQ;
    if (doBypassLDSForQ) {
      Type loadBufferType =
          MemRefType::get({accelParamsGemm0.nRepeats *
                           accelParamsGemm0.kpackPerThread * gemm0kpack},
                          elemTypeQ, AffineMap{}, privateMemoryAddressSpace);
      fromGlobalRegBufferQ = rewriter.create<GpuAllocOp>(loc, loadBufferType);
    } else {
      std::tie(fromGlobalRegBufferQ, toLDSRegBufferQ) =
          createRegBuffersForGemmIn(loc, gemm0KPerBlock, blockSize, elemTypeQ,
                                    gemm0NPerBlock, rewriter);
    }
    auto [fromGlobalRegBufferK, toLDSRegBufferK] = createRegBuffersForGemmIn(
        loc, gemm0KPerBlock, blockSize, elemTypeK, gemm0MPerBlock, rewriter);
    // Note that we dont provide nRepeats because we dont want
    // nRepeats times reg buffer to be created for B of gemm0
    // because we wont be prefetching that.
    auto [preAccelRegBufferK, preAccelRegBuffersQ] =
        createRegInterrimBufferForAccel(loc, accelParamsGemm0, rewriter, 1,
                                        accelParamsGemm0.nRepeats);
    Value accRegBufferGemm0 =
        createBufferForAccelGemmOut(loc, accelParamsGemm0, rewriter);
    // Currently, there is a working assumption that this kernel is meant
    // support fp32/fp16 This should be guranteed by op verifiers.
    Type gemmOutElemType = elemTypeQxK;
    Type softmaxInElemType = elemTypeQxK;
    if (elemTypeQ == rewriter.getI8Type()) {
      gemmOutElemType = rewriter.getI32Type();
    }
    Value gemm0OutBuffer = createBufferForGemmOut(loc, gemmOutElemType,
                                                  accelParamsGemm0, rewriter);
    Value softmaxInBuffer = createBufferForGemmOut(loc, softmaxInElemType,
                                                   accelParamsGemm0, rewriter);

    // Buffers for reductions
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    TypedValue<MemRefType> ldsReductionWorkspaceBuffer =
        viewBufferAs(rewriter, ldsReductionWorkspaceByteBuffer, elemTypeQxK);

    Value gemm0OutBufferMax =
        createBufferForGemmOut(loc, elemTypeQxK, accelParamsGemm0, rewriter);
    Value gemm0OutBufferExp =
        createBufferForGemmOut(loc, elemTypeQxK, accelParamsGemm0, rewriter);
    Value gemm0OutBufferSum =
        createBufferForGemmOut(loc, elemTypeQxK, accelParamsGemm0, rewriter);

    // Buffers for gemm 1
    Value gemm1RegBufferB = gemm0OutBufferExp;
#ifdef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
    llvm::errs() << "Lowering attention op as a gemm-gemm op...\n";
    gemm1RegBufferB = gemm0OutBuffer;
#endif
    if (elemTypeV != elemTypeQxK) {
      gemm1RegBufferB =
          createBufferForGemmOut(loc, elemTypeV, accelParamsGemm0, rewriter);
    }
    Value gemm0ExpOutBufferToLDS =
        createBufferForGemmOut(loc, elemTypeV, accelParamsGemm0, rewriter);
    auto [preAccelRegBufferV, preAccelRegBufferQxK] =
        createRegInterrimBufferForAccel(loc, accelParamsGemm1, rewriter);
    Value accRegBufferGemm1 =
        createBufferForAccelGemmOut(loc, accelParamsGemm1, rewriter);
#ifdef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
    accRegBufferGemm1 = createBufferForAccelGemmOut(loc, accelParamsGemm1,
                                                    rewriter, gemm1MBlocks);
#endif
    Value gemm1OutBuffer =
        createBufferForGemmOut(loc, elemTypeQxK, accelParamsGemm1, rewriter);
#ifdef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
    gemm1OutBuffer = createBufferForGemmOut(loc, elemTypeQxK, accelParamsGemm1,
                                            rewriter, gemm1MBlocks);
#endif

    SmallVector<int64_t, 3> gemm1BidGridLengths = {gemm0G, gemm1MBlocks,
                                                   gemm1NBlocks};
    FailureOr<VectorDimInfo> maybeVectorDimInfoV =
        getVectorDim(rewriter, loc, inV, elemTypeV, blockSize, gemm1KPerBlock,
                     gemm1MPerBlock, gemm1kpack);
    if (failed(maybeVectorDimInfoV)) {
      return failure();
    }
    LDSLayoutConfigDim ldsLayoutCfgMG1 = getLDSLayoutConfigDim(
        elemTypeV, gemm1kpack, maybeVectorDimInfoV.value());
    int64_t gemm1InMPerThread = maybeVectorDimInfoV->inDPerThread;
    RegsAsMatrixSubTiles gemm1OutSubTileViews =
        accelEmitterPtrGemm1->computeOutputTransforms(
            rewriter, loc, gemm1M, gemm1N, blockSize, gemm1BidGridLengths,
            gemm1InMPerThread, gemm1InNPerThread,
            ldsLayoutCfgMG1.doSwapThreadIterSubDims);
    RegsAsMatrixSubTiles gemm1OutSubTileViewsTr =
        transposeSubTileViews(rewriter, loc, gemm1OutSubTileViews);
    auto [fromGlobalRegBufferV, toLDSRegBufferV] = createRegBuffersForGemmIn(
        loc, gemm1KPerBlock, blockSize, elemTypeV, gemm1MPerBlock, rewriter);
    int64_t gemm1MPerThread =
        getLowerShape(gemm1OutSubTileViewsTr.threadSubTile)[0];

    // Buffers for running row state

    // o buffer; this is exactly same as gemm1OutBuffer;
    // we just need another buffer to do the special accumulation
    Value attentionOutAccBuffer = createBufferForGemmOut(
        loc, elemTypeQxK, accelParamsGemm1, rewriter, gemm1MBlocks);
    Value attentionOutAccBufferOutTyped = attentionOutAccBuffer;
    if (elemTypeQxK != elemTypeOut) {
      attentionOutAccBufferOutTyped =
          createBufferForGemmOut(loc, elemTypeOut, accelParamsGemm1, rewriter);
    }
    ArrayAttr attentionOutAccBufferThreadSubTileViewMaps =
        invertTransforms(rewriter, loc, gemm1OutSubTileViewsTr.threadSubTile);
    // m buffer; this only contains a reduced single value per row
    auto reducedBufferType =
        MemRefType::get({gemm1MPerThread}, elemTypeQxK, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
    auto negInfSumTyped =
        createConstantFloatOp(rewriter, loc, reducedBufferType.getElementType(),
                              reducedBufferType.getElementType(),
                              -std::numeric_limits<float>::infinity());
    auto maxRowBuffer =
        rewriter.create<rock::GpuAllocOp>(loc, reducedBufferType);
    auto expMaxDiffRowBuffer =
        rewriter.create<rock::GpuAllocOp>(loc, reducedBufferType);
    rewriter.create<FillOp>(loc, maxRowBuffer, negInfSumTyped);
    // l buffer; this only contains a reduced single value per row
    Value sumRowBuffer =
        rewriter.create<rock::GpuAllocOp>(loc, reducedBufferType);
    rewriter.create<FillOp>(loc, sumRowBuffer,
                            createZeroConstantOp(rewriter, loc, elemTypeQxK));

    zeroAccBuffer(rewriter, loc, attentionOutAccBuffer);
#ifdef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
    zeroAccBuffer(rewriter, loc, accRegBufferGemm1);
#endif
    TypedValue<MemRefType> ldsTileBufferQ;
    // If gemm0K is equal to gemm0KPerBlock that means
    // effectively there is no K loop. Therefore, we
    // can prefetch the Q tile into regs outside of the
    // loop.
    if (gemm0K == gemm0KPerBlock) {
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: gemm0K is equal to gemm0KPerBlock\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: Prefetching Q tile into regs...\n");
      Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
      // it is fine m iteration to be zero as it irrelevant to Q tensor
      // as the first gemm is Kt x Qt.
      auto gridCoordsGemm0LoadQ = layout::makeGxNGridLayout(
          rewriter, loc, bid, zero, gemm0NBlocks, gridSize, arch);
      if (doBypassLDSForQ) {
        LogicalResult statusLoadQTile = loadAndStoreGemmInputTile(
            loc, inQ, /*kiter=*/zero, gridCoordsGemm0LoadQ,
            fromGlobalRegBufferQ, toLDSRegBufferQ, preAccelRegBuffersQ, "n",
            gemm0kpack, gemm0KpacksPerBlock, gemm0NPerBlock, blockSize,
            gridSize, bidGridOrder, gemm0BidGridLengths, forceUnroll, rewriter,
            *accelEmitterPtrGemm0.get(), ldsLayoutCfgNG0);
        if (failed(statusLoadQTile)) {
          return failure();
        }
      } else {
        LogicalResult statusLoadQ = loadAndStoreGemmInputTile(
            loc, inQ, /*kiter=*/zero, gridCoordsGemm0LoadQ,
            fromGlobalRegBufferQ, toLDSRegBufferQ, ldsByteBufferQ, "n",
            gemm0kpack, gemm0KpacksPerBlock, gemm0NPerBlock, blockSize,
            gridSize, bidGridOrder, gemm0BidGridLengths, forceUnroll, rewriter,
            *accelEmitterPtrGemm0.get(), ldsLayoutCfgNG0);
        if (failed(statusLoadQ)) {
          return failure();
        }
        ldsTileBufferQ = viewBufferAs(rewriter, ldsByteBufferQ,
                                      vectorTypeOrSelf(elemTypeQ, gemm0kpack));
        loadGemmOperandsFromLDSToRegs(
            rewriter, loc, ldsTileBufferQ, preAccelRegBuffersQ, "n", blockSize,
            gemm0InMPerThread, *accelEmitterPtrGemm0.get(),
            ldsLayoutCfgNG0.doRotateWithK);
      }
    }

    bool isReverseGrid = succeeded(rock::getReverseGrid(op));
    affine::AffineForOp mLoopOp =
        rewriter.create<affine::AffineForOp>(loc, 0, gemm0MBlocks, 1);
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mLoopOp.getBody());
      int64_t kIterationsGemm0 = gemm0K / gemm0KPerBlock;
      Value kIterationsGemm0Val =
          rewriter.createOrFold<arith::ConstantIndexOp>(loc, kIterationsGemm0);
      Value mIterationsGemm0Val =
          rewriter.createOrFold<arith::ConstantIndexOp>(loc, gemm0MBlocks);
      Value mLoopIV = mLoopOp.getInductionVar();
      if (isReverseGrid) {
        AffineMap reverseMap = rock::getIdxReversalMap(rewriter);
        mLoopIV = rewriter.createOrFold<affine::AffineApplyOp>(
            loc, reverseMap, ValueRange{mLoopIV, mIterationsGemm0Val});
      }
      zeroAccBuffer(rewriter, loc, accRegBufferGemm0);
      layout::GridCoordinates gridCoordsGemm0 = layout::makeGxNGridLayout(
          rewriter, loc, bid, mLoopIV, gemm0NBlocks, gridSize, arch);
      affine::AffineForOp kLoopOp =
          rewriter.create<affine::AffineForOp>(loc, 0, kIterationsGemm0, 1);
      {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(kLoopOp.getBody());
        Value kLoopIV = kLoopOp.getInductionVar();
        // Purpose of reversing the grid is to exploit
        // (if any) temporal locality between producers
        // and consumers of data between kernels.
        // Towards that goal, the kLoop has to be reversed
        // to use latest producer.
        if (isReverseGrid) {
          AffineMap reverseMap = rock::getIdxReversalMap(rewriter);
          kLoopIV = rewriter.createOrFold<affine::AffineApplyOp>(
              loc, reverseMap, ValueRange{kLoopIV, kIterationsGemm0Val});
        }
        // if gemm0K is equal to gemm0KPerBlock, the Q tile
        // is already prefetched into regs. See above.
        if (gemm0K != gemm0KPerBlock) {
          LogicalResult statusLoadQ = loadAndStoreGemmInputTile(
              loc, inQ, kLoopIV, gridCoordsGemm0, fromGlobalRegBufferQ,
              toLDSRegBufferQ, ldsByteBufferQ, "n", gemm0kpack,
              gemm0KpacksPerBlock, gemm0NPerBlock, blockSize, gridSize,
              bidGridOrder, gemm0BidGridLengths, forceUnroll, rewriter,
              *accelEmitterPtrGemm0.get(), ldsLayoutCfgNG0);
          if (failed(statusLoadQ)) {
            return failure();
          }
          ldsTileBufferQ =
              viewBufferAs(rewriter, ldsByteBufferQ,
                           vectorTypeOrSelf(elemTypeQ, gemm0kpack));
        }
        LogicalResult statusLoadKTile = loadAndStoreGemmInputTile(
            loc, inK, kLoopIV, gridCoordsGemm0, fromGlobalRegBufferK,
            toLDSRegBufferK, ldsByteBufferK, "m", gemm0kpack,
            gemm0KpacksPerBlock, gemm0MPerBlock, blockSize, gridSize,
            bidGridOrder, gemm0BidGridLengths, forceUnroll, rewriter,
            *accelEmitterPtrGemm0.get(), ldsLayoutCfgMG0);
        if (failed(statusLoadKTile)) {
          return failure();
        }
        TypedValue<MemRefType> ldsTileBufferK = viewBufferAs(
            rewriter, ldsByteBufferK, vectorTypeOrSelf(elemTypeK, gemm0kpack));
        // LDS barrier.
        rewriter.create<LDSBarrierOp>(loc);
        // if gemm0K is equal to gemm0KPerBlock, the Q tile
        // is already prefetched into regs. See above.
        if (gemm0K != gemm0KPerBlock) {
          loadGemmOperandsFromLDSToRegs(
              rewriter, loc, ldsTileBufferQ, preAccelRegBuffersQ, "n",
              blockSize, gemm0InNPerThread, *accelEmitterPtrGemm0.get(),
              ldsLayoutCfgNG0.doRotateWithK);
        }
        // Emit lowered blockwise GEMM 0.

        // Here we cannot use the full blockwise gemm operation
        // because it expects the operands to be present in the LDS.
        // That limits our ability to prefetch Q tile into regs outside
        // the attention loop. Therefore, we directly do AccelGemmOp as
        // if blockwise gemm would have been lowered to except the Q tile
        // fetching is lifted out.
        Value wrappedLDSBufferForLoadA =
            accelEmitterPtrGemm0->wrapLDSBufferForLoad(
                rewriter, loc, ldsTileBufferK, op.getBlockSize(),
                gemm0InMPerThread, "m", false);
        affine::AffineForOp nRepeatsLoop = rewriter.create<affine::AffineForOp>(
            loc, 0, accelParamsGemm0.nRepeats, 1);
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(nRepeatsLoop.getBody());
          Value ni = nRepeatsLoop.getInductionVar();
          Value preAccelRegBufferQ = preAccelRegBuffersQ;
          if (accelParamsGemm0.nRepeats > 1) {
            preAccelRegBufferQ =
                createSliceOfFirstDim(rewriter, loc, preAccelRegBuffersQ, ni);
          }
          auto mLoop = rewriter.create<affine::AffineForOp>(
              loc, 0, accelParamsGemm0.mRepeats);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mLoop.getBody());
            Value mi = mLoop.getInductionVar();
            // regsB = read B from LDS
            rewriter.create<ThreadwiseReadIntoOp>(
                loc, wrappedLDSBufferForLoadA, preAccelRegBufferK,
                rewriter.getArrayAttr({}), ValueRange{tid, mi}, true, true);
            // regsC += regsA * regsB
            auto kLoop = rewriter.create<affine::AffineForOp>(
                loc, 0, accelParamsGemm0.kBasePerThread);
            {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(kLoop.getBody());
              Value viewA = accelEmitterPtrGemm0->generateThreadwiseViewBufferA(
                  rewriter, loc, preAccelRegBufferK);
              Value viewB = accelEmitterPtrGemm0->generateThreadwiseViewBufferB(
                  rewriter, loc, preAccelRegBufferQ);
              Value viewC = accelEmitterPtrGemm0->generateThreadwiseViewBufferC(
                  rewriter, loc, accRegBufferGemm0);
              Value ki = kLoop.getInductionVar();
              rewriter.create<ThreadwiseAccelGemmOp>(
                  loc, viewA, viewB, viewC, ValueRange{mi, ni, ki},
                  op.getArchAttr(), op.getFeaturesAttr(), op.getParams0Attr());
            }
          }
        }
      }
      accelEmitterPtrGemm0->computeOutputConversion(
          rewriter, loc, accRegBufferGemm0, gemm0OutBuffer, forceUnroll);

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

      // Align the preSoftmaxElementWise (if any) linalg.generic to
      // be performed on the output of the first gemm.
      FailureOr<Value> maybeSoftmaxInBuffer = postProcessFirstGemm(
          rewriter, loc, op, gridCoordsGemm0, gemm0OutBuffer, softmaxInBuffer,
          gemm0OutSubTileViewsTrUnPadded);
      if (failed(maybeSoftmaxInBuffer)) {
        return op.emitError("post processing first gemm failed.\n");
      }
      gemm0OutBuffer = maybeSoftmaxInBuffer.value();
      // Scale gemm0 output by (1/ln2)
      // So that we can use exp2 instead of exp.
#ifndef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
      Value ln2Recip = createConstantFloatOp(rewriter, loc, elemTypeQxK,
                                             elemTypeQxK, 1.44269504);
      postProcessFirstGemmSplat<ElementwiseMultOp>(
          rewriter, loc, gridCoordsGemm0, gemm0OutBuffer, gemm0OutSubTileViews,
          ln2Recip.getDefiningOp<arith::ConstantOp>().getValue());
#endif

      // Handle padding
      bool hasPadding =
          op.getPrePadG0M().has_value() || op.getPrePadG0N().has_value();
      if (hasPadding) {
        createFirstGemmNegInfPadding(rewriter, loc, gridCoordsGemm0,
                                     gemm0OutBuffer,
                                     gemm0OutSubTileViewsTrUnPadded);
      }

      APInt reductionAxis = APInt(64, 1);
      APInt nrDimPerThread = APInt(64, gemm0MPerBlock / gemm0MPerThread);
      // LDS barrier.
      rewriter.create<LDSBarrierOp>(loc);
      rewriter.create<BlockwiseBroadcastReduceOp>(
          loc, gemm0OutBuffer, ldsReductionWorkspaceBuffer, gemm0OutBufferMax,
          /*extraOut=*/nullptr, reductionAxis, rock::ReduceMethod::Max,
          gemm0OutSubTileViewsTr.blockSubTile,
          gemm0OutSubTileViewsTr.blockSubTileTidSlice.value(),
          gemm0OutSubTileViewsTr.threadSubTile, /*extraViews=*/nullptr,
          blockSize);
      // softmax normalization.
      Value gemm0MNThreadwiseView =
          transform(rewriter, gemm0OutBuffer,
                    invertTransforms(rewriter, loc,
                                     gemm0OutSubTileViewsTr.threadSubTile));
      Value gemm0MNExpThreadwiseView =
          transform(rewriter, gemm0OutBufferExp,
                    invertTransforms(rewriter, loc,
                                     gemm0OutSubTileViewsTr.threadSubTile));
      Value gemm0MNMaxThreadwiseView =
          transform(rewriter, gemm0OutBufferMax,
                    invertTransforms(rewriter, loc,
                                     gemm0OutSubTileViewsTr.threadSubTile));
      expSubstractMaxFromGemm0(rewriter, loc, gemm0MNThreadwiseView,
                               gemm0MNExpThreadwiseView,
                               gemm0MNMaxThreadwiseView, maxRowBuffer);
      // LDS barrier is needed because
      // of the LDS workspace reuse.
      rewriter.create<LDSBarrierOp>(loc);
      rewriter.create<BlockwiseBroadcastReduceOp>(
          loc, gemm0OutBufferExp, ldsReductionWorkspaceBuffer,
          gemm0OutBufferSum, /*extraOut=*/nullptr, reductionAxis,
          rock::ReduceMethod::Sum, gemm0OutSubTileViewsTr.blockSubTile,
          gemm0OutSubTileViewsTr.blockSubTileTidSlice.value(),
          gemm0OutSubTileViewsTr.threadSubTile,
          /*extraViews=*/nullptr, blockSize);
      // LDS barrier.
      rewriter.create<LDSBarrierOp>(loc);
      Value gemm0SumThreadwiseView =
          transform(rewriter, gemm0OutBufferSum,
                    invertTransforms(rewriter, loc,
                                     gemm0OutSubTileViewsTr.threadSubTile));
      Value gemm0MaxThreadwiseView =
          transform(rewriter, gemm0OutBufferMax,
                    invertTransforms(rewriter, loc,
                                     gemm0OutSubTileViewsTr.threadSubTile));
      updateRowSum(rewriter, loc, gemm0SumThreadwiseView,
                   gemm0MaxThreadwiseView, sumRowBuffer, maxRowBuffer,
                   expMaxDiffRowBuffer);

      // Emit blockwise GEMM 1.
      {
        if (elemTypeV != elemTypeQxK) {
          createTypeConversionLaGeneric(rewriter, loc, gemm0OutBufferExp,
                                        gemm1RegBufferB);
        }
        Value wrappedLDSBufferForLoadB;
        if (!doBypassLDSSecondGemm) {
          // The output RegsAsSubTile views are N x M where N is reduction dim
          RegsAsMatrixSubTiles gemm0OutSubTileNxMViews = gemm0OutSubTileViews;
          ArrayAttr gemm0ThreadwiseSubtileViewNxMMaps = invertTransforms(
              rewriter, loc, gemm0OutSubTileNxMViews.threadSubTile);
          Value gemm0ExpNMThreadwiseView = transform(
              rewriter, gemm1RegBufferB, gemm0ThreadwiseSubtileViewNxMMaps);
          // Correct the below toLDSViews to be max LDS vectorizable
          // (For now just hacked in the existing view)
          // Copy copyKPerThread is set to 1 because
          // K is not packed as kpack vectors. Therefore, setting
          // copyKPerThread to be 1 will always make the LDS write
          // to be scalars -- which makes the following layout agnostic.
          // We should get rid of storing to LDS altogether with
          // the transposed layout for this gemm.
          LogicalResult storeGemm1ATileStatus = storeGemmInputTile(
              rewriter, loc, gemm1kpack, gemm0ExpNMThreadwiseView,
              gemm0OutSubTileNxMViews, gemm0ExpOutBufferToLDS,
              gemm1LDSByteBufferB, gemm1KpacksPerBlock, "n", gemm1KPerBlock,
              gemm1NPerBlock, /*copyKPerThread=*/1, gemm1InNPerThread,
              forceUnroll, false);
          if (failed(storeGemm1ATileStatus)) {
            return failure();
          }
          TypedValue<MemRefType> gemm1LDSBufferB =
              viewBufferAs(rewriter, gemm1LDSByteBufferB,
                           vectorTypeOrSelf(elemTypeQxK, gemm1kpack));
          wrappedLDSBufferForLoadB = accelEmitterPtrGemm1->wrapLDSBufferForLoad(
              rewriter, loc, gemm1LDSBufferB, op.getBlockSize(),
              gemm1InNPerThread, "n", false);
        }

        affine::AffineForOp g1MLoopOp =
            rewriter.create<affine::AffineForOp>(loc, 0, gemm1MBlocks, 1);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(g1MLoopOp.getBody());
          Value g1MLoopIndVar = g1MLoopOp.getInductionVar();
#ifndef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
          zeroAccBuffer(rewriter, loc, accRegBufferGemm1);
#else
          if (gemm1MBlocks > 1) {
            accRegBufferGemm1 = createSliceOfFirstDim(
                rewriter, loc, accRegBufferGemm1, g1MLoopIndVar);
          }
#endif
          auto gridCoordsGemm1 = layout::makeGxNGridLayout(
              rewriter, loc, bid, g1MLoopIndVar, gemm1NBlocks, gridSize, arch);

          LogicalResult statusLoadVTile = loadAndStoreGemmInputTile(
              loc, inV,
              /*kIter=*/mLoopIV, gridCoordsGemm1, fromGlobalRegBufferV,
              toLDSRegBufferV, ldsByteBufferV, "m", gemm1kpack,
              gemm1KpacksPerBlock, gemm1MPerBlock, blockSize, gridSize,
              bidGridOrder, gemm1BidGridLengths, forceUnroll, rewriter,
              *accelEmitterPtrGemm1.get(), ldsLayoutCfgMG1);
          if (failed(statusLoadVTile)) {
            return failure();
          }
          TypedValue<MemRefType> ldsTileBufferV =
              viewBufferAs(rewriter, ldsByteBufferV,
                           vectorTypeOrSelf(elemTypeV, gemm1kpack));
          // LDS barrier.
          rewriter.create<LDSBarrierOp>(loc);
          // Emit GEMM 1.
          Value wrappedLDSBufferForLoadA =
              accelEmitterPtrGemm1->wrapLDSBufferForLoad(
                  rewriter, loc, ldsTileBufferV, op.getBlockSize(),
                  gemm1InMPerThread, "m", ldsLayoutCfgMG1.doRotateWithK,
                  doBypassLDSSecondGemm);
          ArrayAttr gemm1ThreadwiseSubtileViewDxKMaps = invertTransforms(
              rewriter, loc, gemm0OutSubTileViewsTr.threadSubTile);
          Value gemm1BDxKThreadwiseView = transform(
              rewriter, gemm1RegBufferB, gemm1ThreadwiseSubtileViewDxKMaps);
          affine::AffineForOp nRepeatsLoop =
              rewriter.create<affine::AffineForOp>(
                  loc, 0, accelParamsGemm1.nRepeats, 1);
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(nRepeatsLoop.getBody());
            affine::AffineForOp mRepeatsLoop =
                rewriter.create<affine::AffineForOp>(
                    loc, 0, accelParamsGemm1.mRepeats, 1);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(mRepeatsLoop.getBody());
              Value ni = nRepeatsLoop.getInductionVar();
              Value mi = mRepeatsLoop.getInductionVar();

              // regsA = read A from LDS
              rewriter.create<ThreadwiseReadIntoOp>(
                  loc, wrappedLDSBufferForLoadA, preAccelRegBufferV,
                  rewriter.getArrayAttr({}), ValueRange{tid, mi}, true, true);
              // regsB = read B from LDS
              if (!doBypassLDSSecondGemm) {
                rewriter.create<ThreadwiseReadIntoOp>(
                    loc, wrappedLDSBufferForLoadB, preAccelRegBufferQxK,
                    rewriter.getArrayAttr({}), ValueRange{tid, ni}, true, true);
              } else {
                rewriter.create<ThreadwiseReadIntoOp>(
                    loc, gemm1BDxKThreadwiseView, preAccelRegBufferQxK,
                    rewriter.getArrayAttr({}), ValueRange{ni}, true, true);
              }

              affine::AffineForOp kBasePerThreadLoop =
                  rewriter.create<affine::AffineForOp>(
                      loc, 0, accelParamsGemm1.kBasePerThread, 1);
              {
                PatternRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(kBasePerThreadLoop.getBody());
                Value ki = kBasePerThreadLoop.getInductionVar();

                Value viewA =
                    accelEmitterPtrGemm1->generateThreadwiseViewBufferA(
                        rewriter, loc, preAccelRegBufferV);
                Value viewB =
                    accelEmitterPtrGemm1->generateThreadwiseViewBufferB(
                        rewriter, loc, preAccelRegBufferQxK);
                Value viewC =
                    accelEmitterPtrGemm1->generateThreadwiseViewBufferC(
                        rewriter, loc, accRegBufferGemm1);

                // regsC += regsA * regsB
                rewriter.create<ThreadwiseAccelGemmOp>(
                    loc, viewA, viewB, viewC, ValueRange{mi, ni, ki},
                    op.getArchAttr(), op.getFeaturesAttr(),
                    op.getParams1Attr());
              }
            }
          }

          // There is no second k-loop
          // Therefore can get the output straight away
          Value gemm1OutBufferPerG1MBlock = gemm1OutBuffer;
#ifdef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
          if (gemm1MBlocks > 1) {
            gemm1OutBufferPerG1MBlock = createSliceOfFirstDim(
                rewriter, loc, gemm1OutBuffer, g1MLoopIndVar);
          }
#endif
          accelEmitterPtrGemm1->computeOutputConversion(
              rewriter, loc, accRegBufferGemm1, gemm1OutBufferPerG1MBlock,
              forceUnroll);
          Value attentionOutAccBufferPerG1MBlock = attentionOutAccBuffer;
          if (gemm1MBlocks > 1) {
            attentionOutAccBufferPerG1MBlock = createSliceOfFirstDim(
                rewriter, loc, attentionOutAccBuffer, g1MLoopIndVar);
          }
          ArrayAttr invertedGemm1threadSubTileMaps = invertTransforms(
              rewriter, loc, gemm1OutSubTileViewsTr.threadSubTile);
          Value gemm1MNThreadwiseView =
              transform(rewriter, gemm1OutBufferPerG1MBlock,
                        invertedGemm1threadSubTileMaps);
          // Rescale/correct output, rowMax and rowSums
          Value attentionOutAccBufferView =
              transform(rewriter, attentionOutAccBufferPerG1MBlock,
                        attentionOutAccBufferThreadSubTileViewMaps);
          createAttentionRowStateCorrections(
              rewriter, loc, gemm1MNThreadwiseView, attentionOutAccBufferView,
              expMaxDiffRowBuffer);
        }
      }
    }
    {
      affine::AffineForOp g1MLoopOp =
          rewriter.create<affine::AffineForOp>(loc, 0, gemm1MBlocks, 1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(g1MLoopOp.getBody());
        Value g1MLoopIndVar = g1MLoopOp.getInductionVar();
        Value attentionOutAccBufferPerG1MBlock = attentionOutAccBuffer;
        if (gemm1MBlocks > 1) {
          attentionOutAccBufferPerG1MBlock = createSliceOfFirstDim(
              rewriter, loc, attentionOutAccBuffer, g1MLoopIndVar);
        }
        Value attentionOutAccBufferView =
            transform(rewriter, attentionOutAccBufferPerG1MBlock,
                      attentionOutAccBufferThreadSubTileViewMaps);
        scaleFinalOutput(rewriter, loc, attentionOutAccBufferView,
                         sumRowBuffer);
      }
    }
    if (elemTypeQxK != elemTypeOut) {
      createTypeConversionLaGeneric(rewriter, loc, attentionOutAccBuffer,
                                    attentionOutAccBufferOutTyped);
    }
#ifdef ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX
    attentionOutAccBufferOutTyped = gemm1OutBuffer;
#endif
    // We flatten output buffer in case gemm1MBlocks > 1
    // where those are iterated.
    Value attentionOutAccBufferOutTypedFlat = attentionOutAccBufferOutTyped;
    MemRefType attentionOutAccBufferOutType =
        cast<MemRefType>(attentionOutAccBufferOutTyped.getType());
    int64_t numElementsAttnOut = attentionOutAccBufferOutType.getNumElements();
    if (attentionOutAccBufferOutType.getRank() > 1) {
      Type attentionOutAccBufferOutTypedElType =
          attentionOutAccBufferOutType.getElementType();
      auto attentionOutAccBufferOutTypedFlatType = MemRefType::get(
          {numElementsAttnOut}, attentionOutAccBufferOutTypedElType,
          AffineMap{}, privateMemoryAddressSpace);
      auto reassociation =
          getReassociationForFlattening(attentionOutAccBufferOutType);
      attentionOutAccBufferOutTypedFlat =
          rewriter.create<memref::CollapseShapeOp>(
              loc, attentionOutAccBufferOutTypedFlatType,
              attentionOutAccBufferOutTyped, reassociation);
    }
    // This map will create an upper view [gblock, nblock, flatiter] -> [gblock,
    // miter, nblock, iter]
    TransformMapAttr flatToMiterMap =
        getFlatToMiterMap(rewriter, gemm0G, gemm1MBlocks, gemm1NBlocks,
                          blockSize, numElementsAttnOut);
    ArrayAttr outGridSubTile =
        prependUpperViews(rewriter, rewriter.getArrayAttr({flatToMiterMap}),
                          gemm1OutSubTileViews.gridSubTile);
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    auto gridCoordsGemm1 = layout::makeGxNGridLayout(
        rewriter, loc, bid, zero, gemm1NBlocks, gridSize, arch);
    rewriter.create<ThreadwiseWriteAllOp>(
        loc, attentionOutAccBufferOutTypedFlat, trOut, outGridSubTile,
        /*extraIndices=*/
        ValueRange{gridCoordsGemm1.g_block, gridCoordsGemm1.n_block, tid},
        op.getFeatures(), rock::StoreMethod::Set, forceUnroll,
        /*useIndexDiffs=*/true);
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
    auto elementTypeB = op.getB().getType().getElementType();
    auto destType = op.getC().getType().getElementType();

    // Prepare some useful constants.
    Value matA = op.getA();
    Value matB = op.getB();

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = op.getA().getType().getShape();
    bShape = op.getB().getType().getShape();
    cShape = op.getC().getType().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t K = aShape[1];
    int64_t M = aShape[2];
    int64_t N = bShape[2];

    // Obtain critical tuning parameters.
    StringRef arch = op.getArch();
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

    // Get the vector copy layout for A and B
    FailureOr<VectorDimInfo> maybeVecDimInfoA =
        getVectorDim(b, loc, op.getA(), elementTypeA, blockSize, kPerBlock,
                     mPerBlock, kpack);
    if (failed(maybeVecDimInfoA)) {
      return failure();
    }
    FailureOr<VectorDimInfo> maybeVecDimInfoB =
        getVectorDim(b, loc, op.getB(), elementTypeB, blockSize, kPerBlock,
                     nPerBlock, kpack);
    if (failed(maybeVecDimInfoB)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "gridSize: " << gridSize << "\n"
               << "blockSize: " << blockSize << "\n"
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
               << "copyMPerThread: " << maybeVecDimInfoA->inDPerThread << "\n"
               << "copyNPerThread: " << maybeVecDimInfoB->inDPerThread << "\n");
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    FailureOr<RegsAsMatrixSubTiles> maybeABufferViews = getLoadRegsAsTileViews(
        b, loc, op.getA(), "m", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, mPerBlock, maybeVecDimInfoA->inKPerThread,
        maybeVecDimInfoA->inDPerThread,
        maybeVecDimInfoA->vectorDim == GemmDimension::K);
    if (failed(maybeABufferViews)) {
      return failure();
    }
    Value wrappedA = transform(b, op.getA(), maybeABufferViews->gridSubTile);
    FailureOr<RegsAsMatrixSubTiles> maybeBBufferViews = getLoadRegsAsTileViews(
        b, loc, op.getB(), "n", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, nPerBlock, maybeVecDimInfoB->inKPerThread,
        maybeVecDimInfoB->inDPerThread,
        maybeVecDimInfoB->vectorDim == GemmDimension::K);
    if (failed(maybeBBufferViews)) {
      return failure();
    }
    Value wrappedB = transform(b, op.getB(), maybeBBufferViews->gridSubTile);

    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());
    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    auto loadBufferA =
        gpuAlloc(b, loc, aCopyPerThread, elementTypeA, AddressSpace::Private);
    auto loadBufferB =
        gpuAlloc(b, loc, bCopyPerThread, elementTypeB, AddressSpace::Private);

    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    // Compute grid coordinates
    auto gridCoords = layout::makeGroupedGridLayout(
        b, loc, bid,
        {G, mBlocks, nBlocks, op.getNumCU(), elementTypeA, destType}, arch);

    Value storeBufferA =
        gpuAlloc(b, loc, aCopyPerThread, elementTypeA, AddressSpace::Private);
    Value storeBufferB =
        gpuAlloc(b, loc, bCopyPerThread, elementTypeB, AddressSpace::Private);

    bool isKContiguousDimA = maybeVecDimInfoA->vectorDim == GemmDimension::K;
    bool isKContiguousDimB = maybeVecDimInfoB->vectorDim == GemmDimension::K;
    LDSLayoutConfigDim ldsLayoutConfigA =
        getLDSLayoutConfigDim(elementTypeA, kpack, maybeVecDimInfoA.value());
    LDSLayoutConfigDim ldsLayoutConfigB =
        getLDSLayoutConfigDim(elementTypeB, kpack, maybeVecDimInfoB.value());

    // We invert the transforms that are iter --> K x D slice of the tensor
    // so that we can view loadBuffer as a K x D tensor
    ArrayAttr loadBufferAViews =
        invertTransforms(b, loc, maybeABufferViews->threadSubTile);
    Value viewLoadBufferA = transform(b, loadBufferA, loadBufferAViews);
    // Prior to LDS store, we need re-arrange register buffer to maxmize LDS
    // vectorization Hence, creating the view w.r.t global that correspond to
    // such re-arranged register buffer
    FailureOr<RegsAsMatrixSubTiles> maybeALdsStoreViews =
        getPackedRegsAsTileViews(
            b, loc, op.getA(), "m", bidGridOrder, bidGridLengths, blockSize,
            kPerBlock, mPerBlock, maybeVecDimInfoA->inKPerThread,
            maybeVecDimInfoA->inDPerThread, kpack, isKContiguousDimA,
            ldsLayoutConfigA.doSwapThreadIterSubDims);
    if (failed(maybeALdsStoreViews)) {
      return failure();
    }
    ArrayAttr storeBufferAViews =
        invertTransforms(b, loc, maybeALdsStoreViews->threadSubTile);
    Value viewStoreBufferA = transform(b, storeBufferA, storeBufferAViews);
    ArrayAttr loadBufferBViews =
        invertTransforms(b, loc, maybeBBufferViews->threadSubTile);
    Value viewLoadBufferB = transform(b, loadBufferB, loadBufferBViews);
    // Prior to LDS store, we need re-arrange register buffer to maxmize LDS
    // vectorization Hence, creating the view w.r.t global that correspond to
    // such re-arranged register buffer
    FailureOr<RegsAsMatrixSubTiles> maybeBLdsStoreViews =
        getPackedRegsAsTileViews(
            b, loc, op.getB(), "n", bidGridOrder, bidGridLengths, blockSize,
            kPerBlock, nPerBlock, maybeVecDimInfoB->inKPerThread,
            maybeVecDimInfoB->inDPerThread, kpack, isKContiguousDimB,
            ldsLayoutConfigB.doSwapThreadIterSubDims);
    if (failed(maybeBLdsStoreViews)) {
      return failure();
    }
    ArrayAttr storeBufferBViews =
        invertTransforms(b, loc, maybeBLdsStoreViews->threadSubTile);
    Value viewStoreBufferB = transform(b, storeBufferB, storeBufferBViews);
    // Obtain Accelerator-related attributes.
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();

    auto accelEmitterPtr = accel::AccelEmitter::select(
        op.getFeatures(), elementTypeA, elementTypeB, arch, tuningParams);

    if (!accelEmitterPtr)
      return op.emitOpError("Unable to emit accelerator code.");

    // Extract relevant accelerator parameters
    rock::accel::AccelEmitterParams params = accelEmitterPtr->getParams();
    int64_t nResultVectors = params.nResultVectors;
    int64_t mRepeats = params.mRepeats;
    int64_t nRepeats = params.nRepeats;
    int64_t kBasePerThread = params.kBasePerThread;
    Type argTypeA = params.argTypeA;
    Type argTypeB = params.argTypeB;
    VectorType accVectorType = params.accVectorType;
    int64_t numOutputVectorElements = params.numOutputVectorElements();
    bool useIndexDiffs = true;

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
        kpacksPerBlock * mPerBlock * kpack * getByteWidth(elementTypeA);
    int64_t ldsBlockBSize =
        kpacksPerBlock * nPerBlock * kpack * getByteWidth(elementTypeB);
    LLVM_DEBUG(llvm::dbgs() << "LDS block sizes (bytes): " << ldsBlockASize
                            << " " << ldsBlockBSize << "\n");
    if (failed(checkLDSSize(op, ldsBlockASize, ldsBlockBSize)))
      return op.emitOpError("requires too much LDS");

    // Allocate LDS.
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefAType =
        MemRefType::get({ldsBlockASize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsByteBufferA = b.create<GpuAllocOp>(loc, ldsMemRefAType);
    auto ldsMemRefBType =
        MemRefType::get({ldsBlockBSize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsByteBufferB = b.create<GpuAllocOp>(loc, ldsMemRefBType);

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

    Value ldsViewForGemmA = viewBufferAs(b, ldsByteBufferA, ldsReadTypeA);
    Value ldsViewForGemmB = viewBufferAs(b, ldsByteBufferB, ldsReadTypeB);
    int64_t nOutputVectors = nResultVectors * mRepeats * nRepeats;

    // Logic to setup buffers for blockwise_gemm_accel.
    auto arrayA =
        gpuAlloc(b, loc, kBasePerThread, argTypeA, AddressSpace::Private);
    auto arrayB =
        gpuAlloc(b, loc, kBasePerThread, argTypeB, AddressSpace::Private);
    auto regCAllocOp =
        gpuAlloc(b, loc, nOutputVectors, accVectorType, AddressSpace::Private);

    Value zeroConstantCOp = createZeroConstantOp(b, loc, accVectorType);
    b.create<FillOp>(loc, regCAllocOp, zeroConstantCOp);

    // Emit loop.
    Value nIterations = b.create<ConstantIndexOp>(loc, K / kPerBlock);
    Value step = b.create<ConstantIndexOp>(loc, 1);
    BlockwiseGemmAccelOp blockwiseGemmAccelOp;

    auto loopOp = b.create<scf::ForOp>(loc, zeroConstantOp, nIterations, step);
    loopOp->setAttr(PipelineAttr::getMnemonic(),
                    rock::PipelineAttr::get(b.getContext(), 2));
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());
      Value iv = loopOp.getInductionVar();
      auto stage0 = b.create<StageOp>(loc, "GlobalRead");
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stage0.getRegion().emplaceBlock());
        bool isReverseGrid = succeeded(rock::getReverseGrid(op));
        // Purpose of reversing the grid is to exploit
        // (if any) temporal locality between producers
        // and consumers of data between kernels.
        // Towards that goal, the kLoop has to be reversed
        // to use latest producer.
        if (isReverseGrid) {
          AffineMap reverseMap = rock::getIdxReversalMap(b);
          iv = b.createOrFold<affine::AffineApplyOp>(loc, reverseMap,
                                                    ValueRange{iv, nIterations});
        }
        b.create<ThreadwiseReadIntoOp>(
            loc, wrappedA, loadBufferA, /*extraViews=*/b.getArrayAttr({}),
            /*extraIndices=*/
            ValueRange{/*kIter=*/iv, gridCoords.g_block, gridCoords.m_block,
                       gridCoords.n_block, tid},
            true, true);
        b.create<ThreadwiseReadIntoOp>(
            loc, wrappedB, loadBufferB, /*extraViews=*/b.getArrayAttr({}),
            /*extraIndices=*/
            ValueRange{/*kIter=*/iv, gridCoords.g_block, gridCoords.m_block,
                       gridCoords.n_block, tid},
            true, true);
        b.create<ThreadwiseCopyOp>(loc, viewLoadBufferA, ValueRange{},
                                   viewStoreBufferA, ValueRange{}, false,
                                   false);
        b.create<ThreadwiseCopyOp>(loc, viewLoadBufferB, ValueRange{},
                                   viewStoreBufferB, ValueRange{}, false,
                                   false);
        b.create<rock::YieldOp>(loc);
      }

      auto stage1 = b.create<StageOp>(loc, "LDSWrite");
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stage1.getRegion().emplaceBlock());

        // Emit blockwise stores
        b.create<ThreadwiseWriteAllOp>(loc, storeBufferA, wrappedLdsA,
                                       /*extraViews=*/b.getArrayAttr({}),
                                       /*extraIndices=*/ValueRange{tid},
                                       op.getFeatures(), StoreMethod::Set,
                                       /*forceUnroll=*/forceUnroll,
                                       /*useIndexDiffs=*/true);
        b.create<ThreadwiseWriteAllOp>(loc, storeBufferB, wrappedLdsB,
                                       /*extraViews=*/b.getArrayAttr({}),
                                       /*extraIndices=*/ValueRange{tid},
                                       op.getFeatures(), StoreMethod::Set,
                                       /*forceUnroll=*/forceUnroll,
                                       /*useIndexDiffs=*/true);
        b.create<rock::YieldOp>(loc);
      }

      // Emit blockwise GEMM.
      auto stage2 = b.create<StageOp>(loc, "MMA");
      {
        PatternRewriter::InsertionGuard guard(b);
        b.setInsertionPointToStart(&stage2.getRegion().emplaceBlock());
        blockwiseGemmAccelOp = b.create<BlockwiseGemmAccelOp>(
            loc, ldsViewForGemmA, ldsViewForGemmB,
            b.getI32IntegerAttr(maybeVecDimInfoA->inDPerThread),
            b.getI32IntegerAttr(maybeVecDimInfoB->inDPerThread),
            (ldsLayoutConfigA.doRotateWithK ? b.getUnitAttr() : nullptr),
            (ldsLayoutConfigB.doRotateWithK ? b.getUnitAttr() : nullptr),
            arrayA, arrayB, regCAllocOp, op.getArchAttr(), op.getFeaturesAttr(),
            op.getBlockSizeAttr(), op.getParamsAttr());
        b.create<rock::YieldOp>(loc);
      }
    }

    // Matrix C write out logic.
    Value convertedC = gpuAlloc(b, loc, numOutputVectorElements, destType,
                                AddressSpace::Private);

    ArrayAttr idToMatrixCMaps =
        accelEmitterPtr
            ->computeOutputTransforms(b, loc, M, N, blockSize, bidGridLengths,
                                      maybeVecDimInfoA->inDPerThread,
                                      maybeVecDimInfoB->inDPerThread,
                                      ldsLayoutConfigA.doSwapThreadIterSubDims,
                                      ldsLayoutConfigB.doSwapThreadIterSubDims)
            .gridSubTile;

    accelEmitterPtr->computeOutputConversion(b, loc, regCAllocOp, convertedC,
                                             forceUnroll);

    b.create<ThreadwiseWriteAllOp>(
        loc, convertedC, op.getC(), idToMatrixCMaps,
        /*extraIndices=*/
        ValueRange{gridCoords.g_block, gridCoords.m_block, gridCoords.n_block,
                   tid},
        op.getFeatures(), op.getStoreMethod(), forceUnroll, useIndexDiffs);
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
                         scf::SCFDialect, math::MathDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmRewritePattern, GridwiseGemmAccelRewritePattern,
               GridwiseAttentionAccelRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
