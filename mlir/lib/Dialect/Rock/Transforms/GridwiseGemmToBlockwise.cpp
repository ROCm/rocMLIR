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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "AccelEmitter.h"
#include "GridLayoutEmitter.h"
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

namespace {
struct RockGridwiseGemmToBlockwisePass
    : public rock::impl::RockGridwiseGemmToBlockwisePassBase<
          RockGridwiseGemmToBlockwisePass> {
  void runOnOperation() override;
};

static Type obtainAccumulatorType(OpBuilder &b, Type elementTypeA,
                                  Type elementTypeB, Type destType) {
  // Determine the type used on VGPR to act as accumulator.
  // f32: f32.
  // f16, bf16: f32 to prevent overflow from happening.
  // i16 : i16.
  // fp8 (any combo) : f32.
  // i8: i32, since we have an i32 output
  Type accumulatorType = destType;
  if (elementTypeA.isF16() || elementTypeA.isBF16() ||
      elementTypeA.isFloat8E5M2FNUZ() || elementTypeA.isFloat8E4M3FNUZ()) {
    accumulatorType = b.getF32Type();
  } else if (elementTypeA.isInteger(8)) {
    accumulatorType = b.getI32Type();
  }
  return accumulatorType;
}

/// Construct a `memref.view` operation that interprets the buffer `buffer`,
/// whose elements are bytes, as a buffer of `type`.
static TypedValue<MemRefType> viewBufferAs(OpBuilder &b, Value buffer,
                                           Type type) {
  Location loc = buffer.getLoc();
  Value zeroByteOffset = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  auto bufferType = buffer.getType().cast<MemRefType>();
  int64_t byteWidth = getByteWidth(type);
  int64_t numBytes = bufferType.getShape()[0];
  assert(numBytes % byteWidth == 0 && "Can't evenly fit type into buffer");
  int64_t length = numBytes / byteWidth;
  auto newBufferType = bufferType.cloneWith({length}, type);
  auto view =
      b.create<memref::ViewOp>(loc, newBufferType, buffer, zeroByteOffset,
                               /*dynamic dim sizes=*/ValueRange{});
  return TypedValue<MemRefType>(view.getResult());
}
} // end anonymous namespace

/// Given a copy layout <copyDPerThread, copyKPerThread>, come up with the best
/// vectorization strategy for the layout. For instance, if the layout is <D,K>
/// = <2,16> and K is contiguous, we will vectorize by 16 along K and we will
/// loop over the other dimension
static std::pair<GemmDimension, int64_t>
bestGlobalVectorization(OpBuilder &b, Value matrix, int64_t copyDPerThread,
                        int64_t copyKPerThread, GemmDimension tiebreaker,
                        int64_t kPerBlock, int64_t dPerBlock,
                        Type elementType) {
  Value tensor;
  ArrayAttr transforms;
  std::tie(tensor, transforms, std::ignore) = untransform(b, matrix);
  ArrayRef<int64_t> tensorShape =
      tensor.getType().cast<MemRefType>().getShape();
  int64_t kVectorLen = getMaxVectorizationForDatatype(
      transforms, static_cast<uint32_t>(GemmDimension::K),
      math_util::gcd(copyKPerThread * copyDPerThread, kPerBlock), tensorShape,
      elementType);

  int64_t dVectorLen = getMaxVectorizationForDatatype(
      transforms, static_cast<uint32_t>(GemmDimension::MorN),
      math_util::gcd(copyDPerThread * copyKPerThread, dPerBlock), tensorShape,
      elementType);

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
FailureOr<std::pair<int64_t, int64_t>> static computeCopyPerThread(
    Type elementType, int64_t copyPerThread, int64_t kPerBlock,
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

/// Wraps the LDS buffer "buffer", which is <kOuter * rotate(d) * kpack *
/// sizeof(T) x i8 into a tid x iter view, where `iter` iterates over nominal
/// scalar indices into a buffer of type T. `buffer` will be reinterpreted as a
/// buffer with element type vector<kpackPerThread x T> (with kpackPerThread ==
/// 1 meaning just T). The resulting view must be iterated over with a stride of
/// no less than min(kPerThread, kpack). Also note that the `d` dimension
/// has been rotated to minimize bank conflicts
static FailureOr<Value> wrapLDSBufferForStore(OpBuilder &b, Location loc,
                                              Value buffer, Type ldsReadType,
                                              int64_t kOuter, StringRef dName,
                                              int64_t d, int64_t kPerThread,
                                              int64_t dPerThread,
                                              bool rotateDWithK) {
  MemRefType bufferType = buffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> bufferShape = bufferType.getShape();
  Type dataType = ldsReadType;
  if (bufferShape.size() != 1)
    return emitError(loc, "Expected a flat buffer");
  int64_t kpack = 1;
  if (auto vectorDataType = dataType.dyn_cast<VectorType>()) {
    kpack = vectorDataType.getNumElements();
    dataType = vectorDataType.getElementType();
  }

  if (bufferShape[0] != kOuter * d * kpack * getByteWidth(dataType))
    return emitError(loc, "LDS buffer should have ")
           << kOuter * d * kpack * getByteWidth(dataType)
           << " elements but has " << bufferShape[0];

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
  SmallVector<Attribute, 4> transformAttrs{mergeKpackAttr};

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

/// This function pack the load buffer into a store buffer ready to be copied
/// into LDS:
///  - The load buffer is (viewed as) a KPerThread x DPerThread
///  - The store buffer needs to be packed as a [KOuterPerThread, dPerThread,
///  kpackPerThread]
///    buffer
TransformingForOp packLoadBufferToStoreBuffer(PatternRewriter &b, Location loc,
                                              Type elementType, int64_t kpack,
                                              Value loadBuffer,
                                              Value storeBuffer) {
  ArrayRef<int64_t> loadShape =
      loadBuffer.getType().cast<ShapedType>().getShape();
  Type elemType = loadBuffer.getType().cast<MemRefType>().getElementType();
  int64_t copyDPerThread = loadShape[1];
  int64_t copyKPerThread = loadShape[0];

  // We use kpackPerThread instead of kpack to cover edge cases where
  // copyKPerThread is smaller than kpack
  int64_t kpackPerThread = std::min(copyKPerThread, kpack);

  Value rawLoadBuffer;
  ArrayAttr loadBufferView;
  bool needs64BitIdx;
  std::tie(rawLoadBuffer, loadBufferView, needs64BitIdx) =
      untransform(b, loadBuffer);
  assert(!needs64BitIdx && "Registers shouldn't need 64-bit indexing");
  ArrayRef<int64_t> rawLoadBufferShape =
      rawLoadBuffer.getType().cast<ShapedType>().getShape();

  Value rawStoreBuffer;
  ArrayAttr storeBufferView;
  std::tie(rawStoreBuffer, storeBufferView, needs64BitIdx) =
      untransform(b, storeBuffer);
  assert(!needs64BitIdx && "Registers shouldn't need 64-bit indexing");
  ArrayRef<int64_t> rawStoreBufferShape =
      rawStoreBuffer.getType().cast<ShapedType>().getShape();

  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 2> start(2, zero);
  SmallVector<int64_t, 2> strides(2, 1);
  int64_t vecLen = 1;
  // The store buffer is a flattened < kouter x dPerThread x kpack >.
  if (kpackPerThread == 1) {
    // if kpack == 1, then we can do vectorized loads across d dimension from/to
    // load/store buffer
    vecLen = getMaxVectorizationForDatatype(loadBufferView, /*dim=*/1,
                                            copyDPerThread, rawLoadBufferShape,
                                            elemType);
    vecLen = math_util::gcd(copyDPerThread, vecLen);
    strides[1] = vecLen;
  } else {
    // if kpack > 1, then we are limited by vectorization in k dimension and it
    // could be at most kpack.
    vecLen = getMaxVectorizationForDatatype(loadBufferView, /*dim=*/0,
                                            copyKPerThread, rawLoadBufferShape,
                                            elemType);
    vecLen = math_util::gcd(vecLen, kpackPerThread);
    strides[0] = vecLen;
  }
  loadBufferView = collapseContiguousMerges(loadBufferView, rawLoadBufferShape);
  storeBufferView =
      collapseContiguousMerges(storeBufferView, rawStoreBufferShape);

  // Run the packing loop
  auto packLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{start, start},
      ArrayRef<Attribute>{loadBufferView, storeBufferView},
      /*bounds=*/loadShape,
      /*strides=*/strides, false,
      /*useIndexDiffs=*/false);
  {
    PatternRewriter::InsertionGuard outerGuard(b);
    b.setInsertionPointToStart(packLoop.getBody());
    Type loadType = vectorTypeOrSelf(elementType, vecLen);
    auto val = b.create<InBoundsLoadOp>(loc, loadType, rawLoadBuffer,
                                        packLoop.getLowerCoords(0));

    b.create<InBoundsStoreOp>(loc, val, rawStoreBuffer,
                              packLoop.getLowerCoords(1));
  }
  return packLoop;
}

template <typename OpT>
static LogicalResult checkLDSSize(OpT op, int64_t aBufferBytes,
                                  int64_t bBufferBytes) {
  int64_t ldsBytes = aBufferBytes + bBufferBytes;
  return success(ldsBytes <= 64 * 1024);
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
    Type accumulatorType =
        obtainAccumulatorType(b, elementTypeA, elementTypeB, destType);

    // Prepare some useful constants.
    Value zeroConstantFloatOp = createZeroConstantOp(b, loc, accumulatorType);
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
        MemRefType::get({threadCNumRegisters}, accumulatorType, AffineMap{},
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

    int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
    int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;
    if (aCopyPerThread == 0 || bCopyPerThread == 0) {
      return emitError(loc) << "Block size too large, rejecting as invalid.\n";
    }

    auto maybeCopyAPerThread = computeCopyPerThread(
        elementTypeA, aCopyPerThread, kPerBlock, mPerBlock, kpack, loc);
    if (failed(maybeCopyAPerThread))
      return maybeCopyAPerThread;
    int64_t aCopyKPerThread = (*maybeCopyAPerThread).first;
    int64_t copyMPerThread = (*maybeCopyAPerThread).second;

    auto maybeCopyBPerThread = computeCopyPerThread(
        elementTypeB, bCopyPerThread, kPerBlock, nPerBlock, kpack, loc);
    if (failed(maybeCopyBPerThread))
      return maybeCopyBPerThread;
    int64_t bCopyKPerThread = (*maybeCopyBPerThread).first;
    int64_t copyNPerThread = (*maybeCopyBPerThread).second;

    GemmDimension vectorTiebreaker =
        (kpack > 1) ? GemmDimension::K : GemmDimension::MorN;
    int64_t aVectorLen, bVectorLen;
    GemmDimension aVectorDim, bVectorDim;
    std::tie(aVectorDim, aVectorLen) = bestGlobalVectorization(
        b, op.getA(), copyMPerThread, aCopyKPerThread, vectorTiebreaker,
        kPerBlock, mPerBlock, elementTypeA);
    std::tie(bVectorDim, bVectorLen) = bestGlobalVectorization(
        b, op.getB(), copyNPerThread, bCopyKPerThread, vectorTiebreaker,
        kPerBlock, nPerBlock, elementTypeB);

    LLVM_DEBUG(llvm::dbgs()
               << "aCopyPerThread: " << aCopyPerThread << "\n"
               << "bCopyPerThread: " << bCopyPerThread << "\n"
               << "aVectorDim: " << aVectorDim << "\n"
               << "aVectorLen: " << aVectorLen << "\n"
               << "bVectorDim: " << bVectorDim << "\n"
               << "bVectorLen: " << bVectorLen << "\n"
               << "vectorTiebreaker: " << vectorTiebreaker << "\n");
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    FailureOr<RegsAsMatrixSubTiles> maybeABufferViews = getLoadRegsAsTileViews(
        b, loc, op.getA(), "m", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, mPerBlock, aCopyKPerThread, copyMPerThread,
        aVectorDim == GemmDimension::K);
    if (failed(maybeABufferViews)) {
      return failure();
    }
    Value wrappedA = transform(b, op.getA(), maybeABufferViews->gridSubTile);
    FailureOr<RegsAsMatrixSubTiles> maybeBBufferViews = getLoadRegsAsTileViews(
        b, loc, op.getB(), "n", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, nPerBlock, bCopyKPerThread, copyNPerThread,
        bVectorDim == GemmDimension::K);
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
    auto gridCoords = layout::makeGroupedGridLayout(
        b, loc, bid, {mBlocks, nBlocks, op.getNumCU(), elementTypeA, destType});
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

    bool isKContiguousDimA = (aVectorDim == GemmDimension::K);
    bool isKContiguousDimB = (bVectorDim == GemmDimension::K);

    // We invert the transforms that are iter --> K x D slice of the tensor
    // so that we can view loadBuffer as a K x D tensor
    ArrayAttr loadBufferAViews =
        invertTransforms(b, loc, maybeABufferViews->threadSubTile);
    Value viewLoadBufferA = transform(b, loadBufferA, loadBufferAViews);
    // Prior to LDS store, we need re-arrange register buffer to maxmize LDS
    // vectorization Hence, creating the view w.r.t global that correspond to
    // such re-arranged register buffer
    FailureOr<RegsAsMatrixSubTiles> maybeALdsStoreViews =
        getPackedRegsAsTileViews(b, loc, op.getA(), "m", bidGridOrder,
                                 bidGridLengths, blockSize, kPerBlock,
                                 mPerBlock, aCopyKPerThread, copyMPerThread,
                                 kpack, isKContiguousDimA);
    if (failed(maybeALdsStoreViews)) {
      return failure();
    }
    ArrayAttr storeBufferAViews =
        invertTransforms(b, loc, maybeALdsStoreViews->threadSubTile);
    Value viewStoreBufferA = transform(b, storeBufferA, storeBufferAViews);
    auto packALoop = packLoadBufferToStoreBuffer(
        b, loc, elementTypeA, kpack, viewLoadBufferA, viewStoreBufferA);
    ArrayAttr loadBufferBViews =
        invertTransforms(b, loc, maybeBBufferViews->threadSubTile);
    Value viewLoadBufferB = transform(b, loadBufferB, loadBufferBViews);
    // Prior to LDS store, we need re-arrange register buffer to maxmize LDS
    // vectorization Hence, creating the view w.r.t global that correspond to
    // such re-arranged register buffer
    FailureOr<RegsAsMatrixSubTiles> maybeBLdsStoreViews =
        getPackedRegsAsTileViews(b, loc, op.getB(), "n", bidGridOrder,
                                 bidGridLengths, blockSize, kPerBlock,
                                 nPerBlock, bCopyKPerThread, copyNPerThread,
                                 kpack, isKContiguousDimB);
    if (failed(maybeBLdsStoreViews)) {
      return failure();
    }
    ArrayAttr storeBufferBViews =
        invertTransforms(b, loc, maybeBLdsStoreViews->threadSubTile);
    Value viewStoreBufferB = transform(b, storeBufferB, storeBufferBViews);
    auto packBLoop = packLoadBufferToStoreBuffer(
        b, loc, elementTypeB, kpack, viewLoadBufferB, viewStoreBufferB);

    Type ldsReadTypeA = vectorTypeOrSelf(elementTypeA, kpack);
    FailureOr<Value> maybeWrappedLdsA = wrapLDSBufferForStore(
        b, loc, ldsByteBufferA, ldsReadTypeA, kpacksPerBlock, "m", mPerBlock,
        aCopyKPerThread, copyMPerThread, /*rotateDWithK=*/isKContiguousDimA);
    if (failed(maybeWrappedLdsA))
      return maybeWrappedLdsA;
    // This is KxD view of the flat LDS buffer
    Value wrappedLdsA = std::move(*maybeWrappedLdsA);
    // This will produce a (tid, iter) --> flat LDS view
    wrappedLdsA = transform(b, wrappedLdsA, maybeALdsStoreViews->blockSubTile);

    Type ldsReadTypeB = vectorTypeOrSelf(elementTypeB, kpack);
    FailureOr<Value> maybeWrappedLdsB = wrapLDSBufferForStore(
        b, loc, ldsByteBufferB, ldsReadTypeB, kpacksPerBlock, "n", nPerBlock,
        bCopyKPerThread, copyNPerThread, /*rotateDWithK=*/isKContiguousDimB);
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

      // LDS bank-conflicts parameters
      UnitAttr rotateMWithK = (isKContiguousDimA ? b.getUnitAttr() : nullptr);
      UnitAttr rotateNWithK = (isKContiguousDimB ? b.getUnitAttr() : nullptr);

      // Emit blockwise GEMM.
      blockwiseGemmOp = b.create<BlockwiseGemmOp>(
          loc, ldsMatrixA, ldsMatrixB, registerMatrixCViewOp,
          b.getI32IntegerAttr(copyMPerThread),
          b.getI32IntegerAttr(copyNPerThread), rotateMWithK, rotateNWithK,
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
    toMatrixC.passThrough({"gemmG"}, {0}, {"g_block"});
    toMatrixC.unmerge(
        "gemmM", 1,
        {"m_block", "m_repeat", "m_cuwaves", "m_cuwave", "m_thread"},
        {M / mPerBlock, gemmMRepeat, mCuwavesPerBlock, mThreadsPerCuwave,
         mPerThread});
    toMatrixC.unmerge(
        "gemmN", 2,
        {"n_block", "n_repeat", "n_cuwaves", "n_cuwave", "n_thread"},
        {N / nPerBlock, gemmNRepeat, nCuwavesPerBlock, nThreadsPerCuwave,
         nPerThread});

    swapThreadIdAndIteration(toMatrixC, bidGridLengths, copyMPerThread,
                             copyNPerThread, mPerBlock, nPerBlock,
                             !isKContiguousDimA, !isKContiguousDimB,
                             /*isBlockwise=*/false, transformAttrs);

    Value registerC = registerMatrixCAllocOp;
    // If we need to type-convert the accumulator (currently this is only
    // fp32->f16) then we must do so before the writeback loop in which fusion
    // takes places at this time, since the fusion pass as currently written
    // can't interceps the type conversions.
    if (destType != accumulatorType) {
      auto convertedCType =
          threadCRegisterMemRefType.clone(destType).cast<MemRefType>();
      Value convertedC = b.create<rock::GpuAllocOp>(loc, convertedCType);
      auto convertLoop = b.create<TransformingForOp>(
          loc, ArrayRef<ValueRange>{{zeroConstantOp}},
          ArrayRef<Attribute>{b.getArrayAttr({})},
          /*bounds=*/convertedCType.getShape(), /*strides=*/std::nullopt,
          /*useIndexDiffs=*/true, /*forceUnroll=*/true);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(convertLoop.getBody());
        Value coord = convertLoop.getLowerCoords(/*domain=*/0)[0];
        Value loaded =
            b.create<InBoundsLoadOp>(loc, accumulatorType, registerC, coord);
        Value cast = createTypeConversionOp(b, loc, loaded, destType);
        b.create<InBoundsStoreOp>(loc, cast, convertedC, coord);
      }
      registerC = convertedC;
    }

    ArrayAttr idToMatrixCMaps = b.getArrayAttr(transformAttrs);

    b.create<ThreadwiseWriteAllOp>(loc, registerC, op.getC(), idToMatrixCMaps,
                                   /*extraIndices=*/
                                   ValueRange{gridCoords.g_block,
                                              gridCoords.m_block,
                                              gridCoords.n_block, tid},
                                   op.getFeatures(), StoreMethod::Set,
                                   /*forceUnroll=*/true, useIndexDiffs);
    b.eraseOp(op);

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

    int64_t aVectorLen = 0;
    int64_t bVectorLen = 0;
    GemmDimension aVectorDim;
    GemmDimension bVectorDim;

    int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
    int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;
    if (aCopyPerThread == 0 || bCopyPerThread == 0) {
      return emitError(loc) << "Block size too large, rejecting as invalid.\n";
    }
    int64_t aCopyKpacksPerThread =
        math_util::integer_divide_ceil(aCopyPerThread, kpack);
    int64_t bCopyKpacksPerThread =
        math_util::integer_divide_ceil(bCopyPerThread, kpack);

    // Get the vector copy layout for A and B
    auto maybeCopyAPerThread = computeCopyPerThread(
        elementTypeA, aCopyPerThread, kPerBlock, mPerBlock, kpack, loc);
    if (failed(maybeCopyAPerThread))
      return maybeCopyAPerThread;
    int64_t aCopyKPerThread = (*maybeCopyAPerThread).first;
    int64_t copyMPerThread = (*maybeCopyAPerThread).second;

    auto maybeCopyBPerThread = computeCopyPerThread(
        elementTypeB, bCopyPerThread, kPerBlock, nPerBlock, kpack, loc);
    if (failed(maybeCopyBPerThread))
      return maybeCopyBPerThread;
    int64_t bCopyKPerThread = (*maybeCopyBPerThread).first;
    int64_t copyNPerThread = (*maybeCopyBPerThread).second;

    // Find the best way of vectorizing the layout
    GemmDimension vectorTiebreaker =
        (kpack > 1) ? GemmDimension::K : GemmDimension::MorN;
    std::tie(aVectorDim, aVectorLen) = bestGlobalVectorization(
        b, matA, copyMPerThread, aCopyKPerThread, vectorTiebreaker, kPerBlock,
        mPerBlock, elementTypeA);
    std::tie(bVectorDim, bVectorLen) = bestGlobalVectorization(
        b, matB, copyNPerThread, bCopyKPerThread, vectorTiebreaker, kPerBlock,
        nPerBlock, elementTypeB);

    LLVM_DEBUG(llvm::dbgs()
               << "gridSize: " << gridSize << "\n"
               << "blockSize: " << blockSize << "\n"
               << "aCopyPerThread: " << aCopyPerThread << "\n"
               << "bCopyPerThread: " << bCopyPerThread << "\n"
               << "aCopyKpacksPerThread: " << aCopyKpacksPerThread << "\n"
               << "bCopyKpacksPerThread: " << bCopyKpacksPerThread << "\n"
               << "aVectorDim: " << aVectorDim << "\n"
               << "aVectorLen: " << aVectorLen << "\n"
               << "bVectorDim: " << bVectorDim << "\n"
               << "bVectorLen: " << bVectorLen << "\n"
               << "vectorTiebreaker: " << vectorTiebreaker << "\n"
               << "kPerBlock: " << kPerBlock << "\n"
               << "mPerBlock: " << mPerBlock << "\n"
               << "nPerBlock: " << nPerBlock << "\n"
               << "aCopyKPerThread: " << aCopyKPerThread << "\n"
               << "bCopyKPerThread: " << bCopyKPerThread << "\n"
               << "copyMPerThread: " << copyMPerThread << "\n"
               << "copyNPerThread: " << copyNPerThread << "\n");
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};
    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    FailureOr<RegsAsMatrixSubTiles> maybeABufferViews = getLoadRegsAsTileViews(
        b, loc, op.getA(), "m", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, mPerBlock, aCopyKPerThread, copyMPerThread,
        aVectorDim == GemmDimension::K);
    if (failed(maybeABufferViews)) {
      return failure();
    }
    Value wrappedA = transform(b, op.getA(), maybeABufferViews->gridSubTile);
    FailureOr<RegsAsMatrixSubTiles> maybeBBufferViews = getLoadRegsAsTileViews(
        b, loc, op.getB(), "n", bidGridOrder, bidGridLengths, blockSize,
        kPerBlock, nPerBlock, bCopyKPerThread, copyNPerThread,
        bVectorDim == GemmDimension::K);
    if (failed(maybeBBufferViews)) {
      return failure();
    }
    Value wrappedB = transform(b, op.getB(), maybeBBufferViews->gridSubTile);

    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());
    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    Type loadBufferAType, loadBufferBType;
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    loadBufferAType = MemRefType::get({aCopyPerThread}, elementTypeA,
                                      AffineMap{}, privateMemoryAddressSpace);
    loadBufferBType = MemRefType::get({bCopyPerThread}, elementTypeB,
                                      AffineMap{}, privateMemoryAddressSpace);

    auto loadBufferA = b.create<GpuAllocOp>(loc, loadBufferAType);
    auto loadBufferB = b.create<GpuAllocOp>(loc, loadBufferBType);

    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    // Compute grid coordinates
    auto gridCoords = layout::makeGroupedGridLayout(
        b, loc, bid, {mBlocks, nBlocks, op.getNumCU(), elementTypeA, destType});
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

    bool isKContiguousDimA = aVectorDim == GemmDimension::K;
    bool isKContiguousDimB = bVectorDim == GemmDimension::K;

    // We invert the transforms that are iter --> K x D slice of the tensor
    // so that we can view loadBuffer as a K x D tensor
    ArrayAttr loadBufferAViews =
        invertTransforms(b, loc, maybeABufferViews->threadSubTile);
    Value viewLoadBufferA = transform(b, loadBufferA, loadBufferAViews);
    // Prior to LDS store, we need re-arrange register buffer to maxmize LDS
    // vectorization Hence, creating the view w.r.t global that correspond to
    // such re-arranged register buffer
    FailureOr<RegsAsMatrixSubTiles> maybeALdsStoreViews =
        getPackedRegsAsTileViews(b, loc, op.getA(), "m", bidGridOrder,
                                 bidGridLengths, blockSize, kPerBlock,
                                 mPerBlock, aCopyKPerThread, copyMPerThread,
                                 kpack, isKContiguousDimA);
    if (failed(maybeALdsStoreViews)) {
      return failure();
    }
    ArrayAttr storeBufferAViews =
        invertTransforms(b, loc, maybeALdsStoreViews->threadSubTile);
    Value viewStoreBufferA = transform(b, storeBufferA, storeBufferAViews);
    auto packALoop = packLoadBufferToStoreBuffer(
        b, loc, elementTypeA, kpack, viewLoadBufferA, viewStoreBufferA);
    ArrayAttr loadBufferBViews =
        invertTransforms(b, loc, maybeBBufferViews->threadSubTile);
    Value viewLoadBufferB = transform(b, loadBufferB, loadBufferBViews);
    // Prior to LDS store, we need re-arrange register buffer to maxmize LDS
    // vectorization Hence, creating the view w.r.t global that correspond to
    // such re-arranged register buffer
    FailureOr<RegsAsMatrixSubTiles> maybeBLdsStoreViews =
        getPackedRegsAsTileViews(b, loc, op.getB(), "n", bidGridOrder,
                                 bidGridLengths, blockSize, kPerBlock,
                                 nPerBlock, bCopyKPerThread, copyNPerThread,
                                 kpack, isKContiguousDimB);
    if (failed(maybeBLdsStoreViews)) {
      return failure();
    }
    ArrayAttr storeBufferBViews =
        invertTransforms(b, loc, maybeBLdsStoreViews->threadSubTile);
    Value viewStoreBufferB = transform(b, storeBufferB, storeBufferBViews);
    auto packBLoop = packLoadBufferToStoreBuffer(
        b, loc, elementTypeB, kpack, viewLoadBufferB, viewStoreBufferB);

    // Obtain Accelerator-related attributes.
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();
    int64_t nWaves = nPerBlock / nPerWave;

    auto nWavesConstantOp = b.create<ConstantIndexOp>(loc, nWaves);

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

    const int64_t waveSize = rock::lookupArchInfo(arch).waveSize;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, waveSize);

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
                            << "nPerWave: " << nPerWave << "\n"
                            << "aVectorLen: " << aVectorLen << "\n"
                            << "bVectorLen: " << bVectorLen << "\n"
                            << "aVectorDim: " << aVectorDim << "\n"
                            << "bVectorDim: " << bVectorDim << "\n");

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
        aCopyKPerThread, copyMPerThread, /*rotateDWithK=*/isKContiguousDimA);
    if (failed(maybeWrappedLdsA))
      return maybeWrappedLdsA;
    // This is KxD view of the flat LDS buffer
    Value wrappedLdsA = std::move(*maybeWrappedLdsA);
    // This will produce a (tid, iter) --> flat LDS view
    wrappedLdsA = transform(b, wrappedLdsA, maybeALdsStoreViews->blockSubTile);

    Type ldsReadTypeB = vectorTypeOrSelf(elementTypeB, kpack);
    FailureOr<Value> maybeWrappedLdsB = wrapLDSBufferForStore(
        b, loc, ldsByteBufferB, ldsReadTypeB, kpacksPerBlock, "n", nPerBlock,
        bCopyKPerThread, copyNPerThread, /*rotateDWithK=*/isKContiguousDimB);
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
        /*forceUnroll=*/forceUnroll, /*useIndexDiffs=*/true);
    ThreadwiseWriteAllOp blockwiseStoreB = b.create<ThreadwiseWriteAllOp>(
        loc, storeBufferB, wrappedLdsB,
        /*extraViews=*/b.getArrayAttr({}),
        /*extraIndices=*/ValueRange{tid}, op.getFeatures(), StoreMethod::Set,
        /*forceUnroll=*/forceUnroll, /*useIndexDiffs=*/true);

    Value ldsViewForGemmA = viewBufferAs(b, ldsByteBufferA, ldsReadTypeA);
    Value ldsViewForGemmB = viewBufferAs(b, ldsByteBufferB, ldsReadTypeB);
    int64_t nOutputVectors = nResultVectors * mRepeats * nRepeats;

    // Logic to setup blockwise_gemm_accel parameters.
    //
    // Original C++ logic:
    // index_t mMyWaveOffsetA;
    // index_t mMyWaveOffsetB;
    // const index_t waveId   = get_thread_local_1d_id() / WaveSize;
    // const index_t waveId_m = waveId / GemmNWaves;
    // const index_t waveId_n = waveId % GemmNWaves;
    // mMyWaveOffsetA = waveId_m * GemmMPerWave;
    // mMyWaveOffsetB = waveId_n * GemmNPerWave;
    auto waveId = b.create<DivUIOp>(loc, tid, waveSizeConstantOp);
    auto waveId_m = b.create<DivUIOp>(loc, waveId, nWavesConstantOp);
    auto waveId_n = b.create<RemUIOp>(loc, waveId, nWavesConstantOp);

    Value mMyWaveOffsetA, mMyWaveOffsetB;
    Value waveOffsetAConstantOp =
        b.create<ConstantIndexOp>(loc, params.mPerAccel);
    Value waveOffsetBConstantOp =
        b.create<ConstantIndexOp>(loc, params.nPerAccel);
    mMyWaveOffsetA = b.create<MulIOp>(loc, waveId_m, waveOffsetAConstantOp);
    mMyWaveOffsetB = b.create<MulIOp>(loc, waveId_n, waveOffsetBConstantOp);

    // Logic to setup buffers for blockwise_gemm_accel.

    Type arrayAType, arrayBType;
    arrayAType = MemRefType::get({kBasePerThread}, argTypeA, AffineMap{},
                                 privateMemoryAddressSpace);
    arrayBType = MemRefType::get({kBasePerThread}, argTypeB, AffineMap{},
                                 privateMemoryAddressSpace);
    auto arrayA = b.create<GpuAllocOp>(loc, arrayAType);
    auto arrayB = b.create<GpuAllocOp>(loc, arrayBType);

    // -----
    // Logic to allocate 0-initialized vectors for C.
    MemRefType regCAllocType =
        MemRefType::get(nOutputVectors, accVectorType, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
    Value regCAllocOp = b.create<rock::GpuAllocOp>(loc, regCAllocType);

    Value zeroConstantCOp = createZeroConstantOp(b, loc, accVectorType);
    b.create<FillOp>(loc, regCAllocOp, zeroConstantCOp);

    // Emit loop.
    int64_t nIterations = K / kPerBlock;
    BlockwiseGemmAccelOp blockwiseGemmAccelOp;
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

      // LDS bank-conflicts parameters
      UnitAttr rotateMWithK = (isKContiguousDimA ? b.getUnitAttr() : nullptr);
      UnitAttr rotateNWithK = (isKContiguousDimB ? b.getUnitAttr() : nullptr);

      // Emit blockwise GEMM.
      blockwiseGemmAccelOp = b.create<BlockwiseGemmAccelOp>(
          loc, ldsViewForGemmA, ldsViewForGemmB,
          b.getI32IntegerAttr(copyMPerThread),
          b.getI32IntegerAttr(copyNPerThread), rotateMWithK, rotateNWithK,
          mMyWaveOffsetA, mMyWaveOffsetB, arrayA, arrayB, regCAllocOp,
          op.getArchAttr(), op.getFeaturesAttr(), op.getBlockSizeAttr(),
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

    // Emit loop tail.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    IRMapping tailGemmCloneMap;
    b.clone(*blockwiseGemmAccelOp, tailGemmCloneMap);

    // Apparently, the canonicalizer doesn't get rid of empty loops without
    // results properly, remove them ourselves.
    if (nIterations <= 1) {
      b.eraseOp(loopOp);
    }

    // -----

    // Matrix C write out logic.
    auto convertedCType =
        MemRefType::get(numOutputVectorElements, destType, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
    Value convertedC = b.create<rock::GpuAllocOp>(loc, convertedCType);

    bool doSwapThreadIterSubDimsForM = !isKContiguousDimA;
    bool doSwapThreadIterSubDimsForN = !isKContiguousDimB;
    ArrayAttr idToMatrixCMaps =
        accelEmitterPtr
            ->computeOutputTransforms(b, loc, M, N, doSwapThreadIterSubDimsForM,
                                      doSwapThreadIterSubDimsForN,
                                      copyMPerThread, copyNPerThread, blockSize,
                                      bidGridLengths)
            .gridSubTile;

    Value registerC = accelEmitterPtr->computeOutputConversion(
        b, loc, M, N, blockSize, gridSize, regCAllocOp, convertedC,
        forceUnroll);

    b.create<ThreadwiseWriteAllOp>(
        loc, registerC, op.getC(), idToMatrixCMaps,
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
  target.addIllegalOp<rock::GridwiseGemmOp, rock::GridwiseGemmAccelOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         memref::MemRefDialect, affine::AffineDialect,
                         vector::VectorDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmRewritePattern, GridwiseGemmAccelRewritePattern>(
      ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
