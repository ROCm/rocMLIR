//===- BlockwiseGemmToThreadwise - MLIR Rock ops lowering passes ---===//
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
// This pass converts rock.blockwise_* ops to rock.threadwise_*
// and lowers other higher-level ops like transform and fill in preparation for
// the threadwise lowering
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "mlir/Dialect/Rock/utility/LdsTransposeLoad.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKBLOCKWISEGEMMTOTHREADWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-blockwise-to-threadwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using namespace mlir::affine;

namespace {
struct RockLowerBlockwiseGemmToThreadwisePass
    : public rock::impl::RockBlockwiseGemmToThreadwisePassBase<
          RockLowerBlockwiseGemmToThreadwisePass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Fill lowering.
//===----------------------------------------------------------------------===//

struct FillRewritePattern : public OpConversionPattern<FillOp> {
  using OpConversionPattern<FillOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(FillOp op, FillOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    MemRefType inputType = op.getInput().getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t> lbs(inputShape.size(), 0);
    llvm::SmallVector<int64_t> strides(inputShape.size(), 1);

    affine::buildAffineLoopNest(
        b, loc, lbs, inputShape, strides,
        [value = adaptor.getValue(), input = adaptor.getInput()](
            OpBuilder &b, Location loc, ValueRange ivs) {
          memref::StoreOp::create(b, loc, value, input, ivs);
        });

    b.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseFill lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseFillRewritePattern
    : public OpConversionPattern<BlockwiseFillOp> {
  using OpConversionPattern<BlockwiseFillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BlockwiseFillOp op, BlockwiseFillOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = op.getMemref().getType();
    ArrayRef<int64_t> memrefShape = memrefType.getShape();
    BottomUpTMBuilder threadsToMemrefTrBuilder(rewriter, memrefShape, loc);
    SmallVector<StringRef, 1> lowerNameRefs;
    threadsToMemrefTrBuilder.getStartNames(lowerNameRefs);
    int64_t blockSize = op.getBlockSize();

    Value val = op.getValue();
    int64_t numElements = memrefType.getNumElements();
    Type valueType = val.getType();
    int64_t valueItems = 1;
    Type valueElementType = valueType;
    if (VectorType valueVecType = dyn_cast<VectorType>(val.getType())) {
      valueItems = valueVecType.getNumElements();
      valueElementType = valueVecType.getElementType();
    }
    // guranteed by op verifier that vector length is a factor of memref size
    int64_t numValues = numElements / valueItems;
    int64_t iterLen = ((numValues + blockSize - 1) / blockSize) * valueItems;

    threadsToMemrefTrBuilder.pad(lowerNameRefs[0],
                                 {0, blockSize * iterLen - numElements});
    TransformMapAttr pad = threadsToMemrefTrBuilder.get();

    threadsToMemrefTrBuilder =
        BottomUpTMBuilder::above(threadsToMemrefTrBuilder, pad);
    threadsToMemrefTrBuilder.unmerge({"tid", "iter"}, {0, 1}, lowerNameRefs[0],
                                     {blockSize, iterLen});
    TransformMapAttr unmerge = threadsToMemrefTrBuilder.get();

    gpu::AddressSpaceAttr privateMemoryAddressSpace =
        rewriter.getAttr<gpu::AddressSpaceAttr>(
            gpu::GPUDialect::getPrivateAddressSpace());
    MemRefType valueRegType = MemRefType::get(
        valueItems, valueElementType, AffineMap{}, privateMemoryAddressSpace);
    GpuAllocOp valueReg = GpuAllocOp::create(rewriter, loc, valueRegType);
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    InBoundsStoreOp::create(rewriter, loc, val, valueReg, zero);
    Value tid =
        rewriter.createOrFold<rock::WorkitemIdOp>(loc, rewriter.getIndexType());
    ThreadwiseWriteAllOp::create(rewriter, loc, valueReg, op.getMemref(),
                                 rewriter.getArrayAttr({unmerge, pad}),
                                 /*extraIndices=*/ValueRange{tid},
                                 StoreMethod::Set, true, true);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

// The structure of this lowing is documented at
// https://github.com/ROCm/rocMLIR/issues/719
struct BlockwiseGemmRewritePattern
    : public OpConversionPattern<BlockwiseGemmOp> {
  using OpConversionPattern<BlockwiseGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockwiseGemmOp op,
                                BlockwiseGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Prepare some useful constants.
    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);

    MemRefType blockAType = op.getMatrixA().getType(),
               blockBType = op.getMatrixB().getType(),
               bufferCType = op.getMatrixC().getType();

    auto elementType = bufferCType.getElementType();

    int64_t k = blockAType.getShape()[0];
    int64_t m = blockAType.getShape()[1];
    int64_t n = blockBType.getShape()[1];
    int64_t kPack = blockAType.getShape()[2];

    // Non-accelerator path.

    // Obtain critical attributes.
    int64_t mC = bufferCType.getShape()[0];
    int64_t nC = bufferCType.getShape()[1];

    GeneralGemmParamsAttr params = op.getParams();
    uint32_t blockSize = params.getBlockSize();
    int64_t kPerThread = params.getKPerThread();
    int64_t mPerThread = params.getMPerThread();
    int64_t nPerThread = params.getNPerThread();

    GeneralGemmBlockStructure blockStructure =
        *deriveGeneralGemmBlockStructure(blockSize);

    int64_t mThreadsPerCuwave = blockStructure.mThreadsPerCuwave;
    int64_t nThreadsPerCuwave = blockStructure.nThreadsPerCuwave;
    int64_t cuwaveLen = mThreadsPerCuwave * nThreadsPerCuwave;

    int64_t mCuwavesPerBlock = blockStructure.mCuwavesPerBlock;
    int64_t nCuwavesPerBlock = blockStructure.nCuwavesPerBlock;
    int64_t numCuwaves = mCuwavesPerBlock * nCuwavesPerBlock;
    int64_t derivedBlockSize = numCuwaves * cuwaveLen;
    assert(blockSize == derivedBlockSize &&
           "block structure parameters must multiply to block size");

    int64_t mRepeat = mC / mPerThread;
    int64_t nRepeat = nC / nPerThread;

    if (mRepeat * mCuwavesPerBlock * mThreadsPerCuwave * mPerThread != m)
      return op.emitOpError("The m turing attributes don't multiply to M_LDS");
    if (nRepeat * nCuwavesPerBlock * nThreadsPerCuwave * nPerThread != n)
      return op.emitOpError("The n turing parameters don't multiply to N_LDS");

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << m << "\n"
               << "mRepeat: " << mRepeat << "\n"
               << "mCuwavesPerBlock: " << mCuwavesPerBlock << "\n"
               << "mThreadsPerCuwave: " << mThreadsPerCuwave << "\n"
               << "mPerThread: " << mPerThread << "\n"
               << "n: " << n << "\n"
               << "nRepeat: " << nRepeat << "\n"
               << "nCuwavesPerBlock: " << nCuwavesPerBlock << "\n"
               << "nThreadsPerCuwave: " << nThreadsPerCuwave << "\n"
               << "nPerThread: " << nPerThread << "\n");

    auto ldsTidSplitter = [&](StringRef repeatName, int64_t repeatLen,
                              StringRef perThreadName,
                              int64_t perThreadLen) -> TopDownTMBuilder {
      TopDownTMBuilder splitTidForLDS(
          b, {"k", repeatName, "tid", perThreadName, "kpack"},
          {k, repeatLen, blockSize, perThreadLen, kPack}, loc);
      splitTidForLDS.passThrough({"k", repeatName});
      splitTidForLDS.merge({"m_cuwaves", "n_cuwaves", "m_cuwave", "n_cuwave"},
                           {2, 3, 4, 5}, "tid",
                           {mCuwavesPerBlock, nCuwavesPerBlock,
                            mThreadsPerCuwave, nThreadsPerCuwave});
      splitTidForLDS.passThrough({perThreadName, "kpack"}, {6, 7},
                                 {perThreadName, "kpack"});
      return splitTidForLDS;
    };

    int64_t copyMPerThread = op.getInMPerThread();
    int64_t copyNPerThread = op.getInNPerThread();

    TopDownTMBuilder splitTidA =
        ldsTidSplitter("m_repeat", mRepeat, "m_thread", mPerThread);
    TransformMapAttr splitTidAAttr = splitTidA.get();
    auto toLdsIndexA = TopDownTMBuilder::below(splitTidA, splitTidAAttr);
    toLdsIndexA.passThrough("k");
    toLdsIndexA.unmerge(
        "m", 1, {"m_repeat", "m_cuwaves", "m_cuwave", "m_thread"},
        {mRepeat, mCuwavesPerBlock, mThreadsPerCuwave, mPerThread});
    toLdsIndexA.ignore("n_cuwaves");
    toLdsIndexA.ignore("n_cuwave");
    toLdsIndexA.passThrough({"kpack"}, {2}, {"kpack"});
    TransformMapAttr toLdsIndexAAttr = toLdsIndexA.get();
    SmallVector<Attribute> transformAttrsA{splitTidAAttr, toLdsIndexAAttr};

    // If the dimension `m` has been rotated to minimize bank conflicts we want
    // to apply the same rotation reading from LDS. This rotation happens in
    // `wrapLDSforStore` from
    // mlir/lib/Dialect/Rock/Transforms/GridwiseGemmToBlockwise.cpp which needs
    // to be kept in sync with this function
    int64_t strideA = (kPack == 1 ? copyMPerThread : 1);
    rotateIf(op.getRotateMWithK(), toLdsIndexA, toLdsIndexAAttr, strideA, "m",
             m, 1, "k", k, {"k"}, {"kpack"}, transformAttrsA);

    TopDownTMBuilder splitTidB =
        ldsTidSplitter("n_repeat", nRepeat, "n_thread", nPerThread);
    TransformMapAttr splitTidBAttr = splitTidB.get();
    auto toLdsIndexB = TopDownTMBuilder::below(splitTidB, splitTidBAttr);
    toLdsIndexB.passThrough("k");
    toLdsIndexB.unmerge(
        "n", 1, {"n_repeat", "n_cuwaves", "n_cuwave", "n_thread"},
        {nRepeat, nCuwavesPerBlock, nThreadsPerCuwave, nPerThread});
    toLdsIndexB.ignore("m_cuwaves");
    toLdsIndexB.ignore("m_cuwave");
    toLdsIndexB.passThrough({"kpack"}, {2}, {"kpack"});
    TransformMapAttr toLdsIndexBAttr = toLdsIndexB.get();
    SmallVector<Attribute> transformAttrsB{splitTidBAttr, toLdsIndexBAttr};

    // If the dimension `d` has been rotated to minimize bank conflicts we want
    // to apply the same rotation reading from LDS. This rotation happens in
    // `wrapLDSforStore` from
    // mlir/lib/Dialect/Rock/Transforms/GridwiseGemmToBlockwise.cpp which needs
    // to be kept in sync with this function
    int64_t strideB = (kPack == 1 ? copyNPerThread : 1);
    rotateIf(op.getRotateNWithK(), toLdsIndexB, toLdsIndexBAttr, strideB, "n",
             n, 1, "k", k, {"k"}, {"kpack"}, transformAttrsB);

    Value matrixA, matrixB;
    ArrayAttr transformsA, transformsB;
    bool ldsANeedsi64, ldsBNeedsi64;
    std::tie(matrixA, transformsA, ldsANeedsi64) =
        untransform(b, adaptor.getMatrixA(), b.getArrayAttr(transformAttrsA));
    std::tie(matrixB, transformsB, ldsBNeedsi64) =
        untransform(b, adaptor.getMatrixB(), b.getArrayAttr(transformAttrsB));
    if (ldsANeedsi64 || ldsBNeedsi64)
      return b.notifyMatchFailure(loc, "LDS map can't need 64-bit indexing");

    int64_t threadANumRegisters = kPerThread * mC * kPack;
    int64_t threadBNumRegisters = kPerThread * nC * kPack;

    // Alloc register for thread_a and thread_b.
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    auto threadARegisterMemRefType =
        MemRefType::get(threadANumRegisters, elementType, AffineMap{},
                        privateMemoryAddressSpace);
    auto threadAAllocOp = GpuAllocOp::create(b, loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType =
        MemRefType::get(threadBNumRegisters, elementType, AffineMap{},
                        privateMemoryAddressSpace);
    auto threadBAllocOp = GpuAllocOp::create(b, loc, threadBRegisterMemRefType);

    // Define views of register tiles for copies
    BottomUpTMBuilder viewA(b, {"raw"}, {threadANumRegisters}, loc);
    viewA.unmerge({"k", "m_repeat", "tid", "m_thread", "kpack"},
                  {0, 1, 2, 3, 4}, "raw",
                  {kPerThread, mRepeat, 1, mPerThread, kPack});
    TransformMapAttr threadACopyViewAttr = viewA.get();

    BottomUpTMBuilder viewB(b, {"raw"}, {threadBNumRegisters}, loc);
    viewB.unmerge({"k", "n_repeat", "tid", "n_thread", "kpack"},
                  {0, 1, 2, 3, 4}, "raw",
                  {kPerThread, nRepeat, 1, nPerThread, kPack});
    TransformMapAttr threadBCopyViewAttr = viewB.get();

    // Main loop.
    Value workitem = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());
    LLVM_DEBUG(llvm::dbgs() << "Outer loop:\n "
                            << "k =  " << k << "\n"
                            << " kPerThread = " << kPerThread << "\n");
    auto loopOp =
        b.replaceOpWithNewOp<affine::AffineForOp>(op, 0, k, kPerThread);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loopOp.getBody());
    Value kOffset = loopOp.getInductionVar();

    SmallVector<Value, 5> registerStartCoords(5, zeroConstantOp);
    SmallVector<Value, 5> ldsBufferAStartCoords = {
        kOffset, zeroConstantOp, workitem, zeroConstantOp, zeroConstantOp};
    auto copyALoop = TransformingForOp::create(
        b, loc,
        ArrayRef<ValueRange>{ldsBufferAStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsA, b.getArrayAttr(threadACopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, mRepeat, 1, mPerThread, kPack},
        /*strides=*/std::nullopt, /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard copyAGuard(b);
      b.setInsertionPointToStart(copyALoop.getBody());
      Value aCopy = memref::LoadOp::create(
          b, loc, matrixA, copyALoop.getLowerCoords(/*domain=*/0));
      Value aCast = createTypeConversionOp(b, loc, aCopy, elementType);
      memref::StoreOp::create(b, loc, aCast, threadAAllocOp,
                              copyALoop.getLowerCoords(/*domain=*/1));
    }

    SmallVector<Value, 5> ldsBufferBStartCoords = {
        kOffset, zeroConstantOp, workitem, zeroConstantOp, zeroConstantOp};
    auto copyBLoop = TransformingForOp::create(
        b, loc,
        ArrayRef<ValueRange>{ldsBufferBStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsB, b.getArrayAttr(threadBCopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, nRepeat, 1, nPerThread, kPack},
        /*strides=*/std::nullopt, /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard copyBGuard(b);
      b.setInsertionPointToStart(copyBLoop.getBody());
      Value bCopy = memref::LoadOp::create(
          b, loc, matrixB, copyBLoop.getLowerCoords(/*domain=*/0));
      Value bCast = createTypeConversionOp(b, loc, bCopy, elementType);
      memref::StoreOp::create(b, loc, bCast, threadBAllocOp,
                              copyBLoop.getLowerCoords(/*domain=*/1));
    }

    Value reshapedARegisters = reshapeBuffer(
        b, loc, threadAAllocOp, {"k", "m", "kpack"}, {kPerThread, mC, kPack});
    Value reshapedBRegisters = reshapeBuffer(
        b, loc, threadBAllocOp, {"k", "n", "kpack"}, {kPerThread, nC, kPack});
    // Actually do the gemm - this goes inside the look over kOffset
    ThreadwiseGemmOp::create(b, loc, reshapedARegisters, reshapedBRegisters,
                             op.getMatrixC());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemmAccel lowering.
//===----------------------------------------------------------------------===//
struct BlockwiseGemmAccelRewritePattern
    : public OpConversionPattern<BlockwiseGemmAccelOp> {
  using OpConversionPattern<BlockwiseGemmAccelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockwiseGemmAccelOp op,
                                BlockwiseGemmAccelOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    StringAttr arch = rock::getArchValue(op);
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    int64_t kpackPerBlock = tuningParams.getKpackPerBlock();
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();
    bool loadAFromLDS = adaptor.getMatrixA() != nullptr;
    bool loadBFromLDS = adaptor.getMatrixB() != nullptr;
    BlockwiseMatrixParamsAttr matrixParamsA = op.getMatrixParamsA();
    BlockwiseMatrixParamsAttr matrixParamsB = op.getMatrixParamsB();
    Value scaleA = adaptor.getBufferScaleA();
    Value scaleB = adaptor.getBufferScaleB();

    bool isScaledGemm = (scaleA != Value{} && scaleB != Value{});
    Type dataTypeA = matrixParamsA.getElementType();
    Type dataTypeB = matrixParamsB.getElementType();

    rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
    GemmFeatures features = archInfo.defaultFeatures;
    auto accelEmitterPtr = rock::accel::AccelEmitter::select(
        features, dataTypeA, dataTypeB, arch, tuningParams);

    if (!accelEmitterPtr)
      return op.emitOpError("Unable to emit accelerator code.");

    // Extract relevant accelerator parameters
    rock::accel::AccelEmitterParams params = accelEmitterPtr->getParams();
    Type argTypeA = params.argTypeA;
    Type argTypeB = params.argTypeB;
    int64_t mRepeats = params.mRepeats;
    int64_t nRepeats = params.nRepeats;
    int64_t kBase = params.kBase;
    int64_t kBasePerThread = params.kBasePerThread;
    int64_t kPerBlock = kpackPerBlock * tuningParams.getKpack();

    auto tid = WorkitemIdOp::create(b, loc, b.getIndexType());

    // Build LDS-transpose config attribute using decision from
    // GridwiseGemmToBlockwise
    auto buildTransposeAttr = [&](bool isOperandA) -> LDSTransposeConfigAttr {
      const auto &matrixParams = isOperandA ? matrixParamsA : matrixParamsB;

      // Check if LDS transpose is enabled for this operand
      if (!matrixParams.getLdsTransposeEnabled())
        return nullptr;

      // Get accelerator dimensions from matrix params and tuning params
      // accelDDim = accelDDim (for MFMA instructions with blocksMfma=1)
      // accelKDim = accelKDim from BlockwiseMatrixParamsAttr
      int64_t accelDDim = matrixParams.getAccelDDim();
      int64_t accelKDim = matrixParams.getAccelKDim();

      if (accelDDim <= 0 || accelKDim <= 0)
        return nullptr;

      // Build transpose config attribute using precomputed accelerator geometry
      // Note: doubleBuffering=false because this lowering pass operates in
      // single-buffer mode.
      return hwtranspose::buildTransposeAttrFromParams(
          b, accelDDim, accelKDim, mPerBlock, nPerBlock, kPerBlock, mPerWave,
          nPerWave,
          /*doubleBuffering=*/false, isOperandA);
    };

    LDSTransposeConfigAttr transposeAttrA =
        buildTransposeAttr(/*isOperandA=*/true);
    LDSTransposeConfigAttr transposeAttrB =
        buildTransposeAttr(/*isOperandA=*/false);

    LLVM_DEBUG(llvm::dbgs()
               << "argVectorType A: " << argTypeA << "\n"
               << "argVectorType B: " << argTypeB << "\n"
               << "kBase: " << kBase << "\n"
               << "mPerWave: " << mPerWave << "\n"
               << "nPerWave: " << nPerWave << "\n"
               << "mRepeat: " << mRepeats << "\n"
               << "nRepeat: " << nRepeats << "\n"
               << "kBasePerThread: " << kBasePerThread << "\n"
               << "kpackPerBlock: " << kpackPerBlock << "\n"
               << "loadAFromLDS: " << loadAFromLDS << "\n"
               << "loadBFromLDS: " << loadBFromLDS << "\n"
               << "rotateMWithK: " << matrixParamsA.getRotateDWithK() << "\n"
               << "rotateNWithK: " << matrixParamsB.getRotateDWithK() << "\n"
               << "bufferA type: " << adaptor.getBufferA().getType() << "\n"
               << "bufferB type: " << adaptor.getBufferB().getType() << "\n");

    // The following loop nest hardcodes the following loop schedule:
    //
    // for(index_t m_i = 0; m_i < mRepeats; ++m_i)
    //   regsA = threadwise_readinto[m_i, :]
    //   for(index_t n_i = 0; n_i<nRepeats; ++n_i)
    //       regsB = threadwise_readint[n_i, :]
    //       threadwise_gemm(regsA, regsB)
    //
    // Which mimics:
    // https://github.com/ROCm/composable_kernel/blob/develop/include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp#L304
    //
    // Please note that different schedules might exist, so this can be
    // considered a temporary hack until we have a proper way of "searching"
    // through different schedules (either heuristically or automatically)

    // Determine if the other operand uses LDS transpose load
    // This is needed to select the correct K access pattern for regular loads
    bool bUsesLdsTranspose = matrixParamsB.getLdsTransposeEnabled();
    bool aUsesLdsTranspose = matrixParamsA.getLdsTransposeEnabled();

    Value wrappedLDSBufferForLoadA, wrappedLDSBufferForLoadB;
    if (loadAFromLDS) {
      // When loading A, check if B uses transpose load
      wrappedLDSBufferForLoadA = accelEmitterPtr->wrapLDSBufferForLoad(
          b, loc, op.getMatrixA(), matrixParamsA, op.getBlockSize(), "m",
          /*useLdsTransposeLoad=*/bUsesLdsTranspose);
    }
    if (loadBFromLDS) {
      // When loading B, check if A uses transpose load
      wrappedLDSBufferForLoadB = accelEmitterPtr->wrapLDSBufferForLoad(
          b, loc, op.getMatrixB(), matrixParamsB, op.getBlockSize(), "n",
          /*useLdsTransposeLoad=*/aUsesLdsTranspose);
    }
    Value wrappedLDSBufferForScaleA, wrappedLDSBufferForScaleB;
    if (isScaledGemm) {
      // Scaled GEMM (FP4) doesn't support LDS transpose load yet
      if (loadAFromLDS) {
        wrappedLDSBufferForScaleA = accelEmitterPtr->wrapLDSBufferForLoad(
            b, loc, op.getScaleA(), matrixParamsA, op.getBlockSize(), "m",
            /*useLdsTransposeLoad=*/false);
      }
      if (loadBFromLDS) {
        wrappedLDSBufferForScaleB = accelEmitterPtr->wrapLDSBufferForLoad(
            b, loc, op.getScaleB(), matrixParamsB, op.getBlockSize(), "n",
            /*useLdsTransposeLoad=*/false);
      }
    }

    auto loadBuffer =
        [&](Value buffer, Value wrappedLDSBufferForLoad, Value loopVar,
            Type argType, int64_t repeats, bool loadFromLDS, bool directToLDS,
            bool isA, LDSTransposeConfigAttr transposeAttr = nullptr) -> Value {
      Value inputBuffer = buffer;
      SmallVector<int64_t> shape;
      if (directToLDS) {
        shape.push_back(kBasePerThread);
        auto memrefType = cast<MemRefType>(buffer.getType());
        assert(memrefType.getRank() == 1);
        assert(memrefType.getElementType() == b.getI8Type());
        int64_t numBytes = getPackedByteSize(kBasePerThread, argType);
        if (memrefType.getShape()[0] > numBytes) {
          assert(memrefType.getShape()[0] == numBytes * repeats);
          shape.insert(shape.begin(), repeats);
        } else {
          assert(memrefType.getShape()[0] == numBytes);
        }
        // view for generateThreadwiseViewBuffer()
        buffer = viewBufferAs(b, buffer, argType, shape);
      }

      if (loadFromLDS) {
        Value viewForReadInto = buffer;
        if (directToLDS) {
          SmallVector<int64_t> shapeForLoad(shape);
          if (auto vectorType = dyn_cast<VectorType>(argType)) {
            assert(vectorType.hasRank() == 1 && "Expected rank 1");
            shapeForLoad[shapeForLoad.size() - 1] =
                vectorType.getDimSize(0) *
                shapeForLoad[shapeForLoad.size() - 1];
          }
          viewForReadInto = viewBufferAs(
              b, inputBuffer, getElementTypeOrSelf(argType), shapeForLoad);
        }
        assert(wrappedLDSBufferForLoad != Value{} &&
               "Wrapped LDS buffer for load is empty");
        // regs = read from LDS
        ThreadwiseReadIntoOp::create(
            b, loc, wrappedLDSBufferForLoad, viewForReadInto,
            b.getArrayAttr({}), ValueRange{tid, loopVar}, /*forceUnroll=*/true,
            /*useIndexDiffs=*/true,
            /*ldsTransposeConfig=*/transposeAttr);
      } else {
        if (cast<ShapedType>(buffer.getType()).getRank() == 1) {
          StringRef dk = isA ? "mk" : "nk";
          StringRef indexStr = isA ? "iidx" : "jidx";
          BottomUpTMBuilder regsBuilder(b, {dk}, {repeats * kBasePerThread},
                                        loc);
          regsBuilder.unmerge({indexStr, "k"}, {0, 1}, dk,
                              {repeats, kBasePerThread});
          buffer =
              rock::transform(b, buffer, b.getArrayAttr({regsBuilder.get()}));
        }
        buffer = rock::createSliceOfFirstDim(b, loc, buffer, loopVar);
      }
      return buffer;
    };

    auto mLoop = affine::AffineForOp::create(b, loc, 0, mRepeats);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(mLoop.getBody());
      Value i = mLoop.getInductionVar();

      Value bufferA = adaptor.getBufferA();
      bufferA = loadBuffer(
          bufferA, wrappedLDSBufferForLoadA, i, argTypeA, mRepeats,
          loadAFromLDS, matrixParamsA.getDirectToLDS(), true, transposeAttrA);
      Value viewA =
          accelEmitterPtr->generateThreadwiseViewBufferA(b, loc, bufferA);
      Value viewScaleA = nullptr, viewScaleB = nullptr;
      if (isScaledGemm) {
        if (matrixParamsA.getDirectToLDS()) {
          op->emitOpError("Direct to LDS scaled GEMM is not supported yet.");
          return failure();
        }
        Value bufferScaleA = adaptor.getBufferScaleA();
        bufferScaleA =
            loadBuffer(bufferScaleA, wrappedLDSBufferForScaleA, i,
                       getElementTypeOrSelf(scaleA), mRepeats, loadAFromLDS,
                       matrixParamsA.getDirectToLDS(), true, nullptr);
        viewScaleA = accelEmitterPtr->generateThreadwiseViewBufferA(
            b, loc, bufferScaleA);
      }

      auto nLoop = affine::AffineForOp::create(b, loc, 0, nRepeats);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(nLoop.getBody());
        Value j = nLoop.getInductionVar();

        Value bufferB = adaptor.getBufferB();
        bufferB =
            loadBuffer(bufferB, wrappedLDSBufferForLoadB, j, argTypeB, nRepeats,
                       loadBFromLDS, matrixParamsB.getDirectToLDS(), false,
                       transposeAttrB);
        Value viewB =
            accelEmitterPtr->generateThreadwiseViewBufferB(b, loc, bufferB);
        if (isScaledGemm) {
          if (matrixParamsB.getDirectToLDS()) {
            op->emitOpError("Direct to LDS scaled GEMM is not supported yet.");
            return failure();
          }
          Value bufferScaleB = adaptor.getBufferScaleB();
          bufferScaleB =
              loadBuffer(bufferScaleB, wrappedLDSBufferForScaleB, j,
                         getElementTypeOrSelf(scaleB), nRepeats, loadBFromLDS,
                         matrixParamsB.getDirectToLDS(), false, nullptr);
          viewScaleB = accelEmitterPtr->generateThreadwiseViewBufferB(
              b, loc, bufferScaleB);
        }

        // regsC += regsA * regsB
        auto kLoop = affine::AffineForOp::create(b, loc, 0, kBasePerThread);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(kLoop.getBody());
          Value viewC = accelEmitterPtr->generateThreadwiseViewBufferC(
              b, loc, adaptor.getMatrixC());
          Value k = kLoop.getInductionVar();
          ThreadwiseGemmAccelOp::create(b, loc, viewA, viewB, viewC, viewScaleA,
                                        viewScaleB, ValueRange{i, j, k},
                                        op.getFeaturesAttr(), tuningParams);
        }
      }
    }
    b.eraseOp(op);
    return success();
  }
};

namespace {
struct ThreadwiseReadIntoRewritePattern
    : public OpConversionPattern<ThreadwiseReadIntoOp> {
  using OpConversionPattern<ThreadwiseReadIntoOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseReadIntoOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};

struct ThreadwiseWriteAllRewritePattern
    : public OpConversionPattern<ThreadwiseWriteAllOp> {
  using OpConversionPattern<ThreadwiseWriteAllOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseWriteAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BlockwiseReduce lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseReduceRewritePattern
    : public OpConversionPattern<BlockwiseBroadcastReduceOp> {
  using OpConversionPattern<BlockwiseBroadcastReduceOp>::OpConversionPattern;

  int64_t calculateNonReductionDimProduct(ArrayRef<int64_t> toReduceShape,
                                          int64_t axis) const {
    int64_t dimProduct = 1;
    for (size_t i = 0; i < toReduceShape.size(); i++) {
      if (i != (size_t)axis) {
        dimProduct *= toReduceShape[i];
      }
    }
    return dimProduct;
  }

  // This function will make a 2d view from a multi-dimensional tensors
  // where one axis needs to be reduced.
  ArrayAttr createInput2DView(Location loc, PatternRewriter &rewriter,
                              ArrayAttr regTensorView, int64_t reduceAxis,
                              bool makeRDimZero = false) const {
    TransformMapAttr lowestTr =
        cast<TransformMapAttr>(regTensorView[regTensorView.size() - 1]);
    ArrayRef<int64_t> lowestShape = lowestTr.getLowerBounds().asArrayRef();
    TopDownTMBuilder tensorToLDSViewBuilder(rewriter, lowestShape, loc);
    SmallVector<StringRef, 4> upperNameRefs;
    tensorToLDSViewBuilder.getStartNames(upperNameRefs);

    SmallVector<StringRef, 4> nonReduceNameRefs;
    SmallVector<unsigned, 4> nonReduceDims;
    SmallVector<int64_t, 4> nonReduceDimSizes;
    for (auto [dim, dimSize] : llvm::enumerate(lowestShape)) {
      if (dim != (size_t)reduceAxis) {
        nonReduceNameRefs.push_back(upperNameRefs[dim]);
        nonReduceDims.push_back(dim);
        nonReduceDimSizes.push_back(dimSize);
      }
    }
    tensorToLDSViewBuilder.unmerge("nrDim", 0, nonReduceNameRefs,
                                   nonReduceDimSizes);
    if (makeRDimZero) {
      tensorToLDSViewBuilder.constDim("rDim", 1, 0, lowestShape[reduceAxis]);
    } else {
      tensorToLDSViewBuilder.passThrough({"rDim"}, {1},
                                         {upperNameRefs[reduceAxis]});
    }
    TransformMapAttr twoDimLDSView = tensorToLDSViewBuilder.get();
    return prependUpperViews(rewriter, regTensorView,
                             rewriter.getArrayAttr({twoDimLDSView}));
  }

  ArrayAttr create2DToFlatLDSView(Location loc, PatternRewriter &rewriter,
                                  int64_t dim0, int64_t dim1) const {
    TopDownTMBuilder toLDSViewBuilder(rewriter, {dim0, dim1}, loc);
    SmallVector<StringRef, 4> upperNameRefs;
    toLDSViewBuilder.getStartNames(upperNameRefs);
    toLDSViewBuilder.unmerge("flatDim", 0, upperNameRefs, {dim0, dim1});
    return rewriter.getArrayAttr({toLDSViewBuilder.get()});
  }

  // This function will append views to target a flat LDS buffer
  // where non-reduction dims are laid contiguously as they are expected
  // function on parallel.
  ArrayAttr createLDSWorkspaceView(
      Location loc, PatternRewriter &rewriter, ArrayAttr regTensorView,
      int64_t reduceAxis, bool makeRDimZero = false,
      std::optional<int64_t> rDimZeroLen = std::nullopt) const {

    TransformMapAttr lowestTr =
        cast<TransformMapAttr>(regTensorView[regTensorView.size() - 1]);
    ArrayRef<int64_t> lowestShape = lowestTr.getLowerBounds().asArrayRef();
    TopDownTMBuilder tensorToLDSViewBuilder(rewriter, lowestShape, loc);
    SmallVector<StringRef, 4> upperNameRefs;
    tensorToLDSViewBuilder.getStartNames(upperNameRefs);
    int64_t rDimLen = rDimZeroLen.value_or(lowestShape[reduceAxis]);

    int64_t nonReduceMergeDimSize = 1;
    SmallVector<StringRef, 4> nonReduceNameRefs;
    SmallVector<unsigned, 4> nonReduceDims;
    SmallVector<int64_t, 4> nonReduceDimSizes;
    for (auto [dim, dimSize] : llvm::enumerate(lowestShape)) {
      if (dim != (size_t)reduceAxis) {
        nonReduceMergeDimSize *= dimSize;
        nonReduceNameRefs.push_back(upperNameRefs[dim]);
        nonReduceDims.push_back(dim);
        nonReduceDimSizes.push_back(dimSize);
      }
    }
    tensorToLDSViewBuilder.unmerge("nrDim", 0, nonReduceNameRefs,
                                   nonReduceDimSizes);
    if (makeRDimZero) {
      tensorToLDSViewBuilder.constDim("rDim", 1, 0, rDimLen);
    } else {
      tensorToLDSViewBuilder.passThrough({"rDim"}, {1},
                                         {upperNameRefs[reduceAxis]});
    }
    TransformMapAttr twoDimLDSView = tensorToLDSViewBuilder.get();

    TopDownTMBuilder flatLDSViewBuilder =
        TopDownTMBuilder::below(tensorToLDSViewBuilder, twoDimLDSView);
    flatLDSViewBuilder.unmerge("flatDim", 0, {"nrDim", "rDim"},
                               {nonReduceMergeDimSize, rDimLen});
    TransformMapAttr flatLDSView = flatLDSViewBuilder.get();
    SmallVector<Attribute> threadsToLDSViewAttrs;
    for (Attribute trMap : regTensorView) {
      threadsToLDSViewAttrs.push_back(trMap);
    }
    threadsToLDSViewAttrs.push_back(twoDimLDSView);
    threadsToLDSViewAttrs.push_back(flatLDSView);
    return rewriter.getArrayAttr(threadsToLDSViewAttrs);
  }

  // This should only be used if product non-reduction dims is
  // equal or larger than number threads in a block.
  //
  // Given a input tensor : D0, ... , Dr , ... , DN to reduce,
  // This function creates a view that maps the space of
  // [D0, ... , Dr , ... , DN] --> [tid, nrIter, rIter] where
  // tid is threads within the block, nrIter is non-reducing
  // iterations within a thread and rIter is reducing iterations
  // within a thread.
  ArrayAttr createThreadViewForNRLargerThanThreads(
      Location loc, ArrayRef<int64_t> toReduceShape, int64_t blockSize,
      int64_t reduceAxis, PatternRewriter &rewriter) const {
    BottomUpTMBuilder threadsToTensor(rewriter, toReduceShape, loc);
    SmallVector<StringRef, 4> lowerNameRefs;
    threadsToTensor.getStartNames(lowerNameRefs);

    int64_t nonReduceMergeDimSize = 1;
    SmallVector<StringRef, 4> nonReduceNameRefs;
    for (auto dimAndSize : llvm::enumerate(toReduceShape)) {
      int64_t dim = dimAndSize.index();
      int64_t dimSize = dimAndSize.value();
      if (dim != reduceAxis) {
        nonReduceMergeDimSize *= dimSize;
        nonReduceNameRefs.push_back(lowerNameRefs[dim]);
      }
    }
    threadsToTensor.merge("nrDim", 0, nonReduceNameRefs);
    threadsToTensor.passThrough({"rIter"}, {1}, {lowerNameRefs[reduceAxis]});
    TransformMapAttr mergeTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, mergeTrMap);
    int64_t nrThreads = (nonReduceMergeDimSize + (blockSize - 1)) / blockSize;
    threadsToTensor.pad({"nrDim"},
                        {0, blockSize * nrThreads - nonReduceMergeDimSize});
    threadsToTensor.passThrough({"rIter"}, {1}, {"rIter"});
    TransformMapAttr padTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, padTrMap);
    threadsToTensor.unmerge({"tid", "nrIter"}, {0, 1}, "nrDim",
                            {blockSize, nrThreads});
    threadsToTensor.passThrough({"rIter"}, {2}, {"rIter"});
    TransformMapAttr unmergeTrMap = threadsToTensor.get();

    return rewriter.getArrayAttr({unmergeTrMap, padTrMap, mergeTrMap});
  }

  // This should only be used if product non-reduction dims is
  // less than number threads in a block.
  //
  // Given a input tensor : D0, ... , Dr , ... , DN to reduce,
  // This function creates a view that maps the space of
  // [D0, ... , Dr , ... , DN] --> [nrtid, rtid, rIter] where
  // nrtid = tid / product(non-reduction dims) is a reduction subgroup leader.
  // rtid = tid % product(non-reduction dims) is thread idx within a reduction
  // subgroup. Size of the dimension 'rtid' is the number of threads
  // that'd participate in the reduction
  ArrayAttr createThreadViewforNRSmallerThanThreads(
      Location loc, ArrayRef<int64_t> toReduceShape, int64_t blockSize,
      size_t reduceAxis, PatternRewriter &rewriter) const {
    BottomUpTMBuilder threadsToTensor(rewriter, toReduceShape, loc);
    SmallVector<StringRef, 4> lowerNameRefs;
    threadsToTensor.getStartNames(lowerNameRefs);
    int64_t nonReduceMergeDimSize = 1;
    SmallVector<StringRef, 4> nonReduceNameRefs;
    for (auto [dim, dimSize] : llvm::enumerate(toReduceShape)) {
      if (dim != reduceAxis) {
        nonReduceMergeDimSize *= dimSize;
        nonReduceNameRefs.push_back(lowerNameRefs[dim]);
      }
    }
    threadsToTensor.merge("nrDim", 0, nonReduceNameRefs);
    threadsToTensor.passThrough({"rDim"}, {1}, {lowerNameRefs[reduceAxis]});
    TransformMapAttr mergeTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, mergeTrMap);
    // If this function is being called, then the number of threads is larger
    // than the product of non reduction dimensions. Therefore, we create thread
    // groups (rthreads) per a point in merge(non reduction dimensions).
    int64_t rthreads = blockSize / nonReduceMergeDimSize;

    // Find the largest rthreads that evenly divides rDimSize to avoid LDS
    // aliasing: when rthreads * ceil(rDimSize/rthreads) > rDimSize, padded
    // positions alias into adjacent rows in the flat LDS layout.
    while (rthreads > 1 && toReduceShape[reduceAxis] % rthreads != 0) {
      rthreads--;
    }

    int64_t rDimPerRThread =
        (toReduceShape[reduceAxis] + (rthreads - 1)) / rthreads;
    threadsToTensor.pad(
        {"rDim"}, {0, rthreads * rDimPerRThread - toReduceShape[reduceAxis]});
    threadsToTensor.passThrough({"nrDim"}, {0}, {"nrDim"});
    TransformMapAttr padTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, padTrMap);
    threadsToTensor.unmerge({"rtid", "rIter"}, {1, 2}, "rDim",
                            {rthreads, rDimPerRThread});
    threadsToTensor.passThrough({"nrtid"}, {0}, {"nrDim"});
    TransformMapAttr unmergeTrMap = threadsToTensor.get();

    return rewriter.getArrayAttr({unmergeTrMap, padTrMap, mergeTrMap});
  }

  Value getReductionInitValue(BlockwiseBroadcastReduceOp op,
                              ConversionPatternRewriter &rewriter) const {
    ReduceMethod rMethod = op.getReduceMethod();
    Type elementType = op.getInput().getType().getElementType();
    if (elementType.isIntOrIndex()) {
      if (rMethod == ReduceMethod::Sum) {
        return createConstantIntOp(rewriter, op.getLoc(), elementType,
                                   elementType, 0);
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        return createConstantIntOp(rewriter, op.getLoc(), elementType,
                                   elementType,
                                   std::numeric_limits<int64_t>::min());
      }
    } else {
      if (rMethod == ReduceMethod::Sum) {
        return createConstantFloatOp(rewriter, op.getLoc(), elementType,
                                     elementType, 0.0);
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        return createConstantFloatOp(rewriter, op.getLoc(), elementType,
                                     elementType,
                                     -std::numeric_limits<float>::infinity());
      }
    }
  }

  Value createReducingOp(BlockwiseBroadcastReduceOp op, Value input, Value acc,
                         OpBuilder &builder) const {
    ReduceMethod rMethod = op.getReduceMethod();
    Location loc = op.getLoc();
    // Value loadAcc = InBoundsLoadOp::create(rewriter, loc, input.getType(),
    // acc, zeroConstantOp);
    Type elementType = op.getInput().getType().getElementType();

    if (!isa<VectorType>(acc.getType()) && isa<VectorType>(input.getType())) {
      // This means accumulator is a scalar type and input is a vector type,
      // therefore its a elementwise reduction between two operands.
      vector::CombiningKind kind;
      if (rMethod == ReduceMethod::Sum) {
        kind = vector::CombiningKind::ADD;
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        if (elementType.isIntOrIndex()) {
          kind = vector::CombiningKind::MAXIMUMF;
        } else {
          kind = vector::CombiningKind::MAXIMUMF;
        }
      }
      input = vector::ReductionOp::create(builder, loc, kind, input);
    }

    if (rMethod == ReduceMethod::Sum) {
      Value reduced;
      if (elementType.isIntOrIndex()) {
        reduced = arith::AddIOp::create(builder, loc, acc, input);
      } else {
        reduced = arith::AddFOp::create(builder, loc, acc, input);
      }
      return reduced;
    } else {
      assert(rMethod == ReduceMethod::Max);
      Value reduced;
      if (elementType.isIntOrIndex()) {
        reduced = arith::MaxSIOp::create(builder, loc, acc, input);
      } else {
        reduced = arith::MaximumFOp::create(builder, loc, acc, input);
      }
      return reduced;
    }
  }

  ArrayAttr createReducedView(PatternRewriter &rewriter, Location loc,
                              ArrayAttr subTileView, int64_t axis) const {
    ArrayRef<int64_t> threadSubTileShape = getLowerShape(subTileView);
    TopDownTMBuilder viewBuilder(rewriter, threadSubTileShape, loc);
    for (auto [dim, dimSize] : llvm::enumerate(threadSubTileShape)) {
      if ((int64_t)dim == axis) {
        viewBuilder.constDim("rDim", dim, 0, dimSize);
      } else {
        viewBuilder.passThrough({(unsigned int)dim}, {(unsigned int)dim});
      }
    }
    TransformMapAttr redDimZeroMap = viewBuilder.get();
    ArrayAttr reducedView = prependUpperViews(
        rewriter, subTileView, rewriter.getArrayAttr({redDimZeroMap}));
    return reducedView;
  }

  // Perform threadwise reductions based thread subtile
  // view and store the reduced data to reduced buffer
  void doThreadwiseReductions(PatternRewriter &rewriter, Location loc,
                              BlockwiseBroadcastReduceOp op,
                              Value reducedBuffer,
                              ArrayAttr inputThreadSubTile2dView) const {
    Value inputRawBuffer = op.getInput();
    int64_t numElements =
        cast<MemRefType>(inputRawBuffer.getType()).getNumElements();
    constexpr size_t nrDim = 0;

    Type elemType = cast<MemRefType>(inputRawBuffer.getType()).getElementType();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto loop = TransformingForOp::create(
        rewriter, loc, ArrayRef<ValueRange>{{zero}, {zero}},
        ArrayRef<Attribute>{inputThreadSubTile2dView,
                            rewriter.getArrayAttr({})},
        /*bounds=*/ArrayRef<int64_t>{numElements},
        /*strides=*/ArrayRef<int64_t>{1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Block::BlockArgListType upperCoords = loop.getLowerCoords(1);
      Block::BlockArgListType subtileCoords = loop.getLowerCoords(0);
      Value ldInput = InBoundsLoadOp::create(rewriter, loc, elemType,
                                             inputRawBuffer, upperCoords);
      Value ldInputAcc = InBoundsLoadOp::create(
          rewriter, loc, elemType, reducedBuffer, subtileCoords[nrDim]);
      Value reduced = createReducingOp(op, ldInput, ldInputAcc, rewriter);
      InBoundsStoreOp::create(rewriter, loc, reduced, reducedBuffer,
                              subtileCoords[nrDim]);
    }
  }

  // This function store partial reductions to LDS for
  // inter-thread reductions later on.
  void storePartialReductionstoLDS(PatternRewriter &rewriter, Location loc,
                                   Value reducedBuffer, Value ldsBuffer,
                                   ArrayAttr inputBlockSubTile2dView,
                                   ArrayAttr inputThreadSubTile2dView,
                                   ArrayAttr tidSubTileSliceView,
                                   ArrayAttr toFlatLDSView) const {
    Type elemType = cast<MemRefType>(reducedBuffer.getType()).getElementType();
    constexpr size_t nrDim = 0;
    constexpr size_t rDim = 1;
    FailureOr<ArrayAttr> maybeInputThreadSubTile2dViewInv =
        invertTransforms(rewriter, loc, inputThreadSubTile2dView);
    assert(succeeded(maybeInputThreadSubTile2dViewInv) &&
           "inputThreadSubTile2dView must be invertible");
    ArrayAttr inputThreadSubTile2dViewInv =
        maybeInputThreadSubTile2dViewInv.value();
    ArrayRef<int64_t> threadSubTile2DShape =
        getLowerShape(inputThreadSubTile2dView);
    WorkitemIdOp tid =
        WorkitemIdOp::create(rewriter, loc, rewriter.getIndexType());
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // First we iterate thread subtile along non-reduction
    // axis to get iter coordinate within the register
    auto loop = TransformingForOp::create(
        rewriter, loc, ArrayRef<ValueRange>{{zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{inputThreadSubTile2dViewInv,
                            rewriter.getArrayAttr({})},
        /*bounds=*/ArrayRef<int64_t>{threadSubTile2DShape[nrDim], 1},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value iter = loop.getLowerCoords(0)[0];
      Block::BlockArgListType threadSubTile2DCoords = loop.getLowerCoords(1);

      // Then we plug that iter coordinate along with tid to recover block
      // subtile coordinates. However, we only need non-reduction dimension
      // coordinate from the block subtile.
      auto convertToBlockSubTile = TransformingForOp::create(
          rewriter, loc, ArrayRef<ValueRange>{{tid, iter}},
          ArrayRef<Attribute>{inputBlockSubTile2dView},
          /*bounds=*/ArrayRef<int64_t>{1, 1},
          /*strides=*/ArrayRef<int64_t>{1, 1},
          /*forceUnroll=*/true, /*useIndexDiffs=*/true);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(convertToBlockSubTile.getBody());
        Value blockNrDimCoord = convertToBlockSubTile.getLowerCoords(0)[nrDim];
        Value ldReduced =
            InBoundsLoadOp::create(rewriter, loc, elemType, reducedBuffer,
                                   ValueRange{threadSubTile2DCoords[0]});

        // Here we plug the tid to get the sliced block subtile coordinate find
        // a unique packed coordinate in the reduction axis per each thread to
        // write the partial reductions to the lds.
        auto convertToBlockSubTileTidSlice = TransformingForOp::create(
            rewriter, loc, ArrayRef<ValueRange>{{tid}},
            ArrayRef<Attribute>{tidSubTileSliceView},
            /*bounds=*/ArrayRef<int64_t>{1},
            /*strides=*/ArrayRef<int64_t>{1},
            /*forceUnroll=*/true, /*useIndexDiffs=*/true);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(
              convertToBlockSubTileTidSlice.getBody());
          Value blockTidSliceRDimCoord =
              convertToBlockSubTileTidSlice.getLowerCoords(0)[rDim];
          auto ldsStoreloop = TransformingForOp::create(
              rewriter, loc,
              ArrayRef<ValueRange>{{blockNrDimCoord, blockTidSliceRDimCoord}},
              ArrayRef<Attribute>{toFlatLDSView},
              /*bounds=*/ArrayRef<int64_t>{1, 1},
              /*strides=*/ArrayRef<int64_t>{1, 1},
              /*forceUnroll=*/true, /*useIndexDiffs=*/true);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ldsStoreloop.getBody());
            Block::BlockArgListType ldsFlatCoords =
                ldsStoreloop.getLowerCoords(0);
            InBoundsStoreOp::create(rewriter, loc, ldReduced, ldsBuffer,
                                    ldsFlatCoords);
          }
        }
      }
    }
  }

  LogicalResult
  matchAndRewrite(BlockwiseBroadcastReduceOp op,
                  BlockwiseBroadcastReduceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // inputView should be register {bid, tid, iter} to virtual tensor {bid, d0,
    // ... , Dr , ... , dn} coords transforms where Dr is the reduction axis.
    ArrayAttr inputViewArrayAttr = op.getInputRegViewAttr();
    TypedValue<MemRefType> inputReg = op.getInput();
    TypedValue<MemRefType> outputReg = op.getOutput();
    Type elemType = inputReg.getType().getElementType();
    TypedValue<MemRefType> workspaceLDSBuffer = op.getWorkspaceBuffer();
    Value zeroConstantOp = arith::ConstantIndexOp::create(rewriter, loc, 0);
    int64_t axis = op.getAxis().getSExtValue();
    int64_t blockSize = op.getBlockSize();
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    // Get current workitem ID.
    WorkitemIdOp tid =
        WorkitemIdOp::create(rewriter, loc, rewriter.getIndexType());

    // Create strides and bounds to iterate the virtual tensor
    TransformMapAttr lowerTr = cast<TransformMapAttr>(
        inputViewArrayAttr[inputViewArrayAttr.size() - 1]);
    ArrayRef<int64_t> lowerTrLowerBounds =
        lowerTr.getLowerBounds().asArrayRef();
    SmallVector<int64_t, 4> regTensorShape =
        llvm::to_vector<4>(lowerTrLowerBounds);

    // 2DView is alwasy nrDim x rdim
    constexpr size_t nrDim = 0;
    constexpr size_t rDim = 1;
    ArrayAttr inputThreadSubTile2dView =
        createInput2DView(loc, rewriter, op.getIterSubTileSliceView(), axis);
    ArrayRef<int64_t> inputThreadSubTile2dShape =
        getLowerShape(inputThreadSubTile2dView);
    auto partialReductionBufferType =
        MemRefType::get(inputThreadSubTile2dShape[nrDim], elemType, AffineMap{},
                        privateMemoryAddressSpace);
    Value partialReductionBuffer =
        GpuAllocOp::create(rewriter, loc, partialReductionBufferType);
    Value initVal = getReductionInitValue(op, rewriter);
    FillOp::create(rewriter, loc, partialReductionBuffer, initVal);
    doThreadwiseReductions(rewriter, loc, op, partialReductionBuffer,
                           inputThreadSubTile2dView);

    // Create partially reduced tensor shape
    ArrayAttr inputBlockSubTile2dView =
        createInput2DView(loc, rewriter, inputViewArrayAttr, axis);
    SmallVector<int64_t, 2> partialRegTensorShape =
        llvm::to_vector<2>(getLowerShape(inputBlockSubTile2dView));
    ArrayAttr tidSubTileSliceView =
        createInput2DView(loc, rewriter, op.getTidSubTileSliceView(), axis);
    ArrayRef<int64_t> partialReductionuctionLower2DShape =
        getLowerShape(tidSubTileSliceView);
    partialRegTensorShape[rDim] = partialReductionuctionLower2DShape[rDim];
    ArrayAttr toFlatLDSView =
        create2DToFlatLDSView(loc, rewriter, partialRegTensorShape[nrDim],
                              partialRegTensorShape[rDim]);
    storePartialReductionstoLDS(rewriter, loc, partialReductionBuffer,
                                workspaceLDSBuffer, inputBlockSubTile2dView,
                                inputThreadSubTile2dView, tidSubTileSliceView,
                                toFlatLDSView);

    LDSBarrierOp::create(rewriter, loc);
    // Following RAII scope will create reduction loops.
    {
      int64_t nonReductionDimSizeProduct = partialRegTensorShape[nrDim];
      if (blockSize <= nonReductionDimSizeProduct) {
        // This means there aren't enough threads to do a parallel reduction
        // each individual thread could do its own reduction.
        ArrayAttr threadsToTensorTrs = createThreadViewForNRLargerThanThreads(
            loc, partialRegTensorShape, blockSize, rDim, rewriter);
        ArrayAttr threadToLDSViewTrs =
            createLDSWorkspaceView(loc, rewriter, threadsToTensorTrs, rDim);
        ArrayAttr threadsToLDSViewReducedTrs = createLDSWorkspaceView(
            loc, rewriter, threadsToTensorTrs, rDim, /*makeRDimZero-*/ true);
        ArrayRef<int64_t> threadViewShape =
            cast<TransformMapAttr>(threadToLDSViewTrs[0]).getUpperBounds();
        constexpr size_t nrIterDim = 1;
        constexpr size_t rIterDim = 2;

        // Note: This currently creates a bunch of dead IR because vectorization
        // needs access to a `Value` in order to account for scalarized buffers.
        Value threadToLDSViewed =
            transform(rewriter, workspaceLDSBuffer, threadToLDSViewTrs);
        VectorizationResult nrIterVectorRes =
            getMaxVectorization(threadToLDSViewed, nrIterDim);
        int64_t nrIterVectorLen = nrIterVectorRes.max;
        // Create the accumulation register
        // This will be accumulated over non-reduction iterations.
        auto accRegType = MemRefType::get(
            nrIterVectorLen, elemType, AffineMap{}, privateMemoryAddressSpace);
        Value accReg = GpuAllocOp::create(rewriter, loc, accRegType);
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          Value nrIter;
          if (threadViewShape[nrIterDim] > 1) {
            AffineForOp nrIterLoop = AffineForOp::create(
                rewriter, loc, 0, threadViewShape[nrIterDim], nrIterVectorLen);
            // inside the loop.
            rewriter.setInsertionPointToStart(nrIterLoop.getBody());
            nrIter = nrIterLoop.getInductionVar();
          } else {
            nrIter = zeroConstantOp;
          }
          FillOp::create(rewriter, loc, accReg, initVal);
          VectorizationResult rIterVectorRes =
              getMaxVectorization(threadToLDSViewed, rIterDim);
          int64_t rIterVectorLen = rIterVectorRes.max;
          SmallVector<Value, 4> inits{tid, nrIter, zeroConstantOp};
          SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
          SmallVector<int64_t> strides{1, 1, rIterVectorLen};

          TransformingForOp reductionLoop = TransformingForOp::create(
              rewriter, loc, ArrayRef<ValueRange>{inits, inits, inits},
              ArrayRef<Attribute>{threadToLDSViewTrs, rewriter.getArrayAttr({}),
                                  threadsToLDSViewReducedTrs},
              ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
              /*forceUnroll=*/true,
              /*useIndexDiffs=*/true);
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reductionLoop.getBody());
            Block::BlockArgListType LDSLoadCoords =
                reductionLoop.getLowerCoords(/*domain=*/0);
            // There are two vectorization scenarios :
            // 1) rIterVectorLen > 1 &&  nrIterVectorLen == 1
            //    Here we will have a load vector and accReg that is a scalar
            //    The code in createReducingOp will vector reduce it before
            //    doing a reducing store to accReg
            // 2) nrIterVectorLen > 1 && rIterVectorLen == 1
            //    Here we will have a load vector and accReg that is also a
            //    vector The code in createReducingOp will do vector elementwise
            //    op and store the resulting vector to accReg.
            // NOTE: currently, LDS is viewed as [nrDim x rDim] therefore
            // only scenario 1) is exercised. However, we'd like to keep
            // this code compatible with both approaches for future changes.
            Value loadVal = InBoundsLoadOp::create(
                rewriter, loc,
                vectorTypeOrSelf(elemType,
                                 std::max(rIterVectorLen, nrIterVectorLen)),
                workspaceLDSBuffer, LDSLoadCoords);
            Value loadAcc = InBoundsLoadOp::create(
                rewriter, loc, vectorTypeOrSelf(elemType, nrIterVectorLen),
                accReg, zeroConstantOp);
            Value reduced = createReducingOp(op, loadVal, loadAcc, rewriter);
            InBoundsStoreOp::create(rewriter, loc, reduced, accReg,
                                    zeroConstantOp);
            // Storing the last reduction iter output directly to LDS[..., dr=0,
            // ...]
            Value rIterArg =
                reductionLoop.getLowerCoords(/*domain=*/1)[rIterDim];
            Value boundVal = arith::ConstantIndexOp::create(
                rewriter, loc, threadViewShape[rIterDim]);
            Value strideVal =
                arith::ConstantIndexOp::create(rewriter, loc, rIterVectorLen);
            Value lastIterVal =
                arith::SubIOp::create(rewriter, loc, boundVal, strideVal);
            Value isLastIter = arith::CmpIOp::create(
                rewriter, loc, arith::CmpIPredicate::eq, rIterArg, lastIterVal);
            scf::IfOp ifb = scf::IfOp::create(rewriter, loc, isLastIter,
                                              /*withElseRegion=*/false);
            {
              OpBuilder thenb = ifb.getThenBodyBuilder();
              InBoundsStoreOp::create(
                  thenb, loc, reduced, workspaceLDSBuffer,
                  reductionLoop.getLowerCoords(/*domain=*/2));
            }
          }
        }
        ArrayAttr reducedldsViewArrayAttr = createLDSWorkspaceView(
            loc, rewriter, inputViewArrayAttr, axis, /*makeRDimZero-*/ true,
            partialRegTensorShape[rDim]);
        LDSBarrierOp::create(rewriter, loc);
        ThreadwiseReadIntoOp::create(rewriter, loc, workspaceLDSBuffer,
                                     outputReg, reducedldsViewArrayAttr,
                                     /*extraIndices=*/ValueRange{tid}, true,
                                     false);
        if (ArrayAttr outputViewArrayAttr = op.getExtraOutViewAttr()) {
          ArrayAttr reducedldsViewArrayAttr2 = createLDSWorkspaceView(
              loc, rewriter, outputViewArrayAttr, axis, /*makeRDimZero-*/ true,
              partialRegTensorShape[rDim]);
          ThreadwiseReadIntoOp::create(
              rewriter, loc, workspaceLDSBuffer, op.getExtraOut(),
              reducedldsViewArrayAttr2,
              /*extraIndices=*/ValueRange{tid}, true, false);
        }
      } else {
        // This means there are more threads than elements to be reduced.
        ArrayAttr threadToTensorViewTrs =
            createThreadViewforNRSmallerThanThreads(loc, partialRegTensorShape,
                                                    blockSize, rDim, rewriter);
        ArrayAttr threadToLDSViewTrs =
            createLDSWorkspaceView(loc, rewriter, threadToTensorViewTrs, rDim);
        ArrayRef<int64_t> threadViewShape =
            cast<TransformMapAttr>(threadToLDSViewTrs[0]).getUpperBounds();
        constexpr size_t rTidDim = 1;
        constexpr size_t rIterDim = 2;

        Value threadToLDSViewed =
            transform(rewriter, workspaceLDSBuffer, threadToLDSViewTrs);
        VectorizationResult rIterVectorRes =
            getMaxVectorization(threadToLDSViewed, rIterDim);
        int64_t rIterVectorLen = rIterVectorRes.max;
        Value nrDimSizeProductConst = arith::ConstantIndexOp::create(
            rewriter, loc, nonReductionDimSizeProduct);
        Value rtid =
            arith::DivSIOp::create(rewriter, loc, tid, nrDimSizeProductConst);
        Value nrtid =
            arith::RemSIOp::create(rewriter, loc, tid, nrDimSizeProductConst);

        // We need to do the threadwise reduction
        // here only if rIterDim is meaninfully iterated
        // otherwise this step can be skipped.
        if (threadViewShape[rIterDim] > 1) {
          // This is where thread_wise reduction result is stored.
          Type loadTypeInputReg = vectorTypeOrSelf(elemType, rIterVectorLen);
          Type accRegType = MemRefType::get({1}, elemType, AffineMap{},
                                            privateMemoryAddressSpace);
          Value accReg = GpuAllocOp::create(rewriter, loc, accRegType);
          // This RAII scope would create a loop to iteratively partialy reduce
          // on a thread basis until items to reduce will match the available
          // number of threads.
          {
            SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
            SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
            SmallVector<int64_t> strides{1, 1, rIterVectorLen};

            Value initVal = getReductionInitValue(op, rewriter);
            FillOp::create(rewriter, loc, accReg, initVal);

            TransformingForOp reductionLoop = TransformingForOp::create(
                rewriter, loc, ArrayRef<ValueRange>(inits),
                ArrayRef<Attribute>{threadToLDSViewTrs},
                ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                /*forceUnroll=*/true, /*useIndexDiffs=*/true);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(reductionLoop.getBody());
              Block::BlockArgListType LDSLoadCoords =
                  reductionLoop.getLowerCoords(/*domain=*/0);
              Value loadVal =
                  InBoundsLoadOp::create(rewriter, loc, loadTypeInputReg,
                                         workspaceLDSBuffer, LDSLoadCoords);
              Value loadAcc = InBoundsLoadOp::create(rewriter, loc, elemType,
                                                     accReg, zeroConstantOp);
              Value reduced = createReducingOp(op, loadVal, loadAcc, rewriter);
              InBoundsStoreOp::create(rewriter, loc, reduced, accReg,
                                      zeroConstantOp);
            }
          }

          // This RAII scope would store the partial reductions to
          // LDS
          {
            SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
            SmallVector<int64_t> bounds{1, 1, 1};
            SmallVector<int64_t> strides{1, 1, 1};

            TransformingForOp reductionLoop = TransformingForOp::create(
                rewriter, loc, ArrayRef<ValueRange>(inits),
                ArrayRef<Attribute>{threadToLDSViewTrs},
                ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                /*forceUnroll=*/true, /*useIndexDiffs=*/true);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(reductionLoop.getBody());
              Block::BlockArgListType LDSStoreCoords =
                  reductionLoop.getLowerCoords(/*domain=*/0);
              Value loadVal = InBoundsLoadOp::create(rewriter, loc, elemType,
                                                     accReg, zeroConstantOp);
              InBoundsStoreOp::create(rewriter, loc, loadVal,
                                      workspaceLDSBuffer, LDSStoreCoords);
            }
            LDSBarrierOp::create(rewriter, loc);
          }
        }

        // Branchless reduction: each thread reads ALL rTidDim partial
        // values from LDS and reduces locally in registers. This avoids
        // creating conditional branches (scf.if) that split softmax into
        // multiple basic blocks. Without branches, the LLVM backend
        // scheduler can keep V global loads (issued before softmax) in
        // the same basic block, enabling sched_barrier to prevent them
        // from being sunk past softmax computation.
        //
        // Trade-off: every thread does rTidCount LDS reads (instead of
        // log2(rTidCount) conditional reads in the tree reduction). For
        // typical attention configs where rTidCount is small (e.g., 4),
        // this is negligible overhead.
        {
          int64_t rTidCount = threadViewShape[rTidDim];

          // Accumulator for the full reduction.
          auto accRegType = MemRefType::get(
              {1}, elemType, AffineMap{}, privateMemoryAddressSpace);
          Value accReg = GpuAllocOp::create(rewriter, loc, accRegType);
          FillOp::create(rewriter, loc, accReg, initVal);

          // Read all rTidCount partial values from LDS and reduce.
          // Every thread with the same nrtid computes the identical
          // fully-reduced value.
          for (int64_t i = 0; i < rTidCount; i++) {
            Value iVal = arith::ConstantIndexOp::create(rewriter, loc, i);
            SmallVector<Value, 3> readInits{nrtid, iVal, zeroConstantOp};
            SmallVector<int64_t> bounds{1, 1, 1};
            SmallVector<int64_t> strides{1, 1, 1};

            TransformingForOp readLoop = TransformingForOp::create(
                rewriter, loc, ArrayRef<ValueRange>{readInits},
                ArrayRef<Attribute>{threadToLDSViewTrs},
                ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                /*forceUnroll=*/true, /*useIndexDiffs=*/true);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(readLoop.getBody());
              Block::BlockArgListType ldsCoords =
                  readLoop.getLowerCoords(/*domain=*/0);
              Value ldVal = InBoundsLoadOp::create(
                  rewriter, loc, elemType, workspaceLDSBuffer, ldsCoords);
              Value accVal = InBoundsLoadOp::create(
                  rewriter, loc, elemType, accReg, zeroConstantOp);
              Value reduced = createReducingOp(op, ldVal, accVal, rewriter);
              InBoundsStoreOp::create(rewriter, loc, reduced, accReg,
                                      zeroConstantOp);
            }
          }

          // Write the fully reduced value back to LDS at [nrtid, 0].
          // All threads with the same nrtid compute the same value,
          // so concurrent writes to the same location are safe.
          {
            Value reducedVal = InBoundsLoadOp::create(
                rewriter, loc, elemType, accReg, zeroConstantOp);
            SmallVector<Value, 3> writeInits{nrtid, zeroConstantOp,
                                             zeroConstantOp};
            SmallVector<int64_t> writeBounds{1, 1, 1};
            SmallVector<int64_t> writeStrides{1, 1, 1};

            TransformingForOp writeLoop = TransformingForOp::create(
                rewriter, loc, ArrayRef<ValueRange>{writeInits},
                ArrayRef<Attribute>{threadToLDSViewTrs},
                ArrayRef<int64_t>(writeBounds), ArrayRef<int64_t>(writeStrides),
                /*forceUnroll=*/true, /*useIndexDiffs=*/true);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(writeLoop.getBody());
              Block::BlockArgListType ldsCoords =
                  writeLoop.getLowerCoords(/*domain=*/0);
              InBoundsStoreOp::create(rewriter, loc, reducedVal,
                                      workspaceLDSBuffer, ldsCoords);
            }
          }

          LDSBarrierOp::create(rewriter, loc);
          ArrayAttr reducedldsViewArrayAttr = createLDSWorkspaceView(
              loc, rewriter, inputViewArrayAttr, axis, /*makeRDimZero-*/ true,
              partialRegTensorShape[rDim]);
          ThreadwiseReadIntoOp::create(rewriter, loc, workspaceLDSBuffer,
                                       outputReg, reducedldsViewArrayAttr,
                                       /*extraIndices=*/ValueRange{tid}, true,
                                       false);
          if (ArrayAttr outputViewArrayAttr = op.getExtraOutViewAttr()) {
            ArrayAttr reducedldsViewArrayAttr2 = createLDSWorkspaceView(
                loc, rewriter, outputViewArrayAttr, axis,
                /*makeRDimZero-*/ true, partialRegTensorShape[rDim]);
            ThreadwiseReadIntoOp::create(
                rewriter, loc, workspaceLDSBuffer, op.getExtraOut(),
                reducedldsViewArrayAttr2,
                /*extraIndices=*/ValueRange{tid}, true, false);
          }
        }
      }
      rewriter.eraseOp(op);
      return success();
    }
  }
};

void RockLowerBlockwiseGemmToThreadwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  {
    ConversionTarget writeAllTarget(*ctx);
    writeAllTarget.addIllegalOp<BlockwiseBroadcastReduceOp, BlockwiseFillOp>();
    writeAllTarget.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                                   memref::MemRefDialect, scf::SCFDialect,
                                   vector::VectorDialect, AffineDialect>();
    writeAllTarget.addLegalOp<gpu::PrintfOp>();
    RewritePatternSet writeAllPatterns(ctx);
    writeAllPatterns
        .add<BlockwiseReduceRewritePattern, BlockwiseFillRewritePattern>(ctx);
    if (failed(applyPartialConversion(getOperation(), writeAllTarget,
                                      std::move(writeAllPatterns))))
      signalPassFailure();
  }

  ConversionTarget target(*ctx);
  target.addIllegalOp<FillOp, BlockwiseGemmOp, BlockwiseGemmAccelOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         affine::AffineDialect, vector::VectorDialect,
                         memref::MemRefDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, BlockwiseGemmRewritePattern,
               BlockwiseGemmAccelRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
