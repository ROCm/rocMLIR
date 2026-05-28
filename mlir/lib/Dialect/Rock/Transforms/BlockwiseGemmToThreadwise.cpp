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

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
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

  // Extract m_tid and n_tid counts from the tid slice view's Merge transform.
  static std::pair<int64_t, int64_t>
  getPerWaveThreadCounts(ArrayAttr tidSliceView) {
    if (tidSliceView.empty())
      return {0, 0};
    TransformMapAttr firstMap = cast<TransformMapAttr>(tidSliceView[0]);
    for (TransformAttr tr : firstMap.getOps()) {
      if (tr.getType() != TransformType::Merge)
        continue;
      ArrayRef<StringRef> lowerNames = tr.getLowerNames();
      ArrayRef<int64_t> params = tr.getParams();
      int64_t mTid = 0, nTid = 0;
      for (auto [name, param] : llvm::zip(lowerNames, params)) {
        if (name == "m_tid")
          mTid = param;
        else if (name == "n_tid")
          nTid = param;
      }
      if (mTid > 0 && nTid > 0)
        return {mTid, nTid};
    }
    return {0, 0};
  }

  // Pack a scalar into i32 for cross-lane transfer. Cross-lane intrinsics
  // only move bits, so we avoid value-converting ops (extf/truncf → v_cvt)
  // and use bitcast+zext instead (free register ops).
  //   32-bit float: bitcast f32 → i32
  //   32-bit int:   identity
  //   sub-32-bit:   bitcast fN/iN → iN, then zext iN → i32
  Value toI32(ConversionPatternRewriter &rewriter, Location loc,
              Value val) const {
    auto i32Type = rewriter.getI32Type();
    Type ty = val.getType();
    unsigned bitWidth = ty.getIntOrFloatBitWidth();
    if (bitWidth == 32) {
      if (ty.isIntOrIndex())
        return val;
      return arith::BitcastOp::create(rewriter, loc, i32Type, val);
    }
    auto iNType = rewriter.getIntegerType(bitWidth);
    Value asInt =
        ty.isIntOrIndex()
            ? val
            : Value(arith::BitcastOp::create(rewriter, loc, iNType, val));
    return arith::ExtUIOp::create(rewriter, loc, i32Type, asInt);
  }

  // Unpack i32 back to the original element type (inverse of toI32).
  //   32-bit float: bitcast i32 → f32
  //   32-bit int:   identity
  //   sub-32-bit:   trunc i32 → iN, then bitcast iN → fN/iN
  Value fromI32(ConversionPatternRewriter &rewriter, Location loc, Value word,
                Type origType) const {
    unsigned bitWidth = origType.getIntOrFloatBitWidth();
    if (bitWidth == 32) {
      if (origType.isIntOrIndex())
        return word;
      return arith::BitcastOp::create(rewriter, loc, origType, word);
    }
    auto iNType = rewriter.getIntegerType(bitWidth);
    Value truncated = arith::TruncIOp::create(rewriter, loc, iNType, word);
    if (origType.isIntOrIndex())
      return truncated;
    return arith::BitcastOp::create(rewriter, loc, origType, truncated);
  }

  // Cross-half-wave reduction via v_permlanex16_var_b32 (wave32 only).
  // Lane i exchanges with lane i+16 and reduces.
  void permlaneX16VarReduce(ConversionPatternRewriter &rewriter, Location loc,
                            Value partialReductionBuffer, Value tid,
                            int64_t nrDimSize, int64_t waveSize, Type elemType,
                            BlockwiseBroadcastReduceOp op) const {
    auto i32Type = rewriter.getI32Type();
    Value lane = arith::RemUIOp::create(
        rewriter, loc, tid,
        arith::ConstantIndexOp::create(rewriter, loc, waveSize));
    Value laneI32 = arith::IndexCastOp::create(rewriter, loc, i32Type, lane);
    Value laneIdxInHalf = arith::AndIOp::create(
        rewriter, loc, laneI32,
        arith::ConstantIntOp::create(rewriter, loc, i32Type, 15));

    for (int64_t i = 0; i < nrDimSize; i++) {
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value myVal = InBoundsLoadOp::create(rewriter, loc, elemType,
                                           partialReductionBuffer, idx);
      Value partnerVal = amdgpu::PermlaneVarOp::create(
          rewriter, loc, elemType, myVal, laneIdxInHalf,
          /*cross=*/true, /*fetch_inactive=*/false, /*bound_ctrl=*/false);
      Value reduced = createReducingOp(op, partnerVal, myVal, rewriter);
      InBoundsStoreOp::create(rewriter, loc, reduced, partialReductionBuffer,
                              idx);
    }
  }

  // One permlane-swap reduction step over a per-thread buffer (gfx950 wave64).
  //   swapWidth == 32 -> v_permlane32_swap_b32 (lane i <-> lane i+32)
  //   swapWidth == 16 -> v_permlane16_swap_b32 (lane i <-> lane i+16)
  // The intrinsic returns a {vdst0, vdst1} pair that always contains
  // {self, partner} for every lane (only the order differs across the two
  // halves of the swap-pair). Since our reductions are commutative+associative
  // (Sum / Max) we reduce vdst0 and vdst1 directly, avoiding the cmp+select
  // that would otherwise be needed to identify which is the partner.
  void permlaneSwapReduceStep(ConversionPatternRewriter &rewriter, Location loc,
                              Value buffer, int64_t numElements, Type elemType,
                              int64_t swapWidth,
                              BlockwiseBroadcastReduceOp op) const {
    assert((swapWidth == 16 || swapWidth == 32) &&
           "permlaneSwapReduceStep only supports swap widths 16 and 32");
    auto i32Type = rewriter.getI32Type();
    auto i32PairType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                        {i32Type, i32Type});
    for (int64_t i = 0; i < numElements; i++) {
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value myVal =
          InBoundsLoadOp::create(rewriter, loc, elemType, buffer, idx);
      Value word = toI32(rewriter, loc, myVal);
      Value swapResult = (swapWidth == 32)
                             ? Value(ROCDL::Permlane32SwapOp::create(
                                   rewriter, loc, i32PairType, word, word,
                                   /*fi=*/false, /*boundControl=*/false))
                             : Value(ROCDL::Permlane16SwapOp::create(
                                   rewriter, loc, i32PairType, word, word,
                                   /*fi=*/false, /*boundControl=*/false));
      Value vdst0 = fromI32(
          rewriter, loc,
          LLVM::ExtractValueOp::create(rewriter, loc, swapResult, 0), elemType);
      Value vdst1 = fromI32(
          rewriter, loc,
          LLVM::ExtractValueOp::create(rewriter, loc, swapResult, 1), elemType);
      Value reduced = createReducingOp(op, vdst0, vdst1, rewriter);
      InBoundsStoreOp::create(rewriter, loc, reduced, buffer, idx);
    }
  }

  // Register-only cross-lane reduction over `numElements` slots on gfx950+
  // (wave64) for partner-group sizes 2 or 4:
  //   groupSize == 2: one v_permlane32_swap (lanes 16/32 apart)
  //   groupSize == 4: v_permlane16_swap then v_permlane32_swap
  // After the call, every lane in a partner group holds the fully reduced
  // value for its own nr-positions. Pass `numElements == 1` to reduce a
  // scalar accumulator (used by the NR-Small permlane fast-path).
  void permlaneSwapReduce(ConversionPatternRewriter &rewriter, Location loc,
                          Value buffer, int64_t numElements, int64_t groupSize,
                          Type elemType, BlockwiseBroadcastReduceOp op) const {
    if (groupSize == 4) {
      permlaneSwapReduceStep(rewriter, loc, buffer, numElements, elemType,
                             /*swapWidth=*/16, op);
    }
    permlaneSwapReduceStep(rewriter, loc, buffer, numElements, elemType,
                           /*swapWidth=*/32, op);
  }

  // --- ds_swizzle + ds_bpermute reduction (CDNA wave64) ---
  //
  // Register-only cross-lane reduction for gfx908 (MI100), gfx90a (MI250),
  // and gfx94x (MI300) using two DS permute instructions:
  //   - ds_swizzle_b32: XOR within each 32-lane half (immediate offset, no
  //     address computation). Used for groupSize == 4 as the first step.
  //   - ds_bpermute_b32: XOR 32 across the two 32-lane halves. Byte address
  //     (tid * 4) ^ 128 maps to lane (tid % 64) ^ 32 via the identity
  //     bit 7 in byte-space == bit 5 in lane-space with 256-byte wrapping.
  //
  // Equivalent to permlaneSwapReduce on gfx950 but available on all CDNA.

  void dsSwizzleReduceStep(ConversionPatternRewriter &rewriter, Location loc,
                           Value buffer, int64_t numElements, Type elemType,
                           int64_t xorDistance,
                           BlockwiseBroadcastReduceOp op) const {
    auto i32Type = rewriter.getI32Type();
    auto andMask = rewriter.getI32IntegerAttr(0x1F);
    auto orMask = rewriter.getI32IntegerAttr(0);
    auto xorMask = rewriter.getI32IntegerAttr(xorDistance);

    for (int64_t i = 0; i < numElements; i++) {
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value myVal =
          InBoundsLoadOp::create(rewriter, loc, elemType, buffer, idx);
      Value word = toI32(rewriter, loc, myVal);
      Value partnerWord = amdgpu::SwizzleBitModeOp::create(
          rewriter, loc, i32Type, word, andMask, orMask, xorMask);
      Value partnerVal = fromI32(rewriter, loc, partnerWord, elemType);
      Value reduced = createReducingOp(op, myVal, partnerVal, rewriter);
      InBoundsStoreOp::create(rewriter, loc, reduced, buffer, idx);
    }
  }

  void dsBpermuteReduceStep(ConversionPatternRewriter &rewriter, Location loc,
                            Value buffer, int64_t numElements, Type elemType,
                            Value tid, BlockwiseBroadcastReduceOp op) const {
    auto i32Type = rewriter.getI32Type();
    Value tidI32 = arith::IndexCastOp::create(rewriter, loc, i32Type, tid);
    Value tidBytes = arith::ShLIOp::create(
        rewriter, loc, tidI32,
        arith::ConstantIntOp::create(rewriter, loc, i32Type, 2));
    Value byteAddr = arith::XOrIOp::create(
        rewriter, loc, tidBytes,
        arith::ConstantIntOp::create(rewriter, loc, i32Type, 128));

    for (int64_t i = 0; i < numElements; i++) {
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value myVal =
          InBoundsLoadOp::create(rewriter, loc, elemType, buffer, idx);
      Value word = toI32(rewriter, loc, myVal);
      Value partnerWord =
          ROCDL::DsBpermuteOp::create(rewriter, loc, i32Type, byteAddr, word);
      Value partnerVal = fromI32(rewriter, loc, partnerWord, elemType);
      Value reduced = createReducingOp(op, myVal, partnerVal, rewriter);
      InBoundsStoreOp::create(rewriter, loc, reduced, buffer, idx);
    }
  }

  // groupSize == 2: ds_bpermute only (XOR 32, cross-half).
  // groupSize == 4: ds_swizzle (XOR 16, within-half) then ds_bpermute (XOR 32).
  void dsSwizzleBpermuteReduce(ConversionPatternRewriter &rewriter,
                               Location loc, Value buffer, int64_t numElements,
                               int64_t groupSize, Type elemType, Value tid,
                               BlockwiseBroadcastReduceOp op) const {
    if (groupSize == 4) {
      dsSwizzleReduceStep(rewriter, loc, buffer, numElements, elemType,
                          /*xorDistance=*/16, op);
    }
    dsBpermuteReduceStep(rewriter, loc, buffer, numElements, elemType, tid, op);
  }

  // Wave32 cross-half-wave reduction via ds_swizzle_b32 (XOR=16).
  // Lanes 0-15 exchange with lanes 16-31 and reduce — functionally identical
  // to permlaneX16VarReduce but uses ds_swizzle which is available on gfx11
  // (RDNA3 / Navi3x) where v_permlanex16_var_b32 is not exposed.
  // Only supports partialR=2 (one swap step).
  void dsSwizzleReduceWave32(ConversionPatternRewriter &rewriter, Location loc,
                             Value buffer, int64_t numElements, Type elemType,
                             BlockwiseBroadcastReduceOp op) const {
    dsSwizzleReduceStep(rewriter, loc, buffer, numElements, elemType,
                        /*xorDistance=*/16, op);
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
        // Op verifier guarantees this.
        assert(rMethod == ReduceMethod::Max);
        unsigned bitWidth = elementType.getIntOrFloatBitWidth();
        int64_t signedMin = APInt::getSignedMinValue(bitWidth).getSExtValue();
        return createConstantIntOp(rewriter, op.getLoc(), elementType,
                                   elementType, signedMin);
      }
    } else {
      if (rMethod == ReduceMethod::Sum) {
        return createConstantFloatOp(rewriter, op.getLoc(), elementType,
                                     elementType, 0.0);
      } else {
        // Op verifier guarantees this.
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
        // Op verifier guarantees this.
        assert(rMethod == ReduceMethod::Max);
        if (elementType.isIntOrIndex()) {
          kind = vector::CombiningKind::MAXSI;
        } else {

          kind = vector::CombiningKind::MAXNUMF;
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
        // Use MaxNumFOp (not MaximumFOp) so that NaN does not propagate
        // through the max reduction.
        reduced = arith::MaxNumFOp::create(builder, loc, acc, input);
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

  // Broadcasts fully reduced per-thread values from partialReductionBuffer
  // into outputReg, replacing the LDS write+barrier+read round-trip when
  // every lane already holds its own reduced value (single-wave permlane
  // fast paths). Each outputReg slot reads partialReductionBuffer at the
  // nrDim index of its (nrIdx, rIdx) cell — the rDim coordinate is
  // broadcasted, so multiple iter positions map to the same slot.
  void readReducedResultsFromPrivateBuffer(
      ConversionPatternRewriter &rewriter, Location loc,
      Value partialReductionBuffer, TypedValue<MemRefType> outputReg,
      ArrayAttr inputThreadSubTile2dView) const {
    Type elemType = cast<MemRefType>(outputReg.getType()).getElementType();
    constexpr size_t nrDim = 0;
    constexpr size_t rDim = 1;

    FailureOr<ArrayAttr> maybeInv =
        invertTransforms(rewriter, loc, inputThreadSubTile2dView);
    assert(succeeded(maybeInv) &&
           "inputThreadSubTile2dView must be invertible");
    if (failed(maybeInv))
      return;
    ArrayAttr inputThreadSubTile2dViewInv = maybeInv.value();

    ArrayRef<int64_t> threadSubTile2DShape =
        getLowerShape(inputThreadSubTile2dView);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // domain 0 (inputThreadSubTile2dViewInv): lower coord = iter, the 1D
    //   per-thread index into outputReg.
    // domain 1 (identity): lower coords = (nrIdx, rIdx).
    auto loop = TransformingForOp::create(
        rewriter, loc, ArrayRef<ValueRange>{{zero, zero}, {zero, zero}},
        ArrayRef<Attribute>{inputThreadSubTile2dViewInv,
                            rewriter.getArrayAttr({})},
        /*bounds=*/
        ArrayRef<int64_t>{threadSubTile2DShape[nrDim],
                          threadSubTile2DShape[rDim]},
        /*strides=*/ArrayRef<int64_t>{1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value iter = loop.getLowerCoords(/*domain=*/0)[0];
      Block::BlockArgListType nrAndRCoords = loop.getLowerCoords(/*domain=*/1);
      Value nrIdx = nrAndRCoords[nrDim];

      Value reducedVal = InBoundsLoadOp::create(
          rewriter, loc, elemType, partialReductionBuffer, ValueRange{nrIdx});
      InBoundsStoreOp::create(rewriter, loc, reducedVal, outputReg,
                              ValueRange{iter});
    }
  }

  // Reads fully reduced results from LDS into output (and optional extra
  // output) registers. When withBarrier is true, an LDS barrier is inserted
  // before reading to ensure prior writes are visible to all threads.
  // Note: extraOut, when set, is assumed to be a different *layout* of the
  // same reduced scalar (e.g. for broadcast to a transposed output), not an
  // independently reduced quantity. Both output and extraOut read from the
  // same LDS positions, so this would silently miscompile if extraOut ever
  // carried a separate reduction result (e.g. argmax index).
  void readReducedResultsFromLDS(ConversionPatternRewriter &rewriter,
                                 Location loc, BlockwiseBroadcastReduceOp op,
                                 TypedValue<MemRefType> workspaceLDSBuffer,
                                 TypedValue<MemRefType> outputReg,
                                 ArrayAttr inputViewArrayAttr, int64_t axis,
                                 int64_t rDimPartialSize, Value tid,
                                 bool withBarrier) const {
    ArrayAttr reducedView =
        createLDSWorkspaceView(loc, rewriter, inputViewArrayAttr, axis,
                               /*makeRDimZero=*/true, rDimPartialSize);
    if (withBarrier)
      LDSBarrierOp::create(rewriter, loc);
    ThreadwiseReadIntoOp::create(rewriter, loc, workspaceLDSBuffer, outputReg,
                                 reducedView,
                                 /*extraIndices=*/ValueRange{tid}, true, false);
    if (ArrayAttr extraOutView = op.getExtraOutViewAttr()) {
      ArrayAttr reducedView2 =
          createLDSWorkspaceView(loc, rewriter, extraOutView, axis,
                                 /*makeRDimZero=*/true, rDimPartialSize);
      ThreadwiseReadIntoOp::create(
          rewriter, loc, workspaceLDSBuffer, op.getExtraOut(), reducedView2,
          /*extraIndices=*/ValueRange{tid}, true, false);
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

    // 2DView is always nrDim x rDim
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
    ArrayRef<int64_t> partialReductionLower2DShape =
        getLowerShape(tidSubTileSliceView);
    partialRegTensorShape[rDim] = partialReductionLower2DShape[rDim];
    ArrayAttr toFlatLDSView =
        create2DToFlatLDSView(loc, rewriter, partialRegTensorShape[nrDim],
                              partialRegTensorShape[rDim]);
    int64_t nonReductionDimSizeProduct = partialRegTensorShape[nrDim];

    StringAttr arch = rock::getArchValue(op);
    int64_t waveSize = rock::lookupArchInfo(arch).waveSize;

    int64_t partialR = partialRegTensorShape[rDim];

    // All cross-lane fast paths operate on i32 registers (permlane_swap,
    // ds_bpermute, ds_swizzle, permlanex16_var all move 32-bit values).
    // Cross-lane intrinsics (permlane, ds_swizzle, ds_bpermute) operate on
    // i32. Sub-32-bit types (f16, bf16, i8, f8) are widened to 32-bit
    // before transfer and narrowed back after; 32-bit types are bitcast
    // to i32 directly. Types wider than 32-bit (f64, i64) are not
    // supported here because rock.blockwise_broadcast_reduce currently
    // rejects them at the op verifier level. If the op is ever extended
    // to accept 64-bit types, this gate must be updated and a
    // decomposition into two i32 halves added.
    int64_t elemBitWidth = elemType.getIntOrFloatBitWidth();
    bool elemSupportedByCrossLane = (elemBitWidth <= 32);

    // v_permlanex16_var_b32 (variable selector): gfx950, gfx12 (Navi4x).
    // NOT available on gfx11 (Navi3x) which only has the immediate form.
    bool hasPermlaneVar =
        elemSupportedByCrossLane && (arch.getValue().contains("gfx950") ||
                                     arch.getValue().contains("gfx12"));

    // ds_swizzle_b32 on wave32: gfx11 (RDNA3 / Navi3x). XOR=16 swaps
    // lanes 0-15 <-> 16-31, functionally identical to permlanex16_var.
    // Used as the wave32 cross-half reduction primitive when permlaneVar
    // is not available.
    bool hasDsSwizzleWave32 = elemSupportedByCrossLane && !hasPermlaneVar &&
                              waveSize == 32 &&
                              arch.getValue().contains("gfx11");

    auto [mTidPerWave, nTidPerWave] =
        getPerWaveThreadCounts(op.getTidSubTileSliceView());
    bool has2DThreadLayout = (mTidPerWave > 0 && nTidPerWave > 0);

    // Validate that the 2D layout fully tiles the wave with no gaps
    // or overlap. Without this, XOR-based lane swaps may pair threads
    // that do not hold complementary reduction positions.
    bool layoutTilesWave =
        has2DThreadLayout && (mTidPerWave * nTidPerWave == waveSize);

    // === Wave32 NR-Small gates (blockSize > nrDimProd) ===
    // Cross-half-wave reduction for partialR=2 on wave32 when
    // mTidPerWave=2, nTidPerWave=16 (lanes 0-15 <-> 16-31).
    // - canUsePermlaneX16Var_NRSmall: via v_permlanex16_var (gfx950/gfx12)
    // - canUseDsSwizzleW32_NRSmall: via ds_swizzle XOR=16 (gfx11)
    bool canUsePermlaneX16Var_NRSmall =
        (layoutTilesWave && hasPermlaneVar && waveSize == 32 && partialR == 2 &&
         mTidPerWave == 2 && nTidPerWave == 16);
    bool canUseDsSwizzleW32_NRSmall =
        (layoutTilesWave && hasDsSwizzleWave32 && partialR == 2 &&
         mTidPerWave == 2 && nTidPerWave == 16);

    // === Wave32 NR-Large gates (blockSize <= nrDimProd) ===
    // Single cross-half-wave reduction restricted to partialR == 2
    // (== mTidPerWave on WMMA wave32): one swap step is sufficient
    // for a 2-way reduction only.
    // - canUsePermlaneX16Var_NRLarge: via v_permlanex16_var (gfx950/gfx12)
    // - canUseDsSwizzleW32_NRLarge: via ds_swizzle XOR=16 (gfx11)
    bool canUsePermlaneX16Var_NRLarge = false;
    bool canUseDsSwizzleW32_NRLarge = false;
    if (layoutTilesWave && waveSize == 32 &&
        blockSize <= nonReductionDimSizeProduct && partialR == 2 &&
        partialR == mTidPerWave) {
      canUsePermlaneX16Var_NRLarge = hasPermlaneVar;
      canUseDsSwizzleW32_NRLarge = hasDsSwizzleWave32;
    }

    // Permlane-swap: register-only cross-lane reduction on wave64 (gfx950+)
    // via v_permlane{16,32}_swap_b32. Active in NR-Large path for
    // partialR == 2 (one v_permlane32_swap step) or partialR == 4 (one
    // v_permlane16_swap then one v_permlane32_swap).
    // partialR == mTidPerWave freezes the MFMA layout contract: reduction
    // partners must be at lane distances 32 (2-way) or 16 (4-way).
    // Precedence: gfx950 is checked first and gets permlane_swap;
    // the !hasPermlaneSwap guard below ensures gfx94x (which also matches
    // the "gfx94" substring) falls into the ds_swizzle+bpermute arm instead.
    bool hasPermlaneSwap =
        elemSupportedByCrossLane && arch.getValue().contains("gfx950");
    bool canUsePermlaneSwap_NRLarge =
        (layoutTilesWave && hasPermlaneSwap && waveSize == 64 &&
         (partialR == 2 || partialR == 4) && partialR == mTidPerWave &&
         blockSize <= nonReductionDimSizeProduct);

    // ds_swizzle + ds_bpermute: register-only cross-lane reduction on CDNA
    // wave64 (gfx908/MI100, gfx90a/MI250, gfx94x/MI300) via ds_swizzle_b32
    // (XOR within each 32-lane half) and ds_bpermute_b32 (XOR 32, crossing
    // the half-wave boundary). Same eligibility as canUsePermlaneSwap_NRLarge
    // but for architectures without v_permlane_swap. The !hasPermlaneSwap
    // guard prevents gfx950 from entering this arm despite matching "gfx94".
    bool hasDsSwizzleBpermute = elemSupportedByCrossLane && !hasPermlaneSwap &&
                                waveSize == 64 &&
                                (arch.getValue().contains("gfx908") ||
                                 arch.getValue().contains("gfx90a") ||
                                 arch.getValue().contains("gfx94"));
    bool canUseDsSwizzleBpermute_NRLarge =
        (layoutTilesWave && hasDsSwizzleBpermute &&
         (partialR == 2 || partialR == 4) && partialR == mTidPerWave &&
         blockSize <= nonReductionDimSizeProduct);

    // NR-Small permlane fast-path eligibility for skipping LDS round-trips.
    //
    // Safety constraints (apply to both wave64 and wave32 variants below):
    //   - Single-wave only (blockSize == waveSize): cross-wave merge would
    //     still need LDS since permlane primitives don't cross waves.
    //   - K == 1 (K := inputThreadSubTile2dShape[nrDim]): when K > 1 the
    //     i-th slot of partialReductionBuffer holds the value for a
    //     DIFFERENT nr-position, not an rIter slice; substituting the LDS
    //     read with partialReductionBuffer[i] would mix nr-positions and
    //     produce incorrect results. With K == 1 each thread has exactly
    //     one value (its own (nrtid, rtid) cell), and the LDS read becomes
    //     an identity that we can replace with partialReductionBuffer[0].
    //   - !extraOut: the extraOut path reads from LDS, so it would race
    //     against the skipped writes.
    //
    // NR-Small LDS-skip eligibility (wave64): single-wave, K == 1,
    // partialR ∈ {2, 4}, exact packing (nrDimProd * partialR == waveSize),
    // no extraOut. When true, BOTH the upfront and final LDS round-trips
    // are skipped — the entire reduction stays in registers.
    // Applies to gfx950 (permlane swap) and gfx908/gfx90a/gfx94x (ds_swizzle).
    auto checkLdsSkipEligibility = [&](bool hasFeature) -> bool {
      if (!hasFeature || blockSize != waveSize || waveSize != 64 ||
          blockSize <= nonReductionDimSizeProduct ||
          nonReductionDimSizeProduct <= 0 ||
          !llvm::isPowerOf2_64(nonReductionDimSizeProduct) ||
          op.getExtraOutViewAttr())
        return false;
      int64_t r = blockSize / nonReductionDimSizeProduct;
      int64_t K = inputThreadSubTile2dShape[nrDim];
      return (r == 2 || r == 4) && nonReductionDimSizeProduct * r == waveSize &&
             K == 1 && partialR == r;
    };
    bool canUsePermlaneSwap_NRSmall_LdsSkip =
        checkLdsSkipEligibility(hasPermlaneSwap);
    bool canUseDsSwizzleBpermute_NRSmall_LdsSkip =
        checkLdsSkipEligibility(hasDsSwizzleBpermute);

    // Wave32 NR-Small LDS-skip: both permlane and ds_swizzle wave32 paths
    // skip the upfront LDS write+barrier; this gate additionally skips
    // the END LDS round-trip when !extraOut. partialR==2 guarantees both
    // reduction partners are in the same wave, so multi-wave is safe.
    // K == 1 is required as a defensive guard (see wave64 LDS-skip above).
    int64_t K = inputThreadSubTile2dShape[nrDim];
    bool canUsePermlaneX16Var_NRSmall_LdsSkip =
        canUsePermlaneX16Var_NRSmall && !op.getExtraOutViewAttr() && K == 1;
    bool canUseDsSwizzleW32_NRSmall_LdsSkip =
        canUseDsSwizzleW32_NRSmall && !op.getExtraOutViewAttr() && K == 1;

    LLVM_DEBUG({
      if (!has2DThreadLayout && (hasPermlaneVar || hasPermlaneSwap ||
                                 hasDsSwizzleBpermute || hasDsSwizzleWave32)) {
        llvm::dbgs() << "BlockwiseReduce: has2DThreadLayout=false but arch "
                        "supports cross-lane intrinsics; all register-only "
                        "fast paths disabled. Check tidSubTileSliceView for "
                        "m_tid/n_tid naming.\n";
      }
      if (has2DThreadLayout && !layoutTilesWave) {
        llvm::dbgs() << "BlockwiseReduce: mTidPerWave(" << mTidPerWave
                     << ") * nTidPerWave(" << nTidPerWave << ") != waveSize("
                     << waveSize
                     << "); layout does not tile the wave, "
                        "all register-only fast paths disabled.\n";
      }
      if (!elemSupportedByCrossLane) {
        llvm::dbgs() << "BlockwiseReduce: elemBitWidth=" << elemBitWidth
                     << " > 32; all register-only fast paths disabled "
                        "(only up to 32-bit types supported).\n";
      }
    });

    if (!canUsePermlaneX16Var_NRSmall && !canUseDsSwizzleW32_NRSmall &&
        !canUsePermlaneX16Var_NRLarge && !canUseDsSwizzleW32_NRLarge &&
        !canUsePermlaneSwap_NRLarge && !canUseDsSwizzleBpermute_NRLarge &&
        !canUsePermlaneSwap_NRSmall_LdsSkip &&
        !canUseDsSwizzleBpermute_NRSmall_LdsSkip) {
      storePartialReductionstoLDS(rewriter, loc, partialReductionBuffer,
                                  workspaceLDSBuffer, inputBlockSubTile2dView,
                                  inputThreadSubTile2dView, tidSubTileSliceView,
                                  toFlatLDSView);
      LDSBarrierOp::create(rewriter, loc);
    }
    // Common pattern for all cross-lane fast paths: perform the cross-lane
    // reduction, then either broadcast from registers (LDS-skip) or fall
    // back to the LDS round-trip when multi-wave or extraOut is active.
    auto emitCrossLaneReduceWithBroadcast = [&](auto crossLaneReduceFn,
                                                bool canSkipEndLds) {
      crossLaneReduceFn();
      if (canSkipEndLds) {
        readReducedResultsFromPrivateBuffer(rewriter, loc,
                                            partialReductionBuffer, outputReg,
                                            inputThreadSubTile2dView);
      } else {
        storePartialReductionstoLDS(rewriter, loc, partialReductionBuffer,
                                    workspaceLDSBuffer, inputBlockSubTile2dView,
                                    inputThreadSubTile2dView,
                                    tidSubTileSliceView, toFlatLDSView);
        readReducedResultsFromLDS(rewriter, loc, op, workspaceLDSBuffer,
                                  outputReg, inputViewArrayAttr, axis,
                                  partialRegTensorShape[rDim], tid,
                                  /*withBarrier=*/true);
      }
    };

    // Following RAII scope will create reduction loops.
    {
      if (blockSize <= nonReductionDimSizeProduct) {
        int64_t nrDimSize = inputThreadSubTile2dShape[nrDim];
        bool noExtraOut = !op.getExtraOutViewAttr();
        if (canUsePermlaneSwap_NRLarge) {
          emitCrossLaneReduceWithBroadcast(
              [&] {
                permlaneSwapReduce(rewriter, loc, partialReductionBuffer,
                                   nrDimSize, partialR, elemType, op);
              },
              noExtraOut);
        } else if (canUseDsSwizzleBpermute_NRLarge) {
          emitCrossLaneReduceWithBroadcast(
              [&] {
                dsSwizzleBpermuteReduce(rewriter, loc, partialReductionBuffer,
                                        nrDimSize, partialR, elemType, tid, op);
              },
              noExtraOut);
        } else if (canUsePermlaneX16Var_NRLarge) {
          emitCrossLaneReduceWithBroadcast(
              [&] {
                permlaneX16VarReduce(rewriter, loc, partialReductionBuffer, tid,
                                     nrDimSize, waveSize, elemType, op);
              },
              noExtraOut);
        } else if (canUseDsSwizzleW32_NRLarge) {
          emitCrossLaneReduceWithBroadcast(
              [&] {
                dsSwizzleReduceWave32(rewriter, loc, partialReductionBuffer,
                                      nrDimSize, elemType, op);
              },
              noExtraOut);
        } else {
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

          // Note: This currently creates a bunch of dead IR because
          // vectorization needs access to a `Value` in order to account for
          // scalarized buffers.
          Value threadToLDSViewed =
              transform(rewriter, workspaceLDSBuffer, threadToLDSViewTrs);
          VectorizationResult nrIterVectorRes =
              getMaxVectorization(threadToLDSViewed, nrIterDim);
          int64_t nrIterVectorLen = nrIterVectorRes.max;
          // Create the accumulation register
          // This will be accumulated over non-reduction iterations.
          auto accRegType =
              MemRefType::get(nrIterVectorLen, elemType, AffineMap{},
                              privateMemoryAddressSpace);
          Value accReg = GpuAllocOp::create(rewriter, loc, accRegType);
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            Value nrIter;
            if (threadViewShape[nrIterDim] > 1) {
              AffineForOp nrIterLoop = AffineForOp::create(
                  rewriter, loc, 0, threadViewShape[nrIterDim],
                  nrIterVectorLen);
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
                ArrayRef<Attribute>{threadToLDSViewTrs,
                                    rewriter.getArrayAttr({}),
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
              //    vector The code in createReducingOp will do vector
              //    elementwise op and store the resulting vector to accReg.
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
              // Storing the last reduction iter output directly to LDS[...,
              // dr=0,
              // ...]
              Value rIterArg =
                  reductionLoop.getLowerCoords(/*domain=*/1)[rIterDim];
              Value boundVal = arith::ConstantIndexOp::create(
                  rewriter, loc, threadViewShape[rIterDim]);
              Value strideVal =
                  arith::ConstantIndexOp::create(rewriter, loc, rIterVectorLen);
              Value lastIterVal =
                  arith::SubIOp::create(rewriter, loc, boundVal, strideVal);
              Value isLastIter =
                  arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                        rIterArg, lastIterVal);
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
          readReducedResultsFromLDS(rewriter, loc, op, workspaceLDSBuffer,
                                    outputReg, inputViewArrayAttr, axis,
                                    partialRegTensorShape[rDim], tid,
                                    /*withBarrier=*/true);
        } // end NR-Large-Tree else
      } else {
        int64_t nrDimSize = inputThreadSubTile2dShape[nrDim];
        if (canUsePermlaneX16Var_NRSmall) {
          emitCrossLaneReduceWithBroadcast(
              [&] {
                permlaneX16VarReduce(rewriter, loc, partialReductionBuffer, tid,
                                     nrDimSize, waveSize, elemType, op);
              },
              canUsePermlaneX16Var_NRSmall_LdsSkip);
        } else if (canUseDsSwizzleW32_NRSmall) {
          emitCrossLaneReduceWithBroadcast(
              [&] {
                dsSwizzleReduceWave32(rewriter, loc, partialReductionBuffer,
                                      nrDimSize, elemType, op);
              },
              canUseDsSwizzleW32_NRSmall_LdsSkip);
        } else {
          ArrayAttr threadToTensorViewTrs =
              createThreadViewforNRSmallerThanThreads(
                  loc, partialRegTensorShape, blockSize, rDim, rewriter);
          ArrayAttr threadToLDSViewTrs = createLDSWorkspaceView(
              loc, rewriter, threadToTensorViewTrs, rDim);
          ArrayRef<int64_t> threadViewShape =
              cast<TransformMapAttr>(threadToLDSViewTrs[0]).getUpperBounds();
          constexpr size_t rTidDim = 1;
          constexpr size_t rIterDim = 2;

          Value threadToLDSViewed =
              transform(rewriter, workspaceLDSBuffer, threadToLDSViewTrs);
          VectorizationResult rIterVectorRes =
              getMaxVectorization(threadToLDSViewed, rIterDim);
          int64_t rIterVectorLen = rIterVectorRes.max;

          // Use DPP-based subgroup reduction when all conditions are met:
          // 1. Power-of-2 reduction threads (required by SubgroupReduceOp)
          // 2. More than 1 reduction thread (at least 2 for cross-lane work)
          // 3. partialR > 2: partialR is the block-level LDS reduction
          //    dimension size (number of partial values per non-reduction
          //    position), not the per-thread iteration count. When
          //    partialR == 2, the cluster degenerates to size 2 with only one
          //    reduction element per thread, so DPP setup cost is not amortized
          //    vs the LDS-tree fallback. Threshold chosen from tuning data.
          // 4. Reduction threads fit within a single wave
          // 5. Exact thread packing: blockSize == clusterSize *
          //    nonReductionDimSizeProduct. This guarantees every thread maps to
          //    a valid (nrtid, rtid) pair, so LDS coordinates derived from them
          //    are in-bounds.
          // Otherwise, fall back to LDS-based tree reduction.
          int64_t maxActiveReductionThreads = threadViewShape[rTidDim];
          int64_t clusterSize = llvm::PowerOf2Ceil(maxActiveReductionThreads);
          bool canUseDPP = llvm::isPowerOf2_64(maxActiveReductionThreads) &&
                           (maxActiveReductionThreads > 1) && (partialR > 2) &&
                           (maxActiveReductionThreads <= waveSize) &&
                           (blockSize == maxActiveReductionThreads *
                                             nonReductionDimSizeProduct);
          // Permlane fast-path eligibility (gfx950 wave64): single-wave with
          // rthreads ∈ {2, 4} and exact thread packing
          // (nrDimProd * rthreads == waveSize). Partner lanes are 16/32 apart,
          // so this uses the tree-style rtid layout below.
          bool canUsePermlaneSwap_NRSmall =
              hasPermlaneSwap && waveSize == 64 && blockSize == waveSize &&
              llvm::isPowerOf2_64(maxActiveReductionThreads) &&
              (maxActiveReductionThreads == 2 ||
               maxActiveReductionThreads == 4) &&
              llvm::isPowerOf2_64(nonReductionDimSizeProduct) &&
              nonReductionDimSizeProduct * maxActiveReductionThreads ==
                  waveSize;
          // ds_swizzle+bpermute fast-path (gfx908/gfx90a/gfx94x wave64):
          // same eligibility as canUsePermlaneSwap_NRSmall but for CDNA arches.
          bool canUseDsSwizzleBpermute_NRSmall =
              hasDsSwizzleBpermute && blockSize == waveSize &&
              llvm::isPowerOf2_64(maxActiveReductionThreads) &&
              (maxActiveReductionThreads == 2 ||
               maxActiveReductionThreads == 4) &&
              llvm::isPowerOf2_64(nonReductionDimSizeProduct) &&
              nonReductionDimSizeProduct * maxActiveReductionThreads ==
                  waveSize;
          // The early LDS-skip prediction must remain consistent with the
          // runtime eligibility check.
          assert(!canUsePermlaneSwap_NRSmall_LdsSkip ||
                 canUsePermlaneSwap_NRSmall);
          assert(!canUseDsSwizzleBpermute_NRSmall_LdsSkip ||
                 canUseDsSwizzleBpermute_NRSmall);
          // Two different tid → (rtid, nrtid) factorings are used:
          //
          // DPP path: rtid = tid % clusterSize, nrtid = tid / clusterSize.
          //   SubgroupReduceOp uses DPP lane-swizzle within a cluster, so
          //   consecutive lanes must be reduction partners. Packing rtid
          //   into the low bits achieves this.
          //
          // Tree / permlane / ds_swizzle paths:
          //   rtid = tid / nrDimProd, nrtid = tid % nrDimProd.
          //   Permlane swap and ds_swizzle intrinsics exchange lanes at
          //   fixed distances (16/32), which aligns with rtid occupying
          //   the high bits of tid.
          //
          // The DPP factoring differs from tidSubTileSliceView's mapping
          // of tid, which storePartialReductionstoLDS uses to place each
          // thread's partial into flat LDS. This is correct because:
          //  - Both the store (via toFlatLDSView) and the DPP read (via
          //    threadToLDSViewTrs) address the same row-major flat LDS
          //    array: flat = nrDim * rDimSize + rDim.
          //  - The store writes one value per (nrDim, rDim) slot using
          //    coords from tidSubTileSliceView; the DPP read accesses
          //    fixed flat positions via (nrtid, rtid, rIter) coords.
          //  - The DPP read does not assume which thread stored a given
          //    slot — it simply reads the value at each flat address.
          //  - Complete coverage is guaranteed: tidSubTileSliceView covers
          //    all rDim positions and the iter subtile covers all nrDim
          //    positions, so every (nrDim, rDim) slot is written by at
          //    least one thread with valid (in-bounds) data.
          //    The DPP read covers every rDim position per nrtid row
          //    (clusterSize * rDimPerRThread == rDimSize).
          Value rtid, nrtid;
          if (canUseDPP && !canUsePermlaneSwap_NRSmall &&
              !canUseDsSwizzleBpermute_NRSmall) {
            assert(llvm::isPowerOf2_64(clusterSize) &&
                   "clusterSize must be power of 2");
            // The DPP read uses threadToLDSViewTrs with (nrtid, rtid, rIter)
            // coords — a different decomposition of the same flat array
            // that the store populated via tidSubTileSliceView.
            // This is safe: the DPP read addresses fixed flat positions and
            // does not care which thread stored each value.
            unsigned log2ClusterSize = llvm::Log2_64(clusterSize);
            Value shiftAmt =
                arith::ConstantIndexOp::create(rewriter, loc, log2ClusterSize);
            Value mask =
                arith::ConstantIndexOp::create(rewriter, loc, clusterSize - 1);
            rtid = arith::AndIOp::create(rewriter, loc, tid, mask);
            nrtid = arith::ShRUIOp::create(rewriter, loc, tid, shiftAmt);
          } else {
            if (llvm::isPowerOf2_64(nonReductionDimSizeProduct)) {
              unsigned log2Val = llvm::Log2_64(nonReductionDimSizeProduct);
              Value shiftAmt =
                  arith::ConstantIndexOp::create(rewriter, loc, log2Val);
              Value mask = arith::ConstantIndexOp::create(
                  rewriter, loc, nonReductionDimSizeProduct - 1);
              rtid = arith::ShRUIOp::create(rewriter, loc, tid, shiftAmt);
              nrtid = arith::AndIOp::create(rewriter, loc, tid, mask);
            } else {
              Value nrDimSizeProductConst = arith::ConstantIndexOp::create(
                  rewriter, loc, nonReductionDimSizeProduct);
              rtid = arith::DivSIOp::create(rewriter, loc, tid,
                                            nrDimSizeProductConst);
              nrtid = arith::RemSIOp::create(rewriter, loc, tid,
                                             nrDimSizeProductConst);
            }
          }

          // Threadwise reduction accumulator (populated only when rIterDim >
          // 1). Under the LDS-skip gate (K == 1) rIter span is also 1, so this
          // loop never runs on the LDS-skip path.
          Value accReg;
          bool hasThreadwiseReduction = threadViewShape[rIterDim] > 1;
          assert(
              !(canUsePermlaneSwap_NRSmall_LdsSkip && hasThreadwiseReduction) &&
              "LDS-skip gate (K==1) implies rIter==1");
          assert(!(canUseDsSwizzleBpermute_NRSmall_LdsSkip &&
                   hasThreadwiseReduction) &&
                 "LDS-skip gate (K==1) implies rIter==1");
          if (hasThreadwiseReduction) {
            int64_t localIterVectorLen = rIterVectorLen;
            Type loadTypeInputReg =
                vectorTypeOrSelf(elemType, localIterVectorLen);
            Type accRegType = MemRefType::get({1}, elemType, AffineMap{},
                                              privateMemoryAddressSpace);
            accReg = GpuAllocOp::create(rewriter, loc, accRegType);

            SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
            SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
            SmallVector<int64_t> strides{1, 1, localIterVectorLen};

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

          if (canUsePermlaneSwap_NRSmall) {
            // Permlane fast-path (gfx950 wave64): replaces the DPP/subgroup
            // cross-lane reduction with v_permlane{16,32}_swap_b32 over a
            // scalar accumulator.
            //
            //  1. Populate localAccReg with the per-thread input value
            //     (from accReg if threadwise-reduced, else from
            //      partialReductionBuffer[0] under LDS-skip, else from LDS).
            //  2. Reduce across partner groups via permlane swap.
            //  3a. Under LDS-skip: write the result back to
            //      partialReductionBuffer[0] for the register-only broadcast.
            //  3b. Otherwise: leader writes to LDS so the standard
            //      readReducedResultsFromLDS can broadcast it.
            Value localAccReg = accReg;
            if (!hasThreadwiseReduction) {
              Type accRegType = MemRefType::get({1}, elemType, AffineMap{},
                                                privateMemoryAddressSpace);
              localAccReg = GpuAllocOp::create(rewriter, loc, accRegType);

              if (canUsePermlaneSwap_NRSmall_LdsSkip) {
                // K == 1 under the LDS-skip gate; the single value lives at
                // partialReductionBuffer[0].
                Value loadVal = InBoundsLoadOp::create(rewriter, loc, elemType,
                                                       partialReductionBuffer,
                                                       zeroConstantOp);
                InBoundsStoreOp::create(rewriter, loc, loadVal, localAccReg,
                                        zeroConstantOp);
              } else {
                SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
                SmallVector<int64_t> bounds{1, 1, 1};
                SmallVector<int64_t> strides{1, 1, 1};

                TransformingForOp loadLoop = TransformingForOp::create(
                    rewriter, loc, ArrayRef<ValueRange>(inits),
                    ArrayRef<Attribute>{threadToLDSViewTrs},
                    ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                    /*forceUnroll=*/true, /*useIndexDiffs=*/true);
                {
                  PatternRewriter::InsertionGuard guard(rewriter);
                  rewriter.setInsertionPointToStart(loadLoop.getBody());
                  Block::BlockArgListType LDSCoords =
                      loadLoop.getLowerCoords(/*domain=*/0);
                  Value loadVal = InBoundsLoadOp::create(
                      rewriter, loc, elemType, workspaceLDSBuffer, LDSCoords);
                  InBoundsStoreOp::create(rewriter, loc, loadVal, localAccReg,
                                          zeroConstantOp);
                }
              }
            }

            // After this call every lane in a partner group has the fully
            // reduced value for its nrtid in localAccReg[0].
            permlaneSwapReduce(rewriter, loc, localAccReg, /*numElements=*/1,
                               /*groupSize=*/maxActiveReductionThreads,
                               elemType, op);

            if (canUsePermlaneSwap_NRSmall_LdsSkip) {
              // END LDS-skip: write into partialReductionBuffer[0] so
              // readReducedResultsFromPrivateBuffer can broadcast from
              // registers — no LDS write, no barrier, no LDS read.
              Value reduced = InBoundsLoadOp::create(
                  rewriter, loc, elemType, localAccReg, zeroConstantOp);
              InBoundsStoreOp::create(rewriter, loc, reduced,
                                      partialReductionBuffer, zeroConstantOp);
            } else {
              // Leader (rtid == 0) writes the reduced value to LDS for the
              // standard readReducedResultsFromLDS broadcast path.
              SmallVector<Value, 4> storeInits{nrtid, rtid, zeroConstantOp};
              SmallVector<int64_t> storeBounds{1, 1, 1};
              SmallVector<int64_t> storeStrides{1, 1, 1};

              TransformingForOp storeLoop = TransformingForOp::create(
                  rewriter, loc, ArrayRef<ValueRange>(storeInits),
                  ArrayRef<Attribute>{threadToLDSViewTrs},
                  ArrayRef<int64_t>(storeBounds),
                  ArrayRef<int64_t>(storeStrides),
                  /*forceUnroll=*/true, /*useIndexDiffs=*/true);
              {
                PatternRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(storeLoop.getBody());
                Block::BlockArgListType LDSCoords =
                    storeLoop.getLowerCoords(/*domain=*/0);
                Value zeroIdx =
                    arith::ConstantIndexOp::create(rewriter, loc, 0);
                Value isLeader = arith::CmpIOp::create(
                    rewriter, loc, arith::CmpIPredicate::eq, rtid, zeroIdx);
                scf::IfOp ifStore = scf::IfOp::create(rewriter, loc, isLeader,
                                                      /*withElseRegion=*/false);
                {
                  PatternRewriter::InsertionGuard storeGuard(rewriter);
                  rewriter.setInsertionPointToStart(ifStore.thenBlock());
                  Value valToStore = InBoundsLoadOp::create(
                      rewriter, loc, elemType, localAccReg, zeroConstantOp);
                  InBoundsStoreOp::create(rewriter, loc, valToStore,
                                          workspaceLDSBuffer, LDSCoords);
                }
              }
              LDSBarrierOp::create(rewriter, loc);
            }

          } else if (canUseDsSwizzleBpermute_NRSmall) {
            Value localAccReg = accReg;
            if (!hasThreadwiseReduction) {
              Type accRegType = MemRefType::get({1}, elemType, AffineMap{},
                                                privateMemoryAddressSpace);
              localAccReg = GpuAllocOp::create(rewriter, loc, accRegType);

              if (canUseDsSwizzleBpermute_NRSmall_LdsSkip) {
                Value loadVal = InBoundsLoadOp::create(rewriter, loc, elemType,
                                                       partialReductionBuffer,
                                                       zeroConstantOp);
                InBoundsStoreOp::create(rewriter, loc, loadVal, localAccReg,
                                        zeroConstantOp);
              } else {
                SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
                SmallVector<int64_t> bounds{1, 1, 1};
                SmallVector<int64_t> strides{1, 1, 1};

                TransformingForOp loadLoop = TransformingForOp::create(
                    rewriter, loc, ArrayRef<ValueRange>(inits),
                    ArrayRef<Attribute>{threadToLDSViewTrs},
                    ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                    /*forceUnroll=*/true, /*useIndexDiffs=*/true);
                {
                  PatternRewriter::InsertionGuard guard(rewriter);
                  rewriter.setInsertionPointToStart(loadLoop.getBody());
                  Block::BlockArgListType LDSCoords =
                      loadLoop.getLowerCoords(/*domain=*/0);
                  Value loadVal = InBoundsLoadOp::create(
                      rewriter, loc, elemType, workspaceLDSBuffer, LDSCoords);
                  InBoundsStoreOp::create(rewriter, loc, loadVal, localAccReg,
                                          zeroConstantOp);
                }
              }
            }

            dsSwizzleBpermuteReduce(rewriter, loc, localAccReg,
                                    /*numElements=*/1,
                                    /*groupSize=*/maxActiveReductionThreads,
                                    elemType, tid, op);

            if (canUseDsSwizzleBpermute_NRSmall_LdsSkip) {
              Value reduced = InBoundsLoadOp::create(
                  rewriter, loc, elemType, localAccReg, zeroConstantOp);
              InBoundsStoreOp::create(rewriter, loc, reduced,
                                      partialReductionBuffer, zeroConstantOp);
            } else {
              SmallVector<Value, 4> storeInits{nrtid, rtid, zeroConstantOp};
              SmallVector<int64_t> storeBounds{1, 1, 1};
              SmallVector<int64_t> storeStrides{1, 1, 1};

              TransformingForOp storeLoop = TransformingForOp::create(
                  rewriter, loc, ArrayRef<ValueRange>(storeInits),
                  ArrayRef<Attribute>{threadToLDSViewTrs},
                  ArrayRef<int64_t>(storeBounds),
                  ArrayRef<int64_t>(storeStrides),
                  /*forceUnroll=*/true, /*useIndexDiffs=*/true);
              {
                PatternRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(storeLoop.getBody());
                Block::BlockArgListType LDSCoords =
                    storeLoop.getLowerCoords(/*domain=*/0);
                Value zeroIdx =
                    arith::ConstantIndexOp::create(rewriter, loc, 0);
                Value isLeader = arith::CmpIOp::create(
                    rewriter, loc, arith::CmpIPredicate::eq, rtid, zeroIdx);
                scf::IfOp ifStore = scf::IfOp::create(rewriter, loc, isLeader,
                                                      /*withElseRegion=*/false);
                {
                  PatternRewriter::InsertionGuard storeGuard(rewriter);
                  rewriter.setInsertionPointToStart(ifStore.thenBlock());
                  Value valToStore = InBoundsLoadOp::create(
                      rewriter, loc, elemType, localAccReg, zeroConstantOp);
                  InBoundsStoreOp::create(rewriter, loc, valToStore,
                                          workspaceLDSBuffer, LDSCoords);
                }
              }
              LDSBarrierOp::create(rewriter, loc);
            }

          } else if (canUseDPP) {
            SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
            SmallVector<int64_t> bounds{1, 1, 1};
            SmallVector<int64_t> strides{1, 1, 1};

            gpu::AllReduceOperation gpuReduceOp;
            ReduceMethod rMethod = op.getReduceMethod();
            if (rMethod == ReduceMethod::Sum) {
              gpuReduceOp = gpu::AllReduceOperation::ADD;
            } else {
              gpuReduceOp = isa<FloatType>(elemType)
                                ? gpu::AllReduceOperation::MAXNUMF
                                : gpu::AllReduceOperation::MAXSI;
            }

            TransformingForOp dppLoop = TransformingForOp::create(
                rewriter, loc, ArrayRef<ValueRange>(inits),
                ArrayRef<Attribute>{threadToLDSViewTrs},
                ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                /*forceUnroll=*/true, /*useIndexDiffs=*/true);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(dppLoop.getBody());
              Block::BlockArgListType LDSCoords =
                  dppLoop.getLowerCoords(/*domain=*/0);

              Value valueToReduce;
              if (hasThreadwiseReduction) {
                valueToReduce = InBoundsLoadOp::create(rewriter, loc, elemType,
                                                       accReg, zeroConstantOp);
              } else {
                valueToReduce = InBoundsLoadOp::create(
                    rewriter, loc, elemType, workspaceLDSBuffer, LDSCoords);
              }

              Value reduced = gpu::SubgroupReduceOp::create(
                  rewriter, loc, valueToReduce, gpuReduceOp, /*uniform=*/false,
                  /*cluster_size=*/std::optional<uint32_t>(clusterSize));

              Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
              Value isLeader = arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::eq, rtid, zeroIdx);

              scf::IfOp ifStore = scf::IfOp::create(rewriter, loc, isLeader,
                                                    /*withElseRegion=*/false);
              {
                PatternRewriter::InsertionGuard storeGuard(rewriter);
                rewriter.setInsertionPointToStart(ifStore.thenBlock());
                InBoundsStoreOp::create(rewriter, loc, reduced,
                                        workspaceLDSBuffer, LDSCoords);
              }
            }
            LDSBarrierOp::create(rewriter, loc);

          } else {
            int64_t ceilPowerOf2 =
                llvm::PowerOf2Ceil(maxActiveReductionThreads) / 2;
            if (hasThreadwiseReduction) {
              SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
              SmallVector<int64_t> bounds{1, 1, 1};
              SmallVector<int64_t> strides{1, 1, 1};

              TransformingForOp storeLoop = TransformingForOp::create(
                  rewriter, loc, ArrayRef<ValueRange>(inits),
                  ArrayRef<Attribute>{threadToLDSViewTrs},
                  ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                  /*forceUnroll=*/true, /*useIndexDiffs=*/true);
              {
                PatternRewriter::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(storeLoop.getBody());
                Block::BlockArgListType LDSStoreCoords =
                    storeLoop.getLowerCoords(/*domain=*/0);
                Value loadVal = InBoundsLoadOp::create(rewriter, loc, elemType,
                                                       accReg, zeroConstantOp);
                InBoundsStoreOp::create(rewriter, loc, loadVal,
                                        workspaceLDSBuffer, LDSStoreCoords);
              }
              LDSBarrierOp::create(rewriter, loc);
            }

            int64_t treeMaxActiveThreads = maxActiveReductionThreads;
            for (int64_t offset = ceilPowerOf2; offset >= 1;
                 offset = offset >> 1) {
              Value offsetVal =
                  arith::ConstantIndexOp::create(rewriter, loc, offset);
              Value rtidPlusOffsetVal =
                  arith::AddIOp::create(rewriter, loc, rtid, offsetVal);
              Value maxActiveReductionThreadsVal =
                  arith::ConstantIndexOp::create(rewriter, loc,
                                                 treeMaxActiveThreads);
              treeMaxActiveThreads =
                  llvm::PowerOf2Ceil(treeMaxActiveThreads) >> 1;
              Value isValid = arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::slt, rtidPlusOffsetVal,
                  maxActiveReductionThreadsVal);
              scf::IfOp ifb = scf::IfOp::create(rewriter, loc, isValid,
                                                /*withElseRegion=*/false);
              {
                OpBuilder thenb = ifb.getThenBodyBuilder();
                SmallVector<Value, 4> firstInits{nrtid, rtid, zeroConstantOp};
                SmallVector<Value, 4> secondInits{nrtid, rtidPlusOffsetVal,
                                                  zeroConstantOp};
                SmallVector<int64_t> bounds{1, 1, 1};
                SmallVector<int64_t> strides{1, 1, 1};

                TransformingForOp reductionLoop = TransformingForOp::create(
                    thenb, loc, ArrayRef<ValueRange>{firstInits, secondInits},
                    ArrayRef<Attribute>{threadToLDSViewTrs, threadToLDSViewTrs},
                    ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                    /*forceUnroll=*/true, /*useIndexDiffs=*/true);
                {
                  PatternRewriter::InsertionGuard guard(thenb);
                  thenb.setInsertionPointToStart(reductionLoop.getBody());
                  Block::BlockArgListType firstLDSLoadCoords =
                      reductionLoop.getLowerCoords(/*domain=*/0);
                  Value firstLoadVal = InBoundsLoadOp::create(
                      thenb, loc, elemType, workspaceLDSBuffer,
                      firstLDSLoadCoords);
                  Block::BlockArgListType secondLDSLoadCoords =
                      reductionLoop.getLowerCoords(/*domain=*/1);
                  Value secondLoadVal = InBoundsLoadOp::create(
                      thenb, loc, elemType, workspaceLDSBuffer,
                      secondLDSLoadCoords);
                  Value reduced =
                      createReducingOp(op, firstLoadVal, secondLoadVal, thenb);
                  InBoundsStoreOp::create(thenb, loc, reduced,
                                          workspaceLDSBuffer,
                                          firstLDSLoadCoords);
                }
              }
              LDSBarrierOp::create(rewriter, loc);
            }
          }
          if (canUsePermlaneSwap_NRSmall_LdsSkip ||
              canUseDsSwizzleBpermute_NRSmall_LdsSkip) {
            readReducedResultsFromPrivateBuffer(
                rewriter, loc, partialReductionBuffer, outputReg,
                inputThreadSubTile2dView);
          } else {
            readReducedResultsFromLDS(rewriter, loc, op, workspaceLDSBuffer,
                                      outputReg, inputViewArrayAttr, axis,
                                      partialRegTensorShape[rDim], tid,
                                      /*withBarrier=*/false);
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
    writeAllTarget.addLegalDialect<amdgpu::AMDGPUDialect, arith::ArithDialect,
                                   rock::RockDialect, memref::MemRefDialect,
                                   scf::SCFDialect, vector::VectorDialect,
                                   AffineDialect, gpu::GPUDialect,
                                   LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
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
                         memref::MemRefDialect, gpu::GPUDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, BlockwiseGemmRewritePattern,
               BlockwiseGemmAccelRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
