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
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "AccelEmitter.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

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
          b.create<memref::StoreOp>(loc, value, input, ivs);
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
    GpuAllocOp valueReg = rewriter.create<GpuAllocOp>(loc, valueRegType);
    Value zero = rewriter.createOrFold<ConstantIndexOp>(loc, 0);
    rewriter.create<InBoundsStoreOp>(loc, val, valueReg, zero);
    Value tid =
        rewriter.createOrFold<rock::WorkitemIdOp>(loc, rewriter.getIndexType());
    rewriter.create<ThreadwiseWriteAllOp>(
        loc, valueReg, op.getMemref(), rewriter.getArrayAttr({unmerge, pad}),
        /*extraIndices=*/ValueRange{tid}, GemmFeatures::none, StoreMethod::Set,
        true, true);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

// The structure of this lowing is documented at
// https://github.com/ROCmSoftwarePlatform/rocMLIR/issues/719
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

    Value matrixA, matrixB;
    ArrayAttr transformsA, transformsB;
    std::tie(matrixA, transformsA) =
        untransform(b, adaptor.getMatrixA(),
                    b.getArrayAttr({splitTidAAttr, toLdsIndexAAttr}));
    std::tie(matrixB, transformsB) =
        untransform(b, adaptor.getMatrixB(),
                    b.getArrayAttr({splitTidBAttr, toLdsIndexBAttr}));

    int64_t threadANumRegisters = kPerThread * mC * kPack;
    int64_t threadBNumRegisters = kPerThread * nC * kPack;

    // Alloc register for thread_a and thread_b.
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    auto threadARegisterMemRefType =
        MemRefType::get(threadANumRegisters, elementType, AffineMap{},
                        privateMemoryAddressSpace);
    auto threadAAllocOp = b.create<GpuAllocOp>(loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType =
        MemRefType::get(threadBNumRegisters, elementType, AffineMap{},
                        privateMemoryAddressSpace);
    auto threadBAllocOp = b.create<GpuAllocOp>(loc, threadBRegisterMemRefType);

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
    auto copyALoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferAStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsA, b.getArrayAttr(threadACopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, mRepeat, 1, mPerThread, kPack},
        /*strides=*/std::nullopt, /*forceUnroll=*/true, /*indexDiffs=*/true);
    {
      OpBuilder::InsertionGuard copyAGuard(b);
      b.setInsertionPointToStart(copyALoop.getBody());
      Value aCopy = b.create<memref::LoadOp>(
          loc, matrixA, copyALoop.getLowerCoords(/*domain=*/0));
      Value aCast = createTypeConversionOp(b, loc, aCopy, elementType);
      b.create<memref::StoreOp>(loc, aCast, threadAAllocOp,
                                copyALoop.getLowerCoords(/*domain=*/1));
    }

    SmallVector<Value, 5> ldsBufferBStartCoords = {
        kOffset, zeroConstantOp, workitem, zeroConstantOp, zeroConstantOp};
    auto copyBLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferBStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsB, b.getArrayAttr(threadBCopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, nRepeat, 1, nPerThread, kPack},
        /*strides=*/std::nullopt, /*forceUnroll=*/true, /*indexDiffs=*/true);
    {
      OpBuilder::InsertionGuard copyBGuard(b);
      b.setInsertionPointToStart(copyBLoop.getBody());
      Value bCopy = b.create<memref::LoadOp>(
          loc, matrixB, copyBLoop.getLowerCoords(/*domain=*/0));
      Value bCast = createTypeConversionOp(b, loc, bCopy, elementType);
      b.create<memref::StoreOp>(loc, bCast, threadBAllocOp,
                                copyBLoop.getLowerCoords(/*domain=*/1));
    }

    Value reshapedARegisters = reshapeBuffer(
        b, loc, threadAAllocOp, {"k", "m", "kpack"}, {kPerThread, mC, kPack});
    Value reshapedBRegisters = reshapeBuffer(
        b, loc, threadBAllocOp, {"k", "n", "kpack"}, {kPerThread, nC, kPack});
    // Actually do the gemm - this goes inside the look over kOffset
    b.create<ThreadwiseGemmOp>(loc, reshapedARegisters, reshapedBRegisters,
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

    StringAttr arch = op.getArchAttr();
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    int64_t M = tuningParams.getMPerBlock();
    int64_t N = tuningParams.getNPerBlock();
    int64_t K = tuningParams.getKpackPerBlock();
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();
    int64_t KPack = tuningParams.getKpack();

    Type bufferElemTypeA =
        adaptor.getMatrixA().getType().cast<MemRefType>().getElementType();
    Type bufferElemTypeB =
        adaptor.getMatrixB().getType().cast<MemRefType>().getElementType();
    Type dataTypeA = bufferElemTypeA, dataTypeB = bufferElemTypeB;
    if (auto bufferVecTypeA = bufferElemTypeA.dyn_cast<VectorType>())
      dataTypeA = bufferVecTypeA.getElementType();
    if (auto bufferVecTypeB = bufferElemTypeB.dyn_cast<VectorType>())
      dataTypeB = bufferVecTypeB.getElementType();

    Value sourceOffsetA = adaptor.getWaveOffsetA();
    Value sourceOffsetB = adaptor.getWaveOffsetB();
    int64_t mWaves = M / mPerWave;
    int64_t nWaves = N / nPerWave;

    auto accelEmitterPtr = rock::accel::AccelEmitter::select(
        op.getFeatures(), dataTypeA, dataTypeB, arch, tuningParams);

    if (!accelEmitterPtr)
      return op.emitOpError("Unable to emit accelerator code.");

    // Extract relevant accelerator parameters
    rock::accel::AccelEmitterParams params = accelEmitterPtr->getParams();
    Type argTypeA = params.argTypeA;
    Type argTypeB = params.argTypeB;
    int64_t mRepeats = params.mRepeats;
    int64_t nRepeats = params.nRepeats;
    int64_t mPerAccel = params.mPerAccel;
    int64_t nPerAccel = params.nPerAccel;
    int64_t kBase = params.kBase;
    int64_t kpackPerThread = params.kpackPerThread;
    Value mWavesConstantOp = b.create<ConstantIndexOp>(loc, mWaves);
    Value nWavesConstantOp = b.create<ConstantIndexOp>(loc, nWaves);

    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());
    const int64_t waveSize = rock::lookupArchInfo(arch).waveSize;
    auto laneId =
        b.create<RemUIOp>(loc, tid, b.create<ConstantIndexOp>(loc, waveSize));

    LLVM_DEBUG(llvm::dbgs()
               << "argVectorType A: " << argTypeA << "\n"
               << "argVectorType B: " << argTypeB << "\n"
               << "k_base: " << kBase << "\n"
               << "mPerWave: " << mPerWave << "\n"
               << "nPerWave: " << nPerWave << "\n"
               << "mRepeat: " << mRepeats << "\n"
               << "nRepeat: " << nRepeats << "\n"
               << "K: " << K << "\n"
               << "bufferA type: " << adaptor.getBufferA().getType() << "\n"
               << "bufferB type: " << adaptor.getBufferB().getType() << "\n");

    Value MConstantOp = b.create<ConstantIndexOp>(loc, M);
    Value NConstantOp = b.create<ConstantIndexOp>(loc, N);

    Value mPerAccelConstantOp = b.create<ConstantIndexOp>(loc, mPerAccel);
    Value nPerAccelConstantOp = b.create<ConstantIndexOp>(loc, nPerAccel);

    Value bufferA = adaptor.getBufferA();
    Value bufferB = adaptor.getBufferB();

    Value KPerThreadConstantOp = b.create<ConstantIndexOp>(loc, kpackPerThread);

    auto ldsToRegisterCopy = [&](Location loc, OpBuilder mnb, OpBuilder kb,
                                 Value sourceBase, Value mn_i, Value MN,
                                 Value k_i, Value K, Value mnPerMfmaGroup,
                                 Value mnWaves, Type ldsBufferElemType,
                                 Type dataType, Value ldsOrig, Value regDest) {
      // Compute source offset
      Value sourceOffset = accelEmitterPtr->computeLdsSourceOffset(
          kb, k_i, mnb, mn_i, b, MN, loc, sourceBase, mnWaves, laneId);

      Value value = kb.create<memref::LoadOp>(loc, ldsBufferElemType, ldsOrig,
                                              sourceOffset);

      auto bufferType = regDest.getType().cast<MemRefType>();
      Type bufferElementType = bufferType.getElementType();

      // We're loading in units of kPack, but storing in units of k_base.
      if (KPack == kBase) {
        Value destOffset = k_i;
        kb.create<memref::StoreOp>(loc, value, regDest, ValueRange{destOffset});
      } else if (KPack > kBase) {
        int64_t numStores = KPack / kBase;
        Value baseDestOffset = kb.createOrFold<arith::MulIOp>(
            loc, k_i, kb.createOrFold<arith::ConstantIndexOp>(loc, numStores));
        for (int64_t i = 0; i < numStores; ++i) {
          Value sliceStart =
              kb.createOrFold<arith::ConstantIndexOp>(loc, kBase * i);
          Value slice = kb.create<ExtractSliceOp>(loc, bufferElementType, value,
                                                  sliceStart);
          Value destOffset = kb.createOrFold<arith::AddIOp>(
              loc, baseDestOffset,
              kb.createOrFold<arith::ConstantIndexOp>(loc, i));
          kb.create<memref::StoreOp>(loc, slice, regDest,
                                     ValueRange{destOffset});
        }
      } else if (KPack < kBase) {
        // Here we are gathering loaded values into vectors for passing into
        // MFMAs.
        Value destValsPerKpack =
            kb.createOrFold<arith::ConstantIndexOp>(loc, kBase / KPack);
        // This is fine, since the inputs to MFMAs are contiguous in the k
        // dimension.
        Value destOffset =
            kb.createOrFold<arith::DivUIOp>(loc, k_i, destValsPerKpack);
        Value destVecPart =
            kb.createOrFold<arith::RemUIOp>(loc, k_i, destValsPerKpack);
        Value destSlicePos = kb.createOrFold<arith::MulIOp>(
            loc, destVecPart,
            b.createOrFold<arith::ConstantIndexOp>(loc, KPack));
        Value destVec = kb.create<memref::LoadOp>(
            loc, bufferElementType, regDest, ValueRange{destOffset});
        Value newDestVec = kb.create<InsertSliceOp>(
            loc, bufferElementType, value, destVec, destSlicePos);
        kb.create<memref::StoreOp>(loc, newDestVec, regDest,
                                   ValueRange{destOffset});
      }
    };

    auto ldsToRegisterCopyKdim = [&](OpBuilder outerLoopB,
                                     affine::AffineForOp outerLoopBodyOp,
                                     Value sourceBase, Value MN,
                                     Value mnPerMfmaGroup, Value mnWaves,
                                     Type ldsBufferElemType, Type dataType,
                                     Value ldsOrig, Value regDest) {
      auto innerLoopK =
          outerLoopB.create<affine::AffineForOp>(loc, 0, kpackPerThread);
      auto ilkb = ConversionPatternRewriter::atBlockBegin(innerLoopK.getBody());
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(outerLoopBodyOp);
        OpBuilder::InsertionGuard guardBody(outerLoopB);
        outerLoopB.setInsertionPointToStart(outerLoopBodyOp.getBody());
        ldsToRegisterCopy(loc, outerLoopB, ilkb, sourceBase,
                          outerLoopBodyOp.getInductionVar(), MN,
                          innerLoopK.getInductionVar(), KPerThreadConstantOp,
                          mnPerMfmaGroup, mnWaves, ldsBufferElemType, dataType,
                          ldsOrig, regDest);
      }
    };

    // load A from LDS into registers
    // for(index_t m_i = 0; m_i < mRepeats; ++m_i)
    //   for(index_t k_i = 0; k_i < KPerThread; ++k_i)
    //       ldsToRegisterCopy[m_i, k_i]
    auto outerLoopM = b.create<affine::AffineForOp>(loc, 0, mRepeats);
    auto olmb = ConversionPatternRewriter::atBlockBegin(outerLoopM.getBody());
    ldsToRegisterCopyKdim(olmb, outerLoopM, sourceOffsetA, MConstantOp,
                          mPerAccelConstantOp, mWavesConstantOp,
                          bufferElemTypeA, dataTypeA, op.getMatrixA(), bufferA);

    // load B from LDS into registers
    // for(index_t n_i = 0; n_i < mRepeats; ++n_i)
    //   for(index_t k_i = 0; k_i < KPerThread; ++k_i)
    //       ldsToRegisterCopy[n_i, k_i]
    auto outerLoopN = olmb.create<affine::AffineForOp>(loc, 0, nRepeats);
    auto olnb = ConversionPatternRewriter::atBlockBegin(outerLoopN.getBody());
    ldsToRegisterCopyKdim(olnb, outerLoopN, sourceOffsetB, NConstantOp,
                          nPerAccelConstantOp, nWavesConstantOp,
                          bufferElemTypeB, dataTypeB, op.getMatrixB(), bufferB);

    b.eraseOp(op);
    olnb.create<AccelGemmOp>(loc, outerLoopM.getInductionVar(),
                             outerLoopN.getInductionVar(), adaptor.getBufferA(),
                             adaptor.getBufferB(), adaptor.getMatrixC(), arch,
                             op.getFeaturesAttr(), tuningParams);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GlobalLoadOp lowering.
//===----------------------------------------------------------------------===//
struct GlobalLoadRewritePattern : public OpRewritePattern<GlobalLoadOp> {
  using OpRewritePattern<GlobalLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalLoadOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    MemRefType sourceType = op.getSource().getType();
    Type sourceElemType = sourceType.getElementType();
    int64_t elemsPerWord = (32 / sourceElemType.getIntOrFloatBitWidth());
    int64_t maxLoadLen = 4 * elemsPerWord;

    Type resType = op.getResult().getType();
    int64_t totalLength = 1;
    if (auto vecType = resType.dyn_cast<VectorType>()) {
      totalLength = vecType.getNumElements();
    }
    // Don't use any vector magic if we don't need to
    if ((totalLength <= maxLoadLen) && (maxLoadLen % totalLength == 0)) {
      Type typeToLoad = sourceElemType;
      if (totalLength > 1)
        typeToLoad = VectorType::get({totalLength}, typeToLoad);
      BufferLoadOp load =
          b.create<BufferLoadOp>(loc, typeToLoad, op.getSource(), op.getValid(),
                                 op.getSourceCoord(), IntegerAttr(),
                                 /*oobIsOverload=*/nullptr);
      b.replaceOp(op, load);
      return success();
    }
    int64_t remainingLength = totalLength;
    int64_t offset = 0;

    Value result = createZeroConstantOp(b, loc, resType);

    while (remainingLength > 0) {
      int64_t copyLength = std::min(remainingLength, maxLoadLen);

      // Clean up bad copy lengths
      if (copyLength != maxLoadLen && copyLength > (2 * elemsPerWord))
        copyLength = 2 * elemsPerWord;
      if (copyLength > elemsPerWord && copyLength % elemsPerWord != 0)
        copyLength = elemsPerWord;
      if (copyLength > 1 && copyLength < elemsPerWord)
        // TODO: revisit this to handle things like (2xi8) -> load short
        copyLength = 1;

      Type typeToLoad = sourceElemType;
      if (copyLength > 1)
        typeToLoad = VectorType::get({copyLength}, typeToLoad);

      IntegerAttr offsetAttr =
          (offset > 0) ? b.getIndexAttr(offset) : IntegerAttr();

      Value loaded =
          b.create<BufferLoadOp>(loc, typeToLoad, op.getSource(), op.getValid(),
                                 op.getSourceCoord(), offsetAttr,
                                 /*oobIsOverload=*/nullptr);
      if (totalLength == 1) {
        result = loaded;
      } else {
        Value offsetIdx = b.createOrFold<ConstantIndexOp>(loc, offset);
        result =
            b.create<InsertSliceOp>(loc, resType, loaded, result, offsetIdx);
      }

      remainingLength -= copyLength;
      offset += copyLength;
    }
    b.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GlobalStore lowering.
//===----------------------------------------------------------------------===//
struct GlobalStoreRewritePattern : public OpRewritePattern<GlobalStoreOp> {
  using OpRewritePattern<GlobalStoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalStoreOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value source = op.getSource();
    MemRefType sourceType = op.getSource().getType();
    Value sourceCoord = op.getSourceCoord();

    Type sourceElemType = sourceType.getElementType();
    Type destElemType = op.getDest().getType().getElementType();
    int64_t elemsPerWord = (32 / destElemType.getIntOrFloatBitWidth());
    int64_t maxWriteLen = 4 * elemsPerWord;
    int64_t remainingLength = op.getLength().getSExtValue();
    int64_t offset = 0;
    // Don't use any vector magic if we don't need to
    if ((remainingLength <= maxWriteLen) &&
        (maxWriteLen % remainingLength == 0)) {
      Type typeToLoad = sourceElemType;
      if (remainingLength > 1)
        typeToLoad = VectorType::get({remainingLength}, typeToLoad);
      Value loaded =
          b.create<InBoundsLoadOp>(loc, typeToLoad, source, sourceCoord);
      b.create<BufferStoreOp>(loc, loaded, op.getDest(), op.getValid(),
                              op.getDestCoord(), op.getFeaturesAttr(),
                              op.getStoreMethodAttr(), IntegerAttr(),
                              /*oobIsOverload=*/nullptr);
      b.eraseOp(op);
      return success();
    }
    while (remainingLength > 0) {
      int64_t copyLength = std::min(remainingLength, maxWriteLen);

      // Clean up bad copy lengths
      if (copyLength != maxWriteLen && copyLength > (2 * elemsPerWord))
        copyLength = 2 * elemsPerWord;
      if (copyLength > elemsPerWord && copyLength % elemsPerWord != 0)
        copyLength = elemsPerWord;
      if (copyLength > 1 && copyLength < elemsPerWord)
        copyLength = 1;

      Type typeToLoad = sourceElemType;
      if (copyLength > 1)
        typeToLoad = VectorType::get({copyLength}, typeToLoad);
      Type typeToStore = destElemType;
      if (copyLength > 1)
        typeToStore = VectorType::get({copyLength}, typeToStore);

      Value loadCoord = sourceCoord;
      if (offset > 0)
        loadCoord = b.createOrFold<AddIOp>(
            loc, sourceCoord, b.create<ConstantIndexOp>(loc, offset));
      Value loaded =
          b.create<InBoundsLoadOp>(loc, typeToLoad, source, loadCoord);
      IntegerAttr offsetAttr =
          (offset > 0) ? b.getIndexAttr(offset) : IntegerAttr();
      b.create<BufferStoreOp>(loc, loaded, op.getDest(), op.getValid(),
                              op.getDestCoord(), op.getFeaturesAttr(),
                              op.getStoreMethodAttr(), offsetAttr,
                              /*oobIsOverflow=*/nullptr);
      remainingLength -= copyLength;
      offset += copyLength;
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

LogicalResult ThreadwiseReadIntoRewritePattern::matchAndRewrite(
    ThreadwiseReadIntoOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  auto sourceView = cast<TypedValue<MemRefType>>(adaptor.getSource());
  auto dest = cast<TypedValue<MemRefType>>(adaptor.getDest());

  auto [buffer, transforms] = untransform(b, sourceView, op.getExtraViews());

  int64_t numValues = dest.getType().getNumElements();
  MemRefType srcBufferType = buffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> bufferShape = srcBufferType.getShape();
  size_t extraIdxCount = op.getExtraIndices().size();

  // Unless specified it is assumed to be global
  gpu::AddressSpace srcAddrSpace = gpu::AddressSpace::Global;
  if (srcBufferType.getMemorySpace()) {
    srcAddrSpace =
        srcBufferType.getMemorySpace().cast<gpu::AddressSpaceAttr>().getValue();
  }

  // We are vectorizing in the iter dimension, not block ID or thread ID
  auto elementType = sourceView.getType().getElementType();
  int64_t vectorLen = getMaxVectorizationForDatatype(
      transforms, /*dim=*/extraIdxCount, numValues, bufferShape, elementType);
  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for read_into = " << vectorLen
                          << "\n");

  Type loadType = vectorTypeOrSelf(elementType, vectorLen);
  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  // In the future, this might get merged into the vectorizer.
  transforms = collapseContiguousMerges(transforms, bufferShape);

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  Value bid = b.createOrFold<rock::WorkgroupIdOp>(loc, b.getIndexType());
  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());

  SmallVector<Value, 3> readStartCoords =
      llvm::to_vector<3>(op.getExtraIndices());
  readStartCoords.push_back(zero);
  SmallVector<int64_t, 3> bounds(readStartCoords.size() - 1, 1);
  bounds.push_back(numValues);
  SmallVector<int64_t, 3> strides(readStartCoords.size() - 1, 1);
  strides.push_back(vectorLen);

  auto loadLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{readStartCoords, readStartCoords},
      ArrayRef<Attribute>{transforms, b.getArrayAttr({})}, bounds, strides,
      forceUnroll, useIndexDiffs);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loadLoop.getBody());
    if (srcAddrSpace == gpu::AddressSpace::Global) {
      Value loaded = b.create<GlobalLoadOp>(
          loc, loadType, buffer, loadLoop.getValidity(/*domain=*/0),
          loadLoop.getLowerCoords(/*domain=*/0));
      b.create<InBoundsStoreOp>(loc, loaded, dest,
                                loadLoop.getLowerCoords(
                                    /*domain=*/1)[extraIdxCount]);
    } else {
      TypedValue<IntegerType> valid = loadLoop.getValidity(/*domain=*/0);
      scf::IfOp ifb =
          b.create<scf::IfOp>(loc, loadType, valid, /*withElseRegion=*/true);
      {
        OpBuilder thenb = ifb.getThenBodyBuilder();
        Value loaded = thenb.create<InBoundsLoadOp>(
            loc, loadType, buffer, loadLoop.getLowerCoords(/*domain=*/0));
        thenb.create<scf::YieldOp>(loc, loaded);
      }
      {
        OpBuilder elseb = ifb.getElseBodyBuilder();
        Value zeroVal = createZeroConstantOp(elseb, loc, loadType);
        elseb.create<scf::YieldOp>(loc, zeroVal);
      }
      b.create<InBoundsStoreOp>(loc, ifb.getResult(0), dest,
                                loadLoop.getLowerCoords(
                                    /*domain=*/1)[extraIdxCount]);
    }
  }
  b.eraseOp(op);
  return success();
}

LogicalResult ThreadwiseWriteAllRewritePattern::matchAndRewrite(
    ThreadwiseWriteAllOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();

  auto source = cast<TypedValue<MemRefType>>(adaptor.getSource());
  auto destView = cast<TypedValue<MemRefType>>(adaptor.getDest());

  auto elementType = destView.getType().getElementType();

  ArrayAttr extraViews = op.getExtraViews();
  ArrayRef<int64_t> outputShape;
  if (extraViews.empty())
    outputShape = destView.getType().getShape();
  else
    outputShape = extraViews[0].cast<TransformMapAttr>().getUpperBounds();
  auto [buffer, transforms] = untransform(b, destView, extraViews);

  MemRefType dstBufferType = buffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> bufferShape = dstBufferType.getShape();
  size_t extraIdxCount = op.getExtraIndices().size();

  // Unless specified it is assumed to be global
  gpu::AddressSpace dstAddrSpace = gpu::AddressSpace::Global;
  if (dstBufferType.getMemorySpace()) {
    dstAddrSpace =
        dstBufferType.getMemorySpace().cast<gpu::AddressSpaceAttr>().getValue();
  }
  int64_t iterLen = outputShape[extraIdxCount];

  // We are vectorizing in the iter dimension, not block ID or thread ID
  int64_t vectorLen = getMaxVectorizationForDatatype(
      transforms, /*dim=*/extraIdxCount, iterLen, bufferShape, elementType);
  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for write_all = " << vectorLen
                          << "\n");

  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  transforms = collapseContiguousMerges(transforms, bufferShape);

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  Value bid = b.createOrFold<rock::WorkgroupIdOp>(loc, b.getIndexType());
  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());

  SmallVector<Value, 3> writeStartCoords =
      llvm::to_vector<3>(op.getExtraIndices());
  writeStartCoords.push_back(zero);
  SmallVector<int64_t, 3> bounds(writeStartCoords.size() - 1, 1);
  bounds.push_back(iterLen);
  SmallVector<int64_t, 3> strides(writeStartCoords.size() - 1, 1);
  strides.push_back(vectorLen);

  auto outLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{writeStartCoords, writeStartCoords},
      ArrayRef<Attribute>{b.getArrayAttr({}), transforms}, bounds, strides,
      forceUnroll, useIndexDiffs);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(outLoop.getBody());
    if (dstAddrSpace == gpu::AddressSpace::Global) {
      b.create<GlobalStoreOp>(loc, source, buffer, b.getIndexAttr(vectorLen),
                              op.getFeaturesAttr(), op.getStoreMethodAttr(),
                              outLoop.getLowerCoords(
                                  /*domain=*/0)[extraIdxCount],
                              outLoop.getValidity(/*domain=*/1),
                              outLoop.getLowerCoords(/*domain=*/1));
    } else {
      Type loadType = vectorTypeOrSelf(elementType, vectorLen);
      TypedValue<IntegerType> valid = outLoop.getValidity(/*domain=*/0);
      scf::IfOp ifb = b.create<scf::IfOp>(loc, valid, /*withElseRegion=*/false);
      {
        OpBuilder thenb = ifb.getThenBodyBuilder();
        Value loaded =
            thenb.create<InBoundsLoadOp>(loc, loadType, source,
                                         outLoop.getLowerCoords(
                                             /*domain=*/0)[extraIdxCount]);
        thenb.create<InBoundsStoreOp>(loc, loaded, buffer,
                                      outLoop.getLowerCoords(/*domain=*/1));
      }
    }
  }
  b.eraseOp(op);
  return success();
}

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

  // This function will append views to target a flat LDS buffer
  // where non-reduction dims are laid contigously as they are expected
  // function on parallel.
  ArrayAttr createLDSWorkspaceView(Location loc, PatternRewriter &rewriter,
                                   ArrayAttr regTensorView, int64_t reduceAxis,
                                   bool makeRDimZero = false) const {

    TransformMapAttr lowestTr =
        regTensorView[regTensorView.size() - 1].cast<TransformMapAttr>();
    ArrayRef<int64_t> lowestShape = lowestTr.getLowerBounds().asArrayRef();
    TopDownTMBuilder tensorToLDSViewBuilder(rewriter, lowestShape, loc);
    SmallVector<StringRef, 4> upperNameRefs;
    tensorToLDSViewBuilder.getStartNames(upperNameRefs);

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
    tensorToLDSViewBuilder.unmerge("nrDim", 1, nonReduceNameRefs,
                                   nonReduceDimSizes);
    if (makeRDimZero) {
      tensorToLDSViewBuilder.constDim("rDim", 0, 0, lowestShape[reduceAxis]);
    } else {
      tensorToLDSViewBuilder.passThrough({"rDim"}, {0},
                                         {upperNameRefs[reduceAxis]});
    }
    TransformMapAttr twoDimLDSView = tensorToLDSViewBuilder.get();

    TopDownTMBuilder flatLDSViewBuilder =
        TopDownTMBuilder::below(tensorToLDSViewBuilder, twoDimLDSView);
    flatLDSViewBuilder.unmerge(
        "flatDim", 0, {"nrDim", "rDim"},
        {nonReduceMergeDimSize, lowestShape[reduceAxis]});
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

  arith::ConstantOp
  getReductionInitValue(BlockwiseBroadcastReduceOp op,
                        ConversionPatternRewriter &rewriter) const {
    ReduceMethod rMethod = op.getReduceMethod();
    TypedAttr initValAttr;
    Type elementType = op.getInput().getType().getElementType();
    if (elementType.isIntOrIndex()) {
      int64_t initVal;
      if (rMethod == ReduceMethod::Sum) {
        initVal = 0;
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        initVal = std::numeric_limits<int64_t>::min();
      }
      initValAttr = rewriter.getIntegerAttr(elementType, initVal);
    } else {
      double initVal;
      if (rMethod == ReduceMethod::Sum) {
        initVal = 0.0;
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        initVal = std::numeric_limits<double>::min();
      }
      initValAttr = rewriter.getFloatAttr(elementType, initVal);
    }
    return rewriter.create<arith::ConstantOp>(op.getLoc(), elementType,
                                              initValAttr);
  }

  Value createReducingOp(BlockwiseBroadcastReduceOp op, Value input, Value acc,
                         OpBuilder &builder) const {
    ReduceMethod rMethod = op.getReduceMethod();
    Location loc = op.getLoc();
    // Value loadAcc = rewriter.create<InBoundsLoadOp>(loc, input.getType(),
    // acc, zeroConstantOp);
    Type elementType = op.getInput().getType().getElementType();

    if (!acc.getType().isa<VectorType>() && input.getType().isa<VectorType>()) {
      // This means accumulator is a scalar type and input is a vector type,
      // therefore its a elementwise reduction between two operands.
      vector::CombiningKind kind;
      if (rMethod == ReduceMethod::Sum) {
        kind = vector::CombiningKind::ADD;
      } else {
        // Op verifier gurantees this.
        assert(rMethod == ReduceMethod::Max);
        if (elementType.isIntOrIndex()) {
          kind = vector::CombiningKind::MAXF;
        } else {
          kind = vector::CombiningKind::MAXF;
        }
      }
      input = builder.create<vector::ReductionOp>(loc, kind, input);
    }

    if (rMethod == ReduceMethod::Sum) {
      Value reduced;
      if (elementType.isIntOrIndex()) {
        reduced = builder.create<arith::AddIOp>(loc, acc, input);
      } else {
        reduced = builder.create<arith::AddFOp>(loc, acc, input);
      }
      return reduced;
    } else {
      assert(rMethod == ReduceMethod::Max);
      Value reduced;
      if (elementType.isIntOrIndex()) {
        reduced = builder.create<arith::MaxSIOp>(loc, acc, input);
      } else {
        reduced = builder.create<arith::MaxFOp>(loc, acc, input);
      }
      return reduced;
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
    Value zeroConstantOp = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    int64_t axis = op.getAxis().getSExtValue();
    int64_t blockSize = op.getBlockSize();
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    // Get current workitem ID.
    WorkitemIdOp tid =
        rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());

    // Create strides and bounds to iterate the virtual tensor
    TransformMapAttr lowerTr = inputViewArrayAttr[0].cast<TransformMapAttr>();
    ArrayRef<int64_t> lowerTrLowerBounds =
        lowerTr.getLowerBounds().asArrayRef();
    SmallVector<int64_t, 4> regTensorShape =
        llvm::to_vector<4>(lowerTrLowerBounds);

    rewriter.create<ThreadwiseWriteAllOp>(
        loc, inputReg, workspaceLDSBuffer,
        createLDSWorkspaceView(loc, rewriter, inputViewArrayAttr, axis),
        /*extraIndices=*/ValueRange{tid}, rock::GemmFeatures::none,
        StoreMethod::Set, true, true);

    // Following RAII scope will create reduction loops.
    {
      int64_t nonReductionDimSizeProduct =
          calculateNonReductionDimProduct(regTensorShape, axis);
      if (blockSize <= nonReductionDimSizeProduct) {
        // This means there aren't enough threads to do a parallel reduction
        // each individual thread could do its own reduction.
        ArrayAttr threadsToTensorTrs = createThreadViewForNRLargerThanThreads(
            loc, regTensorShape, blockSize, axis, rewriter);
        ArrayAttr threadToLDSViewTrs =
            createLDSWorkspaceView(loc, rewriter, threadsToTensorTrs, axis);
        ArrayAttr threadsToLDSViewReducedTrs = createLDSWorkspaceView(
            loc, rewriter, threadsToTensorTrs, axis, /*makeRDimZero-*/ true);
        ArrayRef<int64_t> threadViewShape =
            threadToLDSViewTrs[0].cast<TransformMapAttr>().getUpperBounds();
        ArrayRef<int64_t> ldsBufferShape =
            threadToLDSViewTrs[threadToLDSViewTrs.size() - 1]
                .cast<TransformMapAttr>()
                .getLowerBounds();
        constexpr size_t nrIterDim = 1;
        constexpr size_t rIterDim = 2;

        int64_t nrIterVectorLen = getMaxVectorizationForDatatype(
            threadToLDSViewTrs, nrIterDim, threadViewShape[nrIterDim],
            ldsBufferShape, elemType);
        // Create the accumulation register
        // This will be accumulated over non-reduction iterations.
        auto accRegType = MemRefType::get(
            nrIterVectorLen, elemType, AffineMap{}, privateMemoryAddressSpace);
        Value accReg = rewriter.create<GpuAllocOp>(loc, accRegType);
        arith::ConstantOp initVal = getReductionInitValue(op, rewriter);
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          Value nrIter;
          if (threadViewShape[nrIterDim] > 1) {
            AffineForOp nrIterLoop = rewriter.create<AffineForOp>(
                loc, 0, threadViewShape[nrIterDim] - 1, nrIterVectorLen);
            // inside the loop.
            rewriter.setInsertionPointToStart(nrIterLoop.getBody());
            nrIter = nrIterLoop.getInductionVar();
          } else {
            nrIter = zeroConstantOp;
          }
          rewriter.create<FillOp>(loc, accReg, initVal.getResult());
          int64_t rIterVectorLen = getMaxVectorizationForDatatype(
              threadToLDSViewTrs, rIterDim, threadViewShape[rIterDim],
              ldsBufferShape, elemType);
          SmallVector<Value, 4> inits{tid, nrIter, zeroConstantOp};
          SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
          SmallVector<int64_t> strides{1, 1, rIterVectorLen};

          TransformingForOp reductionLoop = rewriter.create<TransformingForOp>(
              loc, ArrayRef<ValueRange>{inits, inits, inits},
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
            Value loadVal = rewriter.create<InBoundsLoadOp>(
                loc, vectorTypeOrSelf(elemType, rIterVectorLen),
                workspaceLDSBuffer, LDSLoadCoords);
            Value loadAcc = rewriter.create<InBoundsLoadOp>(
                loc, vectorTypeOrSelf(elemType, nrIterVectorLen), accReg,
                zeroConstantOp);
            Value reduced = createReducingOp(op, loadVal, loadAcc, rewriter);
            rewriter.create<InBoundsStoreOp>(loc, reduced, accReg,
                                             zeroConstantOp);
            // Storing the last reduction iter output directly to LDS[..., dr=0,
            // ...]
            Value rIterArg =
                reductionLoop.getLowerCoords(/*domain=*/1)[rIterDim];
            Value boundVal = rewriter.create<arith::ConstantIndexOp>(
                loc, threadViewShape[rIterDim]);
            Value strideVal =
                rewriter.create<arith::ConstantIndexOp>(loc, rIterVectorLen);
            Value lastIterVal =
                rewriter.create<arith::SubIOp>(loc, boundVal, strideVal);
            Value isLastIter = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, rIterArg, lastIterVal);
            scf::IfOp ifb = rewriter.create<scf::IfOp>(
                loc, isLastIter, /*withElseRegion=*/false);
            {
              OpBuilder thenb = ifb.getThenBodyBuilder();
              thenb.create<InBoundsStoreOp>(
                  loc, reduced, workspaceLDSBuffer,
                  reductionLoop.getLowerCoords(/*domain=*/2));
            }
          }
        }
        ArrayAttr reducedldsViewArrayAttr = createLDSWorkspaceView(
            loc, rewriter, inputViewArrayAttr, axis, /*makeRDimZero-*/ true);
        rewriter.create<ThreadwiseReadIntoOp>(
            loc, workspaceLDSBuffer, outputReg, reducedldsViewArrayAttr,
            /*extraIndices=*/ValueRange{tid}, true, true);
      } else {
        // This means there are more threads than elements to be reduced.
        ArrayAttr threadToTensorViewTrs =
            createThreadViewforNRSmallerThanThreads(loc, regTensorShape,
                                                    blockSize, axis, rewriter);
        ArrayAttr threadToLDSViewTrs =
            createLDSWorkspaceView(loc, rewriter, threadToTensorViewTrs, axis);
        ArrayRef<int64_t> threadViewShape =
            threadToLDSViewTrs[0].cast<TransformMapAttr>().getUpperBounds();
        ArrayRef<int64_t> ldsBufferShape =
            threadToLDSViewTrs[threadToLDSViewTrs.size() - 1]
                .cast<TransformMapAttr>()
                .getLowerBounds();
        constexpr size_t rTidDim = 1;
        constexpr size_t rIterDim = 2;

        int64_t rIterVectorLen = getMaxVectorizationForDatatype(
            threadToLDSViewTrs, rIterDim, threadViewShape[rIterDim],
            ldsBufferShape, elemType);
        Value nrDimSizeProductConst = rewriter.create<arith::ConstantIndexOp>(
            loc, nonReductionDimSizeProduct);
        Value rtid =
            rewriter.create<arith::DivSIOp>(loc, tid, nrDimSizeProductConst);
        Value nrtid =
            rewriter.create<arith::RemSIOp>(loc, tid, nrDimSizeProductConst);
        Value rtidDimSizeVal = rewriter.create<arith::ConstantIndexOp>(
            loc, threadViewShape[rTidDim]);

        // We need to do the threadwise reduction
        // here only if rIterDim is meaninfully iterated
        // otherwise this step can be skipped.
        if (threadViewShape[rIterDim] > 1) {
          // This is where thread_wise reduction result is stored.
          Type loadTypeInputReg = vectorTypeOrSelf(elemType, rIterVectorLen);
          Value accReg = rewriter.create<GpuAllocOp>(
              loc, MemRefType::get({1}, elemType, AffineMap{},
                                   privateMemoryAddressSpace));
          // This RAII scope would create a loop to iteratively partialy reduce
          // on a thread basis until items to reduce will match the available
          // number of threads.
          {
            SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
            SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
            SmallVector<int64_t> strides{1, 1, rIterVectorLen};

            arith::ConstantOp initVal = getReductionInitValue(op, rewriter);
            rewriter.create<FillOp>(loc, accReg, initVal.getResult());

            TransformingForOp reductionLoop =
                rewriter.create<TransformingForOp>(
                    loc, ArrayRef<ValueRange>(inits),
                    ArrayRef<Attribute>{threadToLDSViewTrs},
                    ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                    /*forceUnroll=*/true, /*useIndexDiffs=*/true);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(reductionLoop.getBody());
              Block::BlockArgListType LDSLoadCoords =
                  reductionLoop.getLowerCoords(/*domain=*/0);
              Value loadVal = rewriter.create<InBoundsLoadOp>(
                  loc, loadTypeInputReg, workspaceLDSBuffer, LDSLoadCoords);
              Value loadAcc = rewriter.create<InBoundsLoadOp>(
                  loc, elemType, accReg, zeroConstantOp);
              Value reduced = createReducingOp(op, loadVal, loadAcc, rewriter);
              rewriter.create<InBoundsStoreOp>(loc, reduced, accReg,
                                               zeroConstantOp);
            }
          }

          // This RAII scope would store the partial reductions to
          // LDS
          {
            SmallVector<Value, 4> inits{nrtid, rtid, zeroConstantOp};
            SmallVector<int64_t> bounds{1, 1, 1};
            SmallVector<int64_t> strides{1, 1, 1};

            TransformingForOp reductionLoop =
                rewriter.create<TransformingForOp>(
                    loc, ArrayRef<ValueRange>(inits),
                    ArrayRef<Attribute>{threadToLDSViewTrs},
                    ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                    /*forceUnroll=*/true, /*useIndexDiffs=*/true);
            {
              PatternRewriter::InsertionGuard guard(rewriter);
              rewriter.setInsertionPointToStart(reductionLoop.getBody());
              Block::BlockArgListType LDSStoreCoords =
                  reductionLoop.getLowerCoords(/*domain=*/0);
              Value loadVal = rewriter.create<InBoundsLoadOp>(
                  loc, loadTypeInputReg, accReg, zeroConstantOp);
              rewriter.create<InBoundsStoreOp>(loc, loadVal, workspaceLDSBuffer,
                                               LDSStoreCoords);
            }
            rewriter.create<LDSBarrierOp>(loc);
          }
        }

        // This RAII scope would do the following :
        // LDS[rtid] = reduce(LDS[rtid], LDS[rtid + offset])
        // where offset is a power of 2.
        // Initial it starts with power = ceil(|rtid| / 2, power of 2)
        // Then keep on reducing the power.
        {
          double log2HalfRtidDimSize =
              std::log2(static_cast<double>(threadViewShape[rTidDim]) / 2);
          int64_t ceilLog2HalfRtidDimSize =
              static_cast<int64_t>(std::ceil(log2HalfRtidDimSize));
          int64_t ceilPowerOf2 = (int64_t)1 << ceilLog2HalfRtidDimSize;

          for (int64_t offset = ceilPowerOf2; offset >= 1;
               offset = offset >> 1) {
            Value offsetVal =
                rewriter.create<arith::ConstantIndexOp>(loc, offset);
            Value rtidPlusOffsetVal =
                rewriter.create<arith::AddIOp>(loc, rtid, offsetVal);
            Value isValid = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, rtidPlusOffsetVal,
                rtidDimSizeVal);
            scf::IfOp ifb = rewriter.create<scf::IfOp>(
                loc, isValid, /*withElseRegion=*/false);
            {
              OpBuilder thenb = ifb.getThenBodyBuilder();
              SmallVector<Value, 4> firstInits{nrtid, rtid, zeroConstantOp};
              SmallVector<Value, 4> secondInits{nrtid, rtidPlusOffsetVal,
                                                zeroConstantOp};
              SmallVector<int64_t> bounds{1, 1, 1};
              SmallVector<int64_t> strides{1, 1, 1};

              TransformingForOp reductionLoop = thenb.create<TransformingForOp>(
                  loc, ArrayRef<ValueRange>{firstInits, secondInits},
                  ArrayRef<Attribute>{threadToLDSViewTrs, threadToLDSViewTrs},
                  ArrayRef<int64_t>(bounds), ArrayRef<int64_t>(strides),
                  /*forceUnroll=*/true, /*useIndexDiffs=*/true);
              {
                PatternRewriter::InsertionGuard guard(thenb);
                thenb.setInsertionPointToStart(reductionLoop.getBody());
                Block::BlockArgListType firstLDSLoadCoords =
                    reductionLoop.getLowerCoords(/*domain=*/0);
                Value firstLoadVal = thenb.create<InBoundsLoadOp>(
                    loc, elemType, workspaceLDSBuffer, firstLDSLoadCoords);
                Block::BlockArgListType secondLDSLoadCoords =
                    reductionLoop.getLowerCoords(/*domain=*/1);
                Value secondLoadVal = thenb.create<InBoundsLoadOp>(
                    loc, elemType, workspaceLDSBuffer, secondLDSLoadCoords);
                Value reduced =
                    createReducingOp(op, firstLoadVal, secondLoadVal, thenb);
                thenb.create<InBoundsStoreOp>(loc, reduced, workspaceLDSBuffer,
                                              firstLDSLoadCoords);
              }
            }
            rewriter.create<LDSBarrierOp>(loc);
          }
          ArrayAttr reducedldsViewArrayAttr = createLDSWorkspaceView(
              loc, rewriter, inputViewArrayAttr, axis, /*makeRDimZero-*/ true);
          rewriter.create<ThreadwiseReadIntoOp>(
              loc, workspaceLDSBuffer, outputReg, reducedldsViewArrayAttr,
              /*extraIndices=*/ValueRange{tid}, true, true);
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
    writeAllTarget.addIllegalOp<ThreadwiseReadIntoOp, ThreadwiseWriteAllOp,
                                BlockwiseBroadcastReduceOp, BlockwiseFillOp>();
    writeAllTarget.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                                   memref::MemRefDialect, scf::SCFDialect,
                                   vector::VectorDialect, AffineDialect>();
    writeAllTarget.addLegalOp<gpu::PrintfOp>();
    RewritePatternSet writeAllPatterns(ctx);
    writeAllPatterns
        .add<ThreadwiseReadIntoRewritePattern, ThreadwiseWriteAllRewritePattern,
             BlockwiseReduceRewritePattern, BlockwiseFillRewritePattern>(ctx);
    if (failed(applyPartialConversion(getOperation(), writeAllTarget,
                                      std::move(writeAllPatterns))))
      signalPassFailure();
  }

  ConversionTarget target(*ctx);
  target.addIllegalOp<FillOp, BlockwiseGemmOp, BlockwiseGemmAccelOp,
                      GlobalLoadOp, GlobalStoreOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         affine::AffineDialect, vector::VectorDialect,
                         memref::MemRefDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, BlockwiseGemmRewritePattern,
               BlockwiseGemmAccelRewritePattern, GlobalLoadRewritePattern,
               GlobalStoreRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
