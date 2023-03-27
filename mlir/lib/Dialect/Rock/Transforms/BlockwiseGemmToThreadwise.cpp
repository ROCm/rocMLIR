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
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

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

    buildAffineLoopNest(b, loc, lbs, inputShape, strides,
                        [value = adaptor.getValue(),
                         input = adaptor.getInput()](OpBuilder &b, Location loc,
                                                     ValueRange ivs) {
                          b.create<memref::StoreOp>(loc, value, input, ivs);
                        });

    b.replaceOp(op, {});
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

    // Non-xdlops path.

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
    auto loopOp = b.replaceOpWithNewOp<AffineForOp>(op, 0, k, kPerThread);
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
// BlockwiseGemmV2 lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmV2RewritePattern
    : public OpConversionPattern<BlockwiseGemmV2Op> {
  using OpConversionPattern<BlockwiseGemmV2Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockwiseGemmV2Op op,
                                BlockwiseGemmV2OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    StringAttr arch = op.getArchAttr();
    XdlopsGemmParamsAttr tuningParams = op.getParams();
    int64_t M = tuningParams.getMPerBlock();
    int64_t N = tuningParams.getNPerBlock();
    int64_t K = tuningParams.getKPerBlock();
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();
    int64_t KPack = tuningParams.getKpack();

    int64_t ldsOffsetA = op.getLdsBufferOffsetA().getSExtValue();
    int64_t ldsOffsetB = op.getLdsBufferOffsetB().getSExtValue();

    assert(ldsOffsetA % KPack == 0 &&
           "LDS buffer segment for A is kpack-aligned");
    assert(ldsOffsetB % KPack == 0 &&
           "LDS buffer segment for B is kpack-aligned");
    auto dataType =
        adaptor.getMatrixA().getType().cast<MemRefType>().getElementType();

    // The address calculations into the LDS buffer assume that the buffer
    // has type vector<KPack x T>. Then, we convert that into an address
    // in a buffer of Ts through a final multiplicaiton by KPack.
    // However, the LDS buffer offset, which was computed when the buffer was
    // allocated, is an offset into a buffer of T. Therefore, to allow it to
    // easily participate in adress calculations (instead of adding it on at the
    // end) we must divide it by KPack here. Fortunately, this offset will be
    // KPack-alligned and so this is safe
    Value sourceOffsetA =
        b.create<AddIOp>(loc, adaptor.getWaveOffsetA(),
                         b.create<ConstantIndexOp>(loc, ldsOffsetA / KPack));
    Value sourceOffsetB =
        b.create<AddIOp>(loc, adaptor.getWaveOffsetB(),
                         b.create<ConstantIndexOp>(loc, ldsOffsetB / KPack));

    auto maybeMfmaInsnGroup =
        MfmaInsnGroup::select(dataType, arch, mPerWave, nPerWave);
    if (failed(maybeMfmaInsnGroup)) {
      return emitError(loc) << "Failed to select xdlops instruction group.\n";
    }
    MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;

    Type argType = mfmaGroup.getArgType();

    MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
    int64_t inputSpanLen = mfmaAttr.inputSpanLen;
    int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
    int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;
    int64_t k_base = mfmaAttr.k_base;

    int64_t mRepeats = mfmaGroup.getMRepeats(mPerWave);
    int64_t nRepeats = mfmaGroup.getNRepeats(nPerWave);

    int64_t mPerMfmaGroup = mfmaGroup.getLenPerMfmaGroup(mPerWave);
    int64_t nPerMfmaGroup = mfmaGroup.getLenPerMfmaGroup(nPerWave);

    bool IsKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);

    if (KPack > 1 && (KPack < k_base || KPack % k_base != 0)) {
      llvm_unreachable(
          "Tuning parameter selection guarantees kPack is multiple of k_base,"
          "this should never happen");
    }

    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());
    constexpr int64_t waveSize = 64;
    auto laneId =
        b.create<RemUIOp>(loc, tid, b.create<ConstantIndexOp>(loc, waveSize));

    LLVM_DEBUG(llvm::dbgs()
               << "argVectorType: " << argType << "\n"
               << "k_base: " << k_base << "\n"
               << "K: " << K << "\n"
               << "bufferA type: " << adaptor.getBufferA().getType() << "\n"
               << "bufferB type: " << adaptor.getBufferB().getType() << "\n");

    Value MConstantOp = b.create<ConstantIndexOp>(loc, M);
    Value NConstantOp = b.create<ConstantIndexOp>(loc, N);

    Value mPerMfmaGroupConstantOp =
        b.create<ConstantIndexOp>(loc, mPerMfmaGroup);
    Value nPerMfmaGroupConstantOp =
        b.create<ConstantIndexOp>(loc, nPerMfmaGroup);

    Value bufferA = adaptor.getBufferA();
    Value bufferB = adaptor.getBufferB();

    int64_t KPerThread = IsKReduction ? K / inputSpansPerMfmaIn : K;
    Value KPerThreadConstantOp = b.create<ConstantIndexOp>(loc, KPerThread);

    auto ldsToRegisterCopy = [&](Location loc, OpBuilder mnb, OpBuilder kb,
                                 Value sourceBase, Value mn_i, Value MN,
                                 Value k_i, Value K, Value mnPerMfmaGroup,
                                 Value ldsOrig, Value regDest) {
      // Compute source offset
      Value sourceOffset = sourceBase;
      if (!IsKReduction) {
        // srcOffset = k_i * MN + laneId + mPerMfmaGroup * mn_i;
        sourceOffset = b.create<AddIOp>(loc, sourceOffset, laneId);
        sourceOffset = mnb.create<AddIOp>(
            loc, sourceOffset, mnb.create<MulIOp>(loc, mnPerMfmaGroup, mn_i));
        sourceOffset = kb.create<AddIOp>(loc, sourceOffset,
                                         kb.create<MulIOp>(loc, MN, k_i));
      } else {
        // srcOffset = (k_i * input_span_per_mfma + blk_id) * MN + blk_td + mn_i
        // * input_span_length;
        Value inputSpanLenConstantOp =
            b.create<ConstantIndexOp>(loc, inputSpanLen);
        Value inputSpansPerMfmaInConstantOp =
            b.create<ConstantIndexOp>(loc, inputSpansPerMfmaIn);
        Value blk_id = b.create<DivUIOp>(loc, laneId, inputSpanLenConstantOp);
        Value blk_td = b.create<RemUIOp>(loc, laneId, inputSpanLenConstantOp);

        sourceOffset = b.create<AddIOp>(loc, sourceOffset, blk_td);
        sourceOffset = mnb.create<AddIOp>(
            loc, sourceOffset,
            mnb.create<MulIOp>(loc, inputSpanLenConstantOp, mn_i));
        sourceOffset = kb.create<AddIOp>(
            loc, sourceOffset,
            kb.create<MulIOp>(
                loc,
                kb.create<AddIOp>(
                    loc,
                    kb.create<MulIOp>(loc, k_i, inputSpansPerMfmaInConstantOp),
                    blk_id),
                MN));
      }
      if (KPack > 1)
        sourceOffset = kb.create<MulIOp>(
            loc, sourceOffset, kb.create<ConstantIndexOp>(loc, KPack));

      Type loadType = vectorTypeOrSelf(dataType, KPack);
      Value value =
          kb.create<InBoundsLoadOp>(loc, loadType, ldsOrig, sourceOffset);
      auto bufferType = regDest.getType().cast<MemRefType>();
      Type bufferElementType = bufferType.getElementType();

      // We're loading in units of kPack, but storing in units of k_base.
      if (KPack == k_base) {
        Value destOffset = k_i;
        kb.create<memref::StoreOp>(loc, value, regDest, ValueRange{destOffset});
      } else if (KPack > k_base) {
        int64_t numStores = KPack / k_base;
        Value baseDestOffset = kb.createOrFold<arith::MulIOp>(
            loc, k_i, kb.createOrFold<arith::ConstantIndexOp>(loc, numStores));
        for (int64_t i = 0; i < numStores; ++i) {
          Value sliceStart =
              kb.createOrFold<arith::ConstantIndexOp>(loc, k_base * i);
          Value slice = kb.create<ExtractSliceOp>(loc, bufferElementType, value,
                                                  sliceStart);
          Value destOffset = kb.createOrFold<arith::AddIOp>(
              loc, baseDestOffset,
              kb.createOrFold<arith::ConstantIndexOp>(loc, i));
          kb.create<memref::StoreOp>(loc, slice, regDest,
                                     ValueRange{destOffset});
        }
      } else if (KPack < k_base) {
        // Here we are gathering loaded values into vectors for passing into
        // MFMAs.
        Value destValsPerKpack =
            kb.createOrFold<arith::ConstantIndexOp>(loc, k_base / KPack);
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

    auto ldsToRegisterCopyKdim =
        [&](OpBuilder outerLoopB, AffineForOp outerLoopBodyOp, Value sourceBase,
            Value MN, Value mnPerMfmaGroup, Value ldsOrig, Value regDest) {
          auto innerLoopK = outerLoopB.create<AffineForOp>(loc, 0, KPerThread);
          auto ilkb = ConversionPatternRewriter::atBlockBegin(
              innerLoopK.getBody(), outerLoopB.getListener());
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPoint(outerLoopBodyOp);
            OpBuilder::InsertionGuard guardBody(outerLoopB);
            outerLoopB.setInsertionPointToStart(outerLoopBodyOp.getBody());
            ldsToRegisterCopy(loc, outerLoopB, ilkb, sourceBase,
                              outerLoopBodyOp.getInductionVar(), MN,
                              innerLoopK.getInductionVar(),
                              KPerThreadConstantOp, mnPerMfmaGroup, ldsOrig,
                              regDest);
          }
        };

    // load A from LDS into registers
    // for(index_t m_i = 0; m_i < mRepeats; ++m_i)
    //   for(index_t k_i = 0; k_i < KPerThread; ++k_i)
    //       ldsToRegisterCopy[m_i, k_i]
    auto outerLoopM = b.create<AffineForOp>(loc, 0, mRepeats);
    auto olmb = ConversionPatternRewriter::atBlockBegin(outerLoopM.getBody(),
                                                        b.getListener());
    ldsToRegisterCopyKdim(olmb, outerLoopM, sourceOffsetA, MConstantOp,
                          mPerMfmaGroupConstantOp, op.getMatrixA(), bufferA);

    // load B from LDS into registers
    // for(index_t n_i = 0; n_i < mRepeats; ++n_i)
    //   for(index_t k_i = 0; k_i < KPerThread; ++k_i)
    //       ldsToRegisterCopy[n_i, k_i]
    auto outerLoopN = olmb.create<AffineForOp>(loc, 0, nRepeats);
    auto olnb = ConversionPatternRewriter::atBlockBegin(outerLoopN.getBody(),
                                                        olmb.getListener());
    ldsToRegisterCopyKdim(olnb, outerLoopN, sourceOffsetB, NConstantOp,
                          nPerMfmaGroupConstantOp, op.getMatrixB(), bufferB);

    b.eraseOp(op);
    olnb.create<XdlopsGemmV2Op>(loc, outerLoopM.getInductionVar(),
                                outerLoopN.getInductionVar(),
                                adaptor.getBufferA(), adaptor.getBufferB(),
                                adaptor.getMatrixC(), arch, tuningParams);
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
      b.replaceOp(op, {load});
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
    b.replaceOp(op, {result});
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
  TypedValue<MemRefType> sourceView = adaptor.getSource();
  TypedValue<MemRefType> dest = adaptor.getDest();

  auto [buffer, transforms] = untransform(b, sourceView, op.getExtraViews());

  int64_t numValues = dest.getType().getNumElements();
  ArrayRef<int64_t> bufferShape =
      buffer.getType().cast<ShapedType>().getShape();

  // We are vectorizing in the iter dimension, not block ID or thread ID
  auto elementType = sourceView.getType().getElementType();
  int64_t vectorLen = getMaxVectorizationForDatatype(
      transforms, /*dim=*/2, numValues, bufferShape, elementType);
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

  SmallVector<Value, 3> readStartCoords = {bid, tid, zero};

  auto loadLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{readStartCoords, readStartCoords},
      ArrayRef<Attribute>{transforms, b.getArrayAttr({})},
      ArrayRef<int64_t>{1, 1, numValues}, ArrayRef<int64_t>{1, 1, vectorLen},
      forceUnroll, useIndexDiffs);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loadLoop.getBody());
    Value loaded = b.create<GlobalLoadOp>(
        loc, loadType, buffer, loadLoop.getValidity(/*domain=*/0),
        loadLoop.getLowerCoords(/*domain=*/0));
    b.create<InBoundsStoreOp>(loc, loaded, dest,
                              loadLoop.getLowerCoords(/*domain=*/1)[2]);
  }
  b.eraseOp(op);
  return success();
}

LogicalResult ThreadwiseWriteAllRewritePattern::matchAndRewrite(
    ThreadwiseWriteAllOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  TypedValue<MemRefType> source = adaptor.getSource();
  TypedValue<MemRefType> destView = adaptor.getDest();

  auto elementType = destView.getType().getElementType();

  auto [buffer, transforms] = untransform(b, destView, op.getExtraViews());

  int64_t numValues = source.getType().getNumElements();
  ArrayRef<int64_t> bufferShape =
      buffer.getType().cast<ShapedType>().getShape();

  // We are vectorizing in the iter dimension, not block ID or thread ID
  int64_t vectorLen = getMaxVectorizationForDatatype(
      transforms, /*dim=*/2, numValues, bufferShape, elementType);
  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for write_all = " << vectorLen
                          << "\n");

  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  transforms = collapseContiguousMerges(transforms, bufferShape);

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  Value bid = b.createOrFold<rock::WorkgroupIdOp>(loc, b.getIndexType());
  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());

  SmallVector<Value, 3> writeStartCoords = {bid, tid, zero};

  auto outLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{writeStartCoords, writeStartCoords},
      ArrayRef<Attribute>{b.getArrayAttr({}), transforms},
      ArrayRef<int64_t>{1, 1, numValues}, ArrayRef<int64_t>{1, 1, vectorLen},
      forceUnroll, useIndexDiffs);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(outLoop.getBody());
    b.create<GlobalStoreOp>(loc, source, buffer, b.getIndexAttr(vectorLen),
                            op.getFeaturesAttr(), op.getStoreMethodAttr(),
                            outLoop.getLowerCoords(/*domain=*/0)[2],
                            outLoop.getValidity(/*domain=*/1),
                            outLoop.getLowerCoords(/*domain=*/1));
  }
  b.eraseOp(op);
  return success();
}

void RockLowerBlockwiseGemmToThreadwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  {
    ConversionTarget writeAllTarget(*ctx);
    writeAllTarget.addIllegalOp<ThreadwiseReadIntoOp, ThreadwiseWriteAllOp>();
    writeAllTarget.addLegalDialect<arith::ArithDialect, rock::RockDialect>();
    RewritePatternSet writeAllPatterns(ctx);
    writeAllPatterns.add<ThreadwiseReadIntoRewritePattern,
                         ThreadwiseWriteAllRewritePattern>(ctx);
    if (failed(applyPartialConversion(getOperation(), writeAllTarget,
                                      std::move(writeAllPatterns))))
      signalPassFailure();
  }

  ConversionTarget target(*ctx);
  target.addIllegalOp<FillOp, BlockwiseGemmOp, BlockwiseGemmV2Op, GlobalLoadOp,
                      GlobalStoreOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect, AffineDialect,
                         memref::MemRefDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, BlockwiseGemmRewritePattern,
               BlockwiseGemmV2RewritePattern, GlobalLoadRewritePattern,
               GlobalStoreRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
