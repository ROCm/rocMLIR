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
                            mThreadsPerCuwave, nThreadsPerCuwave},
                           /*isUnfold=*/true);
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
    auto privateMemoryAddressSpace = gpu::AddressSpaceAttr::get(
        op->getContext(), gpu::GPUDialect::getPrivateAddressSpace());
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
    auto dataType = adaptor.getMatrixA()
                        .getType()
                        .template cast<MemRefType>()
                        .getElementType();

    // The address calculations into the LDS buffer assume that the buffer
    // has type vector<KPack x T>. Then, we convert that into an address
    // in a buffer of Ts through a final multiplicaiton by KPack.
    // However, the LDS buffer offset, which was computed when the buffer was
    // allocated, is an offset into a buffer of T. Therefore, to allow it to
    // easily participate in adress calculations (instead of adding it on at the
    // end) we must divide it by KPack here. Fortunately, this offset will be
    // KPack-alligned and so this is safe
    Value aBase =
        b.create<AddIOp>(loc, adaptor.getWaveOffsetA(),
                         b.create<ConstantIndexOp>(loc, ldsOffsetA / KPack));
    Value bBase =
        b.create<AddIOp>(loc, adaptor.getWaveOffsetB(),
                         b.create<ConstantIndexOp>(loc, ldsOffsetB / KPack));

    auto maybeMfmaInsnGroup =
        MfmaInsnGroup::select(dataType, mPerWave, nPerWave);
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

    int64_t mRepeats = mfmaGroup.getMRepeats();
    int64_t nRepeats = mfmaGroup.getNRepeats();

    int64_t mPerMfmaGroup = mfmaGroup.getLenPerMfmaGroup(mPerWave);
    int64_t nPerMfmaGroup = mfmaGroup.getLenPerMfmaGroup(nPerWave);

    bool IsKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);

    if (KPack > 1 && (KPack < k_base || KPack % k_base != 0)) {
      llvm_unreachable(
          "Tuning parameter selection guarantees kPack is multiple of k_base,"
          "this should never happen");
    }

    // const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
    // FloatA a[KPerThread * mRepeats];
    // FloatB b[KPerThread * nRepeats];
    // constexpr index_t KRepeats = KPack / mfma_type.k_base;
    // auto pa = reinterpret_cast<const data_type*>(&a);
    // auto pb = reinterpret_cast<const data_type*>(&b);
    // constexpr index_t AStride = KPerThread * KRepeats;
    // constexpr index_t BStride = KPerThread * KRepeats;

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

    auto MConstantOp = b.create<ConstantIndexOp>(loc, M);
    auto NConstantOp = b.create<ConstantIndexOp>(loc, N);
    auto KConstantOp = b.create<ConstantIndexOp>(loc, K);

    auto mPerMfmaGroupConstantOp =
        b.create<ConstantIndexOp>(loc, mPerMfmaGroup);
    auto nPerMfmaGroupConstantOp =
        b.create<ConstantIndexOp>(loc, nPerMfmaGroup);

    Value bufferA = adaptor.getBufferA();
    Value bufferB = adaptor.getBufferB();
    auto bufferAType = adaptor.getBufferA().getType().cast<MemRefType>();
    auto bufferBType = adaptor.getBufferB().getType().cast<MemRefType>();
    Type bufferAElementType = bufferAType.getElementType();
    Type bufferBElementType = bufferBType.getElementType();

    int64_t KPerThread = IsKReduction ? K / inputSpansPerMfmaIn : K;

    if (!IsKReduction) {

      // store bufferA logic.
      // for(index_t m_i = 0; m_i < mRepeats; ++m_i)
      //   for(index_t k_i      = 0; k_i < K; ++k_i)
      //     a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + mPerMfmaGroup *
      //     m_i];
      // Note: p_a_wave need to be offseted by waveOffsetA.

      auto outerLoopM = b.create<AffineForOp>(loc, 0, mRepeats);
      auto olmb = ConversionPatternRewriter::atBlockBegin(outerLoopM.getBody(),
                                                          b.getListener());
      auto olmiv = outerLoopM.getInductionVar();
      auto mOffset = olmb.create<AddIOp>(
          loc, aBase, olmb.create<MulIOp>(loc, mPerMfmaGroupConstantOp, olmiv));
      auto kOffsetA = olmb.create<MulIOp>(loc, olmiv, KConstantOp);

      auto innerLoopMK = olmb.create<AffineForOp>(loc, 0, KPerThread);
      auto ilmkb = ConversionPatternRewriter::atBlockBegin(
          innerLoopMK.getBody(), olmb.getListener());
      auto ilmkiv = innerLoopMK.getInductionVar();

      Value sourceOffsetA = ilmkb.create<AddIOp>(
          loc,
          ilmkb.create<AddIOp>(
              loc, ilmkb.create<MulIOp>(loc, ilmkiv, MConstantOp), laneId),
          mOffset);

      if (KPack > 1)
        sourceOffsetA = ilmkb.create<MulIOp>(
            loc, sourceOffsetA, ilmkb.create<ConstantIndexOp>(loc, KPack));

      auto destOffsetA = ilmkb.create<AddIOp>(loc, ilmkiv, kOffsetA);

      Value valueA = ilmkb.create<InBoundsLoadOp>(
          loc, bufferAElementType, op.getMatrixA(), sourceOffsetA);
      ilmkb.create<memref::StoreOp>(loc, valueA, bufferA,
                                    ValueRange{destOffsetA});

      // store bufferB logic.
      // for(index_t n_i = 0; n_i < nRepeats; ++n_i)
      //   for(index_t k_i      = 0; k_i < KPerThread; ++k_i)
      //     b[k_i + n_i * KPerThread] = p_b_wave[k_i * N + laneId +
      //     nPerMfmaGroup
      //     * n_i];
      // Note: p_b_wave need to be offseted by waveOffsetB.

      auto outerLoopN = b.create<AffineForOp>(loc, 0, nRepeats);
      auto olnb = ConversionPatternRewriter::atBlockBegin(outerLoopN.getBody(),
                                                          b.getListener());
      auto olniv = outerLoopN.getInductionVar();
      auto nOffset = olnb.create<AddIOp>(
          loc, bBase, olnb.create<MulIOp>(loc, nPerMfmaGroupConstantOp, olniv));
      auto kOffsetB = olnb.create<MulIOp>(loc, olniv, KConstantOp);

      auto innerLoopNK = olnb.create<AffineForOp>(loc, 0, KPerThread);
      auto ilnkb = ConversionPatternRewriter::atBlockBegin(
          innerLoopNK.getBody(), olnb.getListener());
      auto ilnkiv = innerLoopNK.getInductionVar();

      Value sourceOffsetB = ilnkb.create<AddIOp>(
          loc,
          ilnkb.create<AddIOp>(
              loc, ilnkb.create<MulIOp>(loc, ilnkiv, NConstantOp), laneId),
          nOffset);

      if (KPack > 1)
        sourceOffsetB = ilnkb.create<MulIOp>(
            loc, sourceOffsetB, ilnkb.create<ConstantIndexOp>(loc, KPack));

      auto destOffsetB = ilnkb.create<AddIOp>(loc, ilnkiv, kOffsetB);

      Value valueB = ilnkb.create<InBoundsLoadOp>(
          loc, bufferBElementType, op.getMatrixB(), sourceOffsetB);
      ilnkb.create<memref::StoreOp>(loc, valueB, bufferB,
                                    ValueRange{destOffsetB});
    } else {
      // const index_t blk_id = laneId / mfma_type.num_threads_blk;
      // const index_t blk_td = laneId % mfma_type.num_threads_blk;
      auto inputSpanLenConstantOp =
          b.create<ConstantIndexOp>(loc, inputSpanLen);
      auto blk_id = b.create<DivUIOp>(loc, laneId, inputSpanLenConstantOp);
      auto blk_td = b.create<RemUIOp>(loc, laneId, inputSpanLenConstantOp);

      Value kBaseA = b.create<AddIOp>(loc, aBase, blk_td);
      Value kBaseB = b.create<AddIOp>(loc, bBase, blk_td);

      // for(index_t k_i = 0; k_i < KPerThread; k_i += mfma_type.num_input_blks)
      // {
      //     a[k_i] = p_a_wave[(k_i * num_input_blks + blk_id) * M + blk_td];
      //     b[k_i] = p_b_wave[(k_i * num_input_blks + blk_id) * N + blk_td];
      // }
      // p_a_wave need to be offseted by waveOffsetA.
      // p_b_wave need to be offseted by waveOffsetB.

      auto inputSpansPerMfmaInConstantOp =
          b.create<ConstantIndexOp>(loc, inputSpansPerMfmaIn);

      auto loopKLoad = b.create<AffineForOp>(loc, 0, KPerThread);
      auto lklb = ConversionPatternRewriter::atBlockBegin(loopKLoad.getBody(),
                                                          b.getListener());
      auto lkliv = loopKLoad.getInductionVar();

      Value sourceOffsetA = lklb.create<AddIOp>(
          loc,
          lklb.create<MulIOp>(
              loc,
              lklb.create<AddIOp>(
                  loc,
                  lklb.create<MulIOp>(loc, lkliv,
                                      inputSpansPerMfmaInConstantOp),
                  blk_id),
              MConstantOp),
          kBaseA);

      if (KPack > 1)
        sourceOffsetA = lklb.create<MulIOp>(
            loc, sourceOffsetA, lklb.create<ConstantIndexOp>(loc, KPack));

      Value valueA = lklb.create<InBoundsLoadOp>(
          loc, bufferAElementType, op.getMatrixA(), sourceOffsetA);
      lklb.create<memref::StoreOp>(loc, valueA, bufferA, ValueRange{lkliv});

      Value sourceOffsetB = lklb.create<AddIOp>(
          loc,
          lklb.create<MulIOp>(
              loc,
              lklb.create<AddIOp>(
                  loc,
                  lklb.create<MulIOp>(loc, lkliv,
                                      inputSpansPerMfmaInConstantOp),
                  blk_id),
              NConstantOp),
          kBaseB);

      if (KPack > 1)
        sourceOffsetB = lklb.create<MulIOp>(
            loc, sourceOffsetB, lklb.create<ConstantIndexOp>(loc, KPack));

      Value valueB = lklb.create<InBoundsLoadOp>(
          loc, bufferBElementType, op.getMatrixB(), sourceOffsetB);
      lklb.create<memref::StoreOp>(loc, valueB, bufferB, ValueRange{lkliv});
    }

    int64_t nResultVectors = mfmaGroup.getImms().size();

    Value reshapedARegisters = reshapeBuffer(
        b, loc, adaptor.getBufferA(), {"m", "k"}, {mRepeats, KPerThread});
    Value reshapedBRegisters = reshapeBuffer(
        b, loc, adaptor.getBufferB(), {"n", "k"}, {nRepeats, KPerThread});
    Value reshapedCRegisters =
        reshapeBuffer(b, loc, adaptor.getMatrixC(), {"m", "n", "v"},
                      {mRepeats, nRepeats, nResultVectors});

    b.replaceOpWithNewOp<XdlopsGemmV2Op>(op, reshapedARegisters,
                                         reshapedBRegisters, reshapedCRegisters,
                                         tuningParams);
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

      Value loaded = b.create<BufferLoadOp>(
          loc, typeToLoad, op.getSource(), op.getLeftOobDims(),
          op.getRightOobDims(), op.getSourceCoord(), offsetAttr);
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
      b.create<BufferStoreOp>(loc, loaded, op.getDest(), op.getLeftOobDims(),
                              op.getRightOobDims(), op.getDestCoord(),
                              op.getStoreMethodAttr(), offsetAttr);
      remainingLength -= copyLength;
      offset += copyLength;
    }
    b.eraseOp(op);
    return success();
  }
};

void RockLowerBlockwiseGemmToThreadwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
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
