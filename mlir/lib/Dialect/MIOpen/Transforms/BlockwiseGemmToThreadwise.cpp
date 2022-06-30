//===- BlockwiseGemmToThreadwise - MLIR MIOpen ops lowering passes ---===//
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
// This pass converts miopen.blockwise_* ops to miopen.threadwise_*
// and lowers other higher-level ops like transform and fill in preparation for
// the threadwise lowering
//
//===-----------------------------------------------------===//
#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "miopen-blockwise-to-threadwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;

namespace {
struct MIOpenLowerBlockwiseGemmToThreadwisePass
    : public MIOpenBlockwiseGemmToThreadwisePassBase<
          MIOpenLowerBlockwiseGemmToThreadwisePass> {
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
    auto inputType = op.input().getType().cast<MemRefType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t> lbs(inputShape.size(), 0);
    llvm::SmallVector<int64_t> strides(inputShape.size(), 1);

    buildAffineLoopNest(b, loc, lbs, inputShape, strides,
                        [value = adaptor.value(), input = adaptor.input()](
                            OpBuilder &b, Location loc, ValueRange ivs) {
                          b.create<memref::StoreOp>(loc, value, input, ivs);
                        });

    b.replaceOp(op, {});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmRewritePattern
    : public OpConversionPattern<BlockwiseGemmOp> {
  using OpConversionPattern<BlockwiseGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockwiseGemmOp op,
                                BlockwiseGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Prepare some useful constants.
    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);

    auto blockAType = op.matrixA().getType().cast<MemRefType>();
    auto blockBType = op.matrixB().getType().cast<MemRefType>();
    auto bufferCType = op.matrixC().getType().cast<MemRefType>();

    auto elementType = bufferCType.getElementType();

    int64_t k = blockAType.getShape()[0];
    int64_t m = blockAType.getShape()[1];
    int64_t n = blockBType.getShape()[1];
    int64_t kPack = blockAType.getShape()[2];

    // Non-xdlops path.

    // Obtain critical attributes.
    int64_t mC = bufferCType.getShape()[0];
    int64_t nC = bufferCType.getShape()[1];
    int64_t kPerThread = op.kPerThreadAttr().getInt();
    int64_t mPerThread = op.mPerThreadAttr().getInt();
    int64_t nPerThread = op.nPerThreadAttr().getInt();
    int64_t mRepeatStride = op.mRepeatStrideAttr().getInt();
    int64_t nRepeatStride = op.nRepeatStrideAttr().getInt();
    int64_t mRepeat = mC / mPerThread;
    int64_t nRepeat = nC / nPerThread;

    LLVM_DEBUG(llvm::dbgs() << "M: " << mC << "\n"
                            << "NRepeat: " << mRepeat << "\n"
                            << "MPerThread: " << mPerThread << "\n"
                            << "N: " << nC << "\n"
                            << "NRepeat: " << nRepeat << "\n"
                            << "NPerThread: " << nPerThread << "\n");

    TopDownTMBuilder strideLDSBufferA(b,
                                      {"k", "mRepeat", "mPerThread", "kpack"},
                                      {k, mRepeat, m / mRepeat, kPack}, loc);
    strideLDSBufferA.passThrough("k");
    strideLDSBufferA.embed("m", 1, m, {"mRepeat", "mPerThread"},
                           {mRepeatStride, 1});
    strideLDSBufferA.passThrough({"kpack"}, {2}, {"kpack"});
    TransformMapAttr strideLDSBufferAAttr = strideLDSBufferA.get();

    TopDownTMBuilder strideLDSBufferB(b,
                                      {"k", "nRepeat", "nPerThread", "kpack"},
                                      {k, nRepeat, n / nRepeat, kPack}, loc);
    strideLDSBufferB.passThrough("k");
    strideLDSBufferB.embed("n", 1, n, {"nRepeat", "nPerThread"},
                           {nRepeatStride, 1});
    strideLDSBufferB.passThrough({"kpack"}, {2}, {"kpack"});
    TransformMapAttr strideLDSBufferBAttr = strideLDSBufferB.get();

    Value matrixA, matrixB;
    ArrayAttr transformsA, transformsB;
    std::tie(matrixA, transformsA) = untransform(
        b, adaptor.matrixA(), b.getArrayAttr({strideLDSBufferAAttr}));
    std::tie(matrixB, transformsB) = untransform(
        b, adaptor.matrixB(), b.getArrayAttr({strideLDSBufferBAttr}));

    int64_t threadANumRegisters = kPerThread * mC * kPack;
    int64_t threadBNumRegisters = kPerThread * nC * kPack;

    // Alloc register for thread_a and thread_b.
    auto threadARegisterMemRefType =
        MemRefType::get(threadANumRegisters, elementType, {},
                        gpu::GPUDialect::getPrivateAddressSpace());
    auto threadAAllocOp = b.create<GpuAllocOp>(loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType =
        MemRefType::get(threadBNumRegisters, elementType, {},
                        gpu::GPUDialect::getPrivateAddressSpace());
    auto threadBAllocOp = b.create<GpuAllocOp>(loc, threadBRegisterMemRefType);

    // Define views of register tiles for copies
    BottomUpTMBuilder viewA(b, {"raw"}, {threadANumRegisters}, loc);
    viewA.unmerge({"k", "mRepeat", "mPerThread", "kpack"}, {0, 1, 2, 3}, "raw",
                  {kPerThread, mRepeat, mPerThread, kPack});
    TransformMapAttr threadACopyViewAttr = viewA.get();

    BottomUpTMBuilder viewB(b, {"raw"}, {threadBNumRegisters}, loc);
    viewB.unmerge({"k", "nRepeat", "nPerThread", "kpack"}, {0, 1, 2, 3}, "raw",
                  {kPerThread, nRepeat, nPerThread, kPack});
    TransformMapAttr threadBCopyViewAttr = viewB.get();

    // Main loop.
    LLVM_DEBUG(llvm::dbgs() << "Outer loop:\n "
                            << "k =  " << k << "\n"
                            << " kPerThread = " << kPerThread << "\n");
    auto loopOp = b.replaceOpWithNewOp<AffineForOp>(op, 0, k, kPerThread);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loopOp.getBody());
    Value kOffset = loopOp.getInductionVar();

    SmallVector<Value, 5> registerStartCoords(4, zeroConstantOp);
    SmallVector<Value, 5> ldsBufferAStartCoords = {
        kOffset, zeroConstantOp, op.threadOffsetA(), zeroConstantOp};
    auto copyALoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferAStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsA, b.getArrayAttr(threadACopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, mRepeat, mPerThread, kPack},
        /*strides=*/llvm::None, /*forceUnroll=*/true, /*indexDiffs=*/true);
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
        kOffset, zeroConstantOp, op.threadOffsetB(), zeroConstantOp};
    auto copyBLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferBStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsB, b.getArrayAttr(threadBCopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, nRepeat, nPerThread, kPack},
        /*strides=*/llvm::None, /*forceUnroll=*/true, /*indexDiffs=*/true);
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
                               op.matrixC());

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

    int64_t MPerWave =
        op->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();

    // Original C++ logic.
    // static constexpr index_t MRepeats = (GemmMPerWave > 64) ? (GemmMPerWave /
    // 64) : 1; static constexpr index_t NRepeats = (GemmNPerWave > 64) ?
    // (GemmNPerWave / 64) : 1; static constexpr index_t MPerXdlops =
    // (GemmMPerWave > 64) ? 64 : GemmMPerWave; static constexpr index_t
    // NPerXdlops = (GemmNPerWave > 64) ? 64 : GemmNPerWave;

    int64_t MRepeats = (MPerWave > 64) ? (MPerWave / 64) : 1;
    int64_t NRepeats = (NPerWave > 64) ? (NPerWave / 64) : 1;
    int64_t MPerXdlops = (MPerWave > 64) ? 64 : MPerWave;
    int64_t NPerXdlops = (NPerWave > 64) ? 64 : NPerWave;

    if (MRepeats == 1 && NRepeats == 1) {
      SmallVector<Type, 2> resultTypes;
      for (auto result : op.vectorDs()) {
        resultTypes.push_back(result.getType());
      }

      auto xdlopsGemmV2Op = b.replaceOpWithNewOp<XdlopsGemmV2Op>(
          op, resultTypes, adaptor.matrixA(), adaptor.matrixB(),
          op.ldsBufferOffsetA(), op.ldsBufferOffsetB(), adaptor.waveOffsetA(),
          adaptor.waveOffsetB(), adaptor.bufferA(), adaptor.bufferB(),
          adaptor.vectorCs());

      xdlopsGemmV2Op->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op->setAttr("k", op->getAttr("k"));
      xdlopsGemmV2Op->setAttr("m_per_wave", op->getAttr("m_per_wave"));
      xdlopsGemmV2Op->setAttr("n_per_wave", op->getAttr("n_per_wave"));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op->setAttr("kpack", op->getAttr("kpack"));
    } else if (MRepeats == 2 && NRepeats == 1) {
      // Original C++ logic.
      // p_c_thread.s.x.l = XdlopsGemm.template Run<M, N, K>(p_a_block,
      // p_b_block, p_c_thread.s.x.l); p_c_thread.s.y.l = XdlopsGemm.template
      // Run<M, N, K>(p_a_block + MPerXdlops, p_b_block, p_c_thread.s.y.l);

      SmallVector<Type, 2> resultTypes0;
      resultTypes0.push_back(op.vectorDs()[0].getType());
      resultTypes0.push_back(op.vectorDs()[1].getType());

      auto xdlopsGemmV2Op0 = b.create<XdlopsGemmV2Op>(
          loc, resultTypes0, adaptor.matrixA(), adaptor.matrixB(),
          op.ldsBufferOffsetA(), op.ldsBufferOffsetB(), adaptor.waveOffsetA(),
          adaptor.waveOffsetB(), adaptor.bufferA(), adaptor.bufferB(),
          adaptor.vectorCs().take_front(2));

      xdlopsGemmV2Op0->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op0->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op0->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op0->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op0->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op0->setAttr("kpack", op->getAttr("kpack"));

      SmallVector<Type, 2> resultTypes1;
      resultTypes1.push_back(op.vectorDs()[2].getType());
      resultTypes1.push_back(op.vectorDs()[3].getType());

      auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
      auto xdlopsGemmV2Op1 = b.create<XdlopsGemmV2Op>(
          loc, resultTypes1, adaptor.matrixA(), adaptor.matrixB(),
          op.ldsBufferOffsetA(), op.ldsBufferOffsetB(),
          b.create<AddIOp>(loc, adaptor.waveOffsetA(), MPerXdlopsConstantOp),
          adaptor.waveOffsetB(), adaptor.bufferA(), adaptor.bufferB(),
          adaptor.vectorCs().drop_front(2));

      xdlopsGemmV2Op1->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op1->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op1->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op1->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op1->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op1->setAttr("kpack", op->getAttr("kpack"));

      b.replaceOp(op, ValueRange{xdlopsGemmV2Op0.vectorDs()[0],
                                 xdlopsGemmV2Op0.vectorDs()[1],
                                 xdlopsGemmV2Op1.vectorDs()[0],
                                 xdlopsGemmV2Op1.vectorDs()[1]});
    } else if (MRepeats == 1 && NRepeats == 2) {
      // Original C++ logic.
      // p_c_thread.s.x.l = XdlopsGemm.template Run<M, N, K>(p_a_block,
      // p_b_block, p_c_thread.s.x.l); p_c_thread.s.y.l = XdlopsGemm.template
      // Run<M, N, K>(p_a_block, p_b_block + NPerXdlops, p_c_thread.s.y.l);

      SmallVector<Type, 2> resultTypes0;
      resultTypes0.push_back(op.vectorDs()[0].getType());
      resultTypes0.push_back(op.vectorDs()[1].getType());

      auto xdlopsGemmV2Op0 = b.create<XdlopsGemmV2Op>(
          loc, resultTypes0, adaptor.matrixA(), adaptor.matrixB(),
          op.ldsBufferOffsetA(), op.ldsBufferOffsetB(), adaptor.waveOffsetA(),
          adaptor.waveOffsetB(), adaptor.bufferA(), adaptor.bufferB(),
          adaptor.vectorCs().take_front(2));

      xdlopsGemmV2Op0->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op0->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op0->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op0->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op0->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op0->setAttr("kpack", op->getAttr("kpack"));

      SmallVector<Type, 2> resultTypes1;
      resultTypes1.push_back(op.vectorDs()[2].getType());
      resultTypes1.push_back(op.vectorDs()[3].getType());

      auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
      auto xdlopsGemmV2Op1 = b.create<XdlopsGemmV2Op>(
          loc, resultTypes1, adaptor.matrixA(), adaptor.matrixB(),
          op.ldsBufferOffsetA(), op.ldsBufferOffsetB(), adaptor.waveOffsetA(),
          b.create<AddIOp>(loc, adaptor.waveOffsetB(), NPerXdlopsConstantOp),
          adaptor.bufferA(), adaptor.bufferB(),
          adaptor.vectorCs().drop_front(2));

      xdlopsGemmV2Op1->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op1->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op1->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op1->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op1->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op1->setAttr("kpack", op->getAttr("kpack"));

      b.replaceOp(op, ValueRange{xdlopsGemmV2Op0.vectorDs()[0],
                                 xdlopsGemmV2Op0.vectorDs()[1],
                                 xdlopsGemmV2Op1.vectorDs()[0],
                                 xdlopsGemmV2Op1.vectorDs()[1]});
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ThreadwiseCopyV2 lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseCopyV2RewritePattern
    : public OpRewritePattern<ThreadwiseCopyV2Op> {
  using OpRewritePattern<ThreadwiseCopyV2Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(ThreadwiseCopyV2Op op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value source = op.source();
    auto sourceType = source.getType().cast<MemRefType>();
    Value sourceCoord = op.sourceCoord();

    int64_t copyLength = op.length().getSExtValue();
    Type typeToLoad = sourceType.getElementType();
    if (copyLength > 1)
      typeToLoad = VectorType::get({copyLength}, typeToLoad);
    Type typeToStore = op.dest().getType().cast<MemRefType>().getElementType();
    if (copyLength > 1)
      typeToStore = VectorType::get({copyLength}, typeToStore);

    Value loaded =
        b.create<InBoundsLoadOp>(loc, typeToLoad, source, sourceCoord);
    b.replaceOpWithNewOp<BufferStoreOp>(op, loaded, op.dest(), op.leftOobDims(),
                                        op.rightOobDims(), op.destCoord(),
                                        op.storeMethodAttr());
    return success();
  }
};

void MIOpenLowerBlockwiseGemmToThreadwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<FillOp, BlockwiseGemmOp, BlockwiseGemmV2Op,
                      ThreadwiseCopyV2Op>();
  target.addLegalDialect<arith::ArithmeticDialect, miopen::MIOpenDialect,
                         AffineDialect, memref::MemRefDialect,
                         vector::VectorDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, BlockwiseGemmRewritePattern,
               BlockwiseGemmV2RewritePattern, ThreadwiseCopyV2RewritePattern>(
      ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass>
mlir::miopen::createMIOpenBlockwiseGemmToThreadwisePass() {
  return std::make_unique<MIOpenLowerBlockwiseGemmToThreadwisePass>();
}
