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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "miopen-blockwise-to-threadwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;

namespace {
struct LowerMIOpenOpsStep3Pass
    : public MIOpenOpsStep3PassBase<LowerMIOpenOpsStep3Pass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Fill lowering.
//===----------------------------------------------------------------------===//

struct FillRewritePattern : public OpRewritePattern<FillOp> {
  using OpRewritePattern<FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FillOp op, PatternRewriter &b) const override {
    auto loc = op.getLoc();
    auto inputType = op.input().getType().cast<MemRefType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t> lbs(inputShape.size(), 0);
    llvm::SmallVector<int64_t> strides(inputShape.size(), 1);

    buildAffineLoopNest(b, loc, lbs, inputShape, strides,
                        [value = op.value(), input = op.input()](
                            OpBuilder &b, Location loc, ValueRange ivs) {
                          b.create<memref::StoreOp>(loc, value, input, ivs);
                        });

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmRewritePattern : public OpRewritePattern<BlockwiseGemmOp> {
  using OpRewritePattern<BlockwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseGemmOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Prepare some useful constants.
    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);

    auto blockAType = op.matrixA().getType().cast<MemRefType>();
    auto blockBType = op.matrixB().getType().cast<MemRefType>();

    auto elementType =
        op.matrixC().getType().cast<MemRefType>().getElementType();

    // Obtain critical matrix dimensions.
    int64_t g = blockAType.getShape()[0];
    if (g != 1) {
      // TODO(kdrewnia): Remove this once blockwise_gemm is transitioned to 1D
      // buffers
      return op.emitOpError(
          "Firsct (group) dimension of matrix must be 1 by blockwise gemm\n");
    }
    int64_t k = blockAType.getShape()[1];
    int64_t m = blockAType.getShape()[2];
    int64_t n = blockBType.getShape()[2];
    int64_t kPack = blockAType.getRank() > 3 ? blockAType.getShape()[3] : 1;

    // Non-xdlops path.

    // Obtain critical attributes.
    int64_t mC = op.mCAttr().getInt();
    int64_t nC = op.nCAttr().getInt();
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

    TopDownTMBuilder strideLDSBufferA(
        b, {"g", "k", "mRepeat", "mPerThread", "kpack"},
        {g, k, mRepeat, m / mRepeat, kPack}, loc);
    strideLDSBufferA.passThrough({"g", "k"});
    strideLDSBufferA.embed("m", 2, m, {"mRepeat", "mPerThread"},
                           {mRepeatStride, 1});
    if (kPack > 1)
      strideLDSBufferA.passThrough({"kpack"}, {3}, {"kpack"});
    else
      strideLDSBufferA.ignore("kpack");
    TransformMapAttr strideLDSBufferAAttr = strideLDSBufferA.get();

    TopDownTMBuilder strideLDSBufferB(
        b, {"g", "k", "nRepeat", "nPerThread", "kpack"},
        {g, k, nRepeat, n / nRepeat, kPack}, loc);
    strideLDSBufferB.passThrough({"g", "k"});
    strideLDSBufferB.embed("n", 2, n, {"nRepeat", "nPerThread"},
                           {nRepeatStride, 1});
    if (kPack > 1)
      strideLDSBufferB.passThrough({"kpack"}, {3}, {"kpack"});
    else
      strideLDSBufferB.ignore("kpack");
    TransformMapAttr strideLDSBufferBAttr = strideLDSBufferB.get();

    Value matrixA, matrixB;
    ArrayAttr transformsA, transformsB;
    std::tie(matrixA, transformsA) =
        untransform(b, op.matrixA(), b.getArrayAttr({strideLDSBufferAAttr}));
    std::tie(matrixB, transformsB) =
        untransform(b, op.matrixB(), b.getArrayAttr({strideLDSBufferBAttr}));

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
    // Note: "g" is length 1 and only included here as a temporary compatibility
    // measure until gridwise_gemm is refactored
    BottomUpTMBuilder viewA(b, {"raw"}, {threadANumRegisters}, loc);
    viewA.unmerge({"g", "k", "mRepeat", "mPerThread", "kpack"}, {0, 1, 2, 3, 4},
                  "raw", {g, kPerThread, mRepeat, mPerThread, kPack});
    TransformMapAttr threadACopyViewAttr = viewA.get();

    BottomUpTMBuilder viewB(b, {"raw"}, {threadBNumRegisters}, loc);
    viewB.unmerge({"g", "k", "nRepeat", "nPerThread", "kpack"}, {0, 1, 2, 3, 4},
                  "raw", {1, kPerThread, nRepeat, nPerThread, kPack});
    TransformMapAttr threadBCopyViewAttr = viewB.get();

    // Main loop.
    LLVM_DEBUG(llvm::dbgs() << "Outer loop:\n "
                            << "k =  " << k << "\n"
                            << " kPerThread = " << kPerThread << "\n");
    auto loopOp = b.create<AffineForOp>(loc, 0, k, kPerThread);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loopOp.getBody());
    Value kOffset = loopOp.getInductionVar();

    SmallVector<Value, 5> registerStartCoords(5, zeroConstantOp);
    SmallVector<Value, 5> ldsBufferAStartCoords = {
        zeroConstantOp, kOffset, zeroConstantOp, op.threadOffsetA(),
        zeroConstantOp};
    auto copyALoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferAStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsA, b.getArrayAttr(threadACopyViewAttr)},
        ArrayRef<int64_t>{g, kPerThread, mRepeat, mPerThread, kPack},
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
        zeroConstantOp, kOffset, zeroConstantOp, op.threadOffsetB(),
        zeroConstantOp};
    auto copyBLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferBStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsB, b.getArrayAttr(threadBCopyViewAttr)},
        ArrayRef<int64_t>{g, kPerThread, nRepeat, nPerThread, kPack},
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

    // Actually do the gemm - this goes inside the look over kOffset
    b.create<ThreadwiseGemmOp>(loc, threadAAllocOp, threadBAllocOp,
                               op.matrixC(), op.kPerThreadAttr(), op.mCAttr(),
                               op.nCAttr(), b.getIndexAttr(kPack));

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemmV2 lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmV2RewritePattern
    : public OpRewritePattern<BlockwiseGemmV2Op> {
  using OpRewritePattern<BlockwiseGemmV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseGemmV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

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

      auto xdlopsGemmV2Op = b.create<XdlopsGemmV2Op>(
          loc, resultTypes, op.matrixA(), op.matrixB(), op.ldsBufferOffsetA(),
          op.ldsBufferOffsetB(), op.waveOffsetA(), op.waveOffsetB(),
          op.bufferA(), op.bufferB(), op.vectorCs());

      xdlopsGemmV2Op->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op->setAttr("k", op->getAttr("k"));
      xdlopsGemmV2Op->setAttr("m_per_wave", op->getAttr("m_per_wave"));
      xdlopsGemmV2Op->setAttr("n_per_wave", op->getAttr("n_per_wave"));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op->setAttr("kpack", op->getAttr("kpack"));

      op.replaceAllUsesWith(xdlopsGemmV2Op.vectorDs());
      op.erase();
    } else if (MRepeats == 2 && NRepeats == 1) {
      // Original C++ logic.
      // p_c_thread.s.x.l = XdlopsGemm.template Run<M, N, K>(p_a_block,
      // p_b_block, p_c_thread.s.x.l); p_c_thread.s.y.l = XdlopsGemm.template
      // Run<M, N, K>(p_a_block + MPerXdlops, p_b_block, p_c_thread.s.y.l);

      SmallVector<Type, 2> resultTypes0;
      resultTypes0.push_back(op.vectorDs()[0].getType());
      resultTypes0.push_back(op.vectorDs()[1].getType());

      auto xdlopsGemmV2Op0 = b.create<XdlopsGemmV2Op>(
          loc, resultTypes0, op.matrixA(), op.matrixB(), op.ldsBufferOffsetA(),
          op.ldsBufferOffsetB(), op.waveOffsetA(), op.waveOffsetB(),
          op.bufferA(), op.bufferB(),
          ValueRange{op.vectorCs()[0], op.vectorCs()[1]});

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
          loc, resultTypes1, op.matrixA(), op.matrixB(), op.ldsBufferOffsetA(),
          op.ldsBufferOffsetB(),
          b.create<AddIOp>(loc, op.waveOffsetA(), MPerXdlopsConstantOp),
          op.waveOffsetB(), op.bufferA(), op.bufferB(),
          ValueRange{op.vectorCs()[2], op.vectorCs()[3]});

      xdlopsGemmV2Op1->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op1->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op1->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op1->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op1->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op1->setAttr("kpack", op->getAttr("kpack"));

      op.replaceAllUsesWith(ValueRange{
          xdlopsGemmV2Op0.vectorDs()[0], xdlopsGemmV2Op0.vectorDs()[1],
          xdlopsGemmV2Op1.vectorDs()[0], xdlopsGemmV2Op1.vectorDs()[1]});
      op.erase();
    } else if (MRepeats == 1 && NRepeats == 2) {
      // Original C++ logic.
      // p_c_thread.s.x.l = XdlopsGemm.template Run<M, N, K>(p_a_block,
      // p_b_block, p_c_thread.s.x.l); p_c_thread.s.y.l = XdlopsGemm.template
      // Run<M, N, K>(p_a_block, p_b_block + NPerXdlops, p_c_thread.s.y.l);

      SmallVector<Type, 2> resultTypes0;
      resultTypes0.push_back(op.vectorDs()[0].getType());
      resultTypes0.push_back(op.vectorDs()[1].getType());

      auto xdlopsGemmV2Op0 = b.create<XdlopsGemmV2Op>(
          loc, resultTypes0, op.matrixA(), op.matrixB(), op.ldsBufferOffsetA(),
          op.ldsBufferOffsetB(), op.waveOffsetA(), op.waveOffsetB(),
          op.bufferA(), op.bufferB(),
          ValueRange{op.vectorCs()[0], op.vectorCs()[1]});

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
          loc, resultTypes1, op.matrixA(), op.matrixB(), op.ldsBufferOffsetA(),
          op.ldsBufferOffsetB(), op.waveOffsetA(),
          b.create<AddIOp>(loc, op.waveOffsetB(), NPerXdlopsConstantOp),
          op.bufferA(), op.bufferB(),
          ValueRange{op.vectorCs()[2], op.vectorCs()[3]});

      xdlopsGemmV2Op1->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op1->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op1->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op1->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op1->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op1->setAttr("kpack", op->getAttr("kpack"));

      op.replaceAllUsesWith(ValueRange{
          xdlopsGemmV2Op0.vectorDs()[0], xdlopsGemmV2Op0.vectorDs()[1],
          xdlopsGemmV2Op1.vectorDs()[0], xdlopsGemmV2Op1.vectorDs()[1]});
      op.erase();
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
    b.create<BufferStoreOp>(loc, loaded, op.dest(), op.leftOobDims(),
                            op.rightOobDims(), op.destCoord(),
                            op.storeMethodAttr());
    b.eraseOp(op);
    return success();
  }
};

void LowerMIOpenOpsStep3Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, BlockwiseGemmRewritePattern,
               BlockwiseGemmV2RewritePattern, ThreadwiseCopyV2RewritePattern>(
      ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep3Pass() {
  return std::make_unique<LowerMIOpenOpsStep3Pass>();
}
