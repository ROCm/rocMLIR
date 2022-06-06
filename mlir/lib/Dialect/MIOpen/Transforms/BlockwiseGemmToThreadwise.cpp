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
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
    auto loc = op.getLoc();

    // Prepare some useful constants.
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    auto blockAType = op.matrixA().getType().cast<MemRefType>();

    auto elementType =
        op.matrixC().getType().cast<MemRefType>().getElementType();

    // Obtain critical matrix dimensions.
    int64_t K = blockAType.getShape()[1];

    Value matrixA, matrixB;
    ArrayAttr transformsA, transformsB;
    std::tie(matrixA, transformsA) = untransform(b, op.matrixA());
    std::tie(matrixB, transformsB) = untransform(b, op.matrixB());

    ArrayAttr emptyArr = b.getArrayAttr({});
    // Non-xdlops path.

    // Obtain critical attributes.
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t KPerThread =
        op->getAttr("k_per_thread").template cast<IntegerAttr>().getInt();
    int64_t MPerThread =
        op.matrixC().getType().template cast<MemRefType>().getShape()[1];
    int64_t NPerThread =
        op.matrixC().getType().template cast<MemRefType>().getShape()[2];
    int64_t MPerThreadSubC =
        op->getAttr("m_per_thread").template cast<IntegerAttr>().getInt();
    int64_t NPerThreadSubC =
        op->getAttr("n_per_thread").template cast<IntegerAttr>().getInt();

    LLVM_DEBUG(llvm::dbgs() << "MPerThread: " << MPerThread << "\n"
                            << "MPerThreadSubC: " << MPerThreadSubC << "\n"
                            << "NPerThread: " << NPerThread << "\n"
                            << "NPerThreadSubC: " << NPerThreadSubC << "\n");

    auto MPerThreadSubCConstantOp =
        b.create<ConstantIndexOp>(loc, MPerThreadSubC);
    auto NPerThreadSubCConstantOp =
        b.create<ConstantIndexOp>(loc, NPerThreadSubC);

    int64_t MLevel0Cluster =
        op->getAttr("m_level0_cluster").template cast<IntegerAttr>().getInt();
    int64_t MLevel1Cluster =
        op->getAttr("m_level1_cluster").template cast<IntegerAttr>().getInt();
    int64_t NLevel0Cluster =
        op->getAttr("n_level0_cluster").template cast<IntegerAttr>().getInt();
    int64_t NLevel1Cluster =
        op->getAttr("n_level1_cluster").template cast<IntegerAttr>().getInt();

    int64_t MPerLevel1Cluster =
        MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
    int64_t NPerLevel1Cluster =
        NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;
    auto MPerLevel1ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, MPerLevel1Cluster);
    auto NPerLevel1ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NPerLevel1Cluster);

    int64_t MRepeat = MPerThread / MPerThreadSubC;
    int64_t NRepeat = NPerThread / NPerThreadSubC;

    // Alloc register for thread_a and thread_b.
    Type threadARegisterMemRefType;
    if (KPack > 1) {
      threadARegisterMemRefType =
          MemRefType::get({1, KPerThread, MPerThread, KPack}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
    } else {
      threadARegisterMemRefType =
          MemRefType::get({1, KPerThread, MPerThread}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
    }
    auto threadAAllocOp = b.create<GpuAllocOp>(loc, threadARegisterMemRefType);

    Type threadBRegisterMemRefType;
    if (KPack > 1) {
      threadBRegisterMemRefType =
          MemRefType::get({1, KPerThread, NPerThread, KPack}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
    } else {
      threadBRegisterMemRefType =
          MemRefType::get({1, KPerThread, NPerThread}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
    }
    auto threadBAllocOp = b.create<GpuAllocOp>(loc, threadBRegisterMemRefType);

    // Main loop.
    auto loopIteration = K / KPerThread;
    auto loopOp = b.create<AffineForOp>(loc, 0, loopIteration);

    // inside the main loop.
    auto lb = OpBuilder::atBlockTerminator(loopOp.getBody(), b.getListener());

    auto iv = loopOp.getInductionVar();

    // read matrix A loop.
    auto loopReadMatrixAIteration = MRepeat;
    auto loopReadMatrixAOp =
        lb.create<AffineForOp>(loc, 0, loopReadMatrixAIteration);

    // inside read matrix A loop.
    auto lab = OpBuilder::atBlockTerminator(loopReadMatrixAOp.getBody(),
                                            lb.getListener());

    auto iva = loopReadMatrixAOp.getInductionVar();

    // Threadwise copy from LDS (naive tensor) to register (generic tensor).

    // Set copy sorce and dest coordinate acoording to original C++ logic:
    SmallVector<Value, 4> matrixAThreadwiseCopySourceCoords;
    if (KPack > 1) {
      matrixAThreadwiseCopySourceCoords = {
          zeroConstantOp, zeroConstantOp, iv,
          lab.create<AddIOp>(
              loc, lab.create<MulIOp>(loc, iva, MPerLevel1ClusterConstantOp),
              op.threadOffsetA())};
    } else {
      matrixAThreadwiseCopySourceCoords = {
          zeroConstantOp, iv,
          lab.create<AddIOp>(
              loc, lab.create<MulIOp>(loc, iva, MPerLevel1ClusterConstantOp),
              op.threadOffsetA())};
    }

    SmallVector<Value, 4> matrixAThreadwiseCopyDestCoords;
    if (KPack > 1) {
      matrixAThreadwiseCopyDestCoords = {
          zeroConstantOp, zeroConstantOp, zeroConstantOp,
          lab.create<MulIOp>(loc, iva, MPerThreadSubCConstantOp)};
    } else {
      matrixAThreadwiseCopyDestCoords = {
          zeroConstantOp, zeroConstantOp,
          lab.create<MulIOp>(loc, iva, MPerThreadSubCConstantOp)};
    }

    auto copyALoop = lab.create<TransformingForOp>(
        loc,
        ArrayRef<ValueRange>{matrixAThreadwiseCopySourceCoords,
                             matrixAThreadwiseCopyDestCoords},
        ArrayRef<Attribute>{transformsA, emptyArr},
        ArrayRef<int64_t>{1, KPerThread, MPerThreadSubC},
        /*strides=*/llvm::None, /*forceUnroll=*/true, /*indexDiffs=*/false);
    OpBuilder copyABuilder =
        OpBuilder::atBlockTerminator(copyALoop.getBody(), lab.getListener());
    Value aCopy = copyABuilder.create<memref::LoadOp>(
        loc, matrixA, copyALoop.getLowerCoords(/*domain=*/0));
    Value aCast = createTypeConversionOp(copyABuilder, loc, aCopy, elementType);
    copyABuilder.create<memref::StoreOp>(
        loc, aCast, threadAAllocOp, copyALoop.getLowerCoords(/*domain=*/1));

    // read matrix B loop.
    auto loopReadMatrixBIteration = NRepeat;
    auto loopReadMatrixBOp =
        lb.create<AffineForOp>(loc, 0, loopReadMatrixBIteration);

    // inside read matrix B loop.
    auto lbb = OpBuilder::atBlockTerminator(loopReadMatrixBOp.getBody(),
                                            lb.getListener());

    auto ivb = loopReadMatrixBOp.getInductionVar();

    // Threadwise copy from LDS (naive tensor) to register (generic tensor).

    // Set copy sorce and dest coordinate acoording to original C++ logic:
    SmallVector<Value, 4> matrixBThreadwiseCopySourceCoords;
    if (KPack > 1) {
      matrixBThreadwiseCopySourceCoords = {
          zeroConstantOp, zeroConstantOp, iv,
          lbb.create<AddIOp>(
              loc, lbb.create<MulIOp>(loc, ivb, NPerLevel1ClusterConstantOp),
              op.threadOffsetB())};
    } else {
      matrixBThreadwiseCopySourceCoords = {
          zeroConstantOp, iv,
          lbb.create<AddIOp>(
              loc, lbb.create<MulIOp>(loc, ivb, NPerLevel1ClusterConstantOp),
              op.threadOffsetB())};
    }

    SmallVector<Value, 4> matrixBThreadwiseCopyDestCoords;
    if (KPack > 1) {
      matrixBThreadwiseCopyDestCoords = {
          zeroConstantOp, zeroConstantOp, zeroConstantOp,
          lbb.create<MulIOp>(loc, ivb, NPerThreadSubCConstantOp)};
    } else {
      matrixBThreadwiseCopyDestCoords = {
          zeroConstantOp, zeroConstantOp,
          lbb.create<MulIOp>(loc, ivb, NPerThreadSubCConstantOp)};
    }

    auto copyBLoop = lbb.create<TransformingForOp>(
        loc,
        ArrayRef<ValueRange>{matrixBThreadwiseCopySourceCoords,
                             matrixBThreadwiseCopyDestCoords},
        ArrayRef<Attribute>{transformsB, emptyArr},
        ArrayRef<int64_t>{1, KPerThread, NPerThreadSubC},
        /*strides=*/llvm::None, /*forceUnroll=*/true, /*indexDiffs=*/false);
    OpBuilder copyBBuilder =
        OpBuilder::atBlockTerminator(copyBLoop.getBody(), lbb.getListener());
    Value bCopy = copyBBuilder.create<memref::LoadOp>(
        loc, matrixB, copyBLoop.getLowerCoords(/*domain=*/0));
    Value bCast = createTypeConversionOp(copyBBuilder, loc, bCopy, elementType);
    copyBBuilder.create<memref::StoreOp>(
        loc, bCast, threadBAllocOp, copyBLoop.getLowerCoords(/*domain=*/1));

    // Actually do the gemm
    lb.create<ThreadwiseGemmOp>(loc, threadAAllocOp, threadBAllocOp,
                                op.matrixC());

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
// ThreadwiseCopy lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseCopyRewritePattern
    : public OpRewritePattern<ThreadwiseCopyOp> {
  using OpRewritePattern<ThreadwiseCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseCopyOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    ArrayAttr srcTransformsOnOp = op.transforms()[0].cast<ArrayAttr>();
    ArrayAttr destTransformsOnOp = op.transforms()[1].cast<ArrayAttr>();
    Value source, dest;
    ArrayAttr srcTransforms, destTransforms;
    std::tie(source, srcTransforms) =
        untransform(b, op.source(), srcTransformsOnOp);
    std::tie(dest, destTransforms) =
        untransform(b, op.dest(), destTransformsOnOp);
    MemRefType sourceType = source.getType().cast<MemRefType>();
    MemRefType destType = dest.getType().cast<MemRefType>();

    bool legacyLoad = op.legacyLoad().getValueOr(false);
    bool legacyStore = op.legacyStore().getValueOr(false);
    bool useIndexDiffs = !(legacyLoad || legacyStore);

    ArrayAttr srcLeftOob, srcRightOob, destLeftOob, destRightOob;
    std::tie(srcLeftOob, srcRightOob) =
        computeOobFromTransforms(b, srcTransforms);
    std::tie(destLeftOob, destRightOob) =
        computeOobFromTransforms(b, destTransforms);

    TransformingForOp loop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{op.sourceCoord(), op.destCoord()},
        ArrayRef<Attribute>{srcTransforms, destTransforms}, op.bounds(),
        /*strides=*/ArrayAttr{}, /*forceUnroll=*/true, useIndexDiffs);
    PatternRewriter::InsertionGuard loopGuard(b);
    b.setInsertionPointToStart(loop.getBody());

    bool loadGlobal = sourceType.getMemorySpaceAsInt() == 0;
    bool storeGlobal = destType.getMemorySpaceAsInt() == 0;

    Value loaded;
    if (loadGlobal)
      loaded = b.create<BufferLoadOp>(loc, sourceType.getElementType(), source,
                                      srcLeftOob, srcRightOob,
                                      loop.getLowerCoords(/*domain=*/0));
    else
      loaded = b.create<memref::LoadOp>(loc, source,
                                        loop.getLowerCoords(/*domain=*/0));
    Value cast =
        createTypeConversionOp(b, loc, loaded, destType.getElementType());
    if (storeGlobal)
      b.create<BufferStoreOp>(loc, cast, dest, destLeftOob, destRightOob,
                              loop.getLowerCoords(/*domain=*/1),
                              /*dataOperation=*/nullptr);
    else
      b.create<memref::StoreOp>(loc, cast, dest,
                                loop.getLowerCoords(/*domain=*/1));

    b.eraseOp(op);
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
               BlockwiseGemmV2RewritePattern, ThreadwiseCopyRewritePattern,
               ThreadwiseCopyV2RewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep3Pass() {
  return std::make_unique<LowerMIOpenOpsStep3Pass>();
}
