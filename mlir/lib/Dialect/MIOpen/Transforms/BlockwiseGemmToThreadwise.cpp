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
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

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
    auto inputShape = inputType.getShape();

    AffineForOp currentLoop;
    OpBuilder currentScope = b;
    std::vector<mlir::Value> range;

    for (unsigned i = 0; i < inputShape.size(); ++i) {
      // Rank 1 loop.
      currentLoop = currentScope.create<AffineForOp>(loc, 0, inputShape[i]);

      // collect current loop induction var for store indexes
      range.push_back(currentLoop.getInductionVar());

      // inside loop.
      currentScope = OpBuilder::atBlockTerminator(currentLoop.getBody());
    }

    // Store fill value
    currentScope.create<memref::StoreOp>(loc, op.value(), op.input(), range);

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Transform lowering.
//
// Gathers a chain of transformations and puts them into the appropriate index
// of the transforms attribute of the user of that chain.
//===----------------------------------------------------------------------===//

struct TransformRewritePattern : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransformOp op,
                                PatternRewriter &b) const override {
    // To cut down on the number of intermediate arrays we pull in,
    // we'll deal with entire chains of transform ops at once

    TransformOp lastTransform = op;
    TransformOp firstTransform = op;

    while (auto pred = dyn_cast_or_null<TransformOp>(
               firstTransform.input().getDefiningOp())) {
      firstTransform = pred;
    }

    bool userIsOneTransform = false;
    do {
      userIsOneTransform = false;
      if (lastTransform.output().hasOneUse()) {
        if (auto succ = dyn_cast<TransformOp>(
                *(lastTransform.output().getUsers().begin()))) {
          userIsOneTransform = true;
          lastTransform = succ;
        }
      }
    } while (userIsOneTransform);

    // Each successive transform op creates an upper view from a lower view
    // so the transforms must be composed with the last (uppermost) transform
    // at the front.
    SmallVector<Attribute, 5> transforms;
    SmallVector<TransformOp, 5> transformOpStack;
    {
      Operation *beforeFirstTransform = firstTransform.input().getDefiningOp();
      Operation *currentOp = lastTransform.getOperation();
      while (currentOp != beforeFirstTransform) {
        auto currentTransform = cast<TransformOp>(currentOp);
        transformOpStack.push_back(currentTransform);
        // verify() would've failed if we couldn't cast<> these
        for (Attribute attr : currentTransform.transforms()) {
          TransformMapAttr ta = attr.cast<TransformMapAttr>();
          transforms.push_back(ta);
        }
        currentOp = currentTransform.input().getDefiningOp();
      }
    }

    // Check module-level invariants before applying the rewrite
    for (OpOperand &use : lastTransform.output().getUses()) {
      uint32_t argNum = use.getOperandNumber();
      Operation *useOp = use.getOwner();
      if (auto useTransforms =
              useOp->getAttr("transforms").dyn_cast_or_null<ArrayAttr>()) {
        if (useTransforms.size() <= argNum && !isa<TransformOp>(useOp)) {
          useOp->emitOpError("The transforms attribute does not have an entry "
                             "for each transformed argument");
        }
      } else {
        return useOp->emitOpError(
            "Operation taking a miopen.transform()'ed argument does not have a "
            "transforms attribute in which to place the transforms");
      }
    }

    // And now actually do the amendments
    for (OpOperand &use : lastTransform.output().getUses()) {
      uint32_t argNum = use.getOperandNumber();
      Operation *useOp = use.getOwner();
      auto argTransforms = useOp->getAttrOfType<ArrayAttr>("transforms");

      // Edge case: A transformed value is transformed in multiple ways
      if (auto useTransform = dyn_cast<TransformOp>(useOp)) {
        // The new set of transformations is the composition we're removing
        // going below/after the transforms this operation defines
        llvm::SmallVector<Attribute, 5> newTransforms;
        std::copy(argTransforms.begin(), argTransforms.end(),
                  std::back_inserter(newTransforms));
        newTransforms.append(transforms);
        useTransform->setAttr("transforms", b.getArrayAttr(newTransforms));
      } else {
        ArrayAttr thisArgTransforms = argTransforms[argNum].cast<ArrayAttr>();
        ArrayAttr newTransformsAttr;
        if (thisArgTransforms.size() == 0) {
          newTransformsAttr = b.getArrayAttr(transforms);
        } else {
          // The set of transformations is those being applied to the output of
          // the output transformation followed by those this chain of
          // transformations is performing.
          llvm::SmallVector<Attribute, 5> newTransforms;
          std::copy(thisArgTransforms.begin(), thisArgTransforms.end(),
                    std::back_inserter(newTransforms));
          newTransforms.append(transforms);
          newTransformsAttr = b.getArrayAttr(newTransforms);
        }
        useOp->setAttr("transforms", argTransforms.replaceImmediateSubAttribute(
                                         {{argNum, newTransformsAttr}}));
      }
    }

    Value replacement = firstTransform.input();
    lastTransform.output().replaceAllUsesWith(replacement);
    b.replaceOp(lastTransform, {replacement});

    // The stack of transformations will now remove itself by canonicalization
    return success();
  }
};

// XXX: Figure out a way to do away with isThreadwiseLoad parameter.
template <typename T, typename U>
static void affixThreadwiseCopyAttributes(T &top, U &bop, OpBuilder &b,
                                          bool isThreadwiseLoad) {
  if (isThreadwiseLoad) {
    top->setAttr("vector_read_write_dim",
                 bop->getAttr("source_vector_read_dim"));
    top->setAttr("source_data_per_read", bop->getAttr("source_data_per_read"));
    top->setAttr("dest_data_per_write", bop->getAttr("dest_data_per_write"));
  } else {
    top->setAttr("vector_read_write_dim",
                 bop->getAttr("dest_vector_write_dim"));
    top->setAttr("source_data_per_read", bop->getAttr("source_data_per_read"));
    top->setAttr("dest_data_per_write", bop->getAttr("dest_data_per_write"));
  }
}

// XXX: figure out a better way to get rid of isMatrixA parameter.
void affixThreadwiseCopyAttributes(ThreadwiseCopyOp top, BlockwiseGemmOp bop,
                                   OpBuilder &b, bool isMatrixA) {
  if (isMatrixA) {
    top->setAttr("n_slice_row", bop->getAttr("k_per_thread"));
    top->setAttr("n_slice_col", bop->getAttr("m_per_thread"));
    // XXX: TBD review how vector load/store attributes are passed down.
    // top->setAttr("data_per_access", bop->getAttr("m_per_thread"));
    top->setAttr("data_per_access", b.getI32IntegerAttr(1));
  } else {
    top->setAttr("n_slice_row", bop->getAttr("k_per_thread"));
    top->setAttr("n_slice_col", bop->getAttr("n_per_thread"));
    // XXX: TBD review how vector load/store attributes are passed down.
    // top->setAttr("data_per_access", bop->getAttr("n_per_thread"));
    top->setAttr("data_per_access", b.getI32IntegerAttr(1));
  }
}

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

    Value matrixA = op.matrixA();
    Value matrixB = op.matrixB();
    auto transformsA = op.transforms()[0].cast<ArrayAttr>();
    auto transformsB = op.transforms()[1].cast<ArrayAttr>();
    // ThreadwiseGemm doesn't get transforms for A and B explicitly set, so this
    // is a "did miopen.transform get lowered yet" check
    if (transformsA.empty())
      matrixA = untransform(b, matrixA, transformsA);
    if (transformsB.empty())
      matrixB = untransform(b, matrixB, transformsB);

    ArrayAttr emptyArr = b.getArrayAttr({});
    auto noPadding =
        PaddingInfoAttr::get(b.getContext(), 0, 0, 0, BwdPaddingKernelInfo::NA);
    auto noOobDims = b.getArrayAttr({});
    IntegerAttr noGlobals = b.getIndexAttr(-1);
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

    // llvm::errs() << "MPerThread: " << MPerThread << "\n";
    // llvm::errs() << "MPerThreadSubC: " << MPerThreadSubC << "\n";
    // llvm::errs() << "NPerThread: " << NPerThread << "\n";
    // llvm::errs() << "NPerThreadSubC: " << NPerThreadSubC << "\n";

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
        ArrayRef<int64_t>{1, KPerThread, MPerThread},
        /*forceUnroll=*/true, /*indexDiffs=*/false);
    OpBuilder copyABuilder =
        OpBuilder::atBlockTerminator(copyALoop.getBody(), lab.getListener());
    Value aCopy = copyABuilder.create<memref::LoadOp>(
        loc, matrixA, copyALoop.getLowerCoords(0));
    Value aCast = createTypeConversionOp(copyABuilder, loc, aCopy, elementType);
    copyABuilder.create<memref::StoreOp>(loc, aCast, threadAAllocOp,
                                         copyALoop.getLowerCoords(1));

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
        ArrayRef<int64_t>{1, KPerThread, NPerThread},
        /*forceUnroll=*/true, /*indexDiffs=*/false);
    OpBuilder copyBBuilder =
        OpBuilder::atBlockTerminator(copyBLoop.getBody(), lbb.getListener());
    Value bCopy = copyBBuilder.create<memref::LoadOp>(
        loc, matrixB, copyBLoop.getLowerCoords(0));
    Value bCast = createTypeConversionOp(copyBBuilder, loc, bCopy, elementType);
    copyBBuilder.create<memref::StoreOp>(loc, bCast, threadBAllocOp,
                                         copyBLoop.getLowerCoords(1));

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
// BlockwiseLoad lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseLoadRewritePattern : public OpRewritePattern<BlockwiseLoadOp> {
  using OpRewritePattern<BlockwiseLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseLoadOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();
    TypeRange resultTypes = op.result().getTypes();

    // BlockwiseLoad only accepts the following data movement:
    // - 0 (global) -> 5 (register) : load

    // Threadwise copy from global (generic tensor) to register (naive
    // tensor).

    auto threadwiseLoadOp = b.create<ThreadwiseLoadOp>(
        loc, resultTypes, op.source(), op.bounds(), op.transforms(),
        op.paddingInfo(), op.oobDims(), op.sourceCoord());
    affixThreadwiseCopyAttributes(threadwiseLoadOp, op, b,
                                  /*isThreadwiseLoad=*/true);

    op.replaceAllUsesWith(threadwiseLoadOp.getResults());
    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseStore lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseStoreRewritePattern
    : public OpRewritePattern<BlockwiseStoreOp> {
  using OpRewritePattern<BlockwiseStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseStoreOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();
    // BlockwiseLoad only accepts the following data movement:
    // - 5 (register) -> 3 (LDS) : store

    // Threadwise copy from register (naive tensor) to LDS (naive tensor).
    auto threadwiseStoreOp =
        b.create<ThreadwiseStoreOp>(loc, op.dest(), op.bounds(),
                                    op.transforms(), op.data(), op.destCoord());
    affixThreadwiseCopyAttributes(threadwiseStoreOp, op, b,
                                  /*isThreadwiseLoad=*/false);

    op.erase();
    return success();
  }
};

void LowerMIOpenOpsStep3Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, TransformRewritePattern,
               BlockwiseGemmRewritePattern, BlockwiseGemmV2RewritePattern,
               BlockwiseLoadRewritePattern, BlockwiseStoreRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep3Pass() {
  return std::make_unique<LowerMIOpenOpsStep3Pass>();
}
