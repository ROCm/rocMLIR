//===- LowerMIOpenOps.cpp - MLIR MIOpen ops lowering passes ---------------===//
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
// =============================================================================
//
// These passes convert the MIOpen threadwise ops into constructs from the
// rest of MLIR so that they can be lowered to the GPU and LLVM dialects.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MIOpen/AffineMapHelper.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include <iterator>
#include <numeric>

#define DEBUG_TYPE "miopen-threadwise-gemm-lowering"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;

namespace {
struct MIOpenThreadwiseGemmLoweringPass
    : public MIOpenThreadwiseGemmLoweringPassBase<
          MIOpenThreadwiseGemmLoweringPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// ThreadwiseGemm lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseGemmRewritePattern
    : public OpConversionPattern<ThreadwiseGemmOp> {
  using OpConversionPattern<ThreadwiseGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseGemmOp op,
                                ThreadwiseGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    Value gemmA = adaptor.matrixA();
    Value gemmB = adaptor.matrixB();
    Value gemmC = adaptor.matrixC();
    auto gemmAType = gemmA.getType().cast<MemRefType>();
    Type dataType = gemmAType.getElementType();

    ArrayRef<int64_t> aShape = gemmAType.getShape();
    int64_t k = aShape[0];
    int64_t m = aShape[1];
    int64_t kPack = aShape[2];
    int64_t n = gemmB.getType().cast<MemRefType>().getShape()[1];
    LLVM_DEBUG(llvm::dbgs() << "Threadwise gemm:\n"
                            << "k = " << k << "\n"
                            << "m = " << m << "\n"
                            << "n = " << n << "\n"
                            << "kPack = " << kPack << "\n");
    SmallVector<int64_t> dimensions = {k, m, n, kPack};

    TopDownTMBuilder aView(b, {"k", "m", "n", "kpack"}, dimensions, loc);
    aView.ignore("n");
    aView.passThrough({"k", "m", "kpack"}, {0, 1, 2}, {"k", "m", "kpack"});
    TransformMapAttr aViewAttr = aView.get();

    TopDownTMBuilder bView(b, {"k", "m", "n", "kpack"}, dimensions, loc);
    bView.ignore("m");
    bView.passThrough({"k", "n", "kpack"}, {0, 1, 2}, {"k", "n", "kpack"});
    TransformMapAttr bViewAttr = bView.get();

    TopDownTMBuilder cView(b, {"k", "m", "n", "kpack"}, dimensions, loc);
    cView.ignore("k");
    cView.ignore("kpack");
    cView.passThrough({"m", "n"}, {0, 1}, {"m", "n"});
    TransformMapAttr cViewAttr = cView.get();

    Value zeroConst = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value, 5> startCoords(4, zeroConst);

    ArrayAttr aTransforms, bTransforms, cTransforms;
    Value bufferA, bufferB, bufferC;
    std::tie(bufferA, aTransforms) = untransform(b, gemmA, {aViewAttr});
    std::tie(bufferB, bTransforms) = untransform(b, gemmB, {bViewAttr});
    std::tie(bufferC, cTransforms) = untransform(b, gemmC, {cViewAttr});

    auto gemmLoop = b.replaceOpWithNewOp<TransformingForOp>(
        op, ArrayRef<ValueRange>{startCoords, startCoords, startCoords},
        ArrayRef<Attribute>{aTransforms, bTransforms, cTransforms}, dimensions,
        /*strides=*/llvm::None, /*forceUnroll=*/true, /*useIndexDiffs=*/false);

    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(gemmLoop.getBody());
      Value aVal = b.create<InBoundsLoadOp>(
          loc, dataType, bufferA, gemmLoop.getLowerCoords(/*domain=*/0));
      Value bVal = b.create<InBoundsLoadOp>(
          loc, dataType, bufferB, gemmLoop.getLowerCoords(/*domain=*/1));
      Value mul;
      if (dataType.isa<IntegerType>())
        mul = b.create<MulIOp>(loc, aVal, bVal);
      else if (dataType.isa<FloatType>())
        mul = b.create<MulFOp>(loc, aVal, bVal);
      else
        llvm_unreachable("Validation should make this ints or floats only");

      ValueRange cCoords = gemmLoop.getLowerCoords(/*domain=*/2);
      Value cVal = b.create<InBoundsLoadOp>(loc, dataType, bufferC, cCoords);
      Value add;
      if (dataType.isa<IntegerType>())
        add = b.create<AddIOp>(loc, mul, cVal);
      else if (dataType.isa<FloatType>())
        add = b.create<AddFOp>(loc, mul, cVal);
      else
        llvm_unreachable("Very serously can't happen");

      b.create<InBoundsStoreOp>(loc, add, bufferC, cCoords);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XdlopsGemmV2 lowering.
//===----------------------------------------------------------------------===//
struct XdlopsGemmV2RewritePattern : public OpConversionPattern<XdlopsGemmV2Op> {
  using OpConversionPattern<XdlopsGemmV2Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(XdlopsGemmV2Op op,
                                XdlopsGemmV2OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain critical information.
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t M = op->getAttr("m").template cast<IntegerAttr>().getInt();
    int64_t N = op->getAttr("n").template cast<IntegerAttr>().getInt();
    int64_t K = op->getAttr("k").template cast<IntegerAttr>().getInt();
    int64_t MPerWave =
        op->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();

    int64_t ldsOffsetA = op.ldsBufferOffsetA().getSExtValue();
    int64_t ldsOffsetB = op.ldsBufferOffsetB().getSExtValue();

    assert(ldsOffsetA % KPack == 0 &&
           "LDS buffer segment for A is kpack-aligned");
    assert(ldsOffsetB % KPack == 0 &&
           "LDS buffer segment for B is kpack-aligned");
    auto dataType = adaptor.matrixA()
                        .getType()
                        .template cast<MemRefType>()
                        .getElementType();

    auto MConstantOp = b.create<ConstantIndexOp>(loc, M);
    auto NConstantOp = b.create<ConstantIndexOp>(loc, N);
    auto KConstantOp = b.create<ConstantIndexOp>(loc, K);

    // The address calculations into the LDS buffer assume that the buffer
    // has type vector<KPack x T>. Then, we convert that into an address
    // in a buffer of Ts through a final multiplicaiton by KPack.
    // However, the LDS buffer offset, which was computed when the buffer was
    // allocated, is an offset into a buffer of T. Therefore, to allow it to
    // easily participate in adress calculations (instead of adding it on at the
    // end) we must divide it by KPack here. Fortunately, this offset will be
    // KPack-alligned and so this is safe
    Value aBase =
        b.create<AddIOp>(loc, adaptor.waveOffsetA(),
                         b.create<ConstantIndexOp>(loc, ldsOffsetA / KPack));
    Value bBase =
        b.create<AddIOp>(loc, adaptor.waveOffsetB(),
                         b.create<ConstantIndexOp>(loc, ldsOffsetB / KPack));

    // Logic to do XDLOPS code selection.
    LLVM_DEBUG(llvm::dbgs() << "Invoke XDLOPS code selection logic:\n"
                            << "dataType: " << dataType << "\n"
                            << "MPerWave: " << MPerWave << "\n"
                            << "NPerWave: " << NPerWave << "\n");

    XdlopsCodeSelection xcs =
        XdlopsCodeSelection::get(dataType, MPerWave, NPerWave, b);

    // Extract values from XdlopsCodeSelection.
    amdgpu::MFMAInstr mfmaInstr = xcs.instr;
    LLVM_DEBUG(llvm::dbgs() << "Selected xdlop: "
                            << amdgpu::stringifyMFMAInstr(mfmaInstr) << "\n");
    int64_t MPerXdlops = xcs.MPerXdlops;
    int64_t NPerXdlops = xcs.NPerXdlops;
    int64_t MRepeats = xcs.MRepeats;
    int64_t NRepeats = xcs.NRepeats;
    VectorType vectorType = xcs.vectorType;
    int64_t vectorNumber = xcs.vectorNumber;
    SmallVector<SmallVector<unsigned, 3>, 2> imms = xcs.imms;
    Type argType = xcs.argType;

    int64_t num_threads_blk = xcs.num_threads_blk;
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t k_base = xcs.k_base;

    bool IsKReduction = (num_output_blks == 1) && (num_input_blks > 1);

    if (KPack > 1 && (KPack < k_base || KPack % k_base != 0)) {
      llvm_unreachable(
          "Tuning parameter selection guarantees kPack is multiple of k_base,"
          "this should never happen");
    }

    // const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
    // FloatA a[KPerThread * MRepeats];
    // FloatB b[KPerThread * NRepeats];
    // constexpr index_t KRepeats = KPack / mfma_type.k_base;
    // auto pa = reinterpret_cast<const data_type*>(&a);
    // auto pb = reinterpret_cast<const data_type*>(&b);
    // constexpr index_t AStride = KPerThread * KRepeats;
    // constexpr index_t BStride = KPerThread * KRepeats;

    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());
    constexpr int64_t waveSize = 64;
    auto laneId =
        b.create<RemUIOp>(loc, tid, b.create<ConstantIndexOp>(loc, waveSize));

    int64_t KRepeats = KPack / k_base;
    if (KRepeats == 0)
      KRepeats = 1;
    LLVM_DEBUG(llvm::dbgs()
               << "argVectorType: " << argType << "\n"
               << "k_base: " << k_base << "\n"
               << "KRepeats: " << KRepeats << "\n"
               << "K: " << K << "\n"
               << "bufferA type: " << adaptor.bufferA().getType() << "\n"
               << "bufferB type: " << adaptor.bufferB().getType() << "\n");

    auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
    auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
    auto KBaseConstantOp = b.create<ConstantIndexOp>(loc, k_base);

    Value bufferA = adaptor.bufferA();
    Value bufferB = adaptor.bufferB();
    auto bufferAType = adaptor.bufferA().getType().cast<MemRefType>();
    auto bufferBType = adaptor.bufferB().getType().cast<MemRefType>();
    Type bufferAElementType = bufferAType.getElementType();
    Type bufferBElementType = bufferBType.getElementType();

    int64_t KPerThread = IsKReduction ? K / num_input_blks : K;

    if (!IsKReduction) {

      // store bufferA logic.
      // for(index_t m_i = 0; m_i < MRepeats; ++m_i)
      //   for(index_t k_i      = 0; k_i < K; ++k_i)
      //     a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // Note: p_a_wave need to be offseted by waveOffsetA.

      auto outerLoopM = b.create<AffineForOp>(loc, 0, MRepeats);
      auto olmb = OpBuilder::atBlockBegin(outerLoopM.getBody());
      auto olmiv = outerLoopM.getInductionVar();
      auto mOffset = olmb.create<AddIOp>(
          loc, aBase, olmb.create<MulIOp>(loc, MPerXdlopsConstantOp, olmiv));
      auto kOffsetA = olmb.create<MulIOp>(loc, olmiv, KConstantOp);

      auto innerLoopMK = olmb.create<AffineForOp>(loc, 0, KPerThread);
      auto ilmkb = OpBuilder::atBlockBegin(innerLoopMK.getBody());
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

      Value valueA = ilmkb.create<InBoundsLoadOp>(loc, bufferAElementType,
                                                  op.matrixA(), sourceOffsetA);
      ilmkb.create<memref::StoreOp>(loc, valueA, bufferA,
                                    ValueRange{destOffsetA});

      // store bufferB logic.
      // for(index_t n_i = 0; n_i < NRepeats; ++n_i)
      //   for(index_t k_i      = 0; k_i < KPerThread; ++k_i)
      //     b[k_i + n_i * KPerThread] = p_b_wave[k_i * N + laneId + NPerXdlops
      //     * n_i];
      // Note: p_b_wave need to be offseted by waveOffsetB.

      auto outerLoopN = b.create<AffineForOp>(loc, 0, NRepeats);
      auto olnb = OpBuilder::atBlockBegin(outerLoopN.getBody());
      auto olniv = outerLoopN.getInductionVar();
      auto nOffset = olnb.create<AddIOp>(
          loc, bBase, olnb.create<MulIOp>(loc, NPerXdlopsConstantOp, olniv));
      auto kOffsetB = olnb.create<MulIOp>(loc, olniv, KConstantOp);

      auto innerLoopNK = olnb.create<AffineForOp>(loc, 0, KPerThread);
      auto ilnkb = OpBuilder::atBlockBegin(innerLoopNK.getBody());
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

      Value valueB = ilnkb.create<InBoundsLoadOp>(loc, bufferBElementType,
                                                  op.matrixB(), sourceOffsetB);
      ilnkb.create<memref::StoreOp>(loc, valueB, bufferB,
                                    ValueRange{destOffsetB});
    } else {
      // const index_t blk_id = laneId / mfma_type.num_threads_blk;
      // const index_t blk_td = laneId % mfma_type.num_threads_blk;
      auto NumThreadsBlkConstantOp =
          b.create<ConstantIndexOp>(loc, num_threads_blk);
      auto blk_id = b.create<DivUIOp>(loc, laneId, NumThreadsBlkConstantOp);
      auto blk_td = b.create<RemUIOp>(loc, laneId, NumThreadsBlkConstantOp);

      Value kBaseA = b.create<AddIOp>(loc, aBase, blk_td);
      Value kBaseB = b.create<AddIOp>(loc, bBase, blk_td);

      // for(index_t k_i = 0; k_i < KPerThread; k_i += mfma_type.num_input_blks)
      // {
      //     a[k_i] = p_a_wave[(k_i * num_input_blks + blk_id) * M + blk_td];
      //     b[k_i] = p_b_wave[(k_i * num_input_blks + blk_id) * N + blk_td];
      // }
      // p_a_wave need to be offseted by waveOffsetA.
      // p_b_wave need to be offseted by waveOffsetB.

      auto NumInputBlksConstantOp =
          b.create<ConstantIndexOp>(loc, num_input_blks);

      auto loopKLoad = b.create<AffineForOp>(loc, 0, KPerThread);
      auto lklb = OpBuilder::atBlockBegin(loopKLoad.getBody());
      auto lkliv = loopKLoad.getInductionVar();

      Value sourceOffsetA = lklb.create<AddIOp>(
          loc,
          lklb.create<MulIOp>(
              loc,
              lklb.create<AddIOp>(
                  loc, lklb.create<MulIOp>(loc, lkliv, NumInputBlksConstantOp),
                  blk_id),
              MConstantOp),
          kBaseA);

      if (KPack > 1)
        sourceOffsetA = lklb.create<MulIOp>(
            loc, sourceOffsetA, lklb.create<ConstantIndexOp>(loc, KPack));

      Value valueA = lklb.create<InBoundsLoadOp>(loc, bufferAElementType,
                                                 op.matrixA(), sourceOffsetA);
      lklb.create<memref::StoreOp>(loc, valueA, bufferA, ValueRange{lkliv});

      Value sourceOffsetB = lklb.create<AddIOp>(
          loc,
          lklb.create<MulIOp>(
              loc,
              lklb.create<AddIOp>(
                  loc, lklb.create<MulIOp>(loc, lkliv, NumInputBlksConstantOp),
                  blk_id),
              NConstantOp),
          kBaseB);

      if (KPack > 1)
        sourceOffsetB = lklb.create<MulIOp>(
            loc, sourceOffsetB, lklb.create<ConstantIndexOp>(loc, KPack));

      Value valueB = lklb.create<InBoundsLoadOp>(loc, bufferBElementType,
                                                 op.matrixB(), sourceOffsetB);
      lklb.create<memref::StoreOp>(loc, valueB, bufferB, ValueRange{lkliv});
    }

    // XdlopsGemm Logic, similar between reduction/non-reduction path
    // for(index_t k_i = 0; k_i < KPerThread; ++k_i) {
    //   bufferAElement = a[k_i];
    //   bufferBElement = b[k_i];
    //   // Loop within a kpack
    //   for(index_t ki_i = k_base; ki_i < k_base * KRepeats; ++ki_i)
    //     argA = &bufferAElement[ki_i];
    //     argB = &bufferAElement[ki_i];
    //     p_c_thread = mfma_type.template run<MPerXlops * MRepeats,
    //                                         NPerXdlops * NRepeats,
    //                                         AStride,
    //                                         BStride>(argA, argB,
    //       p_c_thread);
    // }

    auto outerLoop =
        b.create<AffineForOp>(loc, 0, KPerThread, 1, op.vectorCs());
    auto outerLoopb = OpBuilder::atBlockBegin(outerLoop.getBody());
    auto outerLoopiv = outerLoop.getInductionVar();

    Value bufferAElement = outerLoopb.create<memref::LoadOp>(
        loc, bufferAElementType, bufferA, ValueRange{outerLoopiv});
    Value bufferBElement = outerLoopb.create<memref::LoadOp>(
        loc, bufferBElementType, bufferB, ValueRange{outerLoopiv});

    auto innerLoop = outerLoopb.create<AffineForOp>(
        loc, 0, KRepeats * k_base, k_base, outerLoop.getRegionIterArgs());
    auto innerLoopb = OpBuilder::atBlockBegin(innerLoop.getBody());
    auto innerLoopiv = innerLoop.getInductionVar();

    Value argA;
    Value argB;

    if (KPack == 1) {
      // When KPack == 1, we are dealing with memref<T>.
      // Use a combination of inner and outer offset to figure out the actual
      // offset from vgpr
      Value offset = innerLoopb.create<AddIOp>(
          loc, innerLoopb.create<MulIOp>(loc, outerLoopiv, KBaseConstantOp),
          innerLoopb.create<MulIOp>(loc, innerLoopiv, KBaseConstantOp));

      if (k_base == 1) {
        // xdlops needs only 1 element, load directly from buffer.
        argA = innerLoopb.create<memref::LoadOp>(loc, argType, bufferA,
                                                 ValueRange{offset});
        argB = innerLoopb.create<memref::LoadOp>(loc, argType, bufferB,
                                                 ValueRange{offset});
      } else {
        // k_base > 1, use transferRead to load a vector length equivalent
        // with a xdlops argument.
        argA = innerLoopb.create<vector::TransferReadOp>(
            loc, argType.cast<VectorType>(), bufferA, ValueRange{offset});
        argB = innerLoopb.create<vector::TransferReadOp>(
            loc, argType.cast<VectorType>(), bufferB, ValueRange{offset});
      }
    } else {
      // At this point, we are guaranteed that buffer element vectorization
      // length (kPack) must be a multiple of k_base. Use extractsliceop
      // to handle a independent data slice at a time.
      argA = innerLoopb.create<ExtractSliceOp>(loc, argType, bufferAElement,
                                               innerLoopiv);
      argB = innerLoopb.create<ExtractSliceOp>(loc, argType, bufferBElement,
                                               innerLoopiv);
    }

    SmallVector<Value, 4> mfmas;
    for (int64_t i = 0; i < vectorNumber; ++i) {
      auto vectorC = innerLoop.getRegionIterArgs()[i];
      auto mfma = innerLoopb.create<amdgpu::MFMAOp>(
          loc, vectorType, mfmaInstr, argA, argB, vectorC,
          /*cbsz=*/imms[i][0], /*abid=*/imms[i][1], /*blgp=*/imms[i][2]);
      mfmas.push_back(mfma);
    }
    innerLoopb.create<AffineYieldOp>(loc, mfmas);

    outerLoopb.create<AffineYieldOp>(loc, innerLoop.results());

    b.replaceOp(op, outerLoop.results());

    return success();
  }
};

void MIOpenThreadwiseGemmLoweringPass::runOnOperation() {
  FuncOp op = getOperation();
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<miopen::ThreadwiseGemmOp, miopen::XdlopsGemmV2Op>();
  target.addLegalDialect<amdgpu::AMDGPUDialect, arith::ArithmeticDialect,
                         miopen::MIOpenDialect, AffineDialect,
                         memref::MemRefDialect, vector::VectorDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<ThreadwiseGemmRewritePattern, XdlopsGemmV2RewritePattern>(ctx);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();
}

} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createMIOpenThreadwiseGemmLoweringPass() {
  return std::make_unique<MIOpenThreadwiseGemmLoweringPass>();
}
