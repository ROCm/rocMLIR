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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/transformMapUtils.h"

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
    int64_t K = op->getAttr("k").template cast<IntegerAttr>().getInt();
    int64_t MPerWave =
        op->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();

    auto dataType = adaptor.matrixA()
                        .getType()
                        .template cast<MemRefType>()
                        .getElementType();

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

    VectorType vectorType = xcs.vectorType;
    int64_t vectorNumber = xcs.vectorNumber;
    SmallVector<SmallVector<unsigned, 3>, 2> imms = xcs.imms;
    Type argType = xcs.argType;

    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t k_base = xcs.k_base;

    bool IsKReduction = (num_output_blks == 1) && (num_input_blks > 1);

    Value bufferA = adaptor.bufferA();
    Value bufferB = adaptor.bufferB();
    auto bufferAType = adaptor.bufferA().getType().cast<MemRefType>();
    auto bufferBType = adaptor.bufferB().getType().cast<MemRefType>();
    Type bufferAElementType = bufferAType.getElementType();
    Type bufferBElementType = bufferBType.getElementType();

    int64_t KPerThread = IsKReduction ? K / num_input_blks : K;
    int64_t KRepeats = KPack / k_base;
    if (KRepeats == 0)
      KRepeats = 1;

    auto KBaseConstantOp = b.create<ConstantIndexOp>(loc, k_base);

    // XdlopsGemm Logic, similar between reduction/non-reduction path
    // for(index_t k_i = 0; k_i < KPerThread; ++k_i) {
    //   bufferAElement = a[k_i];
    //   bufferBElement = b[k_i];
    //   // Loop within a kpack
    //   for(index_t ki_i = 0; ki_i < k_base * KRepeats; ki_i += k_base)
    //     argA = &bufferAElement[ki_i];
    //     argB = &bufferAElement[ki_i];
    //     p_c_thread = mfma_type.template run<MPerXlops * MRepeats,
    //                                         NPerXdlops * NRepeats,
    //                                         AStride,
    //                                         BStride>(argA, argB,
    //       p_c_thread);
    // }
    int64_t outerLoopUpperBound = KPerThread;
    // In case xdlops consume multiple elements from memref<T>, divide by k_base
    // such that the generated loop is not out of bounds.
    if (KPack == 1) {
      outerLoopUpperBound /= k_base;
    }
    auto regAConstantOp =
        b.create<ConstantIndexOp>(loc, op.regOffsetAAttr().getInt());
    auto regBConstantOp =
        b.create<ConstantIndexOp>(loc, op.regOffsetBAttr().getInt());

    auto outerLoop =
        b.create<AffineForOp>(loc, 0, outerLoopUpperBound, 1, op.vectorCs());
    auto outerLoopb = ConversionPatternRewriter::atBlockBegin(
        outerLoop.getBody(), b.getListener());
    auto outerLoopiv = outerLoop.getInductionVar();

    // bufferAElement and bufferBElement are only useful when KPack > 1
    Value bufferAElement;
    Value bufferBElement;

    if (KPack > 1) {
      Value regAIdx =
          outerLoopb.create<AddIOp>(loc, regAConstantOp, outerLoopiv);
      bufferAElement = outerLoopb.create<memref::LoadOp>(
          loc, bufferAElementType, bufferA, ValueRange{regAIdx});

      Value regBIdx =
          outerLoopb.create<AddIOp>(loc, regBConstantOp, outerLoopiv);
      bufferBElement = outerLoopb.create<memref::LoadOp>(
          loc, bufferBElementType, bufferB, ValueRange{regBIdx});
    }

    auto innerLoop = outerLoopb.create<AffineForOp>(
        loc, 0, KRepeats * k_base, k_base, outerLoop.getRegionIterArgs());
    auto innerLoopb = ConversionPatternRewriter::atBlockBegin(
        innerLoop.getBody(), outerLoopb.getListener());
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

      Value regAIdx = innerLoopb.create<AddIOp>(loc, regAConstantOp, offset);
      Value regBIdx = innerLoopb.create<AddIOp>(loc, regBConstantOp, offset);

      if (k_base == 1) {
        // xdlops needs only 1 element, load directly from buffer.
        argA = innerLoopb.create<memref::LoadOp>(loc, argType, bufferA,
                                                 ValueRange{regAIdx});
        argB = innerLoopb.create<memref::LoadOp>(loc, argType, bufferB,
                                                 ValueRange{regBIdx});
      } else {
        // k_base > 1, use transferRead to load a vector length equivalent
        // with a xdlops argument.
        argA = innerLoopb.create<vector::TransferReadOp>(
            loc, argType.cast<VectorType>(), bufferA, ValueRange{regAIdx});
        argB = innerLoopb.create<vector::TransferReadOp>(
            loc, argType.cast<VectorType>(), bufferB, ValueRange{regBIdx});
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
      auto vectorD = mfma.destD();

      // Note below is assuming only one of MRepeats or NRepeats is larger
      // than 1, which fits the existing blockwisegemmv2op implementation.
      // TODO: Move MRepeats and NRepeats into xdlopsgemmv2op
      int64_t regDOffset = 0;
      if (op.regOffsetAAttr().getInt() > 0 ||
          op.regOffsetBAttr().getInt() > 0) {
        regDOffset += vectorNumber;
      }
      Value offset = innerLoopb.createOrFold<arith::ConstantIndexOp>(
          loc, (regDOffset + i) * vectorType.getNumElements());
      innerLoopb.create<miopen::InBoundsStoreOp>(loc, vectorD,
                                                 adaptor.matrixRes(), offset);
    }
    innerLoopb.create<AffineYieldOp>(loc, mfmas);

    outerLoopb.create<AffineYieldOp>(loc, innerLoop.results());

    b.replaceOp(op, outerLoop.results());

    return success();
  }
};

void MIOpenThreadwiseGemmLoweringPass::runOnOperation() {
  func::FuncOp op = getOperation();
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
