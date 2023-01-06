//===- LowerRockOps.cpp - MLIR Rock ops lowering passes ---------------===//
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
// These passes convert the Rock threadwise ops into constructs from the
// rest of MLIR so that they can be lowered to the GPU and LLVM dialects.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include <iterator>
#include <numeric>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTHREADWISEGEMMLOWERINGPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-threadwise-gemm-lowering"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockThreadwiseGemmLoweringPass
    : public rock::impl::RockThreadwiseGemmLoweringPassBase<
          RockThreadwiseGemmLoweringPass> {
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
    Value gemmA = adaptor.getMatrixA();
    Value gemmB = adaptor.getMatrixB();
    Value gemmC = adaptor.getMatrixC();
    auto gemmAType = gemmA.getType().cast<MemRefType>();
    Type dataType = gemmAType.getElementType();

    ArrayRef<int64_t> aShape = gemmAType.getShape();
    int64_t k = aShape[0];
    int64_t m = aShape[1];
    int64_t kPack = aShape[2];
    int64_t n = gemmB.getType().cast<MemRefType>().getShape()[1];
    // Note for future: when we use dot products, we should increase this to
    // the number of elements supported by the relevant dot product.
    int64_t loadKpackLen = 1;
    LLVM_DEBUG(llvm::dbgs() << "Threadwise gemm:\n"
                            << "k = " << k << "\n"
                            << "m = " << m << "\n"
                            << "n = " << n << "\n"
                            << "kPack = " << kPack << "\n"
                            << "loadKpackLen = " << loadKpackLen << "\n");
    if (loadKpackLen > kPack || kPack % loadKpackLen != 0)
      return op->emitOpError("load length " + Twine(loadKpackLen) +
                             " not compatible with kpack of " + Twine(kPack));
    SmallVector<int64_t, 4> dimensions = {k, m, n, kPack};
    SmallVector<int64_t, 4> strides = {1, 1, 1, loadKpackLen};
    auto abType = VectorType::get(loadKpackLen, dataType);

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
      // These are vector::TransferRead ops so they always return a vector
      // result so that FMA doesn't complain
      Value aVal = b.create<vector::TransferReadOp>(
          loc, abType, bufferA, gemmLoop.getLowerCoords(/*domain=*/0),
          /*inBounds=*/ArrayRef<bool>(true));
      Value bVal = b.create<vector::TransferReadOp>(
          loc, abType, bufferB, gemmLoop.getLowerCoords(/*domain=*/1),
          /*inBounds=*/ArrayRef<bool>(true));
      ValueRange cCoords = gemmLoop.getLowerCoords(/*domain=*/2);
      Value cVal = b.create<InBoundsLoadOp>(loc, dataType, bufferC, cCoords);

      Value cVector = b.create<vector::SplatOp>(loc, abType, cVal);
      Value result;
      if (dataType.isa<IntegerType>()) {
        Value mul = b.create<MulIOp>(loc, aVal, bVal);
        result = b.create<AddIOp>(loc, mul, cVector);
        if (abType.getNumElements() != 1)
          return op.emitOpError(
              "Shouldn't've gone down the scalar code path (int)");
        result = b.create<vector::ExtractElementOp>(loc, result, zeroConst);
      } else if (dataType.isa<FloatType>()) {
        result = b.create<vector::FMAOp>(loc, aVal, bVal, cVector);
        if (abType.getNumElements() != 1)
          return op.emitOpError(
              "Shouldn't've gone down the scalar code path (float)");
        result = b.create<vector::ExtractElementOp>(loc, result, zeroConst);
      } else {
        llvm_unreachable("Validation should make this ints or floats only");
      }

      b.create<InBoundsStoreOp>(loc, result, bufferC, cCoords);
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

    XdlopsGemmParamsAttr tuningParams = op.getParams();
    // Obtain critical information.
    int64_t KPack = tuningParams.getKpack();
    int64_t K = tuningParams.getKPerBlock();
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();

    // Workload of either mPerWave and nPerWave that are larger
    // than wave size of 64 will be executed by repeats
    // TODO: amend this for tuning parameter selection as well
    int64_t waveSize = 64;
    int64_t MPerXdlops = 32;
    int64_t NPerXdlops = 32;

    auto dataType =
        adaptor.getMatrixA().getType().cast<MemRefType>().getElementType();
    if (dataType.isa<VectorType>()) {
      dataType = dataType.cast<VectorType>().getElementType();
    }

    auto maybeMfmaInsnGroup =
        MfmaInsnGroup::select(dataType, MPerXdlops, NPerXdlops);
    if (failed(maybeMfmaInsnGroup)) {
      return emitError(loc) << "Failed to select xdlops instruction group.\n";
    }
    MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;

    VectorType vectorType = mfmaGroup.getRetType();
    auto imms = mfmaGroup.getImms();
    int64_t nResultVectors = imms.size();
    Type argType = mfmaGroup.getArgType();

    MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();

    int64_t mfmaNonKDim = mfmaAttr.mfmaNonKDim;
    int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
    int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;
    int64_t k_base = mfmaAttr.k_base;

    bool IsKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);

    Value matrixA = adaptor.getMatrixA();
    Value matrixB = adaptor.getMatrixB();
    Value matrixC = adaptor.getMatrixC();
    auto matrixAType = matrixA.getType().cast<MemRefType>();
    auto matrixBType = matrixB.getType().cast<MemRefType>();
    Type matrixAElementType = matrixAType.getElementType();
    Type matrixBElementType = matrixBType.getElementType();

    ArrayRef<int64_t> aShape = matrixAType.getShape();
    ArrayRef<int64_t> bShape = matrixBType.getShape();
    int64_t mRepeats = aShape[0];
    int64_t nRepeats = bShape[0];

    int64_t KPerThread = IsKReduction ? K / inputSpansPerMfmaIn : K;

    SmallVector<int64_t> dimensions = {mRepeats, nRepeats, KPerThread,
                                       nResultVectors};
    TopDownTMBuilder aView(b, {"m", "n", "k", "v"}, dimensions, loc);
    aView.ignore("n");
    aView.ignore("v");
    aView.passThrough({"m", "k"}, {0, 1}, {"m", "k"});
    TransformMapAttr aViewAttr = aView.get();

    TopDownTMBuilder bView(b, {"m", "n", "k", "v"}, dimensions, loc);
    bView.ignore("m");
    bView.ignore("v");
    bView.passThrough({"n", "k"}, {0, 1}, {"n", "k"});
    TransformMapAttr bViewAttr = bView.get();

    TopDownTMBuilder cView(b, {"m", "n", "k", "v"}, dimensions, loc);
    cView.ignore("k");
    cView.passThrough({"m", "n", "v"}, {0, 1, 2}, {"m", "n", "v"});
    TransformMapAttr cViewAttr = cView.get();

    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
    SmallVector<Value, 4> startCoords(4, zeroConstantOp);

    ArrayAttr aTransforms, bTransforms, cTransforms;
    Value bufferA, bufferB, bufferC;
    // Use linear coordinate on original 1d buffer
    std::tie(bufferA, aTransforms) = untransform(b, matrixA, {aViewAttr});
    std::tie(bufferB, bTransforms) = untransform(b, matrixB, {bViewAttr});
    std::tie(bufferC, cTransforms) = untransform(b, matrixC, {cViewAttr});

    auto populateMfma = [&](OpBuilder &b, Value &argA, Value &argB,
                            Value &regCOffset) {
      for (int64_t i = 0; i < nResultVectors; ++i) {
        Value offset = b.createOrFold<arith::ConstantIndexOp>(loc, i);
        offset = b.create<AddIOp>(loc, offset, regCOffset);

        auto vectorC =
            b.create<memref::LoadOp>(loc, vectorType, bufferC, offset);
        auto mfma = b.create<amdgpu::MFMAOp>(
            loc, vectorType, mfmaNonKDim, mfmaNonKDim, mfmaAttr.k,
            mfmaAttr.blocksMfma, argA, argB, vectorC, /*cbsz=*/imms[i].cbsz,
            /*abid=*/imms[i].abid,
            /*blgp=*/imms[i].blgp, /*reducePrecision=*/false, /*negateA=*/false,
            /*negateB=*/false, /*negateC=*/false);
        auto vectorD = mfma.getDestD();

        b.create<memref::StoreOp>(loc, vectorD, bufferC, offset);
      }
    };

    auto generateMfmaOnKDim = [&](Value &regAOffset, Value &regBOffset,
                                  Value &regCOffset) {
      if (KPack == 1) {
        auto mfmaLoop = b.create<TransformingForOp>(
            loc, ArrayRef<ValueRange>{{zeroConstantOp}},
            ArrayRef<Attribute>{b.getArrayAttr({})},
            /*bounds=*/ArrayRef<int64_t>{KPerThread},
            /*strides=*/ArrayRef<int64_t>{k_base},
            /*forceUnroll=*/true, /*useIndexDiffs=*/true);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(mfmaLoop.getBody());
          Value coord = mfmaLoop.getLowerCoords(/*domain=*/0)[0];

          auto loadArg = [&](Value regOffset, Value matrix) {
            Value regIdx = b.create<AddIOp>(loc, regOffset, coord);
            Value arg;
            if (k_base == 1) {
              // xdlops needs only 1 element, load directly from buffer.
              arg = b.create<InBoundsLoadOp>(loc, argType, matrix,
                                             ValueRange{regIdx});
            } else {
              // k_base > 1, use transferRead to load a vector length equivalent
              // with a xdlops argument.
              arg = b.create<vector::TransferReadOp>(
                  loc, argType.cast<VectorType>(), matrix, ValueRange{regIdx},
                  /*InBounds*/ ArrayRef<bool>(true));
            }
            return arg;
          };

          Value argA = loadArg(regAOffset, bufferA);
          Value argB = loadArg(regBOffset, bufferB);
          populateMfma(b, argA, argB, regCOffset);
        };
      } else {
        // for(index_t k_i = 0; k_i < KPerThread; ++k_i) {
        //   matrixAElement = a[k_i];
        //   matrixBElement = b[k_i];
        //   // Loop within a kpack
        //   for(index_t ki_i = 0; ki_i < k_base * KRepeats; ki_i += k_base)
        //     argA = &matrixAElement[ki_i];
        //     argB = &matrixAElement[ki_i];
        //     p_c_thread = mfma_type.template run<MPerXlops * mRepeats,
        //                                         NPerXdlops * nRepeats,
        //                                         AStride,
        //                                         BStride>(argA, argB,
        //       p_c_thread);
        // }
        int64_t outerLoopUpperBound = KPerThread;
        int64_t KRepeats = KPack / k_base;

        auto outerLoop = b.create<AffineForOp>(loc, 0, outerLoopUpperBound);
        auto outerLoopb = ConversionPatternRewriter::atBlockBegin(
            outerLoop.getBody(), b.getListener());
        auto outerLoopiv = outerLoop.getInductionVar();

        auto loadSingleKPack = [&](Value regOffset, Type elementType,
                                   Value matrix) {
          Value regIdx = outerLoopb.create<AddIOp>(loc, regOffset, outerLoopiv);
          Value element = outerLoopb.create<memref::LoadOp>(
              loc, elementType, matrix, ValueRange{regIdx});
          return element;
        };

        Value matrixAElement =
            loadSingleKPack(regAOffset, matrixAElementType, bufferA);
        Value matrixBElement =
            loadSingleKPack(regBOffset, matrixBElementType, bufferB);

        auto innerLoop =
            outerLoopb.create<AffineForOp>(loc, 0, KRepeats * k_base, k_base);
        auto innerLoopb = ConversionPatternRewriter::atBlockBegin(
            innerLoop.getBody(), outerLoopb.getListener());
        auto innerLoopiv = innerLoop.getInductionVar();

        // At this point, we are guaranteed that buffer element vectorization
        // length (kPack) must be a multiple of k_base. Use extractsliceop
        // to handle a independent data slice at a time.
        Value argA = innerLoopb.create<ExtractSliceOp>(
            loc, argType, matrixAElement, innerLoopiv);
        Value argB = innerLoopb.create<ExtractSliceOp>(
            loc, argType, matrixBElement, innerLoopiv);
        populateMfma(innerLoopb, argA, argB, regCOffset);
      }
    };

    auto gemmLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{startCoords, startCoords, startCoords},
        ArrayRef<Attribute>{aTransforms, bTransforms, cTransforms}, dimensions,
        /*strides=*/ArrayRef<int64_t>{1, 1, KPerThread, nResultVectors},
        /*forceUnroll=*/true, /*useIndexDiffs=*/false);
    {
      Value regAOffset = gemmLoop.getLowerCoords(/*domain=*/0)[0];
      Value regBOffset = gemmLoop.getLowerCoords(/*domain=*/1)[0];
      Value regCOffset = gemmLoop.getLowerCoords(/*domain=*/2)[0];

      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(gemmLoop.getBody());
      generateMfmaOnKDim(regAOffset, regBOffset, regCOffset);
    };

    b.eraseOp(op);
    return success();
  }
};

void RockThreadwiseGemmLoweringPass::runOnOperation() {
  func::FuncOp op = getOperation();
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::ThreadwiseGemmOp, rock::XdlopsGemmV2Op>();
  target.addLegalDialect<amdgpu::AMDGPUDialect, arith::ArithmeticDialect,
                         rock::RockDialect, AffineDialect,
                         memref::MemRefDialect, vector::VectorDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<ThreadwiseGemmRewritePattern, XdlopsGemmV2RewritePattern>(ctx);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();
}

} // end anonymous namespace
