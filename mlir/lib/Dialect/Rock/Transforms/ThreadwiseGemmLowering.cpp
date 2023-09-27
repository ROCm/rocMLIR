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

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "AccelEmitter.h"
#include "llvm/Support/Debug.h"

#include <iterator>
#include <memory>
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

struct ThreadwiseTransposeRewritePattern
    : public OpConversionPattern<ThreadwiseTransposeOp> {
  using OpConversionPattern<ThreadwiseTransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseTransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};

struct ThreadwiseWriteAllRewritePattern
    : public OpConversionPattern<ThreadwiseWriteAllOp> {
  using OpConversionPattern<ThreadwiseWriteAllOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseWriteAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};

struct ThreadwiseReadIntoRewritePattern
    : public OpConversionPattern<ThreadwiseReadIntoOp> {
  using OpConversionPattern<ThreadwiseReadIntoOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseReadIntoOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
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
    bool isBigA, isBigB, isBigC;
    std::tie(bufferA, aTransforms, isBigA) = untransform(b, gemmA, {aViewAttr});
    std::tie(bufferB, bTransforms, isBigB) = untransform(b, gemmB, {bViewAttr});
    std::tie(bufferC, cTransforms, isBigC) = untransform(b, gemmC, {cViewAttr});
    if (isBigA || isBigB || isBigC)
      return b.notifyMatchFailure(loc, "we don't have 2 GB of registers");

    auto gemmLoop = b.replaceOpWithNewOp<TransformingForOp>(
        op, ArrayRef<ValueRange>{startCoords, startCoords, startCoords},
        ArrayRef<Attribute>{aTransforms, bTransforms, cTransforms}, dimensions,
        /*strides=*/std::nullopt, /*forceUnroll=*/true,
        /*useIndexDiffs=*/false);

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
// ThreadwiseAccelGemm lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseAccelGemmRewritePattern : public OpConversionPattern<ThreadwiseAccelGemmOp> {
  using OpConversionPattern<ThreadwiseAccelGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseAccelGemmOp op, ThreadwiseAccelGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {

  Location loc = op.getLoc();
  auto sourceA = cast<TypedValue<MemRefType>>(adaptor.getSourceA());
  auto sourceB = cast<TypedValue<MemRefType>>(adaptor.getSourceB());
  auto destView = cast<TypedValue<MemRefType>>(adaptor.getDest());

  // ArrayRef<int64_t> inputShapeA, inputShapeB;
  // inputShapeA = sourceViewA.getType().getShape();
  // inputShapeB = sourceViewB.getType().getShape();

  auto [buffer, transforms, needs64BitIdxA] = untransform(b, destView);

  int64_t numValuesA = sourceA.getType().getNumElements();
  int64_t numValuesB = sourceB.getType().getNumElements();

  if (numValuesA != numValuesB){
    return emitError(loc)<<"Values of the source buffers A and Bneed to be the same.\n";
  }

  int64_t numValues = numValuesA;

  // Types are supposed to be vector typed
  VectorType argTypeA = sourceA.getType().cast<MemRefType>().getElementType().dyn_cast<VectorType>();
  auto elementTypeA = argTypeA.getElementType();

  VectorType argTypeB = sourceB.getType().cast<MemRefType>().getElementType().dyn_cast<VectorType>();
  auto elementTypeB = argTypeB.getElementType();

  RockAccelTuningParamAttrInterface tuningParams = op.getParams();
  StringRef arch = op.getArch();

  Value zero = b.createOrFold<ConstantIndexOp>(loc, 0);
  SmallVector<Value, 3> writeStartCoords = llvm::to_vector<3>(op.getExtraIndices());
  writeStartCoords.push_back(zero);
  SmallVector<int64_t, 3> bounds(writeStartCoords.size() - 1, 1);
  bounds.push_back(numValues);
  SmallVector<int64_t, 3> strides(writeStartCoords.size() - 1, 1);
  strides.push_back(1);
  size_t extraIdxCount = op.getExtraIndices().size();

  SmallVector<Attribute> transformAttrs;
  auto emitter = rock::accel::AccelEmitter::select(
      op.getFeatures(), elementTypeA, elementTypeB, arch, tuningParams);

  if (!emitter)
    return emitError(loc)
            << "Failed to select any accelerator instruction.\n";

  //  We should put this on the caller
  // Value regCOffset = b.create<MulIOp>(
  //     loc,
  //     b.create<AddIOp>(
  //         loc, b.create<MulIOp>(loc, mRepeat, nRepeatsConstantOp), nRepeat),
  //     nResultVectorsConstantOp);
  auto computeLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{writeStartCoords, writeStartCoords},
      ArrayRef<Attribute>{b.getArrayAttr({}), b.getArrayAttr({}), transforms}, bounds, strides,
      false, false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(computeLoop.getBody());
      Value argA = b.create<memref::LoadOp>(loc, argTypeA, sourceA, computeLoop.getLowerCoords(0)[extraIdxCount]);
      Value argB = b.create<memref::LoadOp>(loc, argTypeB, sourceB, computeLoop.getLowerCoords(1)[extraIdxCount]);
      emitter->emitThreadwiseLoop(b, loc, argA, argB, buffer, computeLoop.getLowerCoords(2)[0]);
  }

    // // Extract relevant accel emitter parameters
    // rock::accel::AccelEmitterParams params = emitter->getParams();
    // int64_t nRepeats = params.nRepeats;
    // int64_t kBasePerThread = params.kBasePerThread;
    // int64_t nResultVectors = params.nResultVectors;
    // Type argTypeA = params.argTypeA;
    // Type argTypeB = params.argTypeB;

    // if (!emitter)
    //   return emitError(loc)
    //          << "Failed to select any accelerator instruction.\n";

    // SmallVector<Value, 4> startCoords(4, zeroConstantOp);


    // auto mRepeat = op.getMRepeat();
    // auto nRepeat = op.getNRepeat();

    // Value nResultVectorsConstantOp =
    //     b.createOrFold<ConstantIndexOp>(loc, nResultVectors);
    // Value nRepeatsConstantOp = b.create<ConstantIndexOp>(loc, nRepeats);

    // Value regCOffset = b.create<MulIOp>(
    //     loc,
    //     b.create<AddIOp>(
    //         loc, b.create<MulIOp>(loc, mRepeat, nRepeatsConstantOp), nRepeat),
    //     nResultVectorsConstantOp);

    // auto accelLoop = b.create<TransformingForOp>(
    //     loc, ArrayRef<ValueRange>{{zeroConstantOp}},
    //     ArrayRef<Attribute>{b.getArrayAttr({})},
    //     /*bounds=*/ArrayRef<int64_t>{kBasePerThread},
    //     /*strides=*/ArrayRef<int64_t>{1},
    //     /*forceUnroll=*/false, /*useIndexDiffs=*/false);
    // {
    //   OpBuilder::InsertionGuard guard(b);
    //   b.setInsertionPointToStart(accelLoop.getBody());
    // }

    b.eraseOp(op);
    return success();
  }
};

/// Amend the operation chain (and computed shape) for a read/write to add a
/// length-1 iteration index to 0-dimensional (scalar) buffers.
static void addIterationIndexIfScalar(PatternRewriter &b, Location loc,
                                      ArrayRef<int64_t> &shape,
                                      ArrayAttr &extraViews) {
  if (!shape.empty())
    return;
  TopDownTMBuilder addZero(b, {"zero"}, {1}, loc);
  addZero.ignore("zero");
  TransformMapAttr addZeroAttr = addZero.get();
  shape = addZeroAttr.getUpperBounds().asArrayRef();
  SmallVector<Attribute, 4> views = {addZeroAttr};
  if (extraViews)
    views.append(extraViews.begin(), extraViews.end());
  extraViews = b.getArrayAttr(views);
}

LogicalResult ThreadwiseReadIntoRewritePattern::matchAndRewrite(
    ThreadwiseReadIntoOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  auto sourceView = cast<TypedValue<MemRefType>>(adaptor.getSource());
  ArrayAttr extraViews = op.getExtraViews();
  auto dest = cast<TypedValue<MemRefType>>(adaptor.getDest());
  ArrayRef<int64_t> inputShape;
  if (extraViews.empty())
    inputShape = sourceView.getType().getShape();
  else
    inputShape = extraViews[0].cast<TransformMapAttr>().getUpperBounds();
  addIterationIndexIfScalar(b, loc, inputShape, extraViews);

  auto [buffer, transforms, needs64BitIdx] =
      untransform(b, sourceView, extraViews);

  int64_t numValues = dest.getType().getNumElements();
  MemRefType srcBufferType = buffer.getType().cast<MemRefType>();
  MemRefType dstBufferType = dest.getType().cast<MemRefType>();

  bool isSrcVectorBuffer = srcBufferType.getElementType().isa<VectorType>();
  bool isDstVectorBuffer = dstBufferType.getElementType().isa<VectorType>();
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
  Type loadType;
  int64_t vectorSrcLen, vectorDstLen;
  int64_t srcStride;
  VectorType dstVectorType;

  if (isSrcVectorBuffer) {
    loadType = elementType;
    vectorSrcLen = elementType.dyn_cast<VectorType>().getNumElements();
    srcStride = 1;
  } else {
    vectorSrcLen = getMaxVectorizationForDatatype(
        transforms, /*dim=*/extraIdxCount, numValues, bufferShape, elementType);
    // In the future, this might get merged into the vectorizer.
    transforms = collapseContiguousMerges(transforms, bufferShape);
    srcStride = vectorSrcLen;
    loadType = vectorTypeOrSelf(elementType, vectorSrcLen);
  }

  if (isDstVectorBuffer) {
    dstVectorType = dstBufferType.getElementType().dyn_cast<VectorType>();
    vectorDstLen = dstVectorType.dyn_cast<VectorType>().getNumElements();
    numValues = (numValues * vectorDstLen) / vectorSrcLen;
  }

  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for read_into = "
                          << vectorSrcLen << "\n");
  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

  SmallVector<Value, 3> readStartCoords =
      llvm::to_vector<3>(op.getExtraIndices());
  readStartCoords.push_back(zero);
  SmallVector<int64_t, 3> bounds(readStartCoords.size() - 1, 1);
  bounds.push_back(numValues);
  SmallVector<int64_t, 3> strides(readStartCoords.size() - 1, 1);
  strides.push_back(srcStride);

  SmallVector<Attribute> transformAttrs;

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
          loadLoop.getLowerCoords(/*domain=*/0), needs64BitIdx);
      b.create<InBoundsStoreOp>(loc, loaded, dest,
                                loadLoop.getLowerCoords(
                                    /*domain=*/1)[extraIdxCount]);
    } else {
      if (needs64BitIdx)
        return b.notifyMatchFailure(
            loc, "non-global address spaces must have 32-bit pointers");
      TypedValue<IntegerType> valid = loadLoop.getValidity(/*domain=*/0);
      scf::IfOp ifb =
          b.create<scf::IfOp>(loc, loadType, valid, /*withElseRegion=*/true);
      {
        OpBuilder thenb = ifb.getThenBodyBuilder();
        Value loaded;
        if (!isSrcVectorBuffer)
          loaded = thenb.create<InBoundsLoadOp>(
              loc, loadType, buffer, loadLoop.getLowerCoords(/*domain=*/0));
        else
          loaded = thenb.create<memref::LoadOp>(
              loc, loadType, buffer, loadLoop.getLowerCoords(/*domain=*/0));
        thenb.create<scf::YieldOp>(loc, loaded);
      }
      {
        OpBuilder elseb = ifb.getElseBodyBuilder();
        Value zeroVal = createZeroConstantOp(elseb, loc, loadType);
        elseb.create<scf::YieldOp>(loc, zeroVal);
      }

      Value destOffset = loadLoop.getLowerCoords(1)[extraIdxCount];
      if (!isDstVectorBuffer && !isSrcVectorBuffer) {
        b.create<InBoundsStoreOp>(loc, ifb.getResult(0), dest, destOffset);
      } else if (!isDstVectorBuffer && isSrcVectorBuffer) {
        destOffset = b.create<arith::MulIOp>(
            loc, destOffset, b.create<ConstantIndexOp>(loc, vectorSrcLen));
        b.create<InBoundsStoreOp>(loc, ifb.getResult(0), dest, destOffset);
      } else {
        // Destination is a vector buffer
        // Value tid = b.create<WorkitemIdOp>(loc, b.getIndexType());
        // b.create<gpu::PrintfOp>(loc, "thread id %d : %d\n", ValueRange{tid,
        // loadLoop.getLowerCoords(0)[0]});

        if (vectorSrcLen == vectorDstLen) {
          b.create<memref::StoreOp>(loc, ifb.getResult(0), dest,
                                    loadLoop.getLowerCoords(
                                        /*domain=*/1)[extraIdxCount]);
        } else if (vectorSrcLen > vectorDstLen) {
          int64_t numStores = vectorSrcLen / vectorDstLen;
          Value idx = loadLoop.getLowerCoords(1)[extraIdxCount];
          Value value = ifb.getResult(0);

          Value baseDestOffset = b.createOrFold<arith::MulIOp>(
              loc, idx, b.createOrFold<arith::ConstantIndexOp>(loc, numStores));
          for (int64_t i = 0; i < numStores; ++i) {
            Value sliceStart =
                b.createOrFold<arith::ConstantIndexOp>(loc, vectorDstLen * i);
            Value slice =
                b.create<ExtractSliceOp>(loc, dstVectorType, value, sliceStart);
            Value destOffset = b.createOrFold<arith::AddIOp>(
                loc, baseDestOffset,
                b.createOrFold<arith::ConstantIndexOp>(loc, i));
            b.create<memref::StoreOp>(loc, slice, dest, ValueRange{destOffset});
          }
        } else { // srcVecLen < dstVecLen
          // Here we are gathering loaded values into vectors for passing into
          // MFMAs.
          Value value = ifb.getResult(0);
          Value destValsPerKpack = b.createOrFold<arith::ConstantIndexOp>(
              loc, vectorDstLen / vectorSrcLen);
          Value idx = loadLoop.getLowerCoords(1)[extraIdxCount];

          Value destOffset =
              b.createOrFold<arith::DivUIOp>(loc, idx, destValsPerKpack);
          Value destVecPart =
              b.createOrFold<arith::RemUIOp>(loc, idx, destValsPerKpack);
          Value destSlicePos = b.createOrFold<arith::MulIOp>(
              loc, destVecPart,
              b.createOrFold<arith::ConstantIndexOp>(loc, vectorSrcLen));
          Value destVec = b.create<memref::LoadOp>(loc, dstVectorType, dest,
                                                   ValueRange{destOffset});
          Value newDestVec = b.create<InsertSliceOp>(loc, dstVectorType, value,
                                                     destVec, destSlicePos);
          b.create<memref::StoreOp>(loc, newDestVec, dest,
                                    ValueRange{destOffset});
        }
      }
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

  ArrayAttr extraViews = op.getExtraViews();
  ArrayRef<int64_t> outputShape;
  if (extraViews.empty())
    outputShape = destView.getType().getShape();
  else
    outputShape = extraViews[0].cast<TransformMapAttr>().getUpperBounds();
  addIterationIndexIfScalar(b, loc, outputShape, extraViews);

  auto [buffer, transforms, needs64BitIdx] =
      untransform(b, destView, extraViews);

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
  auto destElemType = buffer.getType().cast<MemRefType>().getElementType();
  // We are vectorizing in the iter dimension, not block ID or thread ID
  int64_t maxVecLen = iterLen;
  Type elementType = destElemType;
  int64_t implicitStride = 1;
  // If the dest is already being viewed as vector-typed, there's no good
  // mechanism to, for example, store a vector<8xf32> as two consecutive
  // vector<4xf32>s, and there's unlikely to be any performance benefit from
  // doing so. Therefore, don't bother.
  if (auto elemVecType = destElemType.dyn_cast<VectorType>()) {
    implicitStride = elemVecType.getNumElements();
    maxVecLen = elemVecType.getNumElements();
    elementType = elemVecType.getElementType();
  }
  int64_t vectorLen = getMaxVectorizationForDatatype(
      transforms, /*dim=*/extraIdxCount, maxVecLen, bufferShape, elementType,
      implicitStride);
  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for write_all = " << vectorLen
                          << "\n");

  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  transforms = collapseContiguousMerges(transforms, bufferShape);

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

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
                              outLoop.getLowerCoords(/*domain=*/1),
                              needs64BitIdx ? b.getUnitAttr() : nullptr,
                              /*canStoreOffEnd=*/nullptr);
    } else {
      if (needs64BitIdx)
        return b.notifyMatchFailure(
            loc, "non-global address spaces must have 32-bit pointers");
      Type loadType = vectorTypeOrSelf(elementType, vectorLen);
      TypedValue<IntegerType> valid = outLoop.getValidity(/*domain=*/0);
      scf::IfOp ifb = b.create<scf::IfOp>(loc, valid, /*withElseRegion=*/false);
      {
        OpBuilder thenb = ifb.getThenBodyBuilder();
        Value loaded =
            thenb.create<InBoundsLoadOp>(loc, loadType, source,
                                         outLoop.getLowerCoords(
                                             /*domain=*/0)[extraIdxCount]);
        if (!destElemType.isa<VectorType>()) {
          thenb.create<InBoundsStoreOp>(loc, loaded, buffer,
                                        outLoop.getLowerCoords(/*domain=*/1));
        } else {
          thenb.create<memref::StoreOp>(loc, loaded, buffer,
                                        outLoop.getLowerCoords(/*domain=*/1));
        }
      }
    }
  }
  b.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Threadwise transpose lowering
//===----------------------------------------------------------------------===//
LogicalResult ThreadwiseTransposeRewritePattern::matchAndRewrite(
    ThreadwiseTransposeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  auto sourceView = cast<TypedValue<MemRefType>>(adaptor.getSource());
  auto destView = cast<TypedValue<MemRefType>>(adaptor.getDest());

  ArrayRef<int64_t> loadShape =
      sourceView.getType().cast<ShapedType>().getShape();
  Type elemType = sourceView.getType().cast<MemRefType>().getElementType();
  int64_t copyDPerThread = loadShape[1];
  int64_t copyKPerThread = loadShape[0];
  int64_t kpack = op.getKpack();

  // We use kpackPerThread instead of kpack to cover edge cases where
  // copyKPerThread is smaller than kpack
  int64_t kpackPerThread = std::min(copyKPerThread, kpack);

  Value rawLoadBuffer;
  ArrayAttr loadBufferView;
  bool needs64BitIdx;
  std::tie(rawLoadBuffer, loadBufferView, needs64BitIdx) =
      untransform(b, sourceView);
  assert(!needs64BitIdx && "Registers shouldn't need 64-bit indexing");
  ArrayRef<int64_t> rawLoadBufferShape =
      rawLoadBuffer.getType().cast<ShapedType>().getShape();

  Value rawStoreBuffer;
  ArrayAttr storeBufferView;
  std::tie(rawStoreBuffer, storeBufferView, needs64BitIdx) =
      untransform(b, destView);
  assert(!needs64BitIdx && "Registers shouldn't need 64-bit indexing");
  ArrayRef<int64_t> rawStoreBufferShape =
      rawStoreBuffer.getType().cast<ShapedType>().getShape();

  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 2> start(2, zero);
  SmallVector<int64_t, 2> strides(2, 1);
  int64_t vecLen = 1;
  // The store buffer is a flattened < kouter x dPerThread x kpack >.
  if (kpackPerThread == 1) {
    // if kpack == 1, then we can do vectorized loads across d dimension from/to
    // load/store buffer
    vecLen = getMaxVectorizationForDatatype(loadBufferView, /*dim=*/1,
                                            copyDPerThread, rawLoadBufferShape,
                                            elemType);
    vecLen = math_util::gcd(copyDPerThread, vecLen);
    strides[1] = vecLen;
  } else {
    // if kpack > 1, then we are limited by vectorization in k dimension and it
    // could be at most kpack.
    vecLen = getMaxVectorizationForDatatype(loadBufferView, /*dim=*/0,
                                            copyKPerThread, rawLoadBufferShape,
                                            elemType);
    vecLen = math_util::gcd(vecLen, kpackPerThread);
    strides[0] = vecLen;
  }
  loadBufferView = collapseContiguousMerges(loadBufferView, rawLoadBufferShape);
  storeBufferView =
      collapseContiguousMerges(storeBufferView, rawStoreBufferShape);

  // Run the packing loop
  auto packLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{start, start},
      ArrayRef<Attribute>{loadBufferView, storeBufferView},
      /*bounds=*/loadShape,
      /*strides=*/strides, false,
      /*useIndexDiffs=*/false);
  {
    PatternRewriter::InsertionGuard outerGuard(b);
    b.setInsertionPointToStart(packLoop.getBody());
    Type loadType = vectorTypeOrSelf(elemType, vecLen);
    auto val = b.create<InBoundsLoadOp>(loc, loadType, rawLoadBuffer,
                                        packLoop.getLowerCoords(0));

    b.create<InBoundsStoreOp>(loc, val, rawStoreBuffer,
                              packLoop.getLowerCoords(1));
  }
  b.eraseOp(op);
  return success();
}

void RockThreadwiseGemmLoweringPass::runOnOperation() {
  func::FuncOp op = getOperation();
  MLIRContext *ctx = &getContext();
  {
    ConversionTarget writeAllTarget(*ctx);
    writeAllTarget.addIllegalOp<ThreadwiseReadIntoOp, ThreadwiseWriteAllOp,
                                ThreadwiseTransposeOp>();
    writeAllTarget.addLegalDialect<
        arith::ArithDialect, rock::RockDialect, memref::MemRefDialect,
        scf::SCFDialect, vector::VectorDialect, affine::AffineDialect>();
    writeAllTarget.addLegalOp<gpu::PrintfOp>();
    RewritePatternSet writeAllPatterns(ctx);
    writeAllPatterns
        .add<ThreadwiseReadIntoRewritePattern, ThreadwiseWriteAllRewritePattern,
             ThreadwiseTransposeRewritePattern>(ctx);
    if (failed(applyPartialConversion(getOperation(), writeAllTarget,
                                      std::move(writeAllPatterns))))
      signalPassFailure();
  }
  WalkResult kernelNeeds64Bit = getOperation().walk([](Operation *op) {
    if (auto globalLoad = dyn_cast<GlobalLoadOp>(op))
      return globalLoad.getNeeds64BitIdx() ? WalkResult::interrupt()
                                           : WalkResult::advance();
    if (auto globalStore = dyn_cast<GlobalStoreOp>(op))
      return globalStore.getNeeds64BitIdx() ? WalkResult::interrupt()
                                            : WalkResult::advance();
    return WalkResult::advance();
  });
  if (kernelNeeds64Bit.wasInterrupted())
    getOperation()->setAttr("rock.64bitindex", UnitAttr::get(&getContext()));

  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::ThreadwiseGemmOp, rock::ThreadwiseAccelGemmOp>();
  target.addLegalDialect<amdgpu::AMDGPUDialect, arith::ArithDialect,
                         rock::RockDialect, affine::AffineDialect,
                         memref::MemRefDialect, vector::VectorDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<ThreadwiseGemmRewritePattern, ThreadwiseAccelGemmRewritePattern>(ctx);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();
}

} // end anonymous namespace
