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

#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
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

struct ThreadwiseCopyRewritePattern
    : public OpConversionPattern<ThreadwiseCopyOp> {
  using OpConversionPattern<ThreadwiseCopyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseCopyOp op, OpAdaptor adaptor,
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
// AccelGemm lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseAccelGemmRewritePattern
    : public OpConversionPattern<ThreadwiseAccelGemmOp> {
  using OpConversionPattern<ThreadwiseAccelGemmOp>::OpConversionPattern;

  // Create a normalized view `[startShape, sizes]`. Then convert this to a
  // "real" view by ignoring some of the indices and letting the rest pass
  // through.
  TransformMapAttr normalizeView(OpBuilder &b, Location loc,
                                 ArrayRef<StringRef> names,
                                 ArrayRef<int64_t> sizes, bool ignoreStartShape,
                                 DenseSet<StringRef> ignoreNames) const {
    unsigned pos = 0;
    unsigned newPos = 0;
    TopDownTMBuilder td(b, names, sizes, loc);
    // Convert the normalizedView to a real view by ignoring
    // the names contained in `ignoreNames` and letting the rest pass through
    for (pos = 0; pos < names.size(); pos++) {
      if (ignoreNames.contains(names[pos])) {
        td.ignore(names[pos]);
      } else {
        td.passThrough({newPos++}, {pos});
      }
    }
    return td.get();
  }

  LogicalResult matchAndRewrite(ThreadwiseAccelGemmOp op,
                                ThreadwiseAccelGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    RockAccelTuningParamAttrInterface tuningParams = op.getParams();

    auto dataTypeA =
        adaptor.getMatrixA().getType().cast<MemRefType>().getElementType();
    auto dataTypeB =
        adaptor.getMatrixB().getType().cast<MemRefType>().getElementType();
    if (dataTypeA.isa<VectorType>()) {
      dataTypeA = dataTypeA.cast<VectorType>().getElementType();
    }
    if (dataTypeB.isa<VectorType>()) {
      dataTypeB = dataTypeB.cast<VectorType>().getElementType();
    }

    Value bufferA = adaptor.getMatrixA();
    Value bufferB = adaptor.getMatrixB();
    Value bufferC = adaptor.getMatrixC();

    auto bufferAShape = op.getMatrixA().getType().getShape();
    auto bufferCShape = op.getMatrixC().getType().getShape();

    size_t computeIndices = op.getComputeIndices().size();
    auto emitter = rock::accel::AccelEmitter::select(
        op.getFeatures(), dataTypeA, dataTypeB, op.getArch(), tuningParams);

    if (!emitter)
      return emitError(loc)
             << "Failed to select any accelerator instruction.\n";

    // Extract relevant accel emitter parameters
    rock::accel::AccelEmitterParams params = emitter->getParams();
    Type argTypeA = params.argTypeA;
    Type argTypeB = params.argTypeB;

    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
    SmallVector<Value, 4> startCoords(4, zeroConstantOp);

    // Sizes of the [i,j,k] axis
    int64_t iLen = *(bufferCShape.end() - 2);
    int64_t jLen = bufferCShape.back();
    int64_t kLen = bufferAShape.back();

    // All the view need to be `[i, j, k]` so that
    // we can iterate within the `transforming_for` later. This means that we
    // want to `ignore` some of the indices, e.g, since A is [i,k] we will
    // ignore index `j` and the `extraIndices`
    TransformMapAttr normalizedViewA =
        normalizeView(b, loc, {"i", "j", "k"}, {iLen, jLen, kLen}, true, {"j"});
    TransformMapAttr normalizedViewB =
        normalizeView(b, loc, {"i", "j", "k"}, {iLen, jLen, kLen}, true, {"i"});
    TransformMapAttr normalizedViewC = normalizeView(
        b, loc, {"i", "j", "k"}, {iLen, jLen, kLen}, false, {"k"});

    auto [rawBufferA, bufferViewA, sourceANeeds64BitIdx] =
        untransform(b, bufferA, normalizedViewA);
    auto [rawBufferB, bufferViewB, sourceBNeeds64BitIdx] =
        untransform(b, bufferB, normalizedViewB);
    auto [rawBufferC, bufferViewC, dstNeeds64BitIdx] =
        untransform(b, bufferC, normalizedViewC);

    assert(!sourceANeeds64BitIdx && "Registers shouldn't need 64-bit indexing");
    assert(!sourceBNeeds64BitIdx && "Registers shouldn't need 64-bit indexing");
    assert(!dstNeeds64BitIdx && "Registers shouldn't need 64-bit indexing");

    // Loop properties
    auto computeBounds = SmallVector<int64_t>(computeIndices, 1);
    auto computeStride = SmallVector<int64_t>(computeBounds.size(), 1);
    auto computeStart = llvm::to_vector(op.getComputeIndices());

    // Emit the loop
    auto accelLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{computeStart, computeStart, computeStart},
        ArrayRef<Attribute>{bufferViewA, bufferViewB, bufferViewC},
        /*bounds=*/ArrayRef<int64_t>{1, 1, 1},
        /*strides=*/ArrayRef<int64_t>{1, 1, 1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(accelLoop.getBody());
      auto coordsA = accelLoop.getLowerCoords(/*domain=*/0);
      auto coordsB = accelLoop.getLowerCoords(/*domain=*/1);
      auto coordsC = accelLoop.getLowerCoords(/*domain=*/2);

      Value argA = b.create<memref::LoadOp>(loc, argTypeA, rawBufferA, coordsA);
      Value argB = b.create<memref::LoadOp>(loc, argTypeB, rawBufferB, coordsB);
      emitter->emitThreadwiseLoop(b, loc, argA, argB, rawBufferC, coordsC);
    }
    b.eraseOp(op);
    return success();
  }
};

LogicalResult ThreadwiseCopyRewritePattern::matchAndRewrite(
    ThreadwiseCopyOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  Value sourceView = adaptor.getSource();
  Value destView = adaptor.getDest();
  auto srcViewShape = op.getSource().getType().getShape();
  auto dstViewShape = op.getDest().getType().getShape();

  size_t extraIndicesDestSize = op.getExtraIndicesDest().size();
  size_t extraIndicesSourceSize = op.getExtraIndicesSource().size();
  SmallVector<int64_t> extraIndicesDestShape;
  for (size_t i = 0; i < extraIndicesDestSize; i++) {
    extraIndicesDestShape.push_back(dstViewShape[i]);
  }
  SmallVector<int64_t> extraIndicesSourceShape;
  for (size_t i = 0; i < extraIndicesSourceSize; i++) {
    extraIndicesSourceShape.push_back(srcViewShape[i]);
  }

  // We need to normalize the maps. I.e., if i0,...,iD-1 are the extra
  // destination indices and j0, ..., jS-1 are the extra source indices, both
  // source and destination views need to be [(i0, ..., iD-1), (j0, ..., jS-1),
  // ..., (other dest/source transform indices)]. It is important that the extra
  // dimensions are propagated top to bottom, otherwise the transformation stack
  // might loose invertibility. Note that the this will add an additional series
  // of AddDim{} operators below the existing transforms to protect against
  // shape mismatches.
  sourceView = addPassThroughIndices(b, sourceView, extraIndicesDestShape,
                                     extraIndicesSourceSize);
  destView = addPassThroughIndices(b, destView, extraIndicesSourceShape, 0);

  // Almost certainly a noop, since adding extra indices creates fresh
  // IR, but we call it just in case.
  sourceView = isolateTransforms(b, sourceView);
  destView = isolateTransforms(b, destView);

  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  Type elemType = sourceView.getType().cast<MemRefType>().getElementType();

  auto [rawLoadBuffer, loadBufferView, sourceNeeds64BitIdx] =
      untransform(b, sourceView);
  auto [rawStoreBuffer, storeBufferView, dstNeeds64BitIdx] =
      untransform(b, destView);

  assert(!sourceNeeds64BitIdx && "Registers shouldn't need 64-bit indexing");
  assert(!dstNeeds64BitIdx && "Registers shouldn't need 64-bit indexing");
  ArrayRef<int64_t> rawLoadBufferShape =
      rawLoadBuffer.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rawStoreBufferShape =
      rawStoreBuffer.getType().cast<ShapedType>().getShape();
  if (rawLoadBufferShape.size() > extraIndicesSourceSize + 1)
    return op.emitOpError("Raw load buffers have to be flat or multi buffers.");
  if (rawStoreBufferShape.size() > extraIndicesDestSize + 1)
    return op.emitOpError(
        "Raw store buffers have to be flat or multi buffers.");

  // Basic copy loop. Copy element by element from src to dest view. Only
  // consider un-extended start/stride/bounds. We will extend those later to
  // match our normalized structure
  ArrayAttr copyFromView = loadBufferView;
  ArrayAttr copyToView = storeBufferView;
  SmallVector<Value> start(srcViewShape.size() - extraIndicesSourceSize, zero);
  SmallVector<int64_t> strides(srcViewShape.size() - extraIndicesSourceSize, 1);
  SmallVector<int64_t> bounds(srcViewShape.begin() + extraIndicesSourceSize,
                              srcViewShape.end());
  int64_t vecLen = 1;

  // If we can invert the store view, we can easily find out the common
  // vectorization length.
  // When inverting, we need to peel off the final AddDim{}s used to maintain
  // type consistency if any are needed.
  Attribute storeBufferLoadIdxsAttr;
  ArrayAttr storeBufferViewForInverse = storeBufferView;
  if (extraIndicesSourceSize > 0) {
    ArrayRef<Attribute> storeBufferViews = storeBufferViewForInverse.getValue();
    storeBufferViewForInverse = b.getArrayAttr(storeBufferViews.drop_back());
    storeBufferLoadIdxsAttr = storeBufferViews.back();
  }
  auto storeBufferViewInverted =
      invertTransforms(b, loc, storeBufferViewForInverse);
  if (storeBufferViewInverted) {
    Value srcToDestView = transform(b, sourceView, storeBufferViewInverted);
    // It may be the case that we had an isolated transform stack and didn't
    // need to add extra indices. In that case, all the possible sources of
    // cloning will have declined to trigger on account of everything already
    // being in the right form. However, adding the inverses will introduce
    // additional uses, causing sanity-check assertions to trip in
    // collapseContiguousMerges(). To handle this case, we force a shallow
    // clone.
    srcToDestView = isolateTransforms(b, srcToDestView);
    VectorizationResult vecRes = getMaxVectorization(
        srcToDestView, extraIndicesSourceSize + extraIndicesDestSize);
    vecLen = vecRes.max;
    collapseContiguousMerges(srcToDestView);
    std::tie(rawLoadBuffer, copyFromView, std::ignore) =
        untransform(b, srcToDestView);
    if (storeBufferLoadIdxsAttr)
      copyToView = b.getArrayAttr({storeBufferLoadIdxsAttr});
    else
      copyToView = b.getArrayAttr({});
    start = SmallVector<Value>{zero};
    strides = SmallVector<int64_t>{vecLen};
    bounds = SmallVector<int64_t>{rawStoreBufferShape.back()};
  }

  // Extend start
  SmallVector<Value> extendedStart(op.getExtraIndicesSource());
  extendedStart.insert(extendedStart.end(), op.getExtraIndicesDest().begin(),
                       op.getExtraIndicesDest().end());
  extendedStart.insert(extendedStart.end(), start.begin(), start.end());

  // Extend bounds
  SmallVector<int64_t> extendedBounds(
      extraIndicesSourceSize + extraIndicesDestSize, 1);
  llvm::append_range(extendedBounds, bounds);

  // Extend strides
  SmallVector<int64_t> extendedStrides(
      extraIndicesSourceSize + extraIndicesDestSize, 1);
  llvm::append_range(extendedStrides, strides);

  auto copyLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{extendedStart, extendedStart},
      ArrayRef<Attribute>{copyFromView, copyToView},
      /*bounds=*/extendedBounds,
      /*strides=*/extendedStrides, false,
      /*useIndexDiffs=*/false);
  {
    PatternRewriter::InsertionGuard outerGuard(b);
    b.setInsertionPointToStart(copyLoop.getBody());
    Type loadType = vectorTypeOrSelf(elemType, vecLen);
    auto val = b.create<InBoundsLoadOp>(loc, loadType, rawLoadBuffer,
                                        copyLoop.getLowerCoords(0));
    b.create<InBoundsStoreOp>(loc, val, rawStoreBuffer,
                              copyLoop.getLowerCoords(1));
  }
  b.eraseOp(op);
  return success();
}

/// Add a length-1 iteration index to scalar buffers.
static Value addIterationIndexIfScalar(PatternRewriter &b, Location loc,
                                       Value buffer) {
  auto bufferType = cast<MemRefType>(buffer.getType());
  if (bufferType.getRank() != 0)
    return buffer;
  TopDownTMBuilder addZero(b, {"zero"}, {1}, loc);
  addZero.ignore("zero");
  TransformMapAttr addZeroAttr = addZero.get();
  Value withIter = b.create<TransformOp>(loc, buffer, addZeroAttr);
  return withIter;
}

LogicalResult ThreadwiseReadIntoRewritePattern::matchAndRewrite(
    ThreadwiseReadIntoOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  Value sourceView = adaptor.getSource();
  ArrayAttr extraViews = op.getExtraViews();
  sourceView =
      cast<TypedValue<MemRefType>>(transform(b, sourceView, extraViews));
  sourceView = addIterationIndexIfScalar(b, loc, sourceView);
  sourceView = isolateTransforms(b, sourceView);
  auto sourceViewType = cast<MemRefType>(sourceView.getType());
  Value dest = adaptor.getDest();
  MemRefType dstBufferType = dest.getType().cast<MemRefType>();

  int64_t numValues = dstBufferType.getNumElements();

  bool isSrcVectorBuffer = sourceViewType.getElementType().isa<VectorType>();
  bool isDstVectorBuffer = dstBufferType.getElementType().isa<VectorType>();

  size_t extraIdxCount = op.getExtraIndices().size();
  // We are vectorizing in the iter dimension, not block ID or thread ID
  auto elementType = sourceViewType.getElementType();
  Type loadType;
  int64_t vectorSrcLen, vectorDstLen;
  int64_t srcStride;
  VectorType dstVectorType;

  if (isSrcVectorBuffer) {
    loadType = elementType;
    vectorSrcLen = elementType.dyn_cast<VectorType>().getNumElements();
    elementType = elementType.dyn_cast<VectorType>().getElementType();
    collapseContiguousMerges(sourceView);
    srcStride = 1;
    if (!isDstVectorBuffer) {
      numValues = numValues / vectorSrcLen;
    }
  } else {
    VectorizationResult vectorSrcRes = getMaxVectorization(
        sourceView, extraIdxCount, /*inputDimLen=*/numValues);
    vectorSrcLen = vectorSrcRes.max;
    // In the future, this might get merged into the vectorizer.
    collapseContiguousMerges(sourceView);
    srcStride = vectorSrcLen;
    loadType = vectorTypeOrSelf(elementType, vectorSrcLen);
  }

  if (isDstVectorBuffer) {
    dstVectorType = dstBufferType.getElementType().dyn_cast<VectorType>();
    vectorDstLen = dstVectorType.dyn_cast<VectorType>().getNumElements();
    numValues = numValues * vectorDstLen;
    if (isSrcVectorBuffer) {
      numValues = numValues / vectorSrcLen;
    }
  }

  auto [buffer, transforms, needs64BitIdx] = untransform(b, sourceView);
  // Unless specified it is assumed to be global
  auto srcBufferType = cast<MemRefType>(buffer.getType());
  gpu::AddressSpace srcAddrSpace = gpu::AddressSpace::Global;
  if (srcBufferType.getMemorySpace()) {
    srcAddrSpace =
        srcBufferType.getMemorySpace().cast<gpu::AddressSpaceAttr>().getValue();
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
        Value idx = loadLoop.getLowerCoords(/*domain=*/1)[extraIdxCount];
        // If the source is not a vector buffer, it could still be
        // vectorizable on the load. If that is the case, then
        // the stride would be vectorSrcLen. However, the rest
        // of the code here assumes indexing is done as if the
        // source memref has vector-typed elements. Thus, changing
        // the indexing to be of vector-typed elements.
        if (!isSrcVectorBuffer) {
          Value srcVecLenVal =
              b.createOrFold<arith::ConstantIndexOp>(loc, vectorSrcLen);
          idx = b.createOrFold<arith::DivUIOp>(loc, idx, srcVecLenVal);
        }
        if (vectorSrcLen == vectorDstLen) {
          b.create<memref::StoreOp>(loc, ifb.getResult(0), dest, idx);
        } else {
          // When the vector types differ, we need to find the gcd
          // to make it work for the both source and dest.
          int64_t commonVecLen = math_util::gcd(vectorSrcLen, vectorDstLen);
          Type commonVecType = VectorType::get({commonVecLen}, elementType);
          int64_t numStores = vectorSrcLen / commonVecLen;
          for (int64_t i = 0; i < numStores; ++i) {
            Value loadSliceStart =
                b.createOrFold<arith::ConstantIndexOp>(loc, commonVecLen * i);
            Value value = ifb.getResult(0);
            // Only need to perform slice extraction of vector-typed sources.
            if (vectorSrcLen > 1) {
              value = b.create<ExtractSliceOp>(loc, commonVecType, value,
                                               loadSliceStart);
            }
            // Calculate base element offsets
            Value baseElementOffset = b.createOrFold<arith::MulIOp>(
                loc, idx,
                b.createOrFold<arith::ConstantIndexOp>(loc, vectorSrcLen));
            Value elementOffset = b.createOrFold<arith::AddIOp>(
                loc, baseElementOffset,
                b.createOrFold<arith::ConstantIndexOp>(loc, i * commonVecLen));

            // Base element offset is used to figure out correct dest vector idx
            // and slice idx within the dest vector.
            Value storeVecStart = b.createOrFold<arith::DivUIOp>(
                loc, elementOffset,
                b.createOrFold<arith::ConstantIndexOp>(loc, vectorDstLen));
            Value storeVec = b.create<memref::LoadOp>(
                loc, dstVectorType, dest, ValueRange{storeVecStart});
            Value storeSliceStart = b.createOrFold<arith::RemUIOp>(
                loc, elementOffset,
                b.createOrFold<arith::ConstantIndexOp>(loc, vectorDstLen));
            Value newStoreVec = b.create<InsertSliceOp>(
                loc, dstVectorType, value, storeVec, storeSliceStart);
            b.create<memref::StoreOp>(loc, newStoreVec, dest,
                                      ValueRange{storeVecStart});
          }
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

  Value source = adaptor.getSource();
  Value destView = adaptor.getDest();

  ArrayAttr extraViews = op.getExtraViews();
  destView = transform(b, destView, extraViews);
  destView = addIterationIndexIfScalar(b, loc, destView);
  destView = isolateTransforms(b, destView);
  auto destViewType = cast<MemRefType>(destView.getType());
  ArrayRef<int64_t> outputShape = destViewType.getShape();
  size_t extraIdxCount = op.getExtraIndices().size();

  int64_t iterLen = outputShape[extraIdxCount];
  auto destElemType = destViewType.getElementType();
  // We are vectorizing in the iter dimension, not block ID or thread ID
  int64_t vectorLen;
  Type elementType = destElemType;
  // If the dest is already being viewed as vector-typed, there's no good
  // mechanism to, for example, store a vector<8xf32> as two consecutive
  // vector<4xf32>s, and there's unlikely to be any performance benefit from
  // doing so. Therefore, don't bother. This currently implicitly assumes
  // a scalarized view on top of the destination buffer, which'll be cleaned up
  // in the future.
  if (auto elemVecType = destElemType.dyn_cast<VectorType>()) {
    vectorLen = elemVecType.getNumElements();
    elementType = elemVecType.getElementType();
  } else {
    VectorizationResult vectorRes =
        getMaxVectorization(destView, extraIdxCount);
    vectorLen = vectorRes.max;
  }
  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for write_all = " << vectorLen
                          << "\n");

  collapseContiguousMerges(destView);
  auto [buffer, transforms, needs64BitIdx] = untransform(b, destView);
  MemRefType dstBufferType = buffer.getType().cast<MemRefType>();

  // Unless specified it is assumed to be global
  gpu::AddressSpace dstAddrSpace = gpu::AddressSpace::Global;
  if (dstBufferType.getMemorySpace()) {
    dstAddrSpace =
        dstBufferType.getMemorySpace().cast<gpu::AddressSpaceAttr>().getValue();
  }

  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

  SmallVector<Value, 3> writeStartCoords =
      llvm::to_vector<3>(adaptor.getExtraIndices());
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
                              /*canStoreOffEnd=*/nullptr,
                              /*nontemporal=*/b.getBoolAttr(false));
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

void RockThreadwiseGemmLoweringPass::runOnOperation() {
  func::FuncOp op = getOperation();
  MLIRContext *ctx = &getContext();
  {
    ConversionTarget writeAllTarget(*ctx);
    writeAllTarget.addIllegalOp<ThreadwiseReadIntoOp, ThreadwiseWriteAllOp,
                                ThreadwiseCopyOp>();
    writeAllTarget.addLegalDialect<
        arith::ArithDialect, rock::RockDialect, memref::MemRefDialect,
        scf::SCFDialect, vector::VectorDialect, affine::AffineDialect>();
    writeAllTarget.addLegalOp<gpu::PrintfOp>();
    RewritePatternSet writeAllPatterns(ctx);
    writeAllPatterns
        .add<ThreadwiseReadIntoRewritePattern, ThreadwiseWriteAllRewritePattern,
             ThreadwiseCopyRewritePattern>(ctx);
    if (failed(applyPartialConversion(getOperation(), writeAllTarget,
                                      std::move(writeAllPatterns))))
      signalPassFailure();
  }

  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::ThreadwiseGemmOp, rock::ThreadwiseAccelGemmOp>();
  target.addLegalDialect<amdgpu::AMDGPUDialect, arith::ArithDialect,
                         rock::RockDialect, affine::AffineDialect,
                         memref::MemRefDialect, vector::VectorDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<ThreadwiseGemmRewritePattern, ThreadwiseAccelGemmRewritePattern>(
      ctx);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return signalPassFailure();
}

} // end anonymous namespace
