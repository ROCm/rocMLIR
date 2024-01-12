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
                                 ArrayRef<int64_t> startShape,
                                 ArrayRef<StringRef> names,
                                 ArrayRef<int64_t> sizes, bool ignoreStartShape,
                                 DenseSet<StringRef> ignoreNames) const {
    // Create the full view [startShape, sizes]
    SmallVector<int64_t> normalizedShape(startShape);
    llvm::append_range(normalizedShape, sizes);

    // Create names for the start shape
    SmallVector<SmallString<8>> normalizedNames;
    SmallVector<StringRef> normalizedNamesRef;
    for (size_t i = 0; i < startShape.size(); i++) {
      SmallString<8> normName(Twine("extra_" + Twine(i)).str());
      normalizedNames.push_back(normName);
      normalizedNamesRef.push_back(normalizedNames.back());
      // If we are ignoring the start names, add them into the ignoreNames
      // set
      if (ignoreStartShape)
        ignoreNames.insert(normalizedNames.back());
    }

    // Add the other names in
    llvm::append_range(normalizedNamesRef, names);

    unsigned pos = 0;
    unsigned newPos = 0;
    TopDownTMBuilder td(b, normalizedNamesRef, normalizedShape, loc);
    // Convert the normalizedView to a real view by ignoring
    // the names contained in `ignoreNames` and letting the rest pass through
    for (pos = 0; pos < normalizedNamesRef.size(); pos++) {
      if (ignoreNames.contains(normalizedNamesRef[pos])) {
        td.ignore(normalizedNamesRef[pos]);
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

    size_t extraIndicesCSize = op.getExtraIndicesC().size();
    SmallVector<int64_t> extraIndicesCShape;
    for (size_t i = 0; i < extraIndicesCSize; i++) {
      extraIndicesCShape.push_back(bufferCShape[i]);
    }

    auto emitter = rock::accel::AccelEmitter::select(
        op.getFeatures(), dataTypeA, dataTypeB, op.getArch(), tuningParams);

    // Extract relevant accel emitter parameters
    rock::accel::AccelEmitterParams params = emitter->getParams();
    Type argTypeA = params.argTypeA;
    Type argTypeB = params.argTypeB;

    if (!emitter)
      return emitError(loc)
             << "Failed to select any accelerator instruction.\n";

    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);
    SmallVector<Value, 4> startCoords(4, zeroConstantOp);

    // Sizes of the [i,j,k] axis
    int64_t iLen = *(bufferCShape.end() - 2);
    int64_t jLen = bufferCShape.back();
    int64_t kLen = bufferAShape.back();

    // All the view need to be `[extraIndices, i, j, k]` so that
    // we can iterate within the `transforming_for` later. This means that we
    // want to `ignore` some of the indices, e.g, since A is [i,k] we will
    // ignore index `j` and the `extraIndices`
    TransformMapAttr normalizedViewA =
        normalizeView(b, loc, extraIndicesCShape, {"i", "j", "k"},
                      {iLen, jLen, kLen}, true, {"j"});
    TransformMapAttr normalizedViewB =
        normalizeView(b, loc, extraIndicesCShape, {"i", "j", "k"},
                      {iLen, jLen, kLen}, true, {"i"});
    TransformMapAttr normalizedViewC =
        normalizeView(b, loc, extraIndicesCShape, {"i", "j", "k"},
                      {iLen, jLen, kLen}, false, {"k"});

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
    auto extendedBounds = SmallVector<int64_t>(extraIndicesCSize, 1);
    llvm::append_range(extendedBounds, ArrayRef<int64_t>{iLen, jLen, kLen});
    auto extendedStride = SmallVector<int64_t>(extendedBounds.size(), 1);
    auto extendedStart = llvm::to_vector(op.getExtraIndicesC());
    llvm::append_range(
        extendedStart,
        ArrayRef<Value>{zeroConstantOp, zeroConstantOp, zeroConstantOp});

    // Emit the loop
    auto accelLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{extendedStart, extendedStart, extendedStart},
        ArrayRef<Attribute>{bufferViewA, bufferViewB, bufferViewC},
        /*bounds=*/extendedBounds,
        /*strides=*/extendedStride,
        /*forceUnroll=*/false, /*useIndexDiffs=*/false);
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
  auto sourceView = cast<TypedValue<MemRefType>>(adaptor.getSource());
  auto destView = cast<TypedValue<MemRefType>>(adaptor.getDest());
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

  auto [rawLoadBuffer, loadBufferView, sourceNeeds64BitIdx] =
      untransform(b, sourceView);
  auto [rawStoreBuffer, storeBufferView, dstNeeds64BitIdx] =
      untransform(b, destView);

  // We need to normalize the maps. I.e., if i0,...,iD-1 are the extra
  // destination indices and j0, ..., jS-1 are the extra source indices, both
  // source and destination views need to be [(i0, ..., iD-1), (j0, ..., jS-1),
  // ..., (other dest/source transform indices)]. It is important that the extra
  // dimensions are propagated top to bottom, otherwise the transformation stack
  // might loose invertibility
  loadBufferView = addPassThroughIndices(
      b, loadBufferView, extraIndicesDestShape, extraIndicesSourceSize);
  storeBufferView =
      addPassThroughIndices(b, storeBufferView, extraIndicesSourceShape, 0);

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

  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  Type elemType = sourceView.getType().cast<MemRefType>().getElementType();

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
  // vectorization length
  auto storeBufferViewInverted = invertTransforms(b, loc, storeBufferView);
  if (storeBufferViewInverted) {
    auto srcToDstView =
        prependUpperViews(b, storeBufferViewInverted, loadBufferView);
    int64_t maxVlen = 128 / elemType.getIntOrFloatBitWidth();
    auto shape = srcToDstView[0].cast<TransformMapAttr>().getUpperBounds();
    vecLen = getMaxVectorizationForDatatype(
        srcToDstView, /*dim=*/extraIndicesSourceSize + extraIndicesDestSize,
        maxVlen, shape, elemType);
    copyFromView = collapseContiguousMerges(srcToDstView, rawStoreBufferShape);
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

    // Load coords: get rid of the extra destination indices. The lower source
    // buffer is in the form of [sourceIndices*,extraDestIndices,rawIndex].
    auto loadCoords = SmallVector<Value>(copyLoop.getLowerCoords(0).begin(),
                                         copyLoop.getLowerCoords(0).end() - 1 -
                                             extraIndicesDestSize);
    loadCoords.push_back(copyLoop.getLowerCoords(0).back());

    // Store coords: get rid of the extra source indices. The lower dest buffer
    // is in the form of [extraSourceIndices,destIndices*,rawIndex].
    auto storeCoords = SmallVector<Value>(copyLoop.getLowerCoords(1).begin() +
                                              extraIndicesSourceSize,
                                          copyLoop.getLowerCoords(1).end() - 1);
    storeCoords.push_back(copyLoop.getLowerCoords(1).back());

    auto val =
        b.create<InBoundsLoadOp>(loc, loadType, rawLoadBuffer, loadCoords);
    b.create<InBoundsStoreOp>(loc, val, rawStoreBuffer, storeCoords);
  }
  b.eraseOp(op);
  return success();
}

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
    // transforms = collapseContiguousMerges(transforms, bufferShape);
    srcStride = vectorSrcLen;
    loadType = vectorTypeOrSelf(elementType, vectorSrcLen);
  }

  if (isDstVectorBuffer) {
    dstVectorType = dstBufferType.getElementType().dyn_cast<VectorType>();
    vectorDstLen = dstVectorType.dyn_cast<VectorType>().getNumElements();
    numValues = (numValues * vectorDstLen);
    if(isSrcVectorBuffer){
      numValues = numValues / vectorSrcLen;
    }
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
        Value idx = loadLoop.getLowerCoords(/*domain=*/1)[extraIdxCount];
        if(!isSrcVectorBuffer){
          Value srcVecLenVal = b.createOrFold<arith::ConstantIndexOp>(loc, vectorSrcLen);
          idx = b.createOrFold<arith::DivUIOp>(loc, idx, srcVecLenVal);
        }
        // Destination is a vector buffer
        if (vectorSrcLen == vectorDstLen) {
          b.create<memref::StoreOp>(loc, ifb.getResult(0), dest, idx);
        } else if (vectorSrcLen > vectorDstLen) {
          int64_t numStores = vectorSrcLen / vectorDstLen;
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
