//===- GridwiseGemmToBlockwise - MLIR Rock ops lowering passes -----===//
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
// This pass converts rock.gridwise_gemm[_v2] into block- and threadwise ops
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/IR/XdlopsCodeSelection.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGRIDWISEGEMMTOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gridwise-to-blockwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockGridwiseGemmToBlockwisePass
    : public rock::impl::RockGridwiseGemmToBlockwisePassBase<
          RockGridwiseGemmToBlockwisePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// TODO(kdrewnia): Could rank-0 vectors clear some of this up?
// Utility function for crafting optional vector types
static Type vectorTypeOrSelf(Type elementType, int64_t len) {
  if (len == 1)
    return elementType;
  return VectorType::get({len}, elementType);
}

//===----------------------------------------------------------------------===//
// Utility function to determine the type to be loaded
//===----------------------------------------------------------------------===//
static void computeLoadStoreTypeInfo(OpBuilder &b,
                                     ArrayRef<int64_t> sliceLengths,
                                     int64_t loadLength, uint32_t &vectorDim,
                                     int64_t kPack, Type elementType,
                                     Type &loadType, Type &intermediateType) {

  // In case KPack and vector load is used, and we vector load on GemmK
  // dimension (1), use the last dimension (GemmKPack) instead.
  if ((loadLength > 1) && (kPack > 1) && (vectorDim == 1)) {
    vectorDim = sliceLengths.size() - 1;
  }

  int64_t itemsToCopy = 1;
  for (int64_t len : sliceLengths) {
    itemsToCopy *= len;
  }
  intermediateType = vectorTypeOrSelf(elementType, itemsToCopy);
  loadType = vectorTypeOrSelf(elementType, loadLength);
}

static Type obtainAccumulatorType(OpBuilder &b, Type &elementType,
                                  Type &destType) {
  // Determine the type used on VGPR to act as accumulator.
  // f32: f32.
  // f16, bf16: f32 to prevent overflow from happening.
  // i16 : i16.
  // i8: i32, since we have an i32 output
  Type accumulatorType = destType;
  if (elementType.isF16() || elementType.isBF16()) {
    accumulatorType = b.getF32Type();
  } else if (elementType.isInteger(8)) {
    accumulatorType = b.getI32Type();
  }
  return accumulatorType;
}

// Create a transformation domain that computes the linear, row-major iteration
// index over a rectangular space with dimensions `sliceLengths`.
// The iteration starts at all-zeros
static ArrayAttr makeLinearDomain(OpBuilder &b, Location loc,
                                  ArrayRef<int64_t> sliceLengths) {
  size_t nDims = sliceLengths.size();
  SmallVector<SmallString<4>, 5> dimNames;
  dimNames.reserve(nDims);
  SmallVector<int64_t> strides;
  strides.reserve(nDims);
  int64_t stride = 1;
  for (size_t e = sliceLengths.size(), i = e - 1; i < e; --i) {
    strides.push_back(stride);
    stride *= sliceLengths[i];
    SmallString<4> dimName;
    ("dim" + Twine(i)).toVector(dimName);
    dimNames.push_back(std::move(dimName));
  }
  std::reverse(dimNames.begin(), dimNames.end());
  std::reverse(strides.begin(), strides.end());

  SmallVector<StringRef, 5> dimNameRefs;
  dimNameRefs.reserve(nDims);
  llvm::copy(dimNames, std::back_inserter(dimNameRefs));
  TopDownTMBuilder builder(b, dimNameRefs, sliceLengths, loc);
  builder.embed("iter", 0, stride, dimNameRefs, strides);
  TransformMapAttr ret = builder.get();
  return b.getArrayAttr(ret);
}

//===----------------------------------------------------------------------===//
// Building load/store loops
//===----------------------------------------------------------------------===//
static TransformingForOp
createGlobalLoadLoop(OpBuilder &b, Location loc, Value global,
                     ValueRange globalStart, Type resultType, Type loadType,
                     ArrayRef<int64_t> sliceLengths, uint32_t vectorDim,
                     bool useIndexDiffs) {
  bool fullyScalar = !resultType.isa<ShapedType>();
  int64_t loadLength = 1;
  if (auto loadVectorType = loadType.dyn_cast<VectorType>())
    loadLength = loadVectorType.getNumElements();

  size_t nUpper = globalStart.size();
  bool complexVectorLoad = (loadLength > 1) && (vectorDim != nUpper - 1);

  Value zero = b.createOrFold<ConstantIndexOp>(loc, 0);
  SmallVector<Value, 5> linearInit(nUpper, zero);

  ArrayAttr globalTransforms;
  std::tie(global, globalTransforms) = untransform(b, global);

  ArrayAttr leftOobDims, rightOobDims;
  std::tie(leftOobDims, rightOobDims) =
      computeOobFromTransforms(b, globalTransforms);

  ArrayAttr noTransforms = b.getArrayAttr({});
  ArrayAttr resultIdxMap = makeLinearDomain(b, loc, sliceLengths);

  SmallVector<int64_t, 4> loopBounds(sliceLengths.begin(), sliceLengths.end());
  assert(loopBounds[vectorDim] % loadLength == 0 && "Uneven vector load");
  loopBounds[vectorDim] /= loadLength;

  SmallVector<Attribute> loopTransforms = {globalTransforms, resultIdxMap};
  if (complexVectorLoad)
    loopTransforms[1] = noTransforms;

  Value dest = createZeroConstantOp(b, loc, resultType);
  auto loop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{globalStart, linearInit}, loopTransforms,
      loopBounds, /*strides=*/llvm::None,
      /*forceUnroll=*/true, useIndexDiffs, dest);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(loop.getBody());
  Value loaded =
      b.create<GlobalLoadOp>(loc, loadType, global, leftOobDims, rightOobDims,
                             loop.getLowerCoords(/*domain=*/0));
  Value toYield = loaded;
  if (!fullyScalar) {
    Value loopArg = loop.getIterArgs()[0];
    if (complexVectorLoad) {
      // The results of a vectorized load are not necessarily in the order
      // they'll be stored in. Account for that here with an inner loop that
      // spreads out the loaded elements to appropriate indices. If the
      // vectorization dimension is the fastest-moving dimension of the loop, we
      // don't need to do this
      SmallVector<int64_t, 4> vectorIdxBounds(nUpper, 1);
      vectorIdxBounds[vectorDim] = loadLength;
      ArrayAttr loadedValIdxMap = makeLinearDomain(b, loc, vectorIdxBounds);

      TransformingForOp scatterLoop = b.create<TransformingForOp>(
          loc,
          ArrayRef<ValueRange>{linearInit, loop.getLowerCoords(/*domain=*/1)},
          ArrayRef<Attribute>{loadedValIdxMap, resultIdxMap}, vectorIdxBounds,
          /*strides=*/llvm::None, /*forceUnroll=*/true, /*useIndexDiffs=*/true,
          loopArg);

      {
        OpBuilder::InsertionGuard innerGuard(b);
        b.setInsertionPointToStart(scatterLoop.getBody());
        Value toScatter = b.create<vector::ExtractElementOp>(
            loc, loaded, scatterLoop.getLowerCoords(/*domain=*/0)[0]);
        Value toYieldInner = b.create<vector::InsertElementOp>(
            loc, toScatter, scatterLoop.getIterArgs()[0],
            scatterLoop.getLowerCoords(/*domain=*/1)[0]);
        b.create<rock::YieldOp>(loc, toYieldInner);
      }
      toYield = scatterLoop.getResults()[0];
    } else {
      toYield = b.create<InsertSliceOp>(loc, resultType, loaded, loopArg,
                                        loop.getLowerCoords(/*domain=*/1)[0]);
    }
  }
  b.create<rock::YieldOp>(loc, toYield);
  return loop;
}

static TransformingForOp createLdsStoreLoop(OpBuilder &b, Location loc,
                                            Value loaded, Value buffer,
                                            ValueRange bufferStart,
                                            ArrayRef<int64_t> sliceLengths) {
  Type loadedType = loaded.getType();
  Type elementType = loadedType;
  if (auto loadedVector = loadedType.dyn_cast<ShapedType>())
    elementType = loadedVector.getElementType();
  bool fullyScalar = (loadedType == elementType);

  size_t nUpper = bufferStart.size();

  Value zero = b.createOrFold<ConstantIndexOp>(loc, 0);
  SmallVector<Value, 5> linearInit(nUpper, zero);

  ArrayAttr bufferTransforms;
  std::tie(buffer, bufferTransforms) = untransform(b, buffer);
  ArrayAttr resultIdxMap = makeLinearDomain(b, loc, sliceLengths);

  SmallVector<int64_t, 4> loopBounds(sliceLengths.begin(), sliceLengths.end());

  SmallVector<Attribute> loopTransforms = {resultIdxMap, bufferTransforms};

  auto loop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{linearInit, bufferStart}, loopTransforms,
      loopBounds,
      /*strides=*/llvm::None, /*forceUnroll=*/true, /*useIndexDiffs=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    if (fullyScalar) {
      b.create<InBoundsStoreOp>(loc, loaded, buffer,
                                loop.getLowerCoords(/*domain=*/1));
    } else {
      Value toStore = b.create<vector::ExtractElementOp>(
          loc, loaded, loop.getLowerCoords(/*domain=*/0)[0]);
      b.create<InBoundsStoreOp>(loc, toStore, buffer,
                                loop.getLowerCoords(/*domain=*/1));
    }
  }
  return loop;
}

/// Temporary function to remove the kPack transformation currently being
/// introduced in conv-to-gemm. kPack is a LDS-specific transformation and
/// should not have been applied to the gridwise gemm argument. Future work will
/// make the conversion stop doing that, and we'll be able to get rid of this.
static Value unKpack(OpBuilder &b, Value matrix) {
  ArrayRef<int64_t> shape = matrix.getType().cast<MemRefType>().getShape();
  if (shape.size() != 4)
    return matrix;
  BottomUpTMBuilder unkpack(b, {"g", "k", "d", "kpack"}, shape,
                            matrix.getLoc());
  unkpack.passThrough({"g", "d"});
  unkpack.merge("k", 1, {"k", "kpack"});
  TransformMapAttr unkpackAttr = unkpack.get();
  return b.create<TransformOp>(matrix.getLoc(), matrix, unkpackAttr);
}

/// Given a G x K x D matrix and the block tuning parameters and how much data
/// each thread will load,
/// return the dimension in which the load of this matrix from global memory
/// should be vectorized and the length of that vector load. Also takes
/// `tiebreaker`, the vectorization dimension to be used when both choises are
/// equal, which should be the vectorization dimension of the store to LDS.
static std::pair<GemmDimension, int64_t>
bestVectorization(OpBuilder &b, Value matrix, int64_t dataPerThread,
                  GemmDimension tiebreaker) {
  Value tensor;
  ArrayAttr transforms;
  std::tie(tensor, transforms) = untransform(b, matrix);
  ArrayRef<int64_t> tensorShape =
      tensor.getType().cast<MemRefType>().getShape();
  int64_t kVectorLen =
      getMaxVectorization(transforms, static_cast<uint32_t>(GemmDimension::K),
                          dataPerThread, tensorShape);
  int64_t dVectorLen = getMaxVectorization(
      transforms, static_cast<uint32_t>(GemmDimension::MorN), dataPerThread,
      tensorShape);
  if (kVectorLen > dVectorLen)
    return {GemmDimension::K, kVectorLen};
  if (dVectorLen > kVectorLen)
    return {GemmDimension::MorN, dVectorLen};
  return {tiebreaker, kVectorLen};
}

/// Applies the transforms that take a G x K x D matrix to a k_iter x bid x tid
/// x iter value suitable for using in a global load loop. `dName` should be "m"
/// and "n", and is used to make the maps have the right names for debugging.
///
/// bidGridOrder should
/// contain the strings "g_block", "m_block", and "n_block" in some order
/// indicating how the block ID is to be partitioned into offsets (last element
/// moves fastest) and bidGridLengths should be the lengths of those three
/// dimensions. This is needed because the xdlops and non-xdlops gemms partition
/// their block ID in different orders.
static FailureOr<Value> wrapMatrixForGlobalLoad(
    OpBuilder &b, Location loc, Value matrix, StringRef dName,
    ArrayRef<StringRef> bidGridOrder, ArrayRef<int64_t> bidGridLengths,
    int64_t gridSize, int64_t blockSize, int64_t kPerBlock, int64_t dPerBlock,
    int64_t kPerThread, int64_t dPerThread, GemmDimension vectorDim) {

  if (dName != "m" && dName != "n") {
    return emitError(loc, "expected dName to be m or n but got " + dName);
  }
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  StringRef otherBlockDim = dName == "m" ? "n_block" : "m_block";

  MemRefType matrixType = matrix.getType().cast<MemRefType>();
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
  int64_t kGlobal = matrixShape[1];
  int64_t dGlobal = matrixShape[2];

  int64_t kIters = kGlobal / kPerBlock;
  int64_t dataPerThread = (kPerBlock * dPerBlock) / blockSize;

  SmallString<8> dIterName = llvm::formatv("{0}_iter", dName);
  SmallString<8> dThreadName = llvm::formatv("{0}_thread", dName);

  // Note: (kThreads * dThreads) = (kPerBlock * dPerBlock) / dataPerThread) =
  // blockSize
  int64_t kThreads = kPerBlock / kPerThread;
  int64_t dThreads = dPerBlock / dPerThread;

  TopDownTMBuilder splitId(b, {"k_loop", "bid", "tid", "iter"},
                           {kIters, gridSize, blockSize, dataPerThread}, loc);
  splitId.passThrough("k_loop");
  splitId.merge(bidGridOrder, {1, 2, 3}, "bid", bidGridLengths);
  // That threads are grouped [other dim, k] is important: it menas we can
  // ignore kPack here but then account for it when writing to LDS.
  splitId.merge({dThreadName, "k_thread"}, {4, 5}, "tid", {dThreads, kThreads});
  if (vectorDim == GemmDimension::K) {
    splitId.merge({dIterName, "k_iter"}, {6, 7}, "iter",
                  {dPerThread, kPerThread});
  } else {
    splitId.merge({"k_iter", dIterName}, {6, 7}, "iter",
                  {kPerThread, dPerThread});
  }
  TransformMapAttr splitIdAttr = splitId.get();

  auto toGlobalIdx = TopDownTMBuilder::below(splitId, splitIdAttr);
  toGlobalIdx.passThrough({"g"}, {0}, {"g_block"});
  toGlobalIdx.unmerge("k", 1, {"k_loop", "k_thread", "k_iter"},
                      {kGlobal / kPerBlock, kThreads, kPerThread});
  toGlobalIdx.unmerge(dName, 2, {thisBlockDim, dThreadName, dIterName},
                      {dGlobal / dPerBlock, dThreads, dPerThread});
  toGlobalIdx.ignore(otherBlockDim);
  TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();

  Value intermediate = b.create<TransformOp>(loc, matrix, toGlobalIdxAttr);
  Value transformed = b.create<TransformOp>(loc, intermediate, splitIdAttr);
  return transformed;
}

/// Wraps the LDS buffer "buffer", which is K x D x KPack, into a
/// tid x iter view.
static FailureOr<Value> wrapLDSBufferForStore(OpBuilder &b, Location loc,
                                              Value buffer, StringRef dName,
                                              int64_t kPerThread,
                                              int64_t dPerThread) {
  MemRefType bufferType = buffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> bufferShape = bufferType.getShape();
  if (bufferShape.size() != 3)
    return emitError(loc, "Expected a kOuter x d x kpack LDS  buffer");

  int64_t kOuter = bufferShape[0];
  int64_t d = bufferShape[1];
  int64_t kpack = bufferShape[2];

  int64_t kpackPerThread = std::min(kPerThread, kpack);
  int64_t kOuterPerThread = kPerThread / kpackPerThread;

  SmallString<8> dThreadName = llvm::formatv("{0}_thread", dName);
  SmallString<8> dIterName = llvm::formatv("{0}_iter", dName);
  BottomUpTMBuilder tidIterSplit(b, {"k_outer", dName, "kpack"}, bufferShape,
                                 loc);
  tidIterSplit.unmerge({"k_thread", "k_iter"}, {0, 1}, "k_outer",
                       {kOuter / kOuterPerThread, kOuterPerThread});
  tidIterSplit.unmerge({dThreadName, dIterName}, {2, 3}, dName,
                       {d / dPerThread, dPerThread});
  tidIterSplit.unmerge({"kpack_thread", "kpack_iter"}, {4, 5}, "kpack",
                       {kpack / kpackPerThread, kpackPerThread});
  TransformMapAttr tidIterSplitAttr = tidIterSplit.get();
  Value withTidIterSplit = b.create<TransformOp>(loc, buffer, tidIterSplitAttr);

  // Note: the fact that the global load groups the data each threads loads by
  // k and then by d means that we can smath the k and kpack thread IDs together
  // without any trouble.
  auto tidIter = BottomUpTMBuilder::above(tidIterSplit, tidIterSplitAttr);
  tidIter.merge("tid", 0, {dThreadName, "k_thread", "kpack_thread"});
  tidIter.merge("iter", 1, {"k_iter", dIterName, "kpack_iter"});
  TransformMapAttr tidIterAttr = tidIter.get();
  Value transformed = b.create<TransformOp>(loc, withTidIterSplit, tidIterAttr);
  return transformed;
}

/// Returns the map from (kOuter, bid, tid, iter) to indices in the vector
/// of values loaded from global memory.
static ArrayAttr globalVectorLayout(OpBuilder &b, Location loc, StringRef dName,
                                    int64_t kPerThread, int64_t dPerThread,
                                    int64_t kpack, GemmDimension vectorDim) {
  int64_t kOuter = kPerThread / std::min(kPerThread, kpack);
  int64_t dataPerThread = kPerThread * dPerThread;

  TopDownTMBuilder splitIter(b, {"iter"}, {dataPerThread});
  if (vectorDim == GemmDimension::K)
    splitIter.merge({dName, "k", "kpack"}, {0, 1, 2}, "iter",
                    {dPerThread, kOuter, kpack});
  else
    splitIter.merge({"k", "kpack", dName}, {0, 1, 2}, "iter",
                    {kOuter, kpack, dPerThread});
  TransformMapAttr splitIterAttr = splitIter.get();

  auto toVector = TopDownTMBuilder::below(splitIter, splitIterAttr);
  toVector.unmerge("raw", 0, {"k", dName, "kpack"},
                   {kOuter, dPerThread, kpack});
  TransformMapAttr toVectorAttr = toVector.get();
  return b.getArrayAttr({splitIterAttr, toVectorAttr});
}

/// Returns the map from (tid, iter) to indices the vector of values that will
/// be stored into LDS.
static ArrayAttr ldsVectorLayout(OpBuilder &b, Location loc,
                                 int64_t dataPerThread) {
  TopDownTMBuilder ignoreTid(b, {"tid", "iter"}, {1, dataPerThread}, loc);
  ignoreTid.ignore("tid");
  ignoreTid.passThrough({"raw"}, {0}, {"iter"});
  TransformMapAttr ignoreTidAttr = ignoreTid.get();
  return b.getArrayAttr({ignoreTidAttr});
}

static TransformingForOp
createGlobalLoadLoop(PatternRewriter &b, Location loc, Value wrappedMatrix,
                     ArrayAttr vectorMap, int64_t dataPerThread,
                     int64_t vectorLen, Value bid, Value tid) {
  Value tensor;
  ArrayAttr matrixToTensor;
  std::tie(tensor, matrixToTensor) = untransform(b, wrappedMatrix);

  ArrayAttr leftOobDims, rightOobDims;
  std::tie(leftOobDims, rightOobDims) =
      computeOobFromTransforms(b, matrixToTensor);

  Type elementType =
      wrappedMatrix.getType().cast<MemRefType>().getElementType();
  Type loadType = vectorTypeOrSelf(elementType, vectorLen);
  Type resultType = vectorTypeOrSelf(elementType, dataPerThread);

  Value resultInit = createZeroConstantOp(b, loc, resultType);
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

  SmallVector<Value, 4> globalStart = {zero, bid, tid, zero};
  SmallVector<Value, 4> vectorStartOuter(4, zero);
  auto outerLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{globalStart, vectorStartOuter},
      ArrayRef<Attribute>{matrixToTensor, b.getArrayAttr({})},
      /*bounds=*/ArrayRef<int64_t>{1, 1, 1, dataPerThread},
      /*strides=*/ArrayRef<int64_t>{1, 1, 1, vectorLen},
      /*forceUnroll=*/true, /*useIndexDiffs=*/true, resultInit);
  {
    PatternRewriter::InsertionGuard outerGuard(b);
    b.setInsertionPointToEnd(outerLoop.getBody());
    Value loaded =
        b.create<GlobalLoadOp>(loc, loadType, tensor, leftOobDims, rightOobDims,
                               outerLoop.getLowerCoords(/*domain=*/0));
    auto innerLoop = b.create<TransformingForOp>(
        loc,
        ArrayRef<ValueRange>{zero,
                             outerLoop.getLowerCoords(/*domain=*/1).back()},
        ArrayRef<Attribute>{b.getArrayAttr({}), vectorMap},
        /*bounds=*/ArrayRef<int64_t>{vectorLen},
        /*strides=*/ArrayRef<int64_t>{1},
        /*forceUnroll=*/true, /*useIndexDiffs=*/true,
        outerLoop.getIterArgs()[0]);
    {
      PatternRewriter::InsertionGuard innerGuard(b);
      b.setInsertionPointToEnd(innerLoop.getBody());
      Value loadElem =
          vectorLen == 1
              ? loaded
              : b.create<vector::ExtractElementOp>(
                    loc, loaded, innerLoop.getLowerCoords(/*domain=*/0)[0]);
      Value toYieldInner = dataPerThread == 1
                               ? loadElem
                               : b.create<vector::InsertElementOp>(
                                     loc, loadElem, innerLoop.getIterArgs()[0],
                                     innerLoop.getLowerCoords(/*domain=*/1)[0]);
      b.create<rock::YieldOp>(loc, toYieldInner);
    }
    b.create<rock::YieldOp>(loc, innerLoop.getResult(0));
  }
  return outerLoop;
}

static TransformingForOp createLdsStoreLoop(PatternRewriter &b, Location loc,
                                            Value loaded,
                                            ArrayAttr ldsVectorMap,
                                            Value wrappedBuffer,
                                            int64_t dataPerThread, Value tid) {
  Value rawBuffer;
  ArrayAttr bufferView;
  std::tie(rawBuffer, bufferView) = untransform(b, wrappedBuffer);

  int64_t ldsStoreVectorization =
      getMaxVectorization(bufferView, /*dim=*/1, dataPerThread,
                          rawBuffer.getType().cast<MemRefType>().getShape());
  Type loadedType = loaded.getType();
  Type elementType = loadedType;
  if (auto vectorLoadTy = loadedType.dyn_cast<VectorType>())
    elementType = vectorLoadTy.getElementType();
  Type storeType = vectorTypeOrSelf(elementType, ldsStoreVectorization);

  Value zero = b.createOrFold<ConstantIndexOp>(loc, 0);
  SmallVector<Value, 2> vecCoordInit(2, zero);
  SmallVector<Value, 2> ldsCoordInit = {tid, zero};

  auto loop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{vecCoordInit, ldsCoordInit},
      ArrayRef<Attribute>{ldsVectorMap, bufferView},
      /*bounds=*/ArrayRef<int64_t>{1, dataPerThread},
      /*strides=*/ArrayRef<int64_t>{1, ldsStoreVectorization},
      /*forceUnroll=*/true, /*useIndexDiffs=*/true);
  {
    PatternRewriter::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value toStore =
        dataPerThread == 1
            ? loaded
            : b.create<ExtractSliceOp>(loc, storeType, loaded,
                                       loop.getLowerCoords(/*domain=*/0)[0]);
    b.create<InBoundsStoreOp>(loc, toStore, rawBuffer,
                              loop.getLowerCoords(/*domain=*/1));
  }
  return loop;
}

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

/// Utility function for constructing a subview that slices a buffer as a
/// TransformOp
static Value sliceBufferSubview(OpBuilder &b, Location loc, Value buffer,
                                int64_t start, int64_t length) {
  auto bufferType = buffer.getType().cast<MemRefType>();
  assert(bufferType.getRank() == 1 && "Can't slice multidimensional buffer");
  ArrayRef<int64_t> shape = bufferType.getShape();

  int64_t end = start + length;
  BottomUpTMBuilder transform(b, {"buffer"}, shape, loc);
  transform.slice({"slice"}, {"buffer"}, {start}, {end});

  TransformMapAttr transformAttr = transform.get();
  Value subview = b.create<TransformOp>(loc, buffer, transformAttr);
  return subview;
}

namespace {
struct GridwiseGemmRewritePattern : public OpRewritePattern<GridwiseGemmOp> {
  using OpRewritePattern<GridwiseGemmOp>::OpRewritePattern;

  LogicalResult computeLDSBlockSizes(GridwiseGemmOp op, int64_t &a_block_space,
                                     int64_t &b_block_space,
                                     int64_t &block_space,
                                     int64_t KPack = 1) const {
    GeneralGemmParamsAttr tuningParams = op.getParams();
    int64_t ThreadGemmAThreadCopySrcDataPerRead_M =
        tuningParams.getMPerThread();
    int64_t ThreadGemmBThreadCopySrcDataPerRead_N =
        tuningParams.getNPerThread();

    int64_t max_lds_align =
        math_util::lcm(ThreadGemmAThreadCopySrcDataPerRead_M,
                       ThreadGemmBThreadCopySrcDataPerRead_N);

    int64_t KPerBlock = tuningParams.getKPerBlock();
    int64_t MPerBlock = tuningParams.getMPerBlock();
    int64_t NPerBlock = tuningParams.getNPerBlock();

    int64_t AlignedNPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(NPerBlock, max_lds_align);

    // A matrix in LDS memory, dst of blockwise copy
    //   be careful of LDS alignment
    // Original C++ logic:
    // constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
    //    Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});
    // constexpr index_t a_block_space =
    //    math_util::integer_least_multiple(a_k_m_block_desc.GetElementSpace(),
    //    max_lds_align);
    int64_t AlignedMPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(MPerBlock, max_lds_align);
    a_block_space = math_util::integer_least_multiple(
                        KPerBlock * AlignedMPerBlock, max_lds_align) *
                    KPack;

    // B matrix in LDS memory, dst of blockwise copy
    //   be careful of LDS alignment
    // Original C++ logic:
    // constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
    //    Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});
    // constexpr index_t b_block_space =
    //    math_util::integer_least_multiple(b_k_n_block_desc.GetElementSpace(),
    //    max_lds_align);
    b_block_space = math_util::integer_least_multiple(
                        KPerBlock * AlignedNPerBlock, max_lds_align) *
                    KPack;

    block_space = a_block_space + b_block_space;

    LLVM_DEBUG(llvm::dbgs() << "a_block_space: " << a_block_space << "\n");
    LLVM_DEBUG(llvm::dbgs() << "b_block_space: " << b_block_space << "\n");
    LLVM_DEBUG(llvm::dbgs() << "double_block_space: " << block_space << "\n\n");

    // TODO: adjust for data type and device
    if (block_space * sizeof(float) > 64 * 1024)
      return failure();

    return success();
  }

  LogicalResult matchAndRewrite(GridwiseGemmOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain data type.
    Type elementType = op.getB().getType().getElementType();
    Type destType = op.getC().getType().getElementType();
    Type accumulatorType = obtainAccumulatorType(b, elementType, destType);

    // Prepare some useful constants.
    Value zeroConstantFloatOp = createZeroConstantOp(b, loc, accumulatorType);
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = op.getA().getType().getShape();
    bShape = op.getB().getType().getShape();
    cShape = op.getC().getType().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t K = aShape[1];
    int64_t M = aShape[2];
    int64_t N = bShape[2];

    if (bShape[0] != G || cShape[0] != G) {
      return op.emitOpError("Mismatched G dimensions in matrix multiply;")
             << " A[0] = " << G << " b[0] = " << bShape[0]
             << " C[0] = " << cShape[0];
    }
    if (cShape[1] != M) {
      return op.emitOpError("Mismatched M dimensions in matrix multiply:")
             << " A[2] = " << M << " C[1] = " << cShape[1];
    }
    if (bShape[1] != K) {
      return op.emitOpError("Mismatched K dimensions in matrix multiply:")
             << " A[1] = " << K << " B[1] = " << bShape[1];
    }

    if (cShape[2] != N) {
      return op.emitOpError("Mismatched N dimensions in matrix multiply:")
             << " B[2] = " << N << " C[2] = " << cShape[2];
    }

    // Obtain critical tuning parameters.
    uint32_t blockSize = op.getBlockSize();
    uint32_t gridSize = op.getGridSize();
    GeneralGemmParamsAttr tuningParams = op.getParams();
    int64_t kpack = tuningParams.getKpack();
    // TODO: kPerBlock, as defined in parameter selection etc,
    // is in units of kPack, not individual k. This should be changed
    // at some future point, but it'll be worked around for now.
    int64_t kpacksPerBlock = tuningParams.getKPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();
    int64_t mPerThread = tuningParams.getMPerThread();
    int64_t nPerThread = tuningParams.getNPerThread();

    GeneralGemmBlockStructure blockStructure =
        *deriveGeneralGemmBlockStructure(blockSize);
    int64_t mThreadsPerCuwave = blockStructure.mThreadsPerCuwave;
    int64_t nThreadsPerCuwave = blockStructure.nThreadsPerCuwave;
    int64_t mCuwavesPerBlock = blockStructure.mCuwavesPerBlock;
    int64_t nCuwavesPerBlock = blockStructure.nCuwavesPerBlock;

    int64_t kPerBlock = kpacksPerBlock * kpack;

    bool useIndexDiffs = true;

    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;

    LLVM_DEBUG(llvm::dbgs() << "\ngridwise_gemm op:\n");
    LLVM_DEBUG(op.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << M << "\n"
               << "N: " << N << "\n"
               << "K: " << K << "\n"
               << "G: " << G << "\n"
               << "blockSize: " << blockSize << "\n"
               << "mPerBlock: " << mPerBlock << "\n"
               << "mBlocks = M / mPerBlock: " << mBlocks << "\n"
               << "nPerBlock: " << nPerBlock << "\n"
               << "nBlocks = N / NPerBlock: " << nBlocks << "\n"
               << "kPerBlock: " << kPerBlock << "\n"
               << "kpack: " << kpack << "\n"
               << "mPerThread: " << mPerThread << "\n"
               << "nPerThread: " << nPerThread << "\n"
               << "mThreadsPerCuwave: " << mThreadsPerCuwave << "\n"
               << "mCuwavesPerBlock: " << mCuwavesPerBlock << "\n"
               << "nThreadsPerCuwave: " << nThreadsPerCuwave << "\n"
               << "nCuwavesPerBlock: " << nCuwavesPerBlock << "\n");

    // Compute required LDS sizes.
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    LogicalResult res = computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize,
                                             ldsBlockSize, kpack);
    LLVM_DEBUG(llvm::dbgs() << "LDS block size:" << ldsBlockASize << " "
                            << ldsBlockBSize << " " << ldsBlockSize << "\n");
    if (res.failed())
      return failure();

    // Allocate LDS.
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, elementType, {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsGpuAllocOp = b.create<GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for matrix A tile in LDS buffer.
    int64_t ldsBlockAOffset = 0;
    auto ldsBlockASubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockAOffset, ldsBlockASize);
    Value ldsMatrixASubviewOp =
        reshapeBuffer(b, loc, ldsBlockASubviewOp, {"k", "m", "kpack"},
                      {kpacksPerBlock, mPerBlock, kpack});
    // Subviews for matrix B tile in LDS buffer.
    int64_t ldsBlockBOffset = ldsBlockASize;
    auto ldsBlockBSubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockBOffset, ldsBlockBSize);
    Value ldsMatrixBSubviewOp =
        reshapeBuffer(b, loc, ldsBlockBSubviewOp, {"k", "n", "kpack"},
                      {kpacksPerBlock, nPerBlock, kpack});

    // Alloc for Matrix C on registers.
    // Compute register size from attributes.

    int64_t gemmMRepeat =
        mPerBlock / (mPerThread * mThreadsPerCuwave * mCuwavesPerBlock);
    int64_t gemmNRepeat =
        nPerBlock / (nPerThread * nThreadsPerCuwave * nCuwavesPerBlock);

    LLVM_DEBUG(llvm::dbgs() << "GemmMRepeat: " << gemmMRepeat << "\n");
    LLVM_DEBUG(llvm::dbgs() << "GemmNRepeat: " << gemmNRepeat << "\n");

    int64_t threadCNumM = gemmMRepeat * mPerThread;
    int64_t threadCNumN = gemmNRepeat * nPerThread;
    int64_t threadCNumRegisters = threadCNumM * threadCNumN;
    auto threadCRegisterMemRefType =
        MemRefType::get({threadCNumRegisters}, accumulatorType, {},
                        gpu::GPUDialect::getPrivateAddressSpace());
    Value registerMatrixCAllocOp =
        b.create<GpuAllocOp>(loc, threadCRegisterMemRefType);
    Value registerMatrixCViewOp = reshapeBuffer(
        b, loc, registerMatrixCAllocOp, {"m", "n"}, {threadCNumM, threadCNumN});

    // Zero init Matrix C on registers.
    b.create<FillOp>(loc, registerMatrixCAllocOp, zeroConstantFloatOp);

    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());
    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};

    int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
    int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;

    GemmDimension vectorTiebreaker =
        (kpack > 1) ? GemmDimension::K : GemmDimension::MorN;
    int64_t aVectorLen, bVectorLen;
    GemmDimension aVectorDim, bVectorDim;
    std::tie(aVectorDim, aVectorLen) =
        bestVectorization(b, op.getA(), aCopyPerThread, vectorTiebreaker);
    std::tie(bVectorDim, bVectorLen) =
        bestVectorization(b, op.getB(), bCopyPerThread, vectorTiebreaker);

    LLVM_DEBUG(llvm::dbgs()
               << "aCopyPerThread: " << aCopyPerThread << "\n"
               << "bCopyPerThread: " << bCopyPerThread << "\n"
               << "aVectorDim: " << aVectorDim << "\n"
               << "aVectorLen: " << aVectorLen << "\n"
               << "bVectorDim: " << bVectorDim << "\n"
               << "bVectorLen: " << bVectorLen << "\n"
               << "vectorTiebreaker: " << vectorTiebreaker << "\n");

    // Vectorization lengths evenly devide *CopyPerThread, this is safe.
    int64_t aCopyKPerThread = aVectorDim == GemmDimension::K
                                  ? aVectorLen
                                  : aCopyPerThread / aVectorLen;
    int64_t copyMPerThread = aVectorDim == GemmDimension::MorN
                                 ? aVectorLen
                                 : aCopyPerThread / aVectorLen;
    int64_t bCopyKPerThread = bVectorDim == GemmDimension::K
                                  ? bVectorLen
                                  : bCopyPerThread / bVectorLen;
    int64_t copyNPerThread = bVectorDim == GemmDimension::MorN
                                 ? bVectorLen
                                 : bCopyPerThread / bVectorLen;

    FailureOr<Value> maybeWrappedA = wrapMatrixForGlobalLoad(
        b, loc, op.getA(), "m", bidGridOrder, bidGridLengths, gridSize,
        blockSize, kPerBlock, mPerBlock, aCopyKPerThread, copyMPerThread,
        aVectorDim);
    if (failed(maybeWrappedA))
      return maybeWrappedA;
    FailureOr<Value> maybeWrappedB = wrapMatrixForGlobalLoad(
        b, loc, op.getB(), "n", bidGridOrder, bidGridLengths, gridSize,
        blockSize, kPerBlock, nPerBlock, bCopyKPerThread, copyNPerThread,
        bVectorDim);
    if (failed(maybeWrappedB))
      return maybeWrappedB;
    Value wrappedA = std::move(*maybeWrappedA),
          wrappedB = std::move(*maybeWrappedB);

    ArrayAttr aVectorGlobalMap = globalVectorLayout(
        b, loc, "m", aCopyKPerThread, copyMPerThread, kpack, aVectorDim);
    ArrayAttr bVectorGlobalMap = globalVectorLayout(
        b, loc, "n", bCopyKPerThread, copyNPerThread, kpack, bVectorDim);

    TransformingForOp blockwiseLoadA =
        createGlobalLoadLoop(b, loc, wrappedA, aVectorGlobalMap, aCopyPerThread,
                             aVectorLen, bid, tid);
    TransformingForOp blockwiseLoadB =
        createGlobalLoadLoop(b, loc, wrappedB, bVectorGlobalMap, bCopyPerThread,
                             bVectorLen, bid, tid);

    ArrayAttr aVectorLdsMap = ldsVectorLayout(b, loc, aCopyPerThread);
    ArrayAttr bVectorLdsMap = ldsVectorLayout(b, loc, bCopyPerThread);

    FailureOr<Value> maybeWrappedLdsA = wrapLDSBufferForStore(
        b, loc, ldsMatrixASubviewOp, "m", aCopyKPerThread, copyMPerThread);
    if (failed(maybeWrappedLdsA))
      return maybeWrappedLdsA;
    FailureOr<Value> maybeWrappedLdsB = wrapLDSBufferForStore(
        b, loc, ldsMatrixBSubviewOp, "n", bCopyKPerThread, copyNPerThread);
    if (failed(maybeWrappedLdsB))
      return maybeWrappedLdsB;
    Value wrappedLdsA = std::move(*maybeWrappedLdsA),
          wrappedLdsB = std::move(*maybeWrappedLdsB);

    TransformingForOp blockwiseStoreA =
        createLdsStoreLoop(b, loc, blockwiseLoadA.getResult(0), aVectorLdsMap,
                           wrappedLdsA, aCopyPerThread, tid);
    TransformingForOp blockwiseStoreB =
        createLdsStoreLoop(b, loc, blockwiseLoadB.getResult(0), bVectorLdsMap,
                           wrappedLdsB, bCopyPerThread, tid);

    // Emit loop.
    int64_t nIterations = K / kPerBlock;
    BlockwiseGemmOp blockwiseGemmOp;
    // Start at 1 to make it clearer we have performed software pipelining.
    auto loopOp = b.create<AffineForOp>(loc, 1, nIterations, 1);
    {
      // inside the loop.
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());

      // LDS barrier.
      b.create<LDSBarrierOp>(loc);

      // Emit blockwise GEMM.
      blockwiseGemmOp = b.create<BlockwiseGemmOp>(
          loc, ldsMatrixASubviewOp, ldsMatrixBSubviewOp, registerMatrixCViewOp,
          op.getBlockSizeAttr(), op.getParamsAttr());

      // LDS barrier.
      // This barrier prevents halo part of outputs having weird values.
      b.create<LDSBarrierOp>(loc);

      // We don't update in the clone becasue we might accidentally replace
      // other zeroes.
      Value iv = loopOp.getInductionVar();
      BlockAndValueMapping loadAUpdates, loadBUpdates;
      auto blockwiseLoadAClone = cast<TransformingForOp>(
          b.clone(*blockwiseLoadA.getOperation(), loadAUpdates));
      blockwiseLoadAClone.setOperand(
          blockwiseLoadAClone.getUpperInits(/*domain=*/0)
              .getBeginOperandIndex(),
          iv);

      auto blockwiseLoadBClone = cast<TransformingForOp>(
          b.clone(*blockwiseLoadB.getOperation(), loadBUpdates));
      blockwiseLoadBClone.setOperand(
          blockwiseLoadBClone.getUpperInits(/*domain=*/0)
              .getBeginOperandIndex(),
          iv);

      // Emit blockwise stores
      BlockAndValueMapping storeAUpdates, storeBUpdates;
      storeAUpdates.map(blockwiseLoadA.getResult(0),
                        blockwiseLoadAClone.getResult(0));
      storeBUpdates.map(blockwiseLoadB.getResult(0),
                        blockwiseLoadBClone.getResult(0));
      b.clone(*blockwiseStoreA.getOperation(), storeAUpdates);
      b.clone(*blockwiseStoreB.getOperation(), storeBUpdates);
    }
    // outside the loop.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    BlockAndValueMapping tailGemmCloneMap;
    b.clone(*blockwiseGemmOp, tailGemmCloneMap);

    // Apparently, the canonicalizer doesn't get rid of empty loops without
    // results properly, remove them ourselves.
    if (nIterations <= 1)
      b.eraseOp(loopOp);

    // Threadwise copy from register (naive tensor) to global (generic tensor).
    TopDownTMBuilder splitMemoryCoords(
        b, {"bid", "tid", "iter"}, {gridSize, blockSize, threadCNumRegisters},
        loc);
    splitMemoryCoords.merge({"g", "m_block", "n_block"}, {0, 1, 2}, "bid",
                            {G, mBlocks, nBlocks});
    splitMemoryCoords.merge({"m_cuwaves", "n_cuwaves", "m_cuwave", "n_cuwave"},
                            {3, 4, 5, 6}, "tid",
                            {mCuwavesPerBlock, nCuwavesPerBlock,
                             mThreadsPerCuwave, nThreadsPerCuwave});
    splitMemoryCoords.merge({"m_repeat", "m_thread", "n_repeat", "n_thread"},
                            {7, 8, 9, 10}, "iter",
                            {gemmMRepeat, mPerThread, gemmNRepeat, nPerThread});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

    auto toMatrixC =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {"g"});
    toMatrixC.unmerge(
        "gemmM", 1,
        {"m_block", "m_repeat", "m_cuwaves", "m_cuwave", "m_thread"},
        {M / mPerBlock, gemmMRepeat, mCuwavesPerBlock, mThreadsPerCuwave,
         mPerThread});
    toMatrixC.unmerge(
        "gemmN", 2,
        {"n_block", "n_repeat", "n_cuwaves", "n_cuwave", "n_thread"},
        {N / nPerBlock, gemmNRepeat, nCuwavesPerBlock, nThreadsPerCuwave,
         nPerThread});

    TransformMapAttr toMatrixCAttr = toMatrixC.get();

    TopDownTMBuilder toRegisterC(b, {"bid", "tid", "iter"},
                                 {gridSize, blockSize, threadCNumRegisters},
                                 loc);
    toRegisterC.ignore("bid");
    toRegisterC.ignore("tid");
    toRegisterC.passThrough({"iter"}, {0}, {"iter"});
    TransformMapAttr toRegisterCAttr = toRegisterC.get();

    Value registerC = registerMatrixCAllocOp;
    // If we need to type-convert the accumulator (currently this is only
    // fp32->f16) then we must do so before the writeback loop in which fusion
    // takes places at this time, since the fusion pass as currently written
    // can't interceps the type conversions.
    if (destType != accumulatorType) {
      auto convertedCType =
          threadCRegisterMemRefType.clone(destType).cast<MemRefType>();
      Value convertedC = b.create<rock::GpuAllocOp>(loc, convertedCType);
      auto convertLoop = b.create<TransformingForOp>(
          loc, ArrayRef<ValueRange>{{zeroConstantOp}},
          ArrayRef<Attribute>{b.getArrayAttr({})},
          /*bounds=*/convertedCType.getShape(), /*strides=*/llvm::None,
          /*useIndexDiffs=*/true, /*forceUnroll=*/true);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(convertLoop.getBody());
        Value coord = convertLoop.getLowerCoords(/*domain=*/0)[0];
        Value loaded =
            b.create<InBoundsLoadOp>(loc, accumulatorType, registerC, coord);
        Value cast = createTypeConversionOp(b, loc, loaded, destType);
        b.create<InBoundsStoreOp>(loc, cast, convertedC, coord);
      }
      registerC = convertedC;
    }

    ArrayAttr idToMatrixCMaps =
        b.getArrayAttr({splitMemoryCoordsAttr, toMatrixCAttr});
    Value tensorC;
    ArrayAttr idToTensorCMaps;
    std::tie(tensorC, idToTensorCMaps) =
        untransform(b, op.getC(), idToMatrixCMaps);
    auto writeOobDims = computeOobFromTransforms(b, idToTensorCMaps);

    ArrayRef<int64_t> tensorCShape =
        tensorC.getType().cast<MemRefType>().getShape();
    int64_t tensorCDataPerCopy = getMaxVectorization(
        idToTensorCMaps, /*dim=*/2, threadCNumRegisters, tensorCShape);

    SmallVector<Value, 3> writeStartCoords = {bid, tid, zeroConstantOp};

    auto outLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{writeStartCoords, writeStartCoords},
        ArrayRef<Attribute>{b.getArrayAttr({toRegisterCAttr}), idToTensorCMaps},
        ArrayRef<int64_t>{1, 1, threadCNumRegisters},
        ArrayRef<int64_t>{1, 1, tensorCDataPerCopy},
        /*forceUnroll=*/true, /*useIndexDiffs=*/useIndexDiffs);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(outLoop.getBody());
      b.create<ThreadwiseCopyV2Op>(
          loc, registerC, tensorC,
          /*length=*/b.getIndexAttr(tensorCDataPerCopy),
          StoreMethodAttr::get(op.getContext(), StoreMethod::Set),
          std::get<0>(writeOobDims), std::get<1>(writeOobDims),
          outLoop.getLowerCoords(/*domain=*/0)[0],
          outLoop.getLowerCoords(/*domain=*/1));
    }

    b.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// GridwiseGemmV2 lowering.
//===----------------------------------------------------------------------===//

struct GridwiseGemmV2RewritePattern
    : public OpRewritePattern<GridwiseGemmV2Op> {
  using OpRewritePattern<GridwiseGemmV2Op>::OpRewritePattern;

  LogicalResult computeLDSBlockSizes(GridwiseGemmV2Op op,
                                     int64_t &a_block_space,
                                     int64_t &b_block_space,
                                     int64_t &total_block_space,
                                     int64_t KPack = 1) const {
    int64_t max_lds_align = 1;

    XdlopsGemmParamsAttr tuningParams = op.getParams();
    int64_t KPerBlock = tuningParams.getKPerBlock();
    int64_t MPerBlock = tuningParams.getMPerBlock();
    int64_t NPerBlock = tuningParams.getNPerBlock();

    int64_t AlignedNPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(NPerBlock, max_lds_align);

    // A matrix in LDS memory, dst of blockwise copy
    int64_t AlignedMPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(MPerBlock, max_lds_align);

    LLVM_DEBUG(llvm::dbgs() << "MPerBlock : " << MPerBlock << "\n");
    LLVM_DEBUG(llvm::dbgs() << "NPerBlock : " << NPerBlock << "\n");
    LLVM_DEBUG(llvm::dbgs() << "max_lds_align : " << max_lds_align << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "AlignedMPerBlock : " << AlignedMPerBlock << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "AlignedNPerBlock : " << AlignedNPerBlock << "\n");

    a_block_space = math_util::integer_least_multiple(
                        KPerBlock * AlignedMPerBlock, max_lds_align) *
                    KPack;

    // B matrix in LDS memory, dst of blockwise copy
    b_block_space = math_util::integer_least_multiple(
                        KPerBlock * AlignedNPerBlock, max_lds_align) *
                    KPack;

    total_block_space = a_block_space + b_block_space;

    LLVM_DEBUG(llvm::dbgs() << "a_block_space: " << a_block_space << "\n");
    LLVM_DEBUG(llvm::dbgs() << "b_block_space: " << b_block_space << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "total_block_space: " << total_block_space << "\n\n");

    // TODO: adjust for data type and device
    if (total_block_space * sizeof(float) > 64 * 1024)
      return failure();

    return success();
  }

  LogicalResult matchAndRewrite(GridwiseGemmV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Obtain data type.
    auto elementType = op.getB().getType().getElementType();

    // Prepare some useful constants.
    Value zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    Value matA = unKpack(b, op.getA());
    Value matB = unKpack(b, op.getB());

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = op.getA().getType().getShape();
    bShape = op.getB().getType().getShape();
    cShape = op.getC().getType().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t K = aShape[1];
    int64_t M = aShape[2];
    int64_t N = bShape[2];

    if (bShape[0] != G || cShape[0] != G) {
      return op.emitOpError("Mismatched G dimensions in matrix multiply;")
             << " A[0] = " << G << " b[0] = " << bShape[0]
             << " C[0] = " << cShape[0];
    }
    if (cShape[1] != M) {
      return op.emitOpError("Mismatched M dimensions in matrix multiply:")
             << " A[2] = " << M << " C[1] = " << cShape[1];
    }
    if (bShape[1] != K) {
      return op.emitOpError("Mismatched K dimensions in matrix multiply:")
             << " A[1] = " << K << " B[1] = " << bShape[1];
    }
    if (cShape[2] != N) {
      return op.emitOpError("Mismatched N dimensions in matrix multiply:")
             << " B[2] = " << N << " C[2] = " << cShape[2];
    }

    // Obtain critical tuning parameters.
    uint32_t blockSize = op.getBlockSize();
    uint32_t gridSize = op.getGridSize();
    XdlopsGemmParamsAttr tuningParams = op.getParams();
    int64_t KPack = tuningParams.getKpack();
    int64_t KPerBlock = tuningParams.getKPerBlock();
    int64_t MPerBlock = tuningParams.getMPerBlock();
    int64_t NPerBlock = tuningParams.getNPerBlock();

    int64_t matrix_a_source_data_per_read = 0;
    int64_t matrix_b_source_data_per_read = 0;
    GemmDimension matrix_a_source_vector_read_dim;
    GemmDimension matrix_b_source_vector_read_dim;

    int64_t aCopyPerThread = (KPerBlock * KPack * MPerBlock) / blockSize;
    int64_t bCopyPerThread = (KPerBlock * KPack * NPerBlock) / blockSize;
    GemmDimension vectorTiebreaker =
        (KPack > 1) ? GemmDimension::K : GemmDimension::MorN;
    std::tie(matrix_a_source_vector_read_dim, matrix_a_source_data_per_read) =
        bestVectorization(b, matA, aCopyPerThread, vectorTiebreaker);
    // Temporary clamping hack because the old logic expects certain invariants
    // between the vectorization length and kPack
    if (KPack > 1) {
      if (matrix_a_source_vector_read_dim == GemmDimension::K) {
        matrix_a_source_data_per_read =
            std::min(matrix_a_source_data_per_read, KPack);
      }
      if (matrix_a_source_vector_read_dim == GemmDimension::MorN) {
        matrix_a_source_data_per_read =
            std::min(matrix_a_source_data_per_read, aCopyPerThread / KPack);
      }
    }
    // Similar temporary clamp, avoids division by 0
    if (matrix_a_source_vector_read_dim == GemmDimension::K) {
      matrix_a_source_data_per_read =
          std::min(matrix_a_source_data_per_read, KPerBlock);
    }

    std::tie(matrix_b_source_vector_read_dim, matrix_b_source_data_per_read) =
        bestVectorization(b, matB, bCopyPerThread, vectorTiebreaker);
    // Temporary clamping hack because the old logic expects certain invariants
    // between the vectorization length and kPack
    if (KPack > 1) {
      if (matrix_b_source_vector_read_dim == GemmDimension::K) {
        matrix_b_source_data_per_read =
            std::min(matrix_b_source_data_per_read, KPack);
      }
      if (matrix_b_source_vector_read_dim == GemmDimension::MorN) {
        matrix_b_source_data_per_read =
            std::min(matrix_b_source_data_per_read, bCopyPerThread / KPack);
      }
    }
    // Similar temporary clamp, avoids division by 0
    if (matrix_b_source_vector_read_dim == GemmDimension::K) {
      matrix_b_source_data_per_read =
          std::min(matrix_b_source_data_per_read, KPerBlock);
    }

    // Obtain XDLOPS-related attributes.
    int64_t MPerWave = tuningParams.getMPerWave();
    int64_t NPerWave = tuningParams.getNPerWave();
    // int64_t MWaves = MPerBlock / MPerWave;
    int64_t NWaves = NPerBlock / NPerWave;

    auto MPerWaveConstantOp = b.create<ConstantIndexOp>(loc, MPerWave);
    auto NPerWaveConstantOp = b.create<ConstantIndexOp>(loc, NPerWave);
    auto NWavesConstantOp = b.create<ConstantIndexOp>(loc, NWaves);

    constexpr int64_t waveSize = 64;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, waveSize);

    bool useIndexDiffs = true;

    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());

    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;
    int64_t GStride = MBlockWork * NBlockWork;

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << M << "\n"
               << "N: " << N << "\n"
               << "K: " << K << "\n"
               << "MPerBlock: " << MPerBlock << "\n"
               << "NPerBlock: " << NPerBlock << "\n"
               << "KPerBlock: " << KPerBlock << "\n"
               << "KPack: " << KPack << "\n"
               << "MBlockWork = M / MPerBlock: " << MBlockWork << "\n"
               << "NBlockWork = N / NPerBlock: " << NBlockWork << "\n"
               << "MPerWave: " << MPerWave << "\n"
               << "NPerWave: " << NPerWave << "\n"
               << "matrix_a_source_data_per_read: "
               << matrix_a_source_data_per_read << "\n"
               << "matrix_b_source_data_per_read: "
               << matrix_b_source_data_per_read << "\n"
               << "matrix_a_source_vector_read_dim: "
               << matrix_a_source_vector_read_dim << "\n"
               << "matrix_b_source_vector_read_dim: "
               << matrix_b_source_vector_read_dim << "\n");

    auto MPerBlockConstantOp = b.create<ConstantIndexOp>(loc, MPerBlock);
    auto NPerBlockConstantOp = b.create<ConstantIndexOp>(loc, NPerBlock);
    auto KPerBlockConstantOp = b.create<ConstantIndexOp>(loc, KPerBlock);
    auto MBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, MBlockWork);
    auto GStridOp = b.create<ConstantIndexOp>(loc, GStride);
    // -----

    // Compute the coordinate for the current workgroup on global memory.

    // Original C++ logic:
    // constexpr auto wkgrp_schd_order = NBlock1MBlock0;
    // constexpr auto block_work_sequence =
    //     make_batch_block_work_sequence<G, MBlockWork, NBlockWork,
    //     WorkgroupSchdOrder>{}.get();
    // constexpr auto block_work_desc =
    // make_cluster_descriptor(block_work_sequence); const auto block_work_id =
    // block_work_desc.CalculateClusterIndex(get_block_1d_id());

    // Result block_work_desc is <NBlockWorkd, MBlockWork>

    auto block_work_id_g = b.create<DivUIOp>(loc, bid, GStridOp);
    auto block_work_rem = b.create<RemUIOp>(loc, bid, GStridOp);
    auto block_work_id_m =
        b.create<RemUIOp>(loc, block_work_rem, MBlockWorkConstantOp);
    auto block_work_id_n =
        b.create<DivUIOp>(loc, block_work_rem, MBlockWorkConstantOp);

    auto m_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_m, MPerBlockConstantOp);
    auto n_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_n, NPerBlockConstantOp);

    // -----

    // Logic to prepare parameters for blockwise_copy.

    // Compute ThreadSliceLengths for Matrix A.
    int64_t GemmABlockCopyNumberDataPerThread =
        MPerBlock * KPerBlock * KPack / blockSize;

    LLVM_DEBUG(llvm::dbgs() << "GemmABlockCopyNumberDataPerThread: "
                            << GemmABlockCopyNumberDataPerThread << "\n");

    int64_t GemmABlockCopyThreadSliceLengths_GemmK;
    int64_t GemmABlockCopyThreadSliceLengths_GemmM;
    int64_t GemmABlockCopyThreadSliceLengths_GemmKPack = 1;
    switch (matrix_a_source_vector_read_dim) {
    case GemmDimension::K:
      if (KPack > 1) {
        GemmABlockCopyThreadSliceLengths_GemmKPack =
            matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            KPack / matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmM =
            GemmABlockCopyNumberDataPerThread / KPack;
      } else {
        GemmABlockCopyThreadSliceLengths_GemmK = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmM =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmK;
      }
      break;
    case GemmDimension::MorN:
      // TBD: FIXME. Review logic here.
      if (KPack > 1) {
        GemmABlockCopyThreadSliceLengths_GemmM = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmM / KPack;
        GemmABlockCopyThreadSliceLengths_GemmKPack = KPack;
      } else {
        GemmABlockCopyThreadSliceLengths_GemmM = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmM;
      }
      break;
    case GemmDimension::G:
      LLVM_DEBUG(llvm::dbgs()
                 << "Vector loads/stores aren't possible in the G dimension "
                 << "and should not haven been attempted\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "thread slice lengths for Matrix A: "
               << GemmABlockCopyThreadSliceLengths_GemmK << " "
               << GemmABlockCopyThreadSliceLengths_GemmM << " "
               << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n");

    if (GemmABlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmM == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmKPack == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Blockwise copy slice lengths for matrix A is zero which "
                 << "is invalid.\n");
      return failure();
    }

    // Compute ThreadClusterLengths for Matrix A.
    int64_t GemmABlockCopyClusterLengths_GemmKPack =
        KPack / GemmABlockCopyThreadSliceLengths_GemmKPack;
    int64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock / GemmABlockCopyThreadSliceLengths_GemmK;
    // int64_t GemmABlockCopyClusterLengths_GemmM =
    //    MPerBlock / GemmABlockCopyThreadSliceLengths_GemmM;

    LLVM_DEBUG(llvm::dbgs() << "thread cluster lengths for Matrix A: "
                            << GemmABlockCopyClusterLengths_GemmK << " "
                            << GemmABlockCopyClusterLengths_GemmKPack << "\n");

    // Compute ThreadSliceLengths for Matrix B.
    int64_t GemmBBlockCopyNumberDataPerThread =
        NPerBlock * KPerBlock * KPack / blockSize;

    LLVM_DEBUG(llvm::dbgs() << "GemmBBlockCopyNumberDataPerThread: "
                            << GemmBBlockCopyNumberDataPerThread << "\n");

    int64_t GemmBBlockCopyThreadSliceLengths_GemmK;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmN;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmKPack = 1;
    switch (matrix_b_source_vector_read_dim) {
    case GemmDimension::K:
      if (KPack > 1) {
        GemmBBlockCopyThreadSliceLengths_GemmKPack =
            matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            KPack / matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmN =
            GemmBBlockCopyNumberDataPerThread / KPack;
      } else {
        GemmBBlockCopyThreadSliceLengths_GemmK = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmN =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmK;
      }
      break;
    case GemmDimension::MorN:
      // TBD: FIXME. Review logic here.
      if (KPack > 1) {
        GemmBBlockCopyThreadSliceLengths_GemmN = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmN / KPack;
        GemmBBlockCopyThreadSliceLengths_GemmKPack = KPack;
      } else {
        GemmBBlockCopyThreadSliceLengths_GemmN = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmN;
      }
      break;
    case GemmDimension::G:
      LLVM_DEBUG(llvm::dbgs()
                 << "Vector loads/stores aren't possible in the G dimension "
                 << "and should not haven been attempted.\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "thread slice lengths for Matrix B: "
               << GemmBBlockCopyThreadSliceLengths_GemmK << " "
               << GemmBBlockCopyThreadSliceLengths_GemmN << " "
               << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n");

    if (GemmBBlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmN == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmKPack == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Blockwise copy slice lengths for matrix B is zero which "
                 << "is invalid.\n");
      return failure();
    }

    assert(GemmBBlockCopyThreadSliceLengths_GemmK > 0);
    assert(GemmBBlockCopyThreadSliceLengths_GemmN > 0);
    assert(GemmBBlockCopyThreadSliceLengths_GemmKPack > 0);
    // Compute ThreadClusterLengths for Matrix B.
    uint64_t GemmBBlockCopyClusterLengths_GemmKPack =
        KPack / GemmBBlockCopyThreadSliceLengths_GemmKPack;
    uint64_t GemmBBlockCopyClusterLengths_GemmK =
        KPerBlock / GemmBBlockCopyThreadSliceLengths_GemmK;
    uint64_t GemmBBlockCopyClusterLengths_GemmN =
        NPerBlock / GemmBBlockCopyThreadSliceLengths_GemmN;

    LLVM_DEBUG(llvm::dbgs() << "thread cluster lengths for Matrix B: "
                            << GemmBBlockCopyClusterLengths_GemmK << " "
                            << GemmBBlockCopyClusterLengths_GemmN << " "
                            << GemmBBlockCopyClusterLengths_GemmKPack << "\n");

    // Compute thread_data_id_begin for Matrix A.
    // ClusterArrangeOrder for Matrix A is <1, 0>.
    // So divide by GemmABlockCopyClusterLengths_GemmK.
    auto GemmABlockCopyClusterLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyClusterLengths_GemmK);
    auto GemmABlockCopyThreadSliceLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyThreadSliceLengths_GemmK);
    auto GemmABlockCopyThreadSliceLengths_GemmMConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyThreadSliceLengths_GemmM);

    Value GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp;
    Value GemmABlockCopyClusterLengths_GemmKPackConstantOp;
    Value GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp;
    if (KPack > 1) {
      GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyClusterLengths_GemmK *
                                        GemmABlockCopyClusterLengths_GemmKPack);
      GemmABlockCopyClusterLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyClusterLengths_GemmKPack);
      GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyThreadSliceLengths_GemmKPack);
    }

    Value GemmABlockCopyThreadClusterId_Y;
    Value GemmABlockCopyThreadClusterId_X;
    Value GemmAThreadDataIdBegin_Y;
    Value GemmAThreadDataIdBegin_X;

    Value GemmABlockCopyThreadClusterId_Z;
    Value GemmAThreadDataIdBegin_Z;
    if (KPack > 1) {
      GemmABlockCopyThreadClusterId_Z = b.create<RemUIOp>(
          loc,
          b.create<RemUIOp>(
              loc, tid,
              GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp),
          GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmABlockCopyThreadClusterId_Y = b.create<DivUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp);
      GemmABlockCopyThreadClusterId_X = b.create<DivUIOp>(
          loc,
          b.create<RemUIOp>(
              loc, tid,
              GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp),
          GemmABlockCopyClusterLengths_GemmKConstantOp);

      GemmAThreadDataIdBegin_Z =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Z,
                           GemmABlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Y,
                           GemmABlockCopyThreadSliceLengths_GemmMConstantOp);
      GemmAThreadDataIdBegin_X = b.create<MulIOp>(
          loc, GemmABlockCopyThreadClusterId_X,
          GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp);
    } else {
      GemmABlockCopyThreadClusterId_Y = b.create<RemUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmABlockCopyThreadClusterId_X = b.create<DivUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Y,
                           GemmABlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_X =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_X,
                           GemmABlockCopyThreadSliceLengths_GemmMConstantOp);
    }

    Value GemmABlockCopySourceCoord_Y;
    Value GemmABlockCopySourceCoord_X;

    Value GemmABlockCopySourceCoord_Z;
    if (KPack > 1) {
      GemmABlockCopySourceCoord_Z = GemmAThreadDataIdBegin_Z;
      GemmABlockCopySourceCoord_Y = b.create<AddIOp>(
          loc, m_block_data_on_global, GemmAThreadDataIdBegin_Y);
      GemmABlockCopySourceCoord_X = GemmAThreadDataIdBegin_X;
    } else {
      GemmABlockCopySourceCoord_Y = GemmAThreadDataIdBegin_Y;
      GemmABlockCopySourceCoord_X = b.create<AddIOp>(
          loc, m_block_data_on_global, GemmAThreadDataIdBegin_X);
    }

    Value GemmABlockCopyDestCoord_Y;
    Value GemmABlockCopyDestCoord_X;

    Value GemmABlockCopyDestCoord_Z;
    if (KPack > 1) {
      GemmABlockCopyDestCoord_Z = GemmAThreadDataIdBegin_Z;
    }
    GemmABlockCopyDestCoord_Y = GemmAThreadDataIdBegin_Y;
    GemmABlockCopyDestCoord_X = GemmAThreadDataIdBegin_X;

    // Compute thread_data_id_begin for Matrix B.
    // ClusterArrangeOrder for Matrix B is <0, 1>
    // So divide by GemmBBlockCopyClusterLengths_GemmN.
    auto GemmBBlockCopyClusterLengths_GemmNConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyClusterLengths_GemmN);
    auto GemmBBlockCopyThreadSliceLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyThreadSliceLengths_GemmK);
    auto GemmBBlockCopyThreadSliceLengths_GemmNConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyThreadSliceLengths_GemmN);

    Value GemmBBlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp;
    Value GemmBBlockCopyClusterLengths_GemmKPackConstantOp;
    Value GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp;
    if (KPack > 1) {
      GemmBBlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyClusterLengths_GemmK *
                                        GemmBBlockCopyClusterLengths_GemmKPack);
      GemmBBlockCopyClusterLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyClusterLengths_GemmKPack);
      GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyThreadSliceLengths_GemmKPack);
    }

    Value GemmBBlockCopyThreadClusterId_Y;
    Value GemmBBlockCopyThreadClusterId_X;
    Value GemmBThreadDataIdBegin_Y;
    Value GemmBThreadDataIdBegin_X;

    Value GemmBBlockCopyThreadClusterId_Z;
    Value GemmBThreadDataIdBegin_Z;

    if (KPack > 1) {
      GemmBBlockCopyThreadClusterId_Z = b.create<DivUIOp>(
          loc,
          b.create<DivUIOp>(loc, tid,
                            GemmBBlockCopyClusterLengths_GemmNConstantOp),
          GemmBBlockCopyClusterLengths_GemmKPackConstantOp);
      GemmBBlockCopyThreadClusterId_Y = b.create<RemUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBBlockCopyThreadClusterId_X = b.create<RemUIOp>(
          loc,
          b.create<DivUIOp>(loc, tid,
                            GemmBBlockCopyClusterLengths_GemmNConstantOp),
          GemmBBlockCopyClusterLengths_GemmKPackConstantOp);

      GemmBThreadDataIdBegin_Z =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Z,
                           GemmBBlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmBThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Y,
                           GemmBBlockCopyThreadSliceLengths_GemmNConstantOp);
      GemmBThreadDataIdBegin_X = b.create<MulIOp>(
          loc, GemmBBlockCopyThreadClusterId_X,
          GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp);
    } else {
      GemmBBlockCopyThreadClusterId_Y = b.create<DivUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBBlockCopyThreadClusterId_X = b.create<RemUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Y,
                           GemmBBlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmBThreadDataIdBegin_X =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_X,
                           GemmBBlockCopyThreadSliceLengths_GemmNConstantOp);
    }

    Value GemmBBlockCopySourceCoord_Y;
    Value GemmBBlockCopySourceCoord_X;

    Value GemmBBlockCopySourceCoord_Z;
    if (KPack > 1) {
      GemmBBlockCopySourceCoord_Z = GemmBThreadDataIdBegin_Z;
      GemmBBlockCopySourceCoord_Y = b.create<AddIOp>(
          loc, n_block_data_on_global, GemmBThreadDataIdBegin_Y);
      GemmBBlockCopySourceCoord_X = GemmBThreadDataIdBegin_X;
    } else {
      GemmBBlockCopySourceCoord_Y = GemmBThreadDataIdBegin_Y;
      GemmBBlockCopySourceCoord_X = b.create<AddIOp>(
          loc, n_block_data_on_global, GemmBThreadDataIdBegin_X);
    }

    Value GemmBBlockCopyDestCoord_Y;
    Value GemmBBlockCopyDestCoord_X;

    Value GemmBBlockCopyDestCoord_Z;
    if (KPack > 1) {
      GemmBBlockCopyDestCoord_Z = GemmBThreadDataIdBegin_Z;
    }
    GemmBBlockCopyDestCoord_Y = GemmBThreadDataIdBegin_Y;
    GemmBBlockCopyDestCoord_X = GemmBThreadDataIdBegin_X;

    auto GemmBlockCoord_G = block_work_id_g;
    // -----

    // Alocate LDS and create subviews.

    // Compute required LDS sizes.
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    LogicalResult res = computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize,
                                             ldsBlockSize, KPack);
    LLVM_DEBUG(llvm::dbgs() << "LDS block size:" << ldsBlockASize << " "
                            << ldsBlockBSize << " " << ldsBlockSize << "\n");
    if (res.failed())
      return failure();

    // Allocate LDS.
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, elementType, {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsGpuAllocOp = b.create<GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for Matrix A.
    int64_t ldsBlockAOffset = 0;

    Value ldsBlockASubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockAOffset, ldsBlockASize);

    // Get matrix subviews.
    // Compute matrix A dimension from attributes.
    Value ldsMatrixASubviewOp;
    if (KPack > 1) {
      ldsMatrixASubviewOp =
          reshapeBuffer(b, loc, ldsBlockASubviewOp, {"g", "k", "m", "kpack"},
                        {1, KPerBlock, MPerBlock, KPack});
    } else {
      ldsMatrixASubviewOp =
          reshapeBuffer(b, loc, ldsBlockASubviewOp, {"g", "k", "m"},
                        {1, KPerBlock, MPerBlock});
    }

    // Subviews for Matrix B.
    int64_t ldsBlockBOffset = ldsBlockASize;
    Value ldsBlockBSubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockBOffset, ldsBlockBSize);

    // Get matrix subviews.
    // Compute matrix B dimension from attributes.
    Value ldsMatrixBSubviewOp;
    if (KPack > 1) {
      ldsMatrixBSubviewOp =
          reshapeBuffer(b, loc, ldsBlockBSubviewOp, {"g", "k", "m", "kpack"},
                        {1, KPerBlock, NPerBlock, KPack});
    } else {
      ldsMatrixBSubviewOp =
          reshapeBuffer(b, loc, ldsBlockBSubviewOp, {"g", "k", "m"},
                        {1, KPerBlock, NPerBlock});
    }

    // -----

    // Determine vector / scalar load type for Matrix A / B.
    SmallVector<int64_t, 4> blockwiseCopyABounds;
    if (KPack > 1) {
      blockwiseCopyABounds = {1, GemmABlockCopyThreadSliceLengths_GemmK,
                              GemmABlockCopyThreadSliceLengths_GemmM,
                              GemmABlockCopyThreadSliceLengths_GemmKPack};
    } else {
      blockwiseCopyABounds = {1, GemmABlockCopyThreadSliceLengths_GemmK,
                              GemmABlockCopyThreadSliceLengths_GemmM};
    }

    auto dimToIndex = [](GemmDimension dim) {
      return static_cast<uint32_t>(dim);
    };

    GemmDimension dimA = matrix_a_source_vector_read_dim;
    uint32_t blockwiseVectorDimA = dimToIndex(dimA);

    int64_t blockwiseLoadVectorLenA = matrix_a_source_data_per_read;
    Type aLoadIntermediate, aLoadType;
    computeLoadStoreTypeInfo(b, blockwiseCopyABounds, blockwiseLoadVectorLenA,
                             blockwiseVectorDimA, KPack, elementType, aLoadType,
                             aLoadIntermediate);

    LLVM_DEBUG(llvm::dbgs() << "blockwise copy A bounds: ");
    for (auto v : blockwiseCopyABounds)
      LLVM_DEBUG(llvm::dbgs() << v << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n");

    LLVM_DEBUG(llvm::dbgs()
               << "Corrected blockwise vector dim A: " << blockwiseVectorDimA
               << "\n"
               << "Load type A: " << aLoadType << "\n"
               << "Intermediate type A: " << aLoadIntermediate << "\n");

    SmallVector<int64_t, 4> blockwiseCopyBBounds;
    if (KPack > 1) {
      blockwiseCopyBBounds = {1, GemmBBlockCopyThreadSliceLengths_GemmK,
                              GemmBBlockCopyThreadSliceLengths_GemmN,
                              GemmBBlockCopyThreadSliceLengths_GemmKPack};
    } else {
      blockwiseCopyBBounds = {1, GemmBBlockCopyThreadSliceLengths_GemmK,
                              GemmBBlockCopyThreadSliceLengths_GemmN};
    }
    LLVM_DEBUG(llvm::dbgs() << "blockwise copy B bounds: ");
    for (auto v : blockwiseCopyBBounds)
      LLVM_DEBUG(llvm::dbgs() << v << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n");

    GemmDimension dimB = matrix_b_source_vector_read_dim;
    uint32_t blockwiseVectorDimB = dimToIndex(dimB);
    int64_t blockwiseLoadVectorLenB = matrix_b_source_data_per_read;

    Type bLoadIntermediate, bLoadType;
    computeLoadStoreTypeInfo(b, blockwiseCopyBBounds, blockwiseLoadVectorLenB,
                             blockwiseVectorDimB, KPack, elementType, bLoadType,
                             bLoadIntermediate);
    LLVM_DEBUG(llvm::dbgs()
               << "Corrected blockwise vector dim B: " << blockwiseVectorDimB
               << "\n"
               << "Load type B: " << bLoadType << "\n"
               << "Intermediate type B: " << bLoadIntermediate << "\n");

    // -----

    // Compute source and destination coordinates for BlockwiseCopy ops.
    // Matrix A: {0, 0, m_block_data_on_global}, {0, 0, 0}
    // Matrix B: {0, 0, n_block_data_on_global}, {0, 0, 0}

    // -----

    // Blockwise copies before the loop.
    // Blockwise copy from global (generic tensor) to LDS (naive tensor).

    SmallVector<Value, 4> blockwiseLoadACoords;
    if (KPack > 1) {
      blockwiseLoadACoords = {GemmBlockCoord_G, GemmABlockCopySourceCoord_Z,
                              GemmABlockCopySourceCoord_Y,
                              GemmABlockCopySourceCoord_X};
    } else {
      blockwiseLoadACoords = {GemmBlockCoord_G, GemmABlockCopySourceCoord_Y,
                              GemmABlockCopySourceCoord_X};
    }
    // Emit blockwise load for matrix A.
    TransformingForOp blockwiseLoadA = createGlobalLoadLoop(
        b, loc, op.getA(), blockwiseLoadACoords, aLoadIntermediate, aLoadType,
        blockwiseCopyABounds, blockwiseVectorDimA, useIndexDiffs);

    SmallVector<Value, 4> blockwiseLoadBCoords;
    if (KPack > 1) {
      blockwiseLoadBCoords = {GemmBlockCoord_G, GemmBBlockCopySourceCoord_Z,
                              GemmBBlockCopySourceCoord_Y,
                              GemmBBlockCopySourceCoord_X};
    } else {
      blockwiseLoadBCoords = {GemmBlockCoord_G, GemmBBlockCopySourceCoord_Y,
                              GemmBBlockCopySourceCoord_X};
    }
    // Emit blockwise load for matrix B.
    TransformingForOp blockwiseLoadB = createGlobalLoadLoop(
        b, loc, op.getB(), blockwiseLoadBCoords, bLoadIntermediate, bLoadType,
        blockwiseCopyBBounds, blockwiseVectorDimB, useIndexDiffs);

    SmallVector<Value, 4> blockwiseStoreACoords;
    if (KPack > 1) {
      blockwiseStoreACoords = {zeroConstantOp, GemmABlockCopyDestCoord_Z,
                               GemmABlockCopyDestCoord_Y,
                               GemmABlockCopyDestCoord_X};
    } else {
      blockwiseStoreACoords = {zeroConstantOp, GemmABlockCopyDestCoord_Y,
                               GemmABlockCopyDestCoord_X};
    }
    // Emit blockwise store for matrix A.
    TransformingForOp blockwiseStoreA = createLdsStoreLoop(
        b, loc, blockwiseLoadA.getResult(0), ldsMatrixASubviewOp,
        blockwiseStoreACoords, blockwiseCopyABounds);

    SmallVector<Value, 4> blockwiseStoreBCoords;
    if (KPack > 1) {
      blockwiseStoreBCoords = {zeroConstantOp, GemmBBlockCopyDestCoord_Z,
                               GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    } else {
      blockwiseStoreBCoords = {zeroConstantOp, GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    }
    // Emit blockwise_store for matrix B.
    TransformingForOp blockwiseStoreB = createLdsStoreLoop(
        b, loc, blockwiseLoadB.getResult(0), ldsMatrixBSubviewOp,
        blockwiseStoreBCoords, blockwiseCopyBBounds);

    // -----

    // Logic to do XDLOPS code selection.
    XdlopsCodeSelection xcs =
        XdlopsCodeSelection::get(elementType, MPerWave, NPerWave, b);

    // Extract values from XdlopsCodeSelection.
    int64_t MRepeats = xcs.MRepeats;
    int64_t NRepeats = xcs.NRepeats;
    int64_t mPerRepeat = MPerWave / MRepeats;
    int64_t nPerRepeat = NPerWave / NRepeats;

    VectorType vectorType = xcs.vectorType;
    int64_t nResultVectors = xcs.nResultVectors;
    int64_t rowGroupSize = xcs.rowGroupSize;
    int64_t rowGroupsPerBlock = xcs.rowGroupsPerBlock;
    int64_t inputSpanLen = xcs.inputSpanLen;
    int64_t inputSpansPerMfmaIn = xcs.inputSpansPerMfmaIn;
    int64_t blocksInOutRegs = xcs.blocksInOutRegs;
    int64_t m = xcs.mfmaNonKDim;
    // Note n has the 4x4 => 4x64 behavior that necessitated inputSpansPerMfmaIn
    int64_t n = xcs.inputSpanLen;
    int64_t k_base = xcs.k_base;

    int64_t blocksPerRepeat = (mPerRepeat * nPerRepeat) / (m * n);
    // -----

    // Logic to setup blockwise_gemm_v2 parameters.
    //
    // Original C++ logic:
    // index_t mMyWaveOffsetA;
    // index_t mMyWaveOffsetB;
    // const index_t waveId   = get_thread_local_1d_id() / WaveSize;
    // const index_t waveId_m = waveId / GemmNWaves;
    // const index_t waveId_n = waveId % GemmNWaves;
    // mMyWaveOffsetA = waveId_m * GemmMPerWave;
    // mMyWaveOffsetB = waveId_n * GemmNPerWave;
    auto waveId = b.create<DivUIOp>(loc, tid, waveSizeConstantOp);
    auto waveId_m = b.create<DivUIOp>(loc, waveId, NWavesConstantOp);
    auto waveId_n = b.create<RemUIOp>(loc, waveId, NWavesConstantOp);

    Value mMyWaveOffsetA, mMyWaveOffsetB;
    mMyWaveOffsetA = b.create<MulIOp>(loc, waveId_m, MPerWaveConstantOp);
    mMyWaveOffsetB = b.create<MulIOp>(loc, waveId_n, NPerWaveConstantOp);

    // Logic to setup buffers for blockwise_gemm_v2.

    bool IsKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);
    int64_t arrayASize = (!IsKReduction)
                             ? (KPerBlock * MRepeats)
                             : (KPerBlock / inputSpansPerMfmaIn * MRepeats);
    int64_t arrayBSize = (!IsKReduction)
                             ? (KPerBlock * NRepeats)
                             : (KPerBlock / inputSpansPerMfmaIn * NRepeats);
    Type arrayAType, arrayBType;
    if (KPack > 1) {
      // Should pack at least k_base elements and avoid waste xdlopsgemm
      // cycles
      if (KPack < k_base) {
        return failure();
      }

      // When reduction, KPerBlock must be at least num_input_blks
      if (IsKReduction && KPerBlock < inputSpansPerMfmaIn) {
        return failure();
      }

      arrayAType =
          MemRefType::get({arrayASize}, VectorType::get({KPack}, elementType),
                          {}, gpu::GPUDialect::getPrivateAddressSpace());
      arrayBType =
          MemRefType::get({arrayBSize}, VectorType::get({KPack}, elementType),
                          {}, gpu::GPUDialect::getPrivateAddressSpace());
    } else {
      // When non-reduction, KPerBlock must be at least k_base
      if (!IsKReduction && KPerBlock < k_base) {
        return failure();
      }

      // When reduction, KPerBlock must be at least k_base * num_input_blks
      if (IsKReduction && KPerBlock < k_base * inputSpansPerMfmaIn) {
        return failure();
      }

      arrayAType = MemRefType::get({arrayASize}, elementType, {},
                                   gpu::GPUDialect::getPrivateAddressSpace());
      arrayBType = MemRefType::get({arrayBSize}, elementType, {},
                                   gpu::GPUDialect::getPrivateAddressSpace());
    }
    auto arrayA = b.create<GpuAllocOp>(loc, arrayAType);
    auto arrayB = b.create<GpuAllocOp>(loc, arrayBType);

    // -----
    // Logic to allocate 0-initialized vectors for C.
    int64_t regCVectorLen = vectorType.getNumElements();
    Type destType = op.getC().getType().getElementType();
    Type accumulatorType = obtainAccumulatorType(b, elementType, destType);
    VectorType accumulatorVectorType =
        vectorType.cloneWith({}, accumulatorType);
    MemRefType regCAllocType = MemRefType::get(
        nResultVectors, accumulatorVectorType, {},
        /*memorySpace=*/gpu::GPUDialect::getPrivateAddressSpace());
    Value regCAllocOp = b.create<rock::GpuAllocOp>(loc, regCAllocType);

    Value zeroConstantCOp = createZeroConstantOp(b, loc, vectorType);
    b.create<FillOp>(loc, regCAllocOp, zeroConstantCOp);

    // Emit loop.
    int64_t loopIteration = (K - KPerBlock) / KPerBlock;

    // Assign iter args.
    // 0: blockwise copy A src y coordinate.
    // 1: blockwise copy B src y coordinate.
    // 2-x : vectorCs.
    SmallVector<Value, 6> iterArgs = {blockwiseLoadACoords[1],
                                      blockwiseLoadBCoords[1]};

    auto mfmaLoopOp = b.create<AffineForOp>(loc, 0, loopIteration, 1, iterArgs);

    // inside the loop.
    auto mfmalb = OpBuilder::atBlockBegin(mfmaLoopOp.getBody());

    const auto &mfmalArgs = mfmaLoopOp.getRegionIterArgs();

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    Value blockwiseCopyASrcUpdated =
        mfmalb.create<AddIOp>(loc, mfmalArgs[0], KPerBlockConstantOp);
    BlockAndValueMapping loadAUpdates;
    loadAUpdates.map(blockwiseLoadACoords[1], blockwiseCopyASrcUpdated);
    auto blockwiseLoadAClone = cast<TransformingForOp>(
        mfmalb.clone(*blockwiseLoadA.getOperation(), loadAUpdates));

    // Emit blockwise load for matrix B.
    BlockAndValueMapping loadBUpdates;
    Value blockwiseCopyBSrcUpdated =
        mfmalb.create<AddIOp>(loc, mfmalArgs[1], KPerBlockConstantOp);
    loadBUpdates.map(blockwiseLoadBCoords[1], blockwiseCopyBSrcUpdated);
    auto blockwiseLoadBClone = cast<TransformingForOp>(
        mfmalb.clone(*blockwiseLoadB.getOperation(), loadBUpdates));

    // LDS barrier : guarantees LDS update completion before reading out to
    // register. requires LDS fence + barrier.
    mfmalb.create<LDSBarrierOp>(loc);

    // Emit blockwise V2 GEMM.
    // The xdlops gemms take a 1D buffer because reasons
    mfmalb.create<BlockwiseGemmV2Op>(
        loc, ldsGpuAllocOp, ldsGpuAllocOp, b.getIndexAttr(ldsBlockAOffset),
        b.getIndexAttr(ldsBlockBOffset), mMyWaveOffsetA, mMyWaveOffsetB, arrayA,
        arrayB, regCAllocOp, op.getBlockSizeAttr(), op.getParamsAttr());

    // LDS barrier : defer the next LDS update until this round's GEMM
    // calculation is done. requires barrier only.
    mfmalb.create<LDSBarrierOp>(loc);

    // Blockwise copy from register (naive tensor) to LDS (naive tensor).
    // Emit blockwise stores
    BlockAndValueMapping storeAUpdates, storeBUpdates;
    storeAUpdates.map(blockwiseLoadA.getResult(0),
                      blockwiseLoadAClone.getResult(0));
    storeBUpdates.map(blockwiseLoadB.getResult(0),
                      blockwiseLoadBClone.getResult(0));
    mfmalb.clone(*blockwiseStoreA.getOperation(), storeAUpdates);
    mfmalb.clone(*blockwiseStoreB.getOperation(), storeBUpdates);

    // Update iter args.
    // blockwiseCopyASrcVector and blockwiseCopyBSrcVector are updated.
    iterArgs[0] = blockwiseCopyASrcUpdated;
    iterArgs[1] = blockwiseCopyBSrcUpdated;

    // emit loop yield so iter args can be passed to the next iteration.
    mfmalb.create<AffineYieldOp>(loc, iterArgs);
    // outside the loop.

    // Emit loop tail.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    auto blockwiseGemmV2TailOp = b.create<BlockwiseGemmV2Op>(
        loc, ldsGpuAllocOp, ldsGpuAllocOp, b.getIndexAttr(ldsBlockAOffset),
        b.getIndexAttr(ldsBlockBOffset), mMyWaveOffsetA, mMyWaveOffsetB, arrayA,
        arrayB, regCAllocOp, op.getBlockSizeAttr(), op.getParamsAttr());

    // -----

    // Matrix C write out logic.
    const auto &tailResults = blockwiseGemmV2TailOp->getResults();
    int64_t wavesInKernelBlock = blockSize / waveSize;

    int64_t numElements = regCVectorLen * nResultVectors;
    TopDownTMBuilder splitMemoryCoords(b, {"bid", "tid", "item"},
                                       {gridSize, blockSize, numElements}, loc);
    splitMemoryCoords.merge(
        {"g", "n", "m"}, {0, 1, 2}, {"bid"},
        {gridSize / GStride, GStride / MBlockWork, MBlockWork});
    splitMemoryCoords.merge(
        {"wave", "m_tid", "n_tid"}, {3, 4, 5}, "tid",
        {wavesInKernelBlock, waveSize / inputSpanLen, inputSpanLen});
    splitMemoryCoords.merge(
        {"i", "j", "vec_group", "vec_item"}, {6, 7, 8, 9}, "item",
        {numElements / (blocksPerRepeat * rowGroupsPerBlock * rowGroupSize),
         blocksPerRepeat, rowGroupsPerBlock, rowGroupSize});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

    // "blkMajor" and "blkMinor" are placeholder names because we don't know if
    // they'll be column or row until we check for broadcast-ness.
    auto toRowsAndCols =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    llvm::StringMap<uint32_t> rowsAndColsIdxs = expandNamesInPlace(
        splitMemoryCoords, {{"wave", {"wave_m", "wave_n"}},
                            {"i", {"m_i", "n_i"}},
                            {"j", {"blkMajor", "blkMinor"}}});
    TopDownTMBottomDimsWrapper rowsAndColsWrap(toRowsAndCols, rowsAndColsIdxs);
    rowsAndColsWrap.passThrough({"g", "m", "n"});
    rowsAndColsWrap.merge({"wave_m", "wave_n"}, "wave",
                          {wavesInKernelBlock / NWaves, NWaves});
    rowsAndColsWrap.passThrough({"m_tid", "n_tid"});
    rowsAndColsWrap.merge(
        {"m_i", "n_i"}, "i",
        {splitMemoryCoords.endSize("i") / NRepeats, NRepeats});

    // Here we use the full builder API since we want index and name control
    bool isABroadcast = (nPerRepeat >= mPerRepeat);
    SmallVector<StringRef, 2> rowsFirst = {"blk_row", "blk_col"};
    SmallVector<StringRef, 2> colsFirst = {"blk_col", "blk_row"};
    toRowsAndCols.merge(
        isABroadcast ? rowsFirst : colsFirst,
        {rowsAndColsIdxs["blkMajor"], rowsAndColsIdxs["blkMinor"]}, "j",
        {splitMemoryCoords.endSize("j") / blocksInOutRegs, blocksInOutRegs});
    toRowsAndCols.passThrough(
        {"vec_group", "vec_item"},
        {rowsAndColsIdxs["vec_group"], rowsAndColsIdxs["vec_item"]},
        {"vec_group", "vec_item"});

    TransformMapAttr toRowsAndColsAttr = toRowsAndCols.get();

    auto toMatrixC = TopDownTMBuilder::below(toRowsAndCols, toRowsAndColsAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {"g"});

    toMatrixC.embed(
        "gemmM", 1, M,
        {"m", "wave_m", "m_tid", "m_i", "blk_row", "vec_group", "vec_item"},
        {MPerBlock, MPerWave, rowGroupSize, mPerRepeat, m,
         inputSpansPerMfmaIn * rowGroupSize, 1});
    toMatrixC.embed("gemmN", 2, N, {"n", "wave_n", "n_i", "blk_col", "n_tid"},
                    {NPerBlock, NPerWave, nPerRepeat, n, 1});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();

    ArrayAttr idToMatrixCMaps = b.getArrayAttr(
        {splitMemoryCoordsAttr, toRowsAndColsAttr, toMatrixCAttr});
    Value tensorC;
    ArrayAttr idToTensorCMaps;
    std::tie(tensorC, idToTensorCMaps) =
        untransform(b, op.getC(), idToMatrixCMaps);

    constexpr int64_t swizzleGroup = 4;
    ArrayRef<int64_t> tensorCShape =
        tensorC.getType().cast<MemRefType>().getShape();
    int64_t tensorCDataPerCopy = getMaxVectorization(idToTensorCMaps, /*dim=*/2,
                                                     numElements, tensorCShape);
    int64_t threadsWithConsecutiveElems = getMaxVectorization(
        idToTensorCMaps, /*dim=*/1, swizzleGroup, tensorCShape);
    bool enableOutSwizzles = (tensorCDataPerCopy == 1) &&
                             (threadsWithConsecutiveElems == swizzleGroup);
    if (enableOutSwizzles) {
      // Add the coordinate transformations that reflect the transpose we'll be
      // doing in the emitted kernel.
      tensorCDataPerCopy = threadsWithConsecutiveElems;
      auto indexSplit = TopDownTMBuilder(
          b, {"bid", "tid", "iter"}, {gridSize, blockSize, numElements}, loc);
      indexSplit.passThrough("bid");
      indexSplit.merge({"tid_group", "tid_item"}, {1, 2}, "tid",
                       {blockSize / 4, 4});
      indexSplit.merge({"vec_group", "vec_item"}, {3, 4}, "iter",
                       {numElements / 4, 4});
      TransformMapAttr indexSplitAttr = indexSplit.get();

      // Note that we switch the positions of tid_item and vec_item when
      // recombining the coordinates.
      auto indexCombine = TopDownTMBuilder::below(indexSplit, indexSplitAttr);
      indexCombine.passThrough("bid");
      indexCombine.embed("tid", 1, blockSize, {"tid_group", "vec_item"},
                         {4, 1});
      indexCombine.embed("iter", 2, numElements, {"vec_group", "tid_item"},
                         {4, 1});
      TransformMapAttr indexCombineAttr = indexCombine.get();

      SmallVector<Attribute, 8> newTransforms = {indexSplitAttr,
                                                 indexCombineAttr};
      llvm::copy(idToTensorCMaps, std::back_inserter(newTransforms));
      idToTensorCMaps = b.getArrayAttr(newTransforms);
    }

    // Make the vector slice starting point jump in units of the vectorization.
    TopDownTMBuilder correctVectorCoords(
        b, {"bid", "tid", "item"}, {gridSize, blockSize, numElements}, loc);
    correctVectorCoords.ignore("bid");
    correctVectorCoords.ignore("tid");
    correctVectorCoords.passThrough({"index"}, {0}, {"item"});
    TransformMapAttr correctVectorCoordsAttr = correctVectorCoords.get();

    // Having set up the maps from [block, thread, i] space to gemm space,
    // do all the prep work to make the copy loop correct.

    // Emit vector swizzles if applicable
    SmallVector<Value, 4> transformedTail;
    transformedTail.reserve(tailResults.size());

    if (enableOutSwizzles) {
      Value laneId = b.create<arith::RemUIOp>(loc, tid, waveSizeConstantOp);
      for (int i = 0; i < nResultVectors; ++i) {
        Value indexOp = b.createOrFold<ConstantIndexOp>(loc, i);
        Value loaded =
            b.create<memref::LoadOp>(loc, vectorType, regCAllocOp, indexOp);
        Value swizzle = b.create<InWarpTransposeOp>(
            loc, vectorType, loaded, laneId, b.getI32IntegerAttr(rowGroupSize),
            b.getI32ArrayAttr({0, 1, 2, 3}));
        transformedTail.push_back(swizzle);
        b.create<memref::StoreOp>(loc, swizzle, regCAllocOp, indexOp);
      }
    } else {
      llvm::copy(tailResults, std::back_inserter(transformedTail));
    }

    Value registerC = regCAllocOp;
    auto convertedCType = MemRefType::get(
        numElements, destType, {},
        /*memorySpace=*/gpu::GPUDialect::getPrivateAddressSpace());
    Value convertedC = b.create<rock::GpuAllocOp>(loc, convertedCType);

    BottomUpTMBuilder toRegCScalar(b, {"scalar"}, {numElements}, loc);
    toRegCScalar.embed({"vector"}, {0}, {nResultVectors}, "scalar",
                       {regCVectorLen});
    TransformMapAttr toRegCScalarAttr = toRegCScalar.get();

    // Convert from memref<?xvector<?xT>> to memref<?xT> where the source T
    // is the accumulatorType and destination type is destType
    auto convertLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{{zeroConstantOp}, {zeroConstantOp}},
        ArrayRef<Attribute>{b.getArrayAttr({}),
                            b.getArrayAttr(toRegCScalarAttr)},
        /*bounds=*/regCAllocType.getShape(), /*strides=*/llvm::None,
        /*useIndexDiffs=*/true, /*forceUnroll=*/true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(convertLoop.getBody());
      Value loaded =
          b.create<memref::LoadOp>(loc, accumulatorVectorType, registerC,
                                   convertLoop.getLowerCoords(/*domain*/ 0));
      Value cast = loaded;
      if (destType != accumulatorType) {
        VectorType destVectorType = vectorType.clone(destType);
        cast = createTypeConversionOp(b, loc, loaded, destVectorType);
      }
      b.create<InBoundsStoreOp>(loc, cast, convertedC,
                                convertLoop.getLowerCoords(/*domain*/ 1));
    }
    registerC = convertedC;

    auto writeOobDims = computeOobFromTransforms(b, idToTensorCMaps);

    SmallVector<Value, 3> writeStartCoords = {bid, tid, zeroConstantOp};

    auto outLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{writeStartCoords, writeStartCoords},
        ArrayRef<Attribute>{b.getArrayAttr({correctVectorCoordsAttr}),
                            idToTensorCMaps},
        ArrayRef<int64_t>{1, 1, numElements},
        ArrayRef<int64_t>{1, 1, tensorCDataPerCopy},
        /*forceUnroll=*/true, /*useIndexDiffs=*/useIndexDiffs);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(outLoop.getBody());
      b.create<ThreadwiseCopyV2Op>(
          loc, registerC, tensorC, b.getIndexAttr(tensorCDataPerCopy),
          op.getStoreMethodAttr(), std::get<0>(writeOobDims),
          std::get<1>(writeOobDims), outLoop.getLowerCoords(/*domain=*/0)[0],
          outLoop.getLowerCoords(/*domain=*/1));
    }

    b.eraseOp(op);
    return success();
  }
};
} // end anonymous namespace

void RockGridwiseGemmToBlockwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::GridwiseGemmOp, rock::GridwiseGemmV2Op>();
  target.addLegalDialect<arith::ArithmeticDialect, rock::RockDialect,
                         memref::MemRefDialect, AffineDialect,
                         vector::VectorDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmRewritePattern, GridwiseGemmV2RewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }

  OpPassManager cleanupPasses("func.func");
  cleanupPasses.addPass(mlir::createCanonicalizerPass());
  (void)runPipeline(cleanupPasses, getOperation());
}
