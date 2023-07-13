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
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "AccelEmitter.h"
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

static Type obtainAccumulatorType(OpBuilder &b, Type elementTypeA,
                                  Type elementTypeB, Type destType) {
  // Determine the type used on VGPR to act as accumulator.
  // f32: f32.
  // f16, bf16: f32 to prevent overflow from happening.
  // i16 : i16.
  // fp8 (any combo) : f32.
  // i8: i32, since we have an i32 output
  Type accumulatorType = destType;
  if (elementTypeA.isF16() || elementTypeA.isBF16() ||
      elementTypeA.isFloat8E5M2FNUZ() || elementTypeA.isFloat8E4M3FNUZ()) {
    accumulatorType = b.getF32Type();
  } else if (elementTypeA.isInteger(8)) {
    accumulatorType = b.getI32Type();
  }
  return accumulatorType;
}

/// Construct a `memref.view` operation that interprets the buffer `buffer`,
/// whose elements are bytes, as a buffer of `type`.
static TypedValue<MemRefType> viewBufferAs(OpBuilder &b, Value buffer,
                                           Type type) {
  Location loc = buffer.getLoc();
  Value zeroByteOffset = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  auto bufferType = buffer.getType().cast<MemRefType>();
  int64_t byteWidth = getByteWidth(type);
  int64_t numBytes = bufferType.getShape()[0];
  assert(numBytes % byteWidth == 0 && "Can't evenly fit type into buffer");
  int64_t length = numBytes / byteWidth;
  auto newBufferType = bufferType.cloneWith({length}, type);
  auto view =
      b.create<memref::ViewOp>(loc, newBufferType, buffer, zeroByteOffset,
                               /*dynamic dim sizes=*/ValueRange{});
  return TypedValue<MemRefType>(view.getResult());
}
} // end anonymous namespace

/// Given a copy layout <copyDPerThread, copyKPerThread>, come up with the best
/// vectorization strategy for the layout. For instance, if the layout is <D,K>
/// = <2,16> and K is contiguous, we will vectorize by 16 along K and we will
/// loop over the other dimension
static std::pair<GemmDimension, int64_t>
bestGlobalVectorization(OpBuilder &b, Value matrix, int64_t copyDPerThread,
                        int64_t copyKPerThread, GemmDimension tiebreaker,
                        int64_t kPerBlock, int64_t dPerBlock,
                        Type elementType) {
  Value tensor;
  ArrayAttr transforms;
  std::tie(tensor, transforms) = untransform(b, matrix);
  ArrayRef<int64_t> tensorShape =
      tensor.getType().cast<MemRefType>().getShape();
  int64_t kVectorLen = getMaxVectorizationForDatatype(
      transforms, static_cast<uint32_t>(GemmDimension::K),
      math_util::gcd(copyKPerThread * copyDPerThread, kPerBlock), tensorShape,
      elementType);

  int64_t dVectorLen = getMaxVectorizationForDatatype(
      transforms, static_cast<uint32_t>(GemmDimension::MorN),
      math_util::gcd(copyDPerThread * copyKPerThread, dPerBlock), tensorShape,
      elementType);

  if (kVectorLen > dVectorLen) {
    kVectorLen = math_util::gcd(kVectorLen, copyKPerThread);
    return {GemmDimension::K, kVectorLen};
  }

  if (dVectorLen > kVectorLen) {
    dVectorLen = math_util::gcd(dVectorLen, copyDPerThread);
    return {GemmDimension::MorN, dVectorLen};
  }

  return {tiebreaker, kVectorLen};
}

/// Compute a thread copy layout, i.e., how many elements a single thread (or
/// workitem) reads along K and M (independently on how we vectorize the reads)
static FailureOr<std::pair<int64_t, int64_t>>
computeCopyPerThread(Type elementType, int64_t copyPerThread, int64_t kPerBlock,
                     int64_t dPerBlock, int64_t kpack, Location loc) {

  // By default, we try to maximize the LDS store vectorization. So we will try
  // to read as many elements as possible along the contiguous dimension in LDS
  // and `copyPerThread/elements` in the other dimension
  int64_t maxVlen = 128 / elementType.getIntOrFloatBitWidth();
  int64_t copyKPerThread = 0;
  int64_t copyDPerThread = 0;

  if (kpack == 1) {
    copyDPerThread = math_util::gcd(maxVlen, copyPerThread);
    copyKPerThread = copyPerThread / copyDPerThread;
  } else {
    copyKPerThread =
        math_util::gcd(maxVlen, math_util::gcd(kpack, copyPerThread));
    copyDPerThread = copyPerThread / copyKPerThread;
  }

  if (copyKPerThread == 0 || copyDPerThread == 0) {
    return emitError(loc) << "gemmA copy size too small,"
                          << " copyKPerThread: " << copyKPerThread
                          << " copyDPerThread: " << copyDPerThread << "\n";
  }
  if (kPerBlock < copyKPerThread || dPerBlock < copyDPerThread) {
    return mlir::emitError(loc)
           << "gemmA per thread copy smaller than per"
           << " block copy, incohereant tuning parameters\n";
  }
  return std::make_pair(copyKPerThread, copyDPerThread);
}

/// Applies the transforms that take a G x K x D matrix to a k_iter x bid x tid
/// x iter value suitable for using in a global load loop. `dName` should be "m"
/// and "n", and is used to make the maps have the right names for debugging.
///
/// bidGridOrder should
/// contain the strings "g_block", "m_block", and "n_block" in some order
/// indicating how the block ID is to be partitioned into offsets (last element
/// moves fastest) and bidGridLengths should be the lengths of those three
/// dimensions. This is needed because the accelerated and non-accelerated gemms
/// partition their block ID in different orders.
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

  if (vectorDim == GemmDimension::K) {
    splitId.merge({dThreadName, "k_thread"}, {4, 5}, "tid",
                  {dThreads, kThreads});
    splitId.merge({dIterName, "k_iter"}, {6, 7}, "iter",
                  {dPerThread, kPerThread});
  } else {
    splitId.merge({"k_thread", dThreadName}, {4, 5}, "tid",
                  {kThreads, dThreads});
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

/// Wraps the LDS buffer "buffer", which is <kOuter * d * kpack * sizeof(T) x i8
/// into a tid x iter view, where `iter` iterates over nominal scalar indices
/// into a buffer of type T. `buffer` will be reinterpreted as a buffer with
/// element type vector<kpackPerThread x T> (with kpackPerThread == 1 meaning
/// just T). The resulting view must be iterated over with a stride of no less
/// than min(kPerThread, kpack).
static FailureOr<Value> wrapLDSBufferForStore(OpBuilder &b, Location loc,
                                              Value buffer, Type ldsReadType,
                                              int64_t kOuter, StringRef dName,
                                              int64_t d, int64_t kPerThread,
                                              int64_t dPerThread,
                                              GemmDimension vectorDim) {
  MemRefType bufferType = buffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> bufferShape = bufferType.getShape();
  Type dataType = ldsReadType;
  if (bufferShape.size() != 1)
    return emitError(loc, "Expected a flat buffer");
  int64_t kpack = 1;
  if (auto vectorDataType = dataType.dyn_cast<VectorType>()) {
    kpack = vectorDataType.getNumElements();
    dataType = vectorDataType.getElementType();
  }

  if (bufferShape[0] != kOuter * d * kpack * getByteWidth(dataType))
    return emitError(loc, "LDS buffer should have ")
           << kOuter * d * kpack * getByteWidth(dataType)
           << " elements but has " << bufferShape[0];

  int64_t kpackPerThread = std::min(kPerThread, kpack);
  int64_t kOuterPerThread = kPerThread / kpackPerThread;
  int64_t threadsPerKpack = kpack / kpackPerThread;

  Type ldsWriteType = vectorTypeOrSelf(dataType, kpackPerThread);
  auto typedBuffer = viewBufferAs(b, buffer, ldsWriteType);
  BottomUpTMBuilder reshapeBuf(b, {"raw"}, typedBuffer.getType().getShape(),
                               loc);
  reshapeBuf.unmerge({"k_outer", dName, "kpack_idx"}, {0, 1, 2}, "raw",
                     {kOuter, d, threadsPerKpack});
  // Add this throwaway dimension so that when we're iterating the scalar
  // packing buffer in a vectorized way, the always-zero index gets thrown on
  // the floor.
  reshapeBuf.addDim("kpack_vec", 3, kpackPerThread);
  TransformMapAttr reshapeBufAttr = reshapeBuf.get();
  Value reshaped = b.create<TransformOp>(loc, typedBuffer, reshapeBufAttr);

  auto mergeKpack = BottomUpTMBuilder::above(reshapeBuf, reshapeBufAttr);
  mergeKpack.passThrough({"k_outer", dName});
  mergeKpack.merge("kpack", 2, {"kpack_idx", "kpack_vec"});
  TransformMapAttr mergeKpackAttr = mergeKpack.get();
  Value asMatrix = b.create<TransformOp>(loc, reshaped, mergeKpackAttr);

  SmallString<8> dThreadName = llvm::formatv("{0}_thread", dName);
  SmallString<8> dIterName = llvm::formatv("{0}_iter", dName);
  auto tidIterSplit = BottomUpTMBuilder::above(mergeKpack, mergeKpackAttr);
  tidIterSplit.unmerge({"k_thread", "k_iter"}, {0, 1}, "k_outer",
                       {kOuter / kOuterPerThread, kOuterPerThread});
  tidIterSplit.unmerge({dThreadName, dIterName}, {2, 3}, dName,
                       {d / dPerThread, dPerThread});
  tidIterSplit.unmerge({"kpack_thread", "kpack_iter"}, {4, 5}, "kpack",
                       {kpack / kpackPerThread, kpackPerThread});
  TransformMapAttr tidIterSplitAttr = tidIterSplit.get();
  Value withTidIterSplit =
      b.create<TransformOp>(loc, asMatrix, tidIterSplitAttr);

  auto tidIter = BottomUpTMBuilder::above(tidIterSplit, tidIterSplitAttr);
  if (vectorDim == GemmDimension::K) {
    tidIter.merge("tid", 0, {dThreadName, "k_thread", "kpack_thread"});
  } else {
    tidIter.merge("tid", 0, {"k_thread", "kpack_thread", dThreadName});
  }
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
  int64_t kpackPerThread = std::min(kPerThread, kpack);
  int64_t kOuter = kPerThread / kpackPerThread;

  int64_t dataPerThread = kPerThread * dPerThread;

  TopDownTMBuilder splitIter(b, {"iter"}, {dataPerThread});
  if (vectorDim == GemmDimension::K)
    splitIter.merge({dName, "k", "kpack_thread"}, {0, 1, 2}, "iter",
                    {dPerThread, kOuter, kpackPerThread});
  else
    splitIter.merge({"k", "kpack_thread", dName}, {0, 1, 2}, "iter",
                    {kOuter, kpackPerThread, dPerThread});
  TransformMapAttr splitIterAttr = splitIter.get();

  auto toVector = TopDownTMBuilder::below(splitIter, splitIterAttr);
  toVector.unmerge("raw", 0, {"k", dName, "kpack_thread"},
                   {kOuter, dPerThread, kpackPerThread});
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
createGlobalLoadLoop(PatternRewriter &b, Location loc, GpuAllocOp loadBuffer,
                     Value wrappedMatrix, ArrayAttr vectorMap,
                     int64_t dataPerThread, int64_t vectorLen, Value bid,
                     Value tid, bool forceUnroll) {
  Value tensor;
  ArrayAttr matrixToTensor;
  std::tie(tensor, matrixToTensor) = untransform(b, wrappedMatrix);

  // Optimize the transform chain.
  ArrayRef<int64_t> tensorSize = tensor.getType().cast<ShapedType>().getShape();
  matrixToTensor = collapseContiguousMerges(matrixToTensor, tensorSize);

  Type elementType =
      wrappedMatrix.getType().cast<MemRefType>().getElementType();
  Type loadType = vectorTypeOrSelf(elementType, vectorLen);
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

  SmallVector<Value, 4> globalStart = {zero, bid, tid, zero};
  SmallVector<Value, 4> vectorStartOuter(4, zero);
  auto outerLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{globalStart, vectorStartOuter},
      ArrayRef<Attribute>{matrixToTensor, b.getArrayAttr({})},
      /*bounds=*/ArrayRef<int64_t>{1, 1, 1, dataPerThread},
      /*strides=*/ArrayRef<int64_t>{1, 1, 1, vectorLen}, forceUnroll,
      /*useIndexDiffs=*/true);
  {
    PatternRewriter::InsertionGuard outerGuard(b);
    b.setInsertionPointToStart(outerLoop.getBody());

    Value loaded = b.create<GlobalLoadOp>(
        loc, loadType, tensor, outerLoop.getValidity(/*domain=*/0),
        outerLoop.getLowerCoords(/*domain=*/0));

    b.create<InBoundsStoreOp>(loc, loaded, loadBuffer,
                              outerLoop.getLowerCoords(/*domain*/ 1)[3]);
  }
  return outerLoop;
}

// This function will create a DPerThread x KPerThread view of loaded register
// buffer that may be laid out KPerThread x DPerThread or DPerThread x
// KPerThread depending on the direction of the global vectorization.
Value viewLoadBufferDK(PatternRewriter &b, Location loc, Value loadBuffer,
                       GemmDimension vectorDim, int64_t copyDPerThread,
                       int64_t copyKPerThread) {
  SmallVector<StringRef, 2> loadBufferNames;
  SmallVector<int64_t, 2> loadBufferShape;
  if (vectorDim == GemmDimension::MorN) {
    // If we are vectorizing along the M/N dimension, we have a
    // KxD buffer that we want to transpose into a DxK buffer
    loadBufferShape = {copyKPerThread, copyDPerThread};
    loadBufferNames = {"k_physical", "d_physical"};
  } else {
    // If we are vectorizing along the K dimension, we have a
    // DxK buffer that we want to transpose into a KxD buffer
    loadBufferShape = {copyDPerThread, copyKPerThread};
    loadBufferNames = {"d_physical", "k_physical"};
  }
  assert(loadBuffer.getType().cast<MemRefType>().getNumElements() ==
         copyKPerThread * copyDPerThread);

  Value ret;
  BottomUpTMBuilder rawViewBuilder(b, {"rawLoad"},
                                   {copyKPerThread * copyDPerThread});
  rawViewBuilder.unmerge(loadBufferNames, {0, 1}, "rawLoad", loadBufferShape);
  TransformMapAttr rawView = rawViewBuilder.get();
  ret = b.create<TransformOp>(loc, loadBuffer, rawView);

  BottomUpTMBuilder kdViewBuilder =
      BottomUpTMBuilder::above(rawViewBuilder, rawView);
  kdViewBuilder.passThrough({"d", "k"}, {0, 1}, {"d_physical", "k_physical"});
  TransformMapAttr kdView = kdViewBuilder.get();
  ret = b.create<TransformOp>(loc, ret, kdView);

  return ret;
}

/// This function pack the load buffer into a store buffer ready to be copied
/// into LDS:
///  - The load buffer is (viewed as) a DPerThread x KPerThread
///  - The store buffer needs to be packed as a [KOuterPerThread, dPerThread,
///  kpackPerThread]
///    buffer
TransformingForOp packLoadBufferToStoreBuffer(PatternRewriter &b, Location loc,
                                              Type elementType, int64_t kpack,
                                              Value loadBuffer,
                                              Value storeBuffer) {
  ArrayRef<int64_t> loadShape =
      loadBuffer.getType().cast<ShapedType>().getShape();
  Type elemType = loadBuffer.getType().cast<MemRefType>().getElementType();
  int64_t copyDPerThread = loadShape[0];
  int64_t copyKPerThread = loadShape[1];
  // We use kpackPerThread instead of kpack to cover edge cases where
  // copyKPerThread is smaller than kpack
  int64_t kpackPerThread = std::min(copyKPerThread, kpack);
  int64_t kOuterPerThread = copyKPerThread / kpackPerThread;

  TopDownTMBuilder packStore(b, {"d", "k"}, {copyDPerThread, copyKPerThread});
  packStore.merge({"kouter", "kpack"}, {0, 2}, "k",
                  {kOuterPerThread, kpackPerThread});
  packStore.passThrough({"dPerThread"}, 1, {"d"});
  TransformMapAttr packStoreAttr = packStore.get();
  auto transformPacked = TopDownTMBuilder::below(packStore, packStoreAttr);
  transformPacked.unmerge("rawStore", 0, {"kouter", "dPerThread", "kpack"},
                          {kOuterPerThread, copyDPerThread, kpackPerThread});
  TransformMapAttr transformPackedAttr = transformPacked.get();
  auto storeIdx = b.getArrayAttr({packStoreAttr, transformPackedAttr});

  Value rawLoadBuffer;
  ArrayAttr loadBufferView;
  std::tie(rawLoadBuffer, loadBufferView) = untransform(b, loadBuffer);
  ArrayRef<int64_t> rawLoadBufferShape =
      rawLoadBuffer.getType().cast<ShapedType>().getShape();

  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 2> start(2, zero);
  SmallVector<int64_t, 2> strides(2, 1);
  int64_t vecLen = 1;
  // The store buffer is a flattened < kouter x dPerThread x kpack >.
  if (kpackPerThread == 1) {
    // if kpack == 1, then we can do vectorized loads across d dimension from/to
    // load/store buffer
    vecLen = getMaxVectorizationForDatatype(loadBufferView, /*dim=*/0,
                                            copyDPerThread, rawLoadBufferShape,
                                            elemType);
    vecLen = math_util::gcd(copyDPerThread, vecLen);
    strides[0] = vecLen;
  } else {
    // if kpack > 1, then we are limited by vectorization in k dimension and it
    // could be at most kpack.
    vecLen = getMaxVectorizationForDatatype(loadBufferView, /*dim=*/1,
                                            copyKPerThread, rawLoadBufferShape,
                                            elemType);
    vecLen = math_util::gcd(vecLen, kpackPerThread);
    strides[1] = vecLen;
  }
  loadBufferView = collapseContiguousMerges(loadBufferView, rawLoadBufferShape);

  // Run the packing loop
  auto packLoop =
      b.create<TransformingForOp>(loc, ArrayRef<ValueRange>{start, start},
                                  ArrayRef<Attribute>{loadBufferView, storeIdx},
                                  /*bounds=*/loadShape,
                                  /*strides=*/strides, false,
                                  /*useIndexDiffs=*/false);
  {
    PatternRewriter::InsertionGuard outerGuard(b);
    b.setInsertionPointToStart(packLoop.getBody());
    Type loadType = vectorTypeOrSelf(elementType, vecLen);
    auto val = b.create<InBoundsLoadOp>(loc, loadType, rawLoadBuffer,
                                        packLoop.getLowerCoords(0));

    b.create<InBoundsStoreOp>(loc, val, storeBuffer,
                              packLoop.getLowerCoords(1));
  }
  return packLoop;
}

static TransformingForOp
createLdsStoreLoop(PatternRewriter &b, Location loc, Value storeBuffer,
                   ArrayAttr ldsVectorMap, Value wrappedBuffer,
                   int64_t dataPerThread, Value tid, bool forceUnroll) {
  Value rawBuffer;
  ArrayAttr bufferView;
  std::tie(rawBuffer, bufferView) = untransform(b, wrappedBuffer);

  auto rawBufferType = rawBuffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> bufferShape = rawBufferType.getShape();
  Type ldsBufferElemTy = rawBufferType.getElementType();
  Type dataType = ldsBufferElemTy;
  int64_t ldsWriteLen = 1;
  if (auto ldsBufferVecTy = ldsBufferElemTy.dyn_cast<VectorType>()) {
    ldsWriteLen = ldsBufferVecTy.getNumElements();
    dataType = ldsBufferVecTy.getElementType();
  }

  // If the LDS is already being viewed as vector-typed, there's no good
  // mechanism to, for example, store a vector<8xf32> as two consecutive
  // vector<4xf32>s, and there's unlikely to be any performance benefit from
  // doing so. Therefore, don't bother.
  int64_t ldsMaxAllowedVectorization =
      ldsWriteLen > 1 ? ldsWriteLen : dataPerThread;
  int64_t ldsStoreVectorization = getMaxVectorization(
      bufferView, /*dim=*/1, ldsMaxAllowedVectorization, bufferShape,
      /*implicitStride=*/ldsWriteLen);
  bufferView = collapseContiguousMerges(bufferView, bufferShape);
  Type storeType = vectorTypeOrSelf(dataType, ldsStoreVectorization);

  Value zero = b.createOrFold<ConstantIndexOp>(loc, 0);
  SmallVector<Value, 2> vecCoordInit(2, zero);
  SmallVector<Value, 2> ldsCoordInit = {tid, zero};

  auto loop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{vecCoordInit, ldsCoordInit},
      ArrayRef<Attribute>{ldsVectorMap, bufferView},
      /*bounds=*/ArrayRef<int64_t>{1, dataPerThread},
      /*strides=*/ArrayRef<int64_t>{1, ldsStoreVectorization}, forceUnroll,
      /*useIndexDiffs=*/true);
  {
    PatternRewriter::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value toStore = b.create<InBoundsLoadOp>(loc, storeType, storeBuffer,
                                             loop.getLowerCoords(0));
    if (ldsWriteLen == 1) // kpack = 1, vectorized in D
      b.create<InBoundsStoreOp>(loc, toStore, rawBuffer,
                                loop.getLowerCoords(/*domain=*/1));
    else
      b.create<memref::StoreOp>(loc, toStore, rawBuffer,
                                loop.getLowerCoords(/*domain=*/1));
  }
  return loop;
}

template <typename OpT>
static LogicalResult checkLDSSize(OpT op, int64_t aBufferBytes,
                                  int64_t bBufferBytes) {
  int64_t ldsBytes = aBufferBytes + bBufferBytes;
  return success(ldsBytes <= 64 * 1024);
}

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

namespace {
struct GridwiseGemmRewritePattern : public OpRewritePattern<GridwiseGemmOp> {
  using OpRewritePattern<GridwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GridwiseGemmOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain data type.
    Type elementTypeA = op.getA().getType().getElementType();
    Type elementTypeB = op.getB().getType().getElementType();
    Type destType = op.getC().getType().getElementType();
    Type accumulatorType =
        obtainAccumulatorType(b, elementTypeA, elementTypeB, destType);

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
    uint32_t gridSize = op.getGridSize();
    GeneralGemmParamsAttr tuningParams = op.getParams();
    int64_t kpack = tuningParams.getKpack();
    // TODO: kPerBlock, as defined in parameter selection etc,
    // is in units of kPack, not individual k. This should be changed
    // at some future point, but it'll be worked around for now.
    uint32_t blockSize = tuningParams.getBlockSize();
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
               << "nBlocks = N / nPerBlock: " << nBlocks << "\n"
               << "kPerBlock: " << kPerBlock << "\n"
               << "kpack: " << kpack << "\n"
               << "mPerThread: " << mPerThread << "\n"
               << "nPerThread: " << nPerThread << "\n"
               << "mThreadsPerCuwave: " << mThreadsPerCuwave << "\n"
               << "mCuwavesPerBlock: " << mCuwavesPerBlock << "\n"
               << "nThreadsPerCuwave: " << nThreadsPerCuwave << "\n"
               << "nCuwavesPerBlock: " << nCuwavesPerBlock << "\n");

    // Compute required LDS sizes.
    int64_t ldsBlockASize =
        kpacksPerBlock * mPerBlock * kpack * getByteWidth(elementTypeA);
    int64_t ldsBlockBSize =
        kpacksPerBlock * nPerBlock * kpack * getByteWidth(elementTypeB);
    LLVM_DEBUG(llvm::dbgs() << "LDS block size (in bytes):" << ldsBlockASize
                            << " " << ldsBlockBSize << "\n");
    if (failed(checkLDSSize(op, ldsBlockASize, ldsBlockBSize)))
      return op.emitOpError("requires too much LDS");

    // Allocate LDS.
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefAType =
        MemRefType::get({ldsBlockASize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsBufferA = b.create<GpuAllocOp>(loc, ldsMemRefAType);
    auto ldsMemRefBType =
        MemRefType::get({ldsBlockBSize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsBufferB = b.create<GpuAllocOp>(loc, ldsMemRefBType);

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
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    auto threadCRegisterMemRefType =
        MemRefType::get({threadCNumRegisters}, accumulatorType, AffineMap{},
                        privateMemoryAddressSpace);
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
    if (aCopyPerThread == 0 || bCopyPerThread == 0) {
      return emitError(loc) << "Block size too large, rejecting as invalid.\n";
    }

    auto maybeCopyAPerThread = computeCopyPerThread(
        elementTypeA, aCopyPerThread, kPerBlock, mPerBlock, kpack, loc);
    if (failed(maybeCopyAPerThread))
      return maybeCopyAPerThread;
    int64_t aCopyKPerThread = (*maybeCopyAPerThread).first;
    int64_t copyMPerThread = (*maybeCopyAPerThread).second;

    auto maybeCopyBPerThread = computeCopyPerThread(
        elementTypeB, bCopyPerThread, kPerBlock, nPerBlock, kpack, loc);
    if (failed(maybeCopyBPerThread))
      return maybeCopyBPerThread;
    int64_t bCopyKPerThread = (*maybeCopyBPerThread).first;
    int64_t copyNPerThread = (*maybeCopyBPerThread).second;

    GemmDimension vectorTiebreaker =
        (kpack > 1) ? GemmDimension::K : GemmDimension::MorN;
    int64_t aVectorLen, bVectorLen;
    GemmDimension aVectorDim, bVectorDim;
    std::tie(aVectorDim, aVectorLen) = bestGlobalVectorization(
        b, op.getA(), copyMPerThread, aCopyKPerThread, vectorTiebreaker,
        kPerBlock, mPerBlock, elementTypeA);
    std::tie(bVectorDim, bVectorLen) = bestGlobalVectorization(
        b, op.getB(), copyNPerThread, bCopyKPerThread, vectorTiebreaker,
        kPerBlock, nPerBlock, elementTypeB);

    LLVM_DEBUG(llvm::dbgs()
               << "aCopyPerThread: " << aCopyPerThread << "\n"
               << "bCopyPerThread: " << bCopyPerThread << "\n"
               << "aVectorDim: " << aVectorDim << "\n"
               << "aVectorLen: " << aVectorLen << "\n"
               << "bVectorDim: " << bVectorDim << "\n"
               << "bVectorLen: " << bVectorLen << "\n"
               << "vectorTiebreaker: " << vectorTiebreaker << "\n");

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

    Type loadBufferAType, loadBufferBType;
    loadBufferAType = MemRefType::get({aCopyPerThread}, elementTypeA,
                                      AffineMap{}, privateMemoryAddressSpace);
    loadBufferBType = MemRefType::get({bCopyPerThread}, elementTypeB,
                                      AffineMap{}, privateMemoryAddressSpace);

    auto loadBufferA = b.create<GpuAllocOp>(loc, loadBufferAType);
    auto loadBufferB = b.create<GpuAllocOp>(loc, loadBufferBType);

    TransformingForOp blockwiseLoadA =
        createGlobalLoadLoop(b, loc, loadBufferA, wrappedA, aVectorGlobalMap,
                             aCopyPerThread, aVectorLen, bid, tid, true);
    TransformingForOp blockwiseLoadB =
        createGlobalLoadLoop(b, loc, loadBufferB, wrappedB, bVectorGlobalMap,
                             bCopyPerThread, bVectorLen, bid, tid, true);

    ArrayAttr aVectorLdsMap = ldsVectorLayout(b, loc, aCopyPerThread);
    ArrayAttr bVectorLdsMap = ldsVectorLayout(b, loc, bCopyPerThread);

    Type ldsReadTypeA = vectorTypeOrSelf(elementTypeA, kpack);
    Type ldsReadTypeB = vectorTypeOrSelf(elementTypeB, kpack);
    FailureOr<Value> maybeWrappedLdsA = wrapLDSBufferForStore(
        b, loc, ldsBufferA, ldsReadTypeA, kpacksPerBlock, "m", mPerBlock,
        aCopyKPerThread, copyMPerThread, aVectorDim);
    if (failed(maybeWrappedLdsA))
      return maybeWrappedLdsA;
    FailureOr<Value> maybeWrappedLdsB = wrapLDSBufferForStore(
        b, loc, ldsBufferB, ldsReadTypeB, kpacksPerBlock, "n", nPerBlock,
        bCopyKPerThread, copyNPerThread, bVectorDim);
    if (failed(maybeWrappedLdsB))
      return maybeWrappedLdsB;
    Value wrappedLdsA = std::move(*maybeWrappedLdsA),
          wrappedLdsB = std::move(*maybeWrappedLdsB);

    Value storeBufferA = b.create<GpuAllocOp>(loc, loadBufferA.getType());
    Value storeBufferB = b.create<GpuAllocOp>(loc, loadBufferB.getType());

    Value viewLoadBufferA = viewLoadBufferDK(b, loc, loadBufferA, aVectorDim,
                                             copyMPerThread, aCopyKPerThread);
    auto packALoop = packLoadBufferToStoreBuffer(b, loc, elementTypeA, kpack,
                                                 viewLoadBufferA, storeBufferA);
    Value viewLoadBufferB = viewLoadBufferDK(b, loc, loadBufferB, bVectorDim,
                                             copyNPerThread, bCopyKPerThread);
    auto packBLoop = packLoadBufferToStoreBuffer(b, loc, elementTypeB, kpack,
                                                 viewLoadBufferB, storeBufferB);

    TransformingForOp blockwiseStoreA =
        createLdsStoreLoop(b, loc, storeBufferA, aVectorLdsMap, wrappedLdsA,
                           aCopyPerThread, tid, true);
    TransformingForOp blockwiseStoreB =
        createLdsStoreLoop(b, loc, storeBufferB, bVectorLdsMap, wrappedLdsB,
                           bCopyPerThread, tid, true);

    // The blockwise gemm isn't set up for vector-of-kpack loads and so expects
    // a scalar kpacksPerBlock x dPerBlock x kpack x T buffer unconditionally.
    Value ldsMatrixA = viewBufferAs(b, ldsBufferA, elementTypeA);
    ldsMatrixA = reshapeBuffer(b, loc, ldsMatrixA, {"k", "m", "kpack"},
                               {kpacksPerBlock, mPerBlock, kpack});
    Value ldsMatrixB = viewBufferAs(b, ldsBufferB, elementTypeB);
    ldsMatrixB = reshapeBuffer(b, loc, ldsMatrixB, {"k", "n", "kpack"},
                               {kpacksPerBlock, nPerBlock, kpack});

    // Emit loop.
    int64_t nIterations = K / kPerBlock;
    BlockwiseGemmOp blockwiseGemmOp;
    // Start at 1 to make it clearer we have performed software pipelining.
    auto loopOp = b.create<affine::AffineForOp>(loc, 1, nIterations, 1);
    {
      // inside the loop.
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());

      // We don't update in the clone becasue we might accidentally replace
      // other zeroes.
      Value iv = loopOp.getInductionVar();
      IRMapping loadAUpdates, loadBUpdates;
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

      // LDS barrier.
      b.create<LDSBarrierOp>(loc);

      // Emit blockwise GEMM.
      blockwiseGemmOp =
          b.create<BlockwiseGemmOp>(loc, ldsMatrixA, ldsMatrixB,
                                    registerMatrixCViewOp, op.getParamsAttr());

      // LDS barrier.
      // This barrier prevents halo part of outputs having weird values.
      b.create<LDSBarrierOp>(loc);

      // Packing step
      b.clone(*packALoop.getOperation());
      b.clone(*packBLoop.getOperation());

      // Emit blockwise stores
      b.clone(*blockwiseStoreA.getOperation());
      b.clone(*blockwiseStoreB.getOperation());
    }
    // outside the loop.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    IRMapping tailGemmCloneMap;
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
          /*bounds=*/convertedCType.getShape(), /*strides=*/std::nullopt,
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

    b.create<ThreadwiseWriteAllOp>(loc, registerC, op.getC(), idToMatrixCMaps,
                                   /*extraIndices=*/ValueRange{bid, tid},
                                   op.getFeatures(), StoreMethod::Set,
                                   /*forceUnroll=*/true, useIndexDiffs);
    b.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// GridwiseGemmAccel lowering.
//===----------------------------------------------------------------------===//

struct GridwiseGemmAccelRewritePattern
    : public OpRewritePattern<GridwiseGemmAccelOp> {
  using OpRewritePattern<GridwiseGemmAccelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GridwiseGemmAccelOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain data types of inputs.
    auto elementTypeA = op.getA().getType().getElementType();
    auto elementTypeB = op.getB().getType().getElementType();

    // Prepare some useful constants.
    Value matA = op.getA();
    Value matB = op.getB();

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

    // Obtain critical tuning parameters.
    StringRef arch = op.getArch();
    uint32_t blockSize = op.getBlockSize();
    uint32_t gridSize = op.getGridSize();
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    int64_t kpack = tuningParams.getKpack();
    // TODO: kPerBlock, as defined in parameter selection etc,
    // is in units of kPack, not individual k. This should be changed
    // at some future point, but it'll be worked around for now.
    int64_t kpacksPerBlock = tuningParams.getKpackPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();
    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;
    bool forceUnroll = tuningParams.getForceUnroll();

    int64_t kPerBlock = kpacksPerBlock * kpack;

    int64_t aVectorLen = 0;
    int64_t bVectorLen = 0;
    GemmDimension aVectorDim;
    GemmDimension bVectorDim;

    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};

    int64_t aCopyPerThread = (kPerBlock * mPerBlock) / blockSize;
    int64_t bCopyPerThread = (kPerBlock * nPerBlock) / blockSize;
    if (aCopyPerThread == 0 || bCopyPerThread == 0) {
      return emitError(loc) << "Block size too large, rejecting as invalid.\n";
    }
    int64_t aCopyKpacksPerThread =
        math_util::integer_divide_ceil(aCopyPerThread, kpack);
    int64_t bCopyKpacksPerThread =
        math_util::integer_divide_ceil(bCopyPerThread, kpack);

    // Get the vector copy layout for A and B
    auto maybeCopyAPerThread = computeCopyPerThread(
        elementTypeA, aCopyPerThread, kPerBlock, mPerBlock, kpack, loc);
    if (failed(maybeCopyAPerThread))
      return maybeCopyAPerThread;
    int64_t aCopyKPerThread = (*maybeCopyAPerThread).first;
    int64_t copyMPerThread = (*maybeCopyAPerThread).second;

    auto maybeCopyBPerThread = computeCopyPerThread(
        elementTypeB, bCopyPerThread, kPerBlock, nPerBlock, kpack, loc);
    if (failed(maybeCopyBPerThread))
      return maybeCopyBPerThread;
    int64_t bCopyKPerThread = (*maybeCopyBPerThread).first;
    int64_t copyNPerThread = (*maybeCopyBPerThread).second;

    // Find the best way of vectorizing the layout
    GemmDimension vectorTiebreaker =
        (kpack > 1) ? GemmDimension::K : GemmDimension::MorN;
    std::tie(aVectorDim, aVectorLen) = bestGlobalVectorization(
        b, matA, copyMPerThread, aCopyKPerThread, vectorTiebreaker, kPerBlock,
        mPerBlock, elementTypeA);
    std::tie(bVectorDim, bVectorLen) = bestGlobalVectorization(
        b, matB, copyNPerThread, bCopyKPerThread, vectorTiebreaker, kPerBlock,
        nPerBlock, elementTypeB);

    LLVM_DEBUG(llvm::dbgs()
               << "gridSize: " << gridSize << "\n"
               << "blockSize: " << blockSize << "\n"
               << "aCopyPerThread: " << aCopyPerThread << "\n"
               << "bCopyPerThread: " << bCopyPerThread << "\n"
               << "aCopyKpacksPerThread: " << aCopyKpacksPerThread << "\n"
               << "bCopyKpacksPerThread: " << bCopyKpacksPerThread << "\n"
               << "aVectorDim: " << aVectorDim << "\n"
               << "aVectorLen: " << aVectorLen << "\n"
               << "bVectorDim: " << bVectorDim << "\n"
               << "bVectorLen: " << bVectorLen << "\n"
               << "vectorTiebreaker: " << vectorTiebreaker << "\n"
               << "kPerBlock: " << kPerBlock << "\n"
               << "mPerBlock: " << mPerBlock << "\n"
               << "nPerBlock: " << nPerBlock << "\n"
               << "aCopyKPerThread: " << aCopyKPerThread << "\n"
               << "bCopyKPerThread: " << bCopyKPerThread << "\n"
               << "copyMPerThread: " << copyMPerThread << "\n"
               << "copyNPerThread: " << copyNPerThread << "\n");

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

    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());
    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    Type loadBufferAType, loadBufferBType;
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    loadBufferAType = MemRefType::get({aCopyPerThread}, elementTypeA,
                                      AffineMap{}, privateMemoryAddressSpace);
    loadBufferBType = MemRefType::get({bCopyPerThread}, elementTypeB,
                                      AffineMap{}, privateMemoryAddressSpace);

    auto loadBufferA = b.create<GpuAllocOp>(loc, loadBufferAType);
    auto loadBufferB = b.create<GpuAllocOp>(loc, loadBufferBType);

    TransformingForOp blockwiseLoadA =
        createGlobalLoadLoop(b, loc, loadBufferA, wrappedA, aVectorGlobalMap,
                             aCopyPerThread, aVectorLen, bid, tid, forceUnroll);
    TransformingForOp blockwiseLoadB =
        createGlobalLoadLoop(b, loc, loadBufferB, wrappedB, bVectorGlobalMap,
                             bCopyPerThread, bVectorLen, bid, tid, forceUnroll);

    Value storeBufferA = b.create<GpuAllocOp>(loc, loadBufferA.getType());
    Value storeBufferB = b.create<GpuAllocOp>(loc, loadBufferB.getType());

    Value viewLoadBufferA = viewLoadBufferDK(b, loc, loadBufferA, aVectorDim,
                                             copyMPerThread, aCopyKPerThread);
    auto packALoop = packLoadBufferToStoreBuffer(b, loc, elementTypeA, kpack,
                                                 viewLoadBufferA, storeBufferA);

    Value viewLoadBufferB = viewLoadBufferDK(b, loc, loadBufferB, bVectorDim,
                                             copyNPerThread, bCopyKPerThread);
    auto packBLoop = packLoadBufferToStoreBuffer(b, loc, elementTypeB, kpack,
                                                 viewLoadBufferB, storeBufferB);

    // Obtain Accelerator-related attributes.
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();
    int64_t nWaves = nPerBlock / nPerWave;

    auto nWavesConstantOp = b.create<ConstantIndexOp>(loc, nWaves);

    auto accelEmitterPtr = accel::AccelEmitter::select(
        op.getFeatures(), elementTypeA, elementTypeB, arch, tuningParams);

    if (!accelEmitterPtr)
      return op.emitOpError("Unable to emit accelerator code.");

    // Extract relevant accelerator parameters
    rock::accel::AccelEmitterParams params = accelEmitterPtr->getParams();
    int64_t nResultVectors = params.nResultVectors;
    int64_t mRepeats = params.mRepeats;
    int64_t nRepeats = params.nRepeats;
    int64_t kBasePerThread = params.kBasePerThread;
    Type argTypeA = params.argTypeA;
    Type argTypeB = params.argTypeB;
    VectorType accVectorType = params.accVectorType;
    int64_t numOutputVectorElements = params.numOutputVectorElements();

    const int64_t waveSize = rock::lookupArchInfo(arch).waveSize;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, waveSize);

    bool useIndexDiffs = true;

    LLVM_DEBUG(llvm::dbgs() << "M: " << M << "\n"
                            << "N: " << N << "\n"
                            << "K: " << K << "\n"
                            << "G: " << G << "\n"
                            << "mPerBlock: " << mPerBlock << "\n"
                            << "nPerBlock: " << nPerBlock << "\n"
                            << "kPerBlock: " << kPerBlock << "\n"
                            << "kpack: " << kpack << "\n"
                            << "mBlocks = M / mPerBlock: " << mBlocks << "\n"
                            << "nBlocks = N / nPerBlock: " << nBlocks << "\n"
                            << "mPerWave: " << mPerWave << "\n"
                            << "nPerWave: " << nPerWave << "\n"
                            << "aVectorLen: " << aVectorLen << "\n"
                            << "bVectorLen: " << bVectorLen << "\n"
                            << "aVectorDim: " << aVectorDim << "\n"
                            << "bVectorDim: " << bVectorDim << "\n");

    // Alocate LDS and create subviews.

    // Compute required LDS sizes.
    int64_t ldsBlockASize =
        kpacksPerBlock * mPerBlock * kpack * getByteWidth(elementTypeA);
    int64_t ldsBlockBSize =
        kpacksPerBlock * nPerBlock * kpack * getByteWidth(elementTypeB);
    LLVM_DEBUG(llvm::dbgs() << "LDS block sizes (bytes): " << ldsBlockASize
                            << " " << ldsBlockBSize << "\n");
    if (failed(checkLDSSize(op, ldsBlockASize, ldsBlockBSize)))
      return op.emitOpError("requires too much LDS");

    // Allocate LDS.
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefAType =
        MemRefType::get({ldsBlockASize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsBufferA = b.create<GpuAllocOp>(loc, ldsMemRefAType);
    auto ldsMemRefBType =
        MemRefType::get({ldsBlockBSize}, b.getI8Type(), AffineMap{},
                        workgroupMemoryAddressSpace);
    auto ldsBufferB = b.create<GpuAllocOp>(loc, ldsMemRefBType);

    ArrayAttr aVectorLdsMap = ldsVectorLayout(b, loc, aCopyPerThread);
    ArrayAttr bVectorLdsMap = ldsVectorLayout(b, loc, bCopyPerThread);

    Type ldsReadTypeA = vectorTypeOrSelf(elementTypeA, kpack);
    Type ldsReadTypeB = vectorTypeOrSelf(elementTypeB, kpack);
    FailureOr<Value> maybeWrappedLdsA = wrapLDSBufferForStore(
        b, loc, ldsBufferA, ldsReadTypeA, kpacksPerBlock, "m", mPerBlock,
        aCopyKPerThread, copyMPerThread, aVectorDim);
    if (failed(maybeWrappedLdsA))
      return maybeWrappedLdsA;
    FailureOr<Value> maybeWrappedLdsB = wrapLDSBufferForStore(
        b, loc, ldsBufferB, ldsReadTypeB, kpacksPerBlock, "n", nPerBlock,
        bCopyKPerThread, copyNPerThread, bVectorDim);
    if (failed(maybeWrappedLdsB))
      return maybeWrappedLdsB;
    Value wrappedLdsA = std::move(*maybeWrappedLdsA),
          wrappedLdsB = std::move(*maybeWrappedLdsB);

    TransformingForOp blockwiseStoreA =
        createLdsStoreLoop(b, loc, storeBufferA, aVectorLdsMap, wrappedLdsA,
                           aCopyPerThread, tid, forceUnroll);
    TransformingForOp blockwiseStoreB =
        createLdsStoreLoop(b, loc, storeBufferB, bVectorLdsMap, wrappedLdsB,
                           bCopyPerThread, tid, forceUnroll);

    Value ldsViewForGemmA = viewBufferAs(b, ldsBufferA, ldsReadTypeA);
    Value ldsViewForGemmB = viewBufferAs(b, ldsBufferB, ldsReadTypeB);
    // -----

    Type destType = op.getC().getType().getElementType();

    int64_t nOutputVectors = nResultVectors * mRepeats * nRepeats;

    // -----

    // Logic to setup blockwise_gemm_accel parameters.
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
    auto waveId_m = b.create<DivUIOp>(loc, waveId, nWavesConstantOp);
    auto waveId_n = b.create<RemUIOp>(loc, waveId, nWavesConstantOp);

    Value mMyWaveOffsetA, mMyWaveOffsetB;
    Value waveOffsetAConstantOp =
        b.create<ConstantIndexOp>(loc, params.mPerAccel);
    Value waveOffsetBConstantOp =
        b.create<ConstantIndexOp>(loc, params.nPerAccel);
    mMyWaveOffsetA = b.create<MulIOp>(loc, waveId_m, waveOffsetAConstantOp);
    mMyWaveOffsetB = b.create<MulIOp>(loc, waveId_n, waveOffsetBConstantOp);

    // Logic to setup buffers for blockwise_gemm_accel.

    Type arrayAType, arrayBType;
    arrayAType = MemRefType::get({kBasePerThread}, argTypeA, AffineMap{},
                                 privateMemoryAddressSpace);
    arrayBType = MemRefType::get({kBasePerThread}, argTypeB, AffineMap{},
                                 privateMemoryAddressSpace);
    auto arrayA = b.create<GpuAllocOp>(loc, arrayAType);
    auto arrayB = b.create<GpuAllocOp>(loc, arrayBType);

    // -----
    // Logic to allocate 0-initialized vectors for C.
    MemRefType regCAllocType =
        MemRefType::get(nOutputVectors, accVectorType, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
    Value regCAllocOp = b.create<rock::GpuAllocOp>(loc, regCAllocType);

    Value zeroConstantCOp = createZeroConstantOp(b, loc, accVectorType);
    b.create<FillOp>(loc, regCAllocOp, zeroConstantCOp);

    // Emit loop.
    int64_t nIterations = K / kPerBlock;
    BlockwiseGemmAccelOp blockwiseGemmAccelOp;
    // Start at 1 to make it clearer we have performed software pipelining.
    auto loopOp = b.create<affine::AffineForOp>(loc, 1, nIterations, 1);
    {
      // inside the loop.
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());

      // We don't update in the clone becasue we might accidentally replace
      // other zeroes.
      Value iv = loopOp.getInductionVar();
      IRMapping loadAUpdates, loadBUpdates;
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

      // LDS barrier.
      b.create<LDSBarrierOp>(loc);

      // Emit blockwise GEMM.
      blockwiseGemmAccelOp = b.create<BlockwiseGemmAccelOp>(
          loc, ldsViewForGemmA, ldsViewForGemmB, mMyWaveOffsetA, mMyWaveOffsetB,
          arrayA, arrayB, regCAllocOp, op.getArchAttr(), op.getFeaturesAttr(),
          op.getBlockSizeAttr(), op.getParamsAttr());

      // LDS barrier.
      // This barrier prevents halo part of outputs having weird values.
      b.create<LDSBarrierOp>(loc);

      // Packing step
      b.clone(*packALoop.getOperation());
      b.clone(*packBLoop.getOperation());

      // Emit blockwise stores
      b.clone(*blockwiseStoreA.getOperation());
      b.clone(*blockwiseStoreB.getOperation());
    }
    // outside the loop.

    // Emit loop tail.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    IRMapping tailGemmCloneMap;
    b.clone(*blockwiseGemmAccelOp, tailGemmCloneMap);

    // Apparently, the canonicalizer doesn't get rid of empty loops without
    // results properly, remove them ourselves.
    if (nIterations <= 1) {
      b.eraseOp(loopOp);
    }

    // -----

    // Matrix C write out logic.
    auto convertedCType =
        MemRefType::get(numOutputVectorElements, destType, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
    Value convertedC = b.create<rock::GpuAllocOp>(loc, convertedCType);

    ArrayAttr idToMatrixCMaps = accelEmitterPtr->computeOutputTransforms(
        b, loc, M, N, blockSize, bidGridLengths);

    Value registerC = accelEmitterPtr->computeOutputConversion(
        b, loc, M, N, blockSize, gridSize, regCAllocOp, convertedC,
        forceUnroll);

    b.create<ThreadwiseWriteAllOp>(loc, registerC, op.getC(), idToMatrixCMaps,
                                   /*extraIndices=*/ValueRange{bid, tid},
                                   op.getFeatures(), op.getStoreMethod(),
                                   forceUnroll, useIndexDiffs);

    b.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void RockGridwiseGemmToBlockwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::GridwiseGemmOp, rock::GridwiseGemmAccelOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         memref::MemRefDialect, affine::AffineDialect,
                         vector::VectorDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmRewritePattern, GridwiseGemmAccelRewritePattern>(
      ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
