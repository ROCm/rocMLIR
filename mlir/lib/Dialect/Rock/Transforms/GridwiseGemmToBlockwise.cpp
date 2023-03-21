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
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
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

/// Wraps the LDS buffer "buffer", which is K x D x kpack, into a
/// tid x iter view.
static FailureOr<Value> wrapLDSBufferForStore(OpBuilder &b, Location loc,
                                              Value buffer, StringRef dName,
                                              int64_t kPerThread,
                                              int64_t dPerThread,
                                              GemmDimension vectorDim) {
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

static TransformingForOp createGlobalLoadLoop(PatternRewriter &b, Location loc,
                                              Value wrappedMatrix,
                                              ArrayAttr vectorMap,
                                              int64_t dataPerThread,
                                              int64_t vectorLen, Value bid,
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
  Type resultType = vectorTypeOrSelf(elementType, dataPerThread);

  Value resultInit = createZeroConstantOp(b, loc, resultType);
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);

  SmallVector<Value, 4> globalStart = {zero, bid, tid, zero};
  SmallVector<Value, 4> vectorStartOuter(4, zero);
  auto outerLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{globalStart, vectorStartOuter},
      ArrayRef<Attribute>{matrixToTensor, b.getArrayAttr({})},
      /*bounds=*/ArrayRef<int64_t>{1, 1, 1, dataPerThread},
      /*strides=*/ArrayRef<int64_t>{1, 1, 1, vectorLen}, forceUnroll,
      /*useIndexDiffs=*/true, resultInit);
  {
    PatternRewriter::InsertionGuard outerGuard(b);
    b.setInsertionPointToEnd(outerLoop.getBody());
    Value loaded = b.create<GlobalLoadOp>(
        loc, loadType, tensor, outerLoop.getValidity(/*domain=*/0),
        outerLoop.getLowerCoords(/*domain=*/0));
    auto innerLoop = b.create<TransformingForOp>(
        loc,
        ArrayRef<ValueRange>{zero,
                             outerLoop.getLowerCoords(/*domain=*/1).back()},
        ArrayRef<Attribute>{b.getArrayAttr({}), vectorMap},
        /*bounds=*/ArrayRef<int64_t>{vectorLen},
        /*strides=*/ArrayRef<int64_t>{1}, forceUnroll, /*useIndexDiffs=*/true,
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

static TransformingForOp
createLdsStoreLoop(PatternRewriter &b, Location loc, Value loaded,
                   ArrayAttr ldsVectorMap, Value wrappedBuffer,
                   int64_t dataPerThread, Value tid, bool forceUnroll) {
  Value rawBuffer;
  ArrayAttr bufferView;
  std::tie(rawBuffer, bufferView) = untransform(b, wrappedBuffer);

  ArrayRef<int64_t> bufferShape =
      rawBuffer.getType().cast<MemRefType>().getShape();
  int64_t ldsStoreVectorization =
      getMaxVectorization(bufferView, /*dim=*/1, dataPerThread, bufferShape);
  bufferView = collapseContiguousMerges(bufferView, bufferShape);
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
      /*strides=*/ArrayRef<int64_t>{1, ldsStoreVectorization}, forceUnroll,
      /*useIndexDiffs=*/true);
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
                                     int64_t kpack = 1) const {
    GeneralGemmParamsAttr tuningParams = op.getParams();
    int64_t ThreadGemmAThreadCopySrcDataPerRead_M =
        tuningParams.getMPerThread();
    int64_t ThreadGemmBThreadCopySrcDataPerRead_N =
        tuningParams.getNPerThread();

    int64_t max_lds_align =
        math_util::lcm(ThreadGemmAThreadCopySrcDataPerRead_M,
                       ThreadGemmBThreadCopySrcDataPerRead_N);

    int64_t kPerBlock = tuningParams.getKPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();

    int64_t alignedNperBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(nPerBlock, max_lds_align);

    // A matrix in LDS memory, dst of blockwise copy
    //   be careful of LDS alignment
    // Original C++ logic:
    // constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
    //    Sequence<kPerBlock, mPerBlock>{}, Number<max_lds_align>{});
    // constexpr index_t a_block_space =
    //    math_util::integer_least_multiple(a_k_m_block_desc.GetElementSpace(),
    //    max_lds_align);
    int64_t alignedMperBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(mPerBlock, max_lds_align);
    a_block_space = math_util::integer_least_multiple(
                        kPerBlock * alignedMperBlock, max_lds_align) *
                    kpack;

    // B matrix in LDS memory, dst of blockwise copy
    //   be careful of LDS alignment
    // Original C++ logic:
    // constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
    //    Sequence<kPerBlock, nPerBlock>{}, Number<max_lds_align>{});
    // constexpr index_t b_block_space =
    //    math_util::integer_least_multiple(b_k_n_block_desc.GetElementSpace(),
    //    max_lds_align);
    b_block_space = math_util::integer_least_multiple(
                        kPerBlock * alignedNperBlock, max_lds_align) *
                    kpack;

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
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    LogicalResult res = computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize,
                                             ldsBlockSize, kpack);
    LLVM_DEBUG(llvm::dbgs() << "LDS block size:" << ldsBlockASize << " "
                            << ldsBlockBSize << " " << ldsBlockSize << "\n");
    if (res.failed())
      return failure();

    // Allocate LDS.
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefType = MemRefType::get(
        {ldsBlockSize}, elementType, AffineMap{}, workgroupMemoryAddressSpace);
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
        elementType, aCopyPerThread, kPerBlock, mPerBlock, kpack, loc);
    if (failed(maybeCopyAPerThread))
      return maybeCopyAPerThread;
    int64_t aCopyKPerThread = (*maybeCopyAPerThread).first;
    int64_t copyMPerThread = (*maybeCopyAPerThread).second;

    auto maybeCopyBPerThread = computeCopyPerThread(
        elementType, bCopyPerThread, kPerBlock, nPerBlock, kpack, loc);
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
        kPerBlock, mPerBlock, elementType);
    std::tie(bVectorDim, bVectorLen) = bestGlobalVectorization(
        b, op.getB(), copyNPerThread, bCopyKPerThread, vectorTiebreaker,
        kPerBlock, nPerBlock, elementType);

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

    TransformingForOp blockwiseLoadA =
        createGlobalLoadLoop(b, loc, wrappedA, aVectorGlobalMap, aCopyPerThread,
                             aVectorLen, bid, tid, true);
    TransformingForOp blockwiseLoadB =
        createGlobalLoadLoop(b, loc, wrappedB, bVectorGlobalMap, bCopyPerThread,
                             bVectorLen, bid, tid, true);

    ArrayAttr aVectorLdsMap = ldsVectorLayout(b, loc, aCopyPerThread);
    ArrayAttr bVectorLdsMap = ldsVectorLayout(b, loc, bCopyPerThread);

    FailureOr<Value> maybeWrappedLdsA =
        wrapLDSBufferForStore(b, loc, ldsMatrixASubviewOp, "m", aCopyKPerThread,
                              copyMPerThread, aVectorDim);
    if (failed(maybeWrappedLdsA))
      return maybeWrappedLdsA;
    FailureOr<Value> maybeWrappedLdsB =
        wrapLDSBufferForStore(b, loc, ldsMatrixBSubviewOp, "n", bCopyKPerThread,
                              copyNPerThread, bVectorDim);
    if (failed(maybeWrappedLdsB))
      return maybeWrappedLdsB;
    Value wrappedLdsA = std::move(*maybeWrappedLdsA),
          wrappedLdsB = std::move(*maybeWrappedLdsB);

    TransformingForOp blockwiseStoreA =
        createLdsStoreLoop(b, loc, blockwiseLoadA.getResult(0), aVectorLdsMap,
                           wrappedLdsA, aCopyPerThread, tid, true);
    TransformingForOp blockwiseStoreB =
        createLdsStoreLoop(b, loc, blockwiseLoadB.getResult(0), bVectorLdsMap,
                           wrappedLdsB, bCopyPerThread, tid, true);

    // Emit loop.
    int64_t nIterations = K / kPerBlock;
    BlockwiseGemmOp blockwiseGemmOp;
    // Start at 1 to make it clearer we have performed software pipelining.
    auto loopOp = b.create<AffineForOp>(loc, 1, nIterations, 1);
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
      blockwiseGemmOp = b.create<BlockwiseGemmOp>(
          loc, ldsMatrixASubviewOp, ldsMatrixBSubviewOp, registerMatrixCViewOp,
          op.getParamsAttr());

      // LDS barrier.
      // This barrier prevents halo part of outputs having weird values.
      b.create<LDSBarrierOp>(loc);

      // Emit blockwise stores
      IRMapping storeAUpdates, storeBUpdates;
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
                                   op.getFeatures(), StoreMethod::Set,
                                   /*forceUnroll=*/true, useIndexDiffs);
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
                                     int64_t kpack = 1) const {
    int64_t max_lds_align = 1;

    XdlopsGemmParamsAttr tuningParams = op.getParams();
    int64_t kPerBlock = tuningParams.getKPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();

    int64_t alignedNperBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(nPerBlock, max_lds_align);

    // A matrix in LDS memory, dst of blockwise copy
    int64_t alignedMperBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(mPerBlock, max_lds_align);

    LLVM_DEBUG(llvm::dbgs() << "mPerBlock : " << mPerBlock << "\n");
    LLVM_DEBUG(llvm::dbgs() << "nPerBlock : " << nPerBlock << "\n");
    LLVM_DEBUG(llvm::dbgs() << "max_lds_align : " << max_lds_align << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "alignedMperBlock : " << alignedMperBlock << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "alignedNperBlock : " << alignedNperBlock << "\n");

    a_block_space = math_util::integer_least_multiple(
                        kPerBlock * alignedMperBlock, max_lds_align) *
                    kpack;

    // B matrix in LDS memory, dst of blockwise copy
    b_block_space = math_util::integer_least_multiple(
                        kPerBlock * alignedNperBlock, max_lds_align) *
                    kpack;

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
    XdlopsGemmParamsAttr tuningParams = op.getParams();
    int64_t kpack = tuningParams.getKpack();
    // TODO: kPerBlock, as defined in parameter selection etc,
    // is in units of kPack, not individual k. This should be changed
    // at some future point, but it'll be worked around for now.
    int64_t kpacksPerBlock = tuningParams.getKPerBlock();
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

    // Get the vector copy layout for A and B
    auto maybeCopyAPerThread = computeCopyPerThread(
        elementType, aCopyPerThread, kPerBlock, mPerBlock, kpack, loc);
    if (failed(maybeCopyAPerThread))
      return maybeCopyAPerThread;
    int64_t aCopyKPerThread = (*maybeCopyAPerThread).first;
    int64_t copyMPerThread = (*maybeCopyAPerThread).second;

    auto maybeCopyBPerThread = computeCopyPerThread(
        elementType, bCopyPerThread, kPerBlock, nPerBlock, kpack, loc);
    if (failed(maybeCopyBPerThread))
      return maybeCopyBPerThread;
    int64_t bCopyKPerThread = (*maybeCopyBPerThread).first;
    int64_t copyNPerThread = (*maybeCopyBPerThread).second;

    // Find the best way of vectorizing the layout
    GemmDimension vectorTiebreaker =
        (kpack > 1) ? GemmDimension::K : GemmDimension::MorN;
    std::tie(aVectorDim, aVectorLen) = bestGlobalVectorization(
        b, matA, copyMPerThread, aCopyKPerThread, vectorTiebreaker, kPerBlock,
        mPerBlock, elementType);
    std::tie(bVectorDim, bVectorLen) = bestGlobalVectorization(
        b, matB, copyNPerThread, bCopyKPerThread, vectorTiebreaker, kPerBlock,
        nPerBlock, elementType);

    LLVM_DEBUG(llvm::dbgs() << "gridSize: " << gridSize << "\n"
                            << "blockSize: " << blockSize << "\n"
                            << "aCopyPerThread: " << aCopyPerThread << "\n"
                            << "bCopyPerThread: " << bCopyPerThread << "\n"
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

    TransformingForOp blockwiseLoadA =
        createGlobalLoadLoop(b, loc, wrappedA, aVectorGlobalMap, aCopyPerThread,
                             aVectorLen, bid, tid, forceUnroll);
    TransformingForOp blockwiseLoadB =
        createGlobalLoadLoop(b, loc, wrappedB, bVectorGlobalMap, bCopyPerThread,
                             bVectorLen, bid, tid, forceUnroll);

    // Obtain XDLOPS-related attributes.
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();
    // int64_t MWaves = mPerBlock / mPerWave;
    int64_t nWaves = nPerBlock / nPerWave;

    auto mPerWaveConstantOp = b.create<ConstantIndexOp>(loc, mPerWave);
    auto nPerWaveConstantOp = b.create<ConstantIndexOp>(loc, nPerWave);
    auto nWavesConstantOp = b.create<ConstantIndexOp>(loc, nWaves);

    constexpr int64_t waveSize = 64;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, waveSize);

    bool useIndexDiffs = true;

    int64_t gStride = mBlocks * nBlocks;

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
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    LogicalResult res = computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize,
                                             ldsBlockSize, kpack);
    LLVM_DEBUG(llvm::dbgs() << "LDS block size:" << ldsBlockASize << " "
                            << ldsBlockBSize << " " << ldsBlockSize << "\n");
    if (res.failed())
      return failure();

    // Allocate LDS.
    auto workgroupMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsMemRefType = MemRefType::get(
        {ldsBlockSize}, elementType, AffineMap{}, workgroupMemoryAddressSpace);
    auto ldsGpuAllocOp = b.create<GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for Matrix A.
    int64_t ldsBlockAOffset = 0;

    Value ldsBlockASubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockAOffset, ldsBlockASize);

    // Get matrix subviews.
    // Compute matrix A dimension from attributes.
    Value ldsMatrixASubviewOp =
        reshapeBuffer(b, loc, ldsBlockASubviewOp, {"k", "m", "kpack"},
                      {kpacksPerBlock, mPerBlock, kpack});

    // Subviews for Matrix B.
    int64_t ldsBlockBOffset = ldsBlockASize;
    Value ldsBlockBSubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockBOffset, ldsBlockBSize);

    // Get matrix subviews.
    // Compute matrix B dimension from attributes.
    Value ldsMatrixBSubviewOp =
        reshapeBuffer(b, loc, ldsBlockBSubviewOp, {"k", "n", "kpack"},
                      {kpacksPerBlock, nPerBlock, kpack});

    ArrayAttr aVectorLdsMap = ldsVectorLayout(b, loc, aCopyPerThread);
    ArrayAttr bVectorLdsMap = ldsVectorLayout(b, loc, bCopyPerThread);

    FailureOr<Value> maybeWrappedLdsA =
        wrapLDSBufferForStore(b, loc, ldsMatrixASubviewOp, "m", aCopyKPerThread,
                              copyMPerThread, aVectorDim);
    if (failed(maybeWrappedLdsA))
      return maybeWrappedLdsA;
    FailureOr<Value> maybeWrappedLdsB =
        wrapLDSBufferForStore(b, loc, ldsMatrixBSubviewOp, "n", bCopyKPerThread,
                              copyNPerThread, bVectorDim);
    if (failed(maybeWrappedLdsB))
      return maybeWrappedLdsB;
    Value wrappedLdsA = std::move(*maybeWrappedLdsA),
          wrappedLdsB = std::move(*maybeWrappedLdsB);

    TransformingForOp blockwiseStoreA =
        createLdsStoreLoop(b, loc, blockwiseLoadA.getResult(0), aVectorLdsMap,
                           wrappedLdsA, aCopyPerThread, tid, forceUnroll);
    TransformingForOp blockwiseStoreB =
        createLdsStoreLoop(b, loc, blockwiseLoadB.getResult(0), bVectorLdsMap,
                           wrappedLdsB, bCopyPerThread, tid, forceUnroll);
    // -----

    // Mfma instruction group selection.
    auto maybeMfmaInsnGroup =
        MfmaInsnGroup::select(elementType, arch, mPerWave, nPerWave);
    if (failed(maybeMfmaInsnGroup)) {
      return emitError(loc) << "Failed to select xdlops instruction group.\n";
    }
    MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
    if (!mfmaGroup.isCoherentWithK(kpack, kPerBlock)) {
      return emitError(loc)
             << "Mfma instruction group selection is not compatible with k.\n";
    }

    int64_t mRepeats = mfmaGroup.getMRepeats(mPerWave);
    int64_t nRepeats = mfmaGroup.getNRepeats(nPerWave);
    auto imms = mfmaGroup.getImms();

    int64_t nResultVectors = imms.size() * mRepeats * nRepeats;
    int64_t mPerRepeat = mPerWave / mRepeats;
    int64_t nPerRepeat = nPerWave / nRepeats;

    VectorType vectorType = mfmaGroup.getRetType();
    MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();

    int64_t m = mfmaAttr.mfmaNonKDim;
    // Note n has the 4x4 => 4x64 behavior that necessitated inputSpansPerMfmaIn
    int64_t n = mfmaAttr.inputSpanLen;

    int64_t rowGroupSize = mfmaAttr.rowGroupSize;
    int64_t rowGroupsPerBlock = mfmaAttr.rowGroupsPerBlock;
    int64_t inputSpanLen = mfmaAttr.inputSpanLen;
    int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
    int64_t k_base = mfmaAttr.k_base;
    int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;

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
    auto waveId_m = b.create<DivUIOp>(loc, waveId, nWavesConstantOp);
    auto waveId_n = b.create<RemUIOp>(loc, waveId, nWavesConstantOp);

    Value mMyWaveOffsetA, mMyWaveOffsetB;
    mMyWaveOffsetA = b.create<MulIOp>(loc, waveId_m, mPerWaveConstantOp);
    mMyWaveOffsetB = b.create<MulIOp>(loc, waveId_n, nPerWaveConstantOp);

    // Logic to setup buffers for blockwise_gemm_v2.

    bool isKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);
    int64_t inputBufferSize =
        (kpacksPerBlock * kpack) /
        ((isKReduction ? inputSpansPerMfmaIn : 1) * k_base);

    Type arrayAType, arrayBType;
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    arrayAType = MemRefType::get({inputBufferSize}, mfmaGroup.getArgType(),
                                 AffineMap{}, privateMemoryAddressSpace);
    arrayBType = MemRefType::get({inputBufferSize}, mfmaGroup.getArgType(),
                                 AffineMap{}, privateMemoryAddressSpace);
    auto arrayA = b.create<GpuAllocOp>(loc, arrayAType);
    auto arrayB = b.create<GpuAllocOp>(loc, arrayBType);

    // -----
    // Logic to allocate 0-initialized vectors for C.
    int64_t regCVectorLen = vectorType.getNumElements();
    Type destType = op.getC().getType().getElementType();
    Type accumulatorType = obtainAccumulatorType(b, elementType, destType);
    VectorType accumulatorVectorType =
        vectorType.cloneWith({}, accumulatorType);
    MemRefType regCAllocType =
        MemRefType::get(nResultVectors, accumulatorVectorType, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
    Value regCAllocOp = b.create<rock::GpuAllocOp>(loc, regCAllocType);

    Value zeroConstantCOp = createZeroConstantOp(b, loc, vectorType);
    b.create<FillOp>(loc, regCAllocOp, zeroConstantCOp);

    // Emit loop.
    int64_t nIterations = K / kPerBlock;
    BlockwiseGemmV2Op blockwiseGemmV2Op;
    // Start at 1 to make it clearer we have performed software pipelining.
    auto loopOp = b.create<AffineForOp>(loc, 1, nIterations, 1);
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
      blockwiseGemmV2Op = b.create<BlockwiseGemmV2Op>(
          loc, ldsGpuAllocOp, ldsGpuAllocOp, b.getIndexAttr(ldsBlockAOffset),
          b.getIndexAttr(ldsBlockBOffset), mMyWaveOffsetA, mMyWaveOffsetB,
          arrayA, arrayB, regCAllocOp, op.getArchAttr(), op.getBlockSizeAttr(),
          op.getParamsAttr());

      // LDS barrier.
      // This barrier prevents halo part of outputs having weird values.
      b.create<LDSBarrierOp>(loc);

      // Emit blockwise stores
      IRMapping storeAUpdates, storeBUpdates;
      storeAUpdates.map(blockwiseLoadA.getResult(0),
                        blockwiseLoadAClone.getResult(0));
      storeBUpdates.map(blockwiseLoadB.getResult(0),
                        blockwiseLoadBClone.getResult(0));
      b.clone(*blockwiseStoreA.getOperation(), storeAUpdates);
      b.clone(*blockwiseStoreB.getOperation(), storeBUpdates);
    }
    // outside the loop.

    // Emit loop tail.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    IRMapping tailGemmCloneMap;
    b.clone(*blockwiseGemmV2Op, tailGemmCloneMap);

    // Apparently, the canonicalizer doesn't get rid of empty loops without
    // results properly, remove them ourselves.
    if (nIterations <= 1)
      b.eraseOp(loopOp);

    // -----

    // Matrix C write out logic.
    int64_t wavesInKernelBlock = blockSize / waveSize;

    int64_t numElements = regCVectorLen * nResultVectors;
    TopDownTMBuilder splitMemoryCoords(b, {"bid", "tid", "item"},
                                       {gridSize, blockSize, numElements}, loc);
    splitMemoryCoords.merge({"g", "m", "n"}, {0, 1, 2}, {"bid"},
                            {gridSize / gStride, gStride / nBlocks, nBlocks});
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
                          {wavesInKernelBlock / nWaves, nWaves});
    rowsAndColsWrap.passThrough({"m_tid", "n_tid"});
    rowsAndColsWrap.merge(
        {"m_i", "n_i"}, "i",
        {splitMemoryCoords.endSize("i") / nRepeats, nRepeats});

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
        {mPerBlock, mPerWave, rowGroupSize, mPerRepeat, m,
         inputSpansPerMfmaIn * rowGroupSize, 1});
    toMatrixC.embed("gemmN", 2, N, {"n", "wave_n", "n_i", "blk_col", "n_tid"},
                    {nPerBlock, nPerWave, nPerRepeat, n, 1});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();

    ArrayAttr idToMatrixCMaps = b.getArrayAttr(
        {splitMemoryCoordsAttr, toRowsAndColsAttr, toMatrixCAttr});

    Value registerC = regCAllocOp;
    auto convertedCType =
        MemRefType::get(numElements, destType, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
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
        /*bounds=*/regCAllocType.getShape(), /*strides=*/std::nullopt,
        forceUnroll, /*useIndexDiffs=*/true);
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

    b.create<ThreadwiseWriteAllOp>(loc, registerC, op.getC(), idToMatrixCMaps,
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
  target.addIllegalOp<rock::GridwiseGemmOp, rock::GridwiseGemmV2Op>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         memref::MemRefDialect, AffineDialect,
                         vector::VectorDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmRewritePattern, GridwiseGemmV2RewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
