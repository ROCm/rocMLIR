//===- GridwiseGemmToBlockwise - MLIR MIOpen ops lowering passes -----===//
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
// This pass converts miopen.gridwise_gemm[_v2] into block- and threadwise ops
//
//===-----------------------------------------------------===//
#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;

namespace {
struct LowerMIOpenOpsStep2Pass
    : public MIOpenOpsStep2PassBase<LowerMIOpenOpsStep2Pass> {
  void runOnOperation() override;
};

// TODO(kdrewnia): Could rank-0 vectors clear some of this up?
// Utility function for crafting optional vector types
Type vectorTypeOrSelf(Type elementType, int64_t len) {
  if (len == 1)
    return elementType;
  return VectorType::get({len}, elementType);
}

//===----------------------------------------------------------------------===//
// Utility function to determine the type to be loaded
//===----------------------------------------------------------------------===//
void computeLoadStoreTypeInfo(OpBuilder &b, ArrayRef<int64_t> sliceLengths,
                              int64_t loadLength, int64_t storeLength,
                              uint32_t &vectorDim, int64_t kPack,
                              Type elementType, Type &loadType,
                              Type &intermediateType, Type &storeType) {

  // In case KPack and vector load is used, and we vector load on GemmK
  // dimension (1), use the last dimension (GemmKPack) instead.
  if ((loadLength > 1) && (kPack > 1) && (vectorDim == GemmK)) {
    vectorDim = sliceLengths.size() - 1;
  }

  int64_t itemsToCopy = 1;
  for (int64_t len : sliceLengths) {
    itemsToCopy *= len;
  }
  intermediateType = vectorTypeOrSelf(elementType, itemsToCopy);
  loadType = vectorTypeOrSelf(elementType, loadLength);
  storeType = vectorTypeOrSelf(elementType, storeLength);
}

// Create a transformation domain that computes the linear, row-major iteration
// index over a rectangular space with dimensions `sliceLengths`.
// The iteration starts at all-zeros
ArrayAttr makeLinearDomain(OpBuilder &b, Location loc,
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
  TopDownCTBuilder builder(b, dimNameRefs, sliceLengths, loc);
  builder.embed("iter", 0, stride, dimNameRefs, strides);
  TransformMapAttr ret = builder.get();
  return b.getArrayAttr(ret);
}

//===----------------------------------------------------------------------===//
// Assigning attributes.
//===----------------------------------------------------------------------===//
void affixThreadwiseCopyAttributes(ThreadwiseCopyOp top, GridwiseGemmOp gop,
                                   OpBuilder &b) {
  top->setAttr("vector_read_write_dim",
               gop->getAttr("matrix_c_dest_vector_write_dim"));
  top->setAttr("source_data_per_read", gop->getAttr("matrix_c_data_per_copy"));
  top->setAttr("dest_data_per_write", gop->getAttr("matrix_c_data_per_copy"));
}

void affixThreadwiseCopyV2Attributes(ThreadwiseCopyV2Op top,
                                     GridwiseGemmV2Op gop, OpBuilder &b,
                                     bool isSwizzled, bool canOob) {
  // Account for split m/n dimension
  bool vectorStoreOverride = canOob;
  int64_t vectorGemmDim =
      gop->getAttrOfType<IntegerAttr>("matrix_c_source_vector_read_dim")
          .getInt();
  // Remap vectorized gemm dimensions to account for
  if (vectorGemmDim == gemmCDimM) {
    vectorGemmDim = gemmCSplitDimM2;
  } else if (vectorGemmDim == gemmCDimN) {
    if (isSwizzled) {
      vectorGemmDim = gemmCSplitDimN2;
    } else {
      vectorGemmDim = gemmCSplitDimN;
      // Need swizzles for this to be vector motion but swizzles are off
      vectorStoreOverride = true;
    }
  }
  Attribute dataPerCopy = gop->getAttr("matrix_c_data_per_copy");
  if (vectorStoreOverride) {
    dataPerCopy = b.getI32IntegerAttr(1);
  }
  top->setAttr("upper_vector_read_dim", b.getI32IntegerAttr(vectorGemmDim));
  top->setAttr("vector_read_write_dim",
               gop->getAttr("matrix_c_dest_vector_write_dim"));
  top->setAttr("data_per_copy", dataPerCopy);
}

//===----------------------------------------------------------------------===//
// Building load/store loops
//===----------------------------------------------------------------------===//
TransformingForOp createGlobalLoadLoop(OpBuilder &b, Location loc, Value global,
                                       ValueRange globalStart, Type resultType,
                                       Type loadType,
                                       ArrayRef<int64_t> sliceLengths,
                                       uint32_t vectorDim, ArrayAttr oobDims,
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
  global = untransform(b, global, globalTransforms);

  ArrayAttr noTransforms = b.getArrayAttr({});
  ArrayAttr resultIdxMap = makeLinearDomain(b, loc, sliceLengths);

  SmallVector<int64_t, 4> loopBounds;
  llvm::copy(sliceLengths, std::back_inserter(loopBounds));
  assert(loopBounds[vectorDim] % loadLength == 0 && "Uneven vector load");
  loopBounds[vectorDim] /= loadLength;

  SmallVector<Attribute> loopTransforms = {globalTransforms, resultIdxMap};
  if (complexVectorLoad)
    loopTransforms[1] = noTransforms;

  Value dest = createZeroConstantOp(b, loc, resultType);
  auto loop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{globalStart, linearInit}, loopTransforms,
      loopBounds,
      /*forceUnroll=*/true, useIndexDiffs, dest);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(loop.getBody());
  Value loaded = b.create<BufferLoadOp>(loc, loadType, global, oobDims,
                                        loop.getLowerCoords(0));
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
          loc, ArrayRef<ValueRange>{linearInit, loop.getLowerCoords(1)},
          ArrayRef<Attribute>{loadedValIdxMap, resultIdxMap}, vectorIdxBounds,
          /*forceUnroll=*/true, /*useIndexDiffs=*/true, loopArg);

      {
        OpBuilder::InsertionGuard innerGuard(b);
        b.setInsertionPointToStart(scatterLoop.getBody());
        Value toScatter = b.create<vector::ExtractElementOp>(
            loc, loaded, scatterLoop.getLowerCoords(0)[0]);
        Value toYieldInner = b.create<vector::InsertElementOp>(
            loc, toScatter, scatterLoop.getIterArgs()[0],
            scatterLoop.getLowerCoords(1)[0]);
        b.create<miopen::YieldOp>(loc, toYieldInner);
      }
      toYield = scatterLoop.getResults()[0];
    } else {
      toYield = b.create<InsertSliceOp>(loc, resultType, loaded, loopArg,
                                        loop.getLowerCoords(1)[0]);
    }
  }
  b.create<miopen::YieldOp>(loc, toYield);
  return loop;
}

TransformingForOp createLdsStoreLoop(OpBuilder &b, Location loc, Value loaded,
                                     Value buffer, ValueRange bufferStart,
                                     Type storingType,
                                     ArrayRef<int64_t> sliceLengths,
                                     uint32_t vectorDim) {
  Type loadedType = loaded.getType();
  bool fullyScalar = !loadedType.isa<ShapedType>();

  int64_t storeLength = 1;
  if (auto storingVectorType = storingType.dyn_cast<VectorType>())
    storeLength = storingVectorType.getNumElements();

  size_t nUpper = bufferStart.size();
  bool complexVectorStore = (storeLength > 1) && (vectorDim != nUpper - 1);

  Value zero = b.createOrFold<ConstantIndexOp>(loc, 0);
  SmallVector<Value, 5> linearInit(nUpper, zero);

  ArrayAttr bufferTransforms;
  buffer = untransform(b, buffer, bufferTransforms);
  ArrayAttr noTransforms = b.getArrayAttr({});
  ArrayAttr resultIdxMap = makeLinearDomain(b, loc, sliceLengths);

  SmallVector<int64_t, 4> loopBounds;
  llvm::copy(sliceLengths, std::back_inserter(loopBounds));
  assert(loopBounds[vectorDim] % storeLength == 0 && "Uneven vector store");
  loopBounds[vectorDim] /= storeLength;

  SmallVector<Attribute> loopTransforms = {resultIdxMap, bufferTransforms};
  if (complexVectorStore)
    loopTransforms[0] = noTransforms;

  auto loop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{linearInit, bufferStart}, loopTransforms,
      loopBounds,
      /*forceUnroll=*/true, /*useIndexDiffs=*/true);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(loop.getBody());

  // We can use vector.transfer_write if the vectorized dimension of the
  // load-store loops is the fastest-moving dimension of the store loop.
  // Otherwise, we need to gather elements from the result vector in order to
  // store them
  if (fullyScalar)
    b.create<InBoundsStoreOp>(loc, loaded, buffer, loop.getLowerCoords(1));
  else if (!complexVectorStore) {
    Value toStore = b.create<ExtractSliceOp>(loc, storingType, loaded,
                                             loop.getLowerCoords(0)[0]);
    b.create<InBoundsStoreOp>(loc, toStore, buffer, loop.getLowerCoords(1));
  } else {
    SmallVector<int64_t, 4> vectorIdxBounds(nUpper, 1);
    vectorIdxBounds[vectorDim] = storeLength;
    ArrayAttr loadedValIdxMap = makeLinearDomain(b, loc, vectorIdxBounds);

    Value gatherInit = createZeroConstantOp(b, loc, storingType);
    TransformingForOp gatherLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{loop.getLowerCoords(0), linearInit},
        ArrayRef<Attribute>{resultIdxMap, loadedValIdxMap}, vectorIdxBounds,
        /*forceUnroll=*/true, /*useIndexDiffs=*/true, gatherInit);
    {
      OpBuilder::InsertionGuard innerGuard(b);
      b.setInsertionPointToStart(gatherLoop.getBody());

      Value gatheredScalar = b.create<vector::ExtractElementOp>(
          loc, loaded, gatherLoop.getLowerCoords(0)[0]);
      Value toYield = b.create<vector::InsertElementOp>(
          loc, gatheredScalar, gatherLoop.getIterArgs()[0],
          gatherLoop.getLowerCoords(1)[0]);
      b.create<miopen::YieldOp>(loc, toYield);
    }
    Value gathered = gatherLoop.getResults()[0];
    b.create<InBoundsStoreOp>(loc, gathered, buffer, loop.getLowerCoords(1));
  }

  return loop;
}

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

/// Utility function for constructing a subview that slices a buffer as a
/// TransformOp
Value sliceBufferSubview(OpBuilder &b, Location loc, Value buffer,
                         int64_t start, int64_t length) {
  auto bufferType = buffer.getType().cast<MemRefType>();
  assert(bufferType.getRank() == 1 && "Can't slice multidimensional buffer");
  ArrayRef<int64_t> shape = bufferType.getShape();

  int64_t end = start + length;
  BottomUpCTBuilder transform(b, {"buffer"}, shape, loc);
  transform.slice({"slice"}, {"buffer"}, {start}, {end});

  TransformMapAttr transformAttr = transform.get();
  Value subview = b.create<TransformOp>(loc, buffer, transformAttr,
                                        bufferType.getMemorySpaceAsInt());
  return subview;
}

// Utility function for creating a N-D reshaped view of a subview
Value reshapeBufferSubview(OpBuilder &b, Location loc, Value buffer,
                           ArrayRef<int64_t> shape) {
  MemRefType bufferType = buffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> outShape = bufferType.getShape();
  assert(outShape.size() == 1 && "Buffer being reshaped must start linear");

  SmallVector<int64_t> strides;
  strides.reserve(shape.size());
  int64_t stride = 1;
  for (int64_t v : llvm::reverse(shape)) {
    strides.push_back(stride);
    stride *= v;
  }
  std::reverse(strides.begin(), strides.end());
  assert(stride == outShape[0] && "Strides must multiply to buffer length");

  SmallVector<SmallString<4>, 4> names;
  SmallVector<StringRef, 4> nameRefs;
  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    SmallString<4> name;
    (Twine("dim") + Twine(i)).toVector(name);
    names.push_back(name);
    nameRefs.push_back(StringRef(names[i]));
  }

  TopDownCTBuilder transform(b, nameRefs, shape, loc);
  transform.embed("slice", 0, outShape[0], nameRefs, strides);

  TransformMapAttr transformAttr = transform.get();
  Value ret = b.create<TransformOp>(loc, buffer, transformAttr,
                                    bufferType.getMemorySpaceAsInt());
  return ret;
}

struct GridwiseGemmRewritePattern : public OpRewritePattern<GridwiseGemmOp> {
  using OpRewritePattern<GridwiseGemmOp>::OpRewritePattern;

  void computeLDSBlockSizes(GridwiseGemmOp op, int64_t &a_block_space,
                            int64_t &b_block_space, int64_t &block_space,
                            int64_t KPack = 1) const {
    int64_t ABlockCopyDstDataPerWrite_M =
        op->getAttr("matrix_a_dest_data_per_write_dim_m")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t BBlockCopyDstDataPerWrite_N =
        op->getAttr("matrix_b_dest_data_per_write_dim_n")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t ThreadGemmAThreadCopySrcDataPerRead_M =
        op->getAttr("m_per_thread").template cast<IntegerAttr>().getInt();
    int64_t ThreadGemmBThreadCopySrcDataPerRead_N =
        op->getAttr("n_per_thread").template cast<IntegerAttr>().getInt();

    int64_t max_lds_align =
        math_util::lcm(ABlockCopyDstDataPerWrite_M, BBlockCopyDstDataPerWrite_N,
                       ThreadGemmAThreadCopySrcDataPerRead_M,
                       ThreadGemmBThreadCopySrcDataPerRead_N);

    int64_t KPerBlock =
        op->getAttr("k_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerBlock =
        op->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        op->getAttr("n_per_block").template cast<IntegerAttr>().getInt();

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

    // llvm::errs() << "a_block_space: " << a_block_space << "\n";
    // llvm::errs() << "b_block_space: " << b_block_space << "\n";
    // llvm::errs() << "double_block_space: " << double_block_space << "\n\n";
  }

  void affixBlockwiseGemmAttributes(BlockwiseGemmOp bop, GridwiseGemmOp gop,
                                    OpBuilder &b) const {
    bop->setAttr("block_size", gop->getAttr("block_size"));
    // Attributes used in non-xdlops lowering path.
    bop->setAttr("m_per_thread", gop->getAttr("m_per_thread"));
    bop->setAttr("n_per_thread", gop->getAttr("n_per_thread"));
    bop->setAttr("k_per_thread", gop->getAttr("k_per_thread"));
    bop->setAttr("m_level0_cluster", gop->getAttr("m_level0_cluster"));
    bop->setAttr("m_level1_cluster", gop->getAttr("m_level1_cluster"));
    bop->setAttr("n_level0_cluster", gop->getAttr("n_level0_cluster"));
    bop->setAttr("n_level1_cluster", gop->getAttr("n_level1_cluster"));

    if (gop->hasAttr("kpack"))
      bop->setAttr("kpack", gop->getAttr("kpack"));
  }

  LogicalResult matchAndRewrite(GridwiseGemmOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Obtain data type.
    auto elementType = op.b().getType().cast<MemRefType>().getElementType();

    // Determine the type used on VGPR to act as accumulator.
    // f32: f32.
    // f16: f32 to prevent overflow from happening.
    // i16(bf16) : i16.
    Type accumulatorType = elementType;
    if (elementType == b.getF16Type()) {
      accumulatorType = b.getF32Type();
    }

    // Prepare some useful constants.
    Value zeroConstantFloatOp = createZeroConstantOp(b, loc, accumulatorType);
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = op.a().getType().template cast<MemRefType>().getShape();
    bShape = op.b().getType().template cast<MemRefType>().getShape();
    cShape = op.c().getType().template cast<MemRefType>().getShape();
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

    Attribute noTransforms = b.getArrayAttr({});

    // Obtain critical tuning parameters.
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t BlockSize =
        op->getAttr("block_size").template cast<IntegerAttr>().getInt();
    int64_t KPerBlock =
        op->getAttr("k_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerBlock =
        op->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        op->getAttr("n_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerThread =
        op->getAttr("m_per_thread").template cast<IntegerAttr>().getInt();
    int64_t NPerThread =
        op->getAttr("n_per_thread").template cast<IntegerAttr>().getInt();
    auto MPerThreadConstantOp = b.create<ConstantIndexOp>(loc, MPerThread);
    auto NPerThreadConstantOp = b.create<ConstantIndexOp>(loc, NPerThread);

    int64_t MLevel0Cluster =
        op->getAttr("m_level0_cluster").template cast<IntegerAttr>().getInt();
    int64_t MLevel1Cluster =
        op->getAttr("m_level1_cluster").template cast<IntegerAttr>().getInt();
    int64_t NLevel0Cluster =
        op->getAttr("n_level0_cluster").template cast<IntegerAttr>().getInt();
    int64_t NLevel1Cluster =
        op->getAttr("n_level1_cluster").template cast<IntegerAttr>().getInt();
    auto NLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NLevel0Cluster);
    auto NLevel1ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NLevel1Cluster);

    int64_t matrix_a_source_data_per_read =
        op->getAttr("matrix_a_source_data_per_read")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_source_data_per_read =
        op->getAttr("matrix_b_source_data_per_read")
            .template cast<IntegerAttr>()
            .getInt();
    auto matrix_a_source_vector_read_dim = static_cast<GemmDimensions>(
        op->getAttr("matrix_a_source_vector_read_dim")
            .template cast<IntegerAttr>()
            .getInt());
    auto matrix_b_source_vector_read_dim = static_cast<GemmDimensions>(
        op->getAttr("matrix_b_source_vector_read_dim")
            .template cast<IntegerAttr>()
            .getInt());
    int64_t matrix_a_dest_data_per_write_dim_m =
        op->getAttr("matrix_a_dest_data_per_write_dim_m")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_dest_data_per_write_dim_n =
        op->getAttr("matrix_b_dest_data_per_write_dim_n")
            .template cast<IntegerAttr>()
            .getInt();

    PaddingInfoAttr paddingInfo = op.paddingInfo();
    // TODO(whchung): Determine the conditions for legacy load/store more
    // precisely.

    // Due to a partially-resolved compiler issue, when we had to pad out the
    // gemm so it'd evenly fit into the GPU's grid, the index diff map approach
    // yields incorrect results.
    bool useIndexDiffs = !paddingInfo.hasPadding();
    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;
    int64_t GStride = MBlockWork * NBlockWork;

    // llvm::errs() << "\ngridwise_gemm op:\n";
    // op.dump();
    // llvm::errs() << "\n";

    // llvm::errs() << "M: " << M << "\n";
    // llvm::errs() << "N: " << N << "\n";
    // llvm::errs() << "K: " << K << "\n";
    // llvm::errs() << "BlockSize: " << BlockSize << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "KPack: " << KPack << "\n";
    // llvm::errs() << "MPerThread: " << MPerThread << "\n";
    // llvm::errs() << "NPerThread: " << NPerThread << "\n";
    // llvm::errs() << "MBlockWork = M / MPerBlock: " << MBlockWork << "\n";
    // llvm::errs() << "NBlockWork = N / NPerBlock: " << NBlockWork << "\n";

    auto NBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, NBlockWork);
    auto GStridOp = b.create<ConstantIndexOp>(loc, GStride);
    auto block_work_id_g =
        b.create<DivUIOp>(loc, bid, GStridOp); // id_g of coordinate
    auto block_work_rem = b.create<RemUIOp>(loc, bid, GStridOp);
    auto block_work_id_m =
        b.create<DivUIOp>(loc, block_work_rem, NBlockWorkConstantOp);
    auto block_work_id_n =
        b.create<RemUIOp>(loc, block_work_rem, NBlockWorkConstantOp);
    auto MPerBlockConstantOp = b.create<ConstantIndexOp>(loc, MPerBlock);
    auto NPerBlockConstantOp = b.create<ConstantIndexOp>(loc, NPerBlock);
    auto m_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_m, MPerBlockConstantOp);
    auto n_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_n, NPerBlockConstantOp);

    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "KPack: " << KPack << "\n";
    // llvm::errs() << "matrix_a_source_data_per_read: "
    //              << matrix_a_source_data_per_read << "\n";
    // llvm::errs() << "matrix_b_source_data_per_read: "
    //              << matrix_b_source_data_per_read << "\n";

    // Compute ThreadSliceLengths for Matrix A.
    int64_t GemmABlockCopyNumberDataPerThread =
        MPerBlock * KPerBlock / BlockSize;

    int64_t GemmABlockCopyThreadSliceLengths_GemmK;
    int64_t GemmABlockCopyThreadSliceLengths_GemmM;
    int64_t GemmABlockCopyThreadSliceLengths_GemmKPack = 1;
    switch (matrix_a_source_vector_read_dim) {
    case GemmK:
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
    case GemmMorN:
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
    case GemmG:
      llvm::errs() << "Vector loads/stores aren't possible in the G dimension "
                      "and should not haven been attempted";
      return failure();
    }

    // llvm::errs() << "thread slice lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmM << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n";

    if (GemmABlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmM == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmKPack == 0) {
      llvm::errs() << "Blockwise copy slice lengths for matrix A is zero which "
                      "is invalid.\n";
      return failure();
    }

    // Compute ThreadClusterLengths for Matrix A.
    uint64_t GemmABlockCopyClusterLengths_GemmKPack =
        KPack / GemmABlockCopyThreadSliceLengths_GemmKPack;
    uint64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock / GemmABlockCopyThreadSliceLengths_GemmK;
    // int64_t GemmABlockCopyClusterLengths_GemmM =
    //    MPerBlock / GemmABlockCopyThreadSliceLengths_GemmM;

    // llvm::errs() << "thread cluster lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmM << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmKPack << "\n";

    // Compute ThreadSliceLengths for Matrix B.
    int64_t GemmBBlockCopyNumberDataPerThread =
        NPerBlock * KPerBlock / BlockSize;

    int64_t GemmBBlockCopyThreadSliceLengths_GemmK;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmN;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmKPack = 1;
    assert(matrix_b_source_vector_read_dim != GemmG);
    switch (matrix_b_source_vector_read_dim) {
    case GemmK:
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
    case GemmMorN:
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
    case GemmG:
      llvm::errs() << "Vector loads/stores aren't possible in the G dimension "
                      "and should not haven been attempted";
      return failure();
    }

    // llvm::errs() << "thread slice lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmN << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n";

    if (GemmBBlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmN == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmKPack == 0) {
      llvm::errs() << "Blockwise copy slice lengths for matrix B is zero which "
                      "is invalid.\n";
      return failure();
    }

    // Compute ThreadClusterLengths for Matrix B.
    uint64_t GemmBBlockCopyClusterLengths_GemmKPack =
        KPack / GemmBBlockCopyThreadSliceLengths_GemmKPack;
    uint64_t GemmBBlockCopyClusterLengths_GemmK =
        KPerBlock / GemmBBlockCopyThreadSliceLengths_GemmK;
    uint64_t GemmBBlockCopyClusterLengths_GemmN =
        NPerBlock / GemmBBlockCopyThreadSliceLengths_GemmN;

    // llvm::errs() << "thread cluster lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmN << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmKPack << "\n";

    // Get current workitem ID.

    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

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

    auto GemmDataIdBegin_G = block_work_id_g;
    auto GemmBlockCoord_G = GemmDataIdBegin_G;

    // Compute required LDS sizes.
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize, ldsBlockSize, KPack);

    // llvm::errs() << "LDS block size:" << ldsBlockASize << " " <<
    // ldsBlockBSize
    //             << " " << ldsBlockSize << "\n";

    // Allocate LDS.
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, elementType, {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsGpuAllocOp = b.create<GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for Matrix A.
    auto ldsBlockAOffset = 0;
    auto ldsBlockASubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockAOffset, ldsBlockASize);

    // Get matrix subviews.
    // Compute matrix A dimension from attributes.
    Value ldsMatrixASubviewOp;
    if (KPack > 1) {
      ldsMatrixASubviewOp = reshapeBufferSubview(
          b, loc, ldsBlockASubviewOp, {1, KPerBlock, MPerBlock, KPack});
    } else {
      ldsMatrixASubviewOp = reshapeBufferSubview(b, loc, ldsBlockASubviewOp,
                                                 {1, KPerBlock, MPerBlock});
    }

    // Subviews for Matrix B.
    auto ldsBlockBOffset = ldsBlockASize;
    auto ldsBlockBSubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockBOffset, ldsBlockBSize);

    // Get matrix subviews.
    // Compute matrix B dimension from attributes.
    Value ldsMatrixBSubviewOp;
    if (KPack > 1) {
      ldsMatrixBSubviewOp = reshapeBufferSubview(
          b, loc, ldsBlockBSubviewOp, {1, KPerBlock, NPerBlock, KPack});
    } else {
      ldsMatrixBSubviewOp = reshapeBufferSubview(b, loc, ldsBlockBSubviewOp,
                                                 {1, KPerBlock, NPerBlock});
    }

    // Alloc for Matrix C on registers.
    // Compute register size from attributes.
    int64_t GemmMRepeat = 0, GemmNRepeat = 0;

    // llvm::errs() << "MPerThread: " << MPerThread << "\n";
    // llvm::errs() << "NPerThread: " << NPerThread << "\n";

    GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
    GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

    // llvm::errs() << "GemmMRepeat: " << GemmMRepeat << "\n";
    // llvm::errs() << "GemmNRepeat: " << GemmNRepeat << "\n";

    auto threadCRegisterMemRefType = MemRefType::get(
        {1, GemmMRepeat * MPerThread, GemmNRepeat * NPerThread},
        accumulatorType, {}, gpu::GPUDialect::getPrivateAddressSpace());
    Value registerMatrixCAllocOp =
        b.create<GpuAllocOp>(loc, threadCRegisterMemRefType);

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

    uint32_t blockwiseVectorDimA = matrix_a_source_vector_read_dim;
    int64_t blockwiseLoadVectorLenA = matrix_a_source_data_per_read;
    int64_t blockwiseStoreVectorLenA = matrix_a_dest_data_per_write_dim_m;
    Type aLoadIntermediate, aLoadType, aStoreType;
    computeLoadStoreTypeInfo(b, blockwiseCopyABounds, blockwiseLoadVectorLenA,
                             blockwiseStoreVectorLenA, blockwiseVectorDimA,
                             KPack, elementType, aLoadType, aLoadIntermediate,
                             aStoreType);

    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmK: "
    //              << GemmABlockCopyThreadSliceLengths_GemmK << "\n";
    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmM: "
    //              << GemmABlockCopyThreadSliceLengths_GemmM << "\n";
    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmKPack: "
    //              << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n";
    // llvm::errs() << "blockwise copy A bounds: ";
    // for (auto v : blockwiseCopyABounds)
    //   llvm::errs() << v << " ";
    // llvm::errs() << "\n";

    SmallVector<int64_t, 4> blockwiseCopyBBounds;
    if (KPack > 1) {
      blockwiseCopyBBounds = {1, GemmBBlockCopyThreadSliceLengths_GemmK,
                              GemmBBlockCopyThreadSliceLengths_GemmN,
                              GemmBBlockCopyThreadSliceLengths_GemmKPack};
    } else {
      blockwiseCopyBBounds = {1, GemmBBlockCopyThreadSliceLengths_GemmK,
                              GemmBBlockCopyThreadSliceLengths_GemmN};
    }
    uint32_t blockwiseVectorDimB = matrix_b_source_vector_read_dim;
    int64_t blockwiseLoadVectorLenB = matrix_b_source_data_per_read;
    int64_t blockwiseStoreVectorLenB = matrix_b_dest_data_per_write_dim_n;
    Type bLoadIntermediate, bLoadType, bStoreType;
    computeLoadStoreTypeInfo(b, blockwiseCopyBBounds, blockwiseLoadVectorLenB,
                             blockwiseStoreVectorLenB, blockwiseVectorDimB,
                             KPack, elementType, bLoadType, bLoadIntermediate,
                             bStoreType);

    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmK: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmK << "\n";
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmN: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmN << "\n";
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmKPack: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n";
    // llvm::errs() << "blockwise copy B bounds: ";
    // for (auto v : blockwiseCopyBBounds)
    //   llvm::errs() << v << " ";
    // llvm::errs() << "\n";

    // Zero init Matrix C on registers.
    b.create<FillOp>(loc, registerMatrixCAllocOp, zeroConstantFloatOp);

    // Blockwise copies before the loop.
    // Blockwise copy from global (generic tensor) to LDS (naive tensor).

    // Compute source and destination coordinates for BlockwiseCopy ops.
    // Matrix A: {0, 0, m_block_data_on_global}, {0, 0, 0}
    // Matrix B: {0, 0, n_block_data_on_global}, {0, 0, 0}

    Value mMyThreadOffsetA, mMyThreadOffsetB;
    Value c_thread_mtx_index_row, c_thread_mtx_index_col;
    Value m_thread_data_on_global, n_thread_data_on_global;

    // Compute c_thread_mtx_index for Matrix C.
    int64_t ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;
    auto ThreadPerLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, ThreadPerLevel0Cluster);
    auto level1_id =
        b.create<DivUIOp>(loc, tid, ThreadPerLevel0ClusterConstantOp);
    auto level1_m_id =
        b.create<DivUIOp>(loc, level1_id, NLevel1ClusterConstantOp);
    auto level1_n_id =
        b.create<RemUIOp>(loc, level1_id, NLevel1ClusterConstantOp);

    auto level0_id =
        b.create<RemUIOp>(loc, tid, ThreadPerLevel0ClusterConstantOp);
    auto level0_m_id =
        b.create<DivUIOp>(loc, level0_id, NLevel0ClusterConstantOp);
    auto level0_n_id =
        b.create<RemUIOp>(loc, level0_id, NLevel0ClusterConstantOp);

    int64_t MPerLevel0Cluster = MPerThread * MLevel0Cluster;
    int64_t NPerLevel0Cluster = NPerThread * NLevel0Cluster;
    auto MPerLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, MPerLevel0Cluster);
    auto NPerLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NPerLevel0Cluster);

    // mMyThreadOffsetA = BlockMatrixA::GetOffsetFromMultiIndex{0,
    // c_thread_mtx_index.row} = c_thread_mtx_index_row
    c_thread_mtx_index_row = b.create<AddIOp>(
        loc, b.create<MulIOp>(loc, level1_m_id, MPerLevel0ClusterConstantOp),
        b.create<MulIOp>(loc, level0_m_id, MPerThreadConstantOp));
    mMyThreadOffsetA = c_thread_mtx_index_row;

    // mMyThreadOffsetB = BlockMatrixB::GetOffsetFromMultiIndex{0,
    // c_thread_mtx_index.col} = c_thread_mtx_index_col
    c_thread_mtx_index_col = b.create<AddIOp>(
        loc, b.create<MulIOp>(loc, level1_n_id, NPerLevel0ClusterConstantOp),
        b.create<MulIOp>(loc, level0_n_id, NPerThreadConstantOp));
    mMyThreadOffsetB = c_thread_mtx_index_col;

    m_thread_data_on_global =
        b.create<AddIOp>(loc, m_block_data_on_global, c_thread_mtx_index_row);
    n_thread_data_on_global =
        b.create<AddIOp>(loc, n_block_data_on_global, c_thread_mtx_index_col);

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
    TransformingForOp blockwiseLoadA =
        createGlobalLoadLoop(b, loc, op.a(), blockwiseLoadACoords,
                             aLoadIntermediate, aLoadType, blockwiseCopyABounds,
                             blockwiseVectorDimA, op.aOobDims(), useIndexDiffs);
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
        blockwiseStoreACoords, aStoreType, blockwiseCopyABounds,
        blockwiseVectorDimA);

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
    TransformingForOp blockwiseLoadB =
        createGlobalLoadLoop(b, loc, op.b(), blockwiseLoadBCoords,
                             bLoadIntermediate, bLoadType, blockwiseCopyBBounds,
                             blockwiseVectorDimB, op.bOobDims(), useIndexDiffs);

    SmallVector<Value, 4> blockwiseStoreBCoords;
    if (KPack > 1) {
      blockwiseStoreBCoords = {zeroConstantOp, GemmBBlockCopyDestCoord_Z,
                               GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    } else {
      blockwiseStoreBCoords = {zeroConstantOp, GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    }
    // Emit blockwise store for matrix B.
    TransformingForOp blockwiseStoreB = createLdsStoreLoop(
        b, loc, blockwiseLoadB.getResult(0), ldsMatrixBSubviewOp,
        blockwiseStoreBCoords, bStoreType, blockwiseCopyBBounds,
        blockwiseVectorDimB);

    // Emit loop.
    // Compute loop iterations from attributes.

    auto KPerBlockConstantOp = b.create<ConstantIndexOp>(loc, KPerBlock);

    int64_t loopIteration = (K - KPerBlock) / KPerBlock;

    // Assign iter args.
    // 0: blockwise copy A src y coordinate.
    // 1: blockwise copy B src y coordinate.
    SmallVector<Value, 2> iterArgs = {blockwiseLoadACoords[1],
                                      blockwiseLoadBCoords[1]};

    auto loopOp = b.create<AffineForOp>(loc, 0, loopIteration, 1, iterArgs);

    // inside the loop.
    auto lb = OpBuilder::atBlockBegin(loopOp.getBody(), b.getListener());

    // LDS barrier.
    lb.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM.
    auto blockwiseGemmOp = lb.create<BlockwiseGemmOp>(
        loc, ldsMatrixASubviewOp, ldsMatrixBSubviewOp, registerMatrixCAllocOp,
        mMyThreadOffsetA, mMyThreadOffsetB);
    affixBlockwiseGemmAttributes(blockwiseGemmOp, op, b);

    // LDS barrier.
    // This barrier prevents halo part of outputs having weird values.
    lb.create<LDSBarrierOp>(loc);

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    const auto &args = loopOp.getRegionIterArgs();
    // Emit blockwise load for matrix A.
    Value blockwiseCopyASrcUpdated =
        lb.create<AddIOp>(loc, args[0], KPerBlockConstantOp);
    BlockAndValueMapping loadAUpdates;
    loadAUpdates.map(blockwiseLoadACoords[1], blockwiseCopyASrcUpdated);
    auto blockwiseLoadAClone = cast<TransformingForOp>(
        lb.clone(*blockwiseLoadA.getOperation(), loadAUpdates));

    // Emit blockwise load for matrix B.
    BlockAndValueMapping loadBUpdates;
    Value blockwiseCopyBSrcUpdated =
        lb.create<AddIOp>(loc, args[1], KPerBlockConstantOp);
    loadBUpdates.map(blockwiseLoadBCoords[1], blockwiseCopyBSrcUpdated);
    auto blockwiseLoadBClone = cast<TransformingForOp>(
        lb.clone(*blockwiseLoadB.getOperation(), loadBUpdates));
    // Blockwise copy from register (naive tensor) to LDS (naive tensor).

    // Emit blockwise stores
    BlockAndValueMapping storeAUpdates, storeBUpdates;
    storeAUpdates.map(blockwiseLoadA.getResult(0),
                      blockwiseLoadAClone.getResult(0));
    storeBUpdates.map(blockwiseLoadB.getResult(0),
                      blockwiseLoadBClone.getResult(0));
    lb.clone(*blockwiseStoreA.getOperation(), storeAUpdates);
    lb.clone(*blockwiseStoreB.getOperation(), storeBUpdates);

    // update iter args.
    // blockwiseCopyASrcVector and blockwiseCopyBSrcVector are updated.
    iterArgs[0] = blockwiseCopyASrcUpdated;
    iterArgs[1] = blockwiseCopyBSrcUpdated;
    // emit loop yield so iter args can be passed to the next iteration.
    lb.create<AffineYieldOp>(loc, iterArgs);

    // outside the loop.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    auto blockwiseGemmTailOp = b.create<BlockwiseGemmOp>(
        loc, ldsMatrixASubviewOp, ldsMatrixBSubviewOp, registerMatrixCAllocOp,
        mMyThreadOffsetA, mMyThreadOffsetB);
    affixBlockwiseGemmAttributes(blockwiseGemmTailOp, op, b);

    // Threadwise copy from register (naive tensor) to global (generic tensor).
    int64_t M1 = MPerThread * MLevel0Cluster * MLevel1Cluster;
    int64_t M0 = M / M1;
    int64_t N1 = NPerThread * NLevel0Cluster * NLevel1Cluster;
    int64_t N0 = N / N1;

    auto M1ConstantOp = b.create<ConstantIndexOp>(loc, M1);
    auto N1ConstantOp = b.create<ConstantIndexOp>(loc, N1);

    // Build transformation that unsplits the output matrix for writing
    // by (g, m0, m1, n0, n1) -> (g, m0 * M1 + m1, n0 * N1, n1)
    TopDownCTBuilder cSplitTransform(b, {"G", "M0", "M1", "N0", "N1"},
                                     {G, M0, M1, N0, N1}, loc);
    cSplitTransform.passThrough({"gemmG"}, {0}, {"G"});
    cSplitTransform.embed("gemmM", 1, M1 * M0, {"M0", "M1"}, {M1, 1});
    cSplitTransform.embed("gemmN", 2, N1 * N0, {"N0", "N1"}, {N1, 1});

    TransformMapAttr cSplitTransformAttr = cSplitTransform.get();
    auto cTransformed = b.create<TransformOp>(loc, op.c(), cSplitTransformAttr);

    // Build transformation that maps the in-regester results to
    // three dimensions for writing with
    //  (g, m0, m1, n0, n1) -> (g, m0 * MPerThread + m1, n0 * NPerThread + n1)
    SmallVector<int64_t, 5> copyBounds = {1, GemmMRepeat, MPerThread,
                                          GemmNRepeat, NPerThread};
    TopDownCTBuilder registerCTransform(
        b, {"g", "gemmMRepeat", "mPerThread", "gemmNRepeat", "nPerThread"},
        copyBounds, loc);
    registerCTransform.passThrough({"gemmG"}, {0}, {"g"});
    registerCTransform.embed("gemmM", 1, GemmMRepeat * MPerThread,
                             {"gemmMRepeat", "mPerThread"}, {MPerThread, 1});
    registerCTransform.embed("gemmN", 2, GemmNRepeat * NPerThread,
                             {"gemmNRepeat", "nPerThread"}, {NPerThread, 1});

    TransformMapAttr registerCTransformAttr = registerCTransform.get();
    Value registerCTransformed = b.create<TransformOp>(
        loc, registerMatrixCAllocOp, registerCTransformAttr,
        gpu::GPUDialect::getPrivateAddressSpace());

    SmallVector<Value, 5> matrixCThreadwiseCopySourceCoords;
    std::fill_n(std::back_inserter(matrixCThreadwiseCopySourceCoords), 5,
                zeroConstantOp.getResult());

    SmallVector<Value, 5> matrixCThreadwiseCopyDestCoords = {
        GemmDataIdBegin_G,
        b.create<DivUIOp>(loc, m_thread_data_on_global, M1ConstantOp),
        b.create<RemUIOp>(loc, m_thread_data_on_global, M1ConstantOp),
        b.create<DivUIOp>(loc, n_thread_data_on_global, N1ConstantOp),
        b.create<RemUIOp>(loc, n_thread_data_on_global, N1ConstantOp)};
    // g index

    auto threadwiseCopyCMatrixOp = b.create<ThreadwiseCopyOp>(
        loc, registerCTransformed, cTransformed,
        b.getIndexArrayAttr(copyBounds),
        b.getArrayAttr({noTransforms, noTransforms}), op.paddingInfo(),
        op.cOobDims(), matrixCThreadwiseCopySourceCoords,
        matrixCThreadwiseCopyDestCoords,
        /*legacyLoad=*/nullptr, /*legacyStore=*/nullptr);
    affixThreadwiseCopyAttributes(threadwiseCopyCMatrixOp, op, b);

    op.erase();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// GridwiseGemmV2 lowering.
//===----------------------------------------------------------------------===//

struct GridwiseGemmV2RewritePattern
    : public OpRewritePattern<GridwiseGemmV2Op> {
  using OpRewritePattern<GridwiseGemmV2Op>::OpRewritePattern;

  void computeLDSBlockSizes(GridwiseGemmV2Op op, int64_t &a_block_space,
                            int64_t &b_block_space, int64_t &total_block_space,
                            int64_t KPack = 1) const {
    int64_t ABlockCopyDstDataPerWrite_M =
        op->getAttr("matrix_a_dest_data_per_write_dim_m")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t BBlockCopyDstDataPerWrite_N =
        op->getAttr("matrix_b_dest_data_per_write_dim_n")
            .template cast<IntegerAttr>()
            .getInt();

    int64_t max_lds_align = math_util::lcm(ABlockCopyDstDataPerWrite_M,
                                           BBlockCopyDstDataPerWrite_N);

    int64_t KPerBlock =
        op->getAttr("k_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerBlock =
        op->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        op->getAttr("n_per_block").template cast<IntegerAttr>().getInt();

    int64_t AlignedNPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(NPerBlock, max_lds_align);

    // A matrix in LDS memory, dst of blockwise copy
    int64_t AlignedMPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(MPerBlock, max_lds_align);

    // llvm::errs() << "MPerBlock : " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock : " << NPerBlock << "\n";
    // llvm::errs() << "max_lds_align : " << max_lds_align << "\n";
    // llvm::errs() << "AlignedMPerBlock : " << AlignedMPerBlock << "\n";
    // llvm::errs() << "AlignedNPerBlock : " << AlignedNPerBlock << "\n";

    a_block_space = math_util::integer_least_multiple(
                        KPerBlock * AlignedMPerBlock, max_lds_align) *
                    KPack;

    // B matrix in LDS memory, dst of blockwise copy
    b_block_space = math_util::integer_least_multiple(
                        KPerBlock * AlignedNPerBlock, max_lds_align) *
                    KPack;

    total_block_space = a_block_space + b_block_space;

    // llvm::errs() << "a_block_space: " << a_block_space << "\n";
    // llvm::errs() << "b_block_space: " << b_block_space << "\n";
    // llvm::errs() << "total_block_space: " << total_block_space << "\n\n";
  }

  void affixBlockwiseGemmV2Attributes(BlockwiseGemmV2Op bop,
                                      GridwiseGemmV2Op gop, int64_t m,
                                      int64_t k, int64_t n,
                                      OpBuilder &b) const {
    bop->setAttr("block_size", gop->getAttr("block_size"));

    int64_t MPerBlock =
        gop->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        gop->getAttr("n_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerWave =
        gop->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        gop->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();
    int64_t MWaves = MPerBlock / MPerWave;
    int64_t NWaves = NPerBlock / NPerWave;

    bop->setAttr("m_per_wave", gop->getAttr("m_per_wave"));
    bop->setAttr("n_per_wave", gop->getAttr("n_per_wave"));
    bop->setAttr("m_waves", b.getI32IntegerAttr(MWaves));
    bop->setAttr("n_waves", b.getI32IntegerAttr(NWaves));

    bop->setAttr("m", b.getI32IntegerAttr(m));
    bop->setAttr("n", b.getI32IntegerAttr(n));
    bop->setAttr("k", b.getI32IntegerAttr(k));

    if (gop->hasAttr("kpack"))
      bop->setAttr("kpack", gop->getAttr("kpack"));
  }

  LogicalResult matchAndRewrite(GridwiseGemmV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Obtain data type.
    auto elementType = op.b().getType().cast<MemRefType>().getElementType();

    // Prepare some useful constants.
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = op.a().getType().template cast<MemRefType>().getShape();
    bShape = op.b().getType().template cast<MemRefType>().getShape();
    cShape = op.c().getType().template cast<MemRefType>().getShape();
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
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t BlockSize =
        op->getAttr("block_size").template cast<IntegerAttr>().getInt();
    int64_t KPerBlock =
        op->getAttr("k_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerBlock =
        op->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        op->getAttr("n_per_block").template cast<IntegerAttr>().getInt();

    int64_t matrix_a_source_data_per_read =
        op->getAttr("matrix_a_source_data_per_read")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_source_data_per_read =
        op->getAttr("matrix_b_source_data_per_read")
            .template cast<IntegerAttr>()
            .getInt();
    auto matrix_a_source_vector_read_dim = static_cast<GemmDimensions>(
        op->getAttr("matrix_a_source_vector_read_dim")
            .template cast<IntegerAttr>()
            .getInt());
    auto matrix_b_source_vector_read_dim = static_cast<GemmDimensions>(
        op->getAttr("matrix_b_source_vector_read_dim")
            .template cast<IntegerAttr>()
            .getInt());
    int64_t matrix_a_dest_data_per_write_dim_m =
        op->getAttr("matrix_a_dest_data_per_write_dim_m")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_dest_data_per_write_dim_n =
        op->getAttr("matrix_b_dest_data_per_write_dim_n")
            .template cast<IntegerAttr>()
            .getInt();

    // Obtain XDLOPS-related attributes.
    int64_t MPerWave =
        op->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();
    // int64_t MWaves = MPerBlock / MPerWave;
    int64_t NWaves = NPerBlock / NPerWave;

    auto MPerWaveConstantOp = b.create<ConstantIndexOp>(loc, MPerWave);
    auto NPerWaveConstantOp = b.create<ConstantIndexOp>(loc, NPerWave);
    auto NWavesConstantOp = b.create<ConstantIndexOp>(loc, NWaves);

    int64_t WaveSize = 64;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, WaveSize);

    PaddingInfoAttr paddingInfo = op.paddingInfo();
    // TODO(whchung): Determine the conditions for legacy load/store more
    // precisely.

    // Due to a partially-resolved compiler issue, when we had to pad out the
    // gemm so it'd evenly fit into the GPU's grid, the index diff map approach
    // yields incorrect results.
    bool useIndexDiffs = !paddingInfo.hasPadding();

    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());

    // Get current workitem ID.
    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;
    int64_t GStride = MBlockWork * NBlockWork;

    // llvm::errs() << "M: " << M << "\n";
    // llvm::errs() << "N: " << N << "\n";
    // llvm::errs() << "K: " << K << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "KPack: " << KPack << "\n";
    // llvm::errs() << "MBlockWork = M / MPerBlock: " << MBlockWork << "\n";
    // llvm::errs() << "NBlockWork = N / NPerBlock: " << NBlockWork << "\n";
    // llvm::errs() << "MPerWave: " << MPerWave << "\n";
    // llvm::errs() << "NPerWave: " << NPerWave << "\n";
    // llvm::errs() << "MWaves = MPerBlock / MPerWave: " << MWaves << "\n";
    // llvm::errs() << "NWaves = NPerBlock / NPerWave: " << NWaves << "\n";

    // llvm::errs() << "matrix_a_source_data_per_read: "
    //              << matrix_a_source_data_per_read << "\n";
    // llvm::errs() << "matrix_b_source_data_per_read: "
    //              << matrix_b_source_data_per_read << "\n";
    // llvm::errs() << "matrix_a_source_vector_read_dim: "
    //              << matrix_a_source_vector_read_dim << "\n";
    // llvm::errs() << "matrix_b_source_vector_read_dim: "
    //              << matrix_b_source_vector_read_dim << "\n";

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
        MPerBlock * KPerBlock * KPack / BlockSize;

    // llvm::errs() << "GemmABlockCopyNumberDataPerThread: "
    //              << GemmABlockCopyNumberDataPerThread << "\n";

    int64_t GemmABlockCopyThreadSliceLengths_GemmK;
    int64_t GemmABlockCopyThreadSliceLengths_GemmM;
    int64_t GemmABlockCopyThreadSliceLengths_GemmKPack = 1;
    switch (matrix_a_source_vector_read_dim) {
    case GemmK:
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
    case GemmMorN:
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
    case GemmG:
      llvm::errs() << "Vector loads/stores aren't possible in the G dimension "
                      "and should not haven been attempted";
      return failure();
    }

    // llvm::errs() << "thread slice lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmM << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n";

    if (GemmABlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmM == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmKPack == 0) {
      llvm::errs() << "Blockwise copy slice lengths for matrix A is zero which "
                      "is invalid.\n";
      return failure();
    }

    // Compute ThreadClusterLengths for Matrix A.
    int64_t GemmABlockCopyClusterLengths_GemmKPack =
        KPack / GemmABlockCopyThreadSliceLengths_GemmKPack;
    int64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock / GemmABlockCopyThreadSliceLengths_GemmK;
    // int64_t GemmABlockCopyClusterLengths_GemmM =
    //    MPerBlock / GemmABlockCopyThreadSliceLengths_GemmM;

    // llvm::errs() << "thread cluster lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmM << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmKPack << "\n";

    // Compute ThreadSliceLengths for Matrix B.
    int64_t GemmBBlockCopyNumberDataPerThread =
        NPerBlock * KPerBlock * KPack / BlockSize;

    // llvm::errs() << "GemmBBlockCopyNumberDataPerThread: "
    //              << GemmBBlockCopyNumberDataPerThread << "\n";

    int64_t GemmBBlockCopyThreadSliceLengths_GemmK;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmN;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmKPack = 1;
    switch (matrix_b_source_vector_read_dim) {
    case GemmK:
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
    case GemmMorN:
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
    case GemmG:
      llvm::errs() << "Vector loads/stores aren't possible in the G dimension "
                      "and should not haven been attempted";
      return failure();
    }

    // llvm::errs() << "thread slice lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmN << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n";

    if (GemmBBlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmN == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmKPack == 0) {
      llvm::errs() << "Blockwise copy slice lengths for matrix B is zero which "
                      "is invalid.\n";
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

    // llvm::errs() << "thread cluster lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmN << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmKPack << "\n";

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
    computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize, ldsBlockSize, KPack);

    // llvm::errs() << "KPack: " << KPack << "\n";
    // llvm::errs() << "LDS block size:" << ldsBlockASize << " " <<
    // ldsBlockBSize
    //              << " " << ldsBlockSize << "\n";

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
      ldsMatrixASubviewOp = reshapeBufferSubview(
          b, loc, ldsBlockASubviewOp, {1, KPerBlock, MPerBlock, KPack});
    } else {
      ldsMatrixASubviewOp = reshapeBufferSubview(b, loc, ldsBlockASubviewOp,
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
      ldsMatrixBSubviewOp = reshapeBufferSubview(
          b, loc, ldsBlockBSubviewOp, {1, KPerBlock, NPerBlock, KPack});
    } else {
      ldsMatrixBSubviewOp = reshapeBufferSubview(b, loc, ldsBlockBSubviewOp,
                                                 {1, KPerBlock, NPerBlock});
    }

    ArrayAttr noTransforms = b.getArrayAttr({});
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

    uint32_t blockwiseVectorDimA = matrix_a_source_vector_read_dim;
    int64_t blockwiseLoadVectorLenA = matrix_a_source_data_per_read;
    int64_t blockwiseStoreVectorLenA = matrix_a_dest_data_per_write_dim_m;
    Type aLoadIntermediate, aLoadType, aStoreType;
    computeLoadStoreTypeInfo(b, blockwiseCopyABounds, blockwiseLoadVectorLenA,
                             blockwiseStoreVectorLenA, blockwiseVectorDimA,
                             KPack, elementType, aLoadType, aLoadIntermediate,
                             aStoreType);

    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmK: "
    //              << GemmABlockCopyThreadSliceLengths_GemmK << "\n";
    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmM: "
    //              << GemmABlockCopyThreadSliceLengths_GemmM << "\n";
    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmKPack: "
    //              << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n";
    // llvm::errs() << "blockwise copy A bounds: ";
    // for (auto v : blockwiseCopyABounds)
    //   llvm::errs() << v << " ";
    // llvm::errs() << "\n";

    // llvm::errs() << "vector load dim: " << blockwiseAVectorDim << "\n";
    // llvm::errs() << "element type: " << blockwiseLoadAType << "\n";
    // llvm::errs() << "load size: " << blockwiseLoadAVectorLength << "\n";
    // llvm::errs() << "store size: " << blockwiseStoreAVectorLength << "\n";

    SmallVector<int64_t, 4> blockwiseCopyBBounds;
    if (KPack > 1) {
      blockwiseCopyBBounds = {1, GemmBBlockCopyThreadSliceLengths_GemmK,
                              GemmBBlockCopyThreadSliceLengths_GemmN,
                              GemmBBlockCopyThreadSliceLengths_GemmKPack};
    } else {
      blockwiseCopyBBounds = {1, GemmBBlockCopyThreadSliceLengths_GemmK,
                              GemmBBlockCopyThreadSliceLengths_GemmN};
    }
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmK: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmK << "\n";
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmN: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmN << "\n";
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmKPack: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n";
    // llvm::errs() << "blockwise copy B bounds: ";
    // for (auto v : blockwiseCopyBBounds)
    //   llvm::errs() << v << " ";
    // llvm::errs() << "\n";

    uint32_t blockwiseVectorDimB = matrix_b_source_vector_read_dim;
    int64_t blockwiseLoadVectorLenB = matrix_b_source_data_per_read;
    int64_t blockwiseStoreVectorLenB = matrix_b_dest_data_per_write_dim_n;
    Type bLoadIntermediate, bLoadType, bStoreType;
    computeLoadStoreTypeInfo(b, blockwiseCopyBBounds, blockwiseLoadVectorLenB,
                             blockwiseStoreVectorLenB, blockwiseVectorDimB,
                             KPack, elementType, bLoadType, bLoadIntermediate,
                             bStoreType);

    // llvm::errs() << "vector load dim: " << blockwiseBVectorDim << "\n";
    // llvm::errs() << "element type: " << blockwiseLoadBType << "\n";
    // llvm::errs() << "load size: " << blockwiseLoadBVectorLength << "\n";
    // llvm::errs() << "store size: " << blockwiseStoreBVectorLength << "\n";

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
    TransformingForOp blockwiseLoadA =
        createGlobalLoadLoop(b, loc, op.a(), blockwiseLoadACoords,
                             aLoadIntermediate, aLoadType, blockwiseCopyABounds,
                             blockwiseVectorDimA, op.aOobDims(), useIndexDiffs);

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
        blockwiseStoreACoords, aStoreType, blockwiseCopyABounds,
        blockwiseVectorDimA);

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
    TransformingForOp blockwiseLoadB =
        createGlobalLoadLoop(b, loc, op.b(), blockwiseLoadBCoords,
                             bLoadIntermediate, bLoadType, blockwiseCopyBBounds,
                             blockwiseVectorDimB, op.bOobDims(), useIndexDiffs);

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
        blockwiseStoreBCoords, bStoreType, blockwiseCopyBBounds,
        blockwiseVectorDimB);

    // -----

    // Logic to do XDLOPS code selection.
    // llvm::errs() << "Invoke XDLOPS code selection logic:\n";
    // llvm::errs() << "elementType: "; elementType.dump(); llvm::errs() <<
    // "\n"; llvm::errs() << "MPerWave: " << MPerWave << "\n"; llvm::errs() <<
    // "NPerWave: " << NPerWave << "\n";

    XdlopsCodeSelection xcs =
        XdlopsCodeSelection::get(elementType, MPerWave, NPerWave, b);

    // Extract values from XdlopsCodeSelection.
    int64_t MPerXdlops = xcs.MPerXdlops;
    int64_t NPerXdlops = xcs.NPerXdlops;
    int64_t MRepeats = xcs.MRepeats;
    int64_t NRepeats = xcs.NRepeats;
    VectorType vectorType = xcs.vectorType;
    int64_t vectorNumber = xcs.vectorNumber;
    SmallVector<SmallVector<unsigned, 3>, 2> imms = xcs.imms;

    int64_t group_size = xcs.group_size;
    int64_t num_groups_blk = xcs.num_groups_blk;
    int64_t num_threads_blk = xcs.num_threads_blk;
    int64_t wave_size = xcs.wave_size;
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t m = xcs.m;
    int64_t n = xcs.n;

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

    bool IsKReduction = (num_output_blks == 1) && (num_input_blks > 1);
    int64_t arrayASize = (!IsKReduction)
                             ? (KPerBlock * MRepeats)
                             : (KPerBlock / num_input_blks * MRepeats);
    int64_t arrayBSize = (!IsKReduction)
                             ? (KPerBlock * NRepeats)
                             : (KPerBlock / num_input_blks * NRepeats);
    Type arrayAType, arrayBType;
    if (KPack > 1) {
      arrayAType =
          MemRefType::get({arrayASize}, VectorType::get({KPack}, elementType),
                          {}, gpu::GPUDialect::getPrivateAddressSpace());
      arrayBType =
          MemRefType::get({arrayBSize}, VectorType::get({KPack}, elementType),
                          {}, gpu::GPUDialect::getPrivateAddressSpace());
    } else {
      arrayAType = MemRefType::get({arrayASize}, elementType, {},
                                   gpu::GPUDialect::getPrivateAddressSpace());
      arrayBType = MemRefType::get({arrayBSize}, elementType, {},
                                   gpu::GPUDialect::getPrivateAddressSpace());
    }
    auto arrayA = b.create<GpuAllocOp>(loc, arrayAType);
    auto arrayB = b.create<GpuAllocOp>(loc, arrayBType);

    // -----

    // Logic to allocate 0-initialized vectors for C.
    SmallVector<Value, 4> vectorCs;
    SmallVector<Type, 4> vectorCTypes;
    auto vectorZeroConst = createZeroConstantOp(b, loc, vectorType);
    std::fill_n(std::back_inserter(vectorCs), vectorNumber, vectorZeroConst);
    std::fill_n(std::back_inserter(vectorCTypes), vectorNumber, vectorType);

    // -----

    // Emit loop.

    int64_t loopIteration = (K - KPerBlock) / KPerBlock;

    // Assign iter args.
    // 0: blockwise copy A src y coordinate.
    // 1: blockwise copy B src y coordinate.
    // 2-x : vectorCs.
    SmallVector<Value, 6> iterArgs = {blockwiseLoadACoords[1],
                                      blockwiseLoadBCoords[1]};
    iterArgs.append(vectorCs);

    auto mfmaLoopOp = b.create<AffineForOp>(loc, 0, loopIteration, 1, iterArgs);

    // inside the loop.
    auto mfmalb = OpBuilder::atBlockBegin(mfmaLoopOp.getBody());

    const auto &mfmalArgs = mfmaLoopOp.getRegionIterArgs();
    // get vectorCs for this iteration.
    std::copy(mfmalArgs.begin() + 2, mfmalArgs.end(), vectorCs.begin());

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
    auto blockwiseGemmV2Op = mfmalb.create<BlockwiseGemmV2Op>(
        loc, vectorCTypes, ldsGpuAllocOp, ldsGpuAllocOp,
        b.getIndexAttr(ldsBlockAOffset), b.getIndexAttr(ldsBlockBOffset),
        mMyWaveOffsetA, mMyWaveOffsetB, arrayA, arrayB, vectorCs);
    affixBlockwiseGemmV2Attributes(blockwiseGemmV2Op, op, MPerBlock, KPerBlock,
                                   NPerBlock, b);

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
    // blockwise_gemm_v2 updates iter args[4-].
    std::copy(blockwiseGemmV2Op.getResults().begin(),
              blockwiseGemmV2Op.getResults().end(), iterArgs.begin() + 2);

    // emit loop yield so iter args can be passed to the next iteration.
    mfmalb.create<AffineYieldOp>(loc, iterArgs);
    // outside the loop.

    // Emit loop tail.

    // LDS barrier.
    b.create<LDSBarrierOp>(loc);

    // get vectorCs for loop tail.
    std::copy(mfmaLoopOp.getResults().begin() + 2,
              mfmaLoopOp.getResults().end(), vectorCs.begin());

    // Emit blockwise GEMM for the loop tail.
    auto blockwiseGemmV2TailOp = b.create<BlockwiseGemmV2Op>(
        loc, vectorCTypes, ldsGpuAllocOp, ldsGpuAllocOp,
        b.getIndexAttr(ldsBlockAOffset), b.getIndexAttr(ldsBlockBOffset),
        mMyWaveOffsetA, mMyWaveOffsetB, arrayA, arrayB, vectorCs);
    affixBlockwiseGemmV2Attributes(blockwiseGemmV2TailOp, op, MPerBlock,
                                   KPerBlock, NPerBlock, b);

    // -----

    // Matrix C write out logic.

    // Original C++ logic.
    // __device__ static constexpr index_t GetNumBlksPerXdlops() {
    //     return (MPerXdlops * NPerXdlops) / (mfma_type.m * mfma_type.n);
    // }
    //
    // struct OutputLayout {
    //     __device__ static constexpr index_t GetBlkSize() { return
    //     mfma_type.num_regs_blk; }
    //     __device__ static constexpr index_t GetNumBlks() {
    //         return GetNumBlksPerXdlops() * MRepeats * NRepeats;
    //     }
    // };
    // using CThreadCopySliceLengths = Sequence<M0, 1, M2, 1>;
    // constexpr index_t BlkSize = blockwise_gemm.GetBlkSize();
    // constexpr index_t NumBlks = blockwise_gemm.GetNumBlks();

    // int64_t BlkSize = xcs.num_regs_blk;
    int64_t NumBlksPerXdlops = (MPerXdlops * NPerXdlops) / (m * n);
    int64_t NumBlks = NumBlksPerXdlops * MRepeats * NRepeats;

    int64_t iterationsPerVectorC = NumBlks / vectorNumber;
    int64_t vectorCoffset = vectorType.getShape()[0] / iterationsPerVectorC;
    VectorType vectorCSliceType =
        VectorType::get({vectorCoffset}, vectorType.getElementType());

    // llvm::errs() << "MPerXlops: " << MPerXdlops << "\n";
    // llvm::errs() << "NPerXlops: " << NPerXdlops << "\n";
    // llvm::errs() << "m: " << m << "\n";
    // llvm::errs() << "n: " << n << "\n";
    // llvm::errs() << "MRepeat: " << MRepeats << "\n";
    // llvm::errs() << "NRepeat: " << NRepeats << "\n\n";

    // llvm::errs() << "BlkSize: " << BlkSize << "\n";
    // llvm::errs() << "NumBlksPerXdlops: " << NumBlksPerXdlops << "\n";
    // llvm::errs() << "NumBlks: " << NumBlks << "\n\n";

    // llvm::errs() << "iterationsPerVectorC: " << iterationsPerVectorC << "\n";
    // llvm::errs() << "vectorCoffset: " << vectorCoffset << "\n";

    auto group_size_ConstantOp = b.create<ConstantIndexOp>(loc, group_size);
    auto wave_size_ConstantOp = b.create<ConstantIndexOp>(loc, wave_size);
    auto num_threads_blk_ConstantOp =
        b.create<ConstantIndexOp>(loc, num_threads_blk);

    // Threadwise copy from register (naive tensor) to global (generic tensor).

    int64_t M3 = num_groups_blk;
    int64_t M1 = num_input_blks;
    int64_t M2 = group_size;
    int64_t M0 = M / (M1 * M2);
    int64_t N1 = group_size;
    int64_t N0 = N / N1;
    // llvm::errs() << "M0: " << M0 << "\n";
    // llvm::errs() << "M1: num_input_blks: " << M1 << "\n";
    // llvm::errs() << "M2: group_size: " << M2 << "\n";
    // llvm::errs() << "M3: num_groups_blk: " << M3 << "\n\n";

    auto M2ConstantOp = b.create<ConstantIndexOp>(loc, M2);
    auto M2TimesM1Op = b.create<ConstantIndexOp>(loc, M2 * M1);
    auto N1ConstantOp = M2ConstantOp;

    auto laneId_xdlops_gemm = b.create<RemUIOp>(loc, tid, wave_size_ConstantOp);
    auto blk_id_xdlops_gemm =
        b.create<DivUIOp>(loc, laneId_xdlops_gemm, num_threads_blk_ConstantOp);
    auto blk_td_xdlops_gemm =
        b.create<RemUIOp>(loc, laneId_xdlops_gemm, num_threads_blk_ConstantOp);

    // emit vector swizzles
    auto gemmCVectorizedMatrixDim =
        op->getAttrOfType<IntegerAttr>("matrix_c_source_vector_read_dim");
    int64_t matrixCDataPerCopy =
        op->getAttrOfType<IntegerAttr>("matrix_c_data_per_copy").getInt();

    constexpr int64_t swizzleGroup = 4;
    // Ensure that the prerequisites are met
    // - The N dimension of the output will be stored vectorized
    // - The lowest level of splitting in registers is equal to swizzleGroup
    //    so transpose is well defined
    // - None of the larger dimensions of interest have overhangs that lead to
    //    incomplete transposes
    // - The writes will vectorize: if we're not getting vectorization
    //    due to HW % swizzleGroup != 0, then there's no point
    bool enableOutSwizzles =
        gemmCVectorizedMatrixDim.getInt() == gemmCDimN &&
        (matrixCDataPerCopy >= swizzleGroup) &&
        (M2 == swizzleGroup && (m % swizzleGroup == 0) &&
         (n % swizzleGroup == 0) && (MPerWave % swizzleGroup == 0) &&
         (NPerWave % swizzleGroup == 0));
    const auto &tailResults = blockwiseGemmV2TailOp->getResults();

    TransformMapAttr splitCTransformAttr, cVectorAccessTransformAttr;
    ArrayAttr copyBounds;

    SmallVector<Value, 4> transformedTail;
    transformedTail.reserve(tailResults.size());
    if (enableOutSwizzles) {
      // The swizzle operation doesn't fundamentally affect the mapping
      // of "expanded GEMM" (G x M0 X M1 X M2 X N) to GEMM (G X M X N)
      // space, just how we walk across it and where each thread starts.

      // However, because of the 4x4 transpose we'll be imposing
      // instead of holding N constant and walking up the M2 dimension,
      // we'll need to take 4 steps in the N dimension but hold the
      // divisible-by-4 part of the N coordinate constant. Therefore, we need to
      // break the N dimension into N0 and N1 The affine map remains otherwise
      // unchanged and becomes
      //  (d0, d1, d2, d3, d4, d5) ->
      //  (d0, d1 * M1 * M2 + d2 * M2 + d3, d4 * N1 + d5)
      TopDownCTBuilder splitCTransform(b, {"G", "M0", "M1", "M2", "N0", "N1"},
                                       {G, M0, M1, M2, N0, N1}, loc);
      splitCTransform.passThrough({"gemmG"}, {0}, {"G"});
      splitCTransform.embed("gemmM", 1, M, {"M0", "M1", "M2"},
                            {M1 * M2, M2, 1});
      splitCTransform.embed("gemmN", 2, N, {"N0", "N1"}, {N1, 1});

      splitCTransformAttr = splitCTransform.get();

      // Here is the first main effect of the swizzling transformation
      // Instead of having the fastest coordinate be the M2 dimension
      // it's now the N1 dimension, since each group of 4 values in a vector
      // corresponds to 4 successive N values after the transpose, as opposed
      // to 4 successive M values.
      // The source vector reading map is therefore
      //  (g, m0, m1, m2, n0, n1) -> (m0 * N1 + n1)
      TopDownCTBuilder cVectorAccessTransform(
          b, {"G", "M0", "M1", "M2", "N0", "N1"}, {G, M0, M1, M2, N0, N1}, loc);
      cVectorAccessTransform.embed("raw", 0, M3 * N1,
                                   {"G", "M0", "M1", "M2", "N0", "N1"},
                                   {M3 * N1, N1, N1, N1, N1, 1});
      cVectorAccessTransformAttr = cVectorAccessTransform.get();

      copyBounds = b.getIndexArrayAttr({1, M3, 1, 1, 1, N1});

      // Actually perform the swizzles
      for (Value result : tailResults) {
        auto swizzle = b.create<InWarpTransposeOp>(
            loc, result.getType(), result, laneId_xdlops_gemm,
            b.getI32IntegerAttr(group_size), b.getI32ArrayAttr({0, 1, 2, 3}));
        transformedTail.push_back(swizzle);
      }
    } else {
      // build affine expression: d0 = g
      // (d0, d1, d2, d3, d4) -> (d0, d1 * M1 * M2 + d2 * M2 + d3, d4)
      TopDownCTBuilder splitCTransform(b, {"G", "M0", "M1", "M2", "N"},
                                       {G, M0, M1, M2, N}, loc);
      splitCTransform.passThrough({"gemmG"}, {0}, {"G"});
      splitCTransform.embed("gemmM", 1, M, {"M0", "M1", "M2"},
                            {M1 * M2, M2, 1});
      splitCTransform.passThrough({"gemmN"}, {2}, {"N"});

      splitCTransformAttr = splitCTransform.get();

      // The source vector reading map is
      //  (g, m0, m1, m2, n) -> (m0 * M2 + m2)
      TopDownCTBuilder cVectorAccessTransform(b, {"G", "M0", "M1", "M2", "N"},
                                              {G, M0, M1, M2, N}, loc);
      cVectorAccessTransform.embed("raw", 0, M3 * M2,
                                   {"G", "M0", "M1", "M2", "N"},
                                   {M3 * M2, M2, M2, 1, 1});
      cVectorAccessTransformAttr = cVectorAccessTransform.get();

      copyBounds = b.getIndexArrayAttr({1, M3, 1, M2, 1});

      llvm::copy(tailResults, std::back_inserter(transformedTail));
    }
    // Slice up vectors here to make it clearer that each store loop
    // deals with a distinct set of values.
    llvm::SmallVector<Value, 4> vectors;
    vectors.reserve(transformedTail.size() * iterationsPerVectorC);
    for (Value result : transformedTail) {
      for (int64_t i = 0; i < iterationsPerVectorC; ++i) {
        Value sliceStart =
            b.createOrFold<ConstantIndexOp>(loc, vectorCoffset * i);
        Value slice =
            b.create<ExtractSliceOp>(loc, vectorCSliceType, result, sliceStart);
        vectors.push_back(slice);
      }
    }

    Value cTransformed =
        b.create<TransformOp>(loc, op.c(), splitCTransformAttr);
    // The transform for the destination memref will be copied in
    // by TransformOp lowering
    llvm::SmallVector<Attribute, 2> threadwiseCopyV2Transforms = {
        b.getArrayAttr({cVectorAccessTransformAttr}), noTransforms};
    ArrayAttr threadwiseCopyV2ArgTransform =
        b.getArrayAttr(threadwiseCopyV2Transforms);

    Value c_thread_mtx_index_row, c_thread_mtx_index_col;
    Value m_thread_data_on_global, n_thread_data_on_global;

    // emit unrolled loop.
    for (int64_t iter = 0; iter < NumBlks; ++iter) {
      // In gridwise_gemm_xdlops.hpp:
      //
      // Original C++ logic:
      // const auto c_thread_mtx_on_block =
      // blockwise_gemm.GetBeginOfThreadMatrixC(i); const index_t
      // m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;

      // compute thread_mtx_on_blk_row and thread_mtx_on_blk_col.

      // Original C++ logic.
      //
      // In xdlops_gemm.hpp:
      //
      // static constexpr bool IsABroadcast() { return NPerXdlops >= MPerXdlops;
      // }
      // __device__ static MatrixIndex GetBeginOfThreadBlk(index_t i) {
      //     const index_t xdlops_i = i / GetNumBlksPerXdlops();
      //     const index_t j        = i % GetNumBlksPerXdlops();
      //     const index_t m_i = xdlops_i / NRepeats;
      //     const index_t n_i = xdlops_i % NRepeats;
      //     const index_t laneId = get_thread_local_1d_id() %
      //     mfma_type.wave_size; const index_t blk_id = laneId /
      //     mfma_type.num_threads_blk; const index_t blk_td = laneId %
      //     mfma_type.num_threads_blk; index_t col_blk = j %
      //     mfma_type.num_output_blks; index_t row_blk = j /
      //     mfma_type.num_output_blks; static_if<!IsABroadcast>{}([&](auto) {
      //         col_blk = j / mfma_type.num_output_blks;
      //         row_blk = j % mfma_type.num_output_blks;
      //     });
      //     index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
      //     index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size
      //     + m_i * MPerXdlops; return MatrixIndex{row, col};
      // }
      //
      int64_t xdlops_i_xdlops_gemm = iter / NumBlksPerXdlops;
      int64_t j_xdlops_gemm = iter % NumBlksPerXdlops;
      int64_t m_i_xdlops_gemm = xdlops_i_xdlops_gemm / NRepeats;
      int64_t n_i_xdlops_gemm = xdlops_i_xdlops_gemm % NRepeats;

      int64_t col_blk_xdlops_gemm, row_blk_xdlops_gemm;
      bool IsABroadcast = (NPerXdlops >= MPerXdlops);
      if (IsABroadcast) {
        col_blk_xdlops_gemm = j_xdlops_gemm % num_output_blks;
        row_blk_xdlops_gemm = j_xdlops_gemm / num_output_blks;
      } else {
        col_blk_xdlops_gemm = j_xdlops_gemm / num_output_blks;
        row_blk_xdlops_gemm = j_xdlops_gemm % num_output_blks;
      }

      // Within a group of elements, a non-swizzled loop will output
      // to (ignoring OOB) [(i, j), (i + 1, j), (i + 2, j), (i + 3, j)]
      // for some starting position (i, j) that's a function of coordinates
      // that very slower.

      // The swizzles mean that each thread instead outputs to
      //  [(i, j), (i, j+1), (i, j+2), (i, j+3)]
      // Therefore, in order to ensure that values remain output to the correct
      // place we must map the starting coordinates through
      //  (i, j) -> (i / 4 * 4 + j % 4, j / 4 + 4 + i % 4)
      Value threadMtxColInBlock;
      if (enableOutSwizzles) {
        // The starting coordinate remap means that we must start
        // at (blk_td / 4) * 4, since blk_td % 4 is moved to the
        // row coordinate by the transpose and nothing replaces it
        // (the unswizzled row coordinate is always a multiple of 4
        // in cases where swizzles are enabled)
        threadMtxColInBlock =
            b.create<MulIOp>(loc,
                             b.create<arith::DivUIOp>(loc, blk_td_xdlops_gemm,
                                                      group_size_ConstantOp),
                             group_size_ConstantOp);
      } else {
        // Original C++ logic.
        //     index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
        threadMtxColInBlock = blk_td_xdlops_gemm;
      }
      int64_t thread_mtx_on_blk_col_const =
          col_blk_xdlops_gemm * n + n_i_xdlops_gemm * NPerXdlops;
      Value thread_mtx_on_blk_col = b.create<AddIOp>(
          loc, threadMtxColInBlock,
          b.create<ConstantIndexOp>(loc, thread_mtx_on_blk_col_const));

      // Original C++ logic.
      //     index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size
      //     + m_i * MPerXdlops;
      Value threadMtxRowInBlock =
          b.create<MulIOp>(loc, blk_id_xdlops_gemm, group_size_ConstantOp);
      if (enableOutSwizzles) {
        // Here, we must incorporate the mod-4 parts of blk_td
        // since while, without swizzles, these four values
        // were stored on successive threads, now they're stored
        // in four consecutive vector entries on the same thread
        threadMtxRowInBlock =
            b.create<AddIOp>(loc, threadMtxRowInBlock,
                             b.create<arith::RemUIOp>(loc, blk_td_xdlops_gemm,
                                                      group_size_ConstantOp));
      }
      int64_t thread_mtx_on_blk_row_const =
          row_blk_xdlops_gemm * m + m_i_xdlops_gemm * MPerXdlops;
      auto thread_mtx_on_blk_row = b.create<AddIOp>(
          loc, threadMtxRowInBlock,
          b.create<ConstantIndexOp>(loc, thread_mtx_on_blk_row_const));

      // compute c_thread_mtx_index_row, c_thread_mtx_index_col.
      // compute c_thread_mtx_index_row_i32, c_thread_mtx_index_col_i32.

      // In blockwise_gemm_xdlops.hpp:
      //
      // Original C++ logic:
      //  __device__ static constexpr index_t GetNumBlks()
      //      return GetNumBlksPerXdlops() * MRepeats * NRepeats;
      //
      // __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i) {
      //     const index_t waveId = get_thread_local_1d_id() / WaveSize;
      //     const index_t xdlops_i = i /
      //     XdlopsGemm.GetOutputLayout().GetNumBlks(); const index_t j        =
      //     i % XdlopsGemm.GetOutputLayout().GetNumBlks(); const index_t m =
      //     xdlops_i / NRepeats; const index_t n = xdlops_i % NRepeats; const
      //     auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(j); const
      //     index_t col =
      //         (waveId % GemmNWaves) * GemmNPerWave + n * NPerXdlops +
      //         thread_mtx_on_blk.col;
      //     const index_t row =
      //         (waveId / GemmNWaves) * GemmMPerWave + m * MPerXdlops +
      //         thread_mtx_on_blk.row;
      //     return MatrixIndex{row, col};
      // }

      int64_t xdlops_i_blockwise_gemm = iter / NumBlks;
      int64_t m_blockwise_gemm = xdlops_i_blockwise_gemm / NRepeats;
      int64_t n_blockwise_gemm = xdlops_i_blockwise_gemm % NRepeats;

      // Original C++ logic.
      // const index_t col = (waveId % GemmNWaves) * GemmNPerWave + n *
      // NPerXdlops + thread_mtx_on_blk.col;
      c_thread_mtx_index_col = b.create<AddIOp>(
          loc,
          b.create<AddIOp>(
              loc,
              b.create<MulIOp>(loc,
                               b.create<RemUIOp>(loc, waveId, NWavesConstantOp),
                               NPerWaveConstantOp),
              b.create<ConstantIndexOp>(loc, n_blockwise_gemm * NPerXdlops)),
          thread_mtx_on_blk_col);

      // Original C++ logic.
      // const index_t row = (waveId / GemmNWaves) * GemmMPerWave + m *
      // MPerXdlops + thread_mtx_on_blk.row;
      c_thread_mtx_index_row = b.create<AddIOp>(
          loc,
          b.create<AddIOp>(
              loc,
              b.create<MulIOp>(loc,
                               b.create<DivUIOp>(loc, waveId, NWavesConstantOp),
                               MPerWaveConstantOp),
              b.create<ConstantIndexOp>(loc, m_blockwise_gemm * MPerXdlops)),
          thread_mtx_on_blk_row);

      // In gridwise_gemm_xdlops.hpp:
      //
      // const auto c_thread_mtx_on_block =
      // blockwise_gemm.GetBeginOfThreadMatrixC(i); const index_t
      // m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;

      m_thread_data_on_global =
          b.create<AddIOp>(loc, m_block_data_on_global, c_thread_mtx_index_row);
      n_thread_data_on_global =
          b.create<AddIOp>(loc, n_block_data_on_global, c_thread_mtx_index_col);

      SmallVector<Value, 6> matrixCThreadwiseCopySourceCoords;
      SmallVector<Value, 6> matrixCThreadwiseCopyDestCoords;
      if (enableOutSwizzles) {
        std::fill_n(std::back_inserter(matrixCThreadwiseCopySourceCoords), 6,
                    zeroConstantOp.getResult());
        matrixCThreadwiseCopyDestCoords.append(
            {// g
             GemmBlockCoord_G,
             // m_thread_data_on_global / (M2 * M1)
             b.create<DivUIOp>(loc, m_thread_data_on_global, M2TimesM1Op),
             // m_thread_data_on_global % (M2 * M1) / M2
             b.create<DivUIOp>(
                 loc,
                 b.create<RemUIOp>(loc, m_thread_data_on_global, M2TimesM1Op),
                 M2ConstantOp),
             // m_thread_data_on_global % M2
             b.create<RemUIOp>(loc, m_thread_data_on_global, M2ConstantOp),
             // n_thread_data_on_global / N1
             b.create<DivUIOp>(loc, n_thread_data_on_global, N1ConstantOp),
             // n_thread-data_on_global % N1
             b.create<RemUIOp>(loc, n_thread_data_on_global, N1ConstantOp)});
      } else {
        std::fill_n(std::back_inserter(matrixCThreadwiseCopySourceCoords), 5,
                    zeroConstantOp.getResult());
        matrixCThreadwiseCopyDestCoords.append(
            {// g
             GemmBlockCoord_G,
             // m_thread_data_on_global / (M2 * M1)
             b.create<DivUIOp>(loc, m_thread_data_on_global, M2TimesM1Op),
             // m_thread_data_on_global % (M2 * M1) / M2
             b.create<DivUIOp>(
                 loc,
                 b.create<RemUIOp>(loc, m_thread_data_on_global, M2TimesM1Op),
                 M2ConstantOp),
             // m_thread_data_on_global % M2
             b.create<RemUIOp>(loc, m_thread_data_on_global, M2ConstantOp),
             // n_thread_data_on_global
             n_thread_data_on_global});
      }
      // Emit threadwise_copy_v2.
      auto threadwiseCopyV2CMatrixOp = b.create<ThreadwiseCopyV2Op>(
          loc, vectors[iter], cTransformed, copyBounds,
          threadwiseCopyV2ArgTransform, op.paddingInfo(),
          op.storeOperationAttr(), op.cOobDims(),
          matrixCThreadwiseCopySourceCoords, matrixCThreadwiseCopyDestCoords);
      bool canOob = llvm::any_of(op.cOobDims().getAsValueRange<BoolAttr>(),
                                 [](const bool &v) -> bool { return v; });
      affixThreadwiseCopyV2Attributes(threadwiseCopyV2CMatrixOp, op, b,
                                      enableOutSwizzles, canOob);
    }

    op.erase();

    return success();
  }
};

void LowerMIOpenOpsStep2Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmRewritePattern, GridwiseGemmV2RewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep2Pass() {
  return std::make_unique<LowerMIOpenOpsStep2Pass>();
}
