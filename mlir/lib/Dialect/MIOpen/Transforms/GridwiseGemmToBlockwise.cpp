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

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/TransformMapBuilder.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/math.h"
#include "mlir/Dialect/MIOpen/utility/transformMapUtils.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "miopen-gridwise-to-blockwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;

namespace {
struct MIOpenGridwiseGemmToBlockwisePass
    : public MIOpenGridwiseGemmToBlockwisePassBase<
          MIOpenGridwiseGemmToBlockwisePass> {
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
                              int64_t loadLength, uint32_t &vectorDim,
                              int64_t kPack, Type elementType, Type &loadType,
                              Type &intermediateType) {

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
  TopDownTMBuilder builder(b, dimNameRefs, sliceLengths, loc);
  builder.embed("iter", 0, stride, dimNameRefs, strides);
  TransformMapAttr ret = builder.get();
  return b.getArrayAttr(ret);
}

//===----------------------------------------------------------------------===//
// Building load/store loops
//===----------------------------------------------------------------------===//
TransformingForOp createGlobalLoadLoop(OpBuilder &b, Location loc, Value global,
                                       ValueRange globalStart, Type resultType,
                                       Type loadType,
                                       ArrayRef<int64_t> sliceLengths,
                                       uint32_t vectorDim, bool useIndexDiffs) {
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
  Value loaded = b.create<BufferLoadOp>(
      loc, loadType, global, leftOobDims, rightOobDims,
      loop.getLowerCoords(/*domain=*/0), /*offset=*/IntegerAttr());
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
        b.create<miopen::YieldOp>(loc, toYieldInner);
      }
      toYield = scatterLoop.getResults()[0];
    } else {
      toYield = b.create<InsertSliceOp>(loc, resultType, loaded, loopArg,
                                        loop.getLowerCoords(/*domain=*/1)[0]);
    }
  }
  b.create<miopen::YieldOp>(loc, toYield);
  return loop;
}

TransformingForOp createLdsStoreLoop(OpBuilder &b, Location loc, Value loaded,
                                     Value buffer, ValueRange bufferStart,
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
  BottomUpTMBuilder transform(b, {"buffer"}, shape, loc);
  transform.slice({"slice"}, {"buffer"}, {start}, {end});

  TransformMapAttr transformAttr = transform.get();
  Value subview = b.create<TransformOp>(loc, buffer, transformAttr,
                                        bufferType.getMemorySpaceAsInt());
  return subview;
}

struct GridwiseGemmRewritePattern : public OpRewritePattern<GridwiseGemmOp> {
  using OpRewritePattern<GridwiseGemmOp>::OpRewritePattern;

  void computeLDSBlockSizes(GridwiseGemmOp op, int64_t &a_block_space,
                            int64_t &b_block_space, int64_t &block_space,
                            int64_t KPack = 1) const {
    int64_t ThreadGemmAThreadCopySrcDataPerRead_M =
        op->getAttr("m_per_thread").template cast<IntegerAttr>().getInt();
    int64_t ThreadGemmBThreadCopySrcDataPerRead_N =
        op->getAttr("n_per_thread").template cast<IntegerAttr>().getInt();

    int64_t max_lds_align =
        math_util::lcm(ThreadGemmAThreadCopySrcDataPerRead_M,
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

    LLVM_DEBUG(llvm::dbgs() << "a_block_space: " << a_block_space << "\n");
    LLVM_DEBUG(llvm::dbgs() << "b_block_space: " << b_block_space << "\n");
    LLVM_DEBUG(llvm::dbgs() << "double_block_space: " << block_space << "\n\n");
  }

  LogicalResult matchAndRewrite(GridwiseGemmOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain data type.
    Type elementType = op.b().getType().cast<MemRefType>().getElementType();

    // Determine the type used on VGPR to act as accumulator.
    // f32: f32.
    // f16: f32 to prevent overflow from happening.
    // i16(bf16) : i16.
    // i8: i32, since we have an i32 output
    Type accumulatorType = elementType;
    if (elementType == b.getF16Type()) {
      accumulatorType = b.getF32Type();
    } else if (elementType == b.getI8Type()) {
      accumulatorType = b.getI32Type();
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
    auto kPerThreadAttr =
        b.getIndexAttr(op->getAttrOfType<IntegerAttr>("k_per_thread").getInt());
    auto mPerThreadAttr = op->getAttrOfType<IntegerAttr>("m_per_thread");
    auto nPerThreadAttr = op->getAttrOfType<IntegerAttr>("n_per_thread");
    int64_t MPerThread = mPerThreadAttr.getInt();
    int64_t NPerThread = nPerThreadAttr.getInt();
    Value MPerThreadConstantOp = b.create<ConstantIndexOp>(loc, MPerThread);
    Value NPerThreadConstantOp = b.create<ConstantIndexOp>(loc, NPerThread);

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

    bool useIndexDiffs = true;
    func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
    int64_t kernelBlockSize =
        parentFunc->getAttrOfType<IntegerAttr>("block_size").getInt();
    int64_t kernelGridSize =
        parentFunc->getAttrOfType<IntegerAttr>("grid_size").getInt();

    // Get current workgroup ID.
    auto bid = b.create<WorkgroupIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;
    int64_t GStride = MBlockWork * NBlockWork;

    LLVM_DEBUG(llvm::dbgs() << "\ngridwise_gemm op:\n");
    LLVM_DEBUG(op.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << M << "\n"
               << "N: " << N << "\n"
               << "K: " << K << "\n"
               << "BlockSize: " << BlockSize << "\n"
               << "MPerBlock: " << MPerBlock << "\n"
               << "NPerBlock: " << NPerBlock << "\n"
               << "KPerBlock: " << KPerBlock << "\n"
               << "KPack: " << KPack << "\n"
               << "MPerThread: " << MPerThread << "\n"
               << "NPerThread: " << NPerThread << "\n"
               << "MBlockWork = M / MPerBlock: " << MBlockWork << "\n"
               << "NBlockWork = N / NPerBlock: " << NBlockWork << "\n"
               << "MLevel0Cluster: " << MLevel0Cluster << "\n"
               << "MLevel1Cluster: " << MLevel1Cluster << "\n"
               << "NLevel0Cluster: " << NLevel0Cluster << "\n"
               << "NLevel1Cluster: " << NLevel1Cluster << "\n");

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

    LLVM_DEBUG(llvm::dbgs() << "matrix_a_source_data_per_read: "
                            << matrix_a_source_data_per_read << "\n");
    LLVM_DEBUG(llvm::dbgs() << "matrix_b_source_data_per_read: "
                            << matrix_b_source_data_per_read << "\n");

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

    // Each thread should not read exceed the length of the corresponding tile
    if (GemmABlockCopyThreadSliceLengths_GemmK > KPerBlock ||
        GemmABlockCopyThreadSliceLengths_GemmM > MPerBlock) {
      return failure();
    }

    if (GemmABlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmM == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmKPack == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Blockwise copy slice lengths for matrix A is zero which "
                    "is invalid.\n");
      return failure();
    }

    // Compute ThreadClusterLengths for Matrix A.
    uint64_t GemmABlockCopyClusterLengths_GemmKPack =
        KPack / GemmABlockCopyThreadSliceLengths_GemmKPack;
    uint64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock / GemmABlockCopyThreadSliceLengths_GemmK;
    // int64_t GemmABlockCopyClusterLengths_GemmM =
    //    MPerBlock / GemmABlockCopyThreadSliceLengths_GemmM;

    LLVM_DEBUG(llvm::dbgs() << "thread cluster lengths for Matrix A: "
                            << GemmABlockCopyClusterLengths_GemmK << " "
                            << GemmABlockCopyClusterLengths_GemmKPack << "\n");

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
      LLVM_DEBUG(llvm::dbgs()
                 << "Vector loads/stores aren't possible in the G dimension "
                 << "and should not haven been attempted\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "thread slice lengths for Matrix B: "
               << GemmBBlockCopyThreadSliceLengths_GemmK << " "
               << GemmBBlockCopyThreadSliceLengths_GemmN << " "
               << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n");
    // Each thread should not read exceed the length of the corresponding tile
    if (GemmBBlockCopyThreadSliceLengths_GemmK > KPerBlock ||
        GemmBBlockCopyThreadSliceLengths_GemmN > NPerBlock) {
      return failure();
    }

    if (GemmBBlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmN == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmKPack == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Blockwise copy slice lengths for matrix B is zero which "
                 << "is invalid.\n");
      return failure();
    }

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

    LLVM_DEBUG(llvm::dbgs() << "LDS block size:" << ldsBlockASize << " "
                            << ldsBlockBSize << " " << ldsBlockSize << "\n");

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
    Value ldsMatrixASubviewOp =
        reshapeBuffer(b, loc, ldsBlockASubviewOp, {"k", "m", "kpack"},
                      {KPerBlock, MPerBlock, KPack});
    // TODO: Remove this when kPack branches are unified here
    Value ldsMatrixASubviewForCopy = ldsMatrixASubviewOp;
    if (KPack == 1) {
      ldsMatrixASubviewForCopy = reshapeBuffer(
          b, loc, ldsBlockASubviewOp, {"k", "m"}, {KPerBlock, MPerBlock});
    }

    // Subviews for Matrix B.
    auto ldsBlockBOffset = ldsBlockASize;
    auto ldsBlockBSubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockBOffset, ldsBlockBSize);

    // Get matrix subviews.
    // Compute matrix B dimension from attributes.
    Value ldsMatrixBSubviewOp =
        reshapeBuffer(b, loc, ldsBlockBSubviewOp, {"k", "n", "kpack"},
                      {KPerBlock, NPerBlock, KPack});
    Value ldsMatrixBSubviewOpForCopy = ldsMatrixBSubviewOp;
    if (KPack == 1) {
      ldsMatrixBSubviewOpForCopy = reshapeBuffer(
          b, loc, ldsBlockBSubviewOp, {"k", "n"}, {KPerBlock, NPerBlock});
    }

    // Alloc for Matrix C on registers.
    // Compute register size from attributes.
    int64_t GemmMRepeat = 0, GemmNRepeat = 0;

    GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
    GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

    LLVM_DEBUG(llvm::dbgs() << "GemmMRepeat: " << GemmMRepeat << "\n");
    LLVM_DEBUG(llvm::dbgs() << "GemmNRepeat: " << GemmNRepeat << "\n");

    int64_t threadCNumM = GemmMRepeat * MPerThread;
    int64_t threadCNumN = GemmNRepeat * NPerThread;
    int64_t threadCNumRegisters = threadCNumM * threadCNumN;
    auto threadCRegisterMemRefType =
        MemRefType::get({threadCNumRegisters}, accumulatorType, {},
                        gpu::GPUDialect::getPrivateAddressSpace());
    Value registerMatrixCAllocOp =
        b.create<GpuAllocOp>(loc, threadCRegisterMemRefType);
    Value registerMatrixCViewOp = reshapeBuffer(
        b, loc, registerMatrixCAllocOp, {"m", "n"}, {threadCNumM, threadCNumN});

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
    Type aLoadIntermediate, aLoadType;
    computeLoadStoreTypeInfo(b, blockwiseCopyABounds, blockwiseLoadVectorLenA,
                             blockwiseVectorDimA, KPack, elementType, aLoadType,
                             aLoadIntermediate);
    LLVM_DEBUG(llvm::dbgs()
               << "Corrected blockwise vector dim A: " << blockwiseVectorDimA
               << "\n"
               << "Load type A: " << aLoadType << "\n"
               << "Intermediate type A: " << aLoadIntermediate << "\n");

    LLVM_DEBUG(llvm::dbgs() << "blockwise copy A bounds: ");
    for (auto v : blockwiseCopyABounds)
      LLVM_DEBUG(llvm::dbgs() << v << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n");

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
    Type bLoadIntermediate, bLoadType;
    computeLoadStoreTypeInfo(b, blockwiseCopyBBounds, blockwiseLoadVectorLenB,
                             blockwiseVectorDimB, KPack, elementType, bLoadType,
                             bLoadIntermediate);
    LLVM_DEBUG(llvm::dbgs()
               << "Corrected blockwise vector dim B: " << blockwiseVectorDimB
               << "\n"
               << "Load type B: " << bLoadType << "\n"
               << "Intermediate type B: " << bLoadIntermediate << "\n");

    LLVM_DEBUG(llvm::dbgs() << "blockwise copy B bounds: ");
    for (auto v : blockwiseCopyBBounds)
      LLVM_DEBUG(llvm::dbgs() << v << " ");
    LLVM_DEBUG(llvm::dbgs() << "\n");

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
    int64_t mRepeatLDSStride = MPerLevel0Cluster * MLevel1Cluster;
    int64_t nRepeatLDSStride = NPerLevel0Cluster * NLevel1Cluster;
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
    TransformingForOp blockwiseLoadA = createGlobalLoadLoop(
        b, loc, op.a(), blockwiseLoadACoords, aLoadIntermediate, aLoadType,
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
        b, loc, op.b(), blockwiseLoadBCoords, bLoadIntermediate, bLoadType,
        blockwiseCopyBBounds, blockwiseVectorDimB, useIndexDiffs);

    SmallVector<Value, 4> blockwiseStoreACoords;
    if (KPack > 1) {
      blockwiseStoreACoords = {GemmABlockCopyDestCoord_Z,
                               GemmABlockCopyDestCoord_Y,
                               GemmABlockCopyDestCoord_X};
    } else {
      blockwiseStoreACoords = {GemmABlockCopyDestCoord_Y,
                               GemmABlockCopyDestCoord_X};
    }
    // Emit blockwise store for matrix A.
    // Note: first bound (g dimension) dropped because blockwise store doesn't
    // have it
    TransformingForOp blockwiseStoreA = createLdsStoreLoop(
        b, loc, blockwiseLoadA.getResult(0), ldsMatrixASubviewForCopy,
        blockwiseStoreACoords,
        ArrayRef<int64_t>(blockwiseCopyABounds).drop_front(1));

    SmallVector<Value, 4> blockwiseStoreBCoords;
    if (KPack > 1) {
      blockwiseStoreBCoords = {GemmBBlockCopyDestCoord_Z,
                               GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    } else {
      blockwiseStoreBCoords = {GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    }
    // Emit blockwise store for matrix B.
    TransformingForOp blockwiseStoreB = createLdsStoreLoop(
        b, loc, blockwiseLoadB.getResult(0), ldsMatrixBSubviewOpForCopy,
        blockwiseStoreBCoords,
        ArrayRef<int64_t>(blockwiseCopyBBounds).drop_front(1));

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
        loc, ldsMatrixASubviewOp, ldsMatrixBSubviewOp, registerMatrixCViewOp,
        mMyThreadOffsetA, mMyThreadOffsetB, kPerThreadAttr,
        b.getIndexAttr(MPerThread), b.getIndexAttr(NPerThread),
        b.getIndexAttr(mRepeatLDSStride), b.getIndexAttr(nRepeatLDSStride));

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
    BlockAndValueMapping tailGemmCloneMap;
    b.clone(*blockwiseGemmOp, tailGemmCloneMap);

    // Threadwise copy from register (naive tensor) to global (generic tensor).
    TopDownTMBuilder splitMemoryCoords(
        b, {"bid", "tid", "iter"},
        {kernelGridSize, kernelBlockSize, threadCNumRegisters}, loc);
    splitMemoryCoords.merge(
        {"g", "m_block", "n_block"}, {0, 1, 2}, "bid",
        {kernelGridSize / GStride, GStride / NBlockWork, NBlockWork});
    splitMemoryCoords.merge(
        {"level1", "level0"}, {3, 4}, "tid",
        {kernelBlockSize / ThreadPerLevel0Cluster, ThreadPerLevel0Cluster});
    splitMemoryCoords.merge({"m_iter", "n_iter"}, {5, 6}, "iter",
                            {threadCNumM, threadCNumN});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

    auto toClusters =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    llvm::StringMap<uint32_t> toClustersIdxs = expandNamesInPlace(
        splitMemoryCoords, {{"level1", {"level1_m", "level1_n"}},
                            {"level0", {"level0_m", "level0_n"}},
                            {"m_iter", {"m_iter_level1", "m_iter_level0"}},
                            {"n_iter", {"n_iter_level1", "n_iter_level0"}}});
    TopDownTMBottomDimsWrapper clustersWrap(toClusters, toClustersIdxs);
    clustersWrap.passThrough({"g", "m_block", "n_block"});
    clustersWrap.merge(
        {"level1_m", "level1_n"}, "level1",
        {splitMemoryCoords.endSize("level1") / NLevel1Cluster, NLevel1Cluster});
    clustersWrap.merge(
        {"level0_m", "level0_n"}, "level0",
        {splitMemoryCoords.endSize("level0") / NLevel0Cluster, NLevel0Cluster});
    clustersWrap.merge({"m_iter_level1", "m_iter_level0"}, "m_iter",
                       {GemmMRepeat, MPerThread});
    clustersWrap.merge({"n_iter_level1", "n_iter_level0"}, "n_iter",
                       {GemmNRepeat, NPerThread});
    TransformMapAttr toClustersAttr = toClusters.get();

    auto toMatrixC = TopDownTMBuilder::below(toClusters, toClustersAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {"g"});
    toMatrixC.embed(
        "gemmM", 1, M,
        {"m_block", "level1_m", "level0_m", "m_iter_level1", "m_iter_level0"},
        {MPerBlock, MPerLevel0Cluster, MPerThread, mRepeatLDSStride, 1});
    toMatrixC.embed(
        "gemmN", 2, N,
        {"n_block", "level1_n", "level0_n", "n_iter_level1", "n_iter_level0"},
        {NPerBlock, NPerLevel0Cluster, NPerThread, nRepeatLDSStride, 1});
    TransformMapAttr toTensorCAttr = toMatrixC.get();

    TopDownTMBuilder toRegisterC(
        b, {"bid", "tid", "iter"},
        {kernelGridSize, kernelBlockSize, threadCNumRegisters}, loc);
    toRegisterC.ignore("bid");
    toRegisterC.ignore("tid");
    toRegisterC.passThrough({"iter"}, {0}, {"iter"});
    TransformMapAttr toRegisterCAttr = toRegisterC.get();

    Value registerC = registerMatrixCAllocOp;
    // If we need to type-convert the accumulator (currently this is only
    // fp32->f16) then we must do so before the writeback loop in which fusion
    // takes places at this time, since the fusion pass as currently written
    // can't interceps the type conversions.
    Type destType = op.c().getType().cast<MemRefType>().getElementType();
    if (destType != accumulatorType) {
      auto convertedCType =
          threadCRegisterMemRefType.clone(destType).cast<MemRefType>();
      Value convertedC = b.create<miopen::GpuAllocOp>(loc, convertedCType);
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
        b.getArrayAttr({splitMemoryCoordsAttr, toClustersAttr, toTensorCAttr});
    Value tensorC;
    ArrayAttr idToTensorCMaps;
    std::tie(tensorC, idToTensorCMaps) =
        untransform(b, op.c(), idToMatrixCMaps);
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

  void computeLDSBlockSizes(GridwiseGemmV2Op op, int64_t &a_block_space,
                            int64_t &b_block_space, int64_t &total_block_space,
                            int64_t KPack = 1) const {
    int64_t max_lds_align = 1;

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

    constexpr int64_t waveSize = 64;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, waveSize);

    bool useIndexDiffs = true;

    func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
    int64_t kernelBlockSize =
        parentFunc->getAttrOfType<IntegerAttr>("block_size").getInt();
    int64_t kernelGridSize =
        parentFunc->getAttrOfType<IntegerAttr>("grid_size").getInt();

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
        MPerBlock * KPerBlock * KPack / BlockSize;

    LLVM_DEBUG(llvm::dbgs() << "GemmABlockCopyNumberDataPerThread: "
                            << GemmABlockCopyNumberDataPerThread << "\n");

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
        NPerBlock * KPerBlock * KPack / BlockSize;

    LLVM_DEBUG(llvm::dbgs() << "GemmBBlockCopyNumberDataPerThread: "
                            << GemmBBlockCopyNumberDataPerThread << "\n");

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
    computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize, ldsBlockSize, KPack);

    LLVM_DEBUG(llvm::dbgs() << "LDS block size:" << ldsBlockASize << " "
                            << ldsBlockBSize << " " << ldsBlockSize << "\n");

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

    uint32_t blockwiseVectorDimA = matrix_a_source_vector_read_dim;
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

    uint32_t blockwiseVectorDimB = matrix_b_source_vector_read_dim;
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
        b, loc, op.a(), blockwiseLoadACoords, aLoadIntermediate, aLoadType,
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
        b, loc, op.b(), blockwiseLoadBCoords, bLoadIntermediate, bLoadType,
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
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t m = xcs.m;
    int64_t n = xcs.n;
    int64_t k_base = xcs.k_base;

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
      // Should pack at least k_base elements and avoid waste xdlopsgemm
      // cycles
      if (KPack < k_base) {
        return failure();
      }

      // When reduction, KPerBlock must be at least num_input_blks
      if (IsKReduction && KPerBlock < num_input_blks) {
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
      if (IsKReduction && KPerBlock < k_base * num_input_blks) {
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
    int64_t numBlksPerXdlops = (MPerXdlops * NPerXdlops) / (m * n);
    const auto &tailResults = blockwiseGemmV2TailOp->getResults();
    int64_t wavesInKernelBlock = kernelBlockSize / waveSize;
    int64_t resultCVectorLen = vectorType.getNumElements();
    int64_t numElements = resultCVectorLen * tailResults.size();

    TopDownTMBuilder splitMemoryCoords(
        b, {"bid", "tid", "item"},
        {kernelGridSize, kernelBlockSize, numElements}, loc);
    splitMemoryCoords.merge(
        {"g", "n", "m"}, {0, 1, 2}, {"bid"},
        {kernelGridSize / GStride, GStride / MBlockWork, MBlockWork});
    splitMemoryCoords.merge({"wave", "block", "tid_group", "tid_item"},
                            {3, 4, 5, 6}, "tid",
                            {wavesInKernelBlock, waveSize / num_threads_blk,
                             num_threads_blk / group_size, group_size});
    splitMemoryCoords.merge(
        {"i", "j", "vec_group", "vec_item"}, {7, 8, 9, 10}, "item",
        {numElements / (numBlksPerXdlops * num_groups_blk * group_size),
         numBlksPerXdlops, num_groups_blk, group_size});
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
    rowsAndColsWrap.passThrough({"block", "tid_group", "tid_item"});
    rowsAndColsWrap.merge(
        {"m_i", "n_i"}, "i",
        {splitMemoryCoords.endSize("i") / NRepeats, NRepeats});

    // Here we use the full builder API since we want index and name control
    bool isABroadcast = (NPerXdlops >= MPerXdlops);
    SmallVector<StringRef, 2> rowsFirst = {"blk_row", "blk_col"};
    SmallVector<StringRef, 2> colsFirst = {"blk_col", "blk_row"};
    toRowsAndCols.merge(
        isABroadcast ? rowsFirst : colsFirst,
        {rowsAndColsIdxs["blkMajor"], rowsAndColsIdxs["blkMinor"]}, "j",
        {splitMemoryCoords.endSize("j") / num_output_blks, num_output_blks});
    toRowsAndCols.passThrough(
        {"vec_group", "vec_item"},
        {rowsAndColsIdxs["vec_group"], rowsAndColsIdxs["vec_item"]},
        {"vec_group", "vec_item"});

    TransformMapAttr toRowsAndColsAttr = toRowsAndCols.get();

    auto toMatrixC = TopDownTMBuilder::below(toRowsAndCols, toRowsAndColsAttr);
    toMatrixC.passThrough({"gemmG"}, {0}, {"g"});

    toMatrixC.embed(
        "gemmM", 1, M,
        {"m", "wave_m", "block", "m_i", "blk_row", "vec_group", "vec_item"},
        {MPerBlock, MPerWave, group_size, MPerXdlops, m,
         num_input_blks * group_size, 1});
    toMatrixC.embed("gemmN", 2, N,
                    {"n", "wave_n", "tid_group", "n_i", "blk_col", "tid_item"},
                    {NPerBlock, NPerWave, group_size, NPerXdlops, n, 1});
    TransformMapAttr toMatrixCAttr = toMatrixC.get();

    ArrayAttr idToMatrixCMaps = b.getArrayAttr(
        {splitMemoryCoordsAttr, toRowsAndColsAttr, toMatrixCAttr});
    Value tensorC;
    ArrayAttr idToTensorCMaps;
    std::tie(tensorC, idToTensorCMaps) =
        untransform(b, op.c(), idToMatrixCMaps);

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
      auto indexSplit =
          TopDownTMBuilder(b, {"bid", "tid", "iter"},
                           {kernelGridSize, kernelBlockSize, numElements}, loc);
      indexSplit.passThrough("bid");
      indexSplit.merge({"tid_group", "tid_item"}, {1, 2}, "tid",
                       {kernelBlockSize / 4, 4});
      indexSplit.merge({"vec_group", "vec_item"}, {3, 4}, "iter",
                       {numElements / 4, 4});
      TransformMapAttr indexSplitAttr = indexSplit.get();

      // Note that we switch the positions of tid_item and vec_item when
      // recombining the coordinates.
      auto indexCombine = TopDownTMBuilder::below(indexSplit, indexSplitAttr);
      indexCombine.passThrough("bid");
      indexCombine.embed("tid", 1, kernelBlockSize, {"tid_group", "vec_item"},
                         {4, 1});
      indexCombine.embed("iter", 2, numElements, {"vec_group", "tid_item"},
                         {4, 1});
      TransformMapAttr indexCombineAttr = indexCombine.get();

      SmallVector<Attribute, 8> newTransforms = {indexSplitAttr,
                                                 indexCombineAttr};
      llvm::copy(idToTensorCMaps, std::back_inserter(newTransforms));
      idToTensorCMaps = b.getArrayAttr(newTransforms);
    }

    // Legacy vectorization usage
    int64_t gemmCVectorizedMatrixDim =
        op->getAttrOfType<IntegerAttr>("matrix_c_source_vector_read_dim")
            .getInt();
    int64_t matrixCDataPerCopy =
        op->getAttrOfType<IntegerAttr>("matrix_c_data_per_copy").getInt();

    // Determine if we need to exclude the specified vectorization
    ArrayAttr cLeftOobCheck, cRightOobCheck;
    ArrayAttr cTransforms = std::get<1>(untransform(b, op.c()));
    std::tie(cLeftOobCheck, cRightOobCheck) =
        computeOobFromTransforms(b, cTransforms);
    bool oldCanOutOob = cLeftOobCheck.size() > 0 || cRightOobCheck.size() > 0;

    // Ensure that the prerequisites are met
    // - The N dimension of the output will be stored vectorized
    // - The lowest level of splitting in registers is equal to swizzleGroup
    //    so transpose is well defined
    // - None of the larger dimensions of interest have overhangs that lead to
    //    incomplete transposes
    // - The writes will vectorize: if we're not getting vectorization
    //    due to HW % swizzleGroup != 0, then there's no point
    bool oldEnableOutSwizzles =
        gemmCVectorizedMatrixDim == gemmCDimN &&
        (matrixCDataPerCopy >= swizzleGroup) &&
        (group_size == swizzleGroup && (m % swizzleGroup == 0) &&
         (n % swizzleGroup == 0) && (MPerWave % swizzleGroup == 0) &&
         (NPerWave % swizzleGroup == 0));

    if (oldCanOutOob ||
        (gemmCVectorizedMatrixDim == gemmCDimN && !enableOutSwizzles)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Disabling vectorization of output write. Output oob checks = "
          << oldCanOutOob << "\n");
      matrixCDataPerCopy = 1;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Enable swizzles (new): " << enableOutSwizzles << "\n"
               << "Enable swizzles (old): " << oldEnableOutSwizzles << "\n"
               << "Data per copy (new): " << tensorCDataPerCopy << "\n"
               << "Data per copy (old): " << matrixCDataPerCopy << "\n");
    if ((matrixCDataPerCopy > 1) && (enableOutSwizzles != oldEnableOutSwizzles))
      return op.emitOpError("New vectorizer swizzle enable " +
                            Twine(enableOutSwizzles) +
                            " disagrees with old etting " +
                            Twine(oldEnableOutSwizzles) + " and it matters");
    if (tensorCDataPerCopy < matrixCDataPerCopy)
      return op.emitOpError(
          "New vectorizer calls for " + Twine(tensorCDataPerCopy) +
          " elements but the old one wants " + Twine(matrixCDataPerCopy));

    // Make the vector slice starting point jump in units of the vectorization.
    TopDownTMBuilder correctVectorCoords(
        b, {"bid", "tid", "item"},
        {kernelGridSize, kernelBlockSize, numElements}, loc);
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
      for (Value result : tailResults) {
        Value swizzle = b.create<InWarpTransposeOp>(
            loc, result.getType(), result, laneId,
            b.getI32IntegerAttr(group_size), b.getI32ArrayAttr({0, 1, 2, 3}));
        transformedTail.push_back(swizzle);
      }
    } else {
      llvm::copy(tailResults, std::back_inserter(transformedTail));
    }

    // Convert GEMM results to the expected output type (so we can fuse in)
    // operations expecting that type before writeback and store
    // the result vectors into a allocation of registers to maintain uniformity
    // with the non-xdlops gemm. (These "stores" will be optimized out)
    Type destType = op.c().getType().cast<MemRefType>().getElementType();
    MemRefType mergedType = MemRefType::get(
        numElements, destType, {},
        /*memorySpace=*/gpu::GPUDialect::getPrivateAddressSpace());
    VectorType castVectorType = vectorType.clone(destType);
    Value resultMerged = b.create<miopen::GpuAllocOp>(loc, mergedType);
    for (const auto &pair : llvm::enumerate(transformedTail)) {
      Value cast = createTypeConversionOp(b, loc, pair.value(), castVectorType);
      Value offset = b.createOrFold<arith::ConstantIndexOp>(
          loc, pair.index() * resultCVectorLen);
      b.create<miopen::InBoundsStoreOp>(loc, cast, resultMerged, offset);
    }
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
          loc, resultMerged, tensorC, b.getIndexAttr(tensorCDataPerCopy),
          op.storeMethodAttr(), std::get<0>(writeOobDims),
          std::get<1>(writeOobDims), outLoop.getLowerCoords(/*domain=*/0)[0],
          outLoop.getLowerCoords(/*domain=*/1));
    }

    b.eraseOp(op);
    return success();
  }
};

void MIOpenGridwiseGemmToBlockwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<miopen::GridwiseGemmOp, miopen::GridwiseGemmV2Op>();
  target.addLegalDialect<arith::ArithmeticDialect, miopen::MIOpenDialect,
                         AffineDialect, vector::VectorDialect>();

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
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createMIOpenGridwiseGemmToBlockwisePass() {
  return std::make_unique<MIOpenGridwiseGemmToBlockwisePass>();
}
