//===- loweringUtils.h - functions that often come up during lowering or turing
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ROCK_UTILITY_LOWERINGUTILS_H
#define ROCK_UTILITY_LOWERINGUTILS_H

#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

namespace mlir {
class Operation;
class Type;

namespace gpu {
enum class AddressSpace : uint32_t;
}

namespace rock {
class ThreadwiseReadIntoOp;
struct ConvolutionDims;
struct GemmSize;

// This structure captures three views of
// a register memref. Each view correspond
// to a (strided) slice of a 2D matrix that is
// loaded into the register memref.
struct RegsAsMatrixSubTiles {
  // This is a [gridIdx0, ... , gridIdxN, tid, iter] to a 2D subtile view.
  // Using all grid idxs, tid and iterative idx, this provides access to
  // gridwise sub-tile of the matrix.
  ArrayAttr gridSubTile;
  // This is a [tid, iter] to a 2D subtile view.
  // Using just tid and iterative idx, this provides access to blockwise
  // sub-tile of the matrix.
  ArrayAttr blockSubTile;
  // This is a [iter] to to a 2D subtile view.
  // Using just a iterative dix, this provides access to threadwise sub-tile
  // of the matrix.
  ArrayAttr threadSubTile;
  // This is a [tid] to a 2D subtile view.
  // i.e. [tid] --> [m_tid, n_tid]
  // where |m_tid| x |n_tid| == workgroup size.
  // It is equivalent to removing all iter-dependent components from
  // blockSubTile.
  std::optional<ArrayAttr> blockSubTileTidSlice;
};

// Following structures holds knobs to tweak the
// the LDS layout for gemms/attention ops.
struct LDSLayoutConfigDim {
  bool doRotateWithK;
  bool doSwapThreadIterSubDims;
  bool ldsLayoutDxK;
};

// This is helper struct to aggregate
// derived information w.r.t load vectorization
struct VectorDimInfo {
  GemmDimension vectorDim;
  int64_t vectorLen;
  int64_t inKPerThread;
  int64_t inDPerThread;
  GemmDimension vectorTiebreaker;
};

struct ThreadwiseReadIntoLoopConfigInput {
  Value sourceView;
  MemRefType dstBufferType;
  size_t extraIdxCount;
  Type elementType;
  int64_t numValues;
  bool isSrcVectorBuffer;
  bool isDstVectorBuffer;
  bool hasDynamicValidities;
  bool isGlobalToLDS;
  std::optional<int64_t> maxGlobalToLDSVectorLen;
};

struct ThreadwiseReadIntoLoopInfo {
  int64_t numValues;
  int64_t srcStride;
  int64_t vectorSrcLen;
  int64_t vectorDstLen;
  Type elementType;
  Type loadType;
  VectorType dstVectorType;
};

// The rows and columns of subtile view needs to
// be transposed depending on which operand of
// gemm the view is going to be.
RegsAsMatrixSubTiles transposeSubTileViews(PatternRewriter &rewriter,
                                           Location loc,
                                           RegsAsMatrixSubTiles subTileViews);

// This function will create views of the register buffer of the loaded tile
// of a matrix in global memory. Those views will provide sub-tiles of the
// respective hierarchy within the GPU. See above about RegsAsMatrixSubTiles
FailureOr<RegsAsMatrixSubTiles> getLoadRegsAsTileViews(
    OpBuilder &b, Location loc, Value globalBuffer, StringRef dName,
    ArrayRef<StringRef> bidGridOrder, ArrayRef<int64_t> bidGridLengths,
    int64_t blockSize, int64_t kPerBlock, int64_t dPerBlock, int64_t kPerThread,
    int64_t dPerThread, bool isKContiguousDim, bool directToLDS);

// This function will create views of the register buffer of the loaded tile
// but packed as kOuterPerThread, dPerThread and kPackPerThread for max
// vectorization of LDS storing. Those views will provide sub-tiles of the
// respective hierarchy within the GPU. See above about RegsAsMatrixSubTiles
FailureOr<RegsAsMatrixSubTiles> getPackedRegsAsTileViews(
    OpBuilder &b, Location loc, Value globalBuffer, StringRef dName,
    ArrayRef<StringRef> bidGridOrder, ArrayRef<int64_t> bidGridLengths,
    int64_t blockSize, int64_t kPerBlock, int64_t dPerBlock, int64_t kPerThread,
    int64_t dPerThread, int64_t kpack, bool isKContiguousDim,
    bool doSwapThreadIterSubDimsForD = false);

bool isWrWAtomicKernel(GemmFeatures features, Type dataType,
                       bool requiredPadding);

// Returns true if the provided memory space attribute encodes GPU workgroup
// memory. Returns failure if memorySpace is null (unspecified).
FailureOr<bool> isWorkgroupMemorySpace(Attribute memorySpace);

// Return true if this shaped type will occupy more than 4 GB (2 ^ 32 bytes)
// in memory.
bool is4GBMemoryType(ShapedType type);

// Return true if the Block size is valid
bool isValidBlockSize(int64_t blockSize, int64_t kPerBlock, int64_t mPerBlock,
                      int64_t nPerBlock);

// Heuristic logic to compute KBlock for backward weight atomic add kernel.
// The logic is adopted from MIOpen.
//
// The logic searches within the range of [1, 20 * number of CUs / gridSize],
// where gridSize is the original number of workgroups required for the
// convolution, and find the largest KBlock number which preserves the 2
// contraints:
// - GemmK (before splitting) = KBlock * KPerBlock * KPack * GemmK (after
// splitting).
// - n (batch size) is divisible by KBlock.
//
// 20 is a magic number obtained in MIOpen after empirical testing. It offers a
// reasonable reduction of GemmK after splitting, without incurring too much
// overheads on atomic adds. One potential future work is to make this value be
// tunable.
LogicalResult calculateKBlockNum(const int64_t batchSize,
                                 const GemmSize &gemmSize, int64_t MPerBlock,
                                 int64_t NPerBlock, int64_t KPerBlock,
                                 int64_t KPack, int64_t num_cu,
                                 int64_t &nKBlock);

// Heuristic to determine if every element in the output would be written by the
// backward data convolution algorithm.
bool isEveryElementWrittenBwdData(ArrayRef<int64_t> strideDims,
                                  ArrayRef<int64_t> dilationDims,
                                  ArrayRef<int64_t> filterDims);

/// Populate a vector of kernel IDs to be used by a backward data convolution
/// algorithm. In the current v4r1 algorithm, several kernels may be needed to
/// realize a complete backward data convolution.
///
/// A kernel ID denotes an actual implicit GEMM kernels to
/// partipate the backward data convolution.
SmallVector<int64_t> backwardDataKernelIds(ArrayRef<int64_t> strideDims,
                                           ArrayRef<int64_t> dilationDims,
                                           ArrayRef<int64_t> filterDims,
                                           bool usesV4R1);

/// Return a vector type of length `len` if `len` is more than 1, otherwise,
/// return `type`.
Type vectorTypeOrSelf(Type elementType, int64_t len);

/// Apply padding to a matrix in its `firstDim` and `secondDim` if applicable.
Value padMatrix(Value matrix, OpBuilder &b, Location loc, StringRef firstDim,
                int64_t firstDimPad, StringRef secondDim, int64_t secondDimPad);

// Apply padding to a vector in its `firstDim` if applicable.
Value padVector(Value vector, OpBuilder &b, Location loc, StringRef firstDim,
                int64_t firstDimPad);

/// Normalize the argument into the form requested.
/// If a group dimension is not present, add one.
/// If doTranspose is true, meaning the user's transpose requests don't match
/// what the underlying gridwise gemm expects, transpose the matrix to match,
/// using firstDim as the name of the first dimension in the new value and
/// secondDim as the name of the second dimesion.
Value normalizeMatrix(Value matrix, OpBuilder &b, Location loc,
                      bool doTranspose, StringRef firstDim,
                      StringRef secondDim);
// if K is not the contiguous dimension, we swapped (on each axis) the thread id
// and the iter id dimensions, so that the threads write in a contiguous fashion
// minimizing LDS bank conflicts.  This transformation swap those dimensions
// back before producing the final output view
FailureOr<TopDownTMBuilder>
swapThreadIdAndIteration(TopDownTMBuilder &toMatrixC, int64_t mBlocks,
                         int64_t nBlocks, int64_t copyMPerThread,
                         int64_t copyNPerThread, int64_t mPerBlock,
                         int64_t nPerBlock, bool doSwapThreadIterSubDimsForM,
                         bool doSwapThreadIterSubDimsForN, bool isBlockwise,
                         SmallVector<Attribute> &transformAttrs);

// This is a helper function to create a subview of slice of the first dimension
Value createSliceOfFirstDim(PatternRewriter &rewriter, Location loc,
                            Value buffer, Value sliceIdx);

// Given a `value` traverses its "views" until it finds the real
// `rock::GpuAllocOp` or fails.
FailureOr<rock::GpuAllocOp> findGpuAlloc(Value value);

// Given a `value` traverses its "views" until it finds the real
// `memref::AllocOp` or fails.
FailureOr<memref::AllocOp> findMemrefAlloc(Value value);

/// Trace back a value to find all GpuAllocOps it originates from.
/// Handles views, extract_multibuffer, and transform operations.
/// Returns all allocs that could be the source (for extract_multibuffer with
/// multiple buffers).
SmallVector<rock::GpuAllocOp> findAllGpuAllocs(Value value);

// Get gridSize
FailureOr<IntegerAttr> getGridSize(Operation *op);

// Get blockSize
FailureOr<IntegerAttr> getBlockSize(Operation *op);

// helper to create ReassociationIndices for flattening
ReassociationIndices getReassociationForFlattening(ShapedType srcTp);

// helper to obtain a flattened memref
Value getFlattenedMemref(OpBuilder &b, Value nonFlatMemRef);

/// Construct a `memref.view` operation that interprets the buffer `buffer`,
/// whose elements are bytes, as a buffer of `type`.
TypedValue<MemRefType> viewBufferAs(OpBuilder &b, Value buffer,
                                    Type elementType);

/// Same as above but the user provides output dimensions.
TypedValue<MemRefType> viewBufferAs(OpBuilder &b, Value buffer,
                                    Type elementType,
                                    ArrayRef<int64_t> dimensions);

// helper to allocate memory on the GPU
Value gpuAlloc(OpBuilder &b, Location loc, int64_t bufferDim, Type elementType,
               gpu::AddressSpace memoryAddressSpace);

// helper to verify a lds allocation fits in the GPU
LogicalResult checkLDSSize(StringAttr arch, int64_t ldsBytes);

// Trace gemm output back to its function arguments
FailureOr<SmallVector<BlockArgument>>
traceGemmOutputToArgs(Value matC, func::FuncOp func,
                      const BufferDependencyAnalysis &deps);

// Trace value to a block argument, going through view-like operations
FailureOr<BlockArgument> findBlockArgument(Value value);

// Trace gemm output to all linalg.generic that happen after it (output fusions)
FailureOr<SmallVector<OpOperand *>>
traceGemmOutputToGenericOps(Value matC, func::FuncOp func,
                            const BufferDependencyAnalysis &deps);

/// Wraps the LDS buffer "buffer", which is <kOuter * d * kpack *
/// sizeof(T) x i8> into a tid x iter view, where `iter` iterates over nominal
/// scalar indices into a buffer of type T. `buffer` will be reinterpreted as a
/// buffer with element type vector<kpackPerThread x T> (with kpackPerThread ==
/// 1 meaning just T). The resulting view must be iterated over with a stride of
/// no less than min(kPerThread, kpack). Also note that the `d` dimension
/// might be rotated to minimize bank conflicts (i.e., depending on
/// `rotateDWithK`
// we can apply a transformation similar to `d=(d+kOuter)%D`)
FailureOr<Value> wrapLDSBufferForStore(OpBuilder &b, Location loc, Value buffer,
                                       Type ldsReadType, int64_t kOuter,
                                       StringRef dName, int64_t d,
                                       int64_t kPerThread, int64_t dPerThread,
                                       bool rotateDWithK = false);

FailureOr<VectorDimInfo> getVectorDim(Location loc, Value matrix, Type elemType,
                                      int64_t blockSize, int64_t kPerBlock,
                                      int64_t dPerBlock, int64_t kpack,
                                      bool directToLDS);

// Get the LDS size of the memref
std::optional<int64_t> getWorkgroupMemorySize(MemRefType type);

/// Predict the loop count for a ThreadwiseReadIntoOp by computing the bounds
/// and strides that ThreadwiseReadIntoRewritePattern::matchAndRewrite would
/// create. This function replicates the logic from matchAndRewrite to compute:
/// - numValues: the bound for the iteration dimension
/// - srcStride: the stride for the iteration dimension
/// Returns: numValues / srcStride (the loop count)
FailureOr<ThreadwiseReadIntoLoopInfo>
getThreadwiseReadIntoLoopInfo(const ThreadwiseReadIntoLoopConfigInput &input);

FailureOr<int64_t> predictThreadwiseReadIntoLoopCount(ThreadwiseReadIntoOp op);

} // end namespace rock
} // end namespace mlir
#endif
