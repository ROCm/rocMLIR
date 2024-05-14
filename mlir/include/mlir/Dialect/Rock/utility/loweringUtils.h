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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
struct LogicalResult;
class Operation;
class Type;

namespace rock {
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

// The rows and columns of subtile view needs to
// be transposed depending on which operand of
// gemm the view is going to be.
RegsAsMatrixSubTiles transposeSubTileViews(PatternRewriter &rewriter,
                                           Location loc,
                                           RegsAsMatrixSubTiles subTileViews);

// This function will create views of the register buffer of the loaded tile
// of a matrix in global memory. Those views will provide sub-tiles of the
// respective hierarchy within the GPU. See above about RegsAsMatrixSubTiles
FailureOr<RegsAsMatrixSubTiles>
getLoadRegsAsTileViews(OpBuilder &b, Location loc, Value globalBuffer,
                       StringRef dName, ArrayRef<StringRef> bidGridOrder,
                       ArrayRef<int64_t> bidGridLengths, int64_t blockSize,
                       int64_t kPerBlock, int64_t dPerBlock, int64_t kPerThread,
                       int64_t dPerThread, bool isKContigousDim);

// This function will create views of the register buffer of the loaded tile
// but packed as kOuterPerThread, dPerThread and kPackPerThread for max
// vectorization of LDS storing. Those views will provide sub-tiles of the
// respective hierarchy within the GPU. See above about RegsAsMatrixSubTiles
FailureOr<RegsAsMatrixSubTiles> getPackedRegsAsTileViews(
    OpBuilder &b, Location loc, Value globalBuffer, StringRef dName,
    ArrayRef<StringRef> bidGridOrder, ArrayRef<int64_t> bidGridLengths,
    int64_t blockSize, int64_t kPerBlock, int64_t dPerBlock, int64_t kPerThread,
    int64_t dPerThread, int64_t kpack, bool isKContigousDim,
    bool doSwapThreadIterSubDimsForD = false);

bool isWrWAtomicKernel(GemmFeatures features, Type dataType,
                       bool requiredPadding);

bool isAccel(GemmFeatures features);

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

/// Populate a vector of kernel IDs to be used by a backward data convolution
/// algorithm. In the current v4r1 algorithm, several kernels may be needed to
/// realize a complete backward data convolution.
///
/// A non-negative kernel ID denotes an actual implicit GEMM kernels to
/// partipate the backward data convolution. The ID -1 represents a zero
/// initialization utility kernel The zero initialization kernel, if needed,
/// would be placed in the front of the vector.
SmallVector<int64_t> backwardDataKernelIds(ArrayRef<int64_t> strideDims,
                                           ArrayRef<int64_t> dilationDims,
                                           ArrayRef<int64_t> filterDims);

/// Return a vector type of length `len` if `len` is more than 1, otherwise,
/// return `type`.
Type vectorTypeOrSelf(Type elementType, int64_t len);

/// Apply padding to a matrix in its `firstDim` and `secondDim` if applicable.
Value padMatrix(Value matrix, OpBuilder &b, Location loc, StringRef firstDim,
                int64_t firstDimPad, StringRef secondDim, int64_t secondDimPad);

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
TopDownTMBuilder
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

/// Compute, if possible, the constant different between two values.
std::optional<int64_t> computeConstDiff(Value l, Value u);

// Get the arch from the op
FailureOr<StringAttr> getArch(Operation *op);

// Get the num_cu from the op
FailureOr<int64_t> getNumCU(Operation *op);

// Get whether to reverse the grid
FailureOr<UnitAttr> getReverseGrid(Operation *op);

// Get gridSize
FailureOr<IntegerAttr> getGridSize(Operation *op);

// Return an affine map to reverse loop coordinates
AffineMap getIdxReversalMap(OpBuilder &b);

// helper to create ReassociationIndices for flattening
ReassociationIndices getReassociationForFlattening(ShapedType srcTp);

// Return `mhal::PrefillAttr` attributes for a given function
SmallVector<mhal::PrefillAttr>
getStoredPrefillAttributes(mlir::LLVM::LLVMFuncOp func);

} // end namespace rock
} // end namespace mlir
#endif
