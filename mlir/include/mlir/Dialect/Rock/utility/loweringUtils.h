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

#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
struct LogicalResult;
class Operation;
class Type;

namespace rock {
struct ConvolutionDims;
struct GemmSize;

struct GPUViews {
  ArrayAttr gridwiseView;
  ArrayAttr blockwiseView;
  ArrayAttr threadwiseView;
};



// This is helper data structure to allow
// lowering define lower dimensions re-shuffling
// into thread_id or iter into reg buffers in threads.
// It is particularly useful to arrange data
// in registers before they are loaded/written
// to some memory (global or lds).
struct LowerDimPartInfo{
  SmallVector<StringRef> lowerDimPartNames;
  SmallVector<int64_t> lowerDimPartOrder;
  SmallVector<int64_t> lowerDimPartSizes;
};

// This function creates tid split arrangement for a certain gemm dimension
// Basically, contigous dimension in global memory needs to be the fastest
// changing dimension in the regs as well as between consecutive threads
std::tuple<LowerDimPartInfo, LowerDimPartInfo> createGlobalLdTidSplits(StringRef dThreadName, int64_t dThreads, int64_t kThreads, bool isKContigousDim);
// This function creates iter split arrangement for a certain gemm dimension
// Basically, contigous dimension in global memory needs to be the fastest
// changing dimension in the regs as well as between consecutive threads
std::tuple<LowerDimPartInfo, LowerDimPartInfo> createGlobalLdIterSplits(StringRef dIterName, int64_t dPerThread, int64_t kPerThread, bool isKContigousDim);
// This function creates iter split arrangement that is suitable for LDS store
// The store buffer needs to be packed as a [KOuterPerThread, dPerThread,
// kpackPerThread] buffer
std::tuple<LowerDimPartInfo, LowerDimPartInfo> createLDSStoreIterSplits(int64_t kPerThread, int64_t dPerThread, int64_t kpack);

// This function will create view of the register buffers in threads
// once they are loaded. Among, the typical arguments it accepts
// how tid, iter to splits of lower gemm dimensions (k & d)
FailureOr<GPUViews> createGemmInputTileViews(
    OpBuilder &b, Location loc, Value gBuffer, StringRef dName,
    ArrayRef<StringRef> bidGridOrder, ArrayRef<int64_t> bidGridLengths,
    int64_t gridSize, int64_t blockSize, int64_t kPerBlock, int64_t dPerBlock,
    LowerDimPartInfo kDimTidPartInfo, LowerDimPartInfo dDimTidPartInfo, 
    LowerDimPartInfo kDimIterPartInfo, LowerDimPartInfo dDimIterPartInfo);

bool isWrWAtomicKernel(GemmFeatures features, Type dataType,
                       bool requiredPadding);

bool isAccel(GemmFeatures features);

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
SmallVector<int64_t>
backwardDataKernelIds(int64_t strideHeight, int64_t strideWidth,
                      int64_t dilationHeight, int64_t dilationWidth,
                      int64_t filterHeight, int64_t filterWidth);

/// Return a vector type of length `len` if `len` is more than 1, otherwise,
/// return `type`.
Type vectorTypeOrSelf(Type elementType, int64_t len);

} // end namespace rock
} // end namespace mlir
#endif
