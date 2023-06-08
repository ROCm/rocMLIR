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
