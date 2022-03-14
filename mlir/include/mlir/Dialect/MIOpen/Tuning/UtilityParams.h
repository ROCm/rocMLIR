//===- GridwiseGemmParams.h - MLIR tuning parameter generation --------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines parameters for utility kernels.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_UTILITY_PARAMS_H
#define MLIR_DIALECT_MIOPEN_UTILITY_PARAMS_H

namespace mlir {
namespace miopen {

/// Default grid size and block size for utility kernels used in the lowering
/// process.
constexpr int64_t kUtilityKernelGridSize = 512;
constexpr int64_t kUtilityKernelBlockSize = 64;

} // end namespace miopen
} // end namespace mlir
#endif // MLIR_DIALECT_MIOPEN_UTILITY_PARAMS_H
