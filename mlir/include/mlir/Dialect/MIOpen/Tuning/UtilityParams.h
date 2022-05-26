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

#include <cstdint>

namespace mlir {
namespace miopen {

/// Default block size for utility kernels.
constexpr int64_t kUtilityKernelBlockSize = 64;
/// Default number of elements each utility kernel workitem should handle.
constexpr int64_t kUtilityKernelElemsPerThread = 512;

} // end namespace miopen
} // end namespace mlir
#endif // MLIR_DIALECT_MIOPEN_UTILITY_PARAMS_H
