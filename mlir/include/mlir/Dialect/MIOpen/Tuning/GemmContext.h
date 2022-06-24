//===--------- GemmContext.h - utility struct for GEMM ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility struct, GemmContext, that packages the sizes of a
// matrix multiplication to ensure a cleaner API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_TUNING_GEMMCONTEXT_H
#define MLIR_DIALECT_MIOPEN_TUNING_GEMMCONTEXT_H

#include "mlir/Dialect/MIOpen/MIOpen.h"

namespace mlir {
class Operation;
namespace miopen {
struct ConvolutionDims;
struct GemmContext {
  int64_t m;
  int64_t k;
  int64_t n;

  GemmContext(int64_t m, int64_t k, int64_t n) : m(m), k(k), n(n) {}

  /// Compute the gemm size given a convolution type and its dimensions.
  static GemmContext fromConvolution(ConvOpType type, ConvolutionDims sizes);
};
} // end namespace miopen
} // end namespace mlir

#endif // MLIR_DIALECT_MIOPEN_TUNING_GEMMCONTEXT_H
