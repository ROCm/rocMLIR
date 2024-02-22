//===--------- GemmSize.h - utility struct for GEMM ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility struct, GemmSize, that packages the sizes of a
// matrix multiplication to ensure a cleaner API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_GEMMCONTEXT_H
#define MLIR_DIALECT_ROCK_IR_GEMMCONTEXT_H

#include <cstdint>

namespace mlir {
namespace rock {
struct ConvolutionDims;
enum class ConvOpType : uint32_t;

/// Structure for holding the sizes of a matrix multiplication operation.
struct GemmSize {
  int64_t g;
  int64_t m;
  int64_t k;
  int64_t n;

  GemmSize(int64_t g, int64_t m, int64_t k, int64_t n)
      : g(g), m(m), k(k), n(n) {}

  /// Compute the gemm size given a convolution type and its dimensions.
  static GemmSize fromConvolution(ConvOpType type,
                                  const ConvolutionDims &sizes);

  bool operator==(const GemmSize &other) {
    return (g == other.g) && (m == other.m) && (k == other.k) && (n == other.n);
  }
};
} // end namespace rock
} // end namespace mlir
#endif // MLIR_DIALECT_ROCK_IR_GEMMCONTEXT_H
