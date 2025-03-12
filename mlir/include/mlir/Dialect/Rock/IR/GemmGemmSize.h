//===--------- GemmGemmSize.h - utility struct for gemm+gemm ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility struct, GemmGemmSize, that packages the sizes of
// gemm+gemm to ensure a cleaner API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_GEMMGEMMCONTEXT_H
#define MLIR_DIALECT_ROCK_IR_GEMMGEMMCONTEXT_H

#include <cstdint>

namespace mlir {
namespace rock {

/// Structure for holding the sizes of a matrix multiplication operation.
struct GemmGemmSize {
  int64_t g;
  int64_t m;
  int64_t k;
  int64_t n;
  int64_t o;

  GemmGemmSize(int64_t g, int64_t m, int64_t k, int64_t n, int64_t o)
      : g(g), m(m), k(k), n(n), o(o) {}

  bool operator==(const GemmGemmSize &other) {
    return (g == other.g) && (m == other.m) && (k == other.k) &&
           (n == other.n) && (o == other.o);
  }
};
} // end namespace rock
} // end namespace mlir
#endif // MLIR_DIALECT_ROCK_IR_GEMMGEMMCONTEXT_H
