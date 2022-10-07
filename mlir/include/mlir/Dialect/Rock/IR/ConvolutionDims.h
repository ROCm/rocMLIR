//===--------- ConvolutionDims.h - utility struct for conv dims -------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This file defines a utility struct, ConvolutionDims, that packages the sizes
// of a convolution to allow for a cleaner API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_CONVOLUTIONDIMS_H
#define MLIR_DIALECT_ROCK_IR_CONVOLUTIONDIMS_H

#include <cstdint>

namespace mlir {
class Operation;
namespace rock {

/// Structure for holding the dimensions of a convolution problem
struct ConvolutionDims {
  int64_t y;
  int64_t x;
  int64_t ho;
  int64_t wo;
  int64_t hi;
  int64_t wi;
  int64_t k;
  int64_t c;
  int64_t n;
  int64_t g;

  ConvolutionDims(int64_t y, int64_t x, int64_t ho, int64_t wo, int64_t hi,
                  int64_t wi, int64_t k, int64_t c, int64_t n, int64_t g)
      : y(y), x(x), ho(ho), wo(wo), hi(hi), wi(wi), k(k), c(c), n(n), g(g) {}

  static ConvolutionDims fromOp(Operation *op);
};

} // namespace rock
} // namespace mlir
#endif // MLIR?DIALECT_ROCK_IR_CONVOLUTIONDIMS_H
