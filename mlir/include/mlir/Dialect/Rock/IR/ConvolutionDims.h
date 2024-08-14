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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
namespace rock {

/// Structure for holding the dimensions of a convolution problem
struct ConvolutionDims {
  llvm::SmallVector<int64_t, 4> fil;
  llvm::SmallVector<int64_t, 4> out;
  llvm::SmallVector<int64_t, 4> in;
  int64_t k;
  int64_t c;
  int64_t n;
  int64_t g;

  ConvolutionDims(llvm::ArrayRef<int64_t> fil, llvm::ArrayRef<int64_t> out,
                  llvm::ArrayRef<int64_t> in, int64_t k, int64_t c, int64_t n,
                  int64_t g)
      : fil(fil), out(out), in(in), k(k), c(c), n(n), g(g) {}

  static ConvolutionDims fromOp(Operation *op);
};

} // namespace rock
} // namespace mlir
#endif // MLIR?DIALECT_ROCK_IR_CONVOLUTIONDIMS_H
