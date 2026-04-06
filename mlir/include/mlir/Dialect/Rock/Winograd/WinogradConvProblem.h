//===-- WinogradConvProblem.h - Conv problem descriptor ----------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_CONV_PROBLEM_H
#define MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_CONV_PROBLEM_H

#include <cstdint>
#include <string>

namespace mlir {
namespace rock {
namespace winograd {

enum class WinogradDirection {
  Forward,
  BackwardData,
  BackwardWeight,
};

struct WinogradConvProblem {
  std::string arch;
  int64_t N = 0;
  int64_t C = 0;
  int64_t H = 0;
  int64_t W = 0;
  int64_t K = 0;
  int64_t R = 0;
  int64_t S = 0;
  int64_t outH = 0;
  int64_t outW = 0;
  int64_t padH = 0;
  int64_t padW = 0;
  int64_t strideH = 1;
  int64_t strideW = 1;
  int64_t dilationH = 1;
  int64_t dilationW = 1;
  int64_t groupCount = 1;
  int64_t numCU = 0;
  bool isFp16 = false;
  bool isFp32 = false;
  bool isBf16 = false;
  bool isXnackEnabled = false;
  bool isNCHW = true;
  WinogradDirection direction = WinogradDirection::Forward;
};

} // namespace winograd
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_CONV_PROBLEM_H
