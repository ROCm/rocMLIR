//===- WinogradConsts.h - Winograd transform matrices -----------*- C++ -*-===//
//
// Copyright 2025 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Transform matrices for Winograd F(m, r) convolution.
// All matrices are stored as flat arrays in row-major order.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_WINOGRADCONSTS_H
#define MLIR_DIALECT_ROCK_IR_WINOGRADCONSTS_H

#include <array>
#include <cstdint>
#include <utility>

namespace mlir {
namespace rock {
namespace winograd {

struct WinogradParams {
  int64_t m;
  int64_t r;
  int64_t alpha;
  int64_t alphaSq;
};

inline WinogradParams getParams(int32_t fmr) {
  switch (fmr) {
  case 0: { int64_t a = 4; return {2, 3, a, a * a}; }
  case 1: { int64_t a = 6; return {4, 3, a, a * a}; }
  case 2: { int64_t a = 6; return {2, 5, a, a * a}; }
  default: { int64_t a = 4; return {2, 3, a, a * a}; }
  }
}

// F(2,3) transform matrices
// G: filter transform (4x3), transforms 3x3 filter to 4x4
// G[i][j] stored as G_2_3[i*3 + j]
constexpr double G_2_3[] = {
   1.0,   0.0,   0.0,
   0.5,   0.5,   0.5,
   0.5,  -0.5,   0.5,
   0.0,   0.0,   1.0
};

// G^T: (3x4)
constexpr double GT_2_3[] = {
   1.0,   0.5,   0.5,   0.0,
   0.0,   0.5,  -0.5,   0.0,
   0.0,   0.5,   0.5,   1.0
};

// B^T: input transform (4x4)
constexpr double BT_2_3[] = {
   1.0,   0.0,  -1.0,   0.0,
   0.0,   1.0,   1.0,   0.0,
   0.0,  -1.0,   1.0,   0.0,
   0.0,   1.0,   0.0,  -1.0
};

// B: (4x4)
constexpr double B_2_3[] = {
   1.0,   0.0,   0.0,   0.0,
   0.0,   1.0,  -1.0,   1.0,
  -1.0,   1.0,   1.0,   0.0,
   0.0,   0.0,   0.0,  -1.0
};

// A^T: output transform (2x4)
constexpr double AT_2_3[] = {
   1.0,   1.0,   1.0,   0.0,
   0.0,   1.0,  -1.0,  -1.0
};

// A: (4x2)
constexpr double A_2_3[] = {
   1.0,   0.0,
   1.0,   1.0,
   1.0,  -1.0,
   0.0,  -1.0
};

} // namespace winograd
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_IR_WINOGRADCONSTS_H
