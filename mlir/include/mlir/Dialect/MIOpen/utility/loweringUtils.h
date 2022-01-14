//===- loweringUtil.h - functions that often come up during lowering or turing
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MIOPEN_LOWERING_UTIL_H
#define MIOPEN_LOWERING_UTIL_H

#include "mlir/Dialect/MIOpen/MIOpen.h"

#include "mlir/Dialect/MIOpen/utility/math.h"

using namespace mlir;
using namespace mlir::miopen;

namespace mlir {
namespace miopen {
inline int64_t calculateKBlockNum(int64_t n, int64_t ho, int64_t wo) {
  int64_t gemmK = n * ho * wo;
  int64_t gemmKBlocks = 1;
  if (gemmK % 16 == 0) {
    auto lcm = math_util::lcm(ho * wo, (int64_t)16);
    gemmKBlocks = std::min(gemmK / lcm, n);
  } else if (gemmK % 8 == 0) {
    auto comm = math_util::lcm(ho * wo, (int64_t)8);
    gemmKBlocks = std::min(gemmK / comm, n);
  } else if (gemmK % 4 == 0) {
    auto comm = math_util::lcm(ho * wo, (int64_t)4);
    gemmKBlocks = std::min(gemmK / comm, n);
  }
  // not more than n
  gemmKBlocks = std::min(n, gemmKBlocks);
  // not less than 1
  gemmKBlocks = std::max((__int64_t)1, gemmKBlocks);

  // llvm::errs() << "\n gemmKBlocks: " << gemmKBlocks << " gemmK: " << gemmK
  //               << " ho: " << ho << " wo: " << wo << "\n";
  return gemmKBlocks;
}
} // end namespace miopen
} // end namespace mlir
#endif
