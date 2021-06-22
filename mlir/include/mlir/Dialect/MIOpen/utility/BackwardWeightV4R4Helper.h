//===- BackwardWeightV4r4Helper.h - Utility routines for AffineMap
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provide utility routines to check AffineMap instances.
//
//===----------------------------------------------------------------------===//

#ifndef BACKWARD_WEIGHT_V4R4_HELPER_H
#define BACKWARD_WEIGHT_V4R4_HELPER_H

using namespace mlir;

namespace mlir {
namespace miopen {

inline __int64_t calculateKBlockNum(__int64_t n, __int64_t ho, __int64_t wo) {
  __int64_t gemmK = n * ho * wo;
  __int64_t gemmKBlocks = 1;
  if (gemmK % 16 == 0) {
    auto lcm = math::lcm(ho * wo, (__int64_t)16);
    gemmKBlocks = std::min(gemmK / lcm, n);
  } else if (gemmK % 8 == 0) {
    auto comm = math::lcm(ho * wo, (__int64_t)8);
    gemmKBlocks = std::min(gemmK / comm, n);
  } else if (gemmK % 4 == 0) {
    auto comm = math::lcm(ho * wo, (__int64_t)4);
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