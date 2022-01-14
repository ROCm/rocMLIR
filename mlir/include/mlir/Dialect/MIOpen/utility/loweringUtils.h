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

#include "mlir/Dialect/MIOpen/utility/math.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::miopen;

namespace mlir {
namespace miopen {
inline LogicalResult isSupportedBackwardDataPaddingKernel(
    bool isXdlops, bool isStride2Pad1, int64_t gemmMExtra, int64_t gemmKExtra,
    int64_t gemmNExtra, mlir::miopen::Conv2DBwdDataOp &op) {
  if (gemmNExtra && gemmKExtra) {
    return op.emitOpError(
        "can't support backward data padding kernel when both pad "
        "gemmN and gemmK due to load issue\n");
  }

  if (isXdlops && (gemmMExtra || gemmNExtra)) {
    if (isStride2Pad1) {
      return op->emitOpError(
          "can't support backward data padding kernel when xdlops stride 2 "
          "pad_h,pad_w>0 and pad gemmM or gemmN due to store issue\n");
    }
  }
  return success();
}

using OobCheckSet = llvm::SmallDenseSet<uint32_t, 4>;
inline ArrayAttr getBoundsCheckAttr(Builder &b, OobCheckSet &set,
                                    uint32_t max) {
  llvm::SmallVector<bool, 8> ret;
  ret.reserve(max);
  for (uint32_t i = 0; i < max; ++i) {
    ret.push_back(set.count(i) != 0);
  }
  return b.getBoolArrayAttr(ret);
}

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
