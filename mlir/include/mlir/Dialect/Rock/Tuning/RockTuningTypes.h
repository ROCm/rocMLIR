//===--------- ConvContext.h - MLIR tuning parameter generation ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR base types for tuning
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
#define MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H

#include "mlir/Dialect/Rock/IR/Rock.h"

namespace mlir {
namespace rock {

enum TunableOpType { TunableOpConv2dFwd, TunableOpGemm };

// Tuning parameter struct
typedef std::vector<uint32_t> TuningParam;

// Parameter container holding a parameter and serialized string
struct ParamEntry {
  TuningParam param;
  std::string perfString;
};

// Total tuning space
struct TunableParams {
  std::vector<TuningParam> tuningRange;
  TunableOpType opType;
  int numHeuristicQuick;
};

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
