//===- AmdArchDb.h - Dtabase of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_AMDARCHDB_H
#define MLIR_DIALECT_ROCK_UTILITY_AMDARCHDB_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace rock {
/// A structure containing information about a given AMD chip's features
struct AmdArchInfo {
  GemmFeatures defaultFeatures;
  int64_t waveSize;

  constexpr AmdArchInfo(GemmFeatures defaultFeatures, int64_t waveSize)
      : defaultFeatures(defaultFeatures), waveSize(waveSize) {}
};

AmdArchInfo lookupArchInfo(StringRef arch);
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_AMDARCHDB_H
