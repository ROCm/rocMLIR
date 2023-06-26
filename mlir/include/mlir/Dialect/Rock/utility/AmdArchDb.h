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
  bool hasFp8ConversionInstrs;

  constexpr AmdArchInfo(GemmFeatures defaultFeatures, int64_t waveSize,
                        bool hasFp8ConversionInstrs)
      : defaultFeatures(defaultFeatures), waveSize(waveSize),
        hasFp8ConversionInstrs(hasFp8ConversionInstrs) {}

  /// Get the default features for the pari <arch, datatype>
  GemmFeatures getDefaultFeatures(Type dataType);
};

AmdArchInfo lookupArchInfo(StringRef arch);
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_AMDARCHDB_H
