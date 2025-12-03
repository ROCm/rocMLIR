//===- AmdArchDb.h - Dtabase of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_AMDARCHDB_H
#define MLIR_DIALECT_ROCK_IR_AMDARCHDB_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace rock {
/// A structure containing information about a given AMD chip's features
/// Keep in sync with Python bindings in
/// mlir/lib/Dialect/Rock/utility/Bindings/AmdArchDbBindings.cpp
struct AmdArchInfo {
  GemmFeatures defaultFeatures;
  int64_t waveSize;
  int64_t maxWavesPerEU;
  int64_t totalSGPRPerEU;
  int64_t totalVGPRPerEU;
  int64_t totalSharedMemPerCU;
  int64_t maxSharedMemPerWG; // Not always the same as SharedMemPerCU
  int64_t numEUPerCU;
  int64_t minNumCU;
  bool hasFp8ConversionInstrs;
  bool hasOcpFp8ConversionInstrs;
  bool hasScaledGemm;
  int64_t maxNumXCC;
  bool hasLdsTransposeLoad;

  constexpr AmdArchInfo(GemmFeatures defaultFeatures, int64_t waveSize,
                        int64_t maxWavesPerEU, int64_t totalSGPRPerEU,
                        int64_t totalVGPRPerEU, int64_t sharedMemPerCU,
                        int64_t sharedMemPerWG, int64_t numEUPerCU,
                        int64_t minNumCU, bool hasFp8ConversionInstrs,
                        bool hasOcpFp8ConversionInstrs, bool hasScaledGemm,
                        int64_t maxNumXCC, bool hasLdsTransposeLoad)
      : defaultFeatures(defaultFeatures), waveSize(waveSize),
        maxWavesPerEU(maxWavesPerEU), totalSGPRPerEU(totalSGPRPerEU),
        totalVGPRPerEU(totalVGPRPerEU), totalSharedMemPerCU(sharedMemPerCU),
        maxSharedMemPerWG(sharedMemPerWG), numEUPerCU(numEUPerCU),
        minNumCU(minNumCU), hasFp8ConversionInstrs(hasFp8ConversionInstrs),
        hasOcpFp8ConversionInstrs(hasOcpFp8ConversionInstrs),
        hasScaledGemm(hasScaledGemm), maxNumXCC(maxNumXCC),
        hasLdsTransposeLoad(hasLdsTransposeLoad) {}

  /// Get the default features for the pair <arch, datatype>
  GemmFeatures getDefaultFeatures(Type dataType);

  /// Get the maximum LDS vector length for the given architecture and element
  /// bit width
  int64_t getMaxLDSVectorLength(int64_t elementBitWidth);
};

AmdArchInfo lookupArchInfo(StringRef arch);
bool isDirectToLDSSupported(GemmFeatures features);
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_IR_AMDARCHDB_H
