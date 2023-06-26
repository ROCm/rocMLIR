//===- AmdArchDb.cpp - Dtabase of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::rock;

static constexpr AmdArchInfo gcnInfo(GemmFeatures::none, /*waveSize=*/64,
                                     /*hasFp8ConversionInstrs=*/false),
    cdnaInfo(GemmFeatures::mfma | GemmFeatures::dot | GemmFeatures::atomic_add,
             /*waveSize=*/64, /*hasFp8ConversionInstrs=*/false),
    cdna3Info(GemmFeatures::mfma | GemmFeatures::dot | GemmFeatures::atomic_add,
              /*waveSize=*/64, /*hasFp8ConversionInstrs=*/true),
    rdnaNoDotInfo(GemmFeatures::atomic_fmax_f32, /*waveSize=*/32,
                  /*hasFp8ConversionInstrs=*/false),
    rdnaInfo(GemmFeatures::dot | GemmFeatures::atomic_fmax_f32,
             /*waveSize=*/32, /*hasFp8ConversionInstrs=*/false),
    gfx11Info(GemmFeatures::dot | GemmFeatures::atomic_add |
                  GemmFeatures::atomic_fmax_f32 | GemmFeatures::wmma,
              /*waveSize=*/32, /*hasFp8ConversionInstrs=*/false);

AmdArchInfo mlir::rock::lookupArchInfo(StringRef arch) {
  // Keep this implementation in sync with
  // mlir/test/lit.site.cfg.py.in:set_arch_features()
  StringRef firstPart, remainingParts;
  std::tie(firstPart, remainingParts) = arch.split(':');
  if (firstPart.contains('-')) { // target triple
    std::tie(firstPart, remainingParts) = remainingParts.split(':');
  }
  StringRef chip = firstPart;

  StringRef minor = chip.take_back(2);
  StringRef major = chip.slice(0, chip.size() - 2);
  if (major == "gfx9") {
    return llvm::StringSwitch<AmdArchInfo>(minor)
        .Cases("08", "0a", cdnaInfo)
        .Cases("40", "41", "42", cdna3Info)
        // gfx906 has the dot product instructions, uniquely
        .Case("06", AmdArchInfo(GemmFeatures::dot, /*waveSize=*/64,
                                /*hasFp8ConversionInstrs=*/false))
        .Default(gcnInfo);
  }
  if (major == "gfx10") {
    return llvm::StringSwitch<AmdArchInfo>(minor)
        .Cases("11", "13", rdnaNoDotInfo)
        .Cases("10", "12", rdnaInfo)
        // All gfx103x are the same for us
        .StartsWith("3", rdnaInfo)
        .Default(rdnaNoDotInfo);
  }
  if (major == "gfx11") {
    // We know these chips have common features per backend
    return gfx11Info;
  }
  llvm::errs() << "Warning: unknown architecture, falling back to defaults: "
               << arch << "\n";
  return gcnInfo;
}

GemmFeatures mlir::rock::AmdArchInfo::getDefaultFeatures(Type dataType) {
  GemmFeatures theseFeatures = defaultFeatures;
  bool isWmma = bitEnumContainsAll(theseFeatures, GemmFeatures::wmma);
  Type elementType = getElementTypeOrSelf(dataType);
  if (isWmma) {
    if (!elementType.isF16() && !elementType.isBF16() &&
        !elementType.isInteger(8)) {
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::wmma);
    }
  }
  bool isMfma = bitEnumContainsAll(theseFeatures, GemmFeatures::mfma);
  if (isMfma && !hasFp8ConversionInstrs) {
    if (dataType.isFloat8E4M3FNUZ() || dataType.isFloat8E5M2FNUZ())
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::mfma);
  }
  return theseFeatures;
}
