//===- AmdArchDb.cpp - Dtabase of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Generator/AmdArchDb.h"

#include "mlir/Dialect/Rock/IR/Rock.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::rock;

static constexpr AmdArchInfo gcnInfo(GemmFeatures::none, /*waveSize=*/64),
    cdnaInfo(GemmFeatures::mfma | GemmFeatures::dot | GemmFeatures::atomic_add,
             /*waveSize=*/64),
    rdnaNoDotInfo(GemmFeatures::atomic_fmax_f32, /*waveSize=*/32),
    rdnaInfo(GemmFeatures::dot | GemmFeatures::atomic_fmax_f32,
             /*waveSize=*/32),
    gfx11Info(GemmFeatures::dot | GemmFeatures::atomic_add |
                  GemmFeatures::atomic_fmax_f32,
              /*waveSize=*/32);

AmdArchInfo mlir::rock::lookupArchInfo(StringRef arch) {
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
        .Cases("08", "0a", "40", cdnaInfo)
        // gfx906 has the dot product instructions, uniquely
        .Case("06", AmdArchInfo(GemmFeatures::dot, /*waveSize=*/64))
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
