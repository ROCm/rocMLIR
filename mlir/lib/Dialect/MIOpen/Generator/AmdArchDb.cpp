//===- AmdArchDb.cpp - Dtabase of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/Generator/AmdArchDb.h"

#include "mlir/Dialect/MIOpen/MIOpen.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::miopen;

static constexpr AmdArchInfo gcnInfo(GemmFeatures::none, /*waveSize=*/64),
    cdnaInfo(GemmFeatures::mfma | GemmFeatures::dot | GemmFeatures::atomic_add,
             /*waveSize=*/64),
    rdnaNoDotInfo(GemmFeatures::none, /*waveSize=*/32),
    rdnaInfo(GemmFeatures::dot, /*waveSize=*/32),
    gfx11Info(GemmFeatures::dot | GemmFeatures::atomic_add, /*waveSize=*/32);

AmdArchInfo mlir::miopen::lookupArchInfo(StringRef chip) {
  StringRef minor = chip.take_back(2);
  StringRef major = chip.drop_back(2);
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
  llvm::errs()
      << "Warning: unknown chipset major revision, falling back to defaults: "
      << chip << "\n";
  return gcnInfo;
}
