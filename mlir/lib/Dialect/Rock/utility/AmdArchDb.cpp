//===- AmdArchDb.cpp - Dtabase of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::rock;

static constexpr AmdArchInfo
    gcnInfo(GemmFeatures::none, /*waveSize=*/64,
            /*maxWavesPerEU*/ 10, /*totalSGPRPerEU*/ 512,
            /*totalVGPRPerEU*/ 256, /*totalSharedMemPerCU*/ 65536,
            /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/80,
            /*hasFp8ConversionInstrs=*/false,
            /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdna50Info(GemmFeatures::dot, /*waveSize=*/64, /*maxWavesPerEU*/ 8,
               /*totalSGPRPerEU*/ 512, /*totalVGPRPerEU*/ 256,
               /*totalSharedMemPerCU*/ 65536, /*maxSharedMemPerWG*/ 65536,
               /*numEUPerCU=*/4, /*minNumCU=*/10,
               /*hasFp8ConversionInstrs=*/false,
               /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdnaInfo(GemmFeatures::mfma | GemmFeatures::dot | GemmFeatures::atomic_add |
                 GemmFeatures::atomic_add_f16,
             /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 512,
             /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
             /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/120,
             /*hasFp8ConversionInstrs=*/false,
             /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdna2Info(GemmFeatures::mfma | GemmFeatures::dot |
                  GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16,
              /*waveSize=*/64, /*maxWavesPerEU*/ 8, /*totalSGPRPerEU*/ 512,
              /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/104,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    cdna3Info(GemmFeatures::mfma | GemmFeatures::dot |
                  GemmFeatures::atomic_add | GemmFeatures::atomic_add_f16,
              /*waveSize=*/64, /*maxWavesPerEU*/ 10, /*totalSGPRPerEU*/ 512,
              /*totalVGPRPerEU*/ 512, /*totalSharedMemPerCU*/ 65536,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/228,
              /*hasFp8ConversionInstrs=*/true,
              /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/8),
    // amdgpu target builds all RDNA in WGP Mode
    rdnaNoDotInfo(GemmFeatures::atomic_fmax_f32, /*waveSize=*/32,
                  /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 512,
                  /*totalVGPRPerEU*/ 1024, /*totalSharedMemPerCU*/ 131072,
                  /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4,
                  /*minNumCU=*/36,
                  /*hasFp8ConversionInstrs=*/false,
                  /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    rdnaInfo(GemmFeatures::dot | GemmFeatures::atomic_fmax_f32,
             /*waveSize=*/32, /*maxWavesPerEU*/ 16, /*totalSGPRPerEU*/ 512,
             /*totalVGPRPerEU*/ 1024, /*totalSharedMemPerCU*/ 131072,
             /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/36,
             /*hasFp8ConversionInstrs=*/false,
             /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1),
    gfx11Info(GemmFeatures::dot | GemmFeatures::atomic_add |
                  GemmFeatures::atomic_fmax_f32 | GemmFeatures::wmma,
              /*waveSize=*/32, /*maxWavesPerEU*/ 20, /*totalSGPRPerEU*/ 512,
              /*totalVGPRPerEU*/ 1536, /*totalSharedMemPerCU*/ 131072,
              /*maxSharedMemPerWG*/ 65536, /*numEUPerCU=*/4, /*minNumCU=*/12,
              /*hasFp8ConversionInstrs=*/false,
              /*hasOcpFp8ConversionInstrs=*/false, /*maxNumXCC=*/1);

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
  if (major == "gfx9" && minor != "50") {
    return llvm::StringSwitch<AmdArchInfo>(minor)
        .Case("08", cdnaInfo)
        .Case("0a", cdna2Info)
        .Cases("40", "41", "42", cdna3Info)
        // gfx906 has the dot product instructions, uniquely
        .Case("06", cdna50Info)
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
  if (major == "gfx12") {
    // TODO (gfx12): some of those information are not accurate and need to be
    // adjusted after hardware release
    AmdArchInfo gfx12Info(gfx11Info);
    gfx12Info.hasFp8ConversionInstrs = false;
    gfx12Info.hasOcpFp8ConversionInstrs = true;
    gfx12Info.totalVGPRPerEU = 1028;
    gfx12Info.maxWavesPerEU = 16;
    gfx12Info.totalSGPRPerEU = 128;
    gfx12Info.totalSharedMemPerCU = 65536;
    gfx12Info.maxSharedMemPerWG = 65536;
    gfx12Info.defaultFeatures =
        bitEnumSet(gfx12Info.defaultFeatures, GemmFeatures::atomic_add_f16);
    return gfx12Info;
  }
  if (major == "gfx9" && minor == "50") {
    // TODO (gfx950): some of those information are not accurate and need to be
    // adjusted after hardware release
    AmdArchInfo gfx950Info(cdna3Info);
    gfx950Info.hasFp8ConversionInstrs = false;
    gfx950Info.hasOcpFp8ConversionInstrs = true;
    return gfx950Info;
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
    if (!(isa<Float16Type, BFloat16Type>(elementType) ||
          elementType.isInteger(8) ||
          (hasFp8ConversionInstrs &&
           isa<Float8E5M2FNUZType, Float8E4M3FNUZType>(elementType)) ||
          (hasOcpFp8ConversionInstrs &&
           isa<Float8E5M2Type, Float8E4M3FNType>(elementType)))) {
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::wmma);
    }
  }
  bool isMfma = bitEnumContainsAll(theseFeatures, GemmFeatures::mfma);

  if (isMfma && !hasFp8ConversionInstrs) {
    if (isa<Float8E4M3FNUZType>(elementType) ||
        isa<Float8E5M2FNUZType>(elementType))
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::mfma);
  }
  if (isMfma && !hasOcpFp8ConversionInstrs) {
    if (isa<Float8E4M3FNType>(elementType) || isa<Float8E5M2Type>(elementType))
      theseFeatures = bitEnumClear(theseFeatures, GemmFeatures::mfma);
  }
  return theseFeatures;
}
