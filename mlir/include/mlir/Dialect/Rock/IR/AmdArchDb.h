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
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

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

  /// Get the default features for multiple types (intersects features)
  GemmFeatures getDefaultFeatures(ArrayRef<Type> types);

  /// Get the maximum LDS vector length for the given architecture and element
  /// bit width
  int64_t getMaxLDSVectorLength(int64_t elementBitWidth);

  /// Get features from attribute, falling back to defaultFeatures if attribute
  /// is null
  // TODO: There are methods like this that should be marked as private.
  GemmFeatures getFeaturesFromAttr(ArrayRef<Type> types,
                                   GemmFeaturesAttr featuresAttr);

  // Feature check methods
  bool isAccelEnabled(GemmFeaturesAttr featuresAttr);

  /// Check if accelerator (mfma/wmma) is supported for given types and features
  bool isAccel(Type dataTypeA, Type dataTypeB, GemmFeaturesAttr featuresAttr);

  /// Check if accelerator (mfma/wmma) is supported for given operation
  /// Uses the operation's features attribute internally
  bool isAccel(RockGemmWrapperInterface op);
  bool isAccel(RockGemmGemmWrapperInterface op);

  /// Check if mfma is supported for given types and features
  bool isMfma(Type dataTypeA, Type dataTypeB, GemmFeaturesAttr featuresAttr);

  /// Check if mfma is supported for given operation
  /// Uses the operation's features attribute internally
  bool isMfma(RockGemmWrapperInterface op);
  bool isMfma(RockGemmGemmWrapperInterface op);

  /// Check if wmma is supported for given types and features
  bool isWmma(Type dataTypeA, Type dataTypeB, GemmFeaturesAttr featuresAttr);

  /// Check if wmma is supported for given operation
  /// Uses the operation's features attribute internally
  bool isWmma(RockGemmWrapperInterface op);
  bool isWmma(RockGemmGemmWrapperInterface op);

  /// Check if direct-to-LDS is supported for given type and numBytes
  bool isDirectToLDS(Type dataType, int64_t numBytes = 0);

  /// Check if async direct-to-LDS is supported (needs arch string + type)
  bool isAsyncDirectToLDS(StringRef arch, Type dataType, int64_t numBytes);

  /// Check if dot product is supported (arch-only, no type dependency)
  bool hasDot() const;

  /// Check if atomic add is supported for given type
  bool hasAtomicAdd(Type dataType);

  /// Check if f16 atomic add is supported (arch-only)
  bool hasAtomicAddF16() const;

  /// Check if bf16 atomic add is supported (arch-only)
  bool hasAtomicAddBF16() const;

  /// Check if f32 atomic fmax is supported (arch-only)
  bool hasAtomicFmaxF32() const;

  /// Check if a kernel is a write-read-write atomic kernel
  bool isWrWAtomicKernel(GemmFeaturesAttr featuresAttr, Type dataType,
                         bool requiredPadding);
};

AmdArchInfo lookupArchInfo(StringRef arch);
bool isDirectToLDSSupported(GemmFeatures features);
bool isGlobalPrefetchSupported(StringRef arch);
bool isAsyncDirectToLDSSupported(StringRef arch);
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_IR_AMDARCHDB_H
