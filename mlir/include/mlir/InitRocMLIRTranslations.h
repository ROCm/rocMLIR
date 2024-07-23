//===- InitRocMLIRTranslations.h - rocMLIR Translations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all translations
// in and out of MLIR to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITROCMLIRTRANSLATIONS_H
#define MLIR_INITROCMLIRTRANSLATIONS_H

#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Translation/GpuModuleToRocdir.h"

namespace mlir {
namespace rock {
// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically. It should be called after the registerAllTranslations() from
// MLIR
inline void registerRocMLIRTranslations() {
  static bool initOnce = []() {
    registerGpuModuleToROCDLIRTranslation();
    return true;
  }();
  (void)initOnce;
}
} // namespace rock
} // namespace mlir

#endif // MLIR_INITROCMLIRTRANSLATIONS_H
