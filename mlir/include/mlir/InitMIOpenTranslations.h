//===- InitAllTranslations.h - MLIR Translations Registration ---*- C++ -*-===//
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

#ifndef MLIR_INITALLTRANSLATIONS_H
#define MLIR_INITALLTRANSLATIONS_H

namespace mlir {

void registerFromLLVMIRTranslation();
void registerToLLVMIRTranslation();
void registerToROCDLIRTranslation();
void registerFromMIOpenToCPPTranslation();

namespace miopen {
// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerAllTranslations() {
  static bool initOnce = []() {
    registerFromLLVMIRTranslation();
    registerToLLVMIRTranslation();
    registerToROCDLIRTranslation();
    registerFromMIOpenToCPPTranslation();
    return true;
  }();
  (void)initOnce;
}
} // namespace miopen
} // namespace mlir

#endif // MLIR_INITALLTRANSLATIONS_H
