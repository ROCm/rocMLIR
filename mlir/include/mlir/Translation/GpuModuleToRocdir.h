//===- GpuModuleToRocdlir.h - ROCDL in GPU modules to LLVM IR ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for ROCDL dialect to LLVM IR translation in
// GPU modules.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSLATION_GPUMODULETOROCDLIR_H
#define MLIR_TRANSLATION_GPUMODULETOROCDLIR_H

namespace mlir {
namespace rock {
/// A translation that converts LLVM Dialect within a GPU module
/// to LLVM IR
void registerGpuModuleToROCDLIRTranslation();
} // namespace rock
} // namespace mlir

#endif // MLIR_TRANSLATION_GPUMODULETOROCDLIR_H
