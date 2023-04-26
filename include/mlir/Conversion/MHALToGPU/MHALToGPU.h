//===- MHALToGPU.h - Convert MHAL to GPU dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MHALTOGPU_MHALTOGPU_H
#define MLIR_CONVERSION_MHALTOGPU_MHALTOGPU_H

#include <memory>

namespace mlir {

class Pass;

/// Create a pass to convert MHAL operations to the GPU dialect.
std::unique_ptr<Pass> createConvertMHALToGPUPass();

} // namespace mlir

#endif // MLIR_CONVERSION_MHALTOGPU_MHALTOGPU_H
