//===- XModelToGPU.h - Convert XModel to GPU dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_XMODELTOGPU_XMODELTOGPU_H
#define MLIR_CONVERSION_XMODELTOGPU_XMODELTOGPU_H

#include <memory>

namespace mlir {

class Pass;

/// Create a pass to convert XModel operations to the GPU dialect.
std::unique_ptr<Pass> createConvertXModelToGPUPass();

} // namespace mlir

#endif // MLIR_CONVERSION_XMODELTOGPU_XMODELTOGPU_H
