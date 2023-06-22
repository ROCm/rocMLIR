//===- MHALToCPU.h - Convert MHAL to CPU dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MHALTOCPU_MHALTOCPU_H
#define MLIR_CONVERSION_MHALTOCPU_MHALTOCPU_H

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTMHALTOCPUPASS
#include "mlir/Conversion/MHALPasses.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_MHALTOCPU_MHALTOCPU_H
