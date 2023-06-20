//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MHAL_CONVERSION_PASSES_H
#define MHAL_CONVERSION_PASSES_H

#include "mlir/Conversion/MHALToCPU/MHALToCPU.h"
#include "mlir/Conversion/MHALToGPU/MHALToGPU.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/MHALPasses.h.inc"

} // namespace mlir

#endif // MHAL_CONVERSION_PASSES_H
