//===- RockPasses.h - Conversion Pass Construction and Registration -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_MLIR_CONVERSION_PASSES_H
#define ROCK_MLIR_CONVERSION_PASSES_H

#include "mlir/Conversion/GPUToMIGraphX/GPUToMIGraphX.h"
#include "mlir/Conversion/MIGraphXToTosa/MIGraphXToTosa.h"
#include "mlir/Conversion/RockToGPU/RockToGPU.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TosaToRock/TosaToRock.h"
#include "mlir/Dialect/MIGraphX/Passes.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/RockPasses.h.inc"

} // namespace mlir

#endif // ROCK_MLIR_CONVERSION_PASSES_H
