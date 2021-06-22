//===-- TosaToMIGraphX.h - TOSA optimization pass declarations ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA MIGraphX Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSATOMIGRAPHX_H
#define MLIR_CONVERSION_TOSATOMIGRAPHX_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {

std::unique_ptr<Pass> createTosaToMIGraphXRandom();

/// Populates passes to convert from TOSA to MIOpen on buffers. At the end of
/// the pass, the function will only contain MIOpen ops or standard ops if the
/// pipeline succeeds.
void addTosaToMIGraphXRandomPasses(OpPassManager &pm);

/// Populates conversion passes from TOSA dialect to MIOpen dialect.
void populateConstRandomPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOMIGRAPHX_H