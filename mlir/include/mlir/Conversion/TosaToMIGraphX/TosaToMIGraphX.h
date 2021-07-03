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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Conversion/TosaToMIGraphX/TosaToMIGraphX.h.inc"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h.inc"
#include "mlir/Dialect/Tosa/IR/TosaOps.h.inc"

namespace mlir {
namespace tosa {

std::unique_ptr<Pass> createTosaToMIGraphXRandom();
std::unique_ptr<Pass> createTosaToMIGraphXOnTensors();

/// 
void addTosaToMIGraphXRandomPasses(OpPassManager &pm);
void addTosaToMIGraphXOnTensorsPasses(OpPassManager &pm);

/// Populates conversion passes from TOSA dialect to MIGraphX dialect.
void populateConstRandomPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns);
void populateOPConversionPatterns(
    MLIRContext *context, OwningRewritePatternList *patterns);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOMIGRAPHX_H