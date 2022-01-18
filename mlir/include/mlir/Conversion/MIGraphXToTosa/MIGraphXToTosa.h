//===-- MIGraphXToTosa.h - MIGraphX conversion to Tosa pass declarations
//----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the MIGraphX to Tosa Dialect conversion in
// MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MIGRAPHXTOTOSA_H
#define MLIR_CONVERSION_MIGRAPHXTOTOSA_H

#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace migraphx {

std::unique_ptr<Pass> createMIGraphXToTosaPass();
void addMIGraphXToTosaPasses(OpPassManager &pm);

/// Populates conversion passes from MIGraphX dialect to TOSA dialect.
void populateMIGraphXToTosaConversionPatterns(MLIRContext *context,
                                            OwningRewritePatternList *patterns);

} // namespace migraphx
} // namespace mlir

#endif // MLIR_CONVERSION_MIGRAPHXTOTOSA_H