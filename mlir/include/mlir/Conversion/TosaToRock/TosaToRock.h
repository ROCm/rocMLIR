//===-- TosaToRock.h - TOSA optimization pass declarations ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA Rock Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSATOROCK_TOSATOROCK_H
#define MLIR_CONVERSION_TOSATOROCK_TOSATOROCK_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_TOSATOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace tosa {

/// Populates passes to convert from TOSA to Rock on buffers. At the end of
/// the pass, the function will only contain Rock ops or standard ops if the
/// pipeline succeeds.
void addTosaToRockPasses(OpPassManager &pm);

/// Populates conversion passes from TOSA dialect to Rock dialect.
void populateTosaToRockConversionPatterns(MLIRContext *context,
                                          RewritePatternSet &patterns);

} // namespace tosa

} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOROCK_TOSATOROCK_H
