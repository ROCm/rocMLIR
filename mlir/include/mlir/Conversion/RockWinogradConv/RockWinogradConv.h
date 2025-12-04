//===-- RockWinogradConv.h - Rock Winograd Conv optimization pass declarations
//----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the Rock Winograd Conv Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ROCKWINOGRADCONV_ROCKWINOGRADCONV_H
#define MLIR_CONVERSION_ROCKWINOGRADCONV_ROCKWINOGRADCONV_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_ROCKWINOGRADCONVPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_ROCKWINOGRADCONV_ROCKWINOGRADCONV_H
