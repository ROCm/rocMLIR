//===-- EmulateFp8ExtTrunc pass declarations ----------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2023 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// Declares the passes for remapping `arith.extf` on fp8 types to a table lookup
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_EMULATEFP8EXTTRUNC_EMULATEFP8EXTTRUNC_H
#define MLIR_CONVERSION_EMULATEFP8EXTTRUNC_EMULATEFP8EXTTRUNC_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_EMULATEFP8EXTTRUNCPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

void addEmulateFp8ExtTruncPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOMIGRAPHX_GPUTOMIGRAPHX_H
