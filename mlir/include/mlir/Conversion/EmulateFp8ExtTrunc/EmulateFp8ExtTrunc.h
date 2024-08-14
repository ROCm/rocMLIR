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
class FlatSymbolRefAttr;

#define GEN_PASS_DECL_EMULATEFP8EXTTRUNCPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

// The arguments are functions for converting a float (fp32) to the relevant
// type. If the attribute is `nullptr`, then that truncation pattern is
// disabled.
void addEmulateFp8ExtTruncPatterns(RewritePatternSet &patterns,
                                   FlatSymbolRefAttr f8E4M3FNUZTruncFunc,
                                   FlatSymbolRefAttr f8E5M2FNUZTruncFunc,
                                   FlatSymbolRefAttr f8E4M3FNTruncFunc,
                                   FlatSymbolRefAttr f8E5M2TruncFunc);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOMIGRAPHX_GPUTOMIGRAPHX_H
