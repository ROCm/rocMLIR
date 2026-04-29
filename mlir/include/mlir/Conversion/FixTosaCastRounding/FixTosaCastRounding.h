//===- FixTosaCastRounding.h - Fix tosa.cast rounding -----------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_FIXTOSACASTROUNDING_FIXTOSACASTROUNDING_H
#define MLIR_CONVERSION_FIXTOSACASTROUNDING_FIXTOSACASTROUNDING_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

#define GEN_PASS_DECL_FIXTOSACASTROUNDINGPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace rock {
/// FusedLoc metadata tag used to mark tosa.cast ops that want RTZ rounding.
/// Casts from migraphx.convert carry this tag; casts from quantization do not.
/// Read by the fix-tosa-cast-rounding pass to decide whether to strip the
/// math.roundeven that upstream tosa-to-linalg inserts before arith.fptosi.
constexpr llvm::StringLiteral kRtzCastLocTag("rocmlir.rtz_cast");
} // namespace rock

} // namespace mlir

#endif // MLIR_CONVERSION_FIXTOSACASTROUNDING_FIXTOSACASTROUNDING_H
