//===-MIGraphXToLinalg.h - MIGraphX conversion to Linalg pass declarations-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the MIGraphX to Linalg Dialect conversion
// in MLIR.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MIGRAPHXTOLINALG_H
#define MLIR_CONVERSION_MIGRAPHXTOLINALG_H

#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_MIGRAPHXTOLINALGPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace migraphx {
/// Populates conversion passes from MIGraphX dialect to Linalg dialect.
void populateMIGraphXToLinalgConversionPatterns(TypeConverter &converter,
                                                RewritePatternSet &patterns);

/// Configure legal and illegal operations for MIGraphx to Linalg dialect
void populateMIGraphXToLinalgDialectConversion(ConversionTarget &target);

/// Configure legal and illegal operations for MIGraphX to Linalg boundary
/// conversion
void populateMIGraphXToLinalgBoundaryDialectConversion(
    ConversionTarget &target, TypeConverter &converter);

/// Populates conversion patterns for function boundaries, including
/// migraphx.mlir.as_logical_shape and migraphx.mlir.as_underlying_shape.
void populateMIGraphXFuncBoundaryToLinalgConversionPatterns(
    RewritePatternSet &target, TypeConverter &typeConverter);
} // namespace migraphx
} // namespace mlir

#endif
