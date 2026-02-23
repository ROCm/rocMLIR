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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_MIGRAPHXTOTOSAPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace migraphx {

/// Convert MIXR shaped types to a tensor that's just their shaped part,
/// discarding the strides. Note that this isn't going to going to handle
/// function inputs and return values, which'll temporarily be "converted" with
/// migraphx.mlir.as_logical_shape and migraphx.mlir.as_underlying_shape
/// until those are rewritten away in later passes.
class MIXRShapedToTensorConverter : public TypeConverter {
public:
  MIXRShapedToTensorConverter();
};

/// Convert MIXR shaped types to tensor types that represent the in-memory
/// layout of the shaped type.
class MIXRShapedToMemoryLayoutConverter : public TypeConverter {
public:
  MIXRShapedToMemoryLayoutConverter();
};

/// Configure dialect conversion from MIGraphX to Tosa for partial conversion.
void populateMIGraphXToTosaDialectConversion(ConversionTarget &target,
                                             TypeConverter *typeConverter);

/// Populates conversion passes from MIGraphX dialect to TOSA dialect.
void populateMIGraphXToTosaConversionPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter);

/// Configure dialect conversion for MIXR functions, converting shaped types to
/// tensors that represent the underlying memory layout.
void populateMIGraphXFuncBoundaryToTosaDialectConversion(
    ConversionTarget &target, TypeConverter *typeConverter);

/// Populates conversion patterns for function boundaries, including
/// migraphx.mlir.as_logical_shape and migraphx.mlir.as_underlying_shape.
void populateMIGraphXFuncBoundaryToTosaConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter);

void addMIGraphXToTosaPasses(OpPassManager &pm);
} // namespace migraphx
} // namespace mlir

#endif // MLIR_CONVERSION_MIGRAPHXTOTOSA_H
