//===- Passes.h - MHAL pass entry points ----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MHAL_PASSES_H_
#define MLIR_DIALECT_MHAL_PASSES_H_

#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace arith {
class NarrowTypeEmulationConverter;
} // namespace arith
namespace mhal {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.

#define GEN_PASS_DECL_MHALANNOTATEACCESSKINDSPASS
#define GEN_PASS_DECL_MHALTARGETKERNELSPASS
#define GEN_PASS_DECL_MHALINFERGRAPHPASS
#define GEN_PASS_DECL_MHALPACKAGETARGETSPASS
#define GEN_PASS_DECL_MHALSELECTTARGETSPASS
#define GEN_PASS_DECL_MHALBUFFERIZEPASS
#define GEN_PASS_DECL_MHALEMULATENARROWTYPEPASS
#define GEN_PASS_DECL_MHALPREFILLPASS
#define GEN_PASS_DECL_MHALDROPBINARYMETADATAPASS

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"

/// Set up materializations for function arguments as they're converted to
/// i8 memrefs from i4 (or narrower) ones.
void populateMHalNarrowTypeEmulationConversions(
    arith::NarrowTypeEmulationConverter &typeConverter);

/// Adds patterns for rewriting `mhal.launch` ops to `patterns` that replace
/// 4-bit (or other narrow pattern of two) memrefs to 8-bit ones.
void populateMHalNarrowTypeEmulationBoundaryPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns);

/// Adds patterns that handle `extract_strided_metadata` ops targetting the
/// `builtin.unrealized_conversion_cast` operations that the type conversion
/// process introduces to prevent dialect conversion from failing due to stray
/// `memref.extract_strided_metadata` ops.
void populateMHalNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns);
} // namespace mhal
} // namespace mlir

#endif // MLIR_DIALECT_MHAL_PASSES_H_
