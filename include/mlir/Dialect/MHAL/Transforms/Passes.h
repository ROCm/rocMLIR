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
namespace mhal {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.

#define GEN_PASS_DECL_MHALTARGETKERNELSPASS
#define GEN_PASS_DECL_MHALINFERGRAPHPASS
#define GEN_PASS_DECL_MHALPACKAGETARGETSPASS
#define GEN_PASS_DECL_MHALSELECTTARGETSPASS
#define GEN_PASS_DECL_MHALBUFFERIZEPASS

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"

} // namespace mhal
} // namespace mlir

#endif // MLIR_DIALECT_MHAL_PASSES_H_
