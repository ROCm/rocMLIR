//===- Passes.h - MigraphX pass entry points -------------------*- C++-*-===//
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

#ifndef MLIR_DIALECT_MIGRAPHX_PASSES_H_
#define MLIR_DIALECT_MIGRAPHX_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace migraphx {

#define GEN_PASS_DECL_MIGRAPHXREALIZEINT4PASS
#define GEN_PASS_DECL_MIGRAPHXTRANSFORMPASS
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MIGraphX/Passes.h.inc"

} // namespace migraphx
} // namespace mlir

#endif // MLIR_DIALECT_MIGRAPHX_PASSES_H_
