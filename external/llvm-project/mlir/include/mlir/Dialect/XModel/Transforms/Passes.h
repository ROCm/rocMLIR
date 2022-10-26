//===- Passes.h - XModel pass entry points ----------------------*- C++ -*-===//
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

#ifndef MLIR_DIALECT_XMODEL_PASSES_H_
#define MLIR_DIALECT_XMODEL_PASSES_H_

#include "mlir/Dialect/XModel/IR/XModel.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace xmodel {

#define GEN_PASS_DECL_XMODELTARGETKERNELSPASS
#define GEN_PASS_DECL_XMODELASYNCGRAPHPASS
#define GEN_PASS_DECL_XMODELPACKAGETARGETSPASS
#define GEN_PASS_DECL_XMODELSELECTTARGETSPASS

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/XModel/Transforms/Passes.h.inc"

} // namespace xmodel
} // namespace mlir

#endif // MLIR_DIALECT_XMODEL_PASSES_H_
