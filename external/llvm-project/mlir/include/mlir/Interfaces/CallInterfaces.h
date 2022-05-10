//===- CallInterfaces.h - Call Interfaces for MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the call interfaces defined in
// `CallInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CALLINTERFACES_H
#define MLIR_INTERFACES_CALLINTERFACES_H

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/PointerUnion.h"

namespace mlir {

namespace func {
class FuncOp;
}
/// A callable is either a symbol, or an SSA value, that is referenced by a
/// call-like operation. This represents the destination of the call.
struct CallInterfaceCallable : public PointerUnion<SymbolRefAttr, Value> {
  using PointerUnion<SymbolRefAttr, Value>::PointerUnion;
};
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/CallInterfaces.h.inc"

#endif // MLIR_INTERFACES_CALLINTERFACES_H
