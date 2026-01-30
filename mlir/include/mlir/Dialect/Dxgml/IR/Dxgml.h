//===- Dxgml.h - Dxgml dialect ----------------------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DXGML_IR_DXGML_H
#define MLIR_DIALECT_DXGML_IR_DXGML_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------===//
// Dxgml Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/IR/DxgmlDialect.h.inc"

//===----------------------------------------------------------------------===//
// Dxgml Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Dxgml/IR/DxgmlTypes.h.inc"

//===----------------------------------------------------------------------===//
// Dxgml Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Dxgml/IR/Dxgml.h.inc"

//===----------------------------------------------------------------------===//
// Custom Parsers/Printers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace dxgml {

// Custom parser for DxgmlTensor type in assembly format
ParseResult parseDxgmlTensorType(AsmParser &parser,
                                  SmallVectorImpl<int64_t> &shape,
                                  Type &elementType);

// Custom printer for DxgmlTensor type in assembly format  
void printDxgmlTensorType(AsmPrinter &printer,
                          ArrayRef<int64_t> shape,
                          Type elementType);

} // namespace dxgml
} // namespace mlir

#endif // MLIR_DIALECT_DXGML_IR_DXGML_H
