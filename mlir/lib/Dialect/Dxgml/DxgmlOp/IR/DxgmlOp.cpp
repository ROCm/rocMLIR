//===- DxgmlOp.cpp - DxgmlOp MLIR Operations ------------------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOp.h"
#include "mlir/Dialect/Dxgml/IR/Dxgml.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::dxgml_op;

//===----------------------------------------------------------------------===//
// DxgmlOp Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpDialect.cpp.inc"

void DxgmlOpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOp.cpp.inc"
      >();
}

// Parse/print for dialect
Attribute DxgmlOpDialect::parseAttribute(DialectAsmParser &parser,
                                          Type type) const {
  return Attribute();
}

void DxgmlOpDialect::printAttribute(Attribute attr,
                                     DialectAsmPrinter &os) const {}

Type DxgmlOpDialect::parseType(DialectAsmParser &parser) const {
  return Type();
}

void DxgmlOpDialect::printType(Type type, DialectAsmPrinter &os) const {}

//===----------------------------------------------------------------------===//
// DxgmlOp Enums
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOp.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}
