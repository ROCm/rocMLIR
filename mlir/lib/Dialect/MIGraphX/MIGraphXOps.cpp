//===- MIGraphXOps.cpp - MIGraphX MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::migraphx;

//===----------------------------------------------------------------------===//
// MIGraphXDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {

} // namespace

//===----------------------------------------------------------------------===//
// MIGraphXDialect
//===----------------------------------------------------------------------===//

void MIGraphXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MIGraphX/MIGraphXOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

static ParseResult parseAddOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  SmallVector<Type, 2> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, AddOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(AddOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

static ParseResult parseConvolutionOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  SmallVector<Type, 2> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, ConvolutionOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(ConvolutionOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// LiteralOp
//===----------------------------------------------------------------------===//
/*
static ParseResult parseLiteralOp(OpAsmParser &parser, OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "values", result.attributes))
    return failure();

  // If the attribute is a symbol reference, then we expect a trailing type.
  Type type;
  if (!valueAttr.isa<SymbolRefAttr>())
    type = valueAttr.getType();
  else if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, result.types);	  
}

static void print(OpAsmPrinter &p, LiteralOp &op) {
  p << "Literal ";
  p.printOptionalAttrDict(op.getAttrs(), {"values"});

  if (op.getAttrs().size() > 1)
    p << ' ';
  p << op.getValue();

  // If the value is a symbol reference, print a trailing type.
  if (op.getValue().isa<SymbolRefAttr>())
    p << " : " << op.getType();
}

static LogicalResult verify(LiteralOp op) {
  return success();
}
*/

namespace mlir {

#define GET_OP_CLASSES
#include "mlir/Dialect/MIGraphX/MIGraphXOps.cpp.inc"

} // namespace mlir
