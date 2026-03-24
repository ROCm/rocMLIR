//===- DxgmlOp.cpp - DxgmlOp MLIR Operations ------------------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOp.h"
#include "mlir/Dialect/Dxgml/IR/Dxgml.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

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
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpBaseAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DxgmlOp Enums
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Base AttrDef Definitions (from DxgmlOpBase.td)
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpBaseAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOp.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute valueAttr;
  if (parser.parseLParen() || parser.parseAttribute(valueAttr) ||
      parser.parseRParen())
    return failure();

  // Extract type from TypedAttr
  if (auto typedAttr = dyn_cast<TypedAttr>(valueAttr)) {
    result.addTypes(typedAttr.getType());
  } else {
    return parser.emitError(parser.getNameLoc(),
                           "constant value must be a typed attribute");
  }

  result.getOrAddProperties<ConstantOp::Properties>().value = valueAttr;

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void ConstantOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer.printAttribute(getValue());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// NullPtrOp
//===----------------------------------------------------------------------===//

ParseResult NullPtrOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Result type is !dxgml.null
  result.addTypes(dxgml::NullType::get(parser.getContext()));

  return success();
}

void NullPtrOp::print(OpAsmPrinter &printer) {
  printer.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// GroupQueryAttentionOp
//===----------------------------------------------------------------------===//

ParseResult GroupQueryAttentionOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  // Parse optional op_name: ["name"]
  if (parser.parseOptionalLSquare().succeeded()) {
    StringAttr nameAttr;
    if (parser.parseAttribute(nameAttr) || parser.parseRSquare())
      return failure();
    result.addAttribute("op_name", nameAttr);
  }

  // Parse operands: (op1, op2, ..., opN)
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseLParen())
    return failure();
  if (parser.parseOptionalRParen().failed()) {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand))
      return failure();
    operands.push_back(operand);
    while (parser.parseOptionalComma().succeeded()) {
      if (parser.parseOperand(operand))
        return failure();
      operands.push_back(operand);
    }
    if (parser.parseRParen())
      return failure();
  }

  // Parse attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse : (operand-types) -> (result-types)
  if (parser.parseColon() || parser.parseLParen())
    return failure();
  SmallVector<Type> operandTypes;
  if (parser.parseOptionalRParen().failed()) {
    Type t;
    if (parser.parseType(t))
      return failure();
    operandTypes.push_back(t);
    while (parser.parseOptionalComma().succeeded()) {
      if (parser.parseType(t))
        return failure();
      operandTypes.push_back(t);
    }
    if (parser.parseRParen())
      return failure();
  }

  if (operands.size() != operandTypes.size())
    return parser.emitError(parser.getNameLoc(),
                            "operand count mismatch with type list");

  // Resolve operands: first 7 are fixed, rest are extra_inputs
  if (operands.size() < 7)
    return parser.emitError(parser.getNameLoc(),
                            "expected at least 7 operands for group_query_attention");

  for (size_t i = 0; i < 7; ++i)
    if (parser.resolveOperand(operands[i], operandTypes[i], result.operands))
      return failure();
  for (size_t i = 7; i < operands.size(); ++i)
    if (parser.resolveOperand(operands[i], operandTypes[i], result.operands))
      return failure();

  // Parse -> (result-types)
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  SmallVector<Type> resultTypes;
  if (parser.parseOptionalRParen().failed()) {
    Type t;
    if (parser.parseType(t))
      return failure();
    resultTypes.push_back(t);
    while (parser.parseOptionalComma().succeeded()) {
      if (parser.parseType(t))
        return failure();
      resultTypes.push_back(t);
    }
    if (parser.parseRParen())
      return failure();
  }
  result.addTypes(resultTypes);
  return success();
}

void GroupQueryAttentionOp::print(OpAsmPrinter &printer) {
  if (auto name = getOpName())
    printer << "[\"" << *name << "\"] ";

  printer << "(";
  printer << getQuery() << ", " << getKey() << ", " << getValue() << ", "
          << getPastKey() << ", " << getPastValue() << ", "
          << getSeqlensK() << ", " << getTotalSequenceLength();
  for (auto extra : getExtraInputs())
    printer << ", " << extra;
  printer << ")";

  printer.printOptionalAttrDict((*this)->getAttrs(), {"op_name"});

  printer << " : (";
  printer << getQuery().getType() << ", " << getKey().getType() << ", "
          << getValue().getType() << ", " << getPastKey().getType() << ", "
          << getPastValue().getType() << ", " << getSeqlensK().getType() << ", "
          << getTotalSequenceLength().getType();
  for (auto extra : getExtraInputs())
    printer << ", " << extra.getType();
  printer << ") -> (";
  printer << getOutput().getType() << ", " << getPresentKey().getType() << ", "
          << getPresentValue().getType() << ", "
          << getOutputQkMatrix().getType();
  printer << ")";
}
