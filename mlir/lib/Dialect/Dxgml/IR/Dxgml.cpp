//===- Dxgml.cpp - Dxgml MLIR Operations ----------------------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/IR/Dxgml.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::dxgml;

//===----------------------------------------------------------------------===//
// Dxgml Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/IR/DxgmlDialect.cpp.inc"

void DxgmlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Dxgml/IR/Dxgml.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Dxgml/IR/DxgmlTypes.cpp.inc"
      >();
}

Operation *DxgmlDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return nullptr;
}

// Parse/print for dialect
Attribute DxgmlDialect::parseAttribute(DialectAsmParser &parser, 
                                        Type type) const {
  return Attribute();
}

void DxgmlDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &os) const {}

//===----------------------------------------------------------------------===//
// Custom Type Parsers/Printers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace dxgml {

ParseResult parseDxgmlTensorType(AsmParser &parser,
                                  SmallVectorImpl<int64_t> &shape,
                                  Type &elementType) {
  if (parser.parseDimensionList(shape) || parser.parseType(elementType))
    return failure();
  return success();
}

void printDxgmlTensorType(AsmPrinter &printer,
                          ArrayRef<int64_t> shape,
                          Type elementType) {
  for (int64_t dim : shape) {
    printer << dim << "x";
  }
  printer.printType(elementType);
}

} // namespace dxgml
} // namespace mlir

//===----------------------------------------------------------------------===//
// Operation Implementations
//===----------------------------------------------------------------------===//

// FunctionOp implementations
void FunctionOp::build(OpBuilder &builder, OperationState &state,
                       StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs) {
  state.addAttribute("sym_name", builder.getStringAttr(name));
  state.addAttribute("function_type", TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

ParseResult FunctionOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FunctionOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

// EntryPointOp implementations  
void EntryPointOp::build(OpBuilder &builder, OperationState &state,
                         StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs) {
  state.addAttribute("sym_name", builder.getStringAttr(name));
  state.addAttribute("function_type", TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

ParseResult EntryPointOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void EntryPointOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

// InvokeOp implementation
LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return success();
}

// ModuleOp implementation
void ModuleOp::build(OpBuilder &builder, OperationState &state,
                     std::optional<StringRef> name) {
  state.addAttribute("sym_name",
                     builder.getStringAttr(name.value_or("dxgml_module")));
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  bodyRegion->push_back(body);
}

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Dxgml/IR/DxgmlTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Dxgml/IR/Dxgml.cpp.inc"
