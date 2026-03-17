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
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::dxgml;

//===----------------------------------------------------------------------===//
// Dxgml Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/IR/DxgmlDialect.cpp.inc"

// Enum definitions
#include "mlir/Dialect/Dxgml/IR/DxgmlEnums.cpp.inc"

// Attribute definitions
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Dxgml/IR/DxgmlAttrs.cpp.inc"
#undef GET_ATTRDEF_CLASSES

namespace {
struct DxgmlResourceBlobManagerInterface
    : public ResourceBlobManagerDialectInterfaceBase<ConstantHandle> {
  using ResourceBlobManagerDialectInterfaceBase<
      ConstantHandle>::ResourceBlobManagerDialectInterfaceBase;
};
} // namespace

void DxgmlDialect::initialize() {
  addInterface<
DxgmlResourceBlobManagerInterface
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Dxgml/IR/Dxgml.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Dxgml/IR/DxgmlTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Dxgml/IR/DxgmlAttrs.cpp.inc"
#undef GET_ATTRDEF_LIST
      >();
}

Operation *DxgmlDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return nullptr;
}

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

// ModuleOp implementation
void ModuleOp::build(OpBuilder &builder, OperationState &state,
                     StringRef name, FunctionType type,
                     ArrayRef<NamedAttribute> attrs) {
  state.addAttribute("sym_name", builder.getStringAttr(name.empty() ? StringRef("dxgml_module") : name));
  state.addAttribute("function_type", TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  Region *bodyRegion = state.addRegion();
  Block *body = new Block();
  // Add block arguments corresponding to the function inputs so that %argN
  // SSA values are available inside the module body.
  for (Type inputType : type.getInputs())
    body->addArgument(inputType, state.location);
  bodyRegion->push_back(body);
}

ParseResult ModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  if (failed(function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name))))
    return failure();

  if (auto versionAttr = dyn_cast_or_null<StringAttr>(result.attributes.get("version"))) {
    auto parsedVersion = symbolizeDxgml_VersionEnum(versionAttr.getValue());
    if (!parsedVersion)
      return parser.emitError(parser.getCurrentLocation())
             << "invalid dxgml.module version string '" << versionAttr.getValue()
             << "', expected one of [v0.0.1, v0.0.2]";
    result.attributes.set("version",
                          Dxgml_VersionEnumAttr::get(parser.getContext(),
                                                     *parsedVersion));
  }

  return success();
}

void ModuleOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

// InvokeOp implementation
LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return success();
}

//===----------------------------------------------------------------------===//
// Attribute implementations
//===----------------------------------------------------------------------===//

// ConstantResourceAttr implementation
Attribute ConstantResourceAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};
  // Parse the resource handle identifier
  std::string blobName;
  if (parser.parseKeywordOrString(&blobName))
    return {};
  if (parser.parseGreater())
    return {};

  auto *blobMgrIface = parser.getContext()
                           ->getLoadedDialect<DxgmlDialect>()
                           ->getRegisteredInterface<DxgmlResourceBlobManagerInterface>();
  auto handle = blobMgrIface->insert(blobName, {});
  return ConstantResourceAttr::get(parser.getContext(),
                                   ConstantHandle(handle));
}

void ConstantResourceAttr::print(AsmPrinter &printer) const {
  printer << "<" << getKey().getKey() << ">";
}

// ConstantAttr implementation
Attribute ConstantAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess() || parser.parseLSquare())
    return {};

  SmallVector<Attribute> elements;
  // Empty list is not allowed: at least one element is required.
  do {
    Attribute elem;
    if (parser.parseAttribute(elem))
      return {};
    // Validate: only IntegerAttr or FloatAttr are legal elements.
    if (!isa<IntegerAttr, FloatAttr>(elem)) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected integer or float literal in "
                       "#dxgml.constant_value element list, got: ")
          << elem;
      return {};
    }
    elements.push_back(elem);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare())
    return {};

  if (parser.parseGreater() || parser.parseColon())
    return {};

  Type type;
  if (parser.parseType(type))
    return {};

  return ConstantAttr::get(elements, type);
}

void ConstantAttr::print(AsmPrinter &printer) const {
  printer << "<[";
  llvm::interleaveComma(getValue(), printer,
                        [&](Attribute a) { printer.printAttribute(a); });
  printer << "]> : ";
  printer.printType(getType());
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
