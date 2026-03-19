//===- Dxgml.cpp - Dxgml MLIR Operations ----------------------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/IR/Dxgml.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"

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
#undef GET_OP_LIST
,
#define GET_OP_LIST
#include "mlir/Dialect/Dxgml/IR/DxgmlOp.cpp.inc"
#undef GET_OP_LIST
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

// ---------------------------------------------------------------------------
// Type Implementations
// ---------------------------------------------------------------------------

namespace mlir {
namespace dxgml {

// NF4 type implementations
::mlir::Type NF4Type::parse(::mlir::AsmParser &parser) {
  Attribute scale, lut;
  if (parser.parseLess() ||
      parser.parseKeyword("scale") || parser.parseEqual() ||
      parser.parseAttribute(scale) || parser.parseComma() ||
      parser.parseKeyword("lut") || parser.parseEqual() ||
      parser.parseAttribute(lut) ||
      parser.parseGreater())
    return {};
  return NF4Type::get(parser.getContext(), scale, lut);
}

void NF4Type::print(::mlir::AsmPrinter &printer) const {
  printer << '<';
  printer << "scale=";
  printer.printAttribute(getScale());
  printer << ", lut=";
  printer.printAttribute(getLut());
  printer << '>';
}

// Tensor type implementations
::mlir::Type TensorType::parse(::mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;
  if (parser.parseLess() ||
      parser.parseDimensionList(shape) ||
      parser.parseType(elementType) ||
      parser.parseGreater())
    return {};
  return TensorType::get(parser.getContext(), shape, elementType);
}

void TensorType::print(::mlir::AsmPrinter &printer) const {
  printer << '<';
  for (int64_t dim : getOptionalSizes()) {
    if (dim == ShapedType::kDynamic)
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer.printType(getDtype());
  printer << '>';
}

::llvm::LogicalResult TensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> optionalSizes,
    ::mlir::Type dtype) {
  for (int64_t dim : optionalSizes) {
    if (dim != ShapedType::kDynamic && dim < 0)
      return emitError() << "tensor dimension must be a non-negative integer "
                            "or '?' (dynamic), but got: " << dim;
  }
  return ::mlir::success();
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
    auto parsedVersion = symbolizeVersionEnum(versionAttr.getValue());
    if (!parsedVersion)
      return parser.emitError(parser.getCurrentLocation())
             << "invalid dxgml.module version string '" << versionAttr.getValue()
             << "', expected one of [v0.0.1, v0.0.2]";
    result.attributes.set("version",
                          VersionEnumAttr::get(parser.getContext(),
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

//===----------------------------------------------------------------------===//
// DxgmlOp Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Dxgml/IR/DxgmlOp.cpp.inc"

namespace mlir {
namespace dxgml {
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}
} // namespace dxgml
} // namespace mlir
