//===- MHAL.cpp - MHAL MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MHAL/IR/MHAL.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "mlir/Dialect/MHAL/IR/MHALBaseDialect.cpp.inc"
#include "mlir/Dialect/MHAL/IR/MHALTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// MHALDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
struct MHALAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (attr.isa<MHAL::TargetObjectAttr>()) {
      os << "target_obj";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

namespace mlir {
namespace mhal {

//===---------------------------------------------------------
// TargetObjectAttr
// #mhal.target_obj<<type> : <arch> {<attributes>} -> <binary>>
//===---------------------------------------------------------
mlir::Attribute TargetObjectAttr::parse(mlir::AsmParser &parser,
                                        mlir::Type type) {
  llvm::SMLoc startLoc = parser.getCurrentLocation();
  if (parser.parseLess()) {
    return {};
  }

  std::string typeName;
  if (parser.parseKeywordOrString(&typeName)) {
    return {};
  }

  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  Optional<TargetObjectType> targetType = getTargetObjectTypeForName(typeName);
  if (!targetType.has_value()) {
    parser.emitError(typeLoc, "expected a name of a known target object type");
    return {};
  }

  if (parser.parseEqual()) {
    return {};
  }

  std::string archName;
  if (parser.parseKeywordOrString(&archName)) {
    return {};
  }

  NamedAttrList attrList;
  if (parser.parseOptionalAttrDict(attrList)) {
    return {};
  }
  DictionaryAttr attrs = attrList.getDictionary(parser.getContext());

  if (parser.parseArrow()) {
    return {};
  }

  std::string binary;
  if (parser.parseKeywordOrString(&binary)) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  return parser.getChecked<TargetObjectAttr>(startLoc, parser.getContext(),
                                             targetType.value(), archName,
                                             attrs, binary);
}

void TargetObjectAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  StringRef name = getNameForTargetObjectType(getType());
  printer.printKeywordOrString(name);

  printer << " = ";
  printer.printKeywordOrString(getArch());

  auto attrs = getAttributes();
  if (attrs && attrs.size()) {
    printer << " ";
    printer.printAttributeWithoutType(attrs);
  }

  // print binary
  printer << " -> ";
  printer.printKeywordOrString(getBinary());
  printer << ">";
}

//===---------------------------------------------------------
// KernelPackageAttr
// #mhal.target_kernel<'type' = 'arch' : 'entry_name' ['launch_dims']
// {attributes} -> #mhal.target_obj<>>
//===---------------------------------------------------------
void KernelPackageAttr::walkImmediateSubElements(
    llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
    llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
  walkAttrsFn(getObject());
}
Attribute
KernelPackageAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  assert(0);
  return getObject();
}

template <typename T>
static ParseResult
parseAndGather(mlir::AsmParser &parser, AsmParser::Delimiter delim,
               SmallVectorImpl<T> &ret,
               llvm::function_ref<ParseResult(T &)> getElement) {
  return parser.parseCommaSeparatedList(delim, [&]() -> ParseResult {
    T out;
    ParseResult res = getElement(out);
    if (res.succeeded()) {
      ret.push_back(out);
    }
    return res;
  });
}

mlir::Attribute KernelPackageAttr::parse(mlir::AsmParser &parser,
                                         mlir::Type type) {
  llvm::SMLoc startLoc = parser.getCurrentLocation();
  if (parser.parseLess()) {
    return {};
  }

  std::string typeName;
  if (parser.parseKeywordOrString(&typeName)) {
    return {};
  }

  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  Optional<TargetType> targetType = getTargetTypeForName(typeName);
  if (!targetType.has_value()) {
    parser.emitError(typeLoc, "expected a name of a known target type");
    return {};
  }

  if (parser.parseEqual()) {
    return {};
  }

  std::string targetName;
  if (parser.parseKeywordOrString(&targetName)) {
    return {};
  }

  if (parser.parseColon()) {
    return {};
  }

  std::string entryName;
  if (parser.parseKeywordOrString(&entryName)) {
    return {};
  }

  llvm::SmallVector<unsigned> dimensions;
  if (parseAndGather<unsigned>(parser, AsmParser::Delimiter::Square, dimensions,
                               [&](unsigned &out) -> ParseResult {
                                 return parser.parseInteger(out);
                               })) {
    return {};
  }

  NamedAttrList attrList;
  if (parser.parseOptionalAttrDict(attrList)) {
    return {};
  }
  DictionaryAttr attrs = attrList.getDictionary(parser.getContext());

  if (parser.parseArrow()) {
    return {};
  }

  MHAL::TargetObjectAttr object;
  if (parser.parseAttribute(object)) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  return parser.getChecked<KernelPackageAttr>(
      startLoc, parser.getContext(), targetType.value(), targetName, entryName,
      dimensions, attrs, object);
}

void KernelPackageAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  StringRef name = getNameForTargetType(getType());
  printer.printKeywordOrString(name);

  printer << " = ";

  // target - redundant in target_obj
  printer.printKeywordOrString(getTarget());

  printer << " : ";
  printer.printKeywordOrString(getKernelName());

  // LaunchDimensions
  printer << " [";
  llvm::interleaveComma(getLaunchDims(), printer);
  printer << "]";

  auto attrs = getAttributes();
  if (attrs && attrs.size()) {
    printer << " ";
    printer.printAttributeWithoutType(attrs);
  }

  // print binary
  printer << " -> " << getObject();
  printer << ">";
}

} // namespace mhal
} // namespace mlir

//===----------------------------------------------------------------------===//
// MHALDialect
//===----------------------------------------------------------------------===//

void MHAL::MHALDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/MHAL/IR/MHALAttrDefs.cpp.inc"
      >();
  //   addOperations<
  // #define GET_OP_LIST
  // #include "mlir/Dialect/MHAL/IR/MHALOps.cpp.inc"
  //       >();
  addInterfaces<MHALAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MHAL/IR/MHALAttrDefs.cpp.inc"

// #define GET_OP_CLASSES
// #include "mlir/Dialect/MHAL/IR/MHALOps.cpp.inc"
