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
#include "llvm/Support/ErrorHandling.h"

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
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return {};

  if (attrTag == "convolution_direction_enum_attr") {
    if (parser.parseLess())
      return {};

    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword))) {
      parser.emitError(loc, "expected keyword for Convolution direction");
      return {};
    }

    if (parser.parseGreater())
      return {};

    auto value = symbolizeConvolutionDirection(enumKeyword);
    if (!value) {
      parser.emitError(
          loc,
          "expected one of [convolution_direction_forward, "
          "convolution_direction_backward] for Convolution direction, got: ")
          << enumKeyword;
      return {};
    }

    return ConvolutionDirectionAttr::get(getContext(), *value);
  }

  if (attrTag == "convolution_mode_enum_attr") {
    if (parser.parseLess())
      return {};

    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword))) {
      parser.emitError(loc, "expected keyword for Convolution mode");
      return {};
    }

    if (parser.parseGreater())
      return {};

    auto value = symbolizeConvolutionMode(enumKeyword);
    if (!value) {
      parser.emitError(
          loc,
          "expected one of [convolution_mode_convolution, "
          "convolution_mode_cross_correlation] for Convolution mode, got: ")
          << enumKeyword;
      return {};
    }

    return ConvolutionModeAttr::get(getContext(), *value);
  }

  if (attrTag == "depth_space_order_enum_attr") {
    if (parser.parseLess())
      return {};

    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword))) {
      parser.emitError(loc, "expected keyword for Depth to space order");
      return {};
    }

    if (parser.parseGreater())
      return {};

    auto value = symbolizeDepthSpaceOrder(enumKeyword);
    if (!value) {
      parser.emitError(
          loc,
          "expected one of [depth_space_order_depth_column_row, "
          "depth_space_order_column_row_depth] for Depth to space order, got: ")
          << enumKeyword;
      return {};
    }

    return DepthSpaceOrderAttr::get(getContext(), *value);
  }

  parser.emitError(parser.getCurrentLocation())
      << "unknown " << getNamespace() << " attribute '" << attrTag << "'";
  return {};
}

void DxgmlOpDialect::printAttribute(Attribute attr,
                                     DialectAsmPrinter &os) const {
  if (auto direction = dyn_cast<ConvolutionDirectionAttr>(attr)) {
    os << "convolution_direction_enum_attr<"
       << stringifyConvolutionDirection(direction.getValue()) << ">";
    return;
  }

  if (auto mode = dyn_cast<ConvolutionModeAttr>(attr)) {
    os << "convolution_mode_enum_attr<" << stringifyConvolutionMode(mode.getValue())
       << ">";
    return;
  }

  if (auto order = dyn_cast<DepthSpaceOrderAttr>(attr)) {
    os << "depth_space_order_enum_attr<"
       << stringifyDepthSpaceOrder(order.getValue()) << ">";
    return;
  }

  llvm_unreachable("unexpected 'dxgml_op' attribute kind");
}

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
