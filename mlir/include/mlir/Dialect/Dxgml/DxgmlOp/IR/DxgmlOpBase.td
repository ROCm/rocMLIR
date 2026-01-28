//===- DxgmlOpBase.td - DxgmlOp dialect definitions -------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines base MLIR DxgmlOp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DXGMLOP_BASE
#define DXGMLOP_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def DxgmlOp_Dialect : Dialect {
  let name = "dxgml_op";
  let cppNamespace = "::mlir::dxgml_op";
  
  let summary = "The DxGML operations dialect";
  let description = [{
    The DxGML operations dialect provides machine learning operations
    used by the DXGML compiler.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Base class for DxgmlOp dialect ops
class DxgmlOp_Op<string mnemonic, list<Trait> traits = []> :
    Op<DxgmlOp_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def DxgmlOp_ConvolutionModeEnum : I32EnumAttr<"ConvolutionMode",
    "Convolution mode", [
  I32EnumAttrCase<"convolution", 0, "convolution_mode_convolution">,
  I32EnumAttrCase<"cross_correlation", 1, "convolution_mode_cross_correlation">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_ConvolutionDirectionEnum : I32EnumAttr<"ConvolutionDirection",
    "Convolution direction", [
  I32EnumAttrCase<"forward", 0, "convolution_direction_forward">,
  I32EnumAttrCase<"backward", 1, "convolution_direction_backward">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

def DxgmlOp_DepthSpaceOrderEnum : I32EnumAttr<"DepthSpaceOrder",
    "Depth to space order", [
  I32EnumAttrCase<"depth_column_row", 0, "depth_space_order_depth_column_row">,
  I32EnumAttrCase<"column_row_depth", 1, "depth_space_order_column_row_depth">
]> {
  let cppNamespace = "::mlir::dxgml_op";
}

#endif // DXGMLOP_BASE
