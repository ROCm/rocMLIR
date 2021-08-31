//===- MIGraphXTypes.td - MIGraphX types definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIGraphX operations.
//
//===----------------------------------------------------------------------===//

#ifndef MIGRAPHX_TYPES
#define MIGRAPHX_TYPES

include "mlir/IR/OpBase.td"

def default_ : I32EnumAttrCase<"default_", 0>;
def same     : I32EnumAttrCase<"same", 1>;
def valid    : I32EnumAttrCase<"valid", 2>;
def PaddingModeAttr : I32EnumAttr<"padding_mode_t", "generic enum padding mode for the OPs",
                           [default_, same, valid]> {
  let cppNamespace = "mlir::migraphx";
}

def constant_pad : I32EnumAttrCase<"constant_pad", 0>;
def reflect_pad  : I32EnumAttrCase<"reflect_pad", 1>;
def edge_pad     : I32EnumAttrCase<"edge_pad", 2>;
def PadOpModeAttr : I32EnumAttr<"pad_op_mode_t", "enum padding mode for the pad OP",
                           [constant_pad, reflect_pad, edge_pad]> {
  let cppNamespace = "mlir::migraphx";
}

#endif // MIGRAPHX_TYPES
