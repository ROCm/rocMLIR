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
def PaddingModeAttr : I32EnumAttr<"padding_mode_t", "enum padding mode for the convolution",
                           [default_, same, valid]> {
  let cppNamespace = "mlir::migraphx";
}

#endif // MIGRAPHX_TYPES
