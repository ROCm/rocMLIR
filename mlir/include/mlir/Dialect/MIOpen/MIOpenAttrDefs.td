//===- MIOpenAttrDefs.td - MIOpen MLIR Attributes ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MIOPEN_ATTRS
#define MIOPEN_ATTRS

include "mlir/Dialect/MIOpen/MIOpenBase.td"

class MIOpen_Attr<string name, list<Trait> traits = []> :
    AttrDef<MIOpen_Dialect, name, traits>;

def MIOpen_TransformAttr : MIOpen_Attr<"Transform"> {
    let mnemonic = "transform";

    let description = [{
        A coordinate transform.

        TODO: Move the documentation into here.
    }];

    let parameters = (
        ins
        AttrParameter<"mlir::miopen::TransformType",
            "The type of transformation to perform">:$type,
        ArrayRefOfStringRefParameter<
            "The names of the upper dimensions (input arguments) of this transform">:$upperNames,
        ArrayRefParameter<"unsigned",
            "The indices of the upper dimensions (input arguments) of this transform">:$upperDims,
        ArrayRefOfStringRefParameter<
            "The names of the lower dimensions (outputs) of this transformation">:$lowerNames,
        ArrayRefParameter<"unsigned",
            "The indices of the lower dimensions (outputs) of this transformation">:$lowerDims,
        ArrayRefParameter<"int64_t",
            "The parameters to the transformation, if any">:$params
    );

    let genVerifyDecl = 1;
}

#endif
