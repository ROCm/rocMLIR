//===- DxgmlAttrs.td - Dxgml attribute definitions --------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Dxgml dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef DXGML_ATTRS
#define DXGML_ATTRS

include "mlir/Dialect/Dxgml/IR/DxgmlBase.td"
include "mlir/Dialect/Dxgml/IR/DxgmlTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Version Enum Attribute
//===----------------------------------------------------------------------===//

def Dxgml_V_0_0_1 : I32EnumAttrCase<"V_0_0_1", 0, "v0.0.1">;
def Dxgml_V_0_0_2 : I32EnumAttrCase<"V_0_0_2", 1, "v0.0.2">;
def Dxgml_VersionEnum : I32EnumAttr<"Dxgml_VersionEnum", "Dxgml_VersionEnum",
    [Dxgml_V_0_0_1, Dxgml_V_0_0_2, ]> {
  let cppNamespace = "mlir::dxgml";
}

def Dxgml_VersionAttr : Dxgml_Attr<"Version", "version"> {
  let parameters = (ins
    "::mlir::dxgml::Dxgml_VersionEnum":$ir_version,
    OptionalParameter<"::mlir::StringAttr">:$producer_name,
    OptionalParameter<"::mlir::StringAttr">:$producer_version
  );

  let assemblyFormat = [{
    `<`
    `ir_version` `=` $ir_version
    ( `,` `producer` `=` $producer_name^ )?
    ( `,` `producer_version` `=` $producer_version^ )?
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// Constant Value Attribute
//===----------------------------------------------------------------------===//

def Dxgml_ConstantAttr : Dxgml_Attr<"Constant", "constant_value", [TypedAttrInterface]> {
  let summary = "Inline constant tensor value";
  let description = [{
    Represents an inline constant tensor whose elements are stored directly in
    the IR.

    Example:
    ```mlir
    // 1-D integer tensor
    #dxgml.constant_value<[1, 2, 3]> : !dxgml.tensor<3x!dxgml.int32>

    // Scalar integer
    #dxgml.constant_value<[42]> : !dxgml.tensor<1x!dxgml.int64>

    // Scalar float
    #dxgml.constant_value<[3.14]> : !dxgml.tensor<1x!dxgml.float32>
    ```

    Usage as an op initializer:
    ```mlir
    %0 = #dxgml.constant_value<[1, 2, 3]> : !dxgml.tensor<3x!dxgml.int32>
    ```
  }];
  let cppNamespace = "dxgml";

  let parameters = (ins
    ArrayRefParameter<"::mlir::Attribute">:$value,
    "::mlir::Type":$type
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins
        "::mlir::ArrayRef<::mlir::Attribute>":$value,
        "::mlir::Type":$type), [{
      return $_get(type.getContext(), value, type);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Number of elements in the flat value list.
    size_t getNumElements() const { return getValue().size(); }
  }];
}

//===----------------------------------------------------------------------===//
// Constant Resource Attribute
//===----------------------------------------------------------------------===//

def Dxgml_ConstantResourceAttr : Dxgml_Attr<"ConstantResource", "constant_resource"> {
  let summary = "Reference to an external constant resource blob";
  let description = [{
    This attribute is designed to assign static reference to an independent blob of data 
    that resides as a dialect resource (separate from the body of the MLIR program). 
    This attribute will leverage `DialectResourceBlobManager`, and other internal constructs 
    for splitting programs into .mlir and .mlirres files. Doing this should enable externalizing 
    blobs, and ease the management of large models.

    Example:
    ```mlir
    %0 = dxgml_op.constant(#dxgml.constant_resource<weight.1>) : dxgml.tensor<dxgml.float32>
    ```

    The corresponding resource section in the MLIR file:
    ```mlir
    {-#
      dialect_resources: {
        dxgml: {
          weight.1: "0x08000000..."  // 4bit
        }
      }
    #-}
    ```
  }];
  let cppNamespace = "dxgml";
  let parameters = (ins ResourceHandleParameter<"ConstantHandle">:$key);
  let hasCustomAssemblyFormat = 1;
}

#endif // DXGML_ATTRS
