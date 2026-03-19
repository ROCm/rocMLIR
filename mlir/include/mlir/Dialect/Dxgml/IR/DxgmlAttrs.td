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
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Constant Resource Attribute
//===----------------------------------------------------------------------===//

def Dxgml_ConstantResourceAttr : Dxgml_Attr<"ConstantResource", "ConstantResource"> {
  let summary = "Reference to an external constant resource blob";
  let description = [{
    This attribute is designed to assign static reference to an independent blob of data 
    that resides as a dialect resource (separate from the body of the MLIR program). 
    This attribute will leverage `DialectResourceBlobManager`, and other internal constructs 
    for splitting programs into .mlir and .mlirres files. Doing this should enable externalizing 
    blobs, and ease the management of large models.

    Example:
    ```mlir
    %0 = DxGML.Constant(#DxGML.ConstantResource<weight.1>) : DxGML.Tensor<DxGML.Float32>
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
  let parameters = (ins ResourceHandleParameter<"ConstantHandle">:$key);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Constant Value Attribute
//===----------------------------------------------------------------------===//

def Dxgml_ConstantAttr : Dxgml_Attr<"Constant", "ConstantValue", [TypedAttrInterface]> {
  let summary = "Inline constant tensor value";
  let description = [{
    Represents an inline constant tensor whose elements are stored directly in
    the IR.

    Example:
    ```mlir
    // 1-D integer tensor
    #DxGML.ConstantValue<[1, 2, 3]> : !DxGML.Tensor<3x!DxGML.Int32>

    // Scalar integer
    #DxGML.ConstantValue<[42]> : !DxGML.Tensor<1x!DxGML.Int64>

    // Scalar float
    #DxGML.ConstantValue<[3.14]> : !DxGML.Tensor<1x!DxGML.Float32>
    ```

    Usage as an op initializer:
    ```mlir
    %0 = #DxGML.ConstantValue<[1, 2, 3]> : !DxGML.Tensor<3x!DxGML.Int32>
    ```
  }];

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
// Enums
//===----------------------------------------------------------------------===//

def Dxgml_VersionEnum : I32EnumAttr<"VersionEnum", "DxGML version", [
  I32EnumAttrCase<"V_0_0_1", 0, "v0.0.1">,
  I32EnumAttrCase<"V_0_0_2", 1, "v0.0.2">
]> { let cppNamespace = "::mlir::dxgml"; }

def Dxgml_ConvolutionModeEnum : I32EnumAttr<"ConvolutionMode", "Convolution mode", [
  I32EnumAttrCase<"convolution", 0, "CONVOLUTION_MODE_CONVOLUTION">,
  I32EnumAttrCase<"cross_correlation", 1, "CONVOLUTION_MODE_CROSS_CORRELATION">
]> { let cppNamespace = "::mlir::dxgml"; }

def Dxgml_ConvolutionDirectionEnum : I32EnumAttr<"ConvolutionDirection", "Convolution direction", [
  I32EnumAttrCase<"forward", 0, "CONVOLUTION_DIRECTION_FORWARD">,
  I32EnumAttrCase<"backward", 1, "CONVOLUTION_DIRECTION_BACKWARD">
]> { let cppNamespace = "::mlir::dxgml"; }

def Dxgml_DepthSpaceOrderEnum : I32EnumAttr<"DepthSpaceOrder", "Depth to space order", [
  I32EnumAttrCase<"depth_column_row", 0, "DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW">,
  I32EnumAttrCase<"column_row_depth", 1, "DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH">
]> { let cppNamespace = "::mlir::dxgml"; }

//===----------------------------------------------------------------------===//
// Enum Attribute
//===----------------------------------------------------------------------===//

def Dxgml_VersionAttr
    : Dxgml_Attr<"Version", "version"> {
  let parameters = (
    ins
    "::mlir::dxgml::VersionEnum":$ir_version,
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

def Dxgml_ConvolutionModeEnumAttr
    : Dxgml_Attr<"ConvolutionModeEnum", "ConvolutionModeEnumAttr"> {
  let summary = "Convolution mode enum attribute";
  let parameters = (ins EnumParameter<Dxgml_ConvolutionModeEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
}

def Dxgml_ConvolutionDirectionEnumAttr
    : Dxgml_Attr<"ConvolutionDirectionEnum", "ConvolutionDirectionEnumAttr"> {
  let summary = "Convolution direction enum attribute";
  let parameters = (ins EnumParameter<Dxgml_ConvolutionDirectionEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
}

def Dxgml_DepthSpaceOrderEnumAttr
    : Dxgml_Attr<"DepthSpaceOrderEnum", "DepthSpaceOrderEnumAttr"> {
  let summary = "Depth/space order enum attribute";
  let parameters = (ins EnumParameter<Dxgml_DepthSpaceOrderEnum>:$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // DXGML_ATTRS
