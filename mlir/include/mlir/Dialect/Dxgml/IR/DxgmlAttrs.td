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

//===----------------------------------------------------------------------===//
// Integer Attribute
//===----------------------------------------------------------------------===//

def Dxgml_IntegerAttr : Dxgml_Attr<"Integer", "integer",
                                    [TypedAttrInterface]> {
  let summary = "Attribute holding an integer";
  let description = [{
    An integer attribute with a dxgml integer type.
    
    Example:
    ```mlir
    #dxgml.integer<1 : !dxgml.int64>
    #dxgml.integer<32 : !dxgml.int32>
    ```
  }];
  
  let parameters = (ins
    "Type":$type,
    "APInt":$value
  );
  
  let assemblyFormat = "`<` $value `:` $type `>`";
  
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "int64_t":$value), [{
      return $_get(type.getContext(), type, APInt(64, value));
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type, "const APInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];
  
  let extraClassDeclaration = [{
    int64_t getInt() const { return getValue().getSExtValue(); }
  }];
}

//===----------------------------------------------------------------------===//
// Float Attribute
//===----------------------------------------------------------------------===//

def Dxgml_FloatAttr : Dxgml_Attr<"Float", "float",
                                  [TypedAttrInterface]> {
  let summary = "Attribute holding a float";
  let description = [{
    A float attribute with a dxgml float type.
    
    Example:
    ```mlir
    #dxgml.float<1.0 : !dxgml.float32>
    #dxgml.float<3.14 : !dxgml.float16>
    ```
  }];
  
  let parameters = (ins
    "Type":$type,
    "APFloat":$value
  );
  
  let assemblyFormat = "`<` $value `:` $type `>`";
  
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "double":$value), [{
      return $_get(type.getContext(), type, APFloat(value));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Bool Attribute
//===----------------------------------------------------------------------===//

def Dxgml_BoolAttr : Dxgml_Attr<"Bool", "bool",
                                 [TypedAttrInterface]> {
  let summary = "Boolean attribute";
  let description = [{
    A boolean attribute wrapping DxGML BoolType.
    
    Example:
    ```mlir
    #dxgml.bool<true>
    #dxgml.bool<false>
    ```
  }];
  
  let parameters = (ins "bool":$value);
  let assemblyFormat = "`<` $value `>`";
  
  let builders = [
    AttrBuilder<(ins "bool":$value), [{
      return $_get($_ctxt, value);
    }]>
  ];
  
  let extraClassDeclaration = [{
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// String Attribute
//===----------------------------------------------------------------------===//

def Dxgml_StrAttr : Dxgml_Attr<"Str", "str",
                                [TypedAttrInterface]> {
  let summary = "String attribute";
  let description = [{
    A string attribute.
    
    Example:
    ```mlir
    #dxgml.str<"hello">
    ```
  }];
  
  let parameters = (ins
    "StringRef":$value,
    DefaultValuedParameter<"Type", "NoneType::get($_ctxt)">:$type
  );
  
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Dense Integer Elements Attribute
//===----------------------------------------------------------------------===//

def Dxgml_DenseIntegerElementsAttr : Dxgml_Attr<"DenseIntegerElements", 
                                                  "dense_integer_elements",
                                                  [TypedAttrInterface]> {
  let summary = "Dense array of integer values";
  let description = [{
    A dense integer elements attribute containing a densely packed array
    of integer values.
    
    Example:
    ```mlir
    #dxgml.dense_integer_elements<[1, 2, 3]> : !dxgml.tensor<3x!dxgml.int64>
    #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    ```
  }];
  
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$value,
    "Type":$type
  );
  
  let assemblyFormat = "`<` `[` $value `]` `>` `:` $type";
  
  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$value, "Type":$type), [{
      return $_get(type.getContext(), value, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Dense Float Elements Attribute
//===----------------------------------------------------------------------===//

def Dxgml_DenseFloatElementsAttr : Dxgml_Attr<"DenseFloatElements",
                                                "dense_float_elements",
                                                [TypedAttrInterface]> {
  let summary = "Dense array of float values";
  let description = [{
    A dense float elements attribute containing a densely packed array
    of floating-point values.
    
    Example:
    ```mlir
    #dxgml.dense_float_elements<[1.0, 2.0, 3.0]> : !dxgml.tensor<3x!dxgml.float32>
    ```
  }];
  
  let parameters = (ins
    ArrayRefParameter<"double">:$value,
    "Type":$type
  );
  
  let assemblyFormat = "`<` `[` $value `]` `>` `:` $type";
}

//===----------------------------------------------------------------------===//
// Constant Resource Attribute
//===----------------------------------------------------------------------===//

def Dxgml_ConstantResourceAttr : Dxgml_Attr<"ConstantResource",
                                              "constant_resource",
                                              [TypedAttrInterface]> {
  let summary = "Reference to a constant resource";
  let description = [{
    References a constant resource (weights, biases) by name.
    The actual data is stored externally.
    
    Example:
    ```mlir
    #dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>
    ```
  }];
  
  let parameters = (ins
    "StringAttr":$key,
    "Type":$type
  );
  
  let assemblyFormat = "`<` $key `:` $type `>`";
  
  let builders = [
    AttrBuilderWithInferredContext<(ins "StringRef":$key, "Type":$type), [{
      return $_get(type.getContext(), 
                   StringAttr::get(type.getContext(), key),
                   type);
    }]>
  ];
}

#endif // DXGML_ATTRS
