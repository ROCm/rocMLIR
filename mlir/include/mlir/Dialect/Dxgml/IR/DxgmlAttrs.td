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
    APIntParameter<"">:$value
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
    A float attribute with a dxgml float type. Behavior mirrors builtin FloatAttr:
    missing float type defaults to f64.

    Example:
    ```mlir
    #dxgml.float<1.0 : !dxgml.float32>
    #dxgml.float<3.14>
    ```
  }];

  let parameters = (ins
    "Type":$type,
    APFloatParameter<"">:$value
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "double":$value), [{
      return $_get(type.getContext(), type, APFloat(value));
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type, "const APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let extraClassDeclaration = [{
    double getDouble() const { return getValue().convertToDouble(); }
  }];
}


//===----------------------------------------------------------------------===//
// Bool Attribute
//===----------------------------------------------------------------------===//

def Dxgml_BoolAttr : Dxgml_Attr<"Bool", "bool"> {
  let summary = "Boolean attribute wrapping DxGMLBoolType";
  let description = [{
    This attribute represents a constant boolean value (true or false)
    whose type is the dialect's Bool type.

    Example:
    ```mlir
    #dxgml.bool<true>
    #dxgml.bool<false>
    ```
  }];

  let parameters = (ins
    "bool":$value
  );

  let assemblyFormat = "`<` $value `>`";
}


//===----------------------------------------------------------------------===//
// String Attribute
//===----------------------------------------------------------------------===//

def Dxgml_StrAttr : Dxgml_Attr<"Str", "str"> {
  let summary = "An Attribute containing a string";
  let description = [{
    A string attribute that represents a string literal value.

    Syntax:
    ```
    string-attribute ::= string-literal (`:` type)?
    ```

    Example:
    ```mlir
    "An important string"
    ```
  }];

  let parameters = (ins
    StringRefParameter<"">:$value
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
  let summary = "Dense multi-dimensional array of float values";
  let description = [{
    A dense float elements attribute containing a densely packed array
    of float values (stored as bit-cast int64 for hashing).

    Example:
    ```mlir
    #dxgml.dense_float_elements<[1.0, 2.0, 3.0]> : !dxgml.tensor<3x!dxgml.float32>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$raw_bits,
    "Type":$type
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    SmallVector<double> getValues() const {
      SmallVector<double> vals;
      for (int64_t bits : getRawBits()) {
        double d;
        memcpy(&d, &bits, sizeof(d));
        vals.push_back(d);
      }
      return vals;
    }

    static DenseFloatElementsAttr get(MLIRContext *ctx,
                                      ArrayRef<double> values, Type type) {
      SmallVector<int64_t> bits;
      for (double d : values) {
        int64_t b;
        memcpy(&b, &d, sizeof(b));
        bits.push_back(b);
      }
      return get(ctx, bits, type);
    }
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<double>":$values, "Type":$type), [{
      SmallVector<int64_t> bits;
      for (double d : values) {
        int64_t b;
        memcpy(&b, &d, sizeof(b));
        bits.push_back(b);
      }
      return $_get(type.getContext(), bits, type);
    }]>
  ];
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

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "StringRef":$key, "Type":$type), [{
      return $_get(type.getContext(),
                   StringAttr::get(type.getContext(), key),
                   type);
    }]>
  ];
}

#endif // DXGML_ATTRS
