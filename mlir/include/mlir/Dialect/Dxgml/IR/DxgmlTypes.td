//===- DxgmlTypes.td - Dxgml type definitions -------------*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Dxgml dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef DXGML_TYPES
#define DXGML_TYPES

include "mlir/Dialect/Dxgml/IR/DxgmlBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Dxgml Scalar Types
//===----------------------------------------------------------------------===//

// Float types
def Dxgml_Float16Type : Dxgml_Type<"Float16", "float16"> {
  let summary = "16-bit floating point type";
}

def Dxgml_Float32Type : Dxgml_Type<"Float32", "float32"> {
  let summary = "32-bit floating point type";
}

def Dxgml_Float64Type : Dxgml_Type<"Float64", "float64"> {
  let summary = "64-bit floating point type";
}

def Dxgml_BFloat16Type : Dxgml_Type<"BFloat16", "bfloat16"> {
  let summary = "BFloat16 type";
}

// Integer types
def Dxgml_Int2Type : Dxgml_Type<"Int2", "int2"> {
  let summary = "2-bit integer type";
}

def Dxgml_Int4Type : Dxgml_Type<"Int4", "int4"> {
  let summary = "4-bit integer type";
}

def Dxgml_Int8Type : Dxgml_Type<"Int8", "int8"> {
  let summary = "8-bit integer type";
}

def Dxgml_Int16Type : Dxgml_Type<"Int16", "int16"> {
  let summary = "16-bit integer type";
}

def Dxgml_Int32Type : Dxgml_Type<"Int32", "int32"> {
  let summary = "32-bit integer type";
}

def Dxgml_Int64Type : Dxgml_Type<"Int64", "int64"> {
  let summary = "64-bit integer type";
}

// Unsigned integer types
def Dxgml_UInt2Type : Dxgml_Type<"UInt2", "uint2"> {
  let summary = "2-bit unsigned integer type";
}

def Dxgml_UInt4Type : Dxgml_Type<"UInt4", "uint4"> {
  let summary = "4-bit unsigned integer type";
}

def Dxgml_UInt8Type : Dxgml_Type<"UInt8", "uint8"> {
  let summary = "8-bit unsigned integer type";
}

def Dxgml_UInt16Type : Dxgml_Type<"UInt16", "uint16"> {
  let summary = "16-bit unsigned integer type";
}

def Dxgml_UInt32Type : Dxgml_Type<"UInt32", "uint32"> {
  let summary = "32-bit unsigned integer type";
}

def Dxgml_UInt64Type : Dxgml_Type<"UInt64", "uint64"> {
  let summary = "64-bit unsigned integer type";
}

// Special types
def Dxgml_BoolType : Dxgml_Type<"Bool", "bool"> {
  let summary = "Boolean type";
}

def Dxgml_NullType : Dxgml_Type<"Null", "null"> {
  let summary = "Null type";
}

// Additional float types from spec
def Dxgml_Float8E4M3FNType : Dxgml_Type<"Float8E4M3FN", "float8e4m3fn"> {
  let summary = "8-bit floating point E4M3FN";
}

def Dxgml_Float8E4M3FNUZType : Dxgml_Type<"Float8E4M3FNUZ", "float8e4m3fnuz"> {
  let summary = "8-bit floating point E4M3FNUZ";
}

def Dxgml_Float8E5M2Type : Dxgml_Type<"Float8E5M2", "float8e5m2"> {
  let summary = "8-bit floating point E5M2";
}

def Dxgml_Float8E5M2FNUZType : Dxgml_Type<"Float8E5M2FNUZ", "float8e5m2fnuz"> {
  let summary = "8-bit floating point E5M2FNUZ";
}

def Dxgml_Float8E4M3Type : Dxgml_Type<"Float8E4M3", "float8e4m3"> {
  let summary = "8-bit floating point E4M3";
}

def Dxgml_Float8E3M4Type : Dxgml_Type<"Float8E3M4", "float8e3m4"> {
  let summary = "8-bit floating point E3M4";
}

def Dxgml_Float8E4M3B11FNUZType : Dxgml_Type<"Float8E4M3B11FNUZ", "float8e4m3b11fnuz"> {
  let summary = "8-bit floating point E4M3B11FNUZ";
}

def Dxgml_Float8E8M0FNUType : Dxgml_Type<"Float8E8M0FNU", "float8e8m0fnu"> {
  let summary = "8-bit floating point E8M0FNU";
}

def Dxgml_Float4E2M1FNType : Dxgml_Type<"Float4E2M1FN", "float4e2m1fn"> {
  let summary = "4-bit floating point E2M1FN";
}

def Dxgml_Float6E2M3FNType : Dxgml_Type<"Float6E2M3FN", "float6e2m3fn"> {
  let summary = "6-bit floating point E2M3FN";
}

def Dxgml_Float6E3M2FNType : Dxgml_Type<"Float6E3M2FN", "float6e3m2fn"> {
  let summary = "6-bit floating point E3M2FN";
}

//===----------------------------------------------------------------------===//
// Dxgml Tensor Type
//===----------------------------------------------------------------------===//

def Dxgml_DxgmlTensorType : Dxgml_Type<"DxgmlTensor", "tensor"> {
  let summary = "Multi-dimensional tensor type";
  let description = [{
    Represents a multi-dimensional array with fixed shape and element type.
    
    Syntax: `!dxgml.tensor<DxDx...xElementType>`
    
    Example:
    ```mlir
    !dxgml.tensor<1x32x1080x1920xf16>
    !dxgml.tensor<32xi64>
    ```
  }];
  
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  
  let assemblyFormat = "`<` custom<DxgmlTensorType>($shape, $elementType) `>`";
  
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];
  
  let extraClassDeclaration = [{
    /// Get the number of dimensions
    unsigned getRank() const { return getShape().size(); }
    
    /// Get the number of elements
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape())
        num *= dim;
      return num;
    }
    
    /// Check if all dimensions are static
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), [](int64_t dim) { return dim < 0; });
    }
  }];
}

#endif // DXGML_TYPES
