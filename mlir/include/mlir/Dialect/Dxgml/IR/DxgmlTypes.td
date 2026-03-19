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
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Dxgml Simple Data Types
//===----------------------------------------------------------------------===//

// Integer Data Types
def Dxgml_Int2Type : Dxgml_Type<"Int2", "Int2"> {
  let summary = "2-bit integer type";
}

def Dxgml_Int4Type : Dxgml_Type<"Int4", "Int4"> {
  let summary = "4-bit integer type";
}

def Dxgml_Int8Type : Dxgml_Type<"Int8", "Int8"> {
  let summary = "8-bit integer type";
}

def Dxgml_Int16Type : Dxgml_Type<"Int16", "Int16"> {
  let summary = "16-bit integer type";
}

def Dxgml_Int32Type : Dxgml_Type<"Int32", "Int32"> {
  let summary = "32-bit integer type";
}

def Dxgml_Int64Type : Dxgml_Type<"Int64", "Int64"> {
  let summary = "64-bit integer type";
}

def Dxgml_BoolType : Dxgml_Type<"Bool", "Bool"> {
  let summary = "Boolean type";
}

def Dxgml_UInt2Type : Dxgml_Type<"UInt2", "Uint2"> {
  let summary = "2-bit unsigned integer type";
}

def Dxgml_UInt4Type : Dxgml_Type<"UInt4", "Uint4"> {
  let summary = "4-bit unsigned integer type";
}

def Dxgml_UInt8Type : Dxgml_Type<"UInt8", "Uint8"> {
  let summary = "8-bit unsigned integer type";
}

def Dxgml_UInt16Type : Dxgml_Type<"UInt16", "Uint16"> {
  let summary = "16-bit unsigned integer type";
}

def Dxgml_UInt32Type : Dxgml_Type<"UInt32", "Uint32"> {
  let summary = "32-bit unsigned integer type";
}

def Dxgml_UInt64Type : Dxgml_Type<"UInt64", "Uint64"> {
  let summary = "64-bit unsigned integer type";
}

// Floating Point Data Types
def Dxgml_Float16Type : Dxgml_Type<"Float16", "Float16"> {
  let summary = "16-bit floating point type";
}

def Dxgml_Float32Type : Dxgml_Type<"Float32", "Float32"> {
  let summary = "32-bit floating point type";
}

def Dxgml_Float64Type : Dxgml_Type<"Float64", "Float64"> {
  let summary = "64-bit floating point type";
}

def Dxgml_BFloat16Type : Dxgml_Type<"BFloat16", "BFloat16"> {
  let summary = "BFloat16 type";
}

def Dxgml_Float8E5M2Type : Dxgml_Type<"Float8E5M2", "Float8e5m2"> {
  let summary = "8-bit floating point E5M2";
}

def Dxgml_Float8E4M3Type : Dxgml_Type<"Float8E4M3", "Float8e4m3"> {
  let summary = "8-bit floating point E4M3";
}

def Dxgml_Float4E2M1FNType : Dxgml_Type<"Float4E2M1FN", "Float4e2m1fn"> {
  let summary = "4-bit floating point E2M1FN";
}

def Dxgml_Float4E3M0Type : Dxgml_Type<"Float4E3M0", "Float4e3m0"> {
  let summary = "4-bit floating point E3M0";
}

// Paralletized Data Types
def Dxgml_Nf4Type : Dxgml_Type<"NF4", "NF4"> {
  let summary = "NF4 palletized 4-bit type with scale and lookup table";
  let parameters = (ins
    "::mlir::Attribute":$scale,   // ConstantResourceAttr : Tensor<1xFloat32>
    "::mlir::Attribute":$lut      // ConstantResourceAttr : Tensor<16xFloat32>
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Dxgml Tensor Type
//===----------------------------------------------------------------------===//

// Constraint matching any DxGML scalar type
def AnyDxgmlScalarType : Type<
    Or<[
      CPred<"::llvm::isa<::mlir::dxgml::Int2Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Int4Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Int8Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Int16Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Int32Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Int64Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::BoolType>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::UInt2Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::UInt4Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::UInt8Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::UInt16Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::UInt32Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::UInt64Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Float16Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Float32Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Float64Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::BFloat16Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Float8E5M2Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Float8E4M3Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Float4E2M1FNType>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::Float4E3M0Type>($_self)">,
      CPred<"::llvm::isa<::mlir::dxgml::NF4Type>($_self)">,
    ]>,
    "any DxGML scalar type">;

def Dxgml_TensorType : Dxgml_Type<"Tensor", "Tensor", [ShapedTypeInterface], "::mlir::dxgml::DxgmlTensorBase"> {
  let mnemonic = "Tensor";
  let typeName = "DxGML.Tensor";
  let summary = "Multi-dimensional Tensor type";
  let description = [{ 
    The DxGML tensor represents a logical N-dimensional data values, and have a 
    known element type and a fixed rank with a list of dimensions. Each dimension 
    is either a static non-negative integer or be dynamically specified with "?".
  }];

  let parameters = (
    ins
    ArrayRefParameter<"int64_t", "sizes">:$optionalSizes,
    AnyDxgmlScalarType:$dtype
  );

  let builders = [
    TypeBuilder<
      (
        ins
        "::llvm::ArrayRef<int64_t>":$optionalSizes,
        "::mlir::Type":$dtype
      ),
      [{
        return $_get(context, optionalSizes, dtype);
      }]
    >
  ];

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // ShapedTypeInterface required methods

    /// Returns the element type (maps to the $dtype parameter).
    ::mlir::Type getElementType() const { return getDtype(); }

    /// DxGML tensors always have a known rank.
    bool hasRank() const { return true; }

    /// Returns the shape dimensions (maps to the $optionalSizes parameter).
    ::llvm::ArrayRef<int64_t> getShape() const { return getOptionalSizes(); }

    /// Returns a clone of this type with the given shape and element type.
    ::mlir::ShapedType cloneWith(::std::optional<::llvm::ArrayRef<int64_t>> shape,
                                  ::mlir::Type elementType) const {
      return TensorType::get(getContext(),
                             shape.value_or(getOptionalSizes()),
                             elementType);
    }
  }];

  string extraBaseClassDeclaration = [{
  }];
}

#endif // DXGML_TYPES
