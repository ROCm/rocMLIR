//===- MIGraphXTypes.td - MIGraphX types definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIGraphX types.
//
//===----------------------------------------------------------------------===//

#ifndef MIGRAPHX_TYPES
#define MIGRAPHX_TYPES

include "mlir/Dialect/MIGraphX/IR/MIGraphXBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

class MIGraphX_Type<string name, string typeMnemonic, list<Trait> traits = [],
                    string baseCppClass = "::mlir::Type">
    : TypeDef<MIGraphX_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def MIGraphX_ShapedType : MIGraphX_Type<"MIXRShaped", "shaped", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface, ["cloneWith"]>]> {
  let mnemonic = "shaped";
  let summary = "MIGraphX IR's main tensor type";
  let description = [{
    The type of all values inside MIGraphX's IR, also known as a "shape".

    Note that this is named MIXRShapedType to avoid conflicting with
    the core ShapedType.

    This type has three main components:
    1. A `shape`, which is a set of int64_t lengths that represent the logical
      shape of the value. This does not necessarily correspond to its memory layout.
      This is also the shape of the type for the purposes of MLIR's shaped type
      interface.
    2. A set of `strides`, which is a set of integers of the same length as the
      shape which indicate how each dimension of the shape is mapped to memory.
      These strides are not required to be sorted. A stride of 0 indicates a
      dimension that is broadcasted and not present in the underlying value.
      (We say the strides are "contiguous" when the stlide of dimension `k`
      is equal to the products of the lengths of the dimensions from `k + 1` to
      `D`, the number of dimensions)
    3. An `elementType`, which is the element type of the given shaped value.

    This type is used to ensure that we can faithfully represent the MIGraphX
    IR in MLIR and so we don't fail to handle cases like the non-contiguous
    strides used in NHWC convolution.

  Examples:

  ```mlir
  // Canonical shaped tensor, no broadcasting
  !shaped<5x2xf32, 2x1>

  // The above, but transposed in memory
  !shaped<5x2xf32, 1x5>

  // A shaped tensor with a broadcast axis
  !shaped<5x2xf32, 5x0>

  // A scalar
  !shaped<1xi8, 0>

  // A scalar in a more MLIR-friendly form.
  !shaped<i8>
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    ArrayRefParameter<"int64_t">:$strides,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::llvm::ArrayRef<int64_t>":$shape,
      "::llvm::ArrayRef<int64_t>":$strides,
      "::mlir::Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, strides, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    using ShapedType::Trait<MIXRShapedType>::getElementTypeBitWidth;
    using ShapedType::Trait<MIXRShapedType>::getRank;
    using ShapedType::Trait<MIXRShapedType>::getNumElements;
    using ShapedType::Trait<MIXRShapedType>::isDynamicDim;
    using ShapedType::Trait<MIXRShapedType>::hasStaticShape;
    using ShapedType::Trait<MIXRShapedType>::getNumDynamicDims;
    using ShapedType::Trait<MIXRShapedType>::getDimSize;
    using ShapedType::Trait<MIXRShapedType>::getDynamicDimIndex;

    ::mlir::RankedTensorType asTensor() const;

    // Returns true if any stride is 0, indicating that a dimension of this
    // shaped value is broadcast.
    bool hasBroadcast() const;

    // Return the set of dimensions in this type which are broadcast.
    void getBroadcastDims(::llvm::SmallVectorImpl<uint32_t>& result) const;

    // Clone with the given sizes, strides, and element type if present.
    MIXRShapedType cloneWith(std::optional<::llvm::ArrayRef<int64_t>> shape,
                     std::optional<::llvm::ArrayRef<int64_t>> strides,
                     ::mlir::Type elementType) const;

    // Return the tensor type that represent the momy layout specified by this shape.
    // If there is no such tensor, returns nullptr.
    ::mlir::RankedTensorType asMemoryLayoutTensor() const;

    /// Return the permutation that describes how the strides have been permuted
    /// from what they would be if the tensor were in standard shape. This function
    /// is equivalent to MIGraphX's find_Permutation . Applying the inverse
    /// of the permutation returned here to the strides will put them in standard
    /// shape.
    ///
    /// The permutation is an array in the tosa form, that is, given sequences
    /// in and out, we map in[ret[x]] to out[x].
    void getStridePermutation(::llvm::SmallVectorImpl<int64_t>& ret) const;
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

// As with the C++ class name, we use "MIXR" here as a short prefix to save
// on typing.
class MIXRShapedOf<list<Type> allowedTypes, string summary = "!migraphx.shaped"> :
  ShapedContainerType<allowedTypes, MIGraphX_ShapedType.predicate, summary,
  "::mlir::migraphx::MIXRShapedType">;

def AnyMIXRShaped : MIXRShapedOf<[AnyType]>;

class Strides<string name> :
    StrFunc<"::llvm::cast<::mlir::migraphx::MIXRShapedType>($" # name # ".getType()).getStrides()">;

class AllStridesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, Strides<"_self">.result, "strides">;

class AllLayoutsMatch<list<string> names> :
  TraitList<[AllShapesMatch<names>, AllStridesMatch<names>]>;

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
