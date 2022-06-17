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
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//TODO: submit this to upstream
class ArrayRefOfStringRefParameter<string desc = ""> :
    AttrOrTypeParameter<"::llvm::ArrayRef<::llvm::StringRef>", desc> {
  let allocator = [{ {
    ::llvm::SmallVector<::llvm::StringRef> tmpFields;
    size_t len = $_self.size();
    tmpFields.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      tmpFields.push_back($_allocator.copyInto($_self[i]));
    }
    $_dst = $_allocator.copyInto(::llvm::ArrayRef<StringRef>(tmpFields));
  } }];
  let cppStorageType = "::llvm::SmallVector<StringRef>";
}

def IndexArrayAttr : TypedArrayAttrBase<IndexAttr,
                                      "Index integer array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

class MIOpen_I32Enum<string name, string summary, list<I32EnumAttrCase> cases>
  : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::miopen";

  let genSpecializedAttr = 0;
  let stringToSymbolFnName = "get" # name # "ForName";
  let symbolToStringFnName = "getNameFor" # name;
}

class MIOpen_Attr<string name, list<Trait> traits = []> :
    AttrDef<MIOpen_Dialect, name, traits>;

/// ConvOpType
def ConvOp2DType : I32EnumAttrCase<"Fwd", 0, "conv2d">;
def ConvOp2DBwdDataType : I32EnumAttrCase<"BwdData", 1, "conv2d_bwd_data">;
def ConvOp2DBwdWeightType : I32EnumAttrCase<"BwdWeight", 2, "conv2d_bwd_weight">;

def ConvOpType : MIOpen_I32Enum<"ConvOpType", "The type of a 2D convolution operation",
  [ConvOp2DType, ConvOp2DBwdDataType, ConvOp2DBwdWeightType]>;

/// TransformType
def PassThrough : I32EnumAttrCase<"PassThrough", 0>;
def Pad : I32EnumAttrCase<"Pad", 1>;
def Slice : I32EnumAttrCase<"Slice", 2>;
def Embed : I32EnumAttrCase<"Embed", 3>;
def Unmerge : I32EnumAttrCase<"Unmerge", 4>;
def Merge : I32EnumAttrCase<"Merge", 5>;
def Unfold : I32EnumAttrCase<"Unfold", 6>;
def AddDim : I32EnumAttrCase<"AddDim", 7>;
def Broadcast : I32EnumAttrCase<"Broadcast", 8>;

def TransformType : MIOpen_I32Enum<"TransformType",
    "The operation type for a coordinate transformation",
    [PassThrough, Pad, Slice, Embed, Unmerge, Merge, Unfold, AddDim, Broadcast]>;

/// StoreMethod

def StoreMethod_Set : I32EnumAttrCase<"Set", 0, "set">;
def StoreMethod_AtomicAdd : I32EnumAttrCase<"AtomicAdd", 1, "atomic_add">;

def StoreMethod : MIOpen_I32Enum<"StoreMethod",
    "What operation should be used to write data to memory",
    [StoreMethod_Set, StoreMethod_AtomicAdd]>;

def StoreMethodAttr : EnumAttr<MIOpen_Dialect, StoreMethod, "StoreMethod">;

/// TransformAttr
def MIOpen_TransformAttr : MIOpen_Attr<"Transform"> {
    let mnemonic = "transform";

    let description = [{
        A coordinate transformation, that is, a component of a transform_map.

        A transform_map is a way to describe affine maps between coordinate spaces
        that easily allows one to "take their derivative" when emitting hot unrolled
         loops and that simplifies the process of describing these mappings.

        Each transform produces some number of upper dimensions from some
        number of lower dimensions. A collection of transformation that covers a set
        of upper and lower dimensions can be packaged into a transform_map,
        at which point they specify an affine map between coordinate spaces.

        Each transform is specified as an attribute with the form
        ```
          #miopen.transform<Type{param1, param2} ["upper1", "upper2"] at [dim1, dim2] -> ["lower1", "lower2"] at [dim3, dim4]>
        ```
        where the params block may be omitted for transforms that don't take parameters.

        The available transform types are
        - PassThrough, which maps N lower dimensions to N upper dimensions
            without changing the coordinates in those dimensions
        - Pad{l1, r1, l2, r2, ..., lN, rN} Adds padding to N dimensions in the
          transformed upper output. The parameters are the amount of padding to
          add to the left and right of each dimension, concatenated
        - Slice{b1, e1, b2, e2, ... bN, eN} Takes a slice of N dimensions
          creating, for each upper dimension i, a dimension of size e_i - b_i where 0
          in the upper layer maps to b_i in the lower layer
        - Embed{c1, c2, ... cN} - Creates N upper dimensions from 1 lower dimension
          with the map (d1, d2, ..., dN) -> (d1 * C1 + d2 * C2 + ... dN * cN)
        - Unmerge{c1, c2, ..., cN} - Creates N upper dimensions as in Embed,
          but uses the map (d1, d2, ..., dN) -> (dN + cN * (d_{N - 1} * (... d1)))
        - Merge{c1, c2, ..., cN} - Combines N lower dimensions into 1 upper dimension
          with the parameters holding the sizes of the lower dimensions.
          This has the effect of concatentating the dimensions if they are consecutive
        - Unfold{c1, ..., cN} - Has the semantics of merge, but asserts that the lower
          dimensions are consecutive so that the derivative operation does not need
          to perform carry checks on indices.
        - AddDim{size} - Adds one upper dimension of the given size that does not
          correspond to any lower dimension.
        - Broadcast{c1, c2, ..., cN} - Broadcast lower dimensions of size 1 to
          upper dimensions.
    }];

    let parameters = (ins
    AttrParameter<"::mlir::miopen::TransformType",
        "The type of transformation to perform">:$type,
    ArrayRefParameter<"int64_t",
        "The parameters to the transformation, if any">:$params,
    ArrayRefOfStringRefParameter<
        "The names of the upper dimensions (input arguments) of this transform">:$upperNames,
    ArrayRefParameter<"uint32_t",
        "The indices of the upper dimensions (input arguments) of this transform">:$upperDims,
    ArrayRefOfStringRefParameter<
        "The names of the lower dimensions (outputs) of this transformation">:$lowerNames,
    ArrayRefParameter<"uint32_t",
        "The indices of the lower dimensions (outputs) of this transformation">:$lowerDims
    );

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

/// TransformsAttr
def MIOpen_TransformMapAttr : MIOpen_Attr<"TransformMap"> {
    let mnemonic = "transform_map";
    let description = [{
        A package of a series of coordinate transformation descriptions
        that completely defines how to map from some upper dimensions P
        to some other lower dimensions Q.

        Comes packaged with an affine map describing the transformation and upper and lowr bounds;
    }];

    let parameters = (ins
    ArrayRefParameter<"::mlir::miopen::TransformAttr",
        "The individual coordinate transformations">:$ops,
    AttrParameter<"AffineMapAttr", "The composed map describing the `ops`">:$map,
    // Thes aren't strictly needed and can probably be removed if we're willing to
    // pay an ergonomics penalty.
    ArrayRefParameter<"int64_t", "The shape of the transform input">:$upperBounds,
    ArrayRefParameter<"int64_t", "The shape of the transform output">:$lowerBounds
    );

    let assemblyFormat = [{
        `<` $map `by` ` ` `[` $ops `]` `bounds` `=` `[` $upperBounds `]` `->` `[` $lowerBounds `]` `>`
    }];
    let genVerifyDecl = 1;
}

/// PaddingInfoAttr
def MIOpen_PaddingInfoAttr : MIOpen_Attr<"PaddingInfo"> {
    let mnemonic = "padding_info";

    let description = [{
        A structure packaging information about what padding has been
        applied to the M, K, and N dimensions of a matrix multiplication,
        as well as information about what backward padding kernel workarounds,
        if any, will be needed.
    }];
    let parameters =
    (ins
    "int64_t":$extraM,
    "int64_t":$extraK,
    "int64_t":$extraN);

    let assemblyFormat = "`<` struct($extraM, $extraK, $extraN) `>`";

    let extraClassDeclaration = [{
      bool hasPadding() {
          return getExtraM() > 0 || getExtraK() > 0 || getExtraN() > 0;
      }
    }];
}

#endif
