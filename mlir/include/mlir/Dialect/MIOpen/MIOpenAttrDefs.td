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

def TransformType : MIOpen_I32Enum<"TransformType",
    "The operation type for a coordinate transformation",
    [PassThrough, Pad, Slice, Embed, Unmerge, Merge, Unfold, AddDim]>;

/// InMemoryDataOperation

def DataOperation_Set : I32EnumAttrCase<"Set", 0>;
def DataOperation_AtomicAdd : I32EnumAttrCase<"AtomicAdd", 1>;

def InMemoryDataOperation : MIOpen_I32Enum<"InMemoryDataOperation",
    "What operation should be used to write data to memory",
    [DataOperation_Set, DataOperation_AtomicAdd]>;

/// BwdPaddingKernelInfo
def BwdPadding_NA : BitEnumAttrCase<"NA", 0x0000>;
def BwdPadding_StrideTwo : BitEnumAttrCase<"StrideTwo", 0x0001>;
def BwdPadding_Xdlops : BitEnumAttrCase<"Xdlops", 0x0002>;
def BwdPadding_NCHW : BitEnumAttrCase<"isNCHW", 0x0004>;
def BwdPadding_PadM : BitEnumAttrCase<"PadM", 0x0008>;
def BwdPadding_PadN : BitEnumAttrCase<"PadN", 0x0010>;

def BwdPaddingKernelInfo : BitEnumAttr<"BwdPaddingKernelInfo",
    "Information about what workarounds will be needed in out of bounds checks for backwards data kernels with stride > 1",
    [BwdPadding_NA, BwdPadding_StrideTwo, BwdPadding_Xdlops,
    BwdPadding_NCHW, BwdPadding_PadM, BwdPadding_PadN]> {

    let cppNamespace = "::mlir::miopen";
    let symbolToStringFnName = "getBitsForBwdPaddingKernelInfo";
    let stringToSymbolFnName = "getBwdPaddingKernelInfoForBits";
}

/// TransformAttr
def MIOpen_TransformAttr : MIOpen_Attr<"Transform"> {
    let mnemonic = "transform";

    let description = [{
        A coordinate transform.

        TODO: Move the documentation into here.
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
    // TODO: do we need these, actually?
    ArrayRefParameter<"int64_t", "The shape of the transform input">:$upperBounds,
    ArrayRefParameter<"int64_t", "The shape of the transform output">:$lowerBounds
    );

    let assemblyFormat = [{
        `<` $map `by` `[` $ops `]` `bounds` `=` `[` $upperBounds `]` `->` `[` $lowerBounds `]` `>`
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
    "int64_t":$extraN,
    "::mlir::miopen::BwdPaddingKernelInfo":$bwdPaddingInfo);

    let assemblyFormat = "`<` struct($extraM, $extraK, $extraN, $bwdPaddingInfo) `>`";

    let extraClassDeclaration = [{
      bool hasPadding() {
          return getExtraM() > 0 || getExtraK() > 0 || getExtraN() > 0;
      }
    }];
}

#endif
