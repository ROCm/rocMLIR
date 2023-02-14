//===- RockAttrDefs.td - Rock MLIR Attributes ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_ATTRS
#define ROCK_ATTRS

include "mlir/Dialect/Rock/IR/RockBase.td"
include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

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

class Rock_I32Enum<string name, string summary, list<I32EnumAttrCase> cases>
  : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::rock";

  let genSpecializedAttr = 0;
  let stringToSymbolFnName = "get" # name # "ForName";
  let symbolToStringFnName = "getNameFor" # name;
}

class Rock_Attr<string name, list<Trait> traits = []> :
    AttrDef<Rock_Dialect, name, traits>;

/// ConvOpType
def ConvOp2DType : I32EnumAttrCase<"Fwd", 0, "conv2d">;
def ConvOp2DBwdDataType : I32EnumAttrCase<"BwdData", 1, "conv2d_bwd_data">;
def ConvOp2DBwdWeightType : I32EnumAttrCase<"BwdWeight", 2, "conv2d_bwd_weight">;

def ConvOpType : Rock_I32Enum<"ConvOpType", "The type of a 2D convolution operation",
  [ConvOp2DType, ConvOp2DBwdDataType, ConvOp2DBwdWeightType]>;

/// Kerneltype
def KernelTypeConv2D : I32EnumAttrCase<"Conv2D", 0>;
def KernelTypeConv2DBwdData : I32EnumAttrCase<"Conv2DBwdData", 1>;
def KernelTypeConv2DBwdWeight : I32EnumAttrCase<"Conv2DBwdWeight", 2>;
def KernelTypeGemm : I32EnumAttrCase<"Gemm", 3>;

def KernelType : Rock_I32Enum<"KernelType", "Any of the possible types of a rock kernel",
  [KernelTypeConv2D, KernelTypeConv2DBwdData,
    KernelTypeConv2DBwdWeight, KernelTypeGemm]>;

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
def ConstDim : I32EnumAttrCase<"ConstDim", 9>;

def TransformType : Rock_I32Enum<"TransformType",
    "The operation type for a coordinate transformation",
    [PassThrough, Pad, Slice, Embed, Unmerge, Merge, Unfold, AddDim, Broadcast,
     ConstDim]>;

/// StoreMethod

def StoreMethod_Set : I32EnumAttrCase<"Set", 0, "set">;
def StoreMethod_AtomicAdd : I32EnumAttrCase<"AtomicAdd", 1, "atomic_add">;

def StoreMethod : Rock_I32Enum<"StoreMethod",
    "What operation should be used to write data to memory",
    [StoreMethod_Set, StoreMethod_AtomicAdd]>;

def StoreMethodAttr : EnumAttr<Rock_Dialect, StoreMethod, "StoreMethod">;

/// ReduceMethod

def ReduceMethod_Sum : I32EnumAttrCase<"Sum", 0, "sum">;

def ReduceMethod : Rock_I32Enum<"ReduceMethod",
    "What operation should be used for the reduction",
    [ReduceMethod_Sum]>;

def ReduceMethodAttr : EnumAttr<Rock_Dialect, ReduceMethod, "ReduceMethod">;

/// TransformAttr
def Rock_TransformAttr : Rock_Attr<"Transform"> {
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
          #rock.transform<Type{param1, param2} ["upper1", "upper2"] at [dim1, dim2] -> ["lower1", "lower2"] at [dim3, dim4]>
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
    AttrParameter<"::mlir::rock::TransformType",
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
def Rock_TransformMapAttr : Rock_Attr<"TransformMap"> {
    let mnemonic = "transform_map";
    let description = [{
        A package of a series of coordinate transformation descriptions
        that completely defines how to map from some upper dimensions P
        to some other lower dimensions Q.

        Comes packaged with an affine map describing the transformation and upper and lowr bounds;
    }];

    let parameters = (ins
    ArrayRefParameter<"::mlir::rock::TransformAttr",
        "The individual coordinate transformations">:$ops,
    AttrParameter<"AffineMapAttr", "The composed map describing the `ops`">:$map,
    // Thes aren't strictly needed and can probably be removed if we're willing to
    // pay an ergonomics penalty.
    DefaultValuedParameter<"DenseI64ArrayAttr", "$_builder.getDenseI64ArrayAttr({})">:$upperBounds,
    DefaultValuedParameter<"DenseI64ArrayAttr", "$_builder.getDenseI64ArrayAttr({})">:$lowerBounds
    );

    let assemblyFormat = [{
                             `<` $map `by` ` ` `[` $ops `]` `bounds` `=` ($upperBounds^) : (`[` `]`)? `->` $lowerBounds `>`
    }];
    let genVerifyDecl = 1;
}

def Rock_GeneralGemmParamsAttr : Rock_Attr<"GeneralGemmParams", [RockTuningParamAttrInterface]> {
  let mnemonic = "general_gemm_params";
  let description = [{
    The tuning parameters for a general (no special instructions) matrix
    multiplication.

    - blockSize: Optional parameter to specify the block size of the kernel
    - kPerBlock: The number of values of k to process during each main loop
      iteration within a workgroup
    - mPerBlock: The number of values of m to process in each workgroup
    - nPerBlock: The number of values of n to process in each workgroup
    - kPerThread: The number of values of k to process as a unit on each thread
    - mPerThread: The number of values of m to process as a unit on each thread
    - nPerThread: The number of values of n to process as a unit on each thread
    - kpack: The number of values of k to pack contiguously into the shared buffer
  }];
  let parameters = (ins
    "uint32_t":$blockSize,
    "int64_t":$kPerBlock,
    "int64_t":$mPerBlock,
    "int64_t":$nPerBlock,
    "int64_t":$kPerThread,
    "int64_t":$mPerThread,
    "int64_t":$nPerThread,
    "int64_t":$kpack
  );

  let extraClassDeclaration = [{
    void getPerfConfigStr(std::string &perfStr){
      perfStr =
       (Twine(getBlockSize()) + ","
      + Twine(getMPerBlock()) + ","
      + Twine(getNPerBlock()) + ","
      + Twine(getKPerBlock()) + ","
      + Twine(getMPerThread()) + ","
      + Twine(getNPerThread())).str();
    }
  }];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

def Rock_XdlopsGemmParamsAttr : Rock_Attr<"XdlopsGemmParams", [RockTuningParamAttrInterface]> {
  let mnemonic = "xdlops_gemm_params";
  let description = [{
    The tuning parameters for an xdlops-based matrix multiplication.

    TODO: Document these once they've settled down
  }];
  let parameters = (ins
    "int64_t":$kPerBlock,
    "int64_t":$mPerBlock,
    "int64_t":$nPerBlock,
    "int64_t":$kpack,
    "int64_t":$mPerWave,
    "int64_t":$nPerWave,
    "bool":$forceUnroll
  );

  let extraClassDeclaration = [{
    void getPerfConfigStr(std::string &perfStr){
      perfStr =
         (Twine(getMPerBlock()) + ","
        + Twine(getNPerBlock()) + ","
        + Twine(getKPerBlock()) + ","
        + Twine(getMPerWave()) + ","
        + Twine(getNPerWave()) + ","
        + Twine(getKpack()) + ","
        + Twine(getForceUnroll()) + ","
          + "1") /* *ThreadCopyMore* */
        .str();
    }
  }];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

// GEMM Features
def Rock_GemmFeatures_None : I32BitEnumAttrCaseNone<"none">;
def Rock_GemmFeatures_Mfma : I32BitEnumAttrCaseBit<"mfma", 0>;
def Rock_GemmFeatures_Dot : I32BitEnumAttrCaseBit<"dot", 1>;
def Rock_GemmFeatures_AtomicAdd : I32BitEnumAttrCaseBit<"atomic_add", 2>;

def Rock_GemmFeatures : I32BitEnumAttr<"GemmFeatures",
    "Features that can be enabled on GEMM-based operations",
    [Rock_GemmFeatures_None, Rock_GemmFeatures_Mfma,
     Rock_GemmFeatures_Dot, Rock_GemmFeatures_AtomicAdd]> {
  let cppNamespace = "::mlir::rock";
  let genSpecializedAttr = 0;
}

def Rock_GemmFeaturesAttr : EnumAttr<Rock_Dialect, Rock_GemmFeatures, "GemmFeatures">;
#endif
