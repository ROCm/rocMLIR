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
include "mlir/Dialect/Rock/IR/RockGemmFeaturesInterface.td"
include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.td"
include "mlir/Dialect/Rock/IR/RockAccelTuningParamAttrInterface.td"
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
def ConvOpType : I32EnumAttrCase<"Fwd", 0, "conv">;
def ConvOpBwdDataType : I32EnumAttrCase<"BwdData", 1, "conv_bwd_data">;
def ConvOpBwdWeightType : I32EnumAttrCase<"BwdWeight", 2, "conv_bwd_weight">;

def ConvOpTypes : Rock_I32Enum<"ConvOpType", "The type of a convolution operation",
  [ConvOpType, ConvOpBwdDataType, ConvOpBwdWeightType]>;

/// Kerneltype
def KernelTypeConv : I32EnumAttrCase<"Conv", 0>;
def KernelTypeConvBwdData : I32EnumAttrCase<"ConvBwdData", 1>;
def KernelTypeConvBwdWeight : I32EnumAttrCase<"ConvBwdWeight", 2>;
def KernelTypeGemm : I32EnumAttrCase<"Gemm", 3>;
def KernelTypeAttention : I32EnumAttrCase<"Attention", 4>;
def KernelTypeGemmElementwiseGemm : I32EnumAttrCase<"GemmElementwiseGemm", 5>;
def KernelTypeConvElementwiseGemm : I32EnumAttrCase<"ConvElementwiseGemm", 6>;

def KernelType
    : Rock_I32Enum<"KernelType", "Any of the possible types of a rock kernel",
                   [KernelTypeConv, KernelTypeConvBwdData,
                    KernelTypeConvBwdWeight, KernelTypeGemm,
                    KernelTypeAttention, KernelTypeGemmElementwiseGemm,
                    KernelTypeConvElementwiseGemm]>;

/// TransformType
def PassThrough : I32EnumAttrCase<"PassThrough", 0>;
def Pad : I32EnumAttrCase<"Pad", 1>;
def Slice : I32EnumAttrCase<"Slice", 2>;
def Embed : I32EnumAttrCase<"Embed", 3>;
def Unmerge : I32EnumAttrCase<"Unmerge", 4>;
def Merge : I32EnumAttrCase<"Merge", 5>;
def AddDim : I32EnumAttrCase<"AddDim", 7>;
def Broadcast : I32EnumAttrCase<"Broadcast", 8>;
def ConstDim : I32EnumAttrCase<"ConstDim", 9>;

def TransformType : Rock_I32Enum<"TransformType",
    "The operation type for a coordinate transformation",
    [PassThrough, Pad, Slice, Embed, Unmerge, Merge, AddDim, Broadcast,
     ConstDim]>;

/// StoreMethod

def StoreMethod_Set : I32EnumAttrCase<"Set", 0, "set">;
def StoreMethod_AtomicAdd : I32EnumAttrCase<"AtomicAdd", 1, "atomic_add">;
def StoreMethod_AtomicMax : I32EnumAttrCase<"AtomicMax", 2, "atomic_max">;

def StoreMethod : Rock_I32Enum<"StoreMethod",
    "What operation should be used to write data to memory",
    [StoreMethod_Set, StoreMethod_AtomicAdd, StoreMethod_AtomicMax]>;

def StoreMethodAttr : EnumAttr<Rock_Dialect, StoreMethod, "StoreMethod">;

/// ReduceMethod

def ReduceMethod_Sum : I32EnumAttrCase<"Sum", 0, "sum">;
def ReduceMethod_Max : I32EnumAttrCase<"Max", 1, "max">;

def ReduceMethod : Rock_I32Enum<"ReduceMethod",
    "What operation should be used for the reduction",
    [ReduceMethod_Sum, ReduceMethod_Max]>;

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
        - AddDim{size} - Adds one upper dimension of the given size that does not
          correspond to any lower dimension.
        - Broadcast{c1, c2, ..., cN} - Broadcast lower dimensions of size
          c1, c2, ... cN to upper dimensions whose size isn't (currently) recorded
          in the parameters. Note that the upper dimension's length doesn't have to
          evenly divide into the lower one.
        - ConstDim{c1, l1, c2, l2, .. cN, lN} an operator that sets the lower
          dimensions of length l1, ... lN to the values c1, ..., cN. It takes no
          input arguments.
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

    let builders = [
      AttrBuilderWithInferredContext<
        (ins "ArrayRef<::mlir::rock::TransformAttr>":$transforms,
            "ArrayRef<int64_t>":$upperBounds,
            "ArrayRef<int64_t>":$lowerBounds)>,
    ];

    let assemblyFormat = [{
        `<` $map `by` ` ` `[` $ops `]` `bounds` `=` ($upperBounds^) : (`[` `]`)? `->` ($lowerBounds^) : (`[` `]`)? `>`
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
    - splitKFactor:  Split-k factor for the Split-k GEMM algorithm
    - scheduleVersion: Param to select GEMM schedule
    - outputSwizzle: Whether to enable/disable output swizzle or use heuristics 
  }];
  let parameters = (ins
    "uint32_t":$blockSize,
    "int64_t":$kPerBlock,
    "int64_t":$mPerBlock,
    "int64_t":$nPerBlock,
    "int64_t":$kPerThread,
    "int64_t":$mPerThread,
    "int64_t":$nPerThread,
    "int64_t":$kpack,
    "int64_t":$splitKFactor,
    "int64_t":$scheduleVersion,
    "int64_t":$outputSwizzle
  );

  let extraClassDeclaration = [{
    void getPerfConfigStr(::llvm::SmallVectorImpl<char> &perfStr) {
      ("v3:" + Twine(getBlockSize()) + ","
      + Twine(getMPerBlock()) + ","
      + Twine(getNPerBlock()) + ","
      + Twine(getKPerBlock()) + ","
      + Twine(getMPerThread()) + ","
      + Twine(getNPerThread()) + ","
      + Twine(getSplitKFactor()) + ","
      + Twine(getScheduleVersion()) + ","
      + Twine(getOutputSwizzle())) 
      .toVector(perfStr);
    }
    bool getForceUnroll() { return true; }
  }];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

def Rock_AttnPerfConfig : Rock_Attr<"AttnPerfConfig", [RockTuningParamAttrInterface]> {
  let mnemonic = "attn_perf_config";
  let description = [{
    The perf configs for rock.attention operator.
  }];

  let parameters = (ins "int64_t":$mPerBlockG0, "int64_t":$mPerBlockG1,
      "int64_t":$nPerBlockG0, "int64_t":$kpackPerBlock, "int64_t":$mPerWave,
      "int64_t":$nPerWave, "int64_t":$mnPerXdl, "int64_t":$kpack,
      "int64_t":$splitKFactor, "int64_t":$scheduleVersion,
      "int64_t":$outputSwizzle, "int64_t":$wavesPerEU, "bool":$forceUnroll);

  let extraClassDeclaration = [{
    void getPerfConfigStr(::llvm::SmallVectorImpl<char> &perfStr) {
      ("attn:v3:"
      + Twine(getMPerBlockG0()) + ","
      + Twine(getMPerBlockG1()) + ","
      + Twine(getNPerBlockG0()) + ","
      + Twine(getKpackPerBlock()) + ","
      + Twine(getMPerWave()) + ","
      + Twine(getNPerWave()) + ","
      + Twine(getMnPerXdl()) + ","
      + Twine(getKpack()) + ","
      + Twine(getSplitKFactor()) + ","
      + Twine(getScheduleVersion()) + ","
      + Twine(getOutputSwizzle()) + ","
      + Twine(getWavesPerEU()) + ","
      + Twine(getForceUnroll())).toVector(perfStr);
    }
    AttnPerfConfigAttr withScheduleVersion(int64_t newScheduleVersion) const {
    return AttnPerfConfigAttr::get(
      getContext(), getMPerBlockG0(), getMPerBlockG1(), getNPerBlockG0(),
      getKpackPerBlock(), getMPerWave(), getNPerWave(), getMnPerXdl(), getKpack(),
      getSplitKFactor(), newScheduleVersion, getOutputSwizzle(), getWavesPerEU(), getForceUnroll());
  }
  }];

  let builders = [AttrBuilderWithInferredContext<(
      ins "StringAttr":$perfConfigStr, "bool":$isWmma)>];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

def Rock_MfmaGemmParamsAttr
    : Rock_Attr<"MfmaGemmParams", [RockTuningParamAttrInterface,
                                   RockAccelTuningParamAttrInterface]> {
  let mnemonic = "mfma_gemm_params";
  let description = [{
    The tuning parameters for an mfma-based matrix multiplication.
  }];
  let parameters = (ins "int64_t":$kpackPerBlock, "int64_t":$mPerBlock,
      "int64_t":$nPerBlock, "int64_t":$kpack, "int64_t":$mPerWave,
      "int64_t":$nPerWave, "int64_t":$mnPerXdl, "int64_t":$splitKFactor,
      "int64_t":$scheduleVersion, "int64_t":$outputSwizzle,
      "int64_t":$wavesPerEU, "int64_t":$gridGroupSize, "bool":$forceUnroll);

  let extraClassDeclaration = [{
    void getPerfConfigStr(::llvm::SmallVectorImpl<char> &perfStr) {
        ("v4:" + Twine(getMPerBlock()) + ","
        + Twine(getNPerBlock()) + ","
        + Twine(getKpackPerBlock()) + ","
        + Twine(getMPerWave()) + ","
        + Twine(getNPerWave()) + ","
        + Twine(getMnPerXdl()) + ","
        + Twine(getKpack()) + ","
        + Twine(getSplitKFactor()) + ","
        + Twine(getScheduleVersion()) + ","
        + Twine(getOutputSwizzle()) + ","
        + Twine(getWavesPerEU()) + ","
        + Twine(getGridGroupSize()) + ","
        + Twine(getForceUnroll()) + ","
          + "1") /* *ThreadCopyMore* */
        .toVector(perfStr);
    }
  }];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

def Rock_WmmaGemmParamsAttr : Rock_Attr<"WmmaGemmParams", [RockTuningParamAttrInterface, RockAccelTuningParamAttrInterface]> {
  let mnemonic = "wmma_gemm_params";
  let description = [{
    The tuning parameters for an wmma-based matrix multiplication.
  }];
  let parameters = (ins "int64_t":$kpackPerBlock, "int64_t":$mPerBlock,
      "int64_t":$nPerBlock, "int64_t":$kpack, "int64_t":$mPerWave,
      "int64_t":$nPerWave, "int64_t":$mnPerXdl, "int64_t":$splitKFactor,
      "int64_t":$scheduleVersion, "int64_t":$outputSwizzle,
      "int64_t":$wavesPerEU, "int64_t":$gridGroupSize, "bool":$forceUnroll);

  let extraClassDeclaration = [{
    void getPerfConfigStr(SmallVectorImpl<char> &perfStr) {
        ("v4:" + Twine(getMPerBlock()) + ","
        + Twine(getNPerBlock()) + ","
        + Twine(getKpackPerBlock()) + ","
        + Twine(getMPerWave()) + ","
        + Twine(getNPerWave()) + ","
        + Twine(getMnPerXdl()) + ","
        + Twine(getKpack()) + ","
        + Twine(getSplitKFactor()) + ","
        + Twine(getScheduleVersion()) + ","
        + Twine(getOutputSwizzle()) + ","
        + Twine(getWavesPerEU()) + ","
        + Twine(getGridGroupSize()) + ","
        + Twine(getForceUnroll()) + ","
          + "1") /* *ThreadCopyMore* */
        .toVector(perfStr);
    }
  }];

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

// GEMM Features
/// Keep in sync with Python bindings in
/// mlir/lib/Dialect/Rock/utility/Bindings/AmdArchDbBindings.cpp
def Rock_GemmFeatures_None : I32BitEnumAttrCaseNone<"none">;
def Rock_GemmFeatures_Mfma : I32BitEnumAttrCaseBit<"mfma", 0>;
def Rock_GemmFeatures_Wmma : I32BitEnumAttrCaseBit<"wmma", 1>;
def Rock_GemmFeatures_Dot : I32BitEnumAttrCaseBit<"dot", 2>;
def Rock_GemmFeatures_AtomicAdd : I32BitEnumAttrCaseBit<"atomic_add", 3>;
def Rock_GemmFeatures_AtomicAddBF16 : I32BitEnumAttrCaseBit<"atomic_add_bf16", 4>;
def Rock_GemmFeatures_AtomicAddF16 : I32BitEnumAttrCaseBit<"atomic_add_f16", 5>;
def Rock_GemmFeatures_AtomicFmaxF32 : I32BitEnumAttrCaseBit<"atomic_fmax_f32", 6>;
def Rock_GemmFeatures_DirectToLDS32B
    : I32BitEnumAttrCaseBit<"direct_to_lds_32b", 7>;
def Rock_GemmFeatures_DirectToLDS128B
    : I32BitEnumAttrCaseBit<"direct_to_lds_128b", 8>;

def Rock_GemmFeatures
    : I32BitEnumAttr<
          "GemmFeatures",
          "Features that can be enabled on GEMM-based operations",
          [Rock_GemmFeatures_None, Rock_GemmFeatures_Mfma,
           Rock_GemmFeatures_Wmma, Rock_GemmFeatures_Dot,
           Rock_GemmFeatures_AtomicAdd, Rock_GemmFeatures_AtomicAddBF16,
           Rock_GemmFeatures_AtomicAddF16, Rock_GemmFeatures_AtomicFmaxF32,
           Rock_GemmFeatures_DirectToLDS32B,
           Rock_GemmFeatures_DirectToLDS128B]> {
  let cppNamespace = "::mlir::rock";
  let genSpecializedAttr = 0;
}

def Rock_GemmFeaturesAttr : EnumAttr<Rock_Dialect, Rock_GemmFeatures, "GemmFeatures">;

// GEMM load tile types
def Rock_GemmLoadTileBypassLDS : I32EnumAttrCase<"BypassLDS", 0>;
def Rock_GemmLoadTileDefault : I32EnumAttrCase<"Default", 1>;
def Rock_GemmLoadTileDoubleBuffer : I32EnumAttrCase<"DoubleBuffer", 2>;
def Rock_GemmLoadTileDirectToLDSDefault
    : I32EnumAttrCase<"DirectToLDSDefault", 3>;
def Rock_GemmLoadTileDirectToLDSDoubleBuffer
    : I32EnumAttrCase<"DirectToLDSDoubleBuffer", 4>;

def Rock_GemmLoadTileType
    : Rock_I32Enum<"GemmLoadTileType", "GEMM load tile types",
                   [Rock_GemmLoadTileBypassLDS, Rock_GemmLoadTileDefault,
                    Rock_GemmLoadTileDoubleBuffer,
                    Rock_GemmLoadTileDirectToLDSDefault,
                    Rock_GemmLoadTileDirectToLDSDoubleBuffer]> {
  let cppNamespace = "::mlir::rock";
  let genSpecializedAttr = 0;
}

def Rock_GemmLoadTileTypeAttr
    : EnumAttr<Rock_Dialect, Rock_GemmLoadTileType, "GemmLoadTileType">;

// Misc Attrs
def Rock_PipelineAttr : Rock_Attr<"Pipeline"> {
  let mnemonic = "pipeline";
  let parameters = (ins
    "int64_t":$InitiationInterval
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute to express the properties of a loop to be pipelined
  }];
}

def Rock_ScheduleVersionAttr : Rock_Attr<"ScheduleVersion"> {
  let mnemonic = "schedule_version";
  let parameters = (ins "int64_t":$ScheduleVersion);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Attribute to express which GEMM Scheduling strategies a kernel lowering should use inside rocMLIR
  }];
}

def Rock_EnableSplitKForTuning : Rock_Attr<"EnableSplitKForTuning"> {
  let mnemonic = "enable_splitk_for_tuning";
  let description = [{
    Whether we tune for split-k. If unset, split-k=1.
  }];
}

def Rock_WavesPerEU : Rock_Attr<"WavesPerEU"> {
  let mnemonic = "waves_per_eu";
  let description = [{
    This is a hint to the backend compiler to tune the number of wavefronts that are capable of fitting within the resources of an EU.
  }];
}

def Rock_OutputSwizzle : Rock_Attr<"OutputSwizzle"> {
  let mnemonic = "output_swizzle";
  let description = [{
    Whether we run the output swizzle pass. 0 -> disabled, 1 -> enabled, 2 -> heuristic.
  }];
}

def Rock_PrefillAttr : Rock_Attr<"Prefill"> {
  let mnemonic = "rock.prefill";

  let description = [{
      A user interface parameter for initializing specific memory buffers

      The attribute is a way to describe which specific kernel parameter
      (typically memref) must be initialized with specific value before
      a kernel invocation.

  }];

  let parameters = (ins
    "uint32_t":$argIndex,
    "TypedAttr":$initValue
  );
  let assemblyFormat = "`<` params `>`";
}

def Rock_BlockwiseMatrixParamsAttr : Rock_Attr<"BlockwiseMatrixParams", []> {
  let mnemonic = "blockwise_matrix_params";
  let description = [{
    Encapsulates rock.blockwise_load_tile and rock.blockwise_gemm_accel parameters.
    - elementType: Element type of the matrix operation.
    - elementTypeLoad: Element type of that was actually loaded from memory (before any input fusion).
    - rotateDWithK: Trick to reduce LDS bank conflicts (see more info here: https://github.com/ROCm/rocMLIR/pull/1209)
    - swapThreadIterSubDims: Trick to reduce LDS bank conflicts (see more info here: https://github.com/ROCm/rocMLIR/pull/1209)
    - LDSLayoutDxK: Wheter the layout in LDS is DxK
    - directToLDS: Wheter direct to LDS is enabled
    - splitKAcrossThreadsFirst: Used for attention, when bypassing LDS for the result of the first GEMM, explanation here: https://github.com/ROCm/rocMLIR-internal/issues/1201#issuecomment-1898925539
    - g: gemm parameter G
    - d: gemm parameter D (could be M or N)
    - inDPerThread: How many elements of D (M or N) each thread is going to load from memory.
    - accelDDim: Accelerator instruction D dimension (for LDS transpose support, typically 16 or 32).
    - accelKDim: Accelerator instruction K dimension (for LDS transpose support).
  }];
  let parameters = (ins "Type":$elementType, "Type":$elementTypeLoad,
      "bool":$rotateDWithK, "bool":$swapThreadIterSubDims, "bool":$LDSLayoutDxK,
      "bool":$directToLDS, "bool":$splitKAcrossThreadsFirst, "int64_t":$g,
      "int64_t":$d, "int64_t":$inDPerThread,
      DefaultValuedParameter<"bool", "false">:$ldsTransposeEnabled,
      DefaultValuedParameter<"int64_t", "0">:$accelDDim,
      DefaultValuedParameter<"int64_t", "0">:$accelKDim);

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

def Rock_LDSTransposeConfigAttr : Rock_Attr<"LDSTransposeConfig", []> {
  let mnemonic = "lds_transpose_config";
  let description = [{
    Configuration parameters for hardware-assisted LDS transpose loads.
    This attribute encapsulates all the necessary information to emit
    rock.lds_transpose_load operations with the correct accelerator geometry
    and tiling parameters.
    
    - DDim: Matrix-multiply accelerator instruction D dimension (M or N, typically 16 or 32)
    - KDim: Matrix-multiply accelerator instruction K dimension (typically 8, 16, or 32)
    - mPerBlock: M dimension size per block
    - nPerBlock: N dimension size per block
    - kPerBlock: K dimension size per block
    - mPerWave: M dimension size per wave
    - nPerWave: N dimension size per wave
    - doubleBuffering: Whether double buffering is enabled
    - isOperandA: True if this is operand A, false if operand B
  }];
  let parameters = (ins "int64_t":$dDim, "int64_t":$kDim, "int64_t":$mPerBlock,
      "int64_t":$nPerBlock, "int64_t":$kPerBlock, "int64_t":$mPerWave,
      "int64_t":$nPerWave, "bool":$doubleBuffering, "bool":$isOperandA);

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
  let genVerifyDecl = 1;
}

#endif
