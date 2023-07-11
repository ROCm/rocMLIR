//===- RockOps.td - Rock operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Rock operations.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_OPS
#define ROCK_OPS

include "mlir/Dialect/Rock/IR/RockAttrDefs.td"
include "mlir/Dialect/Rock/IR/RockConvInterface.td"
include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for Rock dialect ops.
class Rock_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rock_Dialect, mnemonic, traits> {
  let extraClassDeclaration = [{
  }];
}

def TransformMapArrayAttr : TypedArrayAttrBase<Rock_TransformMapAttr,
  "Coordinate transforms array attribute, giving the sequence of transform maps applicable to a value, uppermost to lowermost"> {}

def ArgTransformsAttr : TypedArrayAttrBase<TransformMapArrayAttr,
  "An array of arrays of transformations, one per operation argument">;

class ArgTransforms<int n> : ConfinedAttr<ArgTransformsAttr, [ArrayCount<n>]>;

def AnyTensorOrMemRef :
  AnyTypeOf<[AnyTensor, AnyMemRef],
             /*summary=*/"",
             "::mlir::ShapedType">;

class TensorOrMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
  AnyTypeOf<[TensorRankOf<allowedTypes, ranks>,
             MemRefRankOf<allowedTypes, ranks>],
             /*summary=*/"",
             "::mlir::ShapedType">;

class TensorOrMemRefOf<list<Type> allowedTypes> :
  AnyTypeOf<[TensorOf<allowedTypes>,
             MemRefOf<allowedTypes>],
             /*summary=*/"Constraints the type to be either a Tensor or MemRef of certain types of elements.",
             "::mlir::ShapedType">;

class I32ArrayLength<int n> : ConfinedAttr<I32ArrayAttr, [ArrayCount<n>]>;

class Rock_Conv2DOpBase<string mnemonic, list<Type> inputTypes=[F32, F16, BF16], list<Type> outputTypes=[F32, F16, BF16]> :
    Rock_Op<mnemonic, [DeclareOpInterfaceMethods<RockGemmWrapperInterface>,
                       DeclareOpInterfaceMethods<RockConvInterface>]>{
    dag commonConvArgs = (ins TensorOrMemRefRankOf<inputTypes, [5]>:$filter,
                   TensorOrMemRefRankOf<inputTypes, [5]>:$input,
                   TensorOrMemRefRankOf<outputTypes, [5]>:$output,
                   StrAttr:$arch,
                   Rock_GemmFeaturesAttr:$features,
                   OptionalAttr<I32Attr>:$derivedBlockSize,
                   OptionalAttr<I32Attr>:$gridSize,
                   I32ArrayLength<4>:$padding,
                   I32ArrayLength<2>:$strides,
                   I32ArrayLength<2>:$dilations,
                   OptionalAttr<RockTuningParamAttrInterface>:$params);
    let results = (outs Optional<AnyRankedTensor>:$result);
}

defvar GemmInputTypes = [F32, F16, BF16, I8, F8E5M2FNUZ, F8E4M3FNUZ];
// This can be extended for quantization.
defvar GemmOutputTypes = [F32, F16, BF16, I32, I8];

def Rock_Conv2DOp : Rock_Conv2DOpBase<"conv2d", GemmInputTypes, GemmOutputTypes> {
  dag additionalArgs = (ins OptionalAttr<I32Attr>:$numCu);
  let arguments = !con(commonConvArgs, additionalArgs);
  let summary = "2D convolution forward";
  let description = [{
    The `rock.conv2d` op computes 2D convolution forward.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` `features` `=` $features attr-dict
    `:` type(operands) (`->` type($result)^)?
  }];
}

def Rock_Conv2DBwdDataOp : Rock_Conv2DOpBase<"conv2d_bwd_data">
{
  dag additionalArgs = (ins OptionalAttr<I32Attr>:$numCu,
                            IndexAttr:$kernelId);
  let arguments = !con(commonConvArgs, additionalArgs);
  let summary = "2D convolution backward data";
  let description = [{
    The `rock.conv2d_bwd_data` op computes 2D convolution backward data.

    The kernel ID represents which of the multiple backwards data kernels
    needed for the correct computation of the result this kernel is.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` `features` `=` $features attr-dict
    `:` type(operands) (`->` type($result)^)?
  }];
}


def Rock_Conv2DBwdWeightOp : Rock_Conv2DOpBase<"conv2d_bwd_weight">
{
  dag additionalArgs = (ins Optional<TensorOrMemRefRankOf<[F32], [5]>>:$workspace,
                            OptionalAttr<IndexAttr>:$kBlocks,
                            I32Attr:$numCu);
  let arguments = !con(commonConvArgs, additionalArgs);
  let summary = "2D convolution backward weight";
  let description = [{
    The `rock.conv2d_bwd_weight` op computes 2D convolution backward weight.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` `features` `=` $features attr-dict
    `:` type(operands) (`->` type($result)^)?
  }];
}

def Rock_GemmOp :
    Rock_Op<"gemm", [DeclareOpInterfaceMethods<RockGemmWrapperInterface>]>,
    Arguments<(ins Arg<TensorOrMemRefRankOf<GemmInputTypes, [2, 3]>,
                       "matrix A", [MemRead]>:$a,
                   Arg<TensorOrMemRefRankOf<GemmInputTypes, [2, 3]>,
                       "matrix B", [MemRead]>:$b,
                   Arg<TensorOrMemRefRankOf<GemmOutputTypes, [2, 3]>,
                       "matrix C", [MemRead, MemWrite]>:$c,
                   UnitAttr:$aTransposed,
                   UnitAttr:$bTransposed,
                   UnitAttr:$cTransposed,
                   StrAttr:$arch,
                   OptionalAttr<I32Attr>:$numCU,
                   Rock_GemmFeaturesAttr:$features,
                   StoreMethodAttr:$storeMethod,
                   OptionalAttr<I32Attr>:$derivedBlockSize,
                   OptionalAttr<I32Attr>:$gridSize,
                   OptionalAttr<RockTuningParamAttrInterface>:$params)>,
    Results<(outs Optional<AnyRankedTensor>:$result)> {
  let summary = "General matrix multiplication (GEMM)";
  let description = [{
    Performs the operation C += A * B.

    If none of the `transposed` attributes are set, then A is [G] x M x K,
    B is [G] x K x N, and C is [G] x M x N, where G is the optional group dimension
    (which is assumed to be 1 if not set).

    The transpose attributes allow for the non-group dimensions of the matrix to be
    transposed. For example, if `aTransposed` is set, then the argument A should be
    a [G] x K x M memory.

    Those creating a `rock.gemm` must specify the GPU architecture being targetted
    and the number of compute units (numCu) available. The parameters
    `derivedBlockSize`, `gridSize`, and `params` are optional as they can be inferred by
    a tuning process or a heuristic, but they must be set before the `gemm` is
    lowered into the `gridwise_gemm` stage of the code generation pipeline.

    `features` specifies what hardware features can be used in the generated code.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    (`tr` $cTransposed^)? $c `=` (`tr` $aTransposed^)? $a `*` (`tr` $bTransposed^)? $b
    `features` `=` $features `storeMethod` `=` $storeMethod attr-dict
    `:` type($c) `=` type($a) `*` type($b) (`->` type($result)^)?
  }];
}


def Rock_ReduceOp :
  Rock_Op<"reduce", [AllElementTypesMatch<["in", "out"]>]>,
  Arguments<(ins
    Arg<TensorOrMemRefOf<[F32]>, "input", [MemRead]>:$in,
    Arg<TensorOrMemRefOf<[F32]>, "output", [MemRead, MemWrite]>:$out,
    Rock_GemmFeaturesAttr:$features,
    ReduceMethodAttr:$reduceMethod,
    IndexAttr:$axis,
    I32Attr:$blockSize,
    I32Attr:$gridSize,
    UnitAttr:$useLDS,
    UnitAttr:$useDPP
  )>,
  Results<(outs Optional<TensorOf<[F32]>>:$result)> {
  let summary = "Axes-wide reduction operation";
  let hasVerifier = 1;
  let assemblyFormat = [{
    $reduceMethod $in `into` $out `features` `=` $features attr-dict `:` type($in) `into` type($out) (`->` type($result)^)?
  }];
  let extraClassDeclaration = [{
    ::mlir::OpOperand* getOutArgument() { return &(*this)->getOpOperand(1); }
  }];
}

def Rock_AttentionOp :
  Rock_Op<"attention", [AllElementTypesMatch<["queries", "keys", "values", "scale"]>]>,
  Arguments<(ins
    Arg<TensorOrMemRefOf<[F32, F16]>, "queries", [MemRead]>:$queries,
    Arg<TensorOrMemRefOf<[F32, F16]>, "keys", [MemRead]>:$keys,
    Arg<TensorOrMemRefOf<[F32, F16]>, "values", [MemRead]>:$values,
    Arg<TensorOrMemRefOf<[F32, F16]>, "scale", [MemRead]>:$scale,
    Arg<TensorOrMemRefOf<[F32, F16]>, "output", [MemRead, MemWrite]>:$out,
    Rock_GemmFeaturesAttr:$features,
    OptionalAttr<I32Attr>:$blockSize,
    OptionalAttr<I32Attr>:$gridSize
  )>,
  Results<(outs Optional<TensorOf<[F32, F16]>>:$result)> {
  let summary = "Attention operation of transformer models";
  let description = [{
    Performs the operation out = SOFTMAX((queries * keys) .* scale) * values.

    This operation performs attention mechanism of transformer models.

    Those creating a `rock.attention` must specify the GPU architecture being targetted
    and the number of compute units (numCu) available. The parameters
    `gridSize`, and `blockSize` are optional as they can be inferred by
    a tuning process or a heuristic, but they must be set before the `attention` is
    lowered into the `gridwise_attention` stage of the code generation pipeline.

    `features` specifies what hardware features can be used in the generated code.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` `features` `=` $features attr-dict
    `:` type(operands) (`->` type($result)^)?
  }];
  let extraClassDeclaration = [{
    ::mlir::OpOperand* getOutArgument() { return &(*this)->getOpOperand(1); }
  }];
}

def Rock_ZeroInitKernelOp :
    Rock_Op<"zero_init_kernel", []>,
    Arguments<(ins AnyTensorOrMemRef:$buffer,
                  Rock_GemmFeaturesAttr:$features,
                  OptionalAttr<I32Attr>:$blockSize,
                  OptionalAttr<I32Attr>:$gridSize,
                  OptionalAttr<IndexAttr>:$elemsPerThread)>,
    Results<(outs Optional<AnyTensor>:$result)> {
  let summary = "Zero-initialize a buffer";

  let description = [{
    Zero-initialize a given buffer.

    This operation is meant to operate as its own kernel, and is needed
    for correctness in some cases (backwards data kernels where not all
    pixels are written and backwards weight atomic add kernels).
  }];

  let assemblyFormat = [{
    $buffer `features` `=` $features attr-dict `:` type($buffer) (`->` type($result)^)?
  }];

  // Declaration to enable the bufferization implementation to work as if this werr
  // a gemm wrapper kernel
  let extraClassDeclaration = [{
    ::mlir::OpOperand* getOutArgument() { return &(*this)->getOpOperand(0); }
  }];
}

def Rock_ConvertingCopyKernelOp :
    Rock_Op<"converting_copy_kernel", [AllShapesMatch<["input", "output"]>]>,
    Arguments<(ins AnyTensorOrMemRef:$input,
                  AnyTensorOrMemRef:$output,
                  Rock_GemmFeaturesAttr:$features,
                  OptionalAttr<I32Attr>:$blockSize,
                  OptionalAttr<I32Attr>:$gridSize,
                  OptionalAttr<IndexAttr>:$elemsPerThread)>,
    Results<(outs Optional<AnyTensor>:$result)> {
  let summary = "Copy input to output, performing a type conversion";

  let description = [{
    Copies an input buffer to an output buffer, converting the type of each element,
    as its own kernel.

    This operation is needed for backwards weight atomic add kernels where
    the output is a float smaller than f32, since the atomic add instructions
    for smaller floating point types are not accurate enough for our purposes.
  }];

  let assemblyFormat = [{
    $input `to` $output `features` `=` $features attr-dict
    `:` type($input) `to` type($output) (`->` type($result)^)?
  }];

  // Declaration to enable the bufferization implementation to work as if this werr
  // a gemm wrapper kernel
  let extraClassDeclaration = [{
    ::mlir::OpOperand* getOutArgument() { return &(*this)->getOpOperand(1); }
  }];

}

def Rock_TransformOp :
    Rock_Op<"transform", [Pure, ViewLikeOpInterface]>,
    Arguments<(ins AnyShaped:$input, Rock_TransformMapAttr:$transform)>,
    Results<(outs AnyShaped:$output)> {
  let summary = "Tensor transformation";
  let description = [{
    Create a viee `output` of the tensor `input` with the same element type
    such that, when the coordinates used to index into `output` are passed through
    the coordinate transformations `transforms` from left to right,
    they become coordinates into the `input` tensor.
  }];
  let builders = [
   // Custom builder to populate bounds of input and output memrefs as attributes.
   OpBuilder<(ins "Value":$input, "TransformMapAttr":$transform),
   [{
     $_state.addOperands({input});
     ShapedType lowerType = input.getType().template cast<ShapedType>();
     Type elemType = lowerType.getElementType();
     ShapedType upperType;
     if (auto mrType = lowerType.template dyn_cast<MemRefType>()) {
       auto memorySpace = mrType.getMemorySpace().dyn_cast_or_null<gpu::AddressSpaceAttr>();
       upperType = MemRefType::get(transform.getUpperBounds(), elemType,
        MemRefLayoutAttrInterface{}, memorySpace);
     } else {
       upperType = lowerType.cloneWith(transform.getUpperBounds().asArrayRef(), elemType);
     }
     $_state.addAttribute("transform", transform);
     $_state.addTypes(upperType);
   }]>,
  ];
  let assemblyFormat = [{
    $input `by` $transform attr-dict `:` type($input) `to` type($output)
  }];

  let extraClassDeclaration = [{
    Value getViewSource() { return getInput(); }
  }];
}

def Rock_TensorUntransformCastOp :
    Rock_Op<"tensor_untransform_cast",
      [Pure, AllTypesMatch<["transformedResult", "transformedArg"]>]>,
    Arguments<(ins AnyRankedTensor:$transformedResult,
                      AnyRankedTensor:$transformedArg)>,
    Results<(outs AnyRankedTensor:$untransformed)> {
  let summary = "Cast transformed tensor 'result' back to underlying allocation";
  let description = [{
    This operation exists to enable bufferization where we have the pattern
    ```mlir
    %buf = bufferization.alloc_tensor : T
    %t1 = rock.transform %buf : T to I1
    ...
    %output = rock.transform %tK : TK to U
    %result = rock.conv2d(..., %output) : ..., U
    op(%result, ...)
    ```
    where op required a value of type %T.

    This op
    Thi op must be eliminated during bufferization.

    If the alleged result type is not equal to the type of untransform(transformArg),
    it is in error.
  }];
  let assemblyFormat = [{
    attr-dict $transformedResult `aka` $transformedArg
    `:` type($transformedResult) `to` type($untransformed)
  }];
}

def Rock_GridwiseGemmOp :
    Rock_Op<"gridwise_gemm">,
    Arguments<(ins MemRefRankOf<GemmInputTypes, [3]>:$a,
                   MemRefRankOf<GemmInputTypes, [3]>:$b,
                   MemRefRankOf<GemmOutputTypes, [3]>:$c,
                   Rock_GemmFeaturesAttr:$features,
                   I32Attr:$gridSize,
                   Rock_GeneralGemmParamsAttr:$params)> {
  let summary = "Gridwise GEMM";
  let description = [{
    The `rock.gridwise_gemm` op computes gridwise GEMM.
  }];
  let assemblyFormat = [{
    $c `=` $a `*` $b `features` `=` $features attr-dict `:` type($c) `=` type($a) `*` type($b)
  }];
  let hasVerifier = 1;
}

// gridwise_gemm_accel
def Rock_GridwiseGemmAccelOp :
    Rock_Op<"gridwise_gemm_accel">,
    Arguments<(ins MemRefRankOf<GemmInputTypes, [3]>:$a,
                   MemRefRankOf<GemmInputTypes, [3]>:$b,
                   MemRefRankOf<GemmOutputTypes, [3]>:$c,
                   StrAttr:$arch,
                   Rock_GemmFeaturesAttr:$features,
                   StoreMethodAttr:$storeMethod,
                   I32Attr:$blockSize,
                   I32Attr:$gridSize,
                   RockAccelTuningParamAttrInterface:$params)> {
  let summary = "Gridwise GEMM accelerated version";
  let description = [{
    The `rock.gridwise_gemm` op computes gridwise GEMM with acceleration.
  }];
  let assemblyFormat = [{
    `(` operands `)` `storeMethod` `(` $storeMethod `)` `features` `=` $features attr-dict `:` type(operands)
  }];
  let hasVerifier = 1;
}

// Memory allocation on GPU memory hierachy.
def Rock_GpuAllocOp:
    Rock_Op<"alloc">,
    Results<(outs Res<AnyMemRef, "", [MemAlloc]>:$output)> {
  let summary = "Memory allocation on GPU";
  let description = [{
    The `rock.alloc` op allocates memory on GPU.
    - Address space 0 : global.
    - Address space 3 : LDS.
    - Address space 5 : private (VGPR).
    All other values would be considered as allocation on global.
  }];
  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($output)
  }];
}

// TBD: eventually replace this with linalg.fill?
def Rock_FillOp:
    Rock_Op<"fill">,
    Arguments<(ins AnyMemRef:$input,
                   //AnyTypeOf<[AnyInteger, AnyFloat]>:$value)> {
                   AnyTypeOf<[AnyInteger, AnyFloat, AnyVector]>:$value)> {
  let summary = "Fill memory with constant value on GPU";
  let description = [{
    The `rock.fill` op fills a memref on GPU with a constant value.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

def Rock_BlockwiseFillOp:
    Rock_Op<"blockwise_fill", [AllElementTypesMatch<["memref", "value"]>]>,
    Arguments<(ins AnyMemRef:$memref,
                   AnyTypeOf<[AnyInteger, AnyFloat, AnyVector]>:$value,
                   I32Attr:$blockSize)> {
  let summary = "Fill memory with constant value on GPU";
  let description = [{
    The `rock.blockwise_fill` op fills a LDS memref on GPU with a constant value.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
  let hasVerifier = 1;
}

def Rock_WorkgroupBarrierOp:
    Rock_Op<"workgroup_barrier"> {
  let summary = "Setup an workgroup barrier";
  let description = [{
    The `rock.workgroup_barrier` op sets up a workgroup-level barrier.
  }];
  let assemblyFormat = "attr-dict";
}

def Rock_LDSBarrierOp:
    Rock_Op<"lds_barrier"> {
  let summary = "Setup an LDS barrier";
  let description = [{
    The `rock.lds_barrier` op sets up a workgroup-level barrier on LDS activities.
  }];
  let assemblyFormat = "attr-dict";
}

def Rock_WorkgroupIdOp:
    Rock_Op<"workgroup_id", [Pure,
      DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Results<(outs Index:$id)> {
  let summary = "Get current workgroup ID";
  let description = [{
    The `rock.workgroup_id` op gets the current workgroup ID.
  }];
  let assemblyFormat = "attr-dict `:` type($id)";
}

def Rock_WorkitemIdOp:
    Rock_Op<"workitem_id", [Pure,
      DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Results<(outs Index:$id)> {
  let summary = "Get current workitem ID";
  let description = [{
    The `rock.workgroup_id` op gets the current workitem ID.
  }];
  let assemblyFormat = "attr-dict `:` type($id)";
}

// extract_slice
def Rock_ExtractSliceOp :
    Rock_Op<"extract_slice", [Pure,
      AllElementTypesMatch<["vector", "result"]>]>,
    Arguments<(ins
      VectorOfRank<[1]>:$vector,
      Index:$coord)>,
    Results<(outs AnyType:$result)> {
  let summary = "Extract a slice from a vector";

  let description = [{
    Extracts `len(result)` contiguous elements starting at `coord`  in `vector`.
    If `result` is a scalar type, the length is 1 and this is just `vector.extractelement`.
    If this causes an out of bounds read, the result is undefined.
  }];

  let assemblyFormat = [{
    attr-dict $vector `[` $coord `]` `:` type($vector) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

// insert_slice
def Rock_InsertSliceOp :
    Rock_Op<"insert_slice", [Pure,
      AllElementTypesMatch<["source", "dest"]>,
      AllTypesMatch<["dest", "result"]>]>,
    Arguments<(ins
      AnyType:$source,
      VectorOfRank<[1]>:$dest,
      Index:$coord)>,
    Results<(outs VectorOfRank<[1]>:$result)> {
  let summary = "Insert a slice into a vector";

  let description = [{
    Inserts  `source` into contigious indices of `dest`, starting at `coord`.
    If `source` is a scalar type, this is just `vector.insertelement`.
    If the indices into `dest` go out of bounds, the result is undefined.
  }];

  let assemblyFormat = [{
    attr-dict $source `->` $dest `[` $coord `]` `:` type($source) `->` type($dest)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

// yield
def Rock_YieldOp :
    Rock_Op<"yield", [Pure, Terminator, ReturnLike]> {
  let summary = "yield for Rock loops";

  let description = [{
    Op that terminates the Rock looping constructs (currently transforming for).

    If the loops produce any values, then this op's arguments should match the types
    of those values. Otherwise, this op is implicit and can be omitted.

    Note that, in a transforming_for, you don't need to yield back the lower coordinates,
    as these are managed implicitly.
  }];

  // Yoinked from affine.yield
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// Transforming for
def Rock_TransformingForOp :
    Rock_Op<"transforming_for", [RecursivelySpeculatable, RecursiveMemoryEffects,
      DeclareOpInterfaceMethods<LoopLikeOpInterface, ["isDefinedOutsideOfLoop"]>,
      AttrSizedOperandSegments,
      SingleBlockImplicitTerminator<"::mlir::rock::YieldOp">]>,
    Arguments<(ins Variadic<Index>:$upperInits,
      ArgTransformsAttr:$transforms,
      Variadic<AnyType>:$iterInits,
      IndexArrayAttr:$bounds,
      IndexArrayAttr:$strides,
      // This is a derived attribute that holds where in the block arguments
      // the coordinates for each lower domain are stored. It contains
      // one value for each iteration domain, and then two final values:
      // one for the validity booleans and then a final value for
      // the start of the iteration arguments. That is, lowerStarts = [0, 2, 3]
      // means there is one iteration domain with two lower coordinates,
      // while with lowerStarts = [0, 3, 4, 6], there are two iteration domains -
      // the first with three lower coordinates and the second with one
      I32ElementsAttr:$lowerStarts,
      OptionalAttr<UnitAttr>:$useIndexDiffs,
      OptionalAttr<UnitAttr>:$forceUnroll)>,
    Results<(outs Variadic<AnyType>:$results)> {
  let summary = "for loop with coordinate transforms";
  let description = [{
    Loops over several a rectangular regeon of dimensions `bounds` in several
    iteration domains, which are coordinate spaces that are the upper coordinates
    for a sequence of coordinate transformations.

    For each domain, when we have
    `(%l0, %l1, ... %lL) = [#transform_map1](%u0, %u1, ... %uU)`
    with bounds `[b0, b1, ... bU]` and strides `[1, 1, ..., 1]`
    the loop arguments %l0, ... %lL will take on the values
    - (%l0, ... %lL) = #transform_map1(%u0, %u1, ... %uU)
    - (%l0, ... %lL) = #transform_map1(%u0, %u1, ... %uU + 1)
    - ...
    - (%l0, ... %lL) = #transform_map1(%u0, %u1, ... %uU + bU - 1)
    - (%l0, ... %lL) = #transform_map1(%u0, %u1, ... %u(U-1) + 1, %uU)
    - ...
    - (%l0, ... %lL) = #transform_map1(%u0 + b0 - 1, ... %uU + bU - 1)

    That is, the loop
    ```mlir
    %res = rock.transforming_for (%a1, %a2, %a3) = [#transform_map1](%i0, %j0), (%i, %j) = [](%cst0, %cst0)
      (%v1, %v2) = validity
      iter_args(%arg0 = %c0) -> (index) bounds = [4, 8] strides = [1, 1]{
      %v = rock.buffer_load {...} %place[%a1, %a2, %a3] if %v1
      rock.buffer_store {...} %v -> %buffer[%i, %j] if %v2
      %cont = addi %arg0, %c1
      rock.yield %cont : index
    }
    ```
    will lower to
    ```mlir
    %res = affine.for %d0 = 0 to 4 step 1 (%arg0 = %cst0 : index) {
      %res1 = affine.for %d1 = 0 to 8 step 1 (%arg1 = %arg0 : index) {
        %i0_shifted = arith.addi %i0, %d0
        %j0_shifted = arith.addi %j0, %d1
        %a1, %a2, %a3 = [expand affine map in #transform_map1 at %i0_shifted, %j0_shifted]
        %v1 = [compute validity in #transform_map1 as needed]
        %v2 = true
        %v = rock.buffer_load {...} %place[%a1, %a2, %a3]
        rock.buffer_store {...} %v -> %buffer[%d0, %d1]
        %cont = arith.addi %arg1, %c1 : index
        affine.yield %cont : index
      }
      affine.yield %res1
    }
    ```

    When multiple transform_maps are specified for an interatino domain,
    they are composed, with the left one applying first (to the upper coordinates),
    and a lack of transform_maps simply passes the upper cooridate values through.

    If `strides` is not specified during building, it will default to all 1s.

    If the `forceUnroll` attribute is specified, the loops above are unrolled
    after being generated.

    If `useIndexDiffs` is set, instead of computing the affine map within each
    loop, the index diff mechanism is used to determine the change in the lower
    coordinates based on the induction variables of the affine loops and
    the values of the lower coordinates at the initial start values.

    The purpose of this looping construct is to remove redundancy and make the
    application of coordinate transformations more explicit in the IR.
  }];

  let regions = (region SizedRegion<1>:$region);

  DerivedAttr upperLen = DerivedAttr<"uint32_t", [{
    return getUpperInits().size() / getTransforms().size();
  }], [{ $_builder.getI32IntegerAttr($_self) }]>;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ArrayRef<ValueRange>":$inits,
    "ArrayRef<Attribute>":$transforms,
    "ArrayRef<int64_t>":$bounds, "std::optional<ArrayRef<int64_t>>":$strides,
    "bool":$forceUnroll, "bool":$useIndexDiffs,
    CArg<"ValueRange", "std::nullopt">:$iterArgs)>,

    OpBuilder<(ins "ArrayRef<ValueRange>":$inits,
    "ArrayRef<Attribute>":$transforms,
    "ArrayAttr":$bounds, "ArrayAttr":$strides,
    "bool":$forceUnroll, "bool":$useIndexDiffs,
    CArg<"ValueRange", "std::nullopt">:$iterArgs)>,

    OpBuilder<(ins "ArrayRef<ValueRange>":$inits,
    "ArrayAttr":$transforms,
    "ArrayRef<int64_t>":$bounds, "std::optional<ArrayRef<int64_t>>":$strides,
    "bool":$forceUnroll, "bool":$useIndexDiffs,
    CArg<"ValueRange", "std::nullopt">:$iterArgs)>,

    OpBuilder<(ins "ArrayRef<ValueRange>":$inits,
    "ArrayAttr":$transforms,
    "ArrayAttr":$bounds, "ArrayAttr":$strides,
    "bool":$forceUnroll, "bool":$useIndexDiffs,
    CArg<"ValueRange", "std::nullopt">:$iterArgs)>];

  let extraClassDeclaration = [{
    void moveOutOfLoop(ArrayRef<Operation *> ops);

    uint32_t getLowerStart(uint32_t n) {
      return *(getLowerStarts().getValues<uint32_t>().begin() + n);
    }

    // Retreive the block arguments corresponding to the lower coordinates
    // for a given iteration domain.
    Block::BlockArgListType getLowerCoords(uint32_t domain) {
      uint32_t start = getLowerStart(domain);
      uint32_t end = getLowerStart(domain + 1);
      return getBody()->getArguments().slice(start, end - start);
    }
    Block::BlockArgListType getLowerCoords() {
      assert(getLowerStarts().size() == 3 && "Ambiguous call to getLowerCoords() with multple iteration domains");
      return getBody()->getArguments().take_front(getLowerStart(1));
    }

    Block::BlockArgListType getValidities() {
      uint32_t validityDomain = getLowerStarts().size() - 2;
      return getLowerCoords(validityDomain);
    }

    TypedValue<IntegerType> getValidity(uint32_t domain) {
      return cast<TypedValue<IntegerType>>(getValidities()[domain]);
    }

    Block::BlockArgListType getIterArgs() {
      return getBody()->getArguments().drop_front(getLowerStart(getLowerStarts().size() - 1));
    }

    ValueTypeRange<Block::BlockArgListType> getIterArgTypes() {
      return ValueTypeRange<Block::BlockArgListType>(getIterArgs());
    }

    Operation::operand_range getUpperInits(uint32_t domain) {
      uint32_t theUpperLen = getUpperLen();
      return getUpperInits().slice(domain * theUpperLen, theUpperLen);
    }

    ArrayAttr getTransforms(uint32_t domain) {
      return getTransforms()[domain].cast<ArrayAttr>();
    }

    uint32_t domains() {
      return getTransforms().size();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// index_diff_update
def Rock_IndexDiffUpdateOp :
    Rock_Op<"index_diff_update", [Pure, AttrSizedOperandSegments,
      SameVariadicResultSize, AllTypesMatch<["lowerIndices", "lowerDiff"]>]>,
    Arguments<(ins Rock_TransformMapAttr:$map,
      Variadic<Index>:$upperDiffs,
      Variadic<Index>:$lowerOrig)>,
    Results<(outs Variadic<Index>:$lowerIndices,
      Variadic<Index>:$lowerDiff)> {
  let summary = "Compute change in lower indices (map output) using index diffs";
  let description = [{
    `rock.index_diff_update' computes the change in the output of a transform_map
    from adding `upperDiffs` to the inputs that produced the output `lowerOrig'.

    The operation returns both `lowerIndices`, the new indices, and `lowerDiff`,
    the change in the lower indices compared to `lowerOrig'.
  }];

  let builders = [
    OpBuilder<(ins "TransformMapAttr":$transform, "ValueRange":$upperDiffs,
      "ValueRange":$lowerOrig)>
  ];

  let assemblyFormat = [{
    attr-dict $map `(` $upperDiffs `)` `+` `(` $lowerOrig `)` `:`
      type($lowerIndices)
  }];
  let hasVerifier = 1;
}

defvar SupportedMemoryElems = [F32, F16, BF16, I8, I32, F8E5M2FNUZ, F8E4M3FNUZ];
defvar NativeMemoryOpTypes = [F32, F16, BF16, I8, I32,
                           F8E5M2FNUZ, F8E4M3FNUZ,
                           VectorOfLengthAndType<[2, 4], [F32, I32]>,
                           VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                           VectorOfLengthAndType<[2, 4, 8, 16],
                             [I8, F8E5M2FNUZ, F8E4M3FNUZ]>];

// buffer_load
def Rock_BufferLoadOp :
    Rock_Op<"buffer_load">,
    Arguments<(ins Arg<MemRefOf<SupportedMemoryElems>,
        "buffer to load from", [MemRead]>:$source,
      I1:$valid,
      Variadic<Index>:$coords,
      OptionalAttr<IndexAttr>:$offset,
      UnitAttr:$oobIsOverflow)>,
    Results<(outs AnyTypeOf<NativeMemoryOpTypes>:$result)> {
  let summary = "Load data from a global buffer";

  let description = [{
    rock.buffer_load uses, if necessary, GPU buffer load intrinsics to
    ensure that out of bounds loads return 0 instead of a garbage value.

    If `valid` is false, this operation will ignore the provided coordinates
    and instead return 0.

    This op can perform vector reads.

    If provided, `offset` is a constant offset (in units of the underlying scalar
    data type) that is added to the final coordinates.

    If `oobIsOverflow` is set, then `valid` is ignored and the indices passed to
    the load are used unconditionally, but the hardware bounds checking is enabled
    even if if `valid` were always true. This is used to simplify the implementation
    of elementwise kernels.

    The memref must be in global memory (memory space 0).
  }];
  let assemblyFormat = [{
    $source `[` $coords `]` `if` $valid attr-dict
    `:` type($source) (`,` type($coords)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

// buffer_store
def Rock_BufferStoreOp :
    Rock_Op<"buffer_store", []>,
    Arguments<(ins AnyTypeOf<NativeMemoryOpTypes>:$data,
      Arg<MemRefOf<SupportedMemoryElems>,
        "Buffer to store to", [MemWrite]>:$dest,
      I1:$valid,
      Variadic<Index>:$coords,
      Rock_GemmFeaturesAttr:$features,
      StoreMethodAttr:$storeMethod,
      OptionalAttr<IndexAttr>:$offset,
      UnitAttr:$oobIsOverflow)> {
  let summary = "Store data to a global buffer";

  let description = [{
    `rock.buffer_store` stores data to a global buffer, ignoring out of bounds writes.

    If `valid` is false, the provided coordinates will be ignored and an out
    of bounds write will be generated.

    This op supports vector writes and uses buffer store intrinsics if needed.

    `storeMethod` controls whether the data is written to memory, overwriting
    existing contents, or whether it is added to the existing memory atomically.

    If `oobIsOverflow` is set, then `valid` is ignored and the indices passed to
    the load are used unconditionally, but the hardware bounds checking is enabled
    even if if `valid` were always true. This is used to simplify the implementation
    of elementwise kernels.

    The buffer must reside in global memory.
  }];
  let assemblyFormat = [{
    $storeMethod $data `->` $dest `[` $coords `]` `if` $valid `features` `=` $features attr-dict
    `:` type($data) `->` type($dest) (`,` type($coords)^)?
  }];
  let hasVerifier = 1;
}

// in_bounds_load
def Rock_InBoundsLoadOp :
    Rock_Op<"in_bounds_load", [AllElementTypesMatch<["source", "result"]>]>,
    Arguments<(ins Arg<AnyMemRef,
        "buffer to load from", [MemRead]>:$source,
      Variadic<Index>:$coords)>,
    Results<(outs AnyType:$result)> {
  let summary = "Load one or more contiguous items from `source`";

  let description = [{
    rock.in_bounds_loads reads either one scalar value or a vector of values
    from a memref. The memref can be in any memory space. The read begins at
    `coords`.
  }];
  let assemblyFormat = [{
    $source `[` $coords `]` attr-dict
    `:` type($source) `,` type($coords) `->` type($result)
  }];
  let hasVerifier = 1;
}

// in_bounds_store
def Rock_InBoundsStoreOp :
    Rock_Op<"in_bounds_store", [AllElementTypesMatch<["data", "dest"]>]>,
    Arguments<(ins AnyType:$data,
      Arg<MemRefOf<SupportedMemoryElems>,
        "Buffer to store to", [MemWrite]>:$dest,
      Variadic<Index>:$coords)> {
  let summary = "Store one or more items to contiguous indices in `dest`";

  let description = [{
    `rock.in_bounds_store` stores the item or items in `data` (which can be
    a scalar or a vector) to the buffer `dest`, starting at `coords`.
    The memref can be in any memory space.
    This op assumes that all indices will land in bounds.
  }];
  let assemblyFormat = [{
    $data `->` $dest `[` $coords `]` attr-dict
    `:` type($data) `->` type($dest) `,` type($coords)
  }];
  let hasVerifier = 1;
}

// global_load
def Rock_GlobalLoadOp :
    Rock_Op<"global_load", [AllElementTypesMatch<["source", "result"]>]>,
    Arguments<(ins Arg<MemRefOf<SupportedMemoryElems>, "source memory", [MemRead]>:$source,
                   I1:$valid,
                   Variadic<Index>:$sourceCoord)>,
    Results<(outs AnyType:$result)> {
  let summary = "Marker for one or more vectorized global loads";
  let description = [{
    `global_load` is a wrapper around one or more `rock.buffer_load` operations
    that loads a contiguous vector from %source[%sourceCoord, ...].
    This is used both to allow for vectorization lengths greater than one load
    instruction permits and to mark the global load loops to allow fusion.
  }];
  let assemblyFormat = [{
    $source `[` $sourceCoord `]` `if` $valid attr-dict
    `:` type($source) `->`
    type($result)
  }];
}

// global_store
def Rock_GlobalStoreOp :
    Rock_Op<"global_store", [AllElementTypesMatch<["source", "dest"]>]>,
    Arguments<(ins MemRefRankOf<SupportedMemoryElems, [1]>:$source,
                   MemRefOf<SupportedMemoryElems>:$dest,
                   IndexAttr:$length,
                   Rock_GemmFeaturesAttr:$features,
                   StoreMethodAttr:$storeMethod,
                   Index:$sourceCoord,
                   I1:$valid,
                   Variadic<Index>:$destCoord)> {
  let summary = "Marker for stores to global memory";
  let description = [{
    The `global_store` op is a wrapper around storing to a global buffer.
    This is used in order to make it easier for operator
    fusion to identify the write out to global memory produced within a gridwise
    gemm.

    It will lower to one or more buffer store operations.

    The location `%dest[..., %N + (length - 1)]` must be a valid memory
    address, even if it is not in-bounds from the `memref` perspective.
  }];
  let assemblyFormat = [{
    $source `[` $sourceCoord `]`
    `->` $dest `[` $destCoord `]` `if` $valid
    `storeMethod` `(` $storeMethod `)` `features` `=` $features attr-dict
    `:` type($source) `->`
    type($dest) `,` type($destCoord)
  }];
}

// threadwise_read_into
def Rock_ThreadwiseReadIntoOp :
    Rock_Op<"threadwise_read_into">,
    AllElementTypesMatch<["source", "dest"]>,
    Arguments<(ins Arg<MemRefOf<SupportedMemoryElems>, "source view", [MemRead]>:$source,
      Arg<MemRefRankOf<SupportedMemoryElems, [1]>,
        "destination registers", [MemWrite]>:$dest,
      TransformMapArrayAttr:$extraViews,
      Variadic<Index>:$extraIndices,
      UnitAttr:$forceUnroll,
      UnitAttr:$useIndexDiffs
    )> {
  let summary = "Read values from transformed source into destination";

  let description = [{
    A high-level representation of a global -> register read loop that
    accounts for coordinate transformations.

    If `%source = rock.transform #transform_mapN %buffer`
    (with the one transformation representing an entire sequence),
    and `%buffer` is in global memory, the operation

    ```mlir
    rock.threadwise_read_into [#transform_mapM](%source) -> %dest
    ```

    will lower to
    ```mlir
    %bid = rock.workgroup_id
    %tid = rock.workitem_id
    rock.transforming_for
      (%args, ...) = MAPS(%bid, %tid, %c0)
      (%_, %_, %i) = [](%c0, %c0, %c0)
      (%isValid, %isValid) = isValid
      bounds = [1, 1, L], strides = [1, 1, V] {
        %v = rock.global_load {length = V} %buffer[%args] if %isValid
        rock.in_bounds_store %v -> %dest[%i]
    }
    ```
    where MAPS is `[#transform_mapM, #transform_mapN]`,
    L is the length of `%dest`, V is the maximum vectorization computed
    for MAPS.

    The input to extraViews ; (the transforms on %source) must have the form
    (extraIdx0, ... , extraIdxN, iteration_number)

    Primarily, extraIndices would be used to pass in tid and bid. This would need
    to have a matching view in [extraViews]source. The extraIndices could be used
    to integrate loop induction vars that is outside of the op.
    If extraIndices are used, the [extraViews]source must have the form
    (extraIdx0, ... , extraIdxN, iteration_number).

    This is used during fusion to represent loads from addition arguments like
    bias tensors.
  }];

  let assemblyFormat = [{
    attr-dict $extraViews `(` $source `)` (`[` $extraIndices^ `]`)? `->` $dest
    `:` type($source) `->` type($dest)
  }];
  let hasVerifier = 1;
}

// threadwise_write_all
def Rock_ThreadwiseWriteAllOp :
    Rock_Op<"threadwise_write_all">,
    AllElementTypesMatch<["source", "dest"]>,
    Arguments<(ins Arg<MemRefRankOf<SupportedMemoryElems, [1]>,
      "source registers", [MemRead]>:$source,
      Arg<MemRefOf<SupportedMemoryElems>, "detination view", [MemWrite]>:$dest,
      TransformMapArrayAttr:$extraViews,
      Variadic<Index>:$extraIndices,
      Rock_GemmFeaturesAttr:$features,
      StoreMethodAttr:$storeMethod,
      UnitAttr:$forceUnroll,
      UnitAttr:$useIndexDiffs
    )> {
  let summary = "Write out values in registers to transformed destination";

  let description = [{
    A high-level representation of a register -> somewhere write loop that
    accounts for coordinate transformations.

    If `%dest = rock.transform #transform_mapN %buffer`
    (with the one transformation representing an entire sequence),
    and `%buffer` is in global memory, the operation

    ```mlir
    rock.threadwise_write_all %source -> [#transform_mapM](%dest) by set
    ```

    will lower to
    ```mlir
    %bid = rock.workgroup_id
    %tid = rock.workitem_id
    rock.transforming_for (%_, %_, %i) = [](%c0, %c0, %c0)
      (%args, ...) = MAPS(%bid, %tid, %c0)
      (%na, %isValid) = validity
      bounds = [1, 1, L], strides = [1, 1, V] {
        rock.global_store set {length = V} %source[%i] -> %buffer[%args] if %isValid
    }
    ```
    where MAPS is `[#transform_mapM, #transform_mapN]`,
    L is the length of `%source`, V is the maximum vectorization computed
    for MAPS, and the oob dimensions are those combputed on MAPS.

    The input to extraViews ; (the transforms on %dest) must have the form
    (extraIdx0, ... , extraIdxN, iteration_number).

    Primarily, extraIndices would be used to pass in tid and bid. This would need
    to have a matching view in [extraViews]dest. The extraIndices could be used
    to integrate loop induction vars that is outside of the op.

    If extraIndices are used, the [extraViews]dest must have the form
    (extraIdx0, ... , extraIdxN, iteration_number).

    This is used during fusion to allow rewriting
    ```mlir
    %tmp = memref.alloc()
    ...
    rock.threadwise_write_all [...](%result) -> %tmp
    linalg.generic (%tmp) -> (%output) { ... }
    ```

    to
    ```mlir
    linalg.generic (%result) -> (%result_2) { ... }
    rock.threadwise_write_all [...](%result_2) -> %out
    ```
  }];

  let assemblyFormat = [{
    `features` `=` $features attr-dict $source  `->` $extraViews `(` $dest `)` (`[` $extraIndices^ `]`)? `by` $storeMethod
    `:` type($source) `->` type($dest)
  }];
  let hasVerifier = 1;
}

defvar LdsBufferTypes = GemmInputTypes # [VectorOfRankAndType<[1], GemmInputTypes>];

// blockwise_gemm
def Rock_BlockwiseGemmOp:
    Rock_Op<"blockwise_gemm">,
    Arguments<(ins MemRefRankOf<GemmInputTypes, [3]>:$matrixA,
                   MemRefRankOf<GemmInputTypes, [3]>:$matrixB,
                   MemRefRankOf<GemmOutputTypes, [2]>:$matrixC,
                   Rock_GeneralGemmParamsAttr:$params
                   )> {
  let summary = "Blockwise GEMM non accelerated version";
  let description = [{
    The `rock.blockwise_gemm` op does gemm at the blockwise level without acceleration.

    Matrix A resides in LDS and has dimensions [k, m_c * mRepeatStride, kpack].
    Matrix B resides in LDS and has dimensions [k, n_c * nRepeatStride, kpack].

    Matrix C resides in registers and has dimensions [m_c, n_c].

    The two index arguments specify a given threads's offset into the LDS buffer..
    Each {m,n}PerThread group of elements (which may themselves be kPacks)
    is read from LDS by a given thread, then the next mPerThread group is
    mRepeatStride elements later in the buffer.
  }];
  let assemblyFormat = [{
    $matrixC `+` `` `=` $matrixA `*` $matrixB attr-dict
    `:` type($matrixC) `+` `` `=` type($matrixA) `*` type($matrixB)
  }];

  let hasVerifier = 1;
}

// This is less general than the one in the AMDGPU dialect, since it only
// accounts for how we call these operations.
defvar AccelArgTypes = [F32,
                    VectorOfLengthAndType<[2, 4], [F32]>,
                    VectorOfLengthAndType<[2, 4, 8, 16], [F16, BF16]>,
                    VectorOfLengthAndType<[4, 8, 16], [I8, F8E5M2FNUZ, F8E4M3FNUZ]>];
defvar AccelResTypes = [VectorOfLengthAndType<[4, 8, 16, 32], [F32, I32, F16, BF16]>];

// blockwise_gemm_accel
def Rock_BlockwiseGemmAccelOp:
    Rock_Op<"blockwise_gemm_accel">,
    Arguments<(ins MemRefOf<LdsBufferTypes>:$matrixA,
                   MemRefOf<LdsBufferTypes>:$matrixB,
                   Index:$waveOffsetA,
                   Index:$waveOffsetB,
                   MemRefOf<AccelArgTypes>:$bufferA,
                   MemRefOf<AccelArgTypes>:$bufferB,
                   MemRefOf<AccelResTypes>:$matrixC,
                   StrAttr:$arch,
                   Rock_GemmFeaturesAttr:$features,
                   I32Attr:$blockSize,
                   RockAccelTuningParamAttrInterface:$params)>{
  let summary = "Blockwise GEMM accelerated version";
  let description = [{
    The `rock.block_gemm_v2` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on LDS (naive tensor).
    - Matrix C shall be vectors.

    The elements of matrices A and B should be vectors of length kpack, or
    scalars when kpack is 1.
  }];
  let assemblyFormat = [{
    $matrixC `+` `` `=` $bufferA `from` $matrixA `[` $waveOffsetA `]` `*`
                        $bufferB `from` $matrixB `[` $waveOffsetB `]` `features` `=` $features attr-dict
    `:` type($matrixC) `+` `` `=` type($bufferA) `from` type($matrixA) `*`
                                  type($bufferB) `from` type($matrixB)
  }];
}

// threadwise_gemm
def Rock_ThreadwiseGemmOp:
    Rock_Op<"threadwise_gemm",
      [AllElementTypesMatch<["matrixA", "matrixB", "matrixC"]>]>,
    Arguments<(ins MemRefRankOf<GemmOutputTypes, [3]>:$matrixA,
                   MemRefRankOf<GemmOutputTypes, [3]>:$matrixB,
                   MemRefRankOf<GemmOutputTypes, [2]>:$matrixC)> {
  let summary = "Threadwise GEMM non accelerated version";
  let description = [{
    The `rock.threadwise_gemm` op does GEMM at thread level.
    All arguments should be in registers (memref address space 5).

    The dimensions of the multiplication arguments are
     [m, n] = [k, m, kPack] * [k, n, kPack].
  }];
  let assemblyFormat = [{
    $matrixC `+` `` `=` $matrixA `*` $matrixB attr-dict
    `:` type($matrixC) `+` `` `=` type($matrixA) `*` type($matrixB)
  }];
  let hasVerifier = 1;
}

// accel_gemm
def Rock_AccelGemmOp:
    Rock_Op<"accel_gemm">,
    Arguments<(ins Index:$mRepeat,
                   Index:$nRepeat,
                   MemRefRankOf<AccelArgTypes, [1]>:$matrixA,
                   MemRefRankOf<AccelArgTypes, [1]>:$matrixB,
                   MemRefRankOf<AccelResTypes , [1]>:$matrixC,
                   StrAttr:$arch,
                   Rock_GemmFeaturesAttr:$features,
                   RockAccelTuningParamAttrInterface:$params)> {
  let summary = "Accelerated GEMM";
  let description = [{
    The `rock.accel_gemm` op is an abstraction of doing GEMM based on an accelerator.
    It would employ a series of accelerator (e.g., mfma or wmma) operations.

    Matrices A and B reside in LDS, the buffers live in registers, C is a vector
  }];
  let assemblyFormat = [{
    $matrixC `+` `` `=` $matrixA`[`$mRepeat`]` `*` $matrixB`[`$nRepeat`]` `features` `=` $features attr-dict
    `:` type($matrixC) `+` `` `=` type($matrixA) `*` type($matrixB)
  }];
  let hasVerifier = 1;
}

// blockwise_broadcasting_reduction
def Rock_BlockwiseBroadcastReduceOp:
    Rock_Op<"blockwise_broadcast_reduce">,
    Arguments<(ins MemRefOf<GemmOutputTypes>:$input,
                   MemRefOf<LdsBufferTypes>:$workspace_buffer,
                   MemRefOf<GemmOutputTypes>:$output,
                   IndexAttr:$axis,
                   ReduceMethodAttr:$reduceMethod,
                   TransformMapArrayAttr:$inputRegView,
                   I32Attr:$blockSize
                   )> {
  let summary = "Blockwise reduce operation with output broadcasting";
  let description = [{
    The `rock.blockwise_broadcast_reduce` op does reduce at a blockwise level and
    to fill all elements of the input tensor with reduced values. Note that both input/output
    tensors live in registers.

    $input tensor resides in registers, where each thread holds a subset of the elements 
    of the tensor as defined by $inputView.

    $inputRegView should be a coordinate transform taking {tid, iter} (where iter is 
    the register number in $input and tid is the ID of the lane within the workgroup 
    and producing indices into a virtual tensor `{d0, ..., dR, ... dN} 
    which is spread across the lanes of the workgroup. Note that the reduction
    axis (Dr) is not sliced because it is not supported.

    $output tensor resides in registers. It would be of same shape as the input
    tensor because the reduced value will be broadcasted to create the same shape as
    the input tensor.

    $workspace_buffer resides in LDS. This should be a flat tensor with at least
    number of elements in the input tensor. 
  }];
  let assemblyFormat = [{
    $reduceMethod $inputRegView $input `into` $output `using` $workspace_buffer attr-dict `:` type($input) `using` type($workspace_buffer) `into` type($output)
  }];

  let hasVerifier = 1;
}

/// ONLY NEEDEF FOR in_warp_transpose BELOW. DEAD BUT LEFT IN FOR FUTURE
/// USE, JUST LIKE in_warp_transpose ITSELF.

// Whether the vector's length is divisible by `divisor`
class IsVectorOfDivisibleLengthPred<int divisor> :
  And<[IsVectorTypePred,
       CPred<[{$_self.cast<::mlir::VectorType>().getNumElements()
                           % }] # divisor # [{ == 0 }]>]>;

// Any vector where the number of elements is from the given
// `allowedLengths` list
class VectorOfDivisibleLength<int divisor> : Type<
  IsVectorOfDivisibleLengthPred<divisor>,
  " with length divisble by " # divisor,
  "::mlir::VectorType">;
class VectorOfDivisibleLengthAndType<int  divisor,
                          list<Type> allowedTypes> : Type<
  And<[VectorOf<allowedTypes>.predicate,
       VectorOfDivisibleLength<divisor>.predicate]>,
  VectorOf<allowedTypes>.summary # VectorOfDivisibleLength<divisor>.summary,
  "::mlir::VectorType">;

defvar swizzleGroupSize = 4;
def Rock_InWarpTransposeOp :
    Rock_Op<"in_warp_transpose", [AllTypesMatch<["vector", "res"]>]>,
    Arguments<(ins VectorOfDivisibleLengthAndType<swizzleGroupSize,
                                                          [F32, I32]>:$vector,
                  Index:$laneId,
                  ConfinedAttr<I32Attr, [IntMaxValue<swizzleGroupSize>]>:$size,
                  DefaultValuedAttr<ConfinedAttr<I32ArrayAttr,
                        [ArrayCount<swizzleGroupSize>]>,
                    "{0, 1, 2, 3}">:$inGroupPerm)>,
    Results<(outs VectorOfDivisibleLengthAndType<swizzleGroupSize, [F32, I32]>:$res)> {
  let summary = "Transpose blocks of data distributed accross a warp";
  let description = [{
    NOTE: THIS IS CURRENTLY DEAD CODE LEFT IN IN CASE WE FIND A USE FOR IT
    AND TO PROVIDE TESTCASES FOR A PLANNED amdgpu.dpp. YOU CAN, IF NEED
    BE, REMOVE THIS and gpu.warp_swizzle.

    `rock.in_warp_transpose` takes a vector representing a matrix of values
    stored accross the threads in a warp and transposes `size`x`size` blocks
    of this matrix. The rows of the output (if we regard each thread's vector
    as a row in the matrix) are further permuted with the `inGroupPerm`
    permutation.

    It is an ***unchecked invariant*** that laneId contanes the ID of the current
    lane with the wave, and thus is an integer in [0, waveSize).
  }];

  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict $vector `,` $laneId `:` type($vector) `,` type($laneId)
  }];

  let extraClassDeclaration = "static constexpr size_t swizzleGroupSize = "
    # !cast<string>(swizzleGroupSize) # ";\n";
}

#endif // ROCK_OPS
