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
include "mlir/Dialect/Rock/IR/RockGemmFeaturesInterface.td"
include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.td"
include "mlir/Dialect/Rock/IR/RockAcceptingViewOpInterface.td"
include "mlir/Dialect/Rock/IR/RockWriterOpInterface.td"
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

class IndexArrayLength<int n> : ConfinedAttr<IndexArrayAttr, [ArrayMinCount<n>]>;

class Rock_ConvOpBase<string mnemonic, list<Type> inputTypes = [F32, F16, BF16],
                      list<Type> outputTypes = [F32, F16, BF16]>
    : Rock_Op<mnemonic, [DeclareOpInterfaceMethods<RockGemmWrapperInterface>,
                         DeclareOpInterfaceMethods<RockConvInterface>,
                         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                         DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
                         RockFusionRoot]> {
  dag commonConvArgs = (ins TensorOrMemRefOf<inputTypes>:$filter,
      TensorOrMemRefOf<inputTypes>:$input,
      TensorOrMemRefOf<outputTypes>:$output,
      OptionalAttr<Rock_GemmFeaturesAttr>:$features,
      OptionalAttr<I32Attr>:$derivedBlockSize, OptionalAttr<I32Attr>:$gridSize,
      IndexArrayLength<4>:$padding, IndexArrayLength<2>:$strides,
      IndexArrayLength<2>:$dilations,
      OptionalAttr<RockTuningParamAttrInterface>:$params);
  let results = (outs Optional<AnyRankedTensor>:$result);
}

defvar GemmInputTypes = [F32, F16, BF16, I8, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2,
                         F8E4M3FN, F4E2M1FN, F8E8M0FNU];
// This can be extended for quantization.
defvar GemmOutputTypes = [F32, F16, BF16, I32, I8, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2, F8E4M3FN];
defvar GemmAccumulatorTypes = [F32, F16, BF16, I32];

def Rock_ConvOp : Rock_ConvOpBase<"conv", GemmInputTypes, GemmOutputTypes> {
  let arguments = commonConvArgs;
  let summary = "N-D convolution forward";
  let description = [{
    The `rock.conv` op computes N-D convolution forward.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` (`features` `=` $features^)? attr-dict
    `:` type(operands) (`->` type($result)^)?
  }];
}

def Rock_ConvBwdDataOp : Rock_ConvOpBase<"conv_bwd_data">
{
  dag additionalArgs = (ins IndexAttr:$kernelId, BoolAttr:$usesV4R1);
  let arguments = !con(commonConvArgs, additionalArgs);
  let summary = "N-D convolution backward data";
  let description = [{
    The `rock.conv_bwd_data` op computes N-D convolution backward data.

    The kernel ID represents which of the multiple backwards data kernels
    needed for the correct computation of the result this kernel is.

    The boolean `usesV4R1` specifies if the broader compilation flow is using
    the V4R1 algorithm for backwards data convs, which will require multiple
    conv ops to be created to efficiently compute the data.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` (`features` `=` $features^)? attr-dict
    `:` type(operands) (`->` type($result)^)?
  }];
}


def Rock_ConvBwdWeightOp : Rock_ConvOpBase<"conv_bwd_weight">
{
  dag additionalArgs = (ins Optional<TensorOrMemRefOf<[F32]>>:$workspace,
      OptionalAttr<IndexAttr>:$kBlocks);
  let arguments = !con(commonConvArgs, additionalArgs);
  let summary = "N-D convolution backward weight";
  let description = [{
    The `rock.conv_bwd_weight` op computes N-D convolution backward weight.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` operands `)` (`features` `=` $features^)? attr-dict
    `:` type(operands) (`->` type($result)^)?
  }];
}

def Rock_GemmOp
    : Rock_Op<"gemm", [DeclareOpInterfaceMethods<RockGemmWrapperInterface>,
                       DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
                       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                       RockFusionRoot, AttrSizedOperandSegments]>,
      Arguments<(ins Arg<TensorOrMemRefOf<GemmInputTypes>, "matrix A">:$a,
          Arg<TensorOrMemRefOf<GemmInputTypes>, "matrix B">:$b,
          Arg<TensorOrMemRefOf<GemmOutputTypes>, "matrix C">:$c,
          Optional<TensorOrMemRefOf<[F8E8M0FNU, F32]>>:$scaleA,
          Optional<TensorOrMemRefOf<[F8E8M0FNU, F32]>>:$scaleB,
          UnitAttr:$aTransposed, UnitAttr:$bTransposed, UnitAttr:$cTransposed,
          UnitAttr:$aScaleTransposed, UnitAttr:$bScaleTransposed,
          OptionalAttr<Rock_GemmFeaturesAttr>:$features,
          StoreMethodAttr:$storeMethod, OptionalAttr<I32Attr>:$derivedBlockSize,
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

    The parameters `derivedBlockSize`, `gridSize`, and `params` are optional as
    they can be inferred by a tuning process or a heuristic, but they must be
    set before the `gemm` is lowered into the `gridwise_gemm` stage of the code
    generation pipeline.

    `features` specifies what hardware features can be used in the generated code.
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    (`tr` $cTransposed^)? $c `=` (`tr` $aTransposed^)? $a (`scaled` `by` (`tr` $aScaleTransposed^)? $scaleA^)? `*` (`tr` $bTransposed^)? $b (`scaled` `by` (`tr` $bScaleTransposed^)? $scaleB^)?
    (`features` `=` $features^)? `storeMethod` `=` $storeMethod attr-dict
    `:` type($c) `=` type($a) (`scaled` `by` type($scaleA)^)? `*` type($b) (`scaled` `by` type($scaleB)^)? (`->` type($result)^)?
  }];
}

def Rock_ReduceOp
    : Rock_Op<"reduce", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                         AllElementTypesMatch<["in", "out"]>]>,
      Arguments<(ins Arg<TensorOrMemRefOf<[BF16, F16, F32]>, "input">:$in,
          Arg<TensorOrMemRefOf<[BF16, F16, F32]>, "output">:$out,
          ReduceMethodAttr:$reduceMethod, IndexAttr:$axis, I32Attr:$blockSize,
          I32Attr:$gridSize, UnitAttr:$useLDS, UnitAttr:$useDPP)>,
      Results<(outs Optional<TensorOf<[BF16, F16, F32]>>:$result)> {
  let summary = "Axes-wide reduction operation";
  let hasVerifier = 1;
  let assemblyFormat = [{
    $reduceMethod $in `into` $out attr-dict `:` type($in) `into` type($out) (`->` type($result)^)?
  }];
  let extraClassDeclaration = [{
    ::mlir::OpOperand* getOutArgument() { return &(*this)->getOpOperand(1); }

    SmallVector<::mlir::Type> getTypesForFeature() {
      return {getIn().getType()};
    }
  }];
}
def Rock_AttentionOp
    : Rock_Op<
          "attention", [DeclareOpInterfaceMethods<RockGemmGemmWrapperInterface>,
                        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                        DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
                        RockFusionRoot, AttrSizedOperandSegments,
                        AttrSizedResultSegments]>,
      Arguments<(ins TensorOrMemRefOf<[F32, F16, BF16, I8]>:$queries,
          TensorOrMemRefOf<[F32, F16, BF16, I8]>:$keys,
          TensorOrMemRefOf<[F32, F16, BF16]>:$values,
          Variadic<AnyTensorOrMemRef>:$preSoftmaxElemWiseInputs,
          Optional<TensorOrMemRefOf<[I32]>>:$currentSeqLen,
          Optional<TensorOrMemRefOf<[I32]>>:$prefixOffset,
          TensorOrMemRefOf<[F32, F16, BF16]>:$out,
          Optional<TensorOrMemRefOf<[F32, F16, BF16]>>:$lse, I32Attr:$numHeadsQ,
          I32Attr:$numHeadsKV, UnitAttr:$qTransposed, UnitAttr:$kTransposed,
          UnitAttr:$vTransposed, UnitAttr:$oTransposed, UnitAttr:$causal,
          I32Attr:$splitKV, OptionalAttr<Rock_GemmFeaturesAttr>:$features,
          StoreMethodAttr:$storeMethod, OptionalAttr<TypeAttr>:$softmaxType,
          OptionalAttr<RockTuningParamAttrInterface>:$params0,
          OptionalAttr<RockTuningParamAttrInterface>:$params1,
          DenseI64ArrayAttr:$firstGemmIndices,
          DefaultValuedOptionalAttr<BoolAttr,
                                    "false">:$preSoftmaxHasSplitKVTransforms)>,
      Results<(outs Optional<TensorOf<[F32, F16, BF16]>>:$result,
          Optional<TensorOf<[F32, F16, BF16]>>:$lseOut)> {
  let summary = "Attention operation of transformer models";
  let description = [{
    Performs the operation out = SOFTMAX(preSoftmaxBody(queries * keys, preSoftmaxElemWiseInputs)) * values.

    This operation performs attention mechanism of transformer models. There is an optional element-wise 
    fusion just before the softmax, defined by `preSoftmaxBody` with inputs `preSoftmaxElemWiseInputs`.

    If none of the `transposed` attributes are set, then `queries` is [G] x seq_q x head_qk,
    `keys` is [G] x head_qk x seq_k, `values` is [G] x seq_k x head_v and `out` is [G] x seq_q x head_v, 
    where G is the optional group dimension (which is assumed to be 1 if not set).

    The transpose attributes allow for the non-group dimensions of the matrix to be
    transposed. For example, if `qTransposed` is set, then the argument `queries` should be
    a [G] x head_qk x seq_q memory.

    If causal is enabled, we implement causal masking.

    If causal is enabled AND prefixOffset is not null, we implement prefix
    causal masking where: mask = key_pos > (query_pos + offset).
    The prefixOffset can be either:
    - A scalar (shape [1]): a single global offset applied to all sequences
    - A tensor of shape [G]: per-group/batch offsets, allowing different prefix
      lengths for each sequence in the batch

    LSE (log-sum-exp) is an optional output typically used for flash decoding.
    For flash decoding, you can pass splitKV > 1, the default value is 1, which means flash decoding is disabled. 
    Flash decoding multiplies the number of blocks by splitKV. Note that "lse" has to be non-null for splitKV > 1. 
    However, "lse" can be non-null even if splitKV = 1.

    The `preSoftmaxHasSplitKVTransforms` attribute indicates whether the preSoftmaxBody inputs 
    have splitKV transforms applied. It is set to true by DetectFlashDecoding when it detects 
    and removes splitKV patterns. This prevents GridwiseGemmToBlockwise from unnecessarily 
    re-applying splitKV transforms when the IR is generated directly (e.g., via rocmlir-gen) 
    without flash decoding detection.

    The parameters `gridSize`, and `blockSize` are optional as they can be
    inferred by a tuning process or a heuristic, but they must be set before
    the `attention` is lowered into the `gridwise_attention_accel` stage of the
    code generation pipeline.

    `features` specifies what hardware features can be used in the generated code.
  }];
  let hasVerifier = 1;
  let regions = (region AnyRegion:$preSoftmaxBody);
  let assemblyFormat = [{
    `{` `\n`
        ` ` `qk` `=` (`tr` $qTransposed^)? $queries `*` (`tr` $kTransposed^)? $keys `:` type($queries) `,` type($keys) `\n`
        (`currentSeqLen` `=` `(` $currentSeqLen^ `:` type($currentSeqLen) `)` `\n`)?
        (`prefixOffset` `=` `(` $prefixOffset^ `:` type($prefixOffset) `)` `\n`)?
        (`causal` `\n` $causal^)?
        (`lse` `=` $lse^ `:` type($lse) `\n`)?
        (`qk` `=` `elementwise` (`otherIns` `(` $preSoftmaxElemWiseInputs^ `:` type($preSoftmaxElemWiseInputs) `)`)? $preSoftmaxBody^ `\n`)?
        (`tr` $oTransposed^)? $out `=` `softmax` `(` `qk` `)` `*` (`tr` $vTransposed^)? $values `:` type($values) `->` type($out) `\n`
    `}` attr-dict (`->` type($result)^)? (`,` type($lseOut)^)?
  }];

  // Return operand of LSE if it exists.
  let extraClassDeclaration = [{
    ::mlir::OpOperand* getOutLseArgument() { return getLse() ? &(*this)->getOpOperand(getNumOperands() - 1) : nullptr; }
  }];
}

def Rock_GemmElementwiseGemmOp
    : Rock_Op<"gemm_elementwise_gemm",
              [DeclareOpInterfaceMethods<RockGemmGemmWrapperInterface>,
               DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
               DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
               RockFusionRoot]>,
      AllElementTypesMatch<["a", "b"]>,
      Arguments<(ins TensorOrMemRefOf<[F32, F16, BF16]>:$a,
          TensorOrMemRefOf<[F32, F16, BF16]>:$b,
          TensorOrMemRefOf<[F32, F16, BF16]>:$c,
          Variadic<AnyTensorOrMemRef>:$elemwiseInputs,
          TensorOrMemRefOf<[F32, F16, BF16]>:$out, UnitAttr:$aTransposed,
          UnitAttr:$bTransposed, UnitAttr:$cTransposed, UnitAttr:$oTransposed,
          OptionalAttr<Rock_GemmFeaturesAttr>:$features,
          StoreMethodAttr:$storeMethod,
          OptionalAttr<RockTuningParamAttrInterface>:$params0,
          OptionalAttr<RockTuningParamAttrInterface>:$params1,
          DenseI64ArrayAttr:$firstGemmIndices)>,
      Results<(outs Optional<TensorOf<[F32, F16, BF16]>>:$result)> {
  let summary = "GEMM-elementwise-GEMM operation";
  let description = [{
    Performs the operation out = preSecondGemmBody(a * b, elemwiseInputs) * c.

    This operation performs fused GEMM-elementwise-GEMM. There is an optional element-wise
    fusion just before the second GEMM, defined by `preSecondGemmBody` with inputs `elemwiseInputs`.

    If none of the `transposed` attributes are set, then `a` is [G] x M x K,
    `b` is [G] x K x N, `c` is [G] x N x O and `out` is [G] x M x O, where G is the 
    optional group dimension (which is assumed to be 1 if not set).

    The transpose attributes allow for the non-group dimensions of the matrix to be
    transposed. For example, if `aTransposed` is set, then the argument `a` should be
    a [G] x K x M memory.

    The parameters `gridSize`, and `blockSize` are optional as they can be
    inferred by a tuning process or a heuristic, but they must be set before
    the `gemm_elementwise_gemm` is lowered into the `gridwise_attention_accel`
    stage of the code generation pipeline.

    `features` specifies what hardware features can be used in the generated code.
  }];
  let hasVerifier = 1;
  let regions = (region AnyRegion:$preSecondGemmBody);
  let assemblyFormat = [{
    `{` `\n`
        ` ` `ab` `=` (`tr` $aTransposed^)? $a `*` (`tr` $bTransposed^)? $b `:` type($a) `,` type($b) `\n`
        (`ab` `=` `elementwise` (`otherIns` `(` $elemwiseInputs^ `:` type($elemwiseInputs) `)`)? $preSecondGemmBody^ `\n`)?
        (`tr` $oTransposed^)? $out `=` `ab` `*` (`tr` $cTransposed^)? $c `:` type($c) `->` type($out) `\n`
    `}` attr-dict (`->` type($result)^)?
  }];
}

def Rock_ConvElementwiseGemmOp
    : Rock_Op<"conv_elementwise_gemm",
              [DeclareOpInterfaceMethods<RockGemmGemmWrapperInterface>,
               DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
               DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
               RockFusionRoot]>,
      AllElementTypesMatch<["filter", "input"]>,
      Arguments<(ins TensorOrMemRefOf<[F32, F16, BF16]>:$filter,
          TensorOrMemRefOf<[F32, F16, BF16]>:$input,
          TensorOrMemRefOf<[F32, F16, BF16]>:$c,
          Variadic<AnyTensorOrMemRef>:$elemwiseInputs,
          TensorOrMemRefOf<[F32, F16, BF16]>:$out, UnitAttr:$cTransposed,
          UnitAttr:$oTransposed, OptionalAttr<Rock_GemmFeaturesAttr>:$features,
          StoreMethodAttr:$storeMethod, IndexArrayLength<4>:$padding,
          IndexArrayLength<2>:$strides, IndexArrayLength<2>:$dilations,
          OptionalAttr<RockTuningParamAttrInterface>:$params0,
          OptionalAttr<RockTuningParamAttrInterface>:$params1,
          DenseI64ArrayAttr:$firstGemmIndices)>,
      Results<(outs Optional<TensorOf<[F32, F16, BF16]>>:$result)> {
  let summary = "CONV-elementwise-GEMM operation";
  let description = [{
    Performs the operation out = preSecondGemmBody(conv_forward(filter, input), elemwiseInputs) * c.

    This operation performs fused CONV-elementwise-GEMM. There is an optional element-wise
    fusion just before the second GEMM, defined by `preSecondGemmBody` with inputs `elemwiseInputs`.

    If none of the `transposed` attributes are set, then `c` is [G] x N x O and `out` is [G] x M x O, 
    where G is the optional group dimension (which is assumed to be 1 if not set). Note that input and
    filter layouts are defined by attributes "input_layout" and "filter_layout".

    The transpose attributes allow for the non-group dimensions of the matrix to be
    transposed. For example, if `cTransposed` is set, then the argument `c` should be
    a [G] x O x M memory.

    The parameters `gridSize`, and `blockSize` are optional as they can be
    inferred by a tuning process or a heuristic, but they must be set before
    the `conv_elementwise_gemm` is lowered into the `gridwise_attention_accel`
    stage of the code generation pipeline.

    `features` specifies what hardware features can be used in the generated code.
  }];
  let hasVerifier = 1;
  let regions = (region AnyRegion:$preSecondGemmBody);
  let assemblyFormat = [{
    `{` `\n`
        ` ` `ab` `=` `conv` `(` $filter `,` $input `)` `:` type($filter) `,` type($input) `\n`
        (`ab` `=` `elementwise` (`otherIns` `(` $elemwiseInputs^ `:` type($elemwiseInputs) `)`)? $preSecondGemmBody^ `\n`)?
        (`tr` $oTransposed^)? $out `=` `ab` `*` (`tr` $cTransposed^)? $c `:` type($c) `->` type($out) `\n`
    `}` attr-dict (`->` type($result)^)?
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
     ShapedType lowerType = cast<ShapedType>(input.getType());
     Type elemType = lowerType.getElementType();
     ShapedType upperType;
     if (auto mrType = dyn_cast<MemRefType>(lowerType)) {
       auto memorySpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(mrType.getMemorySpace());
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
    %result = rock.conv(..., %output) : ..., U
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

// gridwise_gemm_accel
def Rock_GridwiseGemmAccelOp
    : Rock_Op<"gridwise_gemm_accel",
              [DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
               DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
               RockFusionRoot, AttrSizedOperandSegments]>,
      Arguments<(ins Arg<MemRefRankOf<GemmInputTypes, [3]>, "matrix A view">:$a,
          Arg<MemRefRankOf<GemmInputTypes, [3]>, "matrix B view">:$b,
          Arg<MemRefRankOf<GemmAccumulatorTypes, [3]>, "matrix C view">:$c,
          Optional<MemRefRankOf<[F8E8M0FNU], [3]>>:$scaleA,
          Optional<MemRefRankOf<[F8E8M0FNU], [3]>>:$scaleB,
          OptionalAttr<Rock_GemmFeaturesAttr>:$features,
          StoreMethodAttr:$storeMethod, I32Attr:$blockSize, I32Attr:$gridSize,
          RockAccelTuningParamAttrInterface:$params)> {
  let summary = "Gridwise GEMM accelerated version";
  let description = [{
    The `rock.gridwise_gemm` op computes gridwise GEMM with acceleration.
  }];
  let assemblyFormat = [{
    `(` $a `,` $b `,` $c
        (`,` $scaleA^)? (`,` $scaleB^)? `)`
    `storeMethod` `(` $storeMethod `)`
    (`features` `=` $features^)? attr-dict
    `:` type($a) `,` type($b) `,` type($c)
       (`,` type($scaleA)^)? (`,` type($scaleB)^)?
  }];
  let hasVerifier = 1;
}

// gridwise_attention_accel
def Rock_GridwiseAttentionAccelOp
    : Rock_Op<"gridwise_attention_accel",
              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
               DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
               RockFusionRoot, AttrSizedOperandSegments,
]>,
      Arguments<(ins MemRefRankOf<[F32, F16, BF16, I8], [3]>:$queries,
          MemRefRankOf<[F32, F16, BF16, I8], [3]>:$keys,
          MemRefRankOf<[F32, F16, BF16], [3]>:$values,
          Variadic<AnyTensorOrMemRef>:$preSoftmaxElemWiseInputs,
          Optional<MemRefRankOf<[I32], [1]>>:$currentSeqLen,
          Optional<MemRefRankOf<[I32], [1]>>:$prefixOffset,
          MemRefRankOf<[F32, F16, BF16], [3]>:$out,
          Optional<MemRefRankOf<[F32, F16, BF16], [2]>>:$lse, UnitAttr:$causal,
          I32Attr:$splitKV, OptionalAttr<Rock_GemmFeaturesAttr>:$features,
          StoreMethodAttr:$storeMethod, I32Attr:$blockSize, I32Attr:$gridSize,
          UnitAttr:$disableQBypassLDS, OptionalAttr<IndexAttr>:$prePadG0M,
          OptionalAttr<IndexAttr>:$prePadG0N,
          OptionalAttr<IndexAttr>:$numRepeatsGQA,
          OptionalAttr<TypeAttr>:$softmaxType,
          RockAccelTuningParamAttrInterface:$params0,
          RockAccelTuningParamAttrInterface:$params1,
          DenseI64ArrayAttr:$firstGemmIndices,
          DefaultValuedOptionalAttr<BoolAttr, "true">:$enableSoftmax,
          DefaultValuedOptionalAttr<BoolAttr,
                                    "false">:$preSoftmaxHasSplitKVTransforms)> {
  let summary = "Gridwise attention accelerated version";
  let description = [{
    The `rock.gridwise_attention_accel` op computes gridwise attention with acceleration.
  }];
  let regions = (region AnyRegion:$preSoftmaxBody);
  let assemblyFormat = [{
    `(` operands `)` (`features` `=` $features^)? `preSoftmaxOps` `=` $preSoftmaxBody attr-dict `:` type(operands)
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
  let hasVerifier = 1;
}

// TBD: eventually replace this with linalg.fill?
def Rock_FillOp:
    Rock_Op<"fill">,
    Arguments<(ins AnyMemRef:$input,
                   //AnyTypeOf<[AnyInteger, AnyFloat]>:$value)> {
                   AnyTypeOf<[AnyInteger, AnyFloat, AnyVectorOfNonZeroRank]>:$value)> {
  let summary = "Fill memory with constant value on GPU";
  let description = [{
    The `rock.fill` op fills a memref on GPU with a constant value.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

def Rock_WorkgroupIdOp:
    Rock_Op<"workgroup_id", [Pure,
      DeclareOpInterfaceMethods<InferIntRangeInterface,["inferResultRanges"]>]>,
    Results<(outs I32:$id)> {
  let summary = "Get current workgroup ID";
  let description = [{
    The `rock.workgroup_id` op gets the current workgroup ID.
  }];
  let assemblyFormat = "attr-dict `:` type($id)";
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

  let builders = [OpBuilder<(ins),
                            [{ build($_builder, $_state, ValueRange{}); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

defvar SupportedMemoryElems = [F32, F16, BF16, I1, I<4>, I8, I16, I32,
                               F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2, F8E4M3FN,
                               F4E2M1FN, F8E8M0FNU];
defvar NativeMemoryOpTypes =
    [F32, F16, BF16, I<4>, I8, I16, I32, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2,
     F8E4M3FN, F4E2M1FN, F8E8M0FNU,
     VectorOfLengthAndType<[2, 4, 8, 16, 32], [F32, I32, F4E2M1FN, F8E8M0FNU]>,
     VectorOfLengthAndType<[2, 4, 8, 16], [F16, BF16]>,
     VectorOfLengthAndType<[2, 4, 8, 16, 32, 64], [I8, F8E5M2FNUZ, F8E4M3FNUZ,
                                                   F8E5M2, F8E4M3FN]>];

defvar LdsBufferTypes = GemmInputTypes # [VectorOfRankAndType<[1], GemmInputTypes>];

// This is less general than the one in the AMDGPU dialect, since it only
// accounts for how we call these operations.
defvar AccelArgTypes = [F32, VectorOfLengthAndType<[2, 4], [F32]>,
                        VectorOfLengthAndType<[2, 4, 8, 16], [F16, BF16]>,
                        VectorOfLengthAndType<[4, 8, 16], [I8, F8E5M2FNUZ,
                                                           F8E4M3FNUZ, F8E5M2,
                                                           F8E4M3FN]>,
                        VectorOfLengthAndType<[8, 16, 32], [F4E2M1FN]>,
                        VectorOfLengthAndType<[4, 8, 16, 32], [F8E8M0FNU]>];
defvar AccelResTypes = [VectorOfLengthAndType<[4, 8, 16, 32], [F32, I32, F16, BF16]>];

// blockwise_gemm_accel
def Rock_BlockwiseGemmAccelOp
    : Rock_Op<"blockwise_gemm_accel",
              [AttrSizedOperandSegments,
               DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
               DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>,
      Arguments<(ins MemRefOf<LdsBufferTypes>:$matrixA,
          MemRefOf<LdsBufferTypes>:$matrixB, MemRefOf<LdsBufferTypes>:$matrixC,
          Optional<MemRefOf<LdsBufferTypes>>:$matrixScaleA,
          Optional<MemRefOf<LdsBufferTypes>>:$matrixScaleB)> {
  let summary = "Blockwise GEMM accelerated version";
  let description = [{
    The `rock.blockwise_gemm_accel` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on registers.
    - Matrix C shall be vectors.

    The elements of matrices A and B should be vectors of length kpack, or
    scalars when kpack is 1.
  }];
  let assemblyFormat = [{
    $matrixC `+` `` `=` $matrixA  (`scaled` `by` $matrixScaleA^)? `*`
                        $matrixB (`scaled` `by` $matrixScaleB^)? attr-dict
    `:` type($matrixC) `+` `` `=` type($matrixA) (`scaled` `by` type($matrixScaleA)^)? `*`
                                  type($matrixB) (`scaled` `by` type($matrixScaleB)^)?
  }];
  let hasVerifier = 1;
}

// blockwise_load_tile
def Rock_BlockwiseLoadTileOp
    : Rock_Op<"blockwise_load_tile",
              [DeclareOpInterfaceMethods<RockGemmFeaturesInterface>,
               DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>,
      Arguments<(ins Arg<MemRefOf<NativeMemoryOpTypes>,
                         "source global memory">:$source,
          Arg<MemRefOf<NativeMemoryOpTypes>,
              "destination registers">:$destRegisters,
          Variadic<I32>:$sourceIndices)> {
  let summary =
      "Blockwise load tile from global memory to LDS and/or registers";
  let description = [{
    The `rock.blockwise_load_tile` op does tile loading at workgroup (block) level.
    We will load a tile of data from global memory into LDS and/or registers.

    There are different load types:
    - Default: Creates two stages, (1) load from memory, (2) write to LDS.
    - BypassLDS: Bypasses LDS and loads from device memory to registers directly (only one stage).
    - DoubleBuffer: Creates three stages, (1) load from memory, (2) write to LDS, (3) load to registers.
    - DirectToLDSDefault: Same as Default, but a single stage loads from memory and writes to LDS.
    - DirectToLDSDoubleBuffer: Same as DoubleBuffer, but a single stage loads from memory and writes to LDS.

    `isA` determines if we are loading an A matrix or B matrix. `G`, `M` and `N` are the GEMM sizes.
    `elementTypeA` and `elementTypeB` are used to construct AccelEmitter. They are data types for the Matrix A & B of the GEMMs. 
    `elementLoadType` is the element type of the global buffer this BlockwiseLoadTileOp is trying to load. 
    `elementType` is the elementType in the registers.  Note that it may differ from `elementLoadType` because of input fusions.  
  }];
  let assemblyFormat = [{
    $source (`[` $sourceIndices^ `]`)? `->` $destRegisters attr-dict
    `:` type($source) `->` type($destRegisters)
  }];
}

// blockwise_load_tile
def Rock_BlockwiseLoadTilePtrOp
    : Rock_Op<"blockwise_load_tile_ptr",
              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>,
      Arguments<(ins Arg<AnyMemRef, "Pointer tensor">:$pointerTensor,
          Arg<AnyMemRef, "Mask tensor">:$maskTensor,
          Arg<MemRefOf<NativeMemoryOpTypes>,
              "destination registers">:$destRegisters)> {
  let summary =
      "Blockwise load tile from global memory to LDS and/or registers";
  let description = [{
  }];
  let assemblyFormat = [{
    $pointerTensor `[` $maskTensor `]` `->` $destRegisters attr-dict
    `:` type($pointerTensor) `,` type($maskTensor) `->` type($destRegisters)
  }];
}

// blockwise_store_tile
def Rock_BlockwiseStoreTileOp
    : Rock_Op<"blockwise_store_tile",
              [DeclareOpInterfaceMethods<RockAcceptingViewOpInterface>,
               DeclareOpInterfaceMethods<RockWriterOpInterface>,
               DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>,
      AllElementTypesMatch<["source", "dest"]>,
      Arguments<(ins Arg<MemRefOf<NativeMemoryOpTypes>,
                         "source registers">:$source,
          Arg<MemRefOf<NativeMemoryOpTypes>, "detination view">:$dest,
          Variadic<I32>:$extraIndices,
          StoreMethodAttr:$storeMethod)> {
  let summary = "Write out values in registers to transformed destination";

  let description = [{
  }];

  let assemblyFormat = [{
    attr-dict $source  `->` $dest (`[` $extraIndices^ `]`)? `by` $storeMethod
    `:` type($source) `->` type($dest)
  }];
  let hasVerifier = 1;
}

// blockwise_store_tile
def Rock_BlockwiseStoreTilePtrOp
    : Rock_Op<"blockwise_store_tile_ptr",
              []>,
      Arguments<(ins Arg<AnyMemRef, "Pointer tensor">:$pointerTensor,
          Arg<AnyMemRef, "Mask tensor">:$maskTensor,
          Arg<MemRefOf<NativeMemoryOpTypes>,
              "source registers">:$source, StoreMethodAttr:$storeMethod)> {
  let summary = "Write out values in registers to transformed destination";

  let description = [{
  }];

  let assemblyFormat = [{
    attr-dict $source  `->` $pointerTensor `(` $maskTensor `)` `by` $storeMethod
    `:` type($source) `->` type($pointerTensor) `(` type($maskTensor) `)`
  }];
}

def Rock_TransformsToPtrOp
    : Rock_Op<"transforms_to_ptr",
              []>,
      Arguments<(ins Arg<MemRefOf<NativeMemoryOpTypes>,
                         "source">:$source,
          Arg<AnyMemRef, "pointers">:$pointers,
          Arg<AnyMemRef, "mask">:$mask, Variadic<I32>:$extraIndices)> {
  let summary = "Write out values in registers to transformed destination";

  let description = [{
    ```
  }];

  let assemblyFormat = [{
    attr-dict $source  `->` $pointers `,` $mask (`[` $extraIndices^ `]`)?
    `:` type($source) `->` type($pointers) `,` type($mask)
  }];
}

def Rock_MakeRangeOp : 
    Rock_Op<"make_range", []> {
  let summary = "Create a 1D tensor range";

  let description = [{
    The `rock.make_range` op creates a 1D tensor with integer values 
    spanning from `start` to `end` (exclusive), with step = 1.
    
    Values: [start, start+1, start+2, ..., end-1]
    
    This operation is used in the tensor-based Rock dialect for 
    generating indices and coordinate ranges for GPU computations.
  }];

  let arguments = (ins Arg<AnyMemRef,
                         "source">:$out,
    I32Attr:$start,
    I32Attr:$end
  );

  let assemblyFormat = [{
    $start `to` $end `->` $out attr-dict `:` type($out)
  }];
}

// blockwise_broadcasting_reduction
def Rock_BlockwiseBroadcastReduceOp:
    Rock_Op<"blockwise_broadcast_reduce">,
    Arguments<(ins MemRefOf<GemmOutputTypes>:$input,
                   MemRefOf<LdsBufferTypes>:$workspace_buffer,
                   MemRefOf<GemmOutputTypes>:$output,
                   Optional<MemRefOf<GemmOutputTypes>>:$extraOut,
                   IndexAttr:$axis,
                   ReduceMethodAttr:$reduceMethod,
                   TransformMapArrayAttr:$inputRegView,
                   TransformMapArrayAttr:$tidSubTileSliceView,
                   TransformMapArrayAttr:$iterSubTileSliceView,
                   OptionalAttr<TransformMapArrayAttr>:$extraOutView,
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

    {$extraOut,$extraOutView} optionally be used to broadcast the output in a different
    layout that is different to the input layout.
    $extraOut and $extraOutView are an optional argument and a view stack, respectively,
    that still needs to have upper dimensions equal to the input.
  }];
  let assemblyFormat = [{
    $reduceMethod $inputRegView $tidSubTileSliceView $iterSubTileSliceView $input `into` $output(`,` $extraOutView $extraOut^)? `using` $workspace_buffer attr-dict `:` type($input) `using` type($workspace_buffer) `into` type($output)( `,` type($extraOut)^)?
  }];

  let hasVerifier = 1;
}

def Rock_SplatOp : Rock_Op<"splat", [Pure]> {
  let summary = "Splat a scalar to a memref";

  let description = [{
    For a given scalar value, splat creates a memref filled with that value,
    e.g. %scalar : i32 -> memref<1x64xi32>.
  }];

  let arguments = (ins AnyType:$src);

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def Rock_ArithOp : Rock_Op<"arith_op", []> {
  let summary = "Generic arithmetic operation wrapper";

  let description = [{
    A generic wrapper for arithmetic operations that will be lowered
    when converting memrefs to tensors. The 'name' attribute specifies
    which arithmetic operation to perform.
  }];

  let arguments = (ins StrAttr:$name, OptionalAttr<AnyAttr>:$constantValue, Variadic<AnyType>:$operands);

  let results = (outs AnyType:$result);

  let assemblyFormat = "`{` $name `}` $operands attr-dict `:` functional-type($operands, $result)";
}

def Rock_BroadcastOp : Rock_Op<"broadcast", [Pure]> {
  let summary = "Broadcast a memref to a new shape";

  let description = [{
    For a given memref, broadcast changes one or more dimensions with size 1
    to a new size, e.g. memref<1x32x1xf32> -> memref<2x32x4xf32>. You cannot
    change the size of a non-1 dimension.
  }];

  let arguments = (ins AnyMemRef:$src);

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def Rock_CastToPtrOp : Rock_Op<"cast_to_ptr", [Pure]> {
  let summary = "Cast a tensor to a pointer tensor";

  let description = [{
    Casts a tensor to a pointer tensor representation.
  }];

  let arguments = (ins AnyRankedTensor:$src);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif // ROCK_OPS
