//===- MIGraphXOps.td - MIGraphX operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIGraphX operations.
// https://rocmsoftwareplatform.github.io/AMDMIGraphX/doc/html/dev/operators.html
//
//===----------------------------------------------------------------------===//

#ifndef MIGRAPHX_OPS
#define MIGRAPHX_OPS

include "mlir/Dialect/MIGraphX/MIGraphXTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MIGraphX_Dialect : Dialect {
  let name = "migraphx";
  let cppNamespace = "mlir::migraphx";
}

// Base class for MIGraphX dialect ops.
class MIGraphX_Op<string mnemonic, list<Trait> traits = []> :
    Op<MIGraphX_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.

  let extraClassDeclaration = [{
  }];
}

// Elementwise binary operations

def MIGraphX_AddOp :
    MIGraphX_Op<"add", [Commutative]>,
    Arguments<(ins AnyRankedTensor:$inA,
                   AnyRankedTensor:$inB)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise binary add";
  let description = [{
    Add two tensors elementwise
  }];
  let assemblyFormat = "`(`$inA `,` $inB`)` attr-dict `:` `(`type($inA) `,` type($inB) `)` `->` type($output)";
}

def MIGraphX_SubOp :
    MIGraphX_Op<"sub">,
    Arguments<(ins AnyRankedTensor:$inA,
                   AnyRankedTensor:$inB)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise binary sub";
  let description = [{
    Subtract two tensors elementwise
  }];
  let assemblyFormat = "`(`$inA `,` $inB`)` attr-dict `:` `(`type($inA) `,` type($inB) `)` `->` type($output)";
}

def MIGraphX_MulOp :
    MIGraphX_Op<"mul", [Commutative]>,
    Arguments<(ins AnyRankedTensor:$inA,
                   AnyRankedTensor:$inB)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise binary mul";
  let description = [{
    Multiply two tensors elementwise
  }];
  let assemblyFormat = "`(`$inA `,` $inB`)` attr-dict `:` `(`type($inA) `,` type($inB) `)` `->` type($output)";
}

def MIGraphX_ClipOp :
    MIGraphX_Op<"clip">,
    Arguments<(ins AnyRankedTensor:$x,
      AnyRankedTensor:$minVals,
      AnyRankedTensor:$maxVals)>,
    Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise clip";
  let description = [{
    Elementwise clip: output = min(max(x, minVals), maxVals)
  }];
  let assemblyFormat = [{
    `(`$x `,` $minVals `,` $maxVals`)` attr-dict
    `:` `(`type($x) `,` type($minVals) `,` type($maxVals)`)` `->` type($output)
  }];
}

def MIGraphX_DivOp :
    MIGraphX_Op<"div">,
    Arguments<(ins AnyRankedTensor:$inA,
                   AnyRankedTensor:$inB)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise binary div";
  let description = [{
    Divide two tensors elementwise
  }];
  let assemblyFormat = "`(`$inA `,` $inB`)` attr-dict `:` `(`type($inA) `,` type($inB) `)` `->` type($output)";
}

def MIGraphX_PowOp :
    MIGraphX_Op<"pow">,
    Arguments<(ins AnyRankedTensor:$inA,
                   AnyRankedTensor:$inB)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise pow";
  let description = [{
    Calculate power.
  }];
  let assemblyFormat = "`(`$inA `,` $inB`)` attr-dict `:` `(`type($inA) `,` type($inB) `)` `->` type($output)";
}

// Note: when lowering to kernel calls, MIGraphX represents booleans as i8.
// Keep that logic here.
def MIGraphX_WhereOp :
    MIGraphX_Op<"where">,
    Arguments<(ins TensorOf<[I8]>:$cond,
      AnyRankedTensor:$inA,
      AnyRankedTensor:$inB)>,
    Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise select";
  let description = [{
    output[x] = cond[x] ? inA[x] : inB[x]
  }];
  let assemblyFormat = [{
    `(`$cond `,` $inA `,` $inB`)` attr-dict
    `:` `(`type($cond) `,` type($inA) `,` type($inB)`)` `->` type($output)
  }];
}


// Elementwise unary operations

def MIGraphX_AbsOp :
    MIGraphX_Op<"abs">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise absolute value";
  let description = [{
    Take absolute value elementwise
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_CeilOp :
    MIGraphX_Op<"ceil">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise ceil";
  let description = [{
    Ceil tensor elementwise
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_ConvertOp :
    MIGraphX_Op<"convert">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise type conversion";
  let description = [{
    Type conversion. Due to impedance mismatches between MIGraphX and Tosa,
    currently only supports float to float conversions
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_ErfOp :
    MIGraphX_Op<"erf">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Gauss error function";
  let description = [{
    compute gauss error function
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_ExpOp :
    MIGraphX_Op<"exp">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Exponential";
  let description = [{
    get exp
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_FloorOp :
    MIGraphX_Op<"floor">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise floor";
  let description = [{
    Floor tensor elementwise
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_LogOp :
    MIGraphX_Op<"log">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Logarithm";
  let description = [{
    get log
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_NegOp :
    MIGraphX_Op<"neg">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Negation";
  let description = [{
    get neg
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_RecipOp :
    MIGraphX_Op<"recip">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise reciprocal";
  let description = [{
    reciprocal elementwise
  }];
  let hasFolder = 1;
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_RsqrtOp :
    MIGraphX_Op<"rsqrt">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise rsqrt";
  let description = [{
    Reciprocal square root elementwise
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_SigmoidOp :
    MIGraphX_Op<"sigmoid">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Sigmoid activation function";
  let description = [{
    Sigmoid function, aka 1 / (1 + exp(-x)).
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_SqrtOp :
    MIGraphX_Op<"sqrt">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Elementwise sqrt";
  let description = [{
    square root elementwise
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

def MIGraphX_TanhOp :
    MIGraphX_Op<"tanh">,
    Arguments<(ins AnyRankedTensor:$inA)>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "Hyperbolic Tan";
  let description = [{
    get tanh
  }];
  let assemblyFormat = "`(`$inA`)` attr-dict `:` `(`type($inA)`)` `->` type($output)";
}

// Quantization operations.

def MIGraphX_QuantizeLinearOp :
    MIGraphX_Op<"quantizelinear">,
    Arguments<(ins TensorOf<[F32, F16]>:$input,
                   TensorOf<[F32]>:$scale,
                   Optional<TensorOf<[AnyInteger]>>:$bias)>,
	  Results<(outs TensorOf<[AnyInteger]>:$output)> {
  let summary = "Channelwise quantization";
  let description = [{
    Quantization tensor channelwise. It computes the following:
    tensor[n,c,h,w] = tensor[n,c,h,w] / quantScale[c] + quantBias[c]
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` `(`type(operands)`)` `->` type(results)";
  let hasVerifier = 1;
}

def MIGraphX_DeQuantizeLinearOp :
    MIGraphX_Op<"dequantizelinear">,
    Arguments<(ins TensorOf<[AnyInteger]>:$input,
                   TensorOf<[F32]>:$scale,
                   Optional<TensorOf<[AnyInteger]>>:$bias)>,
	  Results<(outs TensorOf<[F32, F16]>:$output)> {
  let summary = "Channelwise dequantization";
  let description = [{
    De-Quantization tensor channelwise. It computes the following:
    tensor[n,c,h,w] = (tensor[n,c,h,w] - quantBias[c]) * quantScale[c]
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` `(`type(operands)`)` `->` type(results)";
  let hasVerifier = 1;
}

// Convolution operations
class MIGraphX_ConvOpBase<string mnemonic, list<Type> inputTypes=[], list<Type> outputTypes=[]> :
    MIGraphX_Op<mnemonic>{
    let arguments = (ins TensorOf<inputTypes>:$input,
                         TensorOf<inputTypes>:$filter,

                         I64ArrayAttr:$padding,
                         I64ArrayAttr:$stride,
                         I64ArrayAttr:$dilation,
                         I64Attr:$group,
                         OptionalAttr<I64Attr>:$padding_mode,
                         OptionalAttr<BoolAttr>:$xdlopsV2,
                         OptionalAttr<StrAttr>:$perf_config);
	  let results = (outs TensorOf<outputTypes>:$output);
    let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Value":$filter, "::mlir::ArrayAttr":$padding,
      "::mlir::ArrayAttr":$stride, "::mlir::ArrayAttr":$dilation, "::mlir::IntegerAttr":$group,
      "::mlir::migraphx::padding_mode_tAttr":$padding_mode,
      "::mlir::BoolAttr":$xdlopsV2, "::mlir::StringAttr":$perf_config),
    [{
      auto inType = input.getType().cast<ShapedType>().getElementType();
      auto inShape = input.getType().cast<ShapedType>(); //NHWC
      auto filterShape = filter.getType().cast<ShapedType>(); //kHWC
      // output shape = kHWN
      auto outputType = RankedTensorType::get(
        {inShape.getDimSize(0),
        inShape.getDimSize(1),
        inShape.getDimSize(2),
        filterShape.getDimSize(0)}, inType);
      return build($_builder, $_state, outputType, input, filter, padding, stride, dilation,
        group, padding_mode, xdlopsV2, perf_config);
    }]>];

  let assemblyFormat = "`(`$input `,` $filter`)` attr-dict `:` `(`type($input) `,` type($filter) `)` `->` type($output)";
}

def MIGraphX_QuantConvolutionOp :
    MIGraphX_ConvOpBase<"quant_convolution", [I8], [I32]>{
  let summary = "quantized convolution forward";
  let description = [{
    The `migraphx.quant_convolution` op computes quantized convolution forward.
  }];
}

def MIGraphX_ConvolutionOp :
    MIGraphX_ConvOpBase<"convolution", [F32, F16, BF16], [F32, F16, BF16]>{
  let summary = "convolution forward";
  let description = [{
    The `migraphx.convolution` op computes convolution forward.
  }];
}

def MIGraphX_BatchNormOp :
    MIGraphX_Op<"batch_norm_inference">,
    Arguments<(ins AnyRankedTensor:$input,
                   AnyRankedTensor:$a,
                   AnyRankedTensor:$b,
                   AnyRankedTensor:$c,
                   AnyRankedTensor:$d,

                   F32Attr:$epsilon,
                   F32Attr:$momentum,
                   I64Attr:$bn_mode
                   )>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "batch_norm_inference";
  let description = [{
    The `migraphx.batch_norm_inference` op computes batch_norm.
  }];

  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands) `->` type(results)";
}

def MIGraphX_ReluOp :
    MIGraphX_Op<"relu">,
    Arguments<(ins AnyRankedTensor:$input
                   )>,
	  Results<(outs AnyRankedTensor:$output)> {
  let summary = "RELU activation";
  let description = [{
    The `migraphx.relu` op computes relu op.
  }];
  let assemblyFormat = "`(`$input`)` attr-dict `:` `(`type($input)`)` `->` type($output)";
}

def MIGraphX_PadOp :
    MIGraphX_Op<"pad">,
    Arguments<(ins AnyRankedTensor:$input,

                   I64ArrayAttr:$pads,
                   F32Attr:$value,
                   PadOpModeAttr:$mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "Pad operation";
  let description = [{
    The `migraphx.pad` op adds padding to the tensor.
  }];
  let assemblyFormat = "`(`$input`)` attr-dict `:` `(`type($input)`)` `->` type($output)";
}

def MIGraphX_PoolingOp :
    MIGraphX_Op<"pooling">,
    Arguments<(ins AnyRankedTensor:$input,

                   StrAttr:$mode,
                   I64ArrayAttr:$padding,
                   I64ArrayAttr:$stride,
                   I64ArrayAttr:$length,
                   I64Attr:$ceil_mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "Pooling operation";
  let description = [{
    The `migraphx.pooling` op computes average/max pooling op.
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands) `->` type(results)";
}

def MIGraphX_FlattenOp :
    MIGraphX_Op<"flatten">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64Attr:$axis
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "Flatten tensor";
  let description = [{
    The `migraphx.flatten` op.
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands) `->` type(results)";
}

def MIGraphX_TransposeOp :
    MIGraphX_Op<"transpose">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64ArrayAttr:$permutation
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "transpose dimensions";
  let description = [{
    The `migraphx.transpose` op.
  }];
  let assemblyFormat = "`(`$input`)` attr-dict `:` `(`type($input)`)` `->` type($output)";
}

def MIGraphX_ReshapeOp :
    MIGraphX_Op<"reshape">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64ArrayAttr:$dims
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "reshape a tensor";
  let description = [{
    The `migraphx.reshape` op.
  }];
  let assemblyFormat = "`(`$input`)` attr-dict `:` `(`type($input)`)` `->` type($output)";
}

def MIGraphX_SliceOp :
    MIGraphX_Op<"slice">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64ArrayAttr:$axes,
                   I64ArrayAttr:$ends,
                   I64ArrayAttr:$starts
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "slice a tensor";
  let description = [{
    The `migraphx.slice` op.
  }];
  let assemblyFormat = "`(`$input`)` attr-dict `:` `(`type($input)`)` `->` type($output)";
}

def MIGraphX_BroadcastOp :
    MIGraphX_Op<"broadcast">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64Attr:$axis,
                   I64ArrayAttr:$out_lens
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "Broadcast tensor";
  let description = [{
    The `migraphx.broadcast` op.
  }];
  let assemblyFormat = "`(`$input`)` attr-dict `:` `(`type($input)`)` `->` type($output)";
}

def MIGraphX_MultiBroadcastOp :
    MIGraphX_Op<"multibroadcast">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64ArrayAttr:$out_lens
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "Broadcast tensor in multiple dimensions";
  let description = [{
    The `migraphx.multibroadcast` op.
  }];
  let assemblyFormat = "`(`$input`)` attr-dict `:` `(`type($input)`)` `->` type($output)";
}

class MIGraphX_DotOpBase<string mnemonic, list<Type> inputTypes=[], list<Type> outputTypes=[]> :
    MIGraphX_Op<mnemonic>,
    Arguments<(ins TensorOf<inputTypes>:$in_a,
                   TensorOf<inputTypes>:$in_b
                   )>,
	Results<(outs TensorOf<outputTypes>:$output)> {
  let assemblyFormat = "`(` operands `)` attr-dict `:` `(`type(operands)`)` `->` type(results)";
}

def MIGraphX_QuantDotOp :
    MIGraphX_DotOpBase<"quant_dot", [I8], [I32]>{
  let summary = "Dot product of quantized tensors";
  let description = [{
    The `migraphx.quant_dot` op computes the dot product of two tensors.
  }];
}

def MIGraphX_DotOp :
    MIGraphX_DotOpBase<"dot", [F32, F16, BF16], [F32, F16, BF16]>{
  let summary = "Dot product of tensors";
  let description = [{
    The `migraphx.dot` op computes the dot product of two tensors.
  }];
}

def MIGraphX_ConstantOp :
    MIGraphX_Op<"constant">,
    Arguments<(ins OptionalAttr<ElementsAttr>:$value,
                   OptionalAttr<I64ArrayAttr>:$shape,
                   OptionalAttr<TypeAttr>:$valueType
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "constant tensor operation";
  let description = [{
    The `migraphx.constant` op. Should be able to represent everything from literals.
  }];

  let assemblyFormat = "attr-dict `:` `->` type($output)";
}

def MIGraphX_SoftmaxOp :
    MIGraphX_Op<"softmax">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64Attr:$axis
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "softmax operation";
  let description = [{
    The `migraphx.softmax` op.
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands) `->` type(results)";
}

def MIGraphX_ReduceMeanOp :
    MIGraphX_Op<"reduce_mean">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64ArrayAttr:$axes
                   )>,
	Results<(outs AnyRankedTensor:$output)> {
  let summary = "Get the mean of the values in given axis";
  let description = [{
    The `migraphx.reduce_mean` op.
  }];
  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands) `->` type(results)";
}

//--------- Execution layer Ops
def MIGraphX_CodeObjOp :
    MIGraphX_Op<"code_object">,
    Arguments<(ins SymbolRefAttr:$kernel,
      I64ArrayAttr:$globalSize,
      I64ArrayAttr:$localSize,
      Variadic<AnyType>:$kernelArgs)>,
	Results<(outs Variadic<AnyType>:$outputs)> {
  let summary = "OP representing a code object";
  let description = [{
    The `migraphx.code_object` op. Holds the compiled kernel binary and arguments.
  }];
  let assemblyFormat = "attr-dict `(`$kernelArgs`)` `:` `(`type($kernelArgs)`)` `->` type($outputs)";
}

#endif // MIGRAPHX_OPS
