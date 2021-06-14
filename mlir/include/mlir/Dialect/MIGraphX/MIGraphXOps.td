//===- MIGraphXOps.td - MIGraphX operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIGraphX operations.
//
//===----------------------------------------------------------------------===//

#ifndef MIGRAPHX_OPS
#define MIGRAPHX_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MIGraphX_Dialect : Dialect {
  let name = "migraphx";
  let cppNamespace = "migraphx";
}

// Base class for MIGraphX dialect ops.
class MIGraphX_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MIGraphX_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
  }];
}
/*
//@0 = @literal{ ... } -> float_type, {64, 3, 7, 7}, {147, 49, 7, 1}
// literal data_type, dimension, strides
def MIGraphX_LiteralOp :
    MIGraphX_Op<"literal">,
	Arguments<(ins
	  StrAttr:$values,
	  Confined<I64ArrayAttr, [ArrayMinCount<1>]>:$dimension,
	  Confined<I64ArrayAttr, [ArrayMinCount<1>]>:$Stride)>,
    Results<(outs AnyTensor:$output)> {
  let summary = "Literals";
  let description = [{
    Constant data given as literal.
  }];
  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("values"); }
}
*/
def MIGraphX_AddOp :
    MIGraphX_Op<"add">,
    Arguments<(ins AnyRankedTensor:$inA,
                   AnyRankedTensor:$inB)>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Elementwise binary add";
  let description = [{
    Add two tensors elementwise
  }];
}

def MIGraphX_ConvolutionOp :
    MIGraphX_Op<"convolution">,
    Arguments<(ins AnyRankedTensor:$filter,
                   AnyRankedTensor:$input,

                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$pading,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$stride,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$dilation,
                   I32Attr:$group,
                   I32Attr:$padding_mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "2D convolution forward";
  let description = [{
    The `migraphx.conv2d` op computes 2D convolution forward.
  }];
}

//@268 = batch_norm_inference[epsilon=1e-05,momentum=0.9,bn_mode=1](@267,@1,@2,@3,@4) -> float_type, {1, 64, 112, 112}, {802816, 12544, 112, 1}

def MIGraphX_BatchNormOp :
    MIGraphX_Op<"batch_norm_inference">,
    Arguments<(ins AnyRankedTensor:$input,
                   AnyRankedTensor:$a,
                   AnyRankedTensor:$b,
                   AnyRankedTensor:$c,
                   AnyRankedTensor:$d,

                   F32Attr:$epsilon,
                   F32Attr:$momentum,
                   I32Attr:$bn_mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "batch_norm_inference";
  let description = [{
    The `migraphx.batch_norm_inference` op computes batch_norm.
  }];
}

//@269 = relu(@268) -> float_type, {1, 64, 112, 112}, {802816, 12544, 112, 1}
def MIGraphX_ReluOp :
    MIGraphX_Op<"relu">,
    Arguments<(ins AnyRankedTensor:$input
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "RELU activation";
  let description = [{
    The `migraphx.relu` op computes relu op.
  }];
}

//@270 = pooling[mode=max,padding={1, 1},stride={2, 2},lengths={3, 3},ceil_mode=0](@269) -> float_type, {1, 64, 56, 56}, {200704, 3136, 56, 1}
def MIGraphX_PoolingOp :
    MIGraphX_Op<"pooling">,
    Arguments<(ins AnyRankedTensor:$input,

                   StrAttr:$mode,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$pading,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$stride,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$length,
                   I32Attr:$ceil_mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Pooling operation";
  let description = [{
    The `migraphx.pooling` op computes average/max pooling op.
  }];
}

//@440 = flatten[axis=1](@439) -> float_type, {1, 2048}, {2048, 1}
def MIGraphX_FlattenOp :
    MIGraphX_Op<"flatten">,
    Arguments<(ins AnyRankedTensor:$input,
                   I32Attr:$axis
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Flatten tensor";
  let description = [{
    The `migraphx.flatten` op.
  }];
}

//@441 = transpose[dims={1, 0}](@265) -> float_type, {2048, 1000}, {1, 2048}
def MIGraphX_TransposeOp :
    MIGraphX_Op<"transpose">,
    Arguments<(ins AnyRankedTensor:$input,
                   I32ArrayAttr:$dims
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "transpose dimensions";
  let description = [{
    The `migraphx.transpose` op.
  }];
}

//@442 = multibroadcast[output_lens={1, 1000}](@266) -> float_type, {1, 1000}, {0, 1}
def MIGraphX_MultiBroadcastOp :
    MIGraphX_Op<"multibroadcast">,
    Arguments<(ins AnyRankedTensor:$input,
                   I32ArrayAttr:$ouput_lens
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Broadcast tensor in multiple dimensions";
  let description = [{
    The `migraphx.multibroadcast` op.
  }];
}

//@443 = dot[alpha=1,beta=1](@440,@441,@442) -> float_type, {1, 1000}, {1000, 1}
def MIGraphX_DotOp :
    MIGraphX_Op<"dot">,
    Arguments<(ins AnyRankedTensor:$input,
                   AnyRankedTensor:$a,
                   AnyRankedTensor:$b,
                   I32Attr:$alpha,
                   I32Attr:$beta
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Broadcast tensor in multiple dimensions";
  let description = [{
    The `migraphx.multibroadcast` op.
  }];
}

def MIGraphX_ConstantOp :
    MIGraphX_Op<"constant">,
    Arguments<(ins ElementsAttr:$value
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "constant tensor operation";
  let description = [{
    The `migraphx.constant` op. Should be able to represent everything from literals.
  }];
}
/*
def MIGraphX_Conv2DOp :
    MIGraphX_Op<"conv2d">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [5]>:$filter,
                   MemRefRankOf<[F32, F16, I16], [5]>:$input,
                   MemRefRankOf<[F32, F16, I16], [5]>:$output)> {
  let summary = "2D convolution forward";
  let description = [{
    The `migraphx.conv2d` op computes 2D convolution forward.
  }];
}

*/

#endif // MIGRAPHX_OPS
