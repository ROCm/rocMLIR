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
  let cppNamespace = "mlir::migraphx";
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

// Elementwise binary operations

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

def MIGraphX_SubOp :
    MIGraphX_Op<"sub">,
    Arguments<(ins AnyRankedTensor:$inA,
                   AnyRankedTensor:$inB)>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Elementwise binary sub";
  let description = [{
    Subtract two tensors elementwise
  }];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  //let assemblyFormat = "attr-dict `:` type($output)";
}
/*
def MIGraphX_MulOp :
    MIGraphX_Op<"mul">,
    Arguments<(ins AnyRankedTensor:$inA,
                   AnyRankedTensor:$inB)>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Elementwise binary mul";
  let description = [{
    Multiply two tensors elementwise
  }];
}

// Elementwise unary operations

def MIGraphX_RsqrtOp :
    MIGraphX_Op<"rsqrt">,
    Arguments<(ins AnyRankedTensor:$inA>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Elementwise rsqrt";
  let description = [{
    Reciprocal square root elementwise
  }];
}

def MIGraphX_CeilOp :
    MIGraphX_Op<"ceil">,
    Arguments<(ins AnyRankedTensor:$inA>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Elementwise ceil";
  let description = [{
    Ceil tensor elementwise
  }];
}

def MIGraphX_FloorOp :
    MIGraphX_Op<"floor">,
    Arguments<(ins AnyRankedTensor:$inA>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Elementwise floor";
  let description = [{
    Floor tensor elementwise
  }];
}
*/

// Convolution operations

def MIGraphX_ConvolutionOp :
    MIGraphX_Op<"convolution">,
    Arguments<(ins AnyRankedTensor:$input,
                   AnyRankedTensor:$filter,

                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$padding,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$stride,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$dilation,
                   I32Attr:$group,
                   I32Attr:$padding_mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "2D convolution forward";
  let description = [{
    The `migraphx.convolution` op computes 2D convolution forward.
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
                   I32Attr:$bn_mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "batch_norm_inference";
  let description = [{
    The `migraphx.batch_norm_inference` op computes batch_norm.
  }];
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
}
/*
def MIGraphX_PadOp :
    MIGraphX_Op<"pad">,
    Arguments<(ins AnyRankedTensor:$input,

                   I64ArrayAttr:$pads,
                   F32Attr:$value,
                   I32Attr:$mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "Pad operation";
  let description = [{
    The `migraphx.pad` op adds padding to the tensor.
  }];
}
*/
def MIGraphX_PoolingOp :
    MIGraphX_Op<"pooling">,
    Arguments<(ins AnyRankedTensor:$input,

                   StrAttr:$mode,
                   //PoolingEnum:$mode,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$padding,
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

def MIGraphX_TransposeOp :
    MIGraphX_Op<"transpose">,
    Arguments<(ins AnyRankedTensor:$input,
                   I64ArrayAttr:$dims
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "transpose dimensions";
  let description = [{
    The `migraphx.transpose` op.
  }];
}
/*
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
}
*/
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
    Arguments<(ins OptionalAttr<ElementsAttr>:$value,
                   OptionalAttr<I32ArrayAttr>:$shape,
                   OptionalAttr<TypeAttr>:$type
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "constant tensor operation";
  let description = [{
    The `migraphx.constant` op. Should be able to represent everything from literals.
  }];
}


#endif // MIGRAPHX_OPS
