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

def MIGraphX_Conv2DOp :
    MIGraphX_Op<"conv2d">,
    Arguments<(ins AnyRankedTensor:$filter,
                   AnyRankedTensor:$input,

                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$pading,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$stride,
                   Confined<I32ArrayAttr, [ArrayCount<2>]>:$dilation,
                   I32ArrayAttr:$group,
                   I32ArrayAttr:$padding_mode
                   )>,
	Results<(outs AnyRankedTensor:$output)> {	  
  let summary = "2D convolution forward";
  let description = [{
    The `migraphx.conv2d` op computes 2D convolution forward.
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
