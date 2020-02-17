//===- MIOpenOps.td - MIOpen operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIOpen operations.
//
//===----------------------------------------------------------------------===//

#ifndef MIOPEN_OPS
#define MIOPEN_OPS

include "mlir/IR/OpBase.td"
//include "mlir/Transforms/LoopLikeInterface.td"

def MIOpen_Dialect : Dialect {
  let name = "miopen";
  let cppNamespace = "";
}

// Base class for MIOpen dialect ops.
class MIOpen_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MIOpen_Dialect, mnemonic, traits> {
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

def MIOpen_Conv2DOp :
    MIOpen_Op<"conv2d">,
    Arguments<(ins MemRefRankOf<[F32], [4]>:$filter,
                   MemRefRankOf<[F32], [4]>:$input,
                   MemRefRankOf<[F32], [4]>:$output)> {
  let summary = "2D convolution";
  let description = [{
    The `miopen.conv2d` op computes 2D convolution.
  }];
}

def MIOpen_TransformOp :
    MIOpen_Op<"transform">,
    Arguments<(ins AnyMemRef:$input)>,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Tensor transformation";
  let description = [{
    The `miopen.transform` op transforms tensor coordinates.
  }];
}

def MIOpen_GridwiseGemmOp :
    MIOpen_Op<"gridwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>)> {
  let summary = "Gridwise GEMM";
  let description = [{
    The `miopen.gridwise_gemm` op computes gridwise GEMM.
  }];
}

#endif // MIOPEN_OPS
