//===- LetaoOps.td - Letao operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR Letao operations.
//
//===----------------------------------------------------------------------===//

#ifndef LETAO_OPS
#define LETAO_OPS

include "mlir/IR/OpBase.td"

def Letao_Dialect : Dialect {
  let name = "letao";
  let cppNamespace = "letao";
}

// Base class for Letao dialect ops.
class Letao_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Letao_Dialect, mnemonic, traits> {

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let verifier = [{ return ::verify(*this); }];
}
/*
def DotOp: Letao_Op<"dot">{
  let summary = "element-wise multiplication and sum operation";
  let description = [{
    The "mul" operation performs element-wise multiplication and sum between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$first, F64Tensor:$second);
  let results = (outs F64Tensor);
}
*/
def Letao_Conv2DOp :
    Letao_Op<"conv2d">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [4]>:$filter,
                   MemRefRankOf<[F32, F16, BF16], [4]>:$input,
                   MemRefRankOf<[F32, F16, BF16], [4]>:$output)> {
  let summary = "2D convolution forward";
  let description = [{
    The `miopen.conv2d` op computes 2D convolution forward.
  }];
}

def letao_MovePosOp:
    Letao_Op<"movepos">,
    Arguments<(ins AnyMemRef:$memref,
                   Variadic<AnyTypeOf<[AnyInteger, AnyFloat]>>:$values)> {
    let summary = "Add values to indices within the memref.";
    let description = [{
      The `miopen.fill` op adds values to indices within the memref.
    }];
}
#endif

