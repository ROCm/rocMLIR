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
}

def MIOpen_Conv2DOp : MIOpen_Op<"conv2d">;

def MIOpen_TransformOp : MIOpen_Op<"transform">;

def MIOpen_GridwiseGemmOp : MIOpen_Op<"gridwise_gemm">;

#endif // MIOPEN_OPS
