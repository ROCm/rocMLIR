//===- MIOpenBase.td - MIOpen dialect base support ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for base miopen support.
//
//===----------------------------------------------------------------------===//

#ifndef MIOPEN_BASE
#define MIOPEN_BASE

include "mlir/IR/OpBase.td"

def MIOpen_Dialect : Dialect {
  let name = "miopen";
  let description = [{
    The `miopen` dialect groups together a set of types, operations and
    transformations that are useful to implement a structured abstraction on
    buffers and tensors. These abstractions are useful for transformations and
    can lower to scalar load/store and other operations or to more general
    library calls.

    Additional [MIOpen Dialect
    Documentation](https://mlir.llvm.org/docs/Dialects/MIOpen) and a
    [Rationale
    Document](https://mlir.llvm.org/docs/Rationale/RationaleMIOpenDialect) are
    are also available and should be read first before going in the details of
    the op semantics.
  }];
  let cppNamespace = "::mlir::miopen";
  let dependentDialects = [
    "AffineDialect", "StandardOpsDialect", "tensor::TensorDialect"
  ];
}

// Whether a type is a RangeType.
def MIOpenIsRangeTypePred : CPred<"$_self.isa<RangeType>()">;
def Range : DialectType<MIOpen_Dialect, MIOpenIsRangeTypePred, "range">;

#endif // MIOPEN_BASE
