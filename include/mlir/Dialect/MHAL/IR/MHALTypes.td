//===- MHALTypes.td - MHAL dialect types -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MHAL dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MHAL_IR_MHALTYPES
#define MLIR_DIALECT_MHAL_IR_MHALTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/MHAL/IR/MHALDialect.td"

//===----------------------------------------------------------------------===//
// MHAL Types
//===----------------------------------------------------------------------===//

class MHAL_Type<string name, string typeMnemonic> : TypeDef<MHALDialect,
                                                             name> {
  let mnemonic = typeMnemonic;
}

def MHAL_TokenType : MHAL_Type<"Token", "token"> {
  let summary = "mhal token type";
  let description = [{
    `mhal.token` is a type returned by mhalhronous operations, and it becomes
    `available` when the mhalhronous operations that created it is completed.
  }];
}

#endif // MLIR_DIALECT_MHAL_IR_MHALTYPES
