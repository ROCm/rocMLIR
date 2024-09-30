//===- MHALDialect.td - MHAL dialect definition -----------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MHAL_DIALECT
#define MHAL_DIALECT

include "mlir/IR/OpBase.td"

def MHALDialect : Dialect {
  let name = "mhal";
  let cppNamespace = "::mlir::mhal";

  let summary = "Types and operations for mhal dialect";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Arg attributes for access modes
    /// These unit attributes may be added to function args to denote how a
    /// memref pointer is accessed by the function.
    static StringRef getReadAccessAttrName() { return "mhal.read_access"; }
    static StringRef getWriteAccessAttrName() { return "mhal.write_access"; }
  }];
}

#endif // MHAL_DIALECT
