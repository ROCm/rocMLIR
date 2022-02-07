//===- MIOpenBase.td - MIOpen dialect definition -----------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MIOPEN_BASE
#define MIOPEN_BASE

include "mlir/IR/OpBase.td"

def MIOpen_Dialect : Dialect {
  let name = "miopen";
  let cppNamespace = "::mlir::miopen";

  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Global name for MIOpen Kernel Module
    constexpr const static ::llvm::StringLiteral
        kKernelModuleName = "__miopen";
  }];
}

#endif // MIOPEN_BASE
