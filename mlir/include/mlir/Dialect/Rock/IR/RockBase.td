//===- RockBase.td - Rock dialect definition -----------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_BASE
#define ROCK_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Rock_Dialect : Dialect {
  let name = "rock";
  let cppNamespace = "::mlir::rock";

  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Global name for Rock Kernel Module
    constexpr const static ::llvm::StringLiteral
        kKernelModuleName = "__rock";
  }];
}

#endif // ROCK_BASE
