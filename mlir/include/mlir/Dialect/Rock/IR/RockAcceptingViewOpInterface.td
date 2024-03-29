//===- RockAcceptingViewOpInterface.td - ops that accept rock views---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockAcceptingViewOpInterface, which abstracts rock operations
// that accept a view as operands.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_ACCEPTINGVIEWOP_INTERFACE
#define ROCK_ACCEPTINGVIEWOP_INTERFACE

include "mlir/IR/OpBase.td"

def RockAcceptingViewOpInterface : OpInterface<"RockAcceptingViewOpInterface"> {
  let description = [{
    Any rock operation that accepts a view as input should implement this interface.
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [
    InterfaceMethod<
        /*desc=*/[{
          "Clone the operation with a new set of extra indices"
        }],
        /*retType=*/"Operation *",
        /*methodName=*/"cloneWithExtraIndices",
        /*args=*/(ins "OpBuilder& ":$builder, "OpOperand&":$operand, "Value":$view, "ArrayRef<Value>":$newExtraIndices),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          "Return the extra indices used to index the operand accepting the view"
        }],
        /*retType=*/"std::optional<OperandRange>",
        /*methodName=*/"getExtraIndices",
        /*args=*/(ins "OpOperand&":$operand),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          "Return the list of operands that accept a view"
        }],
        /*retType=*/"llvm::SmallPtrSet<OpOperand*, 2>",
        /*methodName=*/"getAcceptingViewOperands",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >
    // TODO: more methods here as needed
  ];
}

#endif // ROCK_ACCEPTINGVIEWOP_INTERFACE
