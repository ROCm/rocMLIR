//===- RockGemmWrapperInterface.td - ops that wrap rock.gemm ---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockGemmWrapperInterface, which abstracts convolutions and
// matrix multiplies to allow code to operate on them generically.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_GEMM_WRAPPER_INTERFACE
#define ROCK_GEMM_WRAPPER_INTERFACE

include "mlir/IR/OpBase.td"

def RockGemmWrapperInterface : OpInterface<"RockGemmWrapperInterface"> {
  let description = [{
    Interface to abstract away gemm-wrapping operators in the rock dialect,
    which mainly include convolutions that can be transformed to matrix
    multiplications.

    This should include functions to get common attributes.
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [
    InterfaceMethod<
        /*desc=*/[{
          Return the OpOperand that corresponds to the operand argument
          that corresponds to the output result of the operation.
        }],
        /*retType=*/"OpOperand *",
        /*methodName=*/"getOutArgument",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Return the size of the matrix multiplication that this op will eventually
          perform.
        }],
        /*retType=*/"::mlir::rock::GemmContext",
        /*methodName=*/"getGemmSize",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >
    // TODO: more methods here as needed
  ];

  let verify = [{
    auto concreteOp = ::mlir::cast<ConcreteOp>($_op);
    if ($_op->getNumResults() == 1) {
      if ($_op->getResult(0).getType() !=
          concreteOp.getOutArgument()->get().getType()) {
        return $_op->emitOpError("result type must match output argument type");
      }
    }
    return ::mlir::success();
  }];
}

#endif // ROCK_GEM_WRAPPER_INTERFACE
