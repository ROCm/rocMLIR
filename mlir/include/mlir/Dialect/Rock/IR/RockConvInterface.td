//===- RockConvInterface.td - common ConvOps access methods ---------=== //
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockConvInterface, which groups common access methods
// conv-like operations
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_CONV_INTERFACE
#define ROCK_CONV_INTERFACE

include "mlir/IR/OpBase.td"

def RockConvInterface : OpInterface<"RockConvInterface"> {
  let description = [{}];
  let cppNamespace = "::mlir::rock";

  let methods = [
    InterfaceMethod<
        /*desc=*/[{Return convolution padding values.}],
        /*retType=*/"ArrayRef<int64_t>",
        /*methodName=*/"getPadding",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.getConvParams().getPadding();
        }]>,
    InterfaceMethod<
        /*desc=*/[{Return convolution stride values.}],
        /*retType=*/"ArrayRef<int64_t>",
        /*methodName=*/"getStride",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.getConvParams().getStride();
        }]>,
    InterfaceMethod<
        /*desc=*/[{Return convolution dilation values.}],
        /*retType=*/"ArrayRef<int64_t>",
        /*methodName=*/"getDilation",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.getConvParams().getDilation();
        }]>,

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

#endif // ROCK_CONV_INTERFACE