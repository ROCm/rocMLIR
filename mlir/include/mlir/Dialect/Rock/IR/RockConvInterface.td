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
        /*desc=*/[{Return filter tensor.}],
        /*retType=*/"::mlir::TypedValue<::mlir::ShapedType>",
        /*methodName=*/"getFilter",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{ return $_op.getFilter(); }]>,
    InterfaceMethod<
        /*desc=*/[{Return input tensor.}],
        /*retType=*/"::mlir::TypedValue<::mlir::ShapedType>",
        /*methodName=*/"getInput",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{ return $_op.getInput(); }]>,
    InterfaceMethod<
        /*desc=*/[{Return output tensor.}],
        /*retType=*/"::mlir::TypedValue<::mlir::ShapedType>",
        /*methodName=*/"getOutput",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{ return $_op.getOutput(); }]>,
    InterfaceMethod<
        /*desc=*/[{Return convolution padding values.}],
        /*retType=*/"ArrayAttr",
        /*methodName=*/"getPadding",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{ return $_op.getPadding(); }]>,
    InterfaceMethod<
        /*desc=*/[{Return convolution stride values.}],
        /*retType=*/"ArrayAttr",
        /*methodName=*/"getStrides",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{ return $_op.getStrides(); }]>,
    InterfaceMethod<
        /*desc=*/[{Return convolution dilation values.}],
        /*retType=*/"ArrayAttr",
        /*methodName=*/"getDilations",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{ return $_op.getDilations(); }]>,
    InterfaceMethod<
        /*desc=*/[{Return features attr values.}],
        /*retType=*/"GemmFeatures",
        /*methodName=*/"getFeatures",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"">,

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
