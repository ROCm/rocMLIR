//===- RockWriterOpInterface.td - ops that write into memory---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockWriterOpInterface, which abstracts rock operations
// that write into memory
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_WRITEROP_INTERFACE
#define ROCK_WRITEROP_INTERFACE

include "mlir/IR/OpBase.td"

def RockWriterOpInterface : OpInterface<"RockWriterOpInterface"> {
  let description = [{
    Any rock operation actually writes into memory
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [
    InterfaceMethod<
        /*desc=*/[{
          "Return the destination of the write"
        }],
        /*retType=*/"Value",
        /*methodName=*/"getDest",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    // TODO: more methods here as needed
  ];
}

#endif //ROCK_WRITEROP_INTERFACE
