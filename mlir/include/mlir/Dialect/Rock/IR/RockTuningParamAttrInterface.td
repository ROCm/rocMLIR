//===- RockTuningParamAttrInterface.td - tuning params for the rock ops ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockTuningParamAttrInterface, which abstracts definitions
// and methods of different tuning parameters for rock dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_TUNING_PARAM_INTERFACE
#define ROCK_TUNING_PARAM_INTERFACE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def RockTuningParamAttrInterface : AttrInterface<"RockTuningParamAttrInterface"> {
  let description = [{
    This interface is providing generic methods to handle tuning parameters
    of different rock ops
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [
    InterfaceMethod<
        /*desc=*/[{
          Return serialize perfconfig (comma separated) string
        }],
        /*retType=*/"void",
        /*methodName=*/"getPerfConfigStr",
        /*args=*/(ins "std::string &":$perfStr),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >

    // TODO: more methods here as needed
  ];

}

#endif // ROCK_TUNING_PARAM_INTERFACE
