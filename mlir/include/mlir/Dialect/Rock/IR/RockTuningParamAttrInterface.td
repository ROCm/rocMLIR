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

include "mlir/Dialect/Rock/IR/RockBase.td"
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
      >,
    InterfaceMethod<
        /*desc=*/[{
          Given that the GEMM input blocks layout is `[kOuter, dPerBlock, kPack]` (where `d` can be `m` or `n`)
          this method returns `kPack`.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getKpack",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Flag used to determine if to unroll the loops in MLIR (return `true`) or leave the decision to the back-end (return `false`)
        }],
        /*retType=*/"bool",
        /*methodName=*/"getForceUnroll",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,

    // TODO: more methods here as needed
  ];

}

#endif // ROCK_TUNING_PARAM_INTERFACE
