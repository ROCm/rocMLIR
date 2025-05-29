//===- RockTuningParamAttrInterface.td - tuning params for the rock ops ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockAccelAttentionTuningParamAttrInterface, which abstracts
// provide a common interface to handle tuning params for attention accelerators
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_ACCEL_ATTENTION_TUNING_PARAM_INTERFACE
#define ROCK_ACCEL_ATTENTION_TUNING_PARAM_INTERFACE

include "mlir/Dialect/Rock/IR/RockBase.td"
include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def RockAccelAttentionTuningParamAttrInterface
    : AttrInterface<"RockAccelAttentionTuningParamAttrInterface",
                    [RockTuningParamAttrInterface]> {
  let description = [{
    This interface is providing generic abstractions to handle tuning params for accelerators
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [
    InterfaceMethod<
        /*desc=*/[{
          Given that the GEMM input blocks layout is `[kOuter, dPerBlock, kPack]` (where `d` can be `m` or `n`)
          this method returns `kOuter`.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getKpackPerBlock",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          M per block of the first GEMM.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getMPerBlockG0",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          M per block of the second GEMM.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getMPerBlockG1",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >

    // TODO: more methods here as needed
  ];
}

#endif // ROCK_ACCEL_ATTENTION_TUNING_PARAM_INTERFACE
