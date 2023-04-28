//===- RockTuningParamAttrInterface.td - tuning params for the rock ops ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockAccelTuningParamAttrInterface, which abstracts provide
// a common interface to handle tuning params for accelerators
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_ACCEL_TUNING_PARAM_INTERFACE
#define ROCK_ACCEL_TUNING_PARAM_INTERFACE

include "mlir/Dialect/Rock/IR/RockBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"


def RockAccelTuningParamAttrInterface : AttrInterface<"RockAccelTuningParamAttrInterface"> {
  let description = [{
    This interface is providing generic abstractions to handle tuning params for accelerators
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [
    InterfaceMethod<
        /*desc=*/[{
          Return how many output rows each block computes. In other words, given that the input block
          A layout is `[kOuter, mPerBlock, kpack]`, this method returns `mPerBlock`.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getMPerBlock",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Return how many output columns each block computes. In other words, given that the input block
          A layout is `[kOuter, nPerBlock, kpack]`, this method returns `nPerBlock`.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getNPerBlock",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Given that the GEMM input blocks layout is `[kOuter, dPerBlock, kPack]` (where `d` can be `m` or `n`)
          this method returns `kOuter`.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getKPerBlock",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Return how many output rows each wave computes.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getMPerWave",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/""
      >,
    InterfaceMethod<
        /*desc=*/[{
          Return how many output columns each wave computes.
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getNPerWave",
        /*args=*/(ins),
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

#endif // ROCK_ACCEL_TUNING_PARAM_INTERFACE
