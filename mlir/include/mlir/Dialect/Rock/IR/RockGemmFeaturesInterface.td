//===------------------ RockGemmFeaturesInterface.td ----------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This file defines RockGemmFeaturesInterface, which abstracts Rock ops for
// which we would expect to extract GemmFeatures for.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_GEMM_FEATURES_INTERFACE
#define ROCK_GEMM_FEATURES_INTERFACE

include "mlir/IR/OpBase.td"

def RockGemmFeaturesInterface : OpInterface<"RockGemmFeaturesInterface"> {
  let description = [{
    Interface to abstract away operations for which we need to extract
    information relating to GemmFeatures from.
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [InterfaceMethod<
      /*desc=*/[{
          Return the type from this op that is needed to calculate GemmFeatures
        }],
      /*retType=*/"SmallVector<::mlir::Type>",
      /*methodName=*/"getTypesForFeature",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"">];
}

#endif // ROCK_GEMM_FEATURES_INTERFACE
