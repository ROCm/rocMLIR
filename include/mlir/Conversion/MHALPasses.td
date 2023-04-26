//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MHAL_CONVERSION_PASSES
#define MHAL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// MHALToGPU
//===----------------------------------------------------------------------===//

def ConvertMHALToGPU : Pass<"convert-mhal-to-gpu", "ModuleOp"> {
  let summary = "Convert the async.launch operations to the GPU dialect";
  let constructor = "mlir::createConvertMHALToGPUPass()";
  let dependentDialects = ["gpu::GPUDialect"];
}

#endif // MHAL_CONVERSION_PASSES
