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

def ConvertMHALToGPUPass : Pass<"convert-mhal-to-gpu", "ModuleOp"> {
  let summary = "Convert the mhal.launch operations to gpu.launch_func";
  let dependentDialects = ["gpu::GPUDialect"];
}

//===----------------------------------------------------------------------===//
// MHALToCPU
//===----------------------------------------------------------------------===//

def ConvertMHALToCPUPass : Pass<"convert-mhal-to-cpu", "ModuleOp"> {
  let summary = "Convert the mhal.launch operations to func.call";
  let dependentDialects = ["func::FuncDialect"];
}

#endif // MHAL_CONVERSION_PASSES
