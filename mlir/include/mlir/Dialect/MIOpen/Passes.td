//===-- Passes.td - MIOpen pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_PASSES
#define MLIR_DIALECT_MIOPEN_PASSES

include "mlir/Pass/PassBase.td"

def MIOpenOpsStep1Pass : Pass<"miopen-lowering", "ModuleOp"> {
  let summary = "expand convolution into coordinate transformations and gridwise gemm";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep1Pass()";
}

def MIOpenOpsAffineTransformPass : FunctionPass<"miopen-affine-transform"> {
  let summary = "build affine maps for miopen.transform ops";
  let constructor = "mlir::miopen::createAffineTransformPass()";
}

def MIOpenOpsAffixTuningParametersPass : FunctionPass<"miopen-affix-params"> {
  let summary = "populate tuning parameters for miopen.gridwise_gemm ops";
  let constructor = "mlir::miopen::createAffixTuningParametersPass()";
}

def MIOpenOpsStep2Pass : Pass<"miopen-lowering-step2", "ModuleOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and threadwise copy";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep2Pass()";
}

def MIOpenOpsStep3Pass : Pass<"miopen-lowering-step3", "ModuleOp"> {
  let summary = "expand blockwise copy into threadwise copy, blockwise gemm into threadwise gemm";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep3Pass()";
}

#endif // MLIR_DIALECT_MIOPEN_PASSES
