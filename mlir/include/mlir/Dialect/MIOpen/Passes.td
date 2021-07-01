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
  let dependentDialects = ["miopen::MIOpenDialect"];
}

def MIOpenOpsAffineTransformPass : FunctionPass<"miopen-affine-transform"> {
  let summary = "build affine maps for miopen.transform ops";
  let constructor = "mlir::miopen::createAffineTransformPass()";
  let dependentDialects = ["miopen::MIOpenDialect"];
}

def MIOpenOpsTestAffineTransformPass : FunctionPass<"miopen-test-affine-transform"> {
  let summary = "test affine maps for miopen.transform ops";
  let constructor = "mlir::miopen::createTestAffineTransformPass()";
  let options = [
    Option<"testMethod", "test-method", "std::string", "\"hasPadding\"",
           "method to be tested">
  ];
}

def MIOpenOpsAffixTuningParametersPass : FunctionPass<"miopen-affix-params"> {
  let summary = "populate tuning parameters for miopen.gridwise_gemm ops";
  let constructor = "mlir::miopen::createAffixTuningParametersPass()";
  let dependentDialects = ["miopen::MIOpenDialect"];
}

def MIOpenOpsStep2Pass : Pass<"miopen-lowering-step2", "ModuleOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and threadwise copy";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep2Pass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "vector::VectorDialect"];
}

def MIOpenOpsStep3Pass : Pass<"miopen-lowering-step3", "ModuleOp"> {
  let summary = "expand blockwise copy into threadwise copy, blockwise gemm into threadwise gemm";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep3Pass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "vector::VectorDialect"];
}

def MIOpenOpsStep4Pass : Pass<"miopen-lowering-step4", "ModuleOp"> {
  let summary = "expand threadwise copy and threadwise gemm to all other dialects.";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep4Pass()";
  let dependentDialects = ["gpu::GPUDialect", "miopen::MIOpenDialect", "scf::SCFDialect", "vector::VectorDialect", "AffineDialect"];
}

def MIOpenOpsStep5Pass : Pass<"miopen-lowering-step5", "ModuleOp"> {
  let summary = "expand loop / affine dialects to std. Notice GPU dialect will explicitly NOT be used in this pass";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep5Pass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "AffineDialect", "StandardOpsDialect"];
}

#endif // MLIR_DIALECT_MIOPEN_PASSES
