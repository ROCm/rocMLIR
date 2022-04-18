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

def MIOpenCloneKernelsPass : Pass<"miopen-clone-kernels", "ModuleOp"> {
  let summary = "clone all kernel funcs into miopen module";
  let constructor = "mlir::miopen::createMIOpenCloneKernelsPass()";
}

def MIOpenApplyImplPass : Pass<"miopen-apply-impl", "ModuleOp"> {
  let summary = "apply target implementation to host kernel funcs";
  let constructor = "mlir::miopen::createMIOpenApplyImplPass()";
}

def MIOpenAsyncLaunchPass : Pass<"miopen-async-launch", "FuncOp"> {
  let summary = "convert kernel func call ops to async.launch ops with dependencies";
  let constructor = "mlir::miopen::createMIOpenAsyncLaunchPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect", "async::AsyncDialect"];
}

def MIOpenOpsStep1Pass : Pass<"miopen-lowering", "ModuleOp"> {
  let summary = "expand convolution into coordinate transformations and gridwise gemm";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep1Pass()";
  let dependentDialects = ["miopen::MIOpenDialect", "memref::MemRefDialect", "AffineDialect", "scf::SCFDialect"];
}

def MIOpenOpsAffixTuningParametersPass : Pass<"miopen-affix-params", "::mlir::FuncOp"> {
  let summary = "populate tuning parameters for miopen.gridwise_gemm ops";
  let constructor = "mlir::miopen::createAffixTuningParametersPass()";
  let dependentDialects = ["miopen::MIOpenDialect"];
}

def MIOpenOpsStep2Pass : Pass<"miopen-lowering-step2", "ModuleOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and threadwise copy";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep2Pass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "vector::VectorDialect", "memref::MemRefDialect", "AffineDialect"];
}

def MIOpenLinalgAlignPass : Pass<"miopen-linalg-align", "FuncOp"> {
  let summary = "expand linalg ops aligned with threadwise copy";
  let constructor = "mlir::miopen::createMIOpenLinalgAlignPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def MIOpenCopyOptPass : Pass<"miopen-copy-opt", "FuncOp"> {
  let summary = "Optimize global tensor allocations";
  let constructor = "mlir::miopen::createMIOpenCopyOptPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def MIOpenOpsStep3Pass : Pass<"miopen-lowering-step3", "ModuleOp"> {
  let summary = "expand blockwise copy into threadwise copy, blockwise gemm into threadwise gemm";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep3Pass()";
  let dependentDialects = ["gpu::GPUDialect", "miopen::MIOpenDialect", "scf::SCFDialect", "vector::VectorDialect", "AffineDialect", "memref::MemRefDialect"];
}

def MIOpenOpsStep4Pass : Pass<"miopen-lowering-step4", "ModuleOp"> {
  let summary = "expand threadwise copy and threadwise gemm to all other dialects.";
  let constructor = "mlir::miopen::createLowerMIOpenOpsStep4Pass()";
  let dependentDialects = ["gpu::GPUDialect", "miopen::MIOpenDialect", "scf::SCFDialect", "vector::VectorDialect", "AffineDialect", "memref::MemRefDialect", "LLVM::LLVMDialect"];
}

def MIOpenExpandShorthandPass : Pass<"miopen-expand-shorthand", "ModuleOp"> {
  let summary = "Expand shorthand, like index_diff_update and extract_slice, to other dialects";
  let constructor = "mlir::miopen::createMIOpenExpandShorthandPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "vector::VectorDialect", "arith::ArithmeticDialect", "memref::MemRefDialect", "gpu::GPUDialect"];
}

def MIOpenLoopsToCfPass : Pass<"miopen-loops-to-cf", "ModuleOp"> {
  let summary = "expand loop / affine dialects to control flow. Notice GPU dialect will explicitly NOT be used in this pass";
  let constructor = "mlir::miopen::createMIOpenLoopsToCfPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "AffineDialect", "func::FuncDialect", "memref::MemRefDialect"];
}

#endif // MLIR_DIALECT_MIOPEN_PASSES
