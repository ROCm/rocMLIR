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

def MIOpenAsyncLaunchPass : Pass<"miopen-async-launch", "func::FuncOp"> {
  let summary = "convert kernel func call ops to async.launch ops with dependencies";
  let constructor = "mlir::miopen::createMIOpenAsyncLaunchPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect", "async::AsyncDialect"];
}

def MIOpenConvToGemmPass : Pass<"miopen-conv-to-gemm", "::mlir::func::FuncOp"> {
  let summary = "expand convolution into coordinate transformations and gridwise gemm";
  let constructor = "mlir::miopen::createMIOpenConvToGemmPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "memref::MemRefDialect", "arith::ArithmeticDialect",
    "AffineDialect", "scf::SCFDialect"];
}

def MIOpenOpsAffixTuningParametersPass : Pass<"miopen-affix-params", "::mlir::func::FuncOp"> {
  let summary = "populate tuning parameters for miopen.gridwise_gemm ops";
  let constructor = "mlir::miopen::createAffixTuningParametersPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "arith::ArithmeticDialect",
                           "func::FuncDialect"];
}

def MIOpenGridwiseGemmToBlockwisePass : Pass<"miopen-gridwise-gemm-to-blockwise", "::mlir::func::FuncOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and threadwise copy";
  let constructor = "mlir::miopen::createMIOpenGridwiseGemmToBlockwisePass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "vector::VectorDialect", "memref::MemRefDialect", "AffineDialect"];
}

def MIOpenLinalgAlignPass : Pass<"miopen-linalg-align", "::mlir::func::FuncOp"> {
  let summary = "expand linalg ops aligned with threadwise copy";
  let constructor = "mlir::miopen::createMIOpenLinalgAlignPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def MIOpenCopyOptPass : Pass<"miopen-copy-opt", "::mlir::func::FuncOp"> {
  let summary = "Optimize global tensor allocations";
  let constructor = "mlir::miopen::createMIOpenCopyOptPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def MIOpenBlockwiseGemmToThreadwisePass : Pass<"miopen-blockwise-gemm-to-threadwise", "::mlir::func::FuncOp"> {
  let summary = "Expand blockwise gemm into threadwise gemm and clean up fusion-related shorthand";
  let constructor = "mlir::miopen::createMIOpenBlockwiseGemmToThreadwisePass()";
  let dependentDialects = ["gpu::GPUDialect", "miopen::MIOpenDialect", "scf::SCFDialect", "vector::VectorDialect", "AffineDialect", "memref::MemRefDialect"];
}

def MIOpenThreadwiseGemmLoweringPass : Pass<"miopen-threadwise-gemm-lowering", "::mlir::func::FuncOp"> {
  let summary = "expand threadwise gemm to final implementation.";
  let constructor = "mlir::miopen::createMIOpenThreadwiseGemmLoweringPass()";
  let dependentDialects = ["gpu::GPUDialect", "miopen::MIOpenDialect", "amdgpu::AMDGPUDialect", "scf::SCFDialect", "vector::VectorDialect", "AffineDialect", "memref::MemRefDialect", "LLVM::LLVMDialect"];
}

def MIOpenSugarToLoopsPass : Pass<"miopen-sugar-to-loops", "::mlir::func::FuncOp"> {
  let summary = "Expand shorthand, like transforming_for and extract_slice, to other dialects";
  let constructor = "mlir::miopen::createMIOpenSugarToLoopsPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "vector::VectorDialect", "arith::ArithmeticDialect", "memref::MemRefDialect", "AffineDialect", "gpu::GPUDialect", "amdgpu::AMDGPUDialect"];
}

def MIOpenCleanMathPass : Pass<"miopen-clean-math", "::mlir::func::FuncOp"> {
  let summary = "Clean up math using integer range analysis and other MLIR passes";
  let constructor = "mlir::miopen::createMIOpenCleanMathPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "arith::ArithmeticDialect"];
}

def MIOpenLoopsToCfPass : Pass<"miopen-loops-to-cf", "ModuleOp"> {
  let summary = "expand loop / affine dialects to control flow. Notice GPU dialect will explicitly NOT be used in this pass";
  let constructor = "mlir::miopen::createMIOpenLoopsToCfPass()";
  let dependentDialects = ["miopen::MIOpenDialect", "scf::SCFDialect", "AffineDialect", "func::FuncDialect", "memref::MemRefDialect"];
}

#endif // MLIR_DIALECT_MIOPEN_PASSES
