//===-- Passes.td - Rock pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_PASSES
#define MLIR_DIALECT_ROCK_PASSES

include "mlir/Pass/PassBase.td"

def RockCloneKernelsPass : Pass<"rock-clone-kernels", "ModuleOp"> {
  let summary = "clone all kernel funcs into rock module";
  let constructor = "mlir::rock::createRockCloneKernelsPass()";
  let options = [
    ListOption<"chips", "chips", "std::string", "List of chip architectures">
  ];
}

def RockApplyImplPass : Pass<"rock-apply-impl", "ModuleOp"> {
  let summary = "apply target implementation to host kernel funcs";
  let constructor = "mlir::rock::createRockApplyImplPass()";
}

def RockAsyncLaunchPass : Pass<"rock-async-launch", "func::FuncOp"> {
  let summary = "convert kernel func call ops to async.launch ops with dependencies";
  let constructor = "mlir::rock::createRockAsyncLaunchPass()";
  let dependentDialects = ["rock::RockDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect", "async::AsyncDialect"];
}

def RockFoldTransposePass : Pass<"rock-fold-transpose", "::mlir::func::FuncOp"> {
  let summary = "fold transposes to enable fusion with convolution and GEMM";
  let constructor = "mlir::rock::createRockFoldTransposePass()";
  let dependentDialects = ["rock::RockDialect", "linalg::LinalgDialect"];
}

def RockConvToGemmPass : Pass<"rock-conv-to-gemm", "::mlir::func::FuncOp"> {
  let summary = "expand convolution into coordinate transformations and gemm";
  let constructor = "mlir::rock::createRockConvToGemmPass()";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect", "arith::ArithmeticDialect",
    "AffineDialect", "scf::SCFDialect"];
}

def RockOpsAffixTuningParametersPass : Pass<"rock-affix-params", "::mlir::func::FuncOp"> {
  let summary = "populate tuning parameters for gemm and convolution ops";
  let constructor = "mlir::rock::createAffixTuningParametersPass()";
  let dependentDialects = ["rock::RockDialect", "arith::ArithmeticDialect",
                           "func::FuncDialect"];
}

def RockGemmToGridwisePass : Pass<"rock-gemm-to-gridwise", "::mlir::func::FuncOp"> {
  let summary = "convert gemm to gridwise gemm, adding padding";
  let constructor = "mlir::rock::createRockGemmToGridwisePass()";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockGridwiseGemmToBlockwisePass : Pass<"rock-gridwise-gemm-to-blockwise", "::mlir::func::FuncOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and threadwise copy";
  let constructor = "mlir::rock::createRockGridwiseGemmToBlockwisePass()";
  let dependentDialects = ["rock::RockDialect", "scf::SCFDialect", "vector::VectorDialect", "memref::MemRefDialect", "AffineDialect"];
}

def RockLinalgAlignPass : Pass<"rock-linalg-align", "::mlir::func::FuncOp"> {
  let summary = "expand linalg ops aligned with threadwise copy";
  let constructor = "mlir::rock::createRockLinalgAlignPass()";
  let dependentDialects = ["rock::RockDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def RockCopyOptPass : Pass<"rock-copy-opt", "::mlir::func::FuncOp"> {
  let summary = "Optimize global tensor allocations";
  let constructor = "mlir::rock::createRockCopyOptPass()";
  let dependentDialects = ["rock::RockDialect", "scf::SCFDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def RockBlockwiseGemmToThreadwisePass : Pass<"rock-blockwise-gemm-to-threadwise", "::mlir::func::FuncOp"> {
  let summary = "Expand blockwise gemm into threadwise gemm and clean up fusion-related shorthand";
  let constructor = "mlir::rock::createRockBlockwiseGemmToThreadwisePass()";
  let dependentDialects = ["gpu::GPUDialect", "rock::RockDialect", "scf::SCFDialect", "vector::VectorDialect", "AffineDialect", "memref::MemRefDialect"];
}

def RockThreadwiseGemmLoweringPass : Pass<"rock-threadwise-gemm-lowering", "::mlir::func::FuncOp"> {
  let summary = "expand threadwise gemm to final implementation.";
  let constructor = "mlir::rock::createRockThreadwiseGemmLoweringPass()";
  let dependentDialects = ["gpu::GPUDialect", "rock::RockDialect", "amdgpu::AMDGPUDialect", "scf::SCFDialect", "vector::VectorDialect", "AffineDialect", "memref::MemRefDialect", "LLVM::LLVMDialect"];
}

def RockSugarToLoopsPass : Pass<"rock-sugar-to-loops", "::mlir::func::FuncOp"> {
  let summary = "Expand shorthand, like transforming_for and extract_slice, to other dialects";
  let constructor = "mlir::rock::createRockSugarToLoopsPass()";
  let dependentDialects = ["rock::RockDialect", "vector::VectorDialect", "arith::ArithmeticDialect", "memref::MemRefDialect", "AffineDialect", "gpu::GPUDialect", "amdgpu::AMDGPUDialect"];
}

def RockCleanMathPass : Pass<"rock-clean-math", "::mlir::func::FuncOp"> {
  let summary = "Clean up math using integer range analysis and other MLIR passes";
  let constructor = "mlir::rock::createRockCleanMathPass()";
  let dependentDialects = ["rock::RockDialect", "arith::ArithmeticDialect"];
}

def RockLoopsToCfPass : Pass<"rock-loops-to-cf", "ModuleOp"> {
  let summary = "expand loop / affine dialects to control flow. Notice GPU dialect will explicitly NOT be used in this pass";
  let constructor = "mlir::rock::createRockLoopsToCfPass()";
  let dependentDialects = ["rock::RockDialect", "scf::SCFDialect", "AffineDialect", "func::FuncDialect", "memref::MemRefDialect"];
}

#endif // MLIR_DIALECT_ROCK_PASSES
