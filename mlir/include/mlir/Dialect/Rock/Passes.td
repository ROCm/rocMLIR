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

def RockViewToTransformPass : Pass<"rock-view-to-transform", "::mlir::func::FuncOp"> {
  let summary = "convert tensor view ops to rock.transform";
  let dependentDialects = ["rock::RockDialect", "tensor::TensorDialect",
    "tosa::TosaDialect"];
}

def RockConvToGemmPass : Pass<"rock-conv-to-gemm", "::mlir::func::FuncOp"> {
  let summary = "expand convolution into coordinate transformations and gemm";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect", "arith::ArithDialect",
    "scf::SCFDialect"];
}

def RockAffixTuningParametersPass : Pass<"rock-affix-params", "::mlir::func::FuncOp"> {
  let summary = "populate tuning parameters for gemm and convolution ops";
  let dependentDialects = ["rock::RockDialect",
                           "func::FuncDialect"];
  let options = [
    Option<"blockSizeOverride", "block_size", "uint32_t", "0",
      "Non-zero values fixes a particular block size">,
    Option<"gridSizeOverride", "grid_size", "uint32_t", "0",
      "Non-zero value fixes a particular grid size">,
    Option<"fallBackNoConfig", "fallback", "bool", "false",
      "Fall back to a heuristic system when hand-specified tuning parameters are invalid">
  ];

}

def RockGemmToGridwisePass : Pass<"rock-gemm-to-gridwise", "::mlir::func::FuncOp"> {
  let summary = "convert gemm to gridwise gemm, adding padding";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockRegularizePass : Pass<"rock-regularize", "::mlir::func::FuncOp"> {
  let summary = "convert linalg.generic to regular form to enable fusion with convolution and GEMM";
  let dependentDialects = ["rock::RockDialect", "linalg::LinalgDialect", "memref::MemRefDialect"];
}

def RockGridwiseGemmToBlockwisePass : Pass<"rock-gridwise-gemm-to-blockwise", "::mlir::func::FuncOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and threadwise copy";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect", "gpu::GPUDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def RockLinalgAlignPass : Pass<"rock-linalg-align", "::mlir::func::FuncOp"> {
  let summary = "expand linalg ops aligned with threadwise copy";
  let dependentDialects = ["rock::RockDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def RockBlockwiseGemmToThreadwisePass : Pass<"rock-blockwise-gemm-to-threadwise", "::mlir::func::FuncOp"> {
  let summary = "Expand blockwise gemm into threadwise gemm and clean up fusion-related shorthand";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect", "gpu::GPUDialect", "memref::MemRefDialect", "scf::SCFDialect", "vector::VectorDialect"];
}

def RockThreadwiseGemmLoweringPass : Pass<"rock-threadwise-gemm-lowering", "::mlir::func::FuncOp"> {
  let summary = "expand threadwise gemm to final implementation.";
  let dependentDialects = ["gpu::GPUDialect", "rock::RockDialect", "amdgpu::AMDGPUDialect", "vector::VectorDialect", "affine::AffineDialect", "memref::MemRefDialect"];
}

def RockSugarToLoopsPass : Pass<"rock-sugar-to-loops", "::mlir::func::FuncOp"> {
  let summary = "Expand shorthand, like transforming_for and extract_slice, to other dialects";
  let dependentDialects = ["rock::RockDialect", "vector::VectorDialect", "arith::ArithDialect",
    "gpu::GPUDialect", "memref::MemRefDialect", "affine::AffineDialect", "amdgpu::AMDGPUDialect", "LLVM::LLVMDialect", "scf::SCFDialect"];
}

def RockCleanMathPass : Pass<"rock-clean-math", "::mlir::func::FuncOp"> {
  let summary = "Clean up math using integer range analysis and other MLIR passes";
}

def RockBufferLoadMergePass : Pass<"rock-buffer-load-merge", "::mlir::func::FuncOp"> {
  let summary = "Merge identical memory loads to buffers only read. Assumes noalias.";
  let dependentDialects = ["::mlir::amdgpu::AMDGPUDialect"];
}

def RockTransformToMemrefPass : Pass<"rock-transform-to-memref", "::mlir::func::FuncOp"> {
  let summary = "convert remaining rock.transform ops to memref.expand/collapse_shape";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect", "gpu::GPUDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def RockLoopsToCfPass : Pass<"rock-loops-to-cf", "ModuleOp"> {
  let summary = "expand loop / affine dialects to control flow. Notice GPU dialect will explicitly NOT be used in this pass";
  let dependentDialects = ["scf::SCFDialect","cf::ControlFlowDialect"];
}

def RockLowerReducePass : Pass<"rock-lower-reduce", "::mlir::func::FuncOp"> {
  let summary = "Lower rock.reduce operator";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect", "gpu::GPUDialect"];
}

#endif // MLIR_DIALECT_ROCK_PASSES
