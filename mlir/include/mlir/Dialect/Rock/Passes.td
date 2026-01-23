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

def RockDetectFlashDecodingPass
    : Pass<"rock-detect-flash-decoding", "::mlir::func::FuncOp"> {
  let summary = "Detect flash decoding (splitKV > 1) from rock.transform "
                "operations and update AttentionOps";
  let description = [{
    This pass analyzes rock.transform operations to detect when flash decoding
    (splitKV > 1) is being used. It identifies broadcast patterns in input
    tensors that indicate the splitKV factor, removes those broadcasts (since
    the attention implementation handles splitting internally), and updates the
    AttentionOp's splitKV attribute accordingly.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockConvToGemmPass : Pass<"rock-conv-to-gemm", "::mlir::func::FuncOp"> {
  let summary = "expand convolution into coordinate transformations and gemm. "
                "Additionally, it also convert rock.conv_elementwise_gemm into "
                "rock.gemm_elementwise_gemm";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect", "arith::ArithDialect",
    "scf::SCFDialect", "gpu::GPUDialect"];
}

def RockAffixTuningParametersPass : Pass<"rock-affix-params", "::mlir::func::FuncOp"> {
  let summary = "populate tuning parameters for gemm and convolution ops";
  let dependentDialects = ["rock::RockDialect",
                           "func::FuncDialect"];
  let options = [
    Option<"fallBackNoConfig", "fallback", "bool", "false",
      "Fall back to a heuristic system when hand-specified tuning parameters are invalid">
  ];
}

def RockGemmToGridwisePass : Pass<"rock-gemm-to-gridwise", "::mlir::func::FuncOp"> {
  let summary = "convert gemm to gridwise gemm, adding padding";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "memref::MemRefDialect", "linalg::LinalgDialect",
                           "arith::ArithDialect"];
}

def RockFoldBroadcastPass: Pass<"rock-fold-broadcast", "::mlir::func::FuncOp"> {
  let summary = "Convert Gemm([g, M, K], [broadcast{1}, K, N]) to Gemm([g*M, K], [K, N])";
  let dependentDialects = ["rock::RockDialect"];
}

def RockGridwiseGemmToBlockwisePass : Pass<"rock-gridwise-gemm-to-blockwise", "::mlir::func::FuncOp"> {
  let summary = "expand gridwise gemm into blockwise copy, blockwise gemm, and threadwise copy";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect", "gpu::GPUDialect", "vector::VectorDialect", "memref::MemRefDialect", "linalg::LinalgDialect", "scf::SCFDialect"];
}

def RockShuffleGemmForReductions : Pass<"rock-shuffle-gemm-for-reductions", "::mlir::func::FuncOp"> {
  let summary = "This pass shuffles parallel gemm dimensions for equally block splits for fused reductions (if any)";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect"];
}

def RockGemmLinalgSplitkNormalizationPass : Pass<"rock-gemm-linalg-splitk-normalization", "::mlir::func::FuncOp"> {
  let summary = "This pass modifies linalg.generic for split-k fusions. It converts any arith.addf/arith.subf gemmOut, other to arith.addf gemmOut, other/splitkFactor";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockTransformsToPtrPass : Pass<"rock-transforms-to-ptr", "::mlir::func::FuncOp"> {
  let summary = "";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockTransformsToPointerArithPass : Pass<"rock-transforms-to-pointer-arith", "::mlir::func::FuncOp"> {
  let summary = "";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockToTTIRPass : Pass<"rock-to-ttir", "::mlir::func::FuncOp"> {
  let summary = "Convert Rock dialect operations to Triton IR";
  let dependentDialects = [
    "rock::RockDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "triton::TritonDialect",
    "bufferization::BufferizationDialect",
  ];
}

def RockMemrefToTensorPass : Pass<"rock-memref-to-tensor", "::mlir::ModuleOp"> {
  let summary = "Convert Memref to Tensor and func.func to tt.func";
  let dependentDialects = [
    "rock::RockDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "triton::TritonDialect",
    "bufferization::BufferizationDialect",
  ];
}

def RockUnbufferizePass : Pass<"rock-unbufferize", "::mlir::triton::FuncOp"> {
  let summary = "Remove bufferization artifacts and convert to pure tensor operations";
  let description = [{
    This pass removes remaining bufferization operations after pointer conversion:
    - Folds to_tensor(to_buffer(x)) -> x
    - Removes dead rock.alloc() and memref.copy operations
  }];
  let dependentDialects = [
    "rock::RockDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "memref::MemRefDialect",
  ];
}

def TritonToHsacoPass : Pass<"triton-to-hsaco", "::mlir::ModuleOp"> {
  let summary = "Convert Triton LLVM dialect IR to HSACO binary";
  let description = [{
    This pass translates Triton LLVM dialect IR to HSACO binary format.
    It wraps the TritonToHsaco translation and implements the functionality
    from Triton's compiler.py:
    - make_llir() lines 358-449: LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    - make_amdgcn() lines 452-473: LLVM -> AMDGCN assembly
    - make_hsaco() lines 476-488: AMDGCN assembly -> HSACO binary

    The resulting HSACO binary is stored as a "triton.hsaco" attribute on the module.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
  let options = [
    Option<"triple", "triple", "std::string", "\"amdgcn-amd-amdhsa\"",
      "Triple">,
    Option<"arch", "arch", "std::string", "\"gfx90a\"",
      "AMDGPU architecture (e.g., gfx90a, gfx942, gfx1100)">,
    Option<"features", "features", "std::string", "\"\"",
      "Arch features">,
    Option<"optLevel", "opt-level", "int", "3",
      "Compiler optimization level">,
    Option<"numWarps", "num-warps", "int", "4",
      "Number of warps per block">,
    Option<"wavesPerEU", "waves-per-eu", "int", "0",
      "Waves per execution unit hint (0 for default)">,
    Option<"enableFpFusion", "enable-fp-fusion", "bool", "true",
      "Enable FP fusion optimizations">,
    Option<"allowFlushDenorm", "allow-flush-denorm", "bool", "false",
      "Allow flushing denormal numbers">,
    Option<"scalarizePackedFops", "scalarize-packed-fops", "bool", "false",
      "Scalarize packed float operations">,
    Option<"scheduleHint", "schedule-hint", "std::string", "\"none\"",
      "Scheduling hint (e.g., 'memory-bound-attention' for iterative-ilp scheduler)">
  ];
}

def RockRestoreHostCodePass : Pass<"rock-restore-host-code", "::mlir::ModuleOp"> {
  let summary = "Restore host functions after Triton kernel compilation";
  let description = [{
    This pass restores host functions (like the GPU wrapper and main) that were
    stored during RockMemrefToTensorPass. It:
    1. Reads the serialized host functions from the module attribute
    2. Parses them back into the module
    3. Updates kernel calls to use the compiled kernel

    This allows the full test harness to work with the Triton-compiled kernel.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"triple", "triple", "std::string", "\"amdgcn-amd-amdhsa\"",
      "Triple">,
    Option<"arch", "arch", "std::string", "\"gfx90a\"",
      "AMDGPU architecture (e.g., gfx90a, gfx942, gfx1100)">,
    Option<"features", "features", "std::string", "\"\"",
      "Arch features">,
    Option<"optLevel", "opt-level", "int", "3",
      "Compiler optimization level">,
  ];
}

#endif // MLIR_DIALECT_ROCK_PASSES
