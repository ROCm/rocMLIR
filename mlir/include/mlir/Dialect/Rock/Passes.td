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

def RockRegularizePass : Pass<"rock-regularize", "::mlir::func::FuncOp"> {
  let summary = "convert linalg.generic to regular form to enable fusion with convolution and GEMM";
  let dependentDialects = ["rock::RockDialect", "linalg::LinalgDialect", "memref::MemRefDialect"];
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

#endif // MLIR_DIALECT_ROCK_PASSES
