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
    "triton::TritonDialect"
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

def RockInsertLoadsPass : Pass<"rock-insert-loads", "::mlir::func::FuncOp"> {
  let summary = "Insert rock.load ops for tensor arguments used by non-viewlike ops";
  let description = [{
    This pass traverses the IR and for each operation that is NOT view-like
    (such as rock.transform), it traces the input operands back to the function
    arguments. For each unique tensor argument that is consumed by a non-viewlike
    operation (possibly through a chain of transforms), a rock.load operation is
    inserted.

    The rock.load operation is a placeholder that marks where actual loads from
    global memory must happen. This is used in later passes for proper memory
    access scheduling.

    If multiple operations load the same tensor argument, only a single rock.load
    is created for that tensor to avoid redundant loads.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockInsertOutputFusionLoadsPass : Pass<"rock-insert-output-fusion-loads", "::mlir::func::FuncOp"> {
  let summary = "Insert BlockwiseLoadTileOp for output fusion loads";
  let description = [{
    This pass runs AFTER GridwiseGemmToBlockwise and BEFORE RockLowerLoads.
    It creates rock.blockwise_load_tile for rock.load ops used in output fusions.

    Output fusion loads are rock.load ops that:
    1. Are NOT reachable from any existing rock.blockwise_load_tile (input loads)
    2. Feed into fusion ops (arith/math) that operate on the GEMM result

    Example:
      Before (after GridwiseGemmToBlockwise):
        %result = scf.for ... {
          %loadedA = rock.blockwise_load_tile %a[indices]
          ...
        }
        %fusionRoot = rock.untile %result
        %bias_t = rock.transform %bias
        %bias_loaded = rock.load %bias_t
        %fused = arith.addf %fusionRoot, %bias_loaded
        %out = rock.store %fused to %dest

      After:
        %result = scf.for ... { ... }
        %fusionRoot = rock.untile %result
        %bias_t = rock.transform %bias
        %bias_loaded = rock.load %bias_t  // Still exists for LowerLoads to trace
        %bias_wrapped = rock.transform %bias_loaded by <output_grid_subtile>
        %bias_tile = rock.blockwise_load_tile %bias_wrapped[g_block, m_block, n_block]
        %fused = arith.addf %fusionRoot, %bias_tile
        %out = rock.store %fused to %dest

    The chain BlockwiseLoadTileOp -> transform -> rock.load allows LowerLoads
    to trace back through rock.load and combine the pre-load transforms with
    the output grid subtile transforms.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect", "math::MathDialect"];
}

def RockLowerLoadsPass : Pass<"rock-lower-loads", "::mlir::func::FuncOp"> {
  let summary = "Lower rock.load ops for fusion roots to prepare for blockwise lowering";
  let description = [{
    This pass prepares rock.load ops for blockwise lowering by analyzing the
    transform chains and fusion patterns that feed into fusion root operations
    (ops with the RockFusionRoot trait like gridwise_gemm, attention, etc.).

    Given IR like:
      %a1 = rock.transform %arg1
      %a2 = rock.transform %arg2
      %l1 = rock.load %a1
      %l2 = rock.load %a2
      %a = arith.addf %l1, %l2      // input fusion
      %aFinal = rock.transform %a
      rock.gridwise_gemm(%aFinal, ...)

    This pass analyzes:
    1. Which rock.load ops feed into fusion roots
    2. The pre-load transforms (transforms before the load)
    3. The post-load transforms (transforms after the load/fusion)

    The ToBlockwise pass will then use this information to create
    blockwise_load_tile ops with the proper transform chains.
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockLowerStoresPass : Pass<"rock-lower-stores", "::mlir::func::FuncOp"> {
  let summary = "Lower rock.store ops and handle output fusions";
  let description = [{
    This pass processes rock.store ops after GridwiseGemmToBlockwise.
    It handles output fusions by:
    1. Finding rock.store ops and tracing back to blockwise_store_tile
    2. For output fusions (arith/math ops between bst and store):
       - Cloning fusion ops to operate on tiles
       - Creating blockwise_load_tile for other fusion operands
       - Creating new blockwise_store_tile with fused result

    Example (output fusion):
      Before:
        %bst = rock.blockwise_store_tile %tile -> %dest[indices]
        %bias_loaded = rock.load %bias_transformed
        %fused = arith.addf %bst, %bias_loaded
        %out = rock.store %fused to %dest_arg

      After:
        %bias_tile = rock.blockwise_load_tile %bias_transformed[indices]
        %fused_tile = arith.addf %tile, %bias_tile
        %out = rock.blockwise_store_tile %fused_tile -> %dest[indices]
  }];
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "arith::ArithDialect", "math::MathDialect"];
}

#endif // MLIR_DIALECT_ROCK_PASSES
