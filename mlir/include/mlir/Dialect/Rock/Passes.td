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

def RockPack4BitGpuOpsTo8BitPass
    : Pass<"rock-pack-4bit-gpu-ops-to-8bit", "::mlir::func::FuncOp"> {
  let summary =
      "Pack GPU memory operations on 4-bit element types to use 8-bit storage";
  let description = [{
    Change the physical storage of gpu.alloc buffers whose memref element type
    is the 4-bit integer or float (printed as f4E2M1FN) to i8 while preserving the original
    logical buffer size. This transformation packs two 4-bit values into one 8-bit value
    (halving the last dimension). It handles gpu.alloc, gpu.dealloc, and gpu.memcpy operations,
    preserving async semantics and host_shared attributes. This is necessary as GPU memory
    operations cannot do subbyte addressing.
  }];
  let dependentDialects = ["gpu::GPUDialect"];
}

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

def RockEmulateNarrowTypePass : Pass<"rock-emulate-narrow-type", "::mlir::func::FuncOp"> {
  let summary = "Emulate memrefs of 4-bit integers (or 2-bit, maybe)";
  let description = [{
    Coordinates the rewrite patterns present in upstream MLIR for supporting
    i4 memrefs by rewriting them to i8 ones. Conveniently, we already have our
    memrefs linearized when we call this pass so they won't need to linearize
    anything.

    This must be called after rock-sugar-to-loops as we won't be writing
    patterns for dealing with transforming_for and its friends here.

    This pass also lowers vector.transfer_read/vector.transfer_write to vector
    load and store in order to enable these narrow type emulation patterns.
  }];
  // Type converter mentions these, make sure they're loaded
  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::func::FuncDialect",
    // Some of the lowering patterns use this
    "::mlir::affine::AffineDialect"
  ];
}

def RockGemmToGridwisePass : Pass<"rock-gemm-to-gridwise", "::mlir::func::FuncOp"> {
  let summary = "convert gemm to gridwise gemm, adding padding";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
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

def RockLinalgAlignPass : Pass<"rock-linalg-align", "::mlir::func::FuncOp"> {
  let summary = "expand linalg ops aligned with threadwise copy";
  let dependentDialects = ["rock::RockDialect", "linalg::LinalgDialect", "vector::VectorDialect", "memref::MemRefDialect", "gpu::GPUDialect"];
}

def RockBlockwiseGemmToThreadwisePass : Pass<"rock-blockwise-gemm-to-threadwise", "::mlir::func::FuncOp"> {
  let summary = "Expand blockwise gemm into threadwise gemm and clean up fusion-related shorthand";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect", "gpu::GPUDialect", "memref::MemRefDialect", "scf::SCFDialect", "vector::VectorDialect"];
}

def RockThreadwiseGemmLoweringPass : Pass<"rock-threadwise-gemm-lowering", "::mlir::func::FuncOp"> {
  let summary = "expand threadwise gemm to final implementation.";
  let dependentDialects = ["gpu::GPUDialect", "rock::RockDialect", "amdgpu::AMDGPUDialect", "vector::VectorDialect", "affine::AffineDialect", "memref::MemRefDialect", "scf::SCFDialect"];
}

def RockAnalyzeMemoryUsePass : Pass<"rock-analyze-memory-use", "::mlir::func::FuncOp"> {
  let summary = "Expand shorthand, like transforming_for and extract_slice, to other dialects";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect", "LLVM::LLVMDialect"];
}

def RockSugarToLoopsPass : Pass<"rock-sugar-to-loops", "::mlir::func::FuncOp"> {
  let summary = "Expand shorthand, like transforming_for and extract_slice, to other dialects";
  let dependentDialects = ["rock::RockDialect", "vector::VectorDialect", "arith::ArithDialect",
    "gpu::GPUDialect", "memref::MemRefDialect", "affine::AffineDialect", "amdgpu::AMDGPUDialect", "LLVM::LLVMDialect", "scf::SCFDialect"];
}

def RockCleanMathPass : Pass<"rock-clean-math", "::mlir::func::FuncOp"> {
  let summary = "Clean up math using integer range analysis and other MLIR passes";
}

def RockPipelinePass : Pass<"rock-pipeline", "::mlir::func::FuncOp"> {
  let options = [
    Option<"removeStages", "rock-pipeline-remove-stages", "bool", "true",
      "Remove pipeline stages once the pipeline pass is completed">
  ];
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect"];
  let summary = "Pipeline loops";
}

def RockVectorizeFusionsPass : Pass<"rock-vectorize-fusions", "::mlir::func::FuncOp"> {
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect"];
  let summary = "Vectorize affine element-wise loops";
}

def RockBufferLoadMergePass : Pass<"rock-buffer-load-merge", "::mlir::func::FuncOp"> {
  let summary = "Merge identical memory loads to buffers only read. Assumes noalias.";
  let dependentDialects = ["::mlir::amdgpu::AMDGPUDialect"];
}

def RockTransformToMemrefPass : Pass<"rock-transform-to-memref", "::mlir::func::FuncOp"> {
  let summary = "convert remaining rock.transform ops to memref.expand/collapse_shape";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect", "gpu::GPUDialect", "vector::VectorDialect", "memref::MemRefDialect"];
}

def RockLoopsToCfPass : Pass<"rock-loops-to-cf", "::mlir::func::FuncOp"> {
  let summary = "expand loop / affine dialects to control flow. Notice GPU dialect will explicitly NOT be used in this pass";
  let dependentDialects = ["scf::SCFDialect","cf::ControlFlowDialect"];
}

def RockLowerReducePass : Pass<"rock-lower-reduce", "::mlir::func::FuncOp"> {
  let summary = "Lower rock.reduce operator";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect", "gpu::GPUDialect"];
}

def RockPrepareLLVMPass : Pass<"rock-prepare-llvm", "::mlir::LLVM::LLVMFuncOp"> {
  let summary = "prepare the generated code for llvm";
  let dependentDialects = ["ROCDL::ROCDLDialect"];
}

def RockCheckResidencyPass : Pass<"rock-check-residency"> {
  let summary = "For global_sync check for full kernel launch residency";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def RockOutputSwizzlePass : Pass<"rock-output-swizzle", "::mlir::func::FuncOp"> {
  let summary = "This pass changes ThreadwiseWriteAllOp to swizzle on LDS before writing to GPU memory";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect"];
}

def RockReuseLDSPass : Pass<"rock-reuse-lds", "::mlir::func::FuncOp"> {
  let summary = "This pass re-uses LDS memory by using the lifetime "
                "annotations (rock.live_in, rock.live_out)";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect"];
}

def RockShuffleGemmForReductions : Pass<"rock-shuffle-gemm-for-reductions", "::mlir::func::FuncOp"> {
  let summary = "This pass shuffles parallel gemm dimensions for equally block splits for fused reductions (if any)";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect"];
}

def RockGemmLinalgSplitkNormalizationPass : Pass<"rock-gemm-linalg-splitk-normalization", "::mlir::func::FuncOp"> {
  let summary = "This pass modifies linalg.generic for split-k fusions. It converts any arith.addf/arith.subf gemmOut, other to arith.addf gemmOut, other/splitkFactor";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockSortDimensionsMemoryLayoutPass : Pass<"rock-sort-dimensions-memory-layout", "::mlir::func::FuncOp"> {
  let summary = "Sort dimensions by stride of the underlying memory layout";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect", "arith::ArithDialect", "linalg::LinalgDialect"];
}

def RockFindFirstGemmIndexPass
    : Pass<"rock-find-first-gemm-index", "::mlir::func::FuncOp"> {
  let summary = "Fusions input arguments and linalg.generic input arguments "
                "don't have to match. This pass sets the first gemm index of "
                "the linalg.generic correctly.";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect",
                           "linalg::LinalgDialect"];
}

def RockRemoveOutputAllocPass
    : Pass<"rock-remove-output-alloc", "::mlir::func::FuncOp"> {
  let summary = "Remove the alloc assigned for output tensors once the output "
                "tensor has already been added to the list of function args";
  let dependentDialects = ["rock::RockDialect", "func::FuncDialect"];
}

def RockBlockwiseLoadTileToThreadwisePass
    : Pass<"rock-blockwise-load-tile-to-threadwise", "::mlir::func::FuncOp"> {
  let summary =
      "This pass lowers `rock.blockwise_load_tile` to rock.threadwise_* ops";
  let dependentDialects = ["rock::RockDialect", "affine::AffineDialect",
                           "arith::ArithDialect", "memref::MemRefDialect"];
}

def RockAnnotateLivenessPass
    : Pass<"rock-annotate-liveness", "::mlir::func::FuncOp"> {
  let summary = "This pass annotates LDS memory with liveness ops "
                "(rock.live_in, rock.live_out)";
  let dependentDialects = ["rock::RockDialect", "memref::MemRefDialect"];
}

def RockAddAsyncWaitPass : Pass<"rock-add-async-wait", "::mlir::func::FuncOp"> {
  let summary = "This pass adds async wait operations for LDS memory";
  let dependentDialects = ["rock::RockDialect"];
}

def RockAddAliasInfoPass : Pass<"rock-add-alias-info", "::mlir::gpu::GPUModuleOp"> {
  let summary = "This pass adds alias information to direct-to-LDS load/store and local load/store operations";
  let description = [{
    This pass annotates operations that perform direct loads or stores to LDS and local loads or stores with
    appropriate alias scope information. This includes:
    - rocdl.load_to_lds operations (direct loads to LDS)
    - llvm.load operations from global memory to LDS  (local loads)
    - llvm.store operations to LDS from global memory (local stores)
    
    The alias information helps the compiler understand that these operations
    access distinct memory regions and can be optimized accordingly.
  }];
  let dependentDialects = ["amdgpu::AMDGPUDialect", "LLVM::LLVMDialect", "gpu::GPUDialect", "ROCDL::ROCDLDialect"];
}

def ConvertRockOpsToROCDLOps
    : Pass<"rock-to-rocdl", "::mlir::LLVM::LLVMFuncOp"> {
  let summary = "This pass lowers async wait operations to ROCDL";
  let options = [Option<"chipset", "chipset", "std::string",
                        /*default=*/"\"gfx000\"",
                        "Chipset that these operations will run on">,
  ];
  let dependentDialects = ["rock::RockDialect", "ROCDL::ROCDLDialect",
                           "LLVM::LLVMDialect"];
}

#endif // MLIR_DIALECT_ROCK_PASSES
