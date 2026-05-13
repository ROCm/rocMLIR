//===-- RocMLIRPasses.td - Conversion pass definition file -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_CONVERSION_PASSES
#define ROCMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// RockToGPU
//===----------------------------------------------------------------------===//

def ConvertRockToGPUPass : Pass<"convert-rock-to-gpu", "ModuleOp"> {
  let summary = "Lower the operations from the Rock dialect into the GPU "
                "dialect";
  let dependentDialects = [
    "rock::RockDialect",
    "amdgpu::AMDGPUDialect",
    "cf::ControlFlowDialect",
    "DLTIDialect",
    "func::FuncDialect",
    "gpu::GPUDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "affine::AffineDialect",
    "mhal::MHALDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FixTosaCastRoundingPass
//===----------------------------------------------------------------------===//

def FixTosaCastRoundingPass
    : Pass<"fix-tosa-cast-rounding", "::mlir::func::FuncOp"> {
  let summary = "Change tosa.cast float-to-int from round-to-nearest-even to "
                "round-towards-zero";
  let description = [{
    The upstream tosa-to-linalg pass inserts math.roundeven before arith.fptosi
    when lowering tosa.cast, implementing TOSA's round-to-nearest-even
    semantics. This pass removes those math.roundeven ops so that
    arith.fptosi's native truncation (round towards zero) is used instead,
    matching ONNX and PyTorch cast semantics. Note that this intentionally
    diverges from the TOSA spec; rocMLIR primarily serves ONNX/MIGraphX
    workloads where RTZ is the expected behavior.

    Only math.roundeven ops that belong to an RTZ-tagged cast lowering are
    removed; this preserves round-to-nearest-even for quantization casts.
    The tag is the FusedLoc metadata exposed as `mlir::rock::kRtzCastLocTag`
    and is set by the migraphx.convert lowering in migraphx-to-tosa. Because
    upstream tosa-to-linalg often rewrites the inner roundeven's location,
    the tag may end up on the math.roundeven itself, on the parent
    linalg.generic's location, or on one of the generic region's *output*
    block-argument locations; the pass checks all three. Input block-arg
    locations are deliberately ignored to avoid false positives when a
    downstream generic merely consumes a previously-tagged cast result.
    To stay safe the pass also bails on multi-output generics and on i1
    outputs.
  }];
  let dependentDialects = ["func::FuncDialect", "linalg::LinalgDialect",
                           "math::MathDialect"];
}

//===----------------------------------------------------------------------===//
// RocmlirCustomTosaDecomposePass
//===----------------------------------------------------------------------===//

def RocmlirCustomTosaDecomposePass : Pass<"rocmlir-custom-tosa-decompose", ""> {
  let summary = "Decompose RocMLIR's custom TOSA ops";
  let description = [{
    Pass that decomposes RocMLIR's custom TOSA operators.
  }];
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect",
                           "tensor::TensorDialect", "tosa::TosaDialect"];
}

//===----------------------------------------------------------------------===//
// RocmlirCustomTosaToLinalgPass
//===----------------------------------------------------------------------===//

def RocmlirCustomTosaToLinalgPass : Pass<"rocmlir-custom-tosa-to-linalg", ""> {
  let summary = "Lower RocMLIR's custom TOSA ops to Linalg";
  let description = [{
    Pass that converts RocMLIR's custom TOSA operators to `linalg`.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "tosa::TosaDialect"
  ];
}

//===----------------------------------------------------------------------===//
// TosaToRock
//===----------------------------------------------------------------------===//

def TosaToRockPass : Pass<"tosa-to-rock", "::mlir::func::FuncOp"> {
  let summary = "Lower TOSA to Rock";
  let description = [{
    Pass that converts TOSA operations to bufferized Rock operations.
  }];
  let dependentDialects = [
    "bufferization::BufferizationDialect",
    "func::FuncDialect",
    "rock::RockDialect",
    "tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MIGraphXToTosa
//===----------------------------------------------------------------------===//

def MIGraphXToTosaPass : Pass<"migraphx-to-tosa", "::mlir::func::FuncOp"> {
  let summary = "Lower MIGraphX to TOSA";
  let description = [{
    Pass that converts MIGraphX operations to TOSA operations.
  }];

  // The `arith`, `rock`, and `tensor` dialects appear in the conversion
  // target's legality rules (addLegalDialect / addDynamicallyLegalDialect /
  // markOpRecursivelyLegal<rock::AttentionOp>) so that rock.attention and
  // any pre-existing arith/tensor ops survive the partial conversion. They
  // need to be loaded by the time the pass runs even when the IR being
  // converted doesn't yet contain ops in those dialects (e.g. an off-tree
  // tool that schedules just this pass before MIGraphXAttentionToRock has
  // produced the rock.attention).
  //
  // The dialects nested inside rock.attention's region (linalg, math,
  // memref, bufferization, ...) are intentionally NOT listed here: this
  // pass only marks them recursively legal, never references their types
  // or creates ops in them, and ops inside an in-flight rock.attention
  // imply their dialect was already loaded at parse time.
  let dependentDialects = ["arith::ArithDialect", "func::FuncDialect",
                           "mhal::MHALDialect", "rock::RockDialect",
                           "tensor::TensorDialect", "tosa::TosaDialect",
  ];
}

//===----------------------------------------------------------------------===//
// EmulateFp8ExtTrunc
//===----------------------------------------------------------------------===//

def EmulateFp8ExtTruncPass : Pass<"emulate-fp8-ext-trunc"> {
  let summary = "Emulate arith.extf/truncf on fp8 types";
  let description = [{
    Pass that emulates arith.extf and arith.truncf on 8-bit float types
    so as to allow testing on non-MI-300 platforms.

    arith.extf is converted into lookups in a table generated at
    code generation-time.

    arith.truncf gets converted to calls to functions which this pass inserts.

    This is a quick implementation that provides the mimimal functionality to
    test 8-bit float kernels.

    This pass must be run at the `builtin.module`/`gpu.module` level
    (the root operation needs to have a symbol table)
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "func::FuncDialect",
    "memref::MemRefDialect",
    "vector::VectorDialect",
  ];
  let options = [
    Option<"hasFp8ConversionInstrs", "f8-conversion-instrs", "bool", /*default=*/"false",
           "if the chip has fp8 Nanoo to fp32 conversion instructions or not">,
    Option<"hasOcpFp8ConversionInstrs", "ocpf8-conversion-instrs", "bool", /*default=*/"false",
           "if the chip has OCP fp8 to fp32 conversion instructions or not">,
  ];
}

//===----------------------------------------------------------------------===//
// MIGraphXToLinalg
//===----------------------------------------------------------------------===//
def MIGraphXToLinalgPass : Pass<"migraphx-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Lower MIGraphX to Linalg";
  let description = [{
    Pass that converts MIGraphX operations to Linalg operations.
  }];

  let dependentDialects = ["arith::ArithDialect", "tensor::TensorDialect",
                           "linalg::LinalgDialect", "rock::RockDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToRock
//===----------------------------------------------------------------------===//
def LinalgToRockPass : Pass<"linalg-to-rock", "::mlir::func::FuncOp"> {
  let summary = "Lower Linalg to Rock";
  let description = [{
    Pass that converts Linalg operations to Rock operations.
  }];

  let dependentDialects = ["rock::RockDialect",
                           "bufferization::BufferizationDialect"];
}
//===----------------------------------------------------------------------===//
// MIGraphXAttentionToRock
//===----------------------------------------------------------------------===//
def MIGraphXAttentionToRockPass
    : Pass<"migraphx-attention-to-rock", "::mlir::func::FuncOp"> {
  let summary = "Lower migraphx.attention to rock.attention";
  let description = [{
    Pass that converts migraphx.attention operations directly to
    rock.attention operations for GPU compilation.

    Anchored to func::FuncOp for consistency with the other migraphx-side
    conversion pass (MIGraphXToTosaPass) and because the pipeline already
    schedules it inside a func-nested pass manager.
  }];
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect",
                           "math::MathDialect", "rock::RockDialect",
                           "bufferization::BufferizationDialect",
                           "memref::MemRefDialect", "tensor::TensorDialect",
                           "migraphx::MIGraphXDialect",
  ];
}

#endif // ROCMLIR_CONVERSION_PASSES
