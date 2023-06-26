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
  let options = [
    Option<"kernelName", "kernel-name", "std::string",
           "\"rock_conv2d_kcyx_nchw_nkhw\"",
           "kernel name to be lowered">,
    Option<"gpuModuleName", "gpu-module-name", "std::string",
           "\"rock_kernel_module\"",
           "GPU kernel module name to be lowered">,
  ];
  let dependentDialects = [
    "rock::RockDialect",
    "amdgpu::AMDGPUDialect",
    "cf::ControlFlowDialect",
    "func::FuncDialect",
    "gpu::GPUDialect",
    "memref::MemRefDialect"
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

  let dependentDialects = [
    "func::FuncDialect",
    "tosa::TosaDialect"
  ];
}

//===----------------------------------------------------------------------===//
// GPUToMIGraphX
//===----------------------------------------------------------------------===//

def GPUToMIGraphXPass : Pass<"gpu-to-migraphx", "::mlir::func::FuncOp"> {
  let summary = "Lower fused gpu launch to MIGraphX code object";
  let description = [{
    Pass that converts func operations with gpu.launch to MIGraphX operation.
  }];

  let dependentDialects = [
    "migraphx::MIGraphXDialect",
    "func::FuncDialect",
    "gpu::GPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Fp8ExtToTables
//===----------------------------------------------------------------------===//

def Fp8ExtToTablesPass : Pass<"fp8-ext-to-tables"> {
  let summary = "Lower arith.extf on fp8 types to table lookup";
  let description = [{
    Pass that converts arith.extf on 8-bit float types to a lookup in a
    constant table.

    This is a quick implementation that provides the mimimal functionality to
    test 8-bit float kernels.

    This pass must be run at the `builtin.module`/`gpu.module` level
    (the root operation needs to have a symbol table)
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "vector::VectorDialect",
  ];
}

#endif // ROCMLIR_CONVERSION_PASSES
