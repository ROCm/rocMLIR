//===-- MIOpenPasses.td - Conversion pass definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MIOPEN_MLIR_CONVERSION_PASSES
#define MIOPEN_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// MIOpenToGPU
//===----------------------------------------------------------------------===//

def ConvertMIOpenToGPU : Pass<"convert-miopen-to-gpu", "ModuleOp"> {
  let summary = "Lower the operations from the MIOpen dialect into the GPU "
                "dialect";
  let constructor = "mlir::createLowerMIOpenOpsToGPUPass()";
  let options = [
    Option<"kernelName", "kernel-name", "std::string",
           "\"miopen_conv2d_kcyx_nchw_nkhw\"",
           "kernel name to be lowered">,
    Option<"gpuModuleName", "gpu-module-name", "std::string",
           "\"miopen_kernel_module\"",
           "GPU kernel module name to be lowered">,
  ];
  let dependentDialects = [
    "miopen::MIOpenDialect",
    "AffineDialect",
    "scf::SCFDialect",
    "StandardOpsDialect",
    "vector::VectorDialect",
    "gpu::GPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// TosaToMIOpen
//===----------------------------------------------------------------------===//

def TosaToMIOpen : Pass<"tosa-to-miopen", "::mlir::FuncOp"> {
  let summary = "Lower TOSA to MIOpen";
  let description = [{
    Pass that converts TOSA operations to bufferized MIOpen operations.
  }];

  let constructor = "tosa::createTosaToMIOpenPass()";
}

//===----------------------------------------------------------------------===//
// MIGraphXToTosa
//===----------------------------------------------------------------------===//

def MIGraphXToTosa : Pass<"migraphx-to-tosa", "::mlir::FuncOp"> {
  let summary = "Lower MIGraphX to TOSA";
  let description = [{
    Pass that converts MIGraphX operations to TOSA operations.
  }];

  let constructor = "migraphx::createMIGraphXToTosaPass()";
}

//===----------------------------------------------------------------------===//
// GPUToMIGraphX
//===----------------------------------------------------------------------===//

def GPUToMIGraphX : Pass<"gpu-to-migraphx", "::mlir::FuncOp"> {
  let summary = "Lower fused gpu launch to MIGraphX code object";
  let description = [{
    Pass that converts func operations with gpu.launch to MIGraphX operation.
  }];

  let constructor = "migraphx::createGPUToMIGraphXPass()";
  let dependentDialects = [
    "migraphx::MIGraphXDialect",
    "StandardOpsDialect",
    "gpu::GPUDialect"
  ];
}

#endif // MIOPEN_MLIR_CONVERSION_PASSES
// GPUToMIGraphX
//===----------------------------------------------------------------------===//

def GPUToMIGraphX : FunctionPass<"gpu-to-migraphx"> {
  let summary = "Lower fused gpu launch to MIGraphX code object";
  let description = [{
    Pass that converts func operations with gpu.launch to MIGraphX operation.
  }];

  let constructor = "migraphx::createGPUToMIGraphX()";
  let dependentDialects = [
    "migraphx::MIGraphXDialect",
    "StandardOpsDialect",
    "gpu::GPUDialect"
  ];
}

#endif // MIOPEN_MLIR_CONVERSION_PASSES
