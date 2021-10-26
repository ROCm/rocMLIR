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
// MIOpenToGPU within an existing GPUModule
//===----------------------------------------------------------------------===//

def ConvertMIOpenWithinGPUModule : Pass<"test-miopen-lowering-gpu-module", "gpu::GPUModuleOp"> {
  let summary = "Lower the operations from the MIOpen dialect within an "
                "existing GPU module";
  let constructor = "mlir::createLowerMIOpenOpsWithinGPUModulePass()";
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

def TosaToMIOpen : FunctionPass<"tosa-to-miopen"> {
  let summary = "Lower TOSA to MIOpen";
  let description = [{
    Pass that converts TOSA operations to bufferized MIOpen operations.
  }];

  let constructor = "tosa::createTosaToMIOpenPass()";
}

//===----------------------------------------------------------------------===//
// MIGraphXToTosa
//===----------------------------------------------------------------------===//

def MIGraphXToTosa : FunctionPass<"migraphx-to-tosa"> {
  let summary = "Lower MIGraphX to TOSA";
  let description = [{
    Pass that converts MIGraphX operations to TOSA operations.
  }];

  let constructor = "migraphx::createMIGraphXToTosa()";
}

#endif // MIOPEN_MLIR_CONVERSION_PASSES
