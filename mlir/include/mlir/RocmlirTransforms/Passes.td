//===-- Passes.td - RocMLIR general pass definitions ---*- tablegen -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2023 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_TRANSFORMS_PASSES
#define ROCMLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CompileGpuWithComgrPass : Pass<"compile-gpu-with-comgr", "::mlir::gpu::GPUModuleOp"> {
  let summary = "compile gpu.module's of IR to HSACO with Comgr";
  let dependentDialects = ["ROCDL::ROCDLDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
  let options = [
    Option<"triple", "triple", "std::string", "",
      "Target triple to compile">,
    Option<"chip", "chip", "std::string", "",
      "Target chipset">,
    Option<"chipFeatures", "feaduters", "std::string", "",
      "Target chipset features">,
    Option<"optLevel", "opt-evel", "uint64_t", "3",
      "Optimization level to use during compilation">
  ];
}

#endif // ROCMLIR_TRANSFORMS_PASSES
