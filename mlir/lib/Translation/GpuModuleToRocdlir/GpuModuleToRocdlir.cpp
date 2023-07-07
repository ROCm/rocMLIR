//===- GpuModuleToRocdlir.cpp - ROCDL in GPU modules to LLVM IR ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for ROCDL dialect to LLVM IR translation in
// GPU modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Translation/GpuModuleToRocdir.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

void mlir::rock::registerGpuModuleToROCDLIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "gpu-module-to-rocdlir", "rocdlir translation in gpu module",
      [](ModuleOp module, raw_ostream &output) {
        // Locate a GPU module within a Module. Use it if we find one.
        Operation *m = nullptr;
        auto *block = module.getBody();
        for (auto op = block->begin(); op != block->end(); ++op)
          if (auto gpuModule = dyn_cast<gpu::GPUModuleOp>(op)) {
            m = gpuModule;
            break;
          }

        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(m, llvmContext);
        if (!llvmModule)
          return failure();

        StringRef amdgcnTriple = "amdgcn-amd-amdhsa";
        StringRef amdgcnDataLayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:"
                                     "64:64-p5:32:32-p6:32:32-i64:64-"
                                     "v16:16-v24:32-v32:32-v48:64-v96:128-v192:"
                                     "256-v256:256-v512:512-v1024:"
                                     "1024-v2048:2048-n32:64-S32-A5-ni:7";
        llvmModule->setTargetTriple(amdgcnTriple);
        llvmModule->setDataLayout(amdgcnDataLayout);

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<mlir::gpu::GPUDialect>();
        mlir::registerGPUDialectTranslation(registry);
        mlir::registerROCDLDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
      });
}
