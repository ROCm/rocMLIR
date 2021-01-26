//===- mlir-rocm-runner.cpp - MLIR ROCM Execution Driver-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the GPU by
// translating MLIR to ROCDL/LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/ROCm/BackendUitls.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> tripleName("triple", cl::desc("target triple"),
                                       cl::value_desc("triple string"),
                                       cl::init(""));

static cl::opt<std::string> targetChip("target", cl::desc("target chip"),
                                       cl::value_desc("AMDGPU ISA version"),
                                       cl::init(""));

static cl::opt<std::string> features("feature", cl::desc("target features"),
                                     cl::value_desc("AMDGPU target features"),
                                     cl::init(""));

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  BackendUtils utils(tripleName, targetChip, features);

  pm.addPass(createGpuKernelOutliningPass());
  auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(createStripDebugInfoPass());
  kernelPm.addPass(createLowerGpuOpsToROCDLOpsPass());
  kernelPm.addPass(createConvertGPUKernelToBlobPass(
      [&utils](Operation *m) { return utils.compileModuleToROCDLIR(m); },
      [&utils](const std::string isa, Location loc, StringRef name) {
        return utils.compileISAToHsaco(isa, loc, name);
      },
      utils.getTriple(), utils.getChip(), utils.getFeatures(),
      /*gpuBinaryAnnotation=*/"rocdl.hsaco"));
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createConvertGpuLaunchFuncToGpuRuntimeCallsPass(
      /*gpuBinaryAnnotation=*/"rocdl.hsaco"));

  return pm.run(m);
}

int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  mlir::registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  // Initialize LLVM AMDGPU backend.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

  mlir::initializeLLVMPasses();
  return mlir::JitRunnerMain(argc, argv, runMLIRPasses);
}
