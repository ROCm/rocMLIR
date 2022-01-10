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

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Async/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdlib>
#include <mutex>

using namespace mlir;
using namespace llvm;

// CLI variables for -On options.
static cl::opt<int> gpuOpt("gO",
                           cl::desc("Optimization level for GPU compilation"),
                           cl::value_desc("Integer from 0 to 3"), cl::init(3));

static cl::opt<std::string> tripleName("triple", cl::desc("target triple"),
                                       cl::value_desc("triple string"),
                                       cl::init(""));

static cl::opt<std::string> targetChip("target", cl::desc("target chip"),
                                       cl::value_desc("AMDGPU ISA version"),
                                       cl::init(""));

static cl::opt<std::string> features("feature", cl::desc("target features"),
                                     cl::value_desc("AMDGPU target features"),
                                     cl::init(""));

static cl::opt<bool>
    rocdlInput("rocdl-input",
               cl::desc("input is in the MLIR LLVM/ROCDL dialect"),
               cl::init(false));

static cl::opt<bool>
    dumpAsm("dump-asm",
            cl::desc("Whether to dump the assembly or intermediate "
                     "IR during GPU kernel convolution"),
            cl::init(false));

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  bool systemOverride = false;
  if (tripleName.empty() && targetChip.empty() && features.empty()) {
    systemOverride = true;
  }

  int optLevel = gpuOpt.getValue();
  if (optLevel < 0 || optLevel > 3) {
    llvm::errs() << "Invalid GPU optimization level: " << optLevel << "\n";
    return failure();
  }
  BackendUtils utils(tripleName, targetChip, features, systemOverride);

  pm.addPass(createLowerToCFGPass());
  pm.addPass(createGpuKernelOutliningPass());
  auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(createStripDebugInfoPass());
  if (!rocdlInput.getValue()) {
    kernelPm.addPass(
        createLowerGpuOpsToROCDLOpsPass(/*indexBitWidth=*/32,
                                        /*runtime=*/gpu::amd::Runtime::HIP));
  }
  kernelPm.addPass(createGpuSerializeToHsacoPass(
      utils.getTriple(), utils.getChip(), utils.getFeatures(), optLevel,
      dumpAsm.getValue()));
  auto &funcPm = pm.nest<FuncOp>();
  funcPm.addPass(createGpuAsyncRegionPass());
  funcPm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createGpuToLLVMConversionPass());
  pm.addPass(createAsyncToAsyncRuntimePass());
  pm.addPass(createConvertAsyncToLLVMPass());
  mlir::LowerToLLVMOptions lower_to_llvm_opts(m.getContext());
  pm.addPass(mlir::createLowerToLLVMPass(lower_to_llvm_opts));

  return pm.run(m);
}

int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Initialize LLVM AMDGPU backend.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

  mlir::initializeLLVMPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
#endif

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
