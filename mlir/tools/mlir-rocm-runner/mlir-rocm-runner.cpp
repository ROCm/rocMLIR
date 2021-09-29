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
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Async/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/DialectConversion.h"

#include <mutex>

using namespace mlir;
using namespace llvm;

static cl::OptionCategory optFlags{"ROCm runner opt-like flags"};

// CLI variables for -On options.
static cl::opt<bool> optO0{
    "Og0", cl::desc("Run GPU opt passes and codegen at O0"), cl::cat(optFlags)};
static cl::opt<bool> optO1{
    "Og1", cl::desc("Run GPU opt passes and codegen at O1"), cl::cat(optFlags)};
static cl::opt<bool> optO2{
    "Og2", cl::desc("Run GPU opt passes and codegen at O2"), cl::cat(optFlags)};
static cl::opt<bool> optO3{
    "Og3", cl::desc("Run GPU opt passes and codegen at O3"), cl::cat(optFlags)};

static cl::opt<std::string> tripleName("triple", cl::desc("target triple"),
                                       cl::value_desc("triple string"),
                                       cl::init(""));

static cl::opt<std::string> targetChip("target", cl::desc("target chip"),
                                       cl::value_desc("AMDGPU ISA version"),
                                       cl::init(""));

static cl::opt<std::string> features("feature", cl::desc("target features"),
                                     cl::value_desc("AMDGPU target features"),
                                     cl::init(""));

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static LogicalResult runMLIRPasses(ModuleOp m) {
  m.getContext()->disableMultithreading();
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  bool systemOverride = false;
  if (tripleName.empty() && targetChip.empty() && features.empty()) {
    systemOverride = true;
  }

  int optLevel = 3;
  if (optO0) {
    optLevel = 0;
  }
  if (optO1) {
    optLevel = 1;
  }
  if (optO2) {
    optLevel = 2;
  }
  if (optO3) {
    optLevel = 3;
  }

  BackendUtils utils(tripleName, targetChip, features, systemOverride);

  pm.addPass(createLowerToCFGPass());
  pm.addPass(createGpuKernelOutliningPass());
  auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(createStripDebugInfoPass());
  kernelPm.addPass(createLowerGpuOpsToROCDLOpsPass(/*indexBitWidth=*/32));
  kernelPm.addPass(createGpuSerializeToHsacoPass(
      utils.getTriple(), utils.getChip(), utils.getFeatures(), optLevel));
  auto &funcPm = pm.nest<FuncOp>();
  funcPm.addPass(createGpuAsyncRegionPass());
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
