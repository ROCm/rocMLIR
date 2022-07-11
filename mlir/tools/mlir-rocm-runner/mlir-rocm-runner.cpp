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


#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TableGen/Record.h"

#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MIOpen/utility/IsaNameSplitter.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include <cstdlib>
#include <mutex>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "hip/hip_runtime.h"
#pragma GCC diagnostic pop

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

static cl::opt<bool> barePointers(
    "bare-ptr-memref-kernels",
    cl::desc("Use bare pointers to represent memrefs when calling kernels"),
    cl::init(true));

static cl::opt<bool>
    rocdlInput("rocdl-input",
               cl::desc("input is in the MLIR LLVM/ROCDL dialect"),
               cl::init(false));

static constexpr const char kTargetTriple[] = "amdgcn-amd-amdhsa";

// As per the coding standard of LLVM, anonymous namespace should only be used
// for class declarations.
// https://llvm.org/docs/CodingStandards.html#anonymous-namespaces
// FIXME: avoid calling hipGetDeviceProperties in mlir-rocm-runner to prevent
// the out-of-handle problem when running multiple rocm instances concurrently.
static void getGpuGCNArchName(hipDevice_t device, std::string &gcnArchName) {
  hipDeviceProp_t props;
  hipError_t result = hipGetDeviceProperties(&props, device);
  if (result != hipSuccess) {
    gcnArchName = "";
    llvm_unreachable("hipGetDeviceProperties() should never fail");
    return;
  }

  const char *pArchName = props.gcnArchName;
  gcnArchName.assign(pArchName);
}

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  int optLevel = gpuOpt.getValue();
  if (optLevel < 0 || optLevel > 3) {
    llvm::errs() << "Invalid GPU optimization level: " << optLevel << "\n";
    return failure();
  }

  if (tripleName.empty() && targetChip.empty() && features.empty()) {
    tripleName = kTargetTriple;
    std::string gcnArchName;
    getGpuGCNArchName(0, gcnArchName);
    auto status =
        IsaNameSplitter::parseArchName(gcnArchName, targetChip, features);
    if (status.failed()) {
      llvm_unreachable("HIP ArchName parsing should never fail.");
    }
  }

  // Find MIOpen module and compile kernel funcs
  ModuleOp kernelModule = m;
  if (auto miopenModule = kernelModule.lookupSymbol<ModuleOp>(
          miopen::MIOpenDialect::kKernelModuleName)) {
    kernelModule = miopenModule;
  }

  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createGpuKernelOutliningPass());
  auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(createStripDebugInfoPass());
  if (!rocdlInput.getValue()) {
    kernelPm.addPass(
        createLowerGpuOpsToROCDLOpsPass(/*chipset=*/targetChip,
                                        /*indexBitWidth=*/32,
                                        /*useBarePtrCallConv=*/barePointers,
                                        /*runtime=*/gpu::amd::Runtime::HIP));
  }
  kernelPm.addPass(createGpuSerializeToHsacoPass(tripleName, targetChip,
                                                 features, optLevel));

  if (failed(pm.run(kernelModule))) {
    return failure();
  }

  // Host Compiler Pipeline
  PassManager pmHost(m.getContext());
  auto &funcPm = pmHost.nest<func::FuncOp>();
  funcPm.addPass(createGpuAsyncRegionPass());
  funcPm.addPass(createConvertMathToLLVMPass());
  pmHost.addPass(
      createGpuToLLVMConversionPass(/*kernelBarePtrCallConv=*/barePointers));
  pmHost.addPass(createAsyncToAsyncRuntimePass());
  pmHost.addPass(createConvertAsyncToLLVMPass());
  mlir::LowerToLLVMOptions lower_to_llvm_opts(m.getContext());
  pmHost.addPass(mlir::createConvertFuncToLLVMPass(lower_to_llvm_opts));
  pmHost.addPass(LLVM::createSoftwareBF16Pass());

  return pmHost.run(m);
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

  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
#endif

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
