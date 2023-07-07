//===- xmir-runner.cpp - MLIR Execution Model Runner ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the CPU/GPU/etc
// by translating MLIR to LLVM IR with launch capabilities before JIT-compiling
// and executing the latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MHAL/Pipelines/Pipelines.h"

#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/CpuSystemDetect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/ExecutionEngine/RocmSystemDetect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdlib>
#include <mutex>

using namespace mlir;
using namespace llvm;

// CLI switch for launch-mode
static cl::opt<std::string> targetType("target-type",
                                       cl::desc("Kernel target type"),
                                       cl::value_desc("valid options: GPU,CPU"),
                                       cl::init("GPU"));
static cl::opt<std::string> targetArch("target-arch",
                                       cl::desc("Specify target architecture"),
                                       cl::init(""));

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static LogicalResult runMLIRPasses(Operation *m, JitRunnerOptions &options) {
  // Canonicalize target arch
  if (targetType == "GPU" && !targetArch.empty()) {
    RocmDeviceName devName;
    if (failed(devName.parse(targetArch))) {
      llvm::errs() << "Invalid ROCm GPU target spec: " << targetArch << "\n";
      return failure();
    }
    SmallString<64> canonicalArch;
    devName.getFullName(canonicalArch);
    targetArch = canonicalArch.str().str();
  }

  SystemDevice::Type targetTypeEnum =
      llvm::StringSwitch<SystemDevice::Type>(targetType)
          .CaseLower("gpu", SystemDevice::Type::EGPU)
          .CaseLower("cpu", SystemDevice::Type::ECPU);
  SmallVector<std::string, 4> targetTypes{targetType};

  SmallVector<std::string, 4> targetArchs;
  if (targetArch.getValue().empty()) {
    auto devices = SystemDevices::get<CpuSystemDetect, RocmSystemDetect>();
    auto device = devices.find(targetTypeEnum);
    if (succeeded(device))
      targetArchs.push_back(device.value()->getArch());
  } else {
    targetArchs.push_back(targetArch);
  }

  // Host Compiler/Scheduler Pipeline
  PassManager pm(m->getContext());
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  mhal::RunnerOptions opts;
  opts.targetTypes = targetTypes;
  opts.targetArchs = targetArchs;

  mhal::buildRunnerPipeline(pm, opts);
  pm.addPass(LLVM::createSoftwareBF16Pass());

  return pm.run(m);
}

int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  registerMLIRContextCLOptions();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  DialectRegistry registry;
  mlir::registerRocMLIRDialects(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
