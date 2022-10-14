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

#include "mlir/Dialect/Rock/Pipelines/XMIRPipelines.h"

#include "mlir/ExecutionEngine/CpuSystemDetect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/ExecutionEngine/RocmSystemDetect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/Pass/PassManager.h"
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
                                       cl::value_desc("valid options: gpu,cpu"),
                                       cl::init("gpu"));
static cl::opt<std::string>
    targetChip("target-chip", cl::desc("Specify target chip"), cl::init(""));

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static LogicalResult runMLIRPasses(ModuleOp m) {
  // Canonicalize target chip
  if (targetType == "gpu" && !targetChip.empty()) {
    RocmDeviceName devName;
    if (failed(devName.parse(targetChip))) {
      llvm::errs() << "Invalid ROCm GPU target spec: " << targetChip << "\n";
      return failure();
    }
    SmallString<64> canonicalArch;
    devName.getFullName(canonicalArch);
    targetChip = canonicalArch.str().str();
  }

  SystemDevice::Type targetTypeEnum =
      llvm::StringSwitch<SystemDevice::Type>(targetType)
          .CaseLower("gpu", SystemDevice::Type::EGPU)
          .CaseLower("cpu", SystemDevice::Type::ECPU);

  auto testChip = [&targetTypeEnum](StringRef chip) -> bool {
    if (targetChip.getValue().empty()) {
      auto devices = SystemDevices::get<CpuSystemDetect, RocmSystemDetect>();
      return succeeded(devices.find(targetTypeEnum, chip));
    }
    return targetChip == chip;
  };

  // walk and select targets
  OpBuilder b(m);
  m->walk([&](func::FuncOp func) {
    if (auto targets = func->getAttrOfType<ArrayAttr>("async.targets")) {
      DictionaryAttr targetDict;
      for (auto targetAttr : targets.getValue()) {
        auto dictAttr = targetAttr.cast<DictionaryAttr>();
        if (auto type = dictAttr.getAs<StringAttr>("type")) {
          auto chip = dictAttr.getAs<StringAttr>("arch");
          if (type == targetType && testChip(chip.getValue())) {
            // test perf?
            targetDict = dictAttr;
            break;
          }
        }
      }
      if (targetDict)
        func->setAttr("async.targets", b.getArrayAttr(targetDict));
      else {
        func->removeAttr("async.targets");
      }
    }
  });

  // Host Compiler/Scheduler Pipeline
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  xmir::buildRunnerPipeline(pm);

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

  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerRocMLIRDialects(registry);
  mlir::registerLLVMDialectTranslation(registry);
#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
#endif

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
