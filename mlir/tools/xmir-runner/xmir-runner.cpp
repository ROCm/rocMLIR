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

#include "mlir/Dialect/MIOpen/XMIRPipelines.h"

#include "mlir/ExecutionEngine/CpuSystemDetect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/RocmSystemDetect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"

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

  auto testChip = [](StringRef chip) -> bool {
    if (targetChip.getValue().empty()) {
      auto devices = SystemDevices::get<CpuSystemDetect, RocmSystemDetect>();
      return devices.find(chip.str()) != devices.end();
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
  mlir::registerLLVMDialectTranslation(registry);
#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
#endif

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
