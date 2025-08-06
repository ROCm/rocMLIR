//===- rocmlir-printer.cpp - MLIR Rock Dialect Printer ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for rocmlir-printer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Pipelines/Pipelines.h"
#include "mlir/Dialect/MIGraphX/Pipeline/Pipeline.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitRocMLIRCLOptions.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init("-"));

static cl::opt<std::string> bcFilename("bc", cl::desc("Bytcode filename"),
                                       cl::value_desc("filename"),
                                       cl::init("-"));

static cl::opt<std::string> kernelPipeline(
    "kernel-pipeline", cl::desc("rocmlir-driver kernel pipeline list"),
    cl::value_desc("comma separated list of rock pipelines: "
                   "applicability,migraphx,highlevel,gpu,rocdl,binary or full"),
    cl::init(""));

static cl::opt<std::string>
    hostPipeline("host-pipeline", cl::desc("rocmlir-driver host pipeline list"),
                 cl::value_desc("comma separated list of rock pipelines: "
                                "migraphx,highlevel,mhal,runner or full"),
                 cl::init(""));

static cl::opt<bool> legacyRockPipeline("c", cl::Hidden, cl::init(false),
                                        cl::Optional,
                                        cl::cb<void, bool>([](bool v) {
                                          if (v) {
                                            kernelPipeline.setValue("full");
                                            hostPipeline.setValue("runner");
                                          }
                                        }));

static cl::opt<bool> verifyPasses(
    "verify-passes", cl::init(false),
    cl::desc("Have the pass manager(s) run verification after each pass"));

static cl::opt<bool> dumpPipelines(
    "dump-pipelines", cl::init(false),
    cl::desc("Print out a textual form of the requested pipelines"));

/////////////////////////////////////////////////////////////////////////////
//// Backend target spec
static cl::opt<bool> cpuOnly("cpu-only", cl::Hidden, cl::init(false),
                             cl::Optional);

static cl::opt<int> gpuOpt("gO",
                           cl::desc("Optimization level for GPU compilation"),
                           cl::value_desc("Integer from 0 to 3"), cl::init(3));

static cl::opt<bool> barePointers(
    "bare-ptr-memref-kernels",
    cl::desc("Use bare pointers to represent memrefs when calling kernels"),
    cl::init(true));

static cl::opt<bool> hostAsyncCoroutines(
    "host-async-coroutines",
    cl::desc("Use coroutines when lowering async ops to LLVM"),
    // FIXME: This should be true to match upstream
    cl::init(false));

static cl::opt<std::string> targets("targets", cl::desc("list of target"),
                                    cl::init(""));

static cl::opt<std::string> arch("arch", cl::desc("target architecture"),
                                 cl::value_desc("Target GPU architecture"),
                                 cl::init(""));

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

void dumpToBytecode(ModuleOp module, llvm::raw_fd_ostream &os) {
  if (failed(writeBytecodeToFile(module.getOperation(), os))) {
    llvm::errs() << "Failed to write bytecode\n";
  }
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  MLIRContext context(registry);
  context.loadDialect<mhal::MHALDialect, rock::RockDialect, func::FuncDialect,
                      scf::SCFDialect, affine::AffineDialect,
                      memref::MemRefDialect, math::MathDialect,
                      arith::ArithDialect, gpu::GPUDialect,
                      bufferization::BufferizationDialect>();
  mlir::registerRocMLIRPasses();
  InitLLVM y(argc, argv);

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR Rock Dialect printer\n");
  OpBuilder builder(&context);
  ModuleOp module;

  std::string errorMessage;
  SourceMgr sourceMgr;
  OwningOpRef<ModuleOp> moduleRef;

  // Set up the input file.
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Parse the input file.
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  moduleRef = parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
    exit(1);
  }
  module = moduleRef.get();
  // Set up the output file.

  auto output = openOutputFile(bcFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  dumpToBytecode(module, output->os());

  output->keep();
  return 0;
}
