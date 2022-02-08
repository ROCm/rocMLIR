//===- mlir-miopen-driver.cpp - MLIR MIOpen Dialect Driver ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-miopen-driver.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Conversion/MIOpenToGPU/MIOpenToGPU.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MIOpen/Generator/Conv2dGenerator.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/Pipeline.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ROCm/IsaNameParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "bf16convert.hpp"
#include <unordered_map>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                          llvm::cl::desc("<input file>"),
                                          llvm::cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<int> blockSize("block_size", cl::desc("Block size"),
                              cl::value_desc("Block size"), cl::init(0));

static cl::opt<int> gridSize("grid_size", cl::desc("Grid size"),
                             cl::value_desc("Grid size"), cl::init(0));

// Set up lowering pipeline.
// The default lowering pipeline compiles down to GPU dialect.
// The output of the pipeline can be piped to mlir-rocm-runner for execution.
//
// When users specify "-c -target=rocdl", compiles down to LLVM dialect.
// The output of the pipeline can be piped to mlir-translate for translation to
// LLVM IR.
static cl::opt<bool> miopenBuiltinPipeline(
    "miopen_pipeline", cl::desc("Compile with the specified pipeline"),
    cl::value_desc("By default, compiles down to GPU dialect. Set "
                   "-target=rocdl compiles to ROCDL dialect."),
    cl::init(false));

static cl::alias
    aliasMIOpenBuiltinPipeline("c", cl::aliasopt(miopenBuiltinPipeline));

static cl::opt<bool>
    highLevelPipeline("high_level_pipeline",
                      cl::desc("Compile with the specified pipeline"),
                      cl::init(false));

static cl::alias aliasHighLevelPipeline("hlp", cl::aliasopt(highLevelPipeline));

static cl::opt<std::string> loweringTargetDialect(
    "target",
    cl::desc("By default, compiles down to GPU dialect. Set "
             "-target=rocdl compiles to ROCDL dialect."),
    cl::value_desc("Target dialect"), cl::init("gpu"));

static cl::opt<int> deviceNum(
    "device",
    cl::desc("Device index on which to run the kernel (only with host code)"),
    cl::value_desc("Between 0 and number of GPUs on system. "
                   "Omission leaves current device intact."));
static cl::alias deviceShort("dev", cl::aliasopt(deviceNum));

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static void populateTuningPipeline(PassManager &pm) {
  pm.addPass(
      mlir::miopen::createAffixTuningParametersPass(blockSize, gridSize));
}

static LogicalResult runMLIRPasses(ModuleOp &module,
                                   mlir::PassPipelineCLParser &passPipeline) {
  PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  // Set up high-level pipeline.
  bool isHighLevel = highLevelPipeline.getValue();
  if (isHighLevel) {
    miopen::addHighLevelPipeline(pm);
  }

  // Set up lowering pipeline.
  if (miopenBuiltinPipeline.getValue()) {
    StringRef pipeline = loweringTargetDialect.getValue();
    if (pipeline == "tuning") {
      // Set up the default lowering pipeline which goes down to affix tuning
      // parameters
      populateTuningPipeline(pm);
    } else if (pipeline == "gpu") {
      // Set up the default lowering pipeline which goes down to GPU dialect.
      miopen::addPipeline(pm);
    } else if (pipeline == "rocdl") {
      // Set up the lowering pipeline which goes down to ROCDL dialect.
      miopen::addPipeline(pm);
      pm.addPass(createLowerGpuOpsToROCDLOpsPass(/*indexBitWidth=*/32));
    }
  } else {
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(module.getContext())) << msg;
      return failure();
    };

    // Use lowering pipeline specified at command line.
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();
  }

  return pm.run(module);
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  MLIRContext context(registry);
  context.loadDialect < miopen::MIOpenDialect, StandardOpsDialect,
      scf::SCFDialect, AffineDialect, memref::MemRefDialect,
      math::MathDialect, arith::ArithmeticDialect>();
  mlir::registerAllPasses();
  mlir::registerMIOpenConversionPasses();
  miopen::registerPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR MIOpen Dialect driver\n");

  OpBuilder builder(&context);
  ModuleOp module;

  std::string errorMessage;
  SourceMgr sourceMgr;
  OwningModuleRef moduleRef;

  // Set up the input file.
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Parse the input file.
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  moduleRef = parseSourceFile(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "Parse host harness " << inputFilename << " failed.\n";
    exit(1);
  }
  module = moduleRef.get();

  // Run MLIR passes with passed in tuning parameters
  if (failed(runMLIRPasses(module, passPipeline))) {
    llvm::errs() << "Lowering failed.\n";
    exit(1);
  }

  // Set up the output file.
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  module.print(output->os());
  output->keep();
  return 0;
}
