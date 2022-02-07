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
#include "mlir/ExecutionEngine/ROCm/BackendUitls.h"
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

static cl::opt<std::string> kernelPipeline(
    "kernel-pipeline", cl::desc("mlir-miopen-driver kernel pipeline list"),
    cl::value_desc(
        "comma separated list of miopen pipelines: tuning,gpu,rocdl,binary"),
    cl::init(""));

static cl::opt<std::string>
    hostPipeline("host-pipeline",
                 cl::desc("mlir-miopen-driver host pipeline list"),
                 cl::value_desc("comma separated list of miopen pipelines: "
                                "partition,highlevel,execmodel"),
                 cl::init(""));

static cl::opt<bool> legacyMiopenPipeline("c", cl::Hidden, cl::init(false),
                                          cl::Optional,
                                          cl::cb<void, bool>([](bool v) {
                                            if (v) {
                                              kernelPipeline.setValue("gpu");
                                            }
                                          }));

/////////////////////////////////////////////////////////////////////////////
//// Backend target spec
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

static cl::opt<int> blockSize("block_size",
                              cl::desc("Override block size for tuning"),
                              cl::value_desc("Block size"), cl::init(0));

static cl::opt<int> gridSize("grid_size",
                             cl::desc("Override grid size for tuning"),
                             cl::value_desc("Grid size"), cl::init(0));

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

static LogicalResult
parsePipeline(StringRef pipeline, llvm::SmallDenseSet<StringRef> &pipelineSet,
              llvm::SmallDenseSet<StringRef> &pipelineOptions) {
  size_t start = 0;
  size_t pos = 0;
  do {
    pos = pipeline.find_first_of(",", start);
    auto opt = pipeline.substr(start, pos).trim();
    if (opt.empty()) {
    } else if (opt == "all" || opt == "full") {
      pipelineSet = pipelineOptions;
    } else if (pipelineOptions.contains(opt)) {
      pipelineSet.insert(opt);
    } else {
      SmallString<256> opts;
      for (auto pipeline : pipelineOptions) {
        if (opts.size())
          opts += ", ";
        opts += pipeline;
      }
      llvm::errs() << "Invalid pipeline: " << opt << "\n"
                   << "   Valid options: " << opts << " or full\n";
      return failure();
    }
    start = pos + 1;
  } while (pos != StringRef::npos);

  return success();
}

static LogicalResult runMLIRPasses(ModuleOp &module,
                                   mlir::PassPipelineCLParser &passPipeline) {
  llvm::SmallDenseSet<StringRef> kernelPipelineOptions{"tuning", "gpu", "rocdl",
                                                       "binary"};
  llvm::SmallDenseSet<StringRef> kernelPipelineSet;
  if (failed(parsePipeline(kernelPipeline.getValue(), kernelPipelineSet,
                           kernelPipelineOptions))) {
    return failure();
  }

  llvm::SmallDenseSet<StringRef> hostPipelineOptions{"partition", "highlevel",
                                                     "xmodel"};
  llvm::SmallDenseSet<StringRef> hostPipelineSet;
  if (failed(parsePipeline(hostPipeline.getValue(), hostPipelineSet,
                           hostPipelineOptions))) {
    return failure();
  }

  // Run partitioning pipeline.
  if (hostPipelineSet.contains("partition")) {
    PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
    pm.addPass(tosa::createTosaPartitionPass());
    pm.addPass(miopen::createMIOpenCloneKernelsPass());

    if (failed(pm.run(module))) {
      return failure();
    }
  }

  // Find kernel module, defaults to top module
  auto kernelModule =
      module.lookupSymbol<ModuleOp>(miopen::MIOpenDialect::kKernelModuleName);
  if (!kernelModule) {
    kernelModule = module;
  }

  PassManager pm(kernelModule.getContext(), PassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  bool isHighLevel = hostPipelineSet.contains("highlevel");
  if (isHighLevel) {
    miopen::addHighLevelPipeline(pm);
  }

  // Set up lowering pipeline.
  if (kernelPipelineSet.size()) {
    if (kernelPipelineSet.size() == 1 && kernelPipelineSet.contains("tuning")) {
      // Set up the default lowering pipeline which goes down to affix tuning
      // parameters
      pm.addPass(
          mlir::miopen::createAffixTuningParametersPass(blockSize, gridSize));
    } else {
      // Set up the default lowering pipeline which goes down to GPU dialect.
      miopen::addPipeline(pm);
      if (kernelPipelineSet.contains("binary")) {
        // Set up the lowering pipeline which goes down to ELF Binary
        int optLevel = gpuOpt.getValue();
        if (optLevel < 0 || optLevel > 3) {
          llvm::errs() << "Invalid GPU optimization level: " << optLevel
                       << "\n";
          return failure();
        }
        if (targetChip.empty()) {
          llvm::errs()
              << "Target chip (-target) not specified for binary backend\n";
          return failure();
        }

        BackendUtils utils(tripleName, targetChip, features);
        miopen::addBackendPipeline(pm, utils.getTriple(), utils.getChip(),
                                   utils.getFeatures(), optLevel);
      } else if (kernelPipelineSet.contains("rocdl")) {
        // Set up the lowering pipeline which goes down to ROCDL dialect.
        pm.addPass(createLowerGpuOpsToROCDLOpsPass(/*indexBitWidth=*/32));
      }
    }
  } else {
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(kernelModule.getContext())) << msg;
      return failure();
    };

    // Use lowering pipeline specified at command line.
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();
  }

  if (failed(pm.run(kernelModule))) {
    return failure();
  }

  if (isHighLevel && kernelModule != module) {
    PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
    miopen::addHighLevelPipeline(pm, false);

    if (failed(pm.run(module))) {
      return failure();
    }
  }

  if (hostPipelineSet.contains("xmodel")) {
    PassManager pm(module.getContext());
    pm.addPass(miopen::createMIOpenApplyImplPass());
    if (failed(pm.run(module))) {
      return failure();
    }
  }

  return success();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  MLIRContext context(registry);
  context
      .loadDialect<miopen::MIOpenDialect, StandardOpsDialect, scf::SCFDialect,
                   AffineDialect, memref::MemRefDialect, math::MathDialect,
                   arith::ArithmeticDialect, gpu::GPUDialect>();
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
