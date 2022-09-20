//===- rocmlir-driver.cpp - MLIR Rock Dialect Driver ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for rocmlir-driver.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Pipelines/XMIRPipelines.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
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

static cl::opt<std::string>
    kernelPipeline("kernel-pipeline",
                   cl::desc("rocmlir-driver kernel pipeline list"),
                   cl::value_desc("comma separated list of rock pipelines: "
                                  "applicability,gpu,rocdl,binary or full"),
                   cl::init(""));

static cl::opt<std::string>
    hostPipeline("host-pipeline",
                 cl::desc("rocmlir-driver host pipeline list"),
                 cl::value_desc("comma separated list of rock pipelines: "
                                "partition,highlevel,execmodel or full"),
                 cl::init(""));

static cl::opt<bool> legacyRockPipeline("c", cl::Hidden, cl::init(false),
                                          cl::Optional,
                                          cl::cb<void, bool>([](bool v) {
                                            if (v) {
                                              kernelPipeline.setValue("gpu");
                                            }
                                          }));

/////////////////////////////////////////////////////////////////////////////
//// Backend target spec
static cl::opt<bool> cpuOnly("cpu-only", cl::Hidden, cl::init(false),
                             cl::Optional);

static cl::opt<int> gpuOpt("gO",
                           cl::desc("Optimization level for GPU compilation"),
                           cl::value_desc("Integer from 0 to 3"), cl::init(3));

static cl::opt<std::string>
    tripleName("triple", cl::desc("target triple: amdgcn-amd-amdhsa"),
               cl::value_desc("triple string"), cl::init(""));

static cl::opt<std::string>
    targetChips("targets", cl::desc("list of target chips"), cl::init(""));

static cl::alias aliasTargetChips("target", cl::aliasopt(targetChips));

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
              llvm::SmallDenseSet<StringRef> &pipelineOptions,
              llvm::SmallDenseSet<StringRef> &fullOptions) {
  SmallVector<StringRef, 8> tokens;
  pipeline.split(tokens, ',');
  for (auto str : tokens) {
    auto opt = str.trim();
    if (opt.empty()) {
    } else if (opt == "full") {
      pipelineSet = fullOptions;
    } else if (pipelineOptions.contains(opt)) {
      pipelineSet.insert(opt);
    } else {
      auto opts = llvm::join(pipelineOptions, ",");
      llvm::errs() << "Invalid pipeline: " << opt << "\n"
                   << "   Valid options: " << opts << " or full\n";
      return failure();
    }
  }

  return success();
}

static LogicalResult
runKernelPipeline(StringRef chip, ModuleOp kmod, bool isHighLevel,
                  llvm::SmallDenseSet<StringRef> &kernelPipelineSet) {
  PassManager pm(kmod.getContext(), PassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  if (isHighLevel) {
    rock::BufferizeOptions opts;
    opts.disableRock = cpuOnly.getValue();
    rock::buildBufferizePipeline(pm, opts);
  }

  // Set up lowering pipeline.
  if (kernelPipelineSet.contains("applicability")) {
    rock::KernelOptions opts;
    opts.enableApplicability = true;
    rock::buildKernelPipeline(pm, opts);
  }
  if (kernelPipelineSet.contains("gpu")) {
    // Set up the default lowering pipeline which goes down to GPU dialect.
    rock::buildKernelPipeline(pm);
  }
  if (kernelPipelineSet.contains("rocdl")) {
    pm.addPass(createLowerGpuOpsToROCDLOpsPass(/*chipset=*/chip.str(),
                                               /*indexBitWidth=*/32,
                                               /*useBarePtrCallConv*/ true));
  }
  if (kernelPipelineSet.contains("binary")) {
    // Set up the lowering pipeline which goes down to ELF Binary
    int optLevel = gpuOpt.getValue();
    if (optLevel < 0 || optLevel > 3) {
      llvm::errs() << "Invalid GPU optimization level: " << optLevel << "\n";
      return failure();
    }

    rock::BackendOptions opts;
    opts.triple = tripleName.getValue();
    opts.chip = chip.str();
    opts.features = features.getValue();
    opts.optLevel = optLevel;
    rock::buildBackendPipeline(pm, opts);
  }

  return pm.run(kmod);
}

static LogicalResult runMLIRPasses(ModuleOp &module,
                                   mlir::PassPipelineCLParser &passPipeline) {

  llvm::SmallVector<StringRef, 4> chips;
  StringRef targetsStr = targetChips.getValue();
  SmallVector<StringRef, 4> tokens;
  targetsStr.split(tokens, ',');
  for (auto str : tokens) {
    auto chip = str.trim();
    if (!chip.empty())
      chips.push_back(chip);
  }

  llvm::SmallDenseSet<StringRef> kernelPipelineOptions{"applicability", "gpu",
                                                       "rocdl", "binary"};
  llvm::SmallDenseSet<StringRef> kernelFullPipeline{"gpu", "binary"};
  llvm::SmallDenseSet<StringRef> kernelPipelineSet;
  if (failed(parsePipeline(kernelPipeline.getValue(), kernelPipelineSet,
                           kernelPipelineOptions, kernelFullPipeline))) {
    return failure();
  }
  if (kernelPipelineSet.size()) {
    // test for target spec
    if (kernelPipelineSet.contains("binary")) {
      if (tripleName.empty() || chips.empty()) {
        llvm::errs() << "Target triple (-triple) and chip (-target) not "
                        "specified for binary backend\n";
        return failure();
      }
    } else if (kernelPipelineSet.contains("rocdl")) {
      if (chips.empty()) {
        llvm::errs()
            << "Target chip (-target) not specified for ROCDL backend\n";
        return failure();
      }
    } else if (!tripleName.empty() || !chips.empty() || !features.empty()) {
      llvm::errs() << "Target (-triple,-target,-features) should not be set "
                      "except for kernel-pipeline=binary\n";
      return failure();
    }

    if (kernelPipelineSet.contains("applicability") &&
        kernelPipelineSet.size() != 1) {
      llvm::errs() << "The `applicability` pipeline cannot be combined with "
                      "any other pipeline options.\n";
      return failure();
    }
  }

  llvm::SmallDenseSet<StringRef> hostPipelineOptions{"partition", "highlevel",
                                                     "xmodel"};
  llvm::SmallDenseSet<StringRef> hostPipelineSet;
  if (failed(parsePipeline(hostPipeline.getValue(), hostPipelineSet,
                           hostPipelineOptions, hostPipelineOptions))) {
    return failure();
  }

  // Run partitioning pipeline.
  if (hostPipelineSet.contains("partition")) {
    PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
    applyPassManagerCLOptions(pm);

    rock::PartitionOptions opts;
    opts.cloneToRockModule = !cpuOnly.getValue();
    opts.targetChips = chips;
    rock::buildPartitionPipeline(pm, opts);

    if (failed(pm.run(module))) {
      return failure();
    }
  }

  bool isHighLevel = hostPipelineSet.contains("highlevel");

  // Find kernel module, defaults to top module
  bool hasKernels = false;
  if (kernelPipelineSet.size() || isHighLevel) {
    LogicalResult kernelResult = success();
    // If sub-modules exists with kernel.chip specified and in set
    // of targetChips, run KernelPipeline
    module->walk([&](ModuleOp kernelModule) {
      auto chipAttr = kernelModule->getAttrOfType<StringAttr>("kernel.chip");
      hasKernels |= (bool)chipAttr;
      if (chipAttr && llvm::find(chips, chipAttr.getValue())) {
        kernelResult = runKernelPipeline(chipAttr.getValue(), kernelModule,
                                         isHighLevel, kernelPipelineSet);
      }
    });
    if (!hasKernels) {
      // If no sub-modules, run KernelPipeline on top-level module
      StringRef chip(chips.size() ? chips.front() : "");
      kernelResult =
          runKernelPipeline(chip, module, isHighLevel, kernelPipelineSet);
    }
    if (failed(kernelResult))
      return kernelResult;
  } else {
    PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
    applyPassManagerCLOptions(pm);

    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(module.getContext())) << msg;
      return failure();
    };

    // Use lowering pipeline specified at command line.
    if (failed(passPipeline.addToPipeline(pm, errorHandler))) {
      return failure();
    }
    if (failed(pm.run(module))) {
      return failure();
    }
  }

  // Run Bufferization on the top module
  if (isHighLevel && hasKernels) {
    PassManager pm(module.getContext(), PassManager::Nesting::Implicit);
    applyPassManagerCLOptions(pm);
    rock::BufferizeOptions opts;
    opts.disableRock = true;
    rock::buildBufferizePipeline(pm, opts);

    if (failed(pm.run(module))) {
      return failure();
    }
  }

  // Run XModel generation on the top module
  if (hostPipelineSet.contains("xmodel")) {
    PassManager pm(module.getContext());
    applyPassManagerCLOptions(pm);
    xmir::buildModelPipeline(pm);
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
      .loadDialect<rock::RockDialect, func::FuncDialect, scf::SCFDialect,
                   AffineDialect, memref::MemRefDialect, math::MathDialect,
                   arith::ArithmeticDialect, gpu::GPUDialect,
                   bufferization::BufferizationDialect, async::AsyncDialect>();
  mlir::registerAllPasses();
  mlir::registerRocMLIRConversionPasses();
  rock::registerPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR Rock Dialect driver\n");

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
