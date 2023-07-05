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
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
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
    hostPipeline("host-pipeline", cl::desc("rocmlir-driver host pipeline list"),
                 cl::value_desc("comma separated list of rock pipelines: "
                                "partition,highlevel,execmodel or full"),
                 cl::init(""));

static cl::opt<bool> legacyRockPipeline("c", cl::Hidden, cl::init(false),
                                        cl::Optional,
                                        cl::cb<void, bool>([](bool v) {
                                          if (v) {
                                            kernelPipeline.setValue("full");
                                            hostPipeline.setValue("runner");
                                          }
                                        }));

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
runKernelPipeline(StringRef arch, ModuleOp kmod, bool isHighLevel,
                  llvm::SmallDenseSet<StringRef> &kernelPipelineSet) {
  PassManager pm(kmod->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  bool needArch = kernelPipelineSet.contains("rocdl") ||
                  kernelPipelineSet.contains("binary");
  RocmDeviceName devName;
  if (arch.empty() && needArch) {
    llvm::errs()
        << "Architecture not specified for this pipeline, but one is required\n"
        << "Use --arch or set mhal.arch\n";
    return failure();
  }
  if (failed(devName.parse(arch)) && needArch) {
    llvm::errs() << "Invalid architecture: " << arch << "\n";
    return failure();
  }

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
    std::string chipset = devName.getChip().str();
    rock::AmdArchInfo archInfo = rock::lookupArchInfo(chipset);
    if (archInfo.hasFp8ConversionInstrs) {
      pm.addNestedPass<gpu::GPUModuleOp>(createArithToAMDGPUConversionPass());
    }
    pm.addPass(createFp8ExtToTablesPass());
    pm.addNestedPass<gpu::GPUModuleOp>(
        amdgpu::createAmdgpuEmulateAtomicsPass({chipset}));
    pm.addPass(
        createLowerGpuOpsToROCDLOpsPass(chipset,
                                        /*indexBitWidth=*/32,
                                        /*useBarePtrCallConv=*/barePointers));
  }
  if (kernelPipelineSet.contains("binary")) {
    // Set up the lowering pipeline which goes down to ELF Binary
    int optLevel = gpuOpt.getValue();
    if (optLevel < 0 || optLevel > 3) {
      llvm::errs() << "Invalid GPU optimization level: " << optLevel << "\n";
      return failure();
    }

    rock::BackendOptions opts;
    opts.triple = devName.getTriple().str();
    opts.chip = devName.getChip().str();
    opts.features = devName.getFeaturesForBackend();
    opts.optLevel = optLevel;
    rock::buildBackendPipeline(pm, opts);
  }

  return pm.run(kmod);
}

static LogicalResult runMLIRPasses(ModuleOp &module,
                                   mlir::PassPipelineCLParser &passPipeline) {

  llvm::SmallVector<std::string, 4> targetList;
  StringRef targetsStr = targets.getValue();
  SmallVector<StringRef, 4> tokens;
  targetsStr.split(tokens, ',');
  for (auto str : tokens) {
    auto target = str.trim();
    if (!target.empty()) {
      RocmDeviceName targetDevName;
      if (failed(targetDevName.parse(target))) {
        llvm::errs() << "Invalid target " << target << " in --targets\n";
        return failure();
      }
      SmallString<64> canonicalTarget;
      targetDevName.getFullName(canonicalTarget);
      targetList.push_back(canonicalTarget.str().str());
    }
  }

  // Canonicalize arch name
  if (!arch.empty()) {
    RocmDeviceName devName;
    if (failed(devName.parse(arch))) {
      llvm::errs() << "Unknown value for --arch " << arch << "\n";
      return failure();
    }
    SmallString<64> canonicalArch;
    devName.getFullName(canonicalArch);
    arch = canonicalArch.str().str();
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
    if (kernelPipelineSet.contains("applicability") &&
        kernelPipelineSet.size() != 1) {
      llvm::errs() << "The `applicability` pipeline cannot be combined with "
                      "any other pipeline options.\n";
      return failure();
    }
  }

  llvm::SmallDenseSet<StringRef> hostPipelineOptions{"partition", "highlevel",
                                                     "mhal", "runner"};
  llvm::SmallDenseSet<StringRef> hostPipelineSet;
  if (failed(parsePipeline(hostPipeline.getValue(), hostPipelineSet,
                           hostPipelineOptions, hostPipelineOptions))) {
    return failure();
  }

  // Run partitioning pipeline.
  if (hostPipelineSet.contains("partition")) {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    mhal::GraphOptions opts;
    opts.targets = targetList;
    mhal::buildGraphPipeline(pm, opts);

    if (failed(pm.run(module))) {
      return failure();
    }
  }

  bool isHighLevel = hostPipelineSet.contains("highlevel");

  StringRef onlyArch;
  if (targetList.size())
    onlyArch = targetList.front();
  else
    onlyArch = arch;

  StringRef targetArch = onlyArch;
  bool hasKernels = false;
  // Find kernel module, defaults to top module
  if (kernelPipelineSet.size() || isHighLevel) {
    LogicalResult kernelResult = success();
    // If sub-modules exists with kernel.chip specified and in set
    // of targetChips, run KernelPipeline
    module->walk([&](ModuleOp kernelModule) {
      auto archAttr = kernelModule->getAttrOfType<StringAttr>("mhal.arch");
      hasKernels |= (bool)archAttr;
      if (archAttr && llvm::find(targetList, archAttr.getValue())) {
        kernelResult = runKernelPipeline(archAttr.getValue(), kernelModule,
                                         isHighLevel, kernelPipelineSet);
        targetArch = archAttr.getValue();
      }
    });
    if (!hasKernels) {
      // If no sub-modules, run KernelPipeline on top-level module
      if (onlyArch.empty()) {
        if (module->hasAttrOfType<StringAttr>("mhal.arch")) {
          onlyArch = module->getAttrOfType<StringAttr>("mhal.arch").getValue();
        }
      }
      targetArch = onlyArch;
      kernelResult =
          runKernelPipeline(onlyArch, module, isHighLevel, kernelPipelineSet);
    }
    if (failed(kernelResult))
      return kernelResult;
  } else {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
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
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    rock::BufferizeOptions opts;
    opts.disableRock = true;
    rock::buildBufferizePipeline(pm, opts);

    if (failed(pm.run(module))) {
      return failure();
    }
  }

  // Run MHAL generation on the top module
  if (hostPipelineSet.contains("mhal")) {
    PassManager pm(module.getContext());
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    mhal::buildPackagePipeline(pm);
    if (failed(pm.run(module))) {
      return failure();
    }
  }

  // Run host code lowering that makes the result of this operation accetable
  // to mlir-cpu-runner. Explicitly aborts in the case of multiple mhal
  // targets to prevent confusing behavior.
  if (hostPipelineSet.contains("runner")) {
    if (targetList.size() > 1) {
      llvm::errs() << "Expected at most one mhal target when compling from "
                      "within rocmlir-driver\n";
      return failure();
    }
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    mhal::RunnerOptions runnerOptions;
    runnerOptions.barePtrMemrefs = barePointers.getValue();
    runnerOptions.enableCoroutines = hostAsyncCoroutines.getValue();
    SmallVector<std::string, 4> targetTypes{"GPU"};
    SmallVector<std::string, 4> targetArchs;
    targetArchs.push_back(targetArch.str());
    runnerOptions.targetTypes = targetTypes;
    runnerOptions.targetArchs = targetArchs;
    mhal::buildRunnerPipeline(pm, runnerOptions);
    pm.addPass(LLVM::createSoftwareBF16Pass());
    if (failed(pm.run(module)))
      return failure();
  }

  // Clean up
  module->walk(
      [&](LLVM::LLVMFuncOp func) { func->removeAttr("xmodel.targets"); });
  return success();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  MLIRContext context(registry);
  context.loadDialect<mhal::MHALDialect, rock::RockDialect, func::FuncDialect,
                      scf::SCFDialect, affine::AffineDialect,
                      memref::MemRefDialect, math::MathDialect,
                      arith::ArithDialect, gpu::GPUDialect,
                      bufferization::BufferizationDialect, mhal::MHALDialect>();
  mlir::registerRocMLIRPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::registerMLIRContextCLOptions();
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
