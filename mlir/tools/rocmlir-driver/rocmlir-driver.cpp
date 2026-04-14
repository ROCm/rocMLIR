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

#include "mlir/Conversion/DxgmlToMIGraphX/DxgmlToMIGraphX.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Pipelines/Pipelines.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Pipeline/Pipeline.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
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

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<std::string> kernelPipeline(
    "kernel-pipeline", cl::desc("rocmlir-driver kernel pipeline list"),
    cl::value_desc("comma separated list of rock pipelines: "
           "applicability,dxgml,migraphx,highlevel,gpu,rocdl,binary or full"),
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

static LogicalResult runHostHighLevelPipeline(ModuleOp m) {
  // Setup pass manager
  PassManager pm(m->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  // Add verification passes
  pm.enableVerifier(verifyPasses);

  // Set disableRock to true since we are just running on the host
  rock::BufferizeOptions opts;
  opts.disableRock = true;

  // Run the bufferize pipeline
  rock::buildBufferizePipeline(pm, opts);

  if (dumpPipelines) {
    llvm::errs() << "Host pipeline:\n";
    pm.printAsTextualPipeline(llvm::errs());
    llvm::errs() << "\n";
  }

  return pm.run(m);
}

/// Outline each tosa.conv2d / tosa.matmul in a func.func into its own
/// function so that Rock tiling (which requires exactly one FusionRoot per
/// function) can process them independently.
///
/// For each compute op found:
///  1. Collect SSA operands that are defined outside the op (function args or
///     results of earlier ops).  tosa.const operands are *cloned* into the new
///     function body rather than passed as arguments.
///  2. Create a new private func.func with the non-const operands as args and
///     the compute result as the return type.
///  3. Clone the needed tosa.const defs + the compute op into the new body.
///  4. Replace the compute op in the original function with a func.call.
///
/// This mirrors how the MIGraphX C++ runtime pre-outlines kernels: each GPU
/// kernel lives in its own function before entering the Rock pipeline.
static void outlineTosaComputeOps(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  OpBuilder moduleBuilder(ctx);
  SymbolTable symbolTable(module);

  // Process each top-level func.func.
  SmallVector<func::FuncOp> funcs;
  for (auto &op : module.getBody()->getOperations())
    if (auto f = dyn_cast<func::FuncOp>(op))
      funcs.push_back(f);

  for (func::FuncOp func : funcs) {
    // Collect all tosa.conv2d and tosa.matmul ops (in order).
    SmallVector<Operation *> computeOps;
    func.walk([&](Operation *op) {
      if (isa<tosa::Conv2DOp, tosa::MatMulOp>(op))
        computeOps.push_back(op);
    });

    if (computeOps.empty())
      continue; // No compute op to outline.

    // Copy relevant attributes from the parent function to propagate to new
    // outlined kernels (kernel="mixr", arch, etc.).
    SmallVector<NamedAttribute> inheritAttrs;
    for (NamedAttribute na : func->getAttrs()) {
      StringRef name = na.getName().getValue();
      if (name == "kernel" || name == "arch" || name == "mhal.arch")
        inheritAttrs.push_back(na);
    }

    unsigned kernelIdx = 0;
    for (Operation *computeOp : computeOps) {
      Location loc = computeOp->getLoc();

      // -------------------------------------------------------------------
      // Step 1: partition the operands into "constants" (tosa.const defined
      // in the same function — clone them) vs "dynamic" (pass as args).
      // -------------------------------------------------------------------
      SmallVector<mlir::Value> constOperands;   // will be cloned inside new func
      SmallVector<mlir::Value> dynamicOperands; // will be passed as arguments

      for (mlir::Value operand : computeOp->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        // tosa.const is side-effect-free; clone it into the new function.
        if (defOp && isa<tosa::ConstOp>(defOp))
          constOperands.push_back(operand);
        else
          dynamicOperands.push_back(operand);
      }

      // -------------------------------------------------------------------
      // Step 2: Build the new function type.
      // -------------------------------------------------------------------
      SmallVector<mlir::Type> argTypes;
      for (mlir::Value v : dynamicOperands)
        argTypes.push_back(v.getType());
      SmallVector<mlir::Type> resultTypes = {
          computeOp->getResult(0).getType()};

      auto newFuncType =
          mlir::FunctionType::get(ctx, argTypes, resultTypes);

      // Generate a unique name: <parent>_conv_<idx> or <parent>_gemm_<idx>.
      std::string suffix =
          isa<tosa::Conv2DOp>(computeOp) ? "_conv_" : "_gemm_";
      std::string newName = func.getName().str() + suffix +
                            std::to_string(kernelIdx++);
      // Guarantee uniqueness.
      while (symbolTable.lookup(newName))
        newName += "_";

      // -------------------------------------------------------------------
      // Step 3: Create the new function.
      // -------------------------------------------------------------------
      moduleBuilder.setInsertionPoint(func); // insert before current func
      auto newFunc = func::FuncOp::create(moduleBuilder, loc, newName, newFuncType);
      newFunc.setPrivate();
      for (NamedAttribute na : inheritAttrs)
        newFunc->setAttr(na.getName(), na.getValue());

      Block *newBody = newFunc.addEntryBlock();
      OpBuilder bodyBuilder(newBody, newBody->begin());

      // Map original const values to their clones inside the new function.
      IRMapping constMapping;
      for (mlir::Value cv : constOperands) {
        Operation *cloned = bodyBuilder.clone(*cv.getDefiningOp(), constMapping);
        constMapping.map(cv, cloned->getResult(0));
      }

      // Map dynamic operands to block arguments.
      IRMapping dynMapping = constMapping;
      for (auto [origVal, arg] :
           llvm::zip(dynamicOperands, newBody->getArguments()))
        dynMapping.map((mlir::Value)origVal, (mlir::Value)arg);

      // Clone the compute op with the remapped operands.
      Operation *clonedCompute = bodyBuilder.clone(*computeOp, dynMapping);
      func::ReturnOp::create(bodyBuilder, loc, clonedCompute->getResult(0));

      // -------------------------------------------------------------------
      // Step 4: Replace the original compute op with a func.call.
      // -------------------------------------------------------------------
      OpBuilder callBuilder(computeOp);
      auto callOp = func::CallOp::create(
          callBuilder, loc, newFunc, dynamicOperands);
      computeOp->getResult(0).replaceAllUsesWith(callOp.getResult(0));
      computeOp->erase();
    }

    // The parent function no longer has FusionRoot ops (they're in the new
    // functions), so it can pass through rock-affix-params without hitting the
    // Multiple-Fusion-Roots check.  Remove the kernel attribute from the
    // parent so it isn't mistakenly tiled by the kernel pipeline.
    func->removeAttr("kernel");
    func->removeAttr("arch");
  }
}

static LogicalResult
runKernelPipeline(StringRef arch, ModuleOp m,
                  llvm::SmallDenseSet<StringRef> &kernelPipelineSet) {
  PassManager pm(m->getName(), PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableVerifier(verifyPasses);
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

  if (kernelPipelineSet.contains("migraphx")) {
    migraphx::addHighLevelPipeline(pm);
  }

  if (kernelPipelineSet.contains("highlevel")) {
    rock::buildBufferizePipeline(pm);
  }

  // Set up lowering pipeline.
  if (kernelPipelineSet.contains("applicability")) {
    rock::KernelOptions opts;
    opts.applicabilityMode = mlir::rock::ApplicabilityMode::Applicability;
    rock::buildKernelPipeline(pm, opts);
  }
  if (kernelPipelineSet.contains("gpu")) {
    // Set up the default lowering pipeline which goes down to GPU dialect.
    rock::buildKernelPipeline(pm);
  }
  bool isRocdlOnly = kernelPipelineSet.contains("rocdl") &&
                     !kernelPipelineSet.contains("binary");
  if (kernelPipelineSet.contains("binary") || isRocdlOnly) {
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
    opts.compile = !isRocdlOnly;
    rock::buildBackendPipeline(pm, opts);
  }

  if (dumpPipelines) {
    llvm::errs() << "Kernel pipeline:\n";
    pm.printAsTextualPipeline(llvm::errs());
    llvm::errs() << "\n";
  }
  return pm.run(m);
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

  llvm::SmallDenseSet<StringRef> kernelPipelineOptions{
      "applicability", "dxgml", "migraphx", "highlevel", "gpu", "rocdl", "binary"};
  llvm::SmallDenseSet<StringRef> kernelFullPipeline{"gpu", "binary"};
  llvm::SmallDenseSet<StringRef> kernelPipelineSet;
  std::string kernelPipelineStr = kernelPipeline.getValue();
  if (failed(parsePipeline(kernelPipelineStr, kernelPipelineSet,
                           kernelPipelineOptions, kernelFullPipeline))) {
    return failure();
  }
  if (!kernelPipelineSet.empty()) {
    if (kernelPipelineSet.contains("applicability") &&
        kernelPipelineSet.size() != 1) {
      llvm::errs() << "The `applicability` pipeline cannot be combined with "
                      "any other pipeline options.\n";
      return failure();
    }
  }

  llvm::SmallDenseSet<StringRef> hostPipelineOptions{"dxgml", "migraphx",
                                                     "highlevel", "mhal",
                                                     "runner"};
  llvm::SmallDenseSet<StringRef> hostPipelineSet;
  std::string hostPipelineStr = hostPipeline.getValue();
  if (failed(parsePipeline(hostPipelineStr, hostPipelineSet,
                           hostPipelineOptions, hostPipelineOptions))) {
    return failure();
  }

  // Lower DXML dialect (dxgml.*) to MIGraphX dialect, then apply the
  // standard MIGraphX high-level pipeline so the same TosaToRock /
  // GPU / ROCDL / binary infrastructure handles the rest.
  // If pass-through dxgml_op ops remain after partial conversion (e.g.
  // depth_to_space, group_query_attention), skip addHighLevelPipeline since
  // those ops have no MIGraphX→TOSA lowering.
  if (hostPipelineSet.contains("dxgml")) {
    {
      PassManager pm(module->getName(), PassManager::Nesting::Implicit);
      pm.addPass(createConvertDxgmlToMIGraphXPass());
      if (failed(pm.run(module))) {
        return failure();
      }
    }
    // Check whether any dxgml_op ops remain (pass-through ops that cannot be
    // lowered through the MIGraphX pipeline).
    bool hasDxgmlOps = false;
    module->walk([&](Operation *op) {
      if (op->getDialect() &&
          op->getDialect()->getNamespace() == StringRef("dxgml_op")) {
        hasDxgmlOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!hasDxgmlOps) {
      PassManager pm(module->getName(), PassManager::Nesting::Implicit);
      migraphx::addHighLevelPipeline(pm);
      if (failed(pm.run(module))) {
        return failure();
      }
      // After MIGraphX->TOSA lowering, stamp kernel="mixr" and arch on every
      // func.func so that tosa-to-rock (which requires the kernel attribute)
      // can process them during the subsequent kernel pipeline.
      // This mirrors what migraphx-format inputs carry as function attributes.
      if (!kernelPipelineSet.empty()) {
        StringRef archVal = arch.getValue();
        module->walk([&](func::FuncOp func) {
          if (!func->hasAttr("kernel"))
            func->setAttr("kernel",
                          StringAttr::get(func->getContext(), "mixr"));
          if (!archVal.empty() && !func->hasAttr("arch"))
            func->setAttr("arch",
                          StringAttr::get(func->getContext(), archVal));
        });
        // Rock tiling (highlevel kernel pipeline) must run before gpu/binary
        // compilation to populate block_size/grid_size on kernel functions.
        // Inject "highlevel" automatically if gpu or binary is requested but
        // highlevel is absent. This handles the "full" case where parsePipeline
        // replaces the set with {gpu, binary} without including highlevel.
        if ((kernelPipelineSet.contains("gpu") ||
             kernelPipelineSet.contains("binary")) &&
            !kernelPipelineSet.contains("highlevel")) {
          kernelPipelineSet.insert("highlevel");
        }
        // Outline multi-conv/matmul functions: Rock tiling allows at most one
        // FusionRoot op per function. Functions with multiple tosa.conv2d /
        // tosa.matmul ops (from DXML models with sequential convolutions) must
        // be split so each kernel goes into its own func.func.
        outlineTosaComputeOps(module);
      }
    } else if (!kernelPipelineSet.empty() &&
               (kernelPipelineSet.contains("gpu") ||
                kernelPipelineSet.contains("binary") ||
                kernelPipelineSet.contains("highlevel"))) {
      // Models with remaining dxgml_op pass-through ops cannot be Rock-tiled
      // or compiled to GPU because those ops have no GPU lowering yet.
      llvm::errs()
          << "error: Cannot lower to GPU/binary — model contains dxgml_op "
             "pass-through ops that have no GPU lowering.\n";
      return failure();
    }
  }

  if (hostPipelineSet.contains("migraphx")) {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    migraphx::addHighLevelPipeline(pm);
    if (failed(pm.run(module))) {
      return failure();
    }
  }

  bool isHighLevel = hostPipelineSet.contains("highlevel") ||
                     kernelPipelineSet.contains("highlevel");

  StringRef onlyArch;
  if (!targetList.empty())
    onlyArch = targetList.front();
  else
    onlyArch = arch;

  StringRef targetArch = onlyArch;
  bool hasKernels = false;
  // Right now we need to update the target architecture used when we
  // are running the kernel pipeline, or if we are running the highlevel host
  // pipeline.
  bool needsTargetArchUpdate =
      !kernelPipelineSet.empty() || hostPipelineSet.contains("highlevel");
  if (needsTargetArchUpdate) {
    LogicalResult kernelResult = success();
    // If sub-modules exists with kernel.chip specified and in set
    // of targetChips, run KernelPipeline
    module->walk([&](ModuleOp kernelModule) {
      auto archAttr = kernelModule->getAttrOfType<StringAttr>("mhal.arch");
      hasKernels |= (bool)archAttr;
      if (archAttr && llvm::find(targetList, archAttr.getValue())) {
        kernelResult = runKernelPipeline(archAttr.getValue(), kernelModule,
                                         kernelPipelineSet);
        // Run host high-level pipeline if specified
        if (hostPipelineSet.contains("highlevel"))
          kernelResult = runHostHighLevelPipeline(kernelModule);

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
      // Propagate arch onto module as mhal.arch so passes like tosa-to-rock
      // can find it via getArchValue (looks for "arch" or "mhal.arch" on op
      // or parents). Without this, tosa-to-rock skips Rock tiling entirely.
      if (!onlyArch.empty() && !module->hasAttr("mhal.arch"))
        module->setAttr("mhal.arch",
                        StringAttr::get(module->getContext(), onlyArch));
      targetArch = onlyArch;
      kernelResult = runKernelPipeline(onlyArch, module, kernelPipelineSet);

      // Run host high-level pipeline if specified
      if (hostPipelineSet.contains("highlevel"))
        kernelResult = runHostHighLevelPipeline(module);
    }
    if (failed(kernelResult))
      return kernelResult;
  } else {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    pm.enableVerifier(verifyPasses);
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(module.getContext())) << msg;
      return failure();
    };

    // Use lowering pipeline specified at command line.
    if (failed(passPipeline.addToPipeline(pm, errorHandler))) {
      return failure();
    }
    if (dumpPipelines) {
      llvm::errs() << "Custom pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
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
    pm.enableVerifier(verifyPasses);
    rock::BufferizeOptions opts;
    opts.disableRock = true;
    rock::buildBufferizePipeline(pm, opts);

    if (dumpPipelines) {
      llvm::errs() << "Bufferization pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
    }
    if (failed(pm.run(module))) {
      return failure();
    }
  }

  // Run MHAL generation on the top module
  if (hostPipelineSet.contains("mhal")) {
    PassManager pm(module.getContext());
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    pm.enableVerifier(verifyPasses);
    mhal::buildPackagePipeline(pm);
    if (dumpPipelines) {
      llvm::errs() << "MHAL package pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
    }
    if (failed(pm.run(module))) {
      return failure();
    }
  }

  // Run host code lowering that makes the result of this operation accetable
  // to mlir-runner. Explicitly aborts in the case of multiple mhal
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
    pm.enableVerifier(verifyPasses);
    mhal::RunnerOptions runnerOptions;
    runnerOptions.barePtrMemrefs = barePointers.getValue();
    runnerOptions.enableCoroutines = hostAsyncCoroutines.getValue();
    SmallVector<std::string, 4> targetTypes{"GPU"};
    SmallVector<std::string, 4> targetArchs;
    targetArchs.push_back(targetArch.str());
    runnerOptions.targetTypes = targetTypes;
    runnerOptions.targetArchs = targetArchs;
    mhal::buildRunnerPipeline(pm, runnerOptions);
    if (dumpPipelines) {
      llvm::errs() << "Host runner pipeline:\n";
      pm.printAsTextualPipeline(llvm::errs());
      llvm::errs() << "\n";
    }
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
                      bufferization::BufferizationDialect,
                      dxgml::DxgmlDialect, dxgml_op::DxgmlOpDialect>();
  mlir::registerRocMLIRPasses();
  InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerMLIRCLOptions();
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
