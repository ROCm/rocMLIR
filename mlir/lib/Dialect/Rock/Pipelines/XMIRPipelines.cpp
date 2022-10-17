//===- XMIRPipeline.cpp - Create XMIR runtime pipeline --------------------===//
//
// Copyright 2021 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This interface adds the Rock compilation pipeline for various flows but
// keeping a unified ordering of the pipeline.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Pipelines/XMIRPipelines.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

//===- Consolidate the XMIR Pipelines here ---------------------===//

void xmir::buildModelPipeline(OpPassManager &pm,
                              const xmir::ModelOptions &options) {
  pm.addPass(rock::createRockApplyImplPass());
}

// Runner takes an Affine/SCF program with async retargetable launchs
// and lowers to host LLVM runtime program. JitRunner then calls ORC
// to generate X86 binary and runs it.
void xmir::buildRunnerPipeline(OpPassManager &pm,
                               const xmir::RunnerOptions &options) {
  auto &funcPm1 = pm.nest<func::FuncOp>();
  funcPm1.addPass(createConvertLinalgToAffineLoopsPass());
  funcPm1.addPass(createLowerAffinePass());
  funcPm1.addPass(createConvertSCFToCFPass());

  // Target async.launch to cpu.coro or gpu.launch_func
  pm.addPass(createConvertAsyncToGPUPass());
  pm.addPass(createAsyncParallelForPass());
  // Make gpu ops async if they didn't come from the async world
  pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());

  auto &funcPm2 = pm.nest<func::FuncOp>();
  funcPm2.addPass(mlir::arith::createArithmeticExpandOpsPass());
  funcPm2.addPass(arith::createConvertArithmeticToLLVMPass());
  funcPm2.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createMemRefToLLVMPass());

  AsyncToAsyncRuntimeOptions a2arOpts;
  a2arOpts.enableCoroutines = options.enableCoroutines;
  pm.addPass(createAsyncToAsyncRuntime(a2arOpts));

  auto &funcPm3 = pm.nest<func::FuncOp>();
  funcPm3.addPass(createAsyncRuntimeRefCountingPass());
  funcPm3.addPass(createAsyncRuntimeRefCountingOptPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createConvertAsyncToLLVMPass());

  pm.addPass(createGpuToLLVMConversionPass(
      /*kernelBarePtrCallConv=*/options.barePtrMemrefs));

  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  pm.addPass(LLVM::createSoftwareBF16Pass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void xmir::registerPipelines() {
  PassPipelineRegistration<xmir::ModelOptions>(
      "xmir-model-pipeline",
      "The XMIR model pipeline collects implementations and applies them"
      " to the appropriate the default func.",
      buildModelPipeline);
  PassPipelineRegistration<xmir::RunnerOptions>(
      "xmir-runner-pipeline",
      "The XMIR runner pipeline selects target implementations and generates"
      " host code to run them.",
      buildRunnerPipeline);
}
