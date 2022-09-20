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

void xmir::buildRunnerPipeline(OpPassManager &pm,
                               const xmir::RunnerOptions &options) {
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToCFPass());
  if (options.cpuOnly) {

  } else {
    pm.addPass(createConvertAsyncToGPUPass());
    pm.addPass(createSymbolDCEPass());
    pm.addPass(createGpuToLLVMConversionPass(
        /*kernelBarePtrCallConv=*/options.barePtrMemrefs));
    pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());
  }
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createConvertLinalgToLLVMPass());
  pm.addPass(createAsyncToAsyncRuntimePass());
  pm.addPass(createAsyncRuntimeRefCountingOptPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createMemRefToLLVMPass());
  pm.addPass(createConvertAsyncToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
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
