//===- Pipeline.cpp - Create MIOpen compilation pipeline ---------------===//
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
// This interface adds the MIOpen compilation pipeline for various flows but
// keeping a unified ordering of the pipeline.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/XMIRPipelines.h"
#include "mlir/Dialect/MIOpen/Pipelines.h"

#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

//===- Consolidate the XMIR Pipelines here ---------------------===//

void xmir::buildModelPipeline(OpPassManager &pm,
                              const xmir::ModelOptions &options) {
  pm.addPass(miopen::createMIOpenApplyImplPass());
}

void xmir::buildRunnerPipeline(OpPassManager &pm,
                               const xmir::RunnerOptions &options) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToCFPass());
  if (!options.cpuOnly) {
    pm.addPass(createConvertAsyncToGPUPass());
    pm.addPass(createSymbolDCEPass());
  }
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createGpuToLLVMConversionPass());
  pm.addPass(createAsyncToAsyncRuntimePass());
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
