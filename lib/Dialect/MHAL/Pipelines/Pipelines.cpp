//===- Pipelines.cpp - Create XModel compilation pipelines ---------------===//
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
// This interface adds the XModel compilation pipeline for various flows but
// keeping a unified ordering of the pipeline.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XModel/Pipelines/Pipelines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/XModelToGPU/XModelToGPU.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/XModel/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

//===- Consolidate the XModel Pipelines here ---------------------===//

void xmodel::buildGraphPipeline(OpPassManager &pm,
                                const xmodel::GraphOptions &options) {
  // TOSA partitioning pass
  // make 'kernel' funcs with tosa dataflow
  /* mlir-opt --tosa-make-broadcastable
         --tosa-partition
   */
  pm.addNestedPass<func::FuncOp>(tosa::createTosaMakeBroadcastablePass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  SmallVector<std::string, 4> anchors{"tosa.conv2d", "tosa.depthwise_conv2d",
                                      "tosa.matmul"};
  tosa::TosaPartitionOptions opts;
  opts.anchorOps = anchors;
  opts.trailingOnly = true;
  pm.addPass(tosa::createTosaPartition(opts));

  // make async kernel launch's
  /* mlir-opt --xmodel-async-graph
   */
  pm.addNestedPass<func::FuncOp>(createXModelAsyncGraphPass());

  // clone 'kernel' funcs into __kernel_<arch> module
  /* mlir-opt --xmodel-target-kernels
   */
  pm.addPass(xmodel::createXModelTargetKernelsPass(
      xmodel::XModelTargetKernelsPassOptions{options.targets}));
}

/// Collect target objects and package with host partitioned kernels
void xmodel::buildPackagePipeline(OpPassManager &pm,
                                  const xmodel::PackageOptions &options) {
  /* mlir-opt --xmodel-package-targets
   */
  pm.addPass(xmodel::createXModelPackageTargetsPass());
}

// Runner takes an Affine/SCF program with async retargetable launchs
// and lowers to host LLVM runtime program. JitRunner then calls ORC
// to generate X86 binary and runs it.
void xmodel::buildRunnerPipeline(OpPassManager &pm,
                                 const xmodel::RunnerOptions &options) {
  // Select targets
  XModelSelectTargetsPassOptions targetOpts;
  targetOpts.targetTypes = options.targetTypes;
  targetOpts.targetArchs = options.targetArchs;
  pm.addNestedPass<func::FuncOp>(createXModelSelectTargetsPass(targetOpts));

  auto &funcPm1 = pm.nest<func::FuncOp>();
  funcPm1.addPass(createConvertLinalgToAffineLoopsPass());
  funcPm1.addPass(createLowerAffinePass());
  funcPm1.addPass(memref::createExpandStridedMetadataPass());

  funcPm1.addPass(createConvertSCFToCFPass());

  // Target async.launch to cpu.coro or gpu.launch_func
  pm.addPass(createConvertXModelToGPUPass());
  pm.addPass(createAsyncParallelForPass());
  // Make gpu ops async if they didn't come from the async world
  pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());

  auto &funcPm2 = pm.nest<func::FuncOp>();
  funcPm2.addPass(arith::createArithExpandOpsPass());
  funcPm2.addPass(createArithToLLVMConversionPass());
  funcPm2.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createMemRefToLLVMConversionPass());

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

void xmodel::registerPipelines() {
  PassPipelineRegistration<xmodel::GraphOptions>(
      "xmodel-graph-pipeline",
      "The XModel graph pipeline optimizes and partitions TOSA dataflow "
      "graphs.",
      buildGraphPipeline);
  PassPipelineRegistration<xmodel::PackageOptions>(
      "xmodel-package-pipeline",
      "The XModel package pipeline collects implementations and applies them"
      " to the appropriate the default func.",
      buildPackagePipeline);
  PassPipelineRegistration<xmodel::RunnerOptions>(
      "xmodel-runner-pipeline",
      "The MHAL runner pipeline selects target implementations and generates"
      " host code to run them.",
      buildRunnerPipeline);
}
