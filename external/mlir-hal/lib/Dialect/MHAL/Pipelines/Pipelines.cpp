//===- Pipelines.cpp - Create MHAL compilation pipelines ---------------===//
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
// This interface adds the MHAL compilation pipeline for various flows but
// keeping a unified ordering of the pipeline.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MHAL/Pipelines/Pipelines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/MHALToCPU/MHALToCPU.h"
#include "mlir/Conversion/MHALToGPU/MHALToGPU.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

//===- Consolidate the MHAL Pipelines here ---------------------===//

void mhal::buildGraphPipeline(OpPassManager &pm,
                              const mhal::GraphOptions &options) {
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

  // make mhal kernel launch's
  /* mlir-opt --mhal-infer-graph
   */
  pm.addNestedPass<func::FuncOp>(createMHALInferGraphPass());

  // clone 'kernel' funcs into __kernel_<arch> module
  /* mlir-opt --mhal-target-kernels
   */
  pm.addPass(mhal::createMHALTargetKernelsPass(
      mhal::MHALTargetKernelsPassOptions{options.targets}));
}

/// Collect target objects and package with host partitioned kernels
void mhal::buildPackagePipeline(OpPassManager &pm,
                                const mhal::PackageOptions &options) {
  /* mlir-opt --mhal-package-targets
   */
  pm.addPass(mhal::createMHALPackageTargetsPass());
}

// Runner takes an Affine/SCF program with mhal retargetable launchs
// and lowers to host LLVM runtime program. JitRunner then calls ORC
// to generate X86 binary and runs it.
void mhal::buildRunnerPipeline(OpPassManager &pm,
                               const mhal::RunnerOptions &options) {
  // Select targets
  MHALSelectTargetsPassOptions targetOpts;
  targetOpts.targetTypes = options.targetTypes;
  targetOpts.targetArchs = options.targetArchs;
  pm.addNestedPass<func::FuncOp>(createMHALSelectTargetsPass(targetOpts));

  auto &funcPm1 = pm.nest<func::FuncOp>();
  funcPm1.addPass(createConvertLinalgToAffineLoopsPass());
  funcPm1.addPass(createLowerAffinePass());
  funcPm1.addPass(memref::createExpandStridedMetadataPass());

  funcPm1.addPass(createConvertSCFToCFPass());

  // Make gpu ops async if they didn't come from the async world
  pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());
  // Target mhal.launch to gpu.launch_func
  pm.addPass(createConvertMHALToGPUPass());
  // Target remaining mhal.launch to cpu.call
  pm.addPass(createConvertMHALToCPUPass());
  pm.addPass(createAsyncParallelForPass());

  auto &funcPm2 = pm.nest<func::FuncOp>();
  funcPm2.addPass(arith::createArithExpandOpsPass());
  funcPm2.addPass(createArithToLLVMConversionPass());
  funcPm2.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  pm.addPass(createAsyncToAsyncRuntimePass());

  auto &funcPm3 = pm.nest<func::FuncOp>();
  funcPm3.addPass(createAsyncRuntimeRefCountingPass());
  funcPm3.addPass(createAsyncRuntimeRefCountingOptPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createConvertAsyncToLLVMPass());

  GpuToLLVMConversionPassOptions opts;
  opts.kernelBarePtrCallConv = options.barePtrMemrefs;
  pm.addPass(createGpuToLLVMConversionPass(opts));

  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void mhal::registerPipelines() {
  PassPipelineRegistration<mhal::GraphOptions>(
      "mhal-graph-pipeline",
      "The MHAL graph pipeline optimizes and partitions TOSA dataflow "
      "graphs.",
      buildGraphPipeline);
  PassPipelineRegistration<mhal::PackageOptions>(
      "mhal-package-pipeline",
      "The MHAL package pipeline collects implementations and applies them"
      " to the appropriate the default func.",
      buildPackagePipeline);
  PassPipelineRegistration<mhal::RunnerOptions>(
      "mhal-runner-pipeline",
      "The MHAL runner pipeline selects target implementations and generates"
      " host code to run them.",
      buildRunnerPipeline);
}
