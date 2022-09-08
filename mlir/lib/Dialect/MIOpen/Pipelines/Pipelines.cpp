//===- Pipelines.cpp - Create MIOpen compilation pipelines ---------------===//
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

#include "mlir/Dialect/MIOpen/Pipelines.h"
#include "mlir/Dialect/MIOpen/XMIRPipelines.h"

#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

//===- Consolidate the MIOpen Pipelines here ---------------------===//

void miopen::buildPartitionPipeline(OpPassManager &pm,
                                    const miopen::PartitionOptions &options) {
  // TOSA partitioning pass
  // make 'kernel' funcs with tosa dataflow
  /* miopen-opt --tosa-partition
   */
  pm.addPass(tosa::createTosaPartitionPass());

  if (options.cloneToMIOpenModule) {
    // clone 'kernel' funcs into __miopen module
    /* miopen-opt --miopen-clone-kernels
     */
    pm.addPass(miopen::createMIOpenCloneKernelsPass(options.targetChips));
  }
}

void miopen::buildBufferizePipeline(OpPassManager &pm,
                                    const miopen::BufferizeOptions &options) {
  bool noMIOpen = options.disableMIOpen;

  // TOSA conversion to miopen and/or linalg with async.launch's
  if (!noMIOpen) {
    // convert tosa.conv2d/matmul to miopen.conv2d
    /* miopen-opt --tosa-to-miopen
     */
    pm.addNestedPass<func::FuncOp>(tosa::createTosaToMIOpenPass());
  }
  // use tosa conversion pipeline
  // (see mlir/lib/Conversion/TosaToLinalg/TosaToLinalgPass.cpp)
  mlir::tosa::addTosaToLinalgPasses(pm);

  // linalg tensor opts
  /* miopen-opt --linalg-fuse-elementwise-ops
   */
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());

  // make async kernel launch's
  /* miopen-opt --miopen-async-launch
   */
  pm.addNestedPass<func::FuncOp>(createMIOpenAsyncLaunchPass());

  // for tosa control flow
  /* miopen-opt --tosa-to-scf --tosa-to-arith
   */
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToSCF());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToArith());

  // bufferization
  /* miopen-opt --canonicalize --cse -convert-tensor-to-linalg
        --one-shot-bufferize="allow-return-allocs=1
     create-deallocs=0 bufferize-function-boundaries=1
     unknown-type-conversion=identity-layout-map
     function-boundary-type-conversion=identity-layout-map"
        --buffer-results-to-out-params
   */
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  pm.addPass(createConvertTensorToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createLinalgInitTensorToAllocTensorPass());

  bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.allowReturnAllocs = true;
  bufOpts.createDeallocs = noMIOpen;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.unknownTypeConverterFn =
    [](Value value, unsigned memorySpace,
                                      const bufferization::BufferizationOptions &options) {
    return bufferization::getMemRefTypeWithStaticIdentityLayout(
        value.getType().cast<TensorType>(), memorySpace);
  };
      //bufferization::BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  bufOpts.functionBoundaryTypeConversion =
      bufferization::BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(createOneShotBufferizePass(bufOpts));

  pm.addPass(bufferization::createBufferResultsToOutParamsPass());

  // copy opt (cleanup from high-level transforms)
  /* miopen-opt --miopen-copy-opt
   */
  pm.addNestedPass<func::FuncOp>(miopen::createMIOpenCopyOptPass());
}

void miopen::buildKernelPipeline(OpPassManager &pm,
                                 const miopen::KernelOptions &options) {
  // Pre kernel lowering fixups for patterns that aren't amenable to lawer
  // fusion
  /* miopen-opt --miopen-fold-transpose */
  if (options.enableFusion) {
    pm.addNestedPass<func::FuncOp>(miopen::createMIOpenFoldTransposePass());
  }
  // miopen lowering (tuning, global to block)
  /* miopen-opt --miopen-affix-params  --miopen-conv-to-gemm
   * --miopen-gemm-to-gridwise --miopen-gridwise-gemm-to-blockwise
   */
  pm.addPass(
      miopen::createAffixTuningParametersPass(0, 0, options.tuningFallback));
  pm.addNestedPass<func::FuncOp>(miopen::createMIOpenConvToGemmPass());
  pm.addNestedPass<func::FuncOp>(miopen::createMIOpenGemmToGridwisePass());
  pm.addNestedPass<func::FuncOp>(miopen::createMIOpenGridwiseGemmToBlockwisePass());

  if (!options.enableApplicability) {
    if (options.enableFusion) {
      // align linalg tiling
      /* miopen-opt --miopen-linalg-align
       * --convert-linalg-to-affine-loops
       */
      pm.addPass(miopen::createMIOpenLinalgAlignPass());
      pm.addPass(createConvertLinalgToAffineLoopsPass());
    }

    // miopen lowering (block to thread)
    /* miopen-opt --miopen-lowering-blockwise-gemm-to-threadwise
         --miopen-threadwise-gemm-lowering
         --miopen-sugar-to-loops --miopen-clean-math --miopen-loops-to-cf
         --convert-miopen-to-gpu
     */
    pm.addPass(miopen::createMIOpenBlockwiseGemmToThreadwisePass());
    pm.addPass(miopen::createMIOpenThreadwiseGemmLoweringPass());
    pm.addPass(miopen::createMIOpenSugarToLoopsPass());
    pm.addPass(miopen::createMIOpenCleanMathPass());
    pm.addPass(miopen::createMIOpenLoopsToCfPass());
    pm.addPass(createLowerMIOpenOpsToGPUPass());

    // lowering linalg to cf
    /* miopen-opt --convert-linalg-to-affine-loops --lower-affine
     * --convert-scf-to-cf
     */
    pm.addPass(createConvertLinalgToAffineLoopsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createConvertSCFToCFPass());
  }
}

void miopen::buildBackendPipeline(OpPassManager &pm,
                                  const miopen::BackendOptions &options) {
  // lowering ROCDL (LLVM) to binary
  /* miopen-opt --strip-debuginfo
   *   "--convert-gpu-to-rocdl=chipset=$chip index-bitwidth=32"
   *   "--gpu-to-hsaco=triple=$triple chip=$chip features=$features opt-level=3"
   */
  pm.addPass(createStripDebugInfoPass());
  pm.addPass(createLowerGpuOpsToROCDLOpsPass(
      options.chip, options.indexBitwidth, /*useBarePtrCallConv=*/true));
  pm.addPass(createGpuSerializeToHsacoPass(options.triple, options.chip,
                                           options.features, options.optLevel));
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void miopen::registerPipelines() {
  PassPipelineRegistration<miopen::PartitionOptions>(
      "miopen-partition-pipeline",
      " representations and algorithms for sparse tensors.",
      buildPartitionPipeline);
  PassPipelineRegistration<miopen::BufferizeOptions>(
      "miopen-bufferize-pipeline",
      " representations and algorithms for sparse tensors.",
      buildBufferizePipeline);
  PassPipelineRegistration<miopen::KernelOptions>(
      "miopen-kernel-pipeline",
      " representations and algorithms for sparse tensors.",
      buildKernelPipeline);
  PassPipelineRegistration<miopen::BackendOptions>(
      "miopen-backend-pipeline",
      " representations and algorithms for sparse tensors.",
      buildBackendPipeline);

  xmir::registerPipelines();
}
