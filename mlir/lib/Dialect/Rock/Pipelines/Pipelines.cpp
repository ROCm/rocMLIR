//===- Pipelines.cpp - Create Rock compilation pipelines ---------------===//
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

#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/RockToGPU/RockToGPU.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

//===- Consolidate the Rock Pipelines here ---------------------===//

void rock::buildBufferizePipeline(OpPassManager &pm,
                                  const rock::BufferizeOptions &options) {
  bool noRock = options.disableRock;

  // TOSA conversion to rock and/or linalg with async.launch's
  if (!noRock) {
    // convert tosa.conv2d/matmul to rock.conv2d
    /* rocmlir-opt --tosa-to-rock
     */
    pm.addNestedPass<func::FuncOp>(createTosaToRockPass());
  }

  // use tosa conversion pipeline
  // (see mlir/lib/Conversion/TosaToLinalg/TosaToLinalgPass.cpp)
  tosa::addTosaToLinalgPasses(pm);

  // for tosa control flow
  /* rocmlir-opt --tosa-to-tensor --tosa-to-scf --tosa-to-arith
   */
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToTensor());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToSCF());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToArith());

  // linalg tensor opts
  /* rocmlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims
   */
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<func::FuncOp>(createLinalgFoldUnitExtentDimsPass());

  // bufferization
  /* rocmlir-opt --canonicalize --cse -convert-tensor-to-linalg
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
  pm.addNestedPass<func::FuncOp>(createLinalgFoldUnitExtentDimsPass());

  bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.allowReturnAllocs = true;
  bufOpts.createDeallocs = noRock;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      bufferization::BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  bufOpts.unknownTypeConverterFn =
      [](Value value, unsigned memorySpace,
         const bufferization::BufferizationOptions &options) {
        return bufferization::getMemRefTypeWithStaticIdentityLayout(
            value.getType().cast<TensorType>(), memorySpace);
      };
  // bufferization::BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(createOneShotBufferizePass(bufOpts));

  pm.addPass(bufferization::createBufferResultsToOutParamsPass());

  // copy opt (cleanup from high-level transforms)
  /* rocmlir-opt --rock-copy-opt
   */
  // pm.addNestedPass<func::FuncOp>(rock::createRockCopyOptPass());
}

void rock::buildKernelPipeline(OpPassManager &pm,
                               const rock::KernelOptions &options) {
  // Pre kernel lowering fixups for patterns that aren't amenable to lawer
  // fusion
  /* rocmlir-opt --rock-fold-transpose */
  // rock lowering (tuning, global to block)
  /* rocmlir-opt --rock-affix-params  --rock-conv-to-gemm
   * --rock-gemm-to-gridwise --rock-gridwise-gemm-to-blockwise
   */
  pm.addPass(rock::createRockAffixTuningParametersPass(
      rock::RockAffixTuningParametersPassOptions{0, 0,
                                                 options.tuningFallback}));
  pm.addNestedPass<func::FuncOp>(rock::createRockConvToGemmPass());
  pm.addNestedPass<func::FuncOp>(rock::createRockGemmToGridwisePass());

  //
  pm.addNestedPass<func::FuncOp>(rock::createRockRegularizePass());

  pm.addNestedPass<func::FuncOp>(rock::createRockGridwiseGemmToBlockwisePass());

  if (!options.enableApplicability) {
    if (options.enableFusion) {
      // align linalg tiling
      /* rocmlir-opt --rock-linalg-align
       * --convert-linalg-to-affine-loops
       */
      pm.addNestedPass<func::FuncOp>(rock::createRockLinalgAlignPass());
      pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
    }
    pm.addPass(rock::createRockLowerReducePass());

    // rock lowering (block to thread)
    /* rocmlir-opt --rock-lowering-blockwise-gemm-to-threadwise
         --rock-threadwise-gemm-lowering
         --rock-sugar-to-loops --rock-clean-math --rock-loops-to-cf
         --convert-rock-to-gpu
     */
    pm.addNestedPass<func::FuncOp>(
        rock::createRockBlockwiseGemmToThreadwisePass());
    pm.addNestedPass<func::FuncOp>(
        rock::createRockThreadwiseGemmLoweringPass());
    pm.addNestedPass<func::FuncOp>(rock::createRockSugarToLoopsPass());
    pm.addNestedPass<func::FuncOp>(rock::createRockCleanMathPass());
    pm.addNestedPass<func::FuncOp>(rock::createRockBufferLoadMergePass());
    pm.addNestedPass<func::FuncOp>(rock::createRockLoopsToCfPass());
    pm.addPass(createConvertRockToGPUPass());

    // lowering linalg to cf
    /* rocmlir-opt --convert-linalg-to-affine-loops --lower-affine
     * --convert-scf-to-cf
     */
    pm.addPass(createConvertLinalgToAffineLoopsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createConvertSCFToCFPass());
  }
}

void rock::buildBackendPipeline(OpPassManager &pm,
                                const rock::BackendOptions &options) {
  // lowering ROCDL (LLVM) to binary
  /* rocmlir-opt --strip-debuginfo
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

void rock::registerPipelines() {
  PassPipelineRegistration<rock::BufferizeOptions>(
      "rock-bufferize-pipeline",
      " representations and algorithms for sparse tensors.",
      buildBufferizePipeline);
  PassPipelineRegistration<rock::KernelOptions>(
      "rock-kernel-pipeline",
      " representations and algorithms for sparse tensors.",
      buildKernelPipeline);
  PassPipelineRegistration<rock::BackendOptions>(
      "rock-backend-pipeline",
      " representations and algorithms for sparse tensors.",
      buildBackendPipeline);
}
