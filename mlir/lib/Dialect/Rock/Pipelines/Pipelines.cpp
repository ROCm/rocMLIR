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
#include "mlir/Conversion/ArithToAMDGPU/ArithToAMDGPU.h"
#include "mlir/Conversion/Fp8ExtToTables/Fp8ExtToTables.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/RockToGPU/RockToGPU.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

//===- Consolidate the Rock Pipelines here ---------------------===//

void rock::buildBufferizePipeline(OpPassManager &pm,
                                  const rock::BufferizeOptions &options) {
  bool noRock = options.disableRock;

  auto &funcPm = pm.nest<func::FuncOp>();
  // TOSA conversion to rock and/or linalg with mhal.launch's
  if (!noRock) {
    // convert tosa.conv2d/matmul to rock.conv2d
    /* rocmlir-opt --tosa-to-tensor --tosa-to-rock --rock-view-to-transform
     */
    funcPm.addPass(tosa::createTosaToTensor());
    funcPm.addPass(createTosaToRockPass());
    funcPm.addPass(rock::createRockViewToTransformPass());
  }

  // use tosa conversion pipeline
  // (see mlir/lib/Conversion/TosaToLinalg/TosaToLinalgPass.cpp)
  tosa::addTosaToLinalgPasses(pm);

  // for tosa control flow
  /* rocmlir-opt --tosa-to-tensor --tosa-to-scf --tosa-to-arith
   */
  auto &funcPm2 = pm.nest<func::FuncOp>();
  funcPm2.addPass(tosa::createTosaToTensor());
  funcPm2.addPass(tosa::createTosaToSCF());
  funcPm2.addPass(tosa::createTosaToArith());

  // linalg tensor opts
  /* rocmlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims
   */
  funcPm2.addPass(createLinalgElementwiseOpFusionPass());
  funcPm2.addPass(createLinalgFoldUnitExtentDimsPass());
  funcPm2.addPass(rock::createRockViewToTransformPass());

  // bufferization
  /* rocmlir-opt --canonicalize --cse -convert-tensor-to-linalg
        --one-shot-bufferize="allow-return-allocs=1
     create-deallocs=0 bufferize-function-boundaries=1
     unknown-type-conversion=identity-layout-map
     function-boundary-type-conversion=identity-layout-map"
        --buffer-results-to-out-params
   */
  funcPm2.addPass(createCanonicalizerPass());
  funcPm2.addPass(createCSEPass());

  pm.addPass(createConvertTensorToLinalgPass());
  auto &funcPm3 = pm.nest<func::FuncOp>();
  funcPm3.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  funcPm3.addPass(createLinalgFoldUnitExtentDimsPass());

  bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.allowReturnAllocs = true;
  bufOpts.createDeallocs = noRock;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  bufOpts.unknownTypeConverterFn =
      [](Value value, Attribute memorySpace,
         const bufferization::BufferizationOptions &options) {
        return bufferization::getMemRefTypeWithStaticIdentityLayout(
            value.getType().cast<TensorType>(), memorySpace);
      };
  // bufferization::BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(createOneShotBufferizePass(bufOpts));

  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
}

void rock::buildKernelPipeline(OpPassManager &pm,
                               const rock::KernelOptions &options) {
  // rock lowering (tuning, global to block)
  /* rocmlir-opt --rock-affix-params --rock-conv-to-gemm
   *   --rock-gemm-to-gridwise --rock-regularize
   *   --rock-gridwise-gemm-to-blockwise
   */
  auto &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(rock::createRockAffixTuningParametersPass(
      rock::RockAffixTuningParametersPassOptions{0, 0,
                                                 options.tuningFallback}));
  funcPm.addPass(rock::createRockConvToGemmPass());
  funcPm.addPass(rock::createRockGemmToGridwisePass());
  funcPm.addPass(rock::createRockRegularizePass());
  funcPm.addPass(rock::createRockGridwiseGemmToBlockwisePass());

  if (!options.enableApplicability) {
    if (options.enableFusion) {
      // align linalg tiling
      /* rocmlir-opt --rock-linalg-align --canonicalize
       * --convert-linalg-to-affine-loops
       */
      funcPm.addPass(rock::createRockLinalgAlignPass());
      funcPm.addPass(createCanonicalizerPass());
      funcPm.addPass(createConvertLinalgToAffineLoopsPass());
    }
    // rock lowering for reductions
    /* rocmlir-opt --rock-lower-reduce
     */
    funcPm.addPass(rock::createRockLowerReducePass());

    // rock lowering (block to thread)
    /* rocmlir-opt --rock-lowering-blockwise-gemm-to-threadwise
     *   --canonicalize --rock-threadwise-gemm-lowering
     *   --rock-sugar-to-loops --rock-clean-math --rock-buffer-load-merge
     *   --rock-transform-to-memref --rock-loops-to-cf
     *   --convert-rock-to-gpu
     */
    funcPm.addPass(rock::createRockBlockwiseGemmToThreadwisePass());
    funcPm.addPass(createCanonicalizerPass());
    funcPm.addPass(rock::createRockThreadwiseGemmLoweringPass());
    funcPm.addPass(rock::createRockSugarToLoopsPass());
    funcPm.addPass(rock::createRockCleanMathPass());
    funcPm.addPass(rock::createRockBufferLoadMergePass());
    funcPm.addPass(rock::createRockTransformToMemrefPass());
    funcPm.addPass(rock::createRockLoopsToCfPass());
    pm.addPass(createConvertRockToGPUPass());
  }
}

void rock::buildBackendPipeline(OpPassManager &pm,
                                const rock::BackendOptions &options) {
  // lowering ROCDL (LLVM) to binary.
  // Leave off --convert-arith-to-amdgpu if not targetting gfx94x+.
  /* rocmlir-opt --strip-debuginfo
   *   --convert-arith-to-amdgpu
   *   --fp8-ext-to-tables
   *   "--amdgpu-emulate-atomics=chipset=$chip"
   *   --arith-emulate-unsupported-floats="source-types=bf16 target-type=f32"
   *   "--convert-gpu-to-rocdl=chipset=$chip index-bitwidth=32"
   *   "--gpu-to-hsaco=triple=$triple chip=$chip features=$features opt-level=3"
   */
  pm.addPass(createStripDebugInfoPass());
  AmdArchInfo archInfo = lookupArchInfo(options.chip);
  if (archInfo.hasFp8ConversionInstrs)
    pm.addNestedPass<gpu::GPUModuleOp>(createArithToAMDGPUConversionPass());
  pm.addPass(createFp8ExtToTablesPass());
  auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
  gpuPm.addPass(amdgpu::createAmdgpuEmulateAtomicsPass({options.chip}));
  arith::ArithEmulateUnsupportedFloatsOptions floatEmuOpts;
  SmallVector<std::string, 1> unsupportedFloats = {"bf16"};
  floatEmuOpts.sourceTypeStrs = unsupportedFloats;
  floatEmuOpts.targetTypeStr = "f32";
  gpuPm.addPass(arith::createArithEmulateUnsupportedFloats(floatEmuOpts));
  gpuPm.addPass(memref::createExpandStridedMetadataPass());
  // We need to lower affine again, because the expand strided metadata pass
  // adds back affine.apply for memref.subview
  gpuPm.addPass(createLowerAffinePass());
  gpuPm.addPass(createLowerGpuOpsToROCDLOpsPass(
      options.chip, /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
      /*useBarePtrCallConv=*/true, gpu::amd::Runtime::Unknown));
  gpuPm.addPass(rock::createRockPrepareLLVMPass());
  if (options.compile)
    gpuPm.addPass(createGpuSerializeToHsacoPass(
        options.triple, options.chip, options.features, options.optLevel));
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
