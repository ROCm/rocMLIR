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
#include "mlir/Conversion/EmulateFp8ExtTrunc/EmulateFp8ExtTrunc.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/RockToGPU/RockToGPU.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
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
    // convert tosa.conv2d/matmul to rock.conv
    /* rocmlir-opt --tosa-to-tensor --tosa-to-rock --rock-view-to-transform
     */
    funcPm.addPass(tosa::createTosaToTensor());
    funcPm.addPass(createTosaToRockPass());
    funcPm.addPass(rock::createRockViewToTransformPass());
  }

  // use tosa conversion pipeline
  // (see mlir/lib/Conversion/TosaToLinalg/TosaToLinalgPass.cpp)
  TosaToLinalgOptions tosaToLinalgOptions;
  TosaToLinalgNamedOptions tosaToLinalgNamedOptions;
  tosa::addTosaToLinalgPasses(pm, tosaToLinalgOptions,
                              tosaToLinalgNamedOptions);

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
  funcPm2.addPass(rock::createRockFoldBroadcastPass());

  // bufferization
  /* rocmlir-opt --canonicalize -convert-tensor-to-linalg --cse
        --one-shot-bufferize="allow-return-allocs=1
     create-deallocs=0 bufferize-function-boundaries=1
     unknown-type-conversion=identity-layout-map
     function-boundary-type-conversion=identity-layout-map"
        --buffer-results-to-out-params
   */
  funcPm2.addPass(createCanonicalizerPass());
  // Note: this is a workaround for an impedance mismatch between bufferization
  // and our fusion code. Specifically, if there are two identical
  // tensor.empty's
  //, they can be CSE'd together, and then, if the bufferizer notices that the
  // allocation that that empty tensor has two independent uses (that is,
  // if op1 and op2 both have the "initial output" %x, and the values produces
  // by op1 are dead by the time op2 rolls around), it'll reuse the buffer.
  // This breaks rocMLIR's fusion code, which assumes allocations aren't reused
  // like this. So, until we move bufferization after rock.regularize (so that
  // we can do the alloc_tensor introductions ourselves), we have to do it up
  // here before CSE.
  funcPm2.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  funcPm2.addPass(createCSEPass());

  pm.addPass(createConvertTensorToLinalgPass());
  auto &funcPm3 = pm.nest<func::FuncOp>();
  funcPm3.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  funcPm3.addPass(createLinalgFoldUnitExtentDimsPass());

  bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.allowReturnAllocsFromLoops = true;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  bufOpts.unknownTypeConverterFn =
      [](Value value, Attribute memorySpace,
         const bufferization::BufferizationOptions &options) {
        return bufferization::getMemRefTypeWithStaticIdentityLayout(
            cast<TensorType>(value.getType()), memorySpace);
      };
  // bufferization::BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(createOneShotBufferizePass(bufOpts));

  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
}

void rock::buildKernelPipeline(OpPassManager &pm,
                               const rock::KernelOptions &options) {
  // rock lowering (tuning, global to block)
  /* rocmlir-opt --rock-affix-params --rock-conv-to-gemm
   *   --rock-fold-broadcast --rock-affix-params --rock-gemm-to-gridwise
   *   --rock-regularize  --rock-gridwise-gemm-to-blockwise
   */
  auto &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(rock::createRockAffixTuningParametersPass(
      rock::RockAffixTuningParametersPassOptions{options.tuningFallback}));
  funcPm.addPass(rock::createRockConvToGemmPass());
  funcPm.addPass(rock::createRockGemmToGridwisePass());
  funcPm.addPass(rock::createRockRegularizePass());
  funcPm.addPass(rock::createRockGridwiseGemmToBlockwisePass());
  funcPm.addPass(rock::createRockBlockwiseGemmToThreadwisePass());

  if (!options.enableApplicability) {
    if (options.enableFusion) {
      // align linalg tiling
      /* rocmlir-opt --rock-linalg-align --canonicalize
       * --convert-linalg-to-affine-loops
       */
      funcPm.addPass(rock::createRockLinalgAlignPass());
      funcPm.addPass(rock::createRockPipelinePass());
      funcPm.addPass(createCanonicalizerPass());
      funcPm.addPass(createConvertLinalgToAffineLoopsPass());
      funcPm.addPass(rock::createRockVectorizeFusionsPass());
    }
    funcPm.addPass(rock::createRockGemmOutputSwizzlePass());
    funcPm.addPass(rock::createRockGemmReuseDeadLDSPass());
    // rock lowering for reductions
    /* rocmlir-opt --rock-lower-reduce
     */
    funcPm.addPass(rock::createRockLowerReducePass());

    // rock lowering (block to thread)
    /* rocmlir-opt --rock-lowering-blockwise-gemm-to-threadwise
     *   --canonicalize --rock-threadwise-gemm-lowering
     *   --rock-analyze-memory-use --rock-sugar-to-loops --rock-clean-math
     *   --math-legalize-to-f32 --rock-buffer-load-merge
     *   --rock-transform-to-memref --rock-loops-to-cf --convert-rock-to-gpu
     */
    funcPm.addPass(rock::createRockThreadwiseGemmLoweringPass());
    funcPm.addPass(rock::createRockAnalyzeMemoryUsePass());
    funcPm.addPass(rock::createRockSugarToLoopsPass());
    funcPm.addPass(rock::createRockCleanMathPass());
    funcPm.addPass(math::createMathLegalizeToF32());
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
   *   --emulate-fp8-ext-trunc
   *   "--amdgpu-emulate-atomics=chipset=$chip"
   *   --arith-emulate-unsupported-floats="source-types=bf16 target-type=f32"
   *   "--convert-gpu-to-rocdl=chipset=$chip index-bitwidth=32"
   *   "--gpu-to-hsaco=triple=$triple chip=$chip features=$features opt-level=3"
   */
  pm.addPass(createStripDebugInfoPass());
  AmdArchInfo archInfo = lookupArchInfo(options.chip);
  auto &gpuPm = pm.nest<gpu::GPUModuleOp>();
  gpuPm.addPass(amdgpu::createAmdgpuEmulateAtomicsPass({options.chip}));
  arith::ArithEmulateUnsupportedFloatsOptions floatEmuOpts;
  SmallVector<std::string, 3> unsupportedFloats = {"bf16", "f8E4M3FNUZ",
                                                   "f8E5M2FNUZ"};
  floatEmuOpts.sourceTypeStrs = unsupportedFloats;
  floatEmuOpts.targetTypeStr = "f32";
  gpuPm.addPass(arith::createArithEmulateUnsupportedFloats(floatEmuOpts));
  ArithToAMDGPUConversionPassOptions arithOptions;
  arithOptions.chipset = options.chip;
  arithOptions.allowPackedF16Rtz = true;
  arithOptions.saturateFP8Truncf = true;
  gpuPm.addPass(createArithToAMDGPUConversionPass(arithOptions));
  if (!archInfo.hasFp8ConversionInstrs)
    gpuPm.addPass(createEmulateFp8ExtTruncPass());
  gpuPm.addPass(memref::createExpandStridedMetadataPass());
  // We need to lower affine again, because the expand strided metadata pass
  // adds back affine.apply for memref.subview
  gpuPm.addPass(createLowerAffinePass());
  gpuPm.addPass(createLowerGpuOpsToROCDLOpsPass(
      options.chip, /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
      /*useBarePtrCallConv=*/true, gpu::amd::Runtime::HIP));
  // Ensure we only run passes on LLVM functions inside GPU modules.
  auto &llvmFuncPm = gpuPm.nest<LLVM::LLVMFuncOp>();
  // -canonicalize -cse so that we don't have to crawl through memref
  // descriptors. (Mainly we want the `extractvalue` fold).
  llvmFuncPm.addPass(createCanonicalizerPass());
  llvmFuncPm.addPass(createCSEPass());
  llvmFuncPm.addPass(rock::createRockPrepareLLVMPass());
  if (options.compile) {
    GpuROCDLAttachTargetOptions opts;
    opts.triple = options.triple;
    opts.chip = options.chip;
    opts.features = options.features;
    opts.optLevel = options.optLevel;
    pm.addPass(createGpuROCDLAttachTarget(opts));
    pm.addPass(createGpuModuleToBinaryPass());
    pm.addPass(createRockCheckResidencyPass());
  }
  // Quick hack around the facct that our host code runner pipeline can't
  // include our fp8 extf implmenentation becasue of MHAL's organization. That
  // pass will ideally be nicely implemented and upstreamed Later (tm).
  pm.addPass(createEmulateFp8ExtTruncPass());
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
