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
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/Passes.h"
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
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Tosa/IR/TargetEnv.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "amd/include/TritonAMDGPUToLLVM/Passes.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"

// Triton includes (for backend pipeline)
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"
#include <optional>

using namespace mlir;
using namespace mlir::triton;

// Based on make_ttir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeTTIR(mlir::OpPassManager *pm) {
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mlir::triton::createTritonRewriteTensorPointer());
  pm->addPass(mlir::triton::createTritonRewriteTensorDescriptorToPointer());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::triton::createTritonCombineOps());
  pm->addPass(mlir::triton::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mlir::triton::createTritonLoopUnroll());
}

static bool isPingpongScheduleEnabled(StringRef arch, bool useAsyncCopy) {
  return arch.starts_with("gfx942") ||
         (arch.starts_with("gfx950") && useAsyncCopy);
}

static bool isInThreadTransposeEnabled(StringRef arch) {
  return arch.starts_with("gfx942");
}

static bool isAsyncCopyEnabled(StringRef arch) {
  return arch.starts_with("gfx950") || arch.starts_with("gfx1250");
}

// Based on make_ttgir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeTTGIR(mlir::OpPassManager *pm, std::string arch, int numWarps,
                      int numCTAs, int numStages, int threadPerWarp,
                      int matrixInstrNonkdim, int kpack) {
  pm->addPass(mlir::triton::createConvertTritonToTritonGPU(
      {"hip:" + arch, numWarps, threadPerWarp, numCTAs}));
  pm->addPass(mlir::triton::gpu::createTritonGPUCoalesce());
  pm->addPass(mlir::triton::gpu::createTritonGPUF32DotTC({false}));
  pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
  pm->addPass(mlir::createTritonAMDGPUAccelerateMatmul(
      {arch, matrixInstrNonkdim, kpack}));
  pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  // TODO ROCm Check if we want to compare MI100 and greater
  pm->addPass(mlir::createTritonAMDGPUOptimizeEpilogue());
  pm->addPass(
      mlir::triton::amdgpu::createTritonAMDGPUOptimizeDotOperands({arch}));
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUHoistLayoutConversions());
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUSinkLayoutConversions());

  pm->addPass(mlir::triton::gpu::createTritonGPUFuseNestedLoops());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createCanonicalizerPass());

  // TODO(ROCm) Modify when corresponding run time flags are introduced.
  std::string scheduleHint = "none";

  bool useAsyncCopy = isAsyncCopyEnabled(arch);
  bool useBlockPingpong = isPingpongScheduleEnabled(arch, useAsyncCopy);

  pm->addPass(mlir::createTritonAMDGPUScheduleLoops({numStages}));
  pm->addPass(
      mlir::createTritonAMDGPUPipeline({useAsyncCopy, useBlockPingpong}));
  if (useAsyncCopy) {
    pm->addPass(mlir::createTritonAMDGPUCoalesceAsyncCopy({arch}));
  }
  pm->addPass(mlir::createCanonicalizerPass());
  if (scheduleHint != "none") {
    pm->addPass(mlir::triton::createTritonAMDGPUInsertInstructionSchedHintsPass(
        {scheduleHint}));
  }
  pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
  if (isInThreadTransposeEnabled(arch)) {
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUInThreadTranspose());
    pm->addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  }
  pm->addPass(mlir::createTritonAMDGPUReorderInstructions());
  if (useBlockPingpong && numStages > 1) {
    pm->addPass(mlir::createTritonAMDGPUBlockPingpong({numStages}));
  }

  // TODO(roctriton): useBufferOps
  // if(false) {
  //   pm->addNestedPass<mlir::triton::FuncOp>(
  //       mlir::createTritonAMDGPUCanonicalizePointers());
  //   pm->addPass(mlir::createCanonicalizerPass());
  //   pm->addPass(mlir::createTritonAMDGPUConvertToBufferOps(
  //       {arch, /*allowBufferAtomics*/ true,
  //       /*analyzeSmallTensorOfst*/ false}));
  // }

  pm->addPass(mlir::createTritonAMDFoldTrueCmpI());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
}

// Based on make_llir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeLLIR(mlir::OpPassManager *pm, const std::string &arch,
                     int numStages) {
  pm->addPass(mlir::createTritonAMDGPUUpdateAsyncWaitCount({arch}));
  pm->addPass(mlir::triton::AMD::createConvertWarpPipelinePass());
  pm->addPass(mlir::createSCFToControlFlowPass());

  // TODO: do we need this?
  // pm->addPass(gluon::createGluonInline());
  pm->addPass(mlir::createConvertIndexToLLVMPass());

  pm->addPass(mlir::triton::createAllocateAMDGPUSharedMemory());

  // ## __HIP_FTZ is used to control the denorm flushing behavior of exp2 op as
  // follows:
  // ## 1. If __HIP_FTZ = 1, exp2 flushes denorms in input and output regardless
  // ##    of the value of kernel arg `allow_flush_denorm`.
  // ## 2. If __HIP_FTZ = 0, whether exp2 flushes denorms in input and output
  // ##    depends on the value of kernel arg `allow_flush_denorm`.
  // ## 3. __HIP_FTZ is default to 1 and not exposed as a kernel argument.
  // ##    For now it is used as a controller for developers only.
  pm->addPass(
      mlir::triton::createConvertTritonAMDGPUToLLVMPass(arch, /*ftz=*/true));
  pm->addPass(
      mlir::triton::AMD::createTritonAMDGPUConvertWarpSpecializeToLLVMPass(
          arch));
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());

  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.
  pm->addPass(mlir::createConvertControlFlowToLLVMPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  if (/*(instruction_sched_variant=="none") == */ /* DISABLES CODE */
      (false)) {
    pm->addPass(mlir::triton::createTritonAMDGPULowerInstructionSchedHintsPass(
        arch, numStages));
  }

  // TODO: add_di_scope

  pm->addPass(mlir::triton::createConvertBuiltinFuncToLLVMPass(/*ftz=*/true));
}

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
    funcPm.addPass(createTosaToTensorPass());
    funcPm.addPass(createTosaToRockPass());
    funcPm.addPass(rock::createRockViewToTransformPass());
    funcPm.addPass(rock::createRockDetectFlashDecodingPass());
  }

  funcPm.addPass(createRocmlirCustomTosaDecomposePass());
  funcPm.addPass(createRocmlirCustomTosaToLinalgPass());

  tosa::TosaAttachTargetOptions tosaOptions;
  tosaOptions.specificationVersion = tosa::SpecificationVersion::V_1_0;
  tosaOptions.level = tosa::Level::none;
  tosaOptions.profiles.push_back("pro_int");
  tosaOptions.profiles.push_back("pro_fp");
  tosaOptions.extensions.push_back("int4");
  tosaOptions.extensions.push_back("bf16");
  tosaOptions.extensions.push_back("fp8e4m3");
  tosaOptions.extensions.push_back("fp8e5m2");
  tosaOptions.extensions.push_back("mxfp");

  funcPm.addPass(tosa::createTosaAttachTarget(tosaOptions));

  // use tosa conversion pipeline
  // (see mlir/lib/Conversion/TosaToLinalg/TosaToLinalgPass.cpp)
  TosaToLinalgOptions tosaToLinalgOptions;
  TosaToLinalgNamedOptions tosaToLinalgNamedOptions;
  // pass std::nullopt as validation options to avoid running tosa-validate pass
  tosa::addTosaToLinalgPasses(pm, tosaToLinalgOptions, tosaToLinalgNamedOptions,
                              /*validationOptions=*/std::nullopt);

  // for tosa control flow
  /* rocmlir-opt --tosa-to-tensor --tosa-to-scf --tosa-to-arith
   */
  auto &funcPm2 = pm.nest<func::FuncOp>();
  funcPm2.addPass(createTosaToTensorPass());
  funcPm2.addPass(createTosaToSCFPass());
  funcPm2.addPass(createTosaToArithPass());

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

  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.allowReturnAllocsFromLoops = true;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  bufOpts.unknownTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;

  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  bufferization::BufferResultsToOutParamsPassOptions bufferResultToOutOptions;
  bufferResultToOutOptions.modifyPublicFunctions = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(
      bufferResultToOutOptions));

  // Sort dimensions according to the underlying memory layout strides
  // TODO(roctriton): RockFindFirstGemmIndexPass for attention fusion support
  // if (!noRock) {
  //   auto &funcPm4 = pm.nest<func::FuncOp>();
  //   funcPm4.addPass(createRockFindFirstGemmIndexPass());
  // }
}

void rock::buildKernelPipeline(OpPassManager &pm,
                               const rock::KernelOptions &options) {
  // rock lowering (tuning, global to block)
  /* rocmlir-opt --rock-affix-params --rock-conv-to-gemm
   *   --rock-fold-broadcast --rock-affix-params --rock-gemm-to-gridwise
   *   --rock-regularize --rock-gridwise-gemm-to-blockwise
   * --rock-blockwise-load-tile-to-threadwise
   */
  auto &funcPm = pm.nest<func::FuncOp>();

  if (options.applicabilityMode == rock::ApplicabilityMode::Applicability ||
      options.applicabilityMode == rock::ApplicabilityMode::Full) {
    funcPm.addPass(rock::createRockAffixTuningParametersPass(
        rock::RockAffixTuningParametersPassOptions{options.tuningFallback}));
    funcPm.addPass(rock::createRockConvToGemmPass());
    funcPm.addPass(rock::createRockGemmLinalgSplitkNormalizationPass());
    funcPm.addPass(rock::createRockGemmToGridwisePass());
    funcPm.addPass(rock::createRockRegularizePass());
    funcPm.addPass(rock::createRockShuffleGemmForReductions());
    funcPm.addPass(rock::createRockGridwiseGemmToBlockwisePass());

    // TODO(roctriton): implement fusions
    // funcPm.addPass(rock::createRockLinalgAlignPass());
    // funcPm.addPass(createConvertLinalgToAffineLoopsPass());
  }

  if (options.applicabilityMode == rock::ApplicabilityMode::NonApplicability ||
      options.applicabilityMode == rock::ApplicabilityMode::Full) {
    funcPm.addPass(rock::createRockTransformsToPtrPass());
    funcPm.addPass(rock::createRockTransformsToPointerArithPass());
    funcPm.addPass(rock::createRockToTTIRPass());
    // RockMemrefToTensorPass operates on ModuleOp (converts func.func to
    // tt.func)
    pm.addPass(rock::createRockMemrefToTensorPass());
    // After this point, function is triton::FuncOp
    auto &ttFuncPm = pm.nest<triton::FuncOp>();
    ttFuncPm.addPass(rock::createRockUnbufferizePass());
    ttFuncPm.addPass(createCanonicalizerPass());
    ttFuncPm.addPass(createCSEPass());
  }
}

void rock::buildTritonPipeline(OpPassManager &pm,
  const rock::TritonOptions &options) {
  StringRef arch = options.arch.getValue();
  AmdArchInfo archInfo = rock::lookupArchInfo(arch);

  makeTTIR(&pm);
  int numWarps = 4;
  int numCTAs = 1;
  int numStages = 2;
  int threadPerWarp = archInfo.waveSize;
  int matrixInstrNonkdim = 16;
  int kpack = 1;
  makeTTGIR(&pm, arch.str(), numWarps, numCTAs, numStages, threadPerWarp,
            matrixInstrNonkdim, kpack);
}


// Build host code lowering pipeline (func + GPU ops -> LLVM)
// Follows the pattern from mlir-hal/lib/Dialect/MHAL/Pipelines/Pipelines.cpp
static void buildHostLoweringPipeline(mlir::OpPassManager &pm) {
  // Lower linalg to loops (for operations like linalg.fill in -pv mode)
  pm.addPass(createConvertLinalgToLoopsPass());

  // Lower affine to standard loops
  pm.addPass(createLowerAffinePass());

  // Expand strided metadata (handles memref.expand_shape, etc.)
  pm.addPass(memref::createExpandStridedMetadataPass());

  // Lower SCF to control flow
  pm.addPass(createSCFToControlFlowPass());

  // Make GPU operations async - required by GpuToLLVMConversionPass patterns
  pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());

  // Lower remaining operations to LLVM (order follows MHAL pipeline)
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());

  // Lower memref operations to LLVM BEFORE GPU conversion (per MHAL pattern)
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());

  // Convert GPU operations to runtime calls
  GpuToLLVMConversionPassOptions gpuOpts;
  gpuOpts.kernelBarePtrCallConv = true; // Use kernel bare ptr, not host
  pm.addPass(createGpuToLLVMConversionPass(gpuOpts));

  // Lower any remaining func operations to LLVM (including external
  // declarations)
  pm.addPass(createConvertFuncToLLVMPass());

  // Cleanup
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void rock::buildBackendPipeline(OpPassManager &pm,
                                const rock::BackendOptions &options) {
  // Get architecture from options or use default
  std::string arch = options.chip.empty() ? "gfx1100" : options.chip.getValue();
  int numStages = 2;

  // Run MLIR passes to convert TritonGPU -> LLVM dialect
  makeLLIR(&pm, arch, numStages);

  // Optionally generate the HSACO binary
  if (options.compile) {
    // Add the TritonToHsaco pass to convert LLVM dialect to HSACO binary
    // This implements the functionality from Triton's compiler.py:
    // - make_llir() lines 358-449: LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    // - make_amdgcn() lines 452-473: LLVM -> AMDGCN assembly
    // - make_hsaco() lines 476-488: AMDGCN assembly -> HSACO binary
    rock::TritonToHsacoPassOptions hsacoOpts;
    hsacoOpts.arch = arch;
    hsacoOpts.numWarps = 4; // TODO: Get from options
    hsacoOpts.wavesPerEU = 0;
    hsacoOpts.enableFpFusion = true;
    hsacoOpts.allowFlushDenorm = false;
    pm.addPass(rock::createTritonToHsacoPass(hsacoOpts));

    // Restore host functions (main, wrapper) that were stored during
    // RockMemrefToTensorPass. This converts func.call @kernel to
    // gpu.launch_func.
    pm.addPass(rock::createRockRestoreHostCodePass());

    // Lower host code (GPU launch + func/memref ops) to LLVM
    buildHostLoweringPipeline(pm);
  }
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
  PassPipelineRegistration<rock::TritonOptions>(
        "rock-triton-pipeline",
        "Convert Triton IR to TritonGPU IR.",
        buildTritonPipeline);  
  PassPipelineRegistration<rock::BackendOptions>(
      "rock-backend-pipeline",
      " representations and algorithms for sparse tensors.",
      buildBackendPipeline);
}
