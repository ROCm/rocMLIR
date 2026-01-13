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

// Triton includes (for backend pipeline)
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"
#include <optional>

using namespace mlir;

namespace mt = ::mlir::triton;

// Based on make_ttir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeTTIR(mlir::OpPassManager *pm) {
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mt::createTritonRewriteTensorPointer());
  pm->addPass(mt::createTritonRewriteTensorDescriptorToPointer());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonCombineOps());
  pm->addPass(mt::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createTritonLoopUnroll());
}

static bool
isPingpongScheduleEnabled(const stream_executor::RocmComputeCapability &rocm_cc,
                          bool use_async_copy) {
  return rocm_cc.gfx9_mi300() || (rocm_cc.gfx9_mi350() && use_async_copy);
}

static bool isInThreadTransposeEnabled(
    const stream_executor::RocmComputeCapability &rocm_cc) {
  return rocm_cc.gfx9_mi300();
}

// Based on make_ttgir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeTTGIR(mlir::OpPassManager *pm,
                      const stream_executor::RocmComputeCapability &rocm_cc,
                      int num_warps, int num_ctas, int num_stages) {
  pm->addPass(mt::createConvertTritonToTritonGPU(
      {absl::StrCat("hip:", rocm_cc.gfx_version()), num_warps,
       rocm_cc.threads_per_warp(), num_ctas}));
  pm->addPass(mt::gpu::createTritonGPUCoalesce());
  pm->addPass(mt::gpu::createTritonGPUF32DotTC({false}));
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  pm->addPass(
      mlir::createTritonAMDGPUAccelerateMatmul({rocm_cc.gfx_version()}));
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  // TODO ROCm Check if we want to compare MI100 and greater
  pm->addPass(mlir::createTritonAMDGPUOptimizeEpilogue());
  pm->addPass(mt::amdgpu::createTritonAMDGPUOptimizeDotOperands(
      {rocm_cc.gfx_version()}));
  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUHoistLayoutConversions());

  pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createCanonicalizerPass());

  // TODO(ROCm) Modify when corresponding run time flags are introduced.
  std::string schedule_hint = "none";

  bool use_async_copy = false; // Not enabled by default.
  bool use_block_pingpong = isPingpongScheduleEnabled(rocm_cc, use_async_copy);

  pm->addPass(mlir::createTritonAMDGPUScheduleLoops({num_stages}));
  pm->addPass(
      mlir::createTritonAMDGPUPipeline({use_async_copy, use_block_pingpong}));
  if (use_async_copy) {
    pm->addPass(
        mlir::createTritonAMDGPUCoalesceAsyncCopy({rocm_cc.gfx_version()}));
  }
  pm->addPass(mlir::createCanonicalizerPass());
  if (schedule_hint != "none") {
    pm->addPass(
        mt::createTritonAMDGPUInsertInstructionSchedHintsPass({schedule_hint}));
  }
  pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  pm->addPass(mt::gpu::createTritonGPUReduceDataDuplication());
  if (isInThreadTransposeEnabled(rocm_cc)) {
    pm->addNestedPass<mlir::triton::FuncOp>(
        mlir::createTritonAMDGPUInThreadTranspose());
    pm->addPass(mt::gpu::createTritonGPURemoveLayoutConversions());
  }
  pm->addPass(mlir::createTritonAMDGPUReorderInstructions());
  if (use_block_pingpong && num_stages > 1) {
    pm->addPass(mlir::createTritonAMDGPUBlockPingpong({num_stages}));
  }

  pm->addNestedPass<mlir::triton::FuncOp>(
      mlir::createTritonAMDGPUCanonicalizePointers());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createTritonAMDGPUConvertToBufferOps(
      {rocm_cc.gfx_version(), /*allowBufferAtomics*/ true,
       /*analyzeSmallTensorOfst*/ false}));

  pm->addPass(mlir::createTritonAMDFoldTrueCmpI());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
}

// Based on make_llir() in
// @triton//:third_party/amd/backend/compiler.py
static void makeLLIR(mlir::OpPassManager *pm,
                     const stream_executor::RocmComputeCapability &rocm_cc,
                     int num_stages) {
  const int custom_lds_size = 0;
  pm->addPass(mlir::createTritonAMDGPUUpdateAsyncWaitCount());
  pm->addPass(mlir::triton::AMD::createOptimizeLDSUsagePass(
      rocm_cc.gfx_version(), custom_lds_size));
  pm->addPass(mlir::createSCFToControlFlowPass());
  pm->addPass(mlir::createConvertIndexToLLVMPass());
  pm->addPass(mt::gpu::createAllocateSharedMemory());
  pm->addPass(
      mt::createConvertTritonAMDGPUToLLVMPass(rocm_cc.gfx_version(), true));
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.
  pm->addPass(mlir::createConvertControlFlowToLLVMPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
  if (/*(instruction_sched_variant=="none") == */ /* DISABLES CODE */ (false)) {
    pm->addPass(mt::createTritonAMDGPULowerInstructionSchedHintsPass(
        rocm_cc.gfx_version(), num_stages));
  }
  pm->addPass(mt::createConvertBuiltinFuncToLLVMPass(/*ftz=*/true));
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
  if (!noRock) {
    auto &funcPm4 = pm.nest<func::FuncOp>();
    funcPm4.addPass(createRockRemoveOutputAllocPass());
    funcPm4.addPass(createRockFindFirstGemmIndexPass());
    funcPm4.addPass(createRockSortDimensionsMemoryLayoutPass());
  }
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
    funcPm.addPass(rock::createRockBlockwiseLoadTileToThreadwisePass());

    // align linalg tiling
    /* rocmlir-opt --rock-linalg-align --canonicalize
     * --convert-linalg-to-affine-loops
     */
    funcPm.addPass(rock::createRockLinalgAlignPass());
    // funcPm.addPass(createConvertLinalgToAffineLoopsPass());
  }

  if (options.applicabilityMode == rock::ApplicabilityMode::NonApplicability ||
      options.applicabilityMode == rock::ApplicabilityMode::Full) {
    funcPm.addPass(rock::createRockSugarToLoopsPass());
    // TODO: RockToTriton

    // Triton backend pipeline
    // This converts Rock dialect to Triton IR and compiles to LLVM

    // 1. Convert Rock to Triton
    // pm.addPass(rock::createRockToTritonPass());

    makeTTIR(pm);
    makeTTGIR(pm, rocm_cc, numWarps, numCtas, numStages);
    makeLLIR(pm, rocm_cc, numStages);
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
  floatEmuOpts.sourceTypeStrs.assign(
      {"f8E4M3FNUZ", "f8E5M2FNUZ", "f8E4M3FN", "f8E5M2", "f8E8M0FNU"});
  floatEmuOpts.targetTypeStr = "f32";
  gpuPm.addPass(arith::createArithEmulateUnsupportedFloats(floatEmuOpts));
  arith::ArithExpandOpsPassOptions arithExpandOpsOptions;
  // emulate truncf(f32)->f8E8M0FNU types. This is used when scales are passed
  // in as f32 for the scaledGemms
  arithExpandOpsOptions.includeF8E8M0 = true;
  gpuPm.addPass(arith::createArithExpandOpsPass(arithExpandOpsOptions));
  ArithToAMDGPUConversionPassOptions arithOptions;
  arithOptions.chipset = options.chip;
  // disable packed truncation to fp16 with rtz (round towards zero) as it
  // generates less accurate results.
  arithOptions.allowPackedF16Rtz = false;
  arithOptions.saturateFP8Truncf = true;
  gpuPm.addPass(createArithToAMDGPUConversionPass(arithOptions));
  EmulateFp8ExtTruncPassOptions f8ConversionOptions;
  f8ConversionOptions.hasFp8ConversionInstrs = archInfo.hasFp8ConversionInstrs;
  f8ConversionOptions.hasOcpFp8ConversionInstrs =
      archInfo.hasOcpFp8ConversionInstrs;
  gpuPm.addPass(createEmulateFp8ExtTruncPass(f8ConversionOptions));
  gpuPm.addPass(memref::createExpandStridedMetadataPass());
  // We need to lower affine again, because the expand strided metadata pass
  // adds back affine.apply for memref.subview
  gpuPm.addPass(createLowerAffinePass());
  ConvertGpuOpsToROCDLOpsOptions rocdlOpts;
  rocdlOpts.chipset = options.chip;
  rocdlOpts.indexBitwidth = kDeriveIndexBitwidthFromDataLayout;
  rocdlOpts.useBarePtrCallConv = true;
  rocdlOpts.runtime = gpu::amd::Runtime::HIP;
  rocdlOpts.allowedDialects.assign(
      {"memref", "math", "cf", "func", "vector", "arith"});
  gpuPm.addPass(createConvertGpuOpsToROCDLOps(rocdlOpts));
  gpuPm.addPass(rock::createRockAddDirectToLDSAliasInfoPass());
  ConvertRockOpsToROCDLOpsOptions rockToROCDLOpts;
  rockToROCDLOpts.chipset = options.chip;
  gpuPm.addPass(rock::createConvertRockOpsToROCDLOps(rockToROCDLOpts));
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
  // Quick hack around the fact that our host code runner pipeline can't
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
