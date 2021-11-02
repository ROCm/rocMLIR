//===- Pipeline.cpp - Create MIOpen compilation pipeline
//------------------===//
//
// Copyright 2020 The MLIR Authors.
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
// This
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/Pipeline.h"

#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllPasses.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

//===- miopen::createPipeline -----------------------------------===//
//===- Consolidate the MIOpen Pipeline here ---------------------===//

void miopen::addPipeline(PassManager &pm, const PipelineConfig &pc) {
  if (pc.highLevelInput) {
    // passes for TOSA and bufferization
    pm.addPass(tosa::createTosaToMIOpenPass());
    pm.addPass(tosa::createTosaToLinalgOnTensors());
    pm.addPass(createLinalgElementwiseOpFusionPass());
    pm.addPass(createLinalgBufferizePass());
    pm.addPass(createFuncBufferizePass());
    pm.addPass(createBufferResultsToOutParamsPass());
    pm.addPass(createFinalizingBufferizePass());
    pm.addPass(miopen::createMIOpenCopyOptPass());
  }

  // Passes for lowering MIOpen dialect.
  pm.addPass(miopen::createAffixTuningParametersPass(
      pc.perf.gridSize, pc.perf.blockSize, pc.perf.config));
  pm.addPass(miopen::createLowerMIOpenOpsStep1Pass());
  pm.addPass(miopen::createAffineTransformPass());
  pm.addPass(miopen::createLowerMIOpenOpsStep2Pass());

  if (!pc.tuningTest) {
    if (pc.highLevelInput) {
      pm.addPass(miopen::createMIOpenLinalgAlignPass());
      pm.addPass(createConvertLinalgToAffineLoopsPass());
    }
    pm.addPass(miopen::createLowerMIOpenOpsStep3Pass());
    pm.addPass(miopen::createLowerMIOpenOpsStep4Pass());
    pm.addPass(miopen::createLowerMIOpenOpsStep5Pass());
    pm.addPass(createLowerMIOpenOpsToGPUPass());

    // Passes for lowering linalg dialect.
    pm.addPass(createConvertLinalgToAffineLoopsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createLowerToCFGPass());

    // Passes for lowering ROCDL dialect
    if (pc.runBackend) {
      pm.addPass(createGpuKernelOutliningPass());
      pm.addPass(createStripDebugInfoPass());
      pm.addPass(createLowerGpuOpsToROCDLOpsPass(/*indexBitWidth=*/32));
      pm.addPass(createGpuSerializeToHsacoPass(pc.target.triple, pc.target.chip,
                                               pc.target.features,
                                               /*optLevel=*/3));
    }
  }
}
