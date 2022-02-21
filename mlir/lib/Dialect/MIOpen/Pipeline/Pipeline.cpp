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

#include "mlir/Dialect/MIOpen/Pipeline.h"

#include "mlir/Conversion/MIOpenPasses.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
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

//===- Consolidate the MIOpen Pipelines here ---------------------===//

void miopen::addHighLevelPipeline(PassManager &pm, bool toMIOpen) {
  // passes for TOSA and bufferization
  if (toMIOpen) {
    pm.addPass(tosa::createTosaToMIOpenPass());
  }
  pm.addPass(tosa::createTosaToLinalgNamed());
  pm.addPass(tosa::createTosaToLinalg());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(createLinalgBufferizePass());
  pm.addPass(arith::createArithmeticBufferizePass());
  pm.addPass(createFuncBufferizePass());
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(bufferization::createFinalizingBufferizePass());
  pm.addPass(miopen::createMIOpenCopyOptPass());
}

void miopen::addPipeline(PassManager &pm, bool applicability, bool highLevel) {
  // Passes for lowering MIOpen dialect.
  pm.addPass(miopen::createAffixTuningParametersPass(0, 0));
  pm.addPass(miopen::createLowerMIOpenOpsStep1Pass());
  pm.addPass(miopen::createLowerMIOpenOpsStep2Pass());

  if (!applicability) {
    if (highLevel) {
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
    pm.addPass(createConvertSCFToCFPass());
  }
}

void miopen::addBackendPipeline(PassManager &pm, const std::string &triple,
                                const std::string &chip,
                                const std::string &features, int32_t optLevel,
                                int32_t indexBitWidth) {
  // Passes for lowering ROCDL dialect
  pm.addPass(createStripDebugInfoPass());
  pm.addPass(createLowerGpuOpsToROCDLOpsPass(indexBitWidth));
  pm.addPass(createGpuSerializeToHsacoPass(triple, chip, features, optLevel));
}
