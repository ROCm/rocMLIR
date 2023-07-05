//===- InitRocMLIRPasses.h - rocMLIR Passes Registration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all our custom
// dialects and passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITROCMLIRPASSES_H_
#define MLIR_INITROCMLIRPASSES_H_

// rocMLIR includes
#include "mlir/Conversion/RocMLIRPasses.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"

// MLIR includes
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/InitMHALPasses.h"
#include "mlir/Transforms/Passes.h"

#include <cstdlib>

namespace mlir {

inline void registerUpstreamPasses() {

  // Conversion passes
  registerConvertAffineToStandard();
  registerArithToAMDGPUConversionPass();
  registerConvertAMDGPUToROCDL();
  registerArithToLLVMConversionPass();
  registerConvertFuncToLLVMPass();
  registerConvertGpuOpsToROCDLOps();
  registerConvertMathToLLVMPass();
  registerFinalizeMemRefToLLVMConversionPass();
  registerReconcileUnrealizedCasts();
  registerSCFToControlFlow();
  registerTosaToArith();
  registerTosaToLinalg();
  registerTosaToLinalgNamed();
  registerTosaToSCF();

  // MLIR passes
  registerTransformsPasses();
  affine::registerAffinePasses();
  arith::registerArithPasses();
  bufferization::registerBufferizationPasses();
  func::registerFuncPasses();
  registerGPUPasses();
  registerGpuSerializeToHsacoPass();
  registerLinalgPasses();
  LLVM::registerLLVMPasses();
  memref::registerMemRefPasses();
  registerSCFPasses();
  tensor::registerTensorPasses();
  tosa::registerTosaOptPasses();
  vector::registerVectorPasses();
}

// This function may be called to register the rocMLIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.
inline void registerRocMLIRPasses() {
  registerRocMLIRConversionPasses();
  migraphx::registerPasses();
  rock::registerPasses();
  rock::registerPipelines();

  registerMHALPasses();

  registerUpstreamPasses();
}

} // namespace mlir

#endif // MLIR_INITROCMLIRPASSES_H_
