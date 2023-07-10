//===- InitRocMLIRDialects.h - rocMLIR Dialects Registration ----*- C++ -*-===//
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

#ifndef MLIR_INITROCMLIRDIALECTS_H_
#define MLIR_INITROCMLIRDIALECTS_H_

// rocMLIR includes
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Transforms/BufferizableOpInterfaceImpl.h"

// MLIR includes
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitMHALDialects.h"

namespace mlir {

inline void registerUpstreamDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<affine::AffineDialect,
                  amdgpu::AMDGPUDialect,
                  arith::ArithDialect,
                  async::AsyncDialect,
                  bufferization::BufferizationDialect,
                  cf::ControlFlowDialect,
                  gpu::GPUDialect,
                  func::FuncDialect,
                  LLVM::LLVMDialect,
                  linalg::LinalgDialect,
                  math::MathDialect,
                  memref::MemRefDialect,
                  scf::SCFDialect,
                  vector::VectorDialect,
                  ROCDL::ROCDLDialect,
                  tensor::TensorDialect,
                  tosa::TosaDialect>();
  // clang-format on

  // Register bufferization hooks for rock interfaces
  rock::registerBufferizableOpInterfaceExternalModels(registry);

  // Register all dialect extensions.
  bufferization::registerTransformDialectExtension(registry);

  // Register all external models.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerTilingInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
}

// Add all the MLIR dialects to the provided registry.
inline void registerRocMLIRDialects(DialectRegistry &registry) {
  // Register rocMLIR specific dialects
  registry.insert<rock::RockDialect, migraphx::MIGraphXDialect>();

  // Register MHAL dialect
  registerMHALDialects(registry);

  // Register auxiliary Upstream dialects
  registerUpstreamDialects(registry);
}

} // namespace mlir

#endif // MLIR_INITROCMLIRDIALECTS_H_
