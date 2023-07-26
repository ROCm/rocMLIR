//===- Pipelines.h - Rock pipelines ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all sparse tensor pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_PIPELINES_H_
#define MLIR_DIALECT_ROCK_PIPELINES_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace rock {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

//===--- Bufferize Pipeline -----------------------------------------------===//
struct BufferizeOptions : public PassPipelineOptions<BufferizeOptions> {

  PassOptions::Option<bool> disableRock{
      *this, "disable-rock",
      desc("Disable Rock dialect targeting when bufferizing"), init(false)};
};

/// Adds the `bufferize` pipeline to the `OpPassManager`.
void buildBufferizePipeline(OpPassManager &pm,
                            const BufferizeOptions &options = {});

//===--- Kernel Pipeline --------------------------------------------------===//
struct KernelOptions : public PassPipelineOptions<KernelOptions> {

  PassOptions::Option<bool> enableApplicability{
      *this, "enable-applicability", desc("Only test for applicability"),
      init(false)};
  PassOptions::Option<bool> enableFusion{
      *this, "enable-fusion",
      desc("Enable fusion alignment between anchor op and peripheral ops"),
      init(true)};
  PassOptions::Option<bool> tuningFallback{
      *this, "tuningFallback",
      desc("Falls back default if invalid config is given"), init(false)};
};

/// Adds the `kernel` pipeline to the `OpPassManager`.
void buildKernelPipeline(OpPassManager &pm, const KernelOptions &options = {});

//===--- Backend Pipeline -------------------------------------------------===//
struct BackendOptions : public PassPipelineOptions<BackendOptions> {

  PassOptions::Option<std::string> triple{
      *this, "triple", desc("AMDGPU target triple: amdgcn-amd-amdhsa"),
      init("")};
  PassOptions::Option<std::string> chip{
      *this, "chip", desc("AMDGPU ISA version: e.g. gfx908"), init("")};
  PassOptions::Option<std::string> features{
      *this, "features", desc("AMDGPU target features"), init("")};
  PassOptions::Option<int32_t> optLevel{
      *this, "opt-level", desc("GPU compiler optimization level"), init(3)};
  PassOptions::Option<int32_t> indexBitwidth{*this, "index-bitwidth",
                                             desc("Index bit-width"), init(32)};
  PassOptions::Option<bool> compile{
      *this, "compile", desc("should the serailization pass be run"),
      init(true)};
};

/// Adds the `kernel` pipeline to the `OpPassManager`.
void buildBackendPipeline(OpPassManager &pm,
                          const BackendOptions &options = {});

/// Registers all pipelines for the `rock` dialect.
void registerPipelines();

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PIPELINES_H_
