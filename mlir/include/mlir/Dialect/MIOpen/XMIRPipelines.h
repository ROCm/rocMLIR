//===- XMRunner.h - XMIR Runner pipeline ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all the XMIR pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XMIR_PIPELINES_H_
#define MLIR_DIALECT_XMIR_PIPELINES_H_

#include "mlir/Pass/PassOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace xmir {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

//===--- Model Pipeline ---------------------------------------------------===//
struct ModelOptions : public PassPipelineOptions<ModelOptions> {};

/// Build the XMIR Model Pipeline.
void buildModelPipeline(OpPassManager &pm,
                        const xmir::ModelOptions &options = {});

//===--- Runner Pipeline --------------------------------------------------===//
struct RunnerOptions : public PassPipelineOptions<RunnerOptions> {

  PassOptions::Option<bool> cpuOnly{
      *this, "cpu-only", desc("Generate CPU-only code "), init(false)};

  PassOptions::Option<bool> barePtrMemrefs{
      *this, "bare-ptr-memref-kernels",
      desc("Use bare pointers to pass memrefs to GPU kernels"), init(true)};
};

/// Build the XMIR Runner Pipeline.
void buildRunnerPipeline(OpPassManager &pm, const RunnerOptions &options = {});

/// Registers all pipelines for the `xmir` dialect.
void registerPipelines();

} // namespace xmir
} // namespace mlir

#endif // MLIR_DIALECT_XMIR_PIPELINES_H_
