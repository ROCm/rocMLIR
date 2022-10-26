//===- Pipelines.h - XModel pipelines ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all the XModel pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XMODEL_PIPELINES_H_
#define MLIR_DIALECT_XMODEL_PIPELINES_H_

#include "mlir/Pass/PassOptions.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir::detail;
using namespace llvm::cl;

namespace mlir {
namespace xmodel {

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

//===--- Graph Pipeline -----------------------------------------------===//
struct GraphOptions : public PassPipelineOptions<GraphOptions> {

  PassOptions::ListOption<std::string> targets{
      *this, "targets",
      desc("list of target architectures to clone kernels for")};
};

/// Adds the "partition" pipeline to the `OpPassManager`.
void buildGraphPipeline(OpPassManager &pm, const GraphOptions &options = {});

//===--- Model Pipeline ---------------------------------------------------===//
struct PackageOptions : public PassPipelineOptions<PackageOptions> {};

/// Build the XMIR Model Pipeline.
void buildPackagePipeline(OpPassManager &pm,
                          const PackageOptions &options = {});

//===--- Runner Pipeline --------------------------------------------------===//
struct RunnerOptions : public PassPipelineOptions<RunnerOptions> {

  PassOptions::ListOption<std::string> targetTypes{
      *this, "target-types",
      desc("list of target architecture types to select from")};
  PassOptions::ListOption<std::string> targetArchs{
      *this, "target-archs",
      desc("list of target architecture chips to select from")};

  PassOptions::Option<bool> enableCoroutines{
      *this, "enable-coroutines", desc("Generate coroutines in CPU execution"),
      init(false)};

  PassOptions::Option<bool> barePtrMemrefs{
      *this, "bare-ptr-memref-kernels",
      desc("Use bare pointers to pass memrefs to GPU kernels"), init(true)};
};

/// Build the XMIR Runner Pipeline.
void buildRunnerPipeline(OpPassManager &pm, const RunnerOptions &options = {});

/// Registers all pipelines for the `xmodel` dialect.
void registerPipelines();

} // namespace xmodel
} // namespace mlir

#endif // MLIR_DIALECT_XMODEL_PIPELINES_H_
