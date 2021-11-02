//===- Pipeline.h - MIOpen pipeline creation ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the miopen::addPipeline routine to consolidate the
// MIOpen pipeline creation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_PIPELINE_H
#define MLIR_DIALECT_MIOPEN_PIPELINE_H

#include <mlir/Pass/PassManager.h>

#include <string>

namespace mlir {
namespace miopen {

struct PipelineConfig {
  bool highLevelInput = false;
  bool tuningTest = false;
  bool runBackend = false;
  struct PerfConfig {
    const std::string &config = "";
    int64_t gridSize = 0;
    int64_t blockSize = 0;
  } perf;
  struct TargetConfig {
    const std::string &triple = "";
    const std::string &chip = "";
    const std::string &features = "";
  } target;
};

struct TuningPipeline : PipelineConfig {
  TuningPipeline(const std::string &perfConfig, int64_t gridSize = 0,
                 int64_t blockSize = 0)
      : PipelineConfig{false, true, false, {perfConfig, gridSize, blockSize}} {}
};

struct DriverPipeline : PipelineConfig {
  DriverPipeline(const std::string &perfConfig)
      : PipelineConfig{false, false, false, {perfConfig}} {}
};

template <bool highLevel = false> struct KernelPipeline : PipelineConfig {
  KernelPipeline(const std::string &triple, const std::string &chip,
                 const std::string &features,
                 const std::string &perfConfig = "")
      : PipelineConfig{
            highLevel, false, true, {perfConfig}, {triple, chip, features}} {}
};

// Consolidate MIOpen pipeline creation
void addPipeline(mlir::PassManager &pm, const PipelineConfig &pc);

} // namespace miopen
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_PIPELINE_H
