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

// Compilation pipeline for TOSA/MIOpen partitioning
void addPartitionPipeline(PassManager &pm, bool toMIOpen = true);

// Compilation pipeline from MIXR/TOSA to MIOpen
void addHighLevelPipeline(PassManager &pm, bool toMIOpen = true);

// Compilation pipeline from MIOpen to LLVM
void addPipeline(PassManager &pm, bool applicability = false,
                 bool highLevel = true);

// Compilation pipeline from LLVM to Binary
void addBackendPipeline(PassManager &pm, const std::string &triple,
                        const std::string &chip, const std::string &features,
                        int32_t optLevel = 3, int32_t indexBitWidth = 32);

} // namespace miopen
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_PIPELINE_H
