//===- Pipeline.h - MIGraphx pipeline creation ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the migraphx::addPipeline routine to consolidate the
// MIGraphX pipeline creation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIGRAPHX_PIPELINE_H
#define MLIR_DIALECT_MIGRAPHX_PIPELINE_H

#include <mlir/Pass/PassManager.h>

#include <string>

namespace mlir {
namespace migraphx {

// Compilation pipeline from MIXR to TOSA
void addHighLevelPipeline(PassManager &pm);

// Compilation pipeline from GPU to MIXR code object
void addBackendPipeline(PassManager &pm);

} // namespace migraphx
} // namespace mlir

#endif // MLIR_DIALECT_MIGRAPHX_PIPELINE_H
