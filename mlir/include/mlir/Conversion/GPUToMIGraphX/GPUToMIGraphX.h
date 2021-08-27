//===-- GPUToMIGraphX fused func to code obj pass declarations ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the MIGraphX Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOMIGRAPHX_GPUTOMIGRAPHX_H
#define MLIR_CONVERSION_GPUTOMIGRAPHX_GPUTOMIGRAPHX_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"

namespace mlir {
class LLVMTypeConverter;

namespace migraphx {

std::unique_ptr<Pass> createGPUToMIGraphX();

/// Populates passes to convert from GPU to MIGraphX.
/// The pass lowers a func only contains a gpu launch and result allocation
/// with constant numbers representing dimension into a code_obj operation
void addGPUToMIGraphXPasses(OpPassManager &pm);

/// Populates conversion passes GPU to MIGraphX.
void populateFuncToCOBJPatterns(MLIRContext *context,
                                            OwningRewritePatternList *patterns);

} // namespace migraphx
} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOMIGRAPHX_GPUTOMIGRAPHX_H
