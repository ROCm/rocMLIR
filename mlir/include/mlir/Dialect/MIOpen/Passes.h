//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_PASSES_H_
#define MLIR_DIALECT_MIOPEN_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace miopen {

/// Create a pass to clone kernel funcs into MIOpen module
std::unique_ptr<Pass> createMIOpenCloneKernelsPass();

/// Create a pass to apply target implementation to host kernel funcs
std::unique_ptr<Pass> createMIOpenApplyImplPass();

/// Create a pass to
std::unique_ptr<Pass> createMIOpenAsyncLaunchPass();

/// Create a pass to convert MIOpen conv2d operations to transform and
/// gridwise_gemm operations.
std::unique_ptr<Pass> createMIOpenConvToGemmPass();

/// Create a pass to convert MIOpen gridwise_gemm operations to blockwise
/// operations.
std::unique_ptr<Pass> createMIOpenGridwiseGemmToBlockwisePass();

/// Create a pass to align tiling of subsequent linalg.generic ops with
/// the miopen.conv2d op after the gridwise -> blockwise pass.
std::unique_ptr<Pass> createMIOpenLinalgAlignPass();

/// Create a pass to optimize out global copies.
std::unique_ptr<Pass> createMIOpenCopyOptPass();

/// Create a pass to convert MIOpen blockwise operations to threadwise
/// operations.
std::unique_ptr<Pass> createMIOpenBlockwiseGemmToThreadwisePass();

/// Create a pass to convert MIOpen threadwise operations to other dialects.
std::unique_ptr<Pass> createMIOpenThreadwiseGemmLoweringPass();

/// Create a pass to expand transforming_for and other MIOpen shorthand to other
/// dialects.
std::unique_ptr<Pass> createMIOpenSugarToLoopsPass();

/// Create a pass to clean up math using integer range analysis and other MLIR
/// passes.
std::unique_ptr<Pass> createMIOpenCleanMathPass();

/// Create a pass to convert affine / loop to cf dialect.
std::unique_ptr<Pass> createMIOpenLoopsToCfPass();

/// Create a pass to affix tuning parameters to gridwise gemm ops.
std::unique_ptr<Pass>
createAffixTuningParametersPass(int64_t blockSizeOverride = 0,
                                int64_t gridSizeOverride = 0,
                                bool fallBackNoConfig = false);

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MIOpen/Passes.h.inc"

} // namespace miopen
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_PASSES_H_
