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

#ifndef MLIR_DIALECT_ROCK_PASSES_H_
#define MLIR_DIALECT_ROCK_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace rock {

/// Create a pass to clone kernel funcs into Kernel modules
std::unique_ptr<Pass>
createRockCloneKernelsPass(llvm::ArrayRef<llvm::StringRef> _chips = {});

/// Create a pass to apply target implementation to host kernel funcs
std::unique_ptr<Pass> createRockApplyImplPass();

/// Create a pass to
std::unique_ptr<Pass> createRockAsyncLaunchPass();

/// Create a pass to fold certain transpose operations in order to enable
/// fusing generic operations with convolution and GEMM kernels.
std::unique_ptr<Pass> createRockFoldTransposePass();

/// Create a pass to convert Rock conv2d operations to transform and
/// gridwise_gemm operations.
std::unique_ptr<Pass> createRockConvToGemmPass();

/// Create a pass to convert Rock gemm operations to gridwise operations,
/// adding padding.
std::unique_ptr<Pass> createRockGemmToGridwisePass();

/// Create a pass to convert Rock gridwise_gemm operations to blockwise
/// operations.
std::unique_ptr<Pass> createRockGridwiseGemmToBlockwisePass();

/// Create a pass to align tiling of subsequent linalg.generic ops with
/// the rock.conv2d op after the gridwise -> blockwise pass.
std::unique_ptr<Pass> createRockLinalgAlignPass();

/// Create a pass to optimize out global copies.
std::unique_ptr<Pass> createRockCopyOptPass();

/// Create a pass to convert Rock blockwise operations to threadwise
/// operations.
std::unique_ptr<Pass> createRockBlockwiseGemmToThreadwisePass();

/// Create a pass to convert Rock threadwise operations to other dialects.
std::unique_ptr<Pass> createRockThreadwiseGemmLoweringPass();

/// Create a pass to expand transforming_for and other Rock shorthand to other
/// dialects.
std::unique_ptr<Pass> createRockSugarToLoopsPass();

/// Create a pass to clean up math using integer range analysis and other MLIR
/// passes.
std::unique_ptr<Pass> createRockCleanMathPass();

/// Create a pass to convert affine / loop to cf dialect.
std::unique_ptr<Pass> createRockLoopsToCfPass();

/// Create a pass to affix tuning parameters to gridwise gemm ops.
std::unique_ptr<Pass>
createAffixTuningParametersPass(uint32_t blockSizeOverride = 0,
                                uint32_t gridSizeOverride = 0,
                                bool fallBackNoConfig = false);

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Rock/Passes.h.inc"

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PASSES_H_
