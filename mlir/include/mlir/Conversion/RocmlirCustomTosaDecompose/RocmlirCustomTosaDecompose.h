//===-- RocmlirCustomTosaDecompose.h Lower Rocmlir tosa custom ops--*- C++
//-*-*-===//
//
// Part of the rocMLIRProject, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// Pass for decomposing tosa::CustomOp representing backward convolution into
// lower level tosa ops. This pass is essentially a downstream version of
// mlir/lib/Dialect/Tosa/Transforms/TosaDecomposeTransposeConv.cpp (code was
// copied from rocMLIR commit ec067ce842b1580e02e222ec444b877f0f861e1b)
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ROCMLIRCUSTOMTOSADECOMPOSE_ROCMLIRCUSTOMTOSADECOMPOSE_H
#define MLIR_CONVERSION_ROCMLIRCUSTOMTOSADECOMPOSE_ROCMLIRCUSTOMTOSADECOMPOSE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ConversionTarget;

#define GEN_PASS_DECL_ROCMLIRCUSTOMTOSADECOMPOSEPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace rock {
/// Configure legality for converting the rocmlir domain Tosa custom ops to
/// linalg.generic ops
void populateRocmlirCustomTosaDecomposeTarget(ConversionTarget &target);

/// Populates conversion passes from rocMLIR's Tosa custom ops to linalg.generic
/// ops.
void populateRocmlirCustomTosaDecomposeConversionPatterns(
    RewritePatternSet &patterns);
} // namespace rock
} // namespace mlir

#endif // MLIR_CONVERSION_ROCMLIRCUSTOMTOSADECOMPOSE_ROCMLIRCUSTOMTOSADECOMPOSE_H
