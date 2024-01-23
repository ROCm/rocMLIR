//===- Multibuffer.h - Adaptation of multibuffer to rock dialect ---===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TRANSFORMS_MULTIBUFFER_H
#define MLIR_DIALECT_ROCK_TRANSFORMS_MULTIBUFFER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"

namespace mlir {

namespace rock {
FailureOr<SmallVector<rock::GpuAllocOp>> multiBuffer(RewriterBase &rewriter,
                                                     rock::GpuAllocOp allocOp,
                                                     unsigned multiplier,
                                                     bool skipOverrideAnalysis);

FailureOr<SmallVector<rock::GpuAllocOp>> multiBuffer(rock::GpuAllocOp allocOp,
                                                     unsigned multiplier,
                                                     bool skipOverrideAnalysis);

FailureOr<SmallVector<rock::GpuAllocOp>>
updateMultiBuffer(RewriterBase &rewriter, Location loc,
                  ArrayRef<rock::GpuAllocOp> multiBuffer,
                  unsigned newMultiplier);

FailureOr<SmallVector<rock::GpuAllocOp>>
updateMultiBuffer(ArrayRef<rock::GpuAllocOp> multiBuffer,
                  unsigned newMultiplier);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TRANSFORMS_MULTIBUFFER_H
