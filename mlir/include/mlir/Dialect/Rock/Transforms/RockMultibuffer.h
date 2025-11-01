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
LogicalResult multiBuffer(RewriterBase &rewriter, rock::GpuAllocOp allocOp,
                          SmallVectorImpl<rock::GpuAllocOp> &newAllocs,
                          unsigned multiBufferingFactor,
                          bool skipOverrideAnalysis);

FailureOr<SmallVector<rock::GpuAllocOp>>
multiBuffer(rock::GpuAllocOp allocOp,
            SmallVectorImpl<rock::GpuAllocOp> &newAllocs,
            unsigned multiBufferingFactor, bool skipOverrideAnalysis);

LogicalResult updateMultiBuffer(RewriterBase &rewriter, Location loc,
                                ArrayRef<rock::GpuAllocOp> allocs,
                                SmallVectorImpl<rock::GpuAllocOp> &newAllocs,
                                unsigned newMultiBufferingFactor);

LogicalResult updateMultiBuffer(ArrayRef<rock::GpuAllocOp> allocs,
                                SmallVectorImpl<rock::GpuAllocOp> &newAllocs,
                                unsigned newMultiBufferingFactor);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TRANSFORMS_MULTIBUFFER_H
