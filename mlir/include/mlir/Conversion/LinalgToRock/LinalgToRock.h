//==- LinalgToRock.h - Linalg conversion to Rock pass declarations -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the Linalg to Rock Dialect conversion
// in MLIR.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_LINALGTOROCK_H
#define MLIR_CONVERSION_LINALGTOROCK_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGTOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace rock {
void populateLinalgToRockConversionPattern(RewritePatternSet &pattern,
                                           MLIRContext *context);

/// A tensor.insert_slice is said to be a rock.expand_strides
bool isRockExpandStride(tensor::InsertSliceOp op);
} // namespace rock
} // namespace mlir

#endif
