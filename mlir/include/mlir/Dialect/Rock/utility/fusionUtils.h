//===- fusionUtils.h - Rock utility for fusion -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//
#ifndef ROCK_UTILITY_FUSION_H
#define ROCK_UTILITY_FUSION_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace memref {
class AllocOp;
} // namespace memref

namespace func {
class FuncOp;
} // namespace func

namespace rock {
// Checks whether a function is valid for split-k.
LogicalResult testFusionLegalitySplitK(func::FuncOp func);

// Checks whether a function contains any `rock::ReduceOp` and
// the atomic operation is supported by the hardware.
LogicalResult testFusionLegalityReduce(func::FuncOp func);

// Checks whether a function contains any `rock::BwdDataConv` ops, and if they
// are using the v4r1 algorithm (for splitting)
LogicalResult testFusionLegalityBwdDataConv(func::FuncOp func);

// This is an overload of the `testFusionLegalitySplitK` which is more
// convenient to use in CAPI. Given a `ModuleOp`, the function retrieve the
// embedded `func:FuncOp` and calls the implementation
// `testFusionLegalitySplitK` (see above). Note, this overloaded function
// assumes that `ModuleOp` contains a single `func:FuncOp`
LogicalResult testFusionLegalitySplitK(ModuleOp mod);

// Same as above, overload of `testFusionLegalityReduce` for `ModuleOp`.
LogicalResult testFusionLegalityReduce(ModuleOp mod);

// Same as above, overload of `testFusionLegalityBwdDataConv` for `ModuleOp`.
LogicalResult testFusionLegalityBwdDataConv(ModuleOp mod);

// Checks whether the output fusion linalg::GenericOp is valid. Assuming a
// split-k kernel.
LogicalResult
checkValidOutputFusion(linalg::GenericOp genericOp, Value gemmResult,
                       GemmFeatures features,
                       SmallVector<std::tuple<Operation *, int>> &adds);

// Checks whether an operation is a valid elementwise operation for GEMM output
// fusion (used for both split-K and reduction fusion analysis).
bool validOperationGemmOut(Operation &op);

} // end namespace rock
} // end namespace mlir

#endif // ROCK_UTILITY_FUSION_H
