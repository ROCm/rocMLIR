//===- fusionUtils.h - Rock utility for fusion -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//
#ifndef ROCK_UTILITY_FISION_H
#define ROCK_UTILITY_FISION_H

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
// Checks whether a function contains any `linalg::GenericOp` which
// reads or writes to the output of any `Operation` implementing
// `RockGemmWrapperInterface`. The result of this test can be ignored
// if the Data Parallel GEMM scheme is used.
LogicalResult testFusionLegality(func::FuncOp func);

// This is an overload of the `testFusionLegality` which is more convenient
// to use in CAPI. Given a `ModuleOp`, the function retrieve the embedded
// `func:FuncOp` and calls the implementation `testFusionLegality` (see above).
// Note, this overloaded function assumes that `ModuleOp` contains
// a single `func:FuncOp`
LogicalResult testFusionLegality(ModuleOp mod);
} // end namespace rock
} // end namespace mlir

#endif // ROCK_UTILITY_FISION_H
