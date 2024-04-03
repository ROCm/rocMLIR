//===- fusionUtils.h - Rock utility for fusion -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//
#ifndef ROCK_UTILITY_FISION_H
#define ROCK_UTILITY_FISION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace rock {
bool testFusibility(ModuleOp mod);
bool testFusibility(func::FuncOp func);
} // end namespace rock
} // end namespace mlir

#endif // ROCK_UTILITY_FISION_H
