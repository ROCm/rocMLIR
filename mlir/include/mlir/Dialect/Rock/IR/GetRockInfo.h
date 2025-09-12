//===- GetRockInfo.h - functions used to calculate information about Rock ops
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_ROCK_IR_GETROCKINFO_H
#define MLIR_DIALECT_ROCK_IR_GETROCKINFO_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
class Operation;
class Type;

namespace rock {

// This function returns the func or gpu.func of a given op
Operation *getParentFuncOp(Operation *op);

// Return a boolean if the features contain accel properties
bool isAccel(rock::GemmFeatures features);

// Get the arch from the op
FailureOr<StringAttr> getArch(Operation *op);

// Get the arch from the op and error out if it cannot be found
StringAttr getArchValue(Operation *op);

// Get the num_cu from the op
FailureOr<int64_t> getNumCU(Operation *op);

// Get the num_cu from the op, and error out if it cannot be found
int64_t getNumCUValue(Operation *op);

inline rock::GemmFeatures intersectGemmFeatures(rock::GemmFeatures a,
                                                rock::GemmFeatures b) {
  return a & b;
}

// Get the features enabled for the specified op. These will be dependent on
// the architecture being used, and the type of the op.
rock::GemmFeatures getFeatures(Operation *op);

} // End namespace rock
} // End namespace mlir
#endif // MLIR_DIALECT_ROCK_IR_GETROCKINFO_H
