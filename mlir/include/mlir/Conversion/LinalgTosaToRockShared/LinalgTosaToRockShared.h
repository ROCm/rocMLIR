//===- LinalgTosaToRockShared.h - Shared utilities for *ToRock --*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOSATOROCKSHARED_LINALGTOSATOROCKSHARED_H
#define MLIR_CONVERSION_LINALGTOSATOROCKSHARED_LINALGTOSATOROCKSHARED_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class Operation;

namespace rock {

Value traceToRes(Value tensor, DenseMap<Value, Value> &cache,
                 Value expectedTensor);

SetVector<int64_t> traceToRes(Value expectedTensor, func::FuncOp func);

void addZeroInitPrefillAttribute(Operation *op,
                                 llvm::ArrayRef<int64_t> strideDims,
                                 llvm::ArrayRef<int64_t> dilationDims,
                                 llvm::ArrayRef<int64_t> filterDims);

} // namespace rock
} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOSATOROCKSHARED_LINALGTOSATOROCKSHARED_H
