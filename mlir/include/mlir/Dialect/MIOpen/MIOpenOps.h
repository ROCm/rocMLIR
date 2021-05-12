//===- MIOpenOps.h - MIOpen MLIR Operations ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MIOpen memref operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MIOPENOPS_OPS_H_
#define MLIR_MIOPENOPS_OPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

namespace miopen {

enum ConvOpType { Conv2DOpType, Conv2DBwdDataOpType, Conv2DBwdWeightOpType };
} // end namespace miopen

#include "mlir/Dialect/MIOpen/MIOpenOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenOps.h.inc"

} // end namespace mlir

#endif // MLIR_MIOPENOPS_OPS_H_
