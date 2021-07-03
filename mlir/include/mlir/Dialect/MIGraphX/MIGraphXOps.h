//===- MIGraphXOps.h - MIGraphX MLIR Operations ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MIGraphX operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MIGRAPHXOPS_OPS_H_
#define MLIR_MIGRAPHXOPS_OPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

namespace migraphx {


} // end namespace migraphx

#include "mlir/Dialect/MIGraphX/MIGraphXOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h.inc"

} // end namespace mlir

#endif // MLIR_MIGRAPHXOPS_OPS_H_
