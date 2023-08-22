//===- MIGraphX.h - MIGraphX MLIR Dialect --------------*- C++-* -===//
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

#ifndef MLIR_MIGRAPHX_IR_MIGRAPHX_H_
#define MLIR_MIGRAPHX_IR_MIGRAPHX_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace migraphx {} // end namespace migraphx
} // end namespace mlir

#include "mlir/Dialect/MIGraphX/IR/MIGraphXDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MIGraphX/IR/MIGraphXTypes.h.inc"

#include "mlir/Dialect/MIGraphX/IR/MIGraphXEnums.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h.inc"

#endif // MLIR_MIGRAPHX_IR_MIGRAPHX_H_
