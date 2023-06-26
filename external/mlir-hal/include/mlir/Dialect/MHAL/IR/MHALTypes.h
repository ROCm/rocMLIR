//===- MHALTypes.h - MHAL Dialect Types -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the MHAL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MHAL_IR_MHALTYPES_H_
#define MLIR_DIALECT_MHAL_IR_MHALTYPES_H_

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// MHAL Dialect Types
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MHAL/IR/MHALTypes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MHAL/IR/MHALOpsTypes.h.inc"

#endif // MLIR_DIALECT_MHAL_IR_MHALTYPES_H_
