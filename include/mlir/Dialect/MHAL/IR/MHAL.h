//===- MHAL.h - MHAL MLIR Dialect ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MHAL attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MHAL_IR_MHAL_H_
#define MLIR_MHAL_IR_MHAL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"

//===----------------------------------------------------------------------===//
//  MHAL Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MHAL/IR/MHALOpsDialect.h.inc"
#include "mlir/Dialect/MHAL/IR/MHALTypes.h"
#define GET_OP_CLASSES
#include "mlir/Dialect/MHAL/IR/MHALOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MHAL/IR/MHALAttrDefs.h.inc"

#endif // MLIR_MHAL_IR_MHAL_H_
