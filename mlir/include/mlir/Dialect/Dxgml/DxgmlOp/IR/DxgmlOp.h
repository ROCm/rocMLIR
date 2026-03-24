//===- DxgmlOp.h - DxgmlOp dialect ------------------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DXGML_DXGMLOP_IR_DXGMLOP_H
#define MLIR_DIALECT_DXGML_DXGMLOP_IR_DXGMLOP_H

#include "mlir/Dialect/Dxgml/IR/Dxgml.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// DxgmlOp Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpDialect.h.inc"

//===----------------------------------------------------------------------===//
// DxgmlOp Enums
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpEnums.h.inc"

//===----------------------------------------------------------------------===//
// DxgmlOp Base Attributes (enum wrapper AttrDefs from DxgmlOpBase.td)
// Must be included before DxgmlOp.h.inc which uses these types.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DXGMLOP_BASE_ATTRS_INCLUDED
#define MLIR_DXGMLOP_BASE_ATTRS_INCLUDED
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOpBaseAttrs.h.inc"
#endif // MLIR_DXGMLOP_BASE_ATTRS_INCLUDED

//===----------------------------------------------------------------------===//
// DxgmlOp Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOp.h.inc"

#endif // MLIR_DIALECT_DXGML_DXGMLOP_IR_DXGMLOP_H
