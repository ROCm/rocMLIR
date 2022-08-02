//===- MIOpen.h - MIOpen MLIR Dialect ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MIOpen memref attributes and operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MIOPENOPS_OPS_H_
#define MLIR_MIOPENOPS_OPS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
class PatternRewriter;
} // namespace mlir

//===----------------------------------------------------------------------===//
//  MIOpen Dialect
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/MIOpen/MIOpenOpsDialect.h.inc"
#include "mlir/Dialect/MIOpen/MIOpenTypes.h.inc"

namespace mlir {
namespace miopen {
//===----------------------------------------------------------------------===//
// Utility method for creating an array attribute of n empty array attributes.
// We use this structure so transforms can be uniformly copied onto the final
// user(s) of the transformed value
//
// TODO(kdrewnia) See if this declaration should be elsewhere
//===----------------------------------------------------------------------===//
ArrayAttr noTransformsArray(Builder &b, size_t n);

ArrayAttr getIndexArrayAttr(Builder &b, ArrayRef<int64_t> values);
} // end namespace miopen
} // end namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenOps.h.inc"

namespace mlir {
namespace miopen {
TransformAttr getTransformAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    MLIRContext *context, TransformType type, ArrayRef<int64_t> params,
    ArrayRef<StringRef> upperNames, ArrayRef<uint32_t> upperDims,
    ArrayRef<StringRef> lowerNames, ArrayRef<uint32_t> lowerDims);

TransformMapAttr getTransformMapAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    MLIRContext *context, ArrayRef<TransformAttr> ops, AffineMapAttr map,
    ArrayRef<int64_t> upperBounds, ArrayRef<int64_t> lowerBounds);

} // namespace miopen
} // namespace mlir
#endif // MLIR_MIOPENOPS_OPS_H_
