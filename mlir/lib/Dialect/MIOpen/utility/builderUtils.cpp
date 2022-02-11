//===- utilities.cpp - MIOpen utility functions ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/MIOpen/utility/builderUtils.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using mlir::arith::ConstantOp;

namespace mlir {
namespace miopen {
namespace {
Value createConstantI8Op(OpBuilder &b, Location loc, Type type,
                         Type elementType, int8_t value) {
  Value retValue;
  if (auto vecType = type.dyn_cast<VectorType>()) {
    retValue =
        b.create<ConstantOp>(loc, SplatElementsAttr::get(vecType, value));
  } else {
    retValue = b.create<ConstantOp>(loc, b.getI8IntegerAttr(value), type);
  }

  return retValue;
}

Value createConstantFloatOp(OpBuilder &b, Location loc, Type type,
                            Type elementType, float value) {
  auto semantics = static_cast<APFloat::Semantics>(-1);
  if (elementType == b.getF32Type()) {
    semantics = APFloat::S_IEEEsingle;
  } else if (elementType == b.getF16Type()) {
    semantics = APFloat::S_IEEEhalf;
  } else if (elementType == b.getBF16Type()) {
    semantics = APFloat::S_BFloat;
  } else {
    llvm_unreachable("Unexpected float semantics");
  }

  APFloat apValue(value);
  bool lostInfo = false;
  apValue.convert(APFloat::EnumToSemantics(semantics),
                  APFloat::rmNearestTiesToEven, &lostInfo);
  Value retValue;

  if (auto vecType = type.dyn_cast<VectorType>()) {
    Attribute constValue = b.getFloatAttr(elementType, apValue);
    retValue =
        b.create<ConstantOp>(loc, SplatElementsAttr::get(vecType, constValue));
  } else {
    retValue =
        b.create<ConstantOp>(loc, b.getFloatAttr(elementType, value), type);
  }

  return retValue;
}
} // anonymous namespace

Value createZeroConstantOp(OpBuilder &b, Location loc, Type type) {
  Type elementType = type;
  if (type.isa<VectorType>())
    elementType = type.template cast<VectorType>().getElementType();

  if (elementType == b.getIntegerType(8)) {
    return createConstantI8Op(b, loc, type, elementType, 0);
  } else {
    return createConstantFloatOp(b, loc, type, elementType, 0.0);
  }
}

} // namespace miopen
} // namespace mlir
