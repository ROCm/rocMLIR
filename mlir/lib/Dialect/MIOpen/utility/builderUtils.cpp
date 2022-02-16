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
Value createConstantIntOp(OpBuilder &b, Location loc, Type type,
                          Type elementType, int32_t value) {
  APInt apValue(elementType.getIntOrFloatBitWidth(), value, true);
  Attribute constValue = b.getIntegerAttr(elementType, apValue);

  Value retValue;
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    retValue =
        b.create<ConstantOp>(loc, SplatElementsAttr::get(shapedType, value));
  } else {
    retValue = b.create<ConstantOp>(loc, constValue, type);
  }

  return retValue;
}

} // anonymous namespace

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

  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    Attribute constValue = b.getFloatAttr(elementType, apValue);
    retValue = b.create<ConstantOp>(
        loc, SplatElementsAttr::get(shapedType, constValue));
  } else {
    retValue =
        b.create<ConstantOp>(loc, b.getFloatAttr(elementType, value), type);
  }

  return retValue;
}

Value createZeroConstantOp(OpBuilder &b, Location loc, Type type) {
  Type elementType = type;
  if (auto shaped = type.dyn_cast<ShapedType>())
    elementType = shaped.getElementType();

  if (elementType.isIntOrIndex()) {
    return createConstantIntOp(b, loc, type, elementType, 0);
  } else {
    return createConstantFloatOp(b, loc, type, elementType, 0.0);
  }
}

} // namespace miopen
} // namespace mlir
