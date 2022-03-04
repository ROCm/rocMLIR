//===- utilities.cpp - MIOpen utility functions ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/MIOpen/utility/builderUtils.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using mlir::arith::ConstantOp;

namespace mlir {
namespace miopen {
namespace {} // anonymous namespace

Value createConstantIntOp(OpBuilder &b, Location loc, Type type,
                          Type elementType, int64_t value) {
  APInt apValue(elementType.getIntOrFloatBitWidth(), value, true);
  Attribute constValue = b.getIntegerAttr(elementType, apValue);

  Value retValue;
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    retValue = b.create<ConstantOp>(
        loc, SplatElementsAttr::get(shapedType, constValue));
  } else {
    retValue = b.create<ConstantOp>(loc, constValue, type);
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
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isIntOrIndex()) {
    return createConstantIntOp(b, loc, type, elementType, 0);
  } else {
    return createConstantFloatOp(b, loc, type, elementType, 0.0);
  }
}

//===----------------------------------------------------------------------===//
// Utility function to emit type conversion ops.
//===----------------------------------------------------------------------===//
Value createTypeConversionOp(OpBuilder &b, Location loc, Value source,
                             Type destType) {
  // Convert from sourceType to destType if necessary.
  Value result = source;
  Type sourceType = source.getType();
  if (auto sourceVec = sourceType.dyn_cast<VectorType>()) {
    if (auto destVec = destType.dyn_cast<VectorType>()) {
      assert(sourceVec.getNumElements() == destVec.getNumElements() &&
             "source and destinatioon have same length");
    } else {
      llvm_unreachable("Can't store vector sources to scalar destinations in "
                       "output writeback");
    }
  }
  Type sourceElemType = getElementTypeOrSelf(sourceType);
  Type destElemType = getElementTypeOrSelf(destType);
  if (sourceElemType != destElemType) {
    // Possible cases:
    // - fp16/bf16 -> fp32 : use fpext.
    // - fp32 -> fp16/bf16 : use fptrunc.
    // - fp16/fp32 -> bf16(i16) : use miopen.data_convert.
    // All these ops act elementwise on vectors
    if (sourceElemType.getIntOrFloatBitWidth() == 16 &&
        destElemType == b.getF32Type()) {
      result = b.create<arith::ExtFOp>(loc, destType, source);
    } else if (sourceElemType == b.getF32Type() &&
               destElemType.getIntOrFloatBitWidth() == 16) {
      result = b.create<arith::TruncFOp>(loc, destType, source);
    } else {
      llvm_unreachable("Only fp32, fp16, or bf16 targets for data conversion");
    }
  }
  return result;
}

Value createCollapseShapeOp(OpBuilder &b, Location loc, Value source) {
  auto ctx = b.getContext();
  auto sourceType = source.getType().cast<ShapedType>();
  assert(sourceType.hasStaticShape() &&
         "Only memrefs with static shapes are allowed");

  auto shape = sourceType.getShape();
  uint64_t collapsedDim = 1;
  SmallVector<AffineExpr, 2> exprs;
  for (uint32_t dim = 0; dim < shape.size(); ++dim) {
    collapsedDim *= shape[dim];
    exprs.push_back(getAffineDimExpr(dim, ctx));
  }

  SmallVector<int64_t, 1> collapsedShape;
  SmallVector<ReassociationExprs, 1> reassocs;
  collapsedShape.push_back(collapsedDim);
  reassocs.push_back(exprs);

  auto collapsedType =
      MemRefType::get(collapsedShape, sourceType.getElementType());
  Value result =
      b.create<memref::CollapseShapeOp>(loc, collapsedType, source, reassocs);
  return result;
}

} // namespace miopen
} // namespace mlir
