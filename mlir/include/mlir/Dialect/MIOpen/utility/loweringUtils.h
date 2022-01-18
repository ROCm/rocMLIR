//===- loweringUtil.h - functions that often come up during lowering or turing
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MIOPEN_LOWERING_UTIL_H
#define MIOPEN_LOWERING_UTIL_H

#include "mlir/Dialect/MIOpen/MIOpen.h"

#include "mlir/Dialect/MIOpen/utility/math.h"

using namespace mlir;
using namespace mlir::miopen;

namespace mlir {
namespace miopen {
inline int64_t calculateKBlockNum(int64_t n, int64_t ho, int64_t wo) {
  int64_t gemmK = n * ho * wo;
  int64_t gemmKBlocks = 1;
  if (gemmK % 16 == 0) {
    auto lcm = math_util::lcm(ho * wo, (int64_t)16);
    gemmKBlocks = std::min(gemmK / lcm, n);
  } else if (gemmK % 8 == 0) {
    auto comm = math_util::lcm(ho * wo, (int64_t)8);
    gemmKBlocks = std::min(gemmK / comm, n);
  } else if (gemmK % 4 == 0) {
    auto comm = math_util::lcm(ho * wo, (int64_t)4);
    gemmKBlocks = std::min(gemmK / comm, n);
  }
  // not more than n
  gemmKBlocks = std::min(n, gemmKBlocks);
  // not less than 1
  gemmKBlocks = std::max((__int64_t)1, gemmKBlocks);

  // llvm::errs() << "\n gemmKBlocks: " << gemmKBlocks << " gemmK: " << gemmK
  //               << " ho: " << ho << " wo: " << wo << "\n";
  return gemmKBlocks;
}

//===----------------------------------------------------------------------===//
// Utility function to emit constant float op. Returns a scalar.
//===----------------------------------------------------------------------===//
inline Value createConstantFloatOp(OpBuilder &b, Location loc, Type elementType,
                                   float value) {
  Value ret;
  if (elementType == b.getF32Type()) {
    ret = b.create<arith::ConstantFloatOp>(loc, APFloat(value), b.getF32Type());
  } else if (elementType == b.getF16Type()) {
    bool lossy = false;
    APFloat constant(value);
    constant.convert(APFloat::IEEEhalf(), llvm::RoundingMode::TowardZero,
                     &lossy);
    ret = b.create<arith::ConstantFloatOp>(loc, constant, b.getF16Type());
  } else if (elementType == b.getIntegerType(16)) {
    ret = b.create<arith::ConstantIntOp>(loc, static_cast<int>(value),
                                         b.getIntegerType(16));
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// Utility function to emit constant zero op. Can return scalars or vectors.
//===----------------------------------------------------------------------===//
inline Value createZeroConstantFloatOp(OpBuilder &b, Location loc, Type type) {
  Type elementType = type;
  if (type.isa<VectorType>())
    elementType = type.template cast<VectorType>().getElementType();
  auto semantics = static_cast<APFloat::Semantics>(-1);
  if (elementType == b.getF32Type()) {
    semantics = APFloat::S_IEEEsingle;
  } else if (elementType == b.getF16Type()) {
    semantics = APFloat::S_IEEEhalf;
  } else if (elementType == b.getIntegerType(16)) {
    semantics = APFloat::S_BFloat;
  } else {
    llvm_unreachable("Unexpected float semantics");
  }

  auto zero = APFloat::getZero(APFloat::EnumToSemantics(semantics));
  Value retValue;

  if (auto vecType = type.dyn_cast<VectorType>()) {
    Attribute constValue;
    if (auto intType = elementType.dyn_cast<IntegerType>()) {
      auto intZero = zero.bitcastToAPInt();
      assert(intType.getIntOrFloatBitWidth() == intZero.getBitWidth());
      constValue = b.getIntegerAttr(elementType, intZero);
    } else {
      constValue = b.getFloatAttr(elementType, zero);
    }
    llvm::SmallVector<Attribute> constValues;
    std::fill_n(std::back_inserter(constValues), vecType.getNumElements(),
                constValue);
    retValue = b.create<mlir::ConstantOp>(
        loc, DenseElementsAttr::get(vecType, constValues), type);
  } else {
    if (auto intType = elementType.dyn_cast<IntegerType>()) {
      auto intZero = zero.bitcastToAPInt();
      assert(intType.getIntOrFloatBitWidth() == intZero.getBitWidth());
      retValue = b.create<mlir::ConstantOp>(
          loc, b.getIntegerAttr(intType, intZero), type);
    } else {
      retValue = b.create<mlir::ConstantOp>(
          loc, b.getFloatAttr(elementType, zero), type);
    }
  }
  return retValue;
}
} // end namespace miopen
} // end namespace mlir
#endif
