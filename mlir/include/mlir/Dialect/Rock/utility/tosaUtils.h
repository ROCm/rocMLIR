//===- tosa Utility Functions  ------===//
//
// Copyright 2025 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
#ifndef MLIR_DIALECT_TOSA_UTILITY_H
#define MLIR_DIALECT_TOSA_UTILITY_H

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir {
namespace rock {
bool isSpecificValueAttribute(Attribute value, double target);
bool isConstantValue(Value v, double target);
bool isConstantZero(Value v);
bool isConstantOne(Value v);
bool isConstNegInf(Value v);
bool isConstRange(Value v);

template <typename Rewriter>
Value getTosaOneTensor(Rewriter &builder, Location loc, RankedTensorType type) {
  auto value = cast<ElementsAttr>(builder.getOneAttr(type));
  return tosa::ConstOp::create(builder, loc, type, value);
}

template <typename Rewriter>
Type getTosaAccType(Rewriter &rewriter, Type inputType) {
  Type accType;
  if (isa<FloatType>(inputType)) {
    accType = rewriter.getF32Type();
  } else if (isa<IntegerType>(inputType)) {
    accType = rewriter.getI32Type();
  } else {
    llvm_unreachable("not expected type");
  }
  return accType;
}

template <typename TosaOp, typename Rewriter, typename... Args>
TosaOp createTosaOpAndInfer(Rewriter &rewriter, Location loc, Type elemType,
                            Args &&...args) {
  auto op =
      TosaOp::create(rewriter, loc, UnrankedTensorType::get(elemType), args...);
  InferShapedTypeOpInterface shapeInterface =
      cast<InferShapedTypeOpInterface>(op.getOperation());
  SmallVector<ShapedTypeComponents> returnShape;
  LogicalResult shapeInferenceStatus = shapeInterface.inferReturnTypeComponents(
      op.getContext(), op.getLoc(), op->getOperands(), op->getAttrDictionary(),
      op->getPropertiesStorage(), op->getRegions(), returnShape);
  assert(shapeInferenceStatus.succeeded());
  Type newOutTy = RankedTensorType::get({returnShape[0].getDims()}, elemType);
  auto result = op->getResult(0);
  result.setType(newOutTy);
  return op;
}

template <typename Rewriter>
Value getTosaZeroTensor(Location loc, RankedTensorType type,
                        Rewriter &rewriter) {
  auto value = cast<ElementsAttr>(rewriter.getZeroAttr(type));
  return tosa::ConstOp::create(rewriter, loc, type, value);
}

template <typename Rewriter>
tosa::TransposeOp getTosaTransposeOp(Rewriter &rewriter, Location loc,
                                     Value input,
                                     ArrayRef<int32_t> permutation) {
  ShapedType inputTy = cast<ShapedType>(input.getType());
  auto inputShape = inputTy.getShape();
  SmallVector<int64_t> newShape;
  newShape.reserve(permutation.size());
  for (int32_t fromIdx : permutation)
    newShape.push_back(inputShape[fromIdx]);
  Type newTy = RankedTensorType::get(newShape, inputTy.getElementType());

  auto newOp =
      tosa::TransposeOp::create(rewriter, loc, newTy, input, permutation);
  return newOp;
}

template <typename Rewriter>
tosa::MulOp getTosaMulOp(Rewriter &rewriter, Location loc, Value input1,
                         Value input2, Type elemType) {
  auto shiftType = RankedTensorType::get({1}, rewriter.getIntegerType(8));
  elemType = getElementTypeOrSelf(elemType);
  auto shiftZeroAttr = DenseElementsAttr::get(
      shiftType, rewriter.getZeroAttr(rewriter.getIntegerType(8)));
  Value constZero =
      tosa::ConstOp::create(rewriter, loc, shiftType, shiftZeroAttr);
  auto mulOp = createTosaOpAndInfer<tosa::MulOp>(rewriter, loc, elemType,
                                                 input1, input2, constZero);
  return mulOp;
}

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_UTILITY_H