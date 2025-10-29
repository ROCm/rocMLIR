//===- tosa Utility Functions -===//
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
#ifndef MLIR_DIALECT_ROCK_TOSA_UTILITY_H
#define MLIR_DIALECT_ROCK_TOSA_UTILITY_H

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
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

namespace tosa {
template <typename TosaOp, typename... Args>
TosaOp createOpAndInfer(OpBuilder &rewriter, Location loc, Type elemType,
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

Value getOneTensor(OpBuilder &builder, Location loc, RankedTensorType type);

Type getAccType(OpBuilder &builder, Type inputType);

Value getZeroTensor(OpBuilder &builder, Location loc, RankedTensorType type);

mlir::tosa::TransposeOp getTransposeOp(OpBuilder &builder, Location loc,
                                       Value input,
                                       ArrayRef<int32_t> permutation);

mlir::tosa::MulOp getMulOp(OpBuilder &builder, Location loc, Value input1,
                           Value input2, Type elemType);
} // namespace tosa
} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TOSA_UTILITY_H
