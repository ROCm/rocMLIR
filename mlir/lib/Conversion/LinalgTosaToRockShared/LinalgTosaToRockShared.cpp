//===- LinalgTosaToRockShared.cpp - Shared utilities for *ToRock ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgTosaToRockShared/LinalgTosaToRockShared.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace mlir {
namespace rock {

Value traceToRes(Value tensor, DenseMap<Value, Value> &cache,
                 Value expectedTensor) {
  if (cache.contains(tensor))
    return cache.at(tensor);

  Value res = nullptr;
  if (tensor.getDefiningOp()) {
    if (expectedTensor == tensor) {
      res = tensor;
    } else if (auto view = tensor.getDefiningOp<ViewLikeOpInterface>()) {
      res = traceToRes(view.getViewSource(), cache, expectedTensor);
    } else if (auto expand = tensor.getDefiningOp<tensor::ExpandShapeOp>()) {
      res = traceToRes(expand.getSrc(), cache, expectedTensor);
    } else if (auto collapse =
                   tensor.getDefiningOp<tensor::CollapseShapeOp>()) {
      res = traceToRes(collapse.getSrc(), cache, expectedTensor);
    } else if (auto untransform =
                   tensor.getDefiningOp<TensorUntransformCastOp>()) {
      res =
          traceToRes(untransform.getTransformedResult(), cache, expectedTensor);
    } else if (auto tosaOp = tensor.getDefiningOp<tosa::TosaOp>()) {
      for (auto operand : tosaOp->getOperands()) {
        if (llvm::isa<TensorType>(operand.getType())) {
          res = traceToRes(operand, cache, expectedTensor);
          if (res)
            break;
        }
      }
    } else if (auto linalgOp = tensor.getDefiningOp<linalg::LinalgOp>()) {
      for (auto operand : linalgOp->getOperands()) {
        if (llvm::isa<TensorType>(operand.getType())) {
          res = traceToRes(operand, cache, expectedTensor);
          if (res)
            break;
        }
      }
    }
  }

  cache.insert({tensor, res});
  return res;
}

SetVector<int64_t> traceToRes(Value expectedTensor, func::FuncOp func) {
  llvm::DenseMap<Value, Value> cache;

  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
  assert(returns.size() == 1 && "Number of returns is not one");
  func::ReturnOp returnOp = returns[0];

  SetVector<int64_t> resIndices;
  for (auto [i, res] : llvm::enumerate(returnOp->getOperands())) {
    Value out = traceToRes(res, cache, expectedTensor);
    if (out == expectedTensor)
      resIndices.insert(i);
  }
  return resIndices;
}

void addZeroInitPrefillAttribute(Operation *op, ArrayRef<int64_t> strideDims,
                                 ArrayRef<int64_t> dilationDims,
                                 ArrayRef<int64_t> filterDims) {
  if (isEveryElementWrittenBwdData(strideDims, dilationDims, filterDims))
    return;

  Value output = op->getResult(0);
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return;

  SetVector<int64_t> resIndices = traceToRes(output, func);
  if (resIndices.empty())
    assert(false &&
           "Output of TransposeConv2D op cannot be traced to result index");

  OpBuilder builder(op->getContext());
  for (int64_t resNumber : resIndices) {
    Type funcResType = func.getFunctionType().getResult(resNumber);
    auto shapedResType = cast<ShapedType>(funcResType);
    Type elementType = shapedResType.getElementType();

    Attribute outputInitVal;
    if (isa<FloatType>(elementType)) {
      outputInitVal = builder.getFloatAttr(elementType, 0.0);
    } else if (isa<IntegerType>(elementType)) {
      outputInitVal = builder.getIntegerAttr(elementType, 0);
    } else {
      assert(false && "Unsupported element type for prefill attribute");
    }

    func.setResultAttr(resNumber, PrefillAttr::getMnemonic(), outputInitVal);
  }
}

} // namespace rock
} // namespace mlir
