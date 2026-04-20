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
#include "llvm/ADT/SetVector.h"

using namespace mlir;

Value rock::traceToRes(Value tensor, DenseMap<Value, Value> &cache,
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
                   tensor.getDefiningOp<rock::TensorUntransformCastOp>()) {
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
    }
  }

  cache.insert({tensor, res});
  return res;
}

SetVector<int64_t> rock::traceToRes(Value expectedTensor,
                                    func::FuncOp func) {
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

void rock::addZeroInitPrefillAttribute(Operation *op,
                                       ArrayRef<int64_t> strideDims,
                                       ArrayRef<int64_t> dilationDims,
                                       ArrayRef<int64_t> filterDims) {
  // If there is no zeroinit kernel needed, then there is nothing more we need
  // to do here.
  if (rock::isEveryElementWrittenBwdData(strideDims, dilationDims, filterDims))
    return;

  // Now we need to determine where to add the prefill attributes. Trace through
  // the output of the TransposeConv2D op to find where the result is used.
  Value output = op->getResult(0);
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return;

  SetVector<int64_t> resIndices = traceToRes(output, func);
  // If the output cannot be traced to a result index, then we have a case that
  // we cannot yet handle
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
      // We only expect integer and float types for now
      assert(false && "Unsupported element type for prefill attribute");
    }

    func.setResultAttr(resNumber, rock::PrefillAttr::getMnemonic(),
                       outputInitVal);
  }
}
