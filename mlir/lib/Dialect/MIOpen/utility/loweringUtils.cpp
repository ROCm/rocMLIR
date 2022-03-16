//===- builderUtils.cpp - MIOpen utility functions ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"
#include "mlir/Dialect/MIOpen/MIOpenOpTraits.h"

namespace mlir {
namespace miopen {
std::tuple<Value, ArrayAttr> untransform(OpBuilder &b, Value transformed,
                                         ArrayAttr existing) {
  SmallVector<Attribute> transformList;
  if (existing)
    transformList.append(existing.begin(), existing.end());
  Value ret = transformed;
  while (auto transform = dyn_cast_or_null<TransformOp>(ret.getDefiningOp())) {
    llvm::copy(transform.transforms(), std::back_inserter(transformList));
    ret = transform.input();
  }
  return {ret, b.getArrayAttr(transformList)};
}

static llvm::StringMap<int64_t> canonicalizeDims(ArrayRef<int64_t> dims,
                                                 ArrayAttr layout) {
  llvm::StringMap<int64_t> result;
  for (auto tuple : llvm::zip(layout.getValue(), dims)) {
    StringRef key = std::get<0>(tuple).cast<StringAttr>().getValue();
    result[key] = std::get<1>(tuple);
  }
  return result;
}

std::tuple<llvm::StringMap<int64_t>, llvm::StringMap<int64_t>,
           llvm::StringMap<int64_t>>
fetchDimensions(Operation *op) {
  assert(op->hasTrait<OpTrait::miopen::ConvolutionOp>() &&
         "Op is not a convolution.");

  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  // Get shape of filter tensor.
  auto filterType = op->getOperand(0).getType().template cast<MemRefType>();
  auto filterShape = filterType.getShape();

  // Get shape of input tensor.
  auto inputType = op->getOperand(1).getType().template cast<MemRefType>();
  auto inputShape = inputType.getShape();

  // Get shape of output tensor.
  auto outputType = op->getOperand(2).getType().template cast<MemRefType>();
  auto outputShape = outputType.getShape();

  auto filterDim = canonicalizeDims(filterShape, filterLayoutAttr);
  auto inputDim = canonicalizeDims(inputShape, inputLayoutAttr);
  auto outputDim = canonicalizeDims(outputShape, outputLayoutAttr);

  return std::make_tuple(filterDim, inputDim, outputDim);
}
} // namespace miopen
} // namespace mlir
