//===- MIGraphX.cpp - MIGraphX MLIR Operations
//-----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SMLoc.h"

#include "mlir/Dialect/MIGraphX/IR/MIGraphXDialect.cpp.inc"

#include "mlir/Dialect/MIGraphX/IR/MIGraphXEnums.cpp.inc"

#define DEBUG_TYPE "migraphx"

using namespace mlir;
using namespace mlir::migraphx;

//===----------------------------------------------------------------------===//
// MIGraphXDialect
//===----------------------------------------------------------------------===//

void MIGraphXDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/MIGraphX/IR/MIGraphXTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MIGraphX/IR/MIGraphXTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.cpp.inc"

Operation *MIGraphXDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  if (!isa<MIXRShapedType>(type))
    return nullptr;
  ElementsAttr elemsValue = dyn_cast<ElementsAttr>(value);
  if (!elemsValue)
    return nullptr;
  return builder.create<LiteralOp>(loc, type, elemsValue);
}

//===----------------------------------------------------------------------===//
// MIXRShapedType
//===----------------------------------------------------------------------===//

LogicalResult
MIXRShapedType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
                       Type elementType) {
  if (shape.size() != strides.size())
    return emitError() << "migraphx.shaped type has " << shape.size()
                       << " elements in its shape but " << strides.size()
                       << " strides defined";
  if (!TensorType::isValidElementType(elementType))
    return emitError() << "cannot put the type " << elementType
                       << " into a migraphx.shaped type";
  return success();
}

Type MIXRShapedType::parse(AsmParser &parser) {
  llvm::SMLoc currentLoc = parser.getCurrentLocation();
  SmallVector<int64_t, 4> shape;
  Type elementType;
  SmallVector<int64_t, 4> strides;
  if (parser.parseLess() || parser.parseDimensionList(shape) ||
      parser.parseType(elementType)) {
    parser.emitError(currentLoc, "expected shaped dimension list with type");
    return Type();
  }
  currentLoc = parser.getCurrentLocation();
  if (!shape.empty()) {
    if (parser.parseComma() ||
        parser.parseDimensionList(strides, /*allowDynamic=*/true,
                                  /*withTrailingX=*/false)) {
      parser.emitError(currentLoc, "expected `,` and a `x`-separated list in "
                                   "non-scalar migraphx.shaped type");
      return Type();
    }
  }
  currentLoc = parser.getCurrentLocation();
  if (parser.parseGreater()) {
    parser.emitError(currentLoc, "expected `>`");
    return Type();
  }
  return get(shape, strides, elementType);
}

void MIXRShapedType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape()) {
    if (ShapedType::isDynamic(dim))
      printer << "?";
    else
      printer << dim;
    printer << "x";
  }
  printer.printType(getElementType());
  ArrayRef<int64_t> strides = getStrides();
  if (!strides.empty()) {
    printer << ", ";
    for (size_t i = 0, e = strides.size(); i < e; ++i) {
      int64_t stride = strides[i];
      if (ShapedType::isDynamic(stride))
        printer << "?";
      else
        printer << stride;

      if (i + 1 != e)
        printer << "x";
    }
  }
  printer << ">";
}

MIXRShapedType
MIXRShapedType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                          std::optional<ArrayRef<int64_t>> strides,
                          Type elementType) const {
  return get(shape ? *shape : getShape(), strides ? *strides : getStrides(),
             elementType ? elementType : getElementType());
}

ShapedType MIXRShapedType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                     Type elementType) const {
  return cloneWith(shape, std::nullopt, elementType);
}

RankedTensorType MIXRShapedType::asTensor() const {
  return RankedTensorType::get(getShape(), getElementType());
}

bool MIXRShapedType::isStandard() const {
  ArrayRef<int64_t> strides = getStrides();
  if (strides.empty())
    return true;
  if (strides.size() == 1 && strides[0] == 0 && getShape()[0] == 1)
    return true;
  return llvm::is_sorted(llvm::reverse(strides)) &&
         llvm::is_contained(strides, 1);
}

bool MIXRShapedType::hasBroadcast() const {
  return llvm::any_of(getStrides(), [](int64_t s) { return s == 0; });
}

void MIXRShapedType::getBroadcastDims(SmallVectorImpl<uint32_t> &result) const {
  for (auto [i, val] : llvm::enumerate(getStrides())) {
    if (val == 0)
      result.emplace_back(val);
  }
}

bool MIXRShapedType::hasRank() const { return true; }

RankedTensorType MIXRShapedType::asMemoryLayoutTensor() const {
  ArrayRef<int64_t> shape = getShape();
  ArrayRef<int64_t> strides = getStrides();

  size_t nStrides = strides.size();
  SmallVector<int64_t> stridesToStandardPerm;
  getStridePermutation(stridesToStandardPerm);
  SmallVector<int64_t, 4> orderedShape;
  SmallVector<int64_t, 4> orderedStrides;
  orderedShape.resize_for_overwrite(nStrides);
  orderedStrides.resize_for_overwrite(nStrides);
  for (auto [to, from] : llvm::enumerate(stridesToStandardPerm)) {
    orderedShape[to] = shape[from];
    orderedStrides[to] = strides[from];
    // Broadcasts become a length-1 dimension
    if (strides[from] == 0)
      orderedShape[to] = 1;
  }
  // Ensure we have a unit stride.
  for (auto stride : llvm::reverse(orderedStrides)) {
    if (stride == 0)
      continue;
    if (stride == 1)
      break;
    emitError(UnknownLoc::get(getContext()),
              "!migraphx.shaped type with smallest stride " + Twine(stride) +
                  " has no supported in-memory layout");
    return nullptr;
  }
  // Check for the case where we're taking slices.
  for (auto [idx, stride] : llvm::enumerate(orderedStrides)) {
    // We can stop checking after we've hit the fastest-moving dimension
    if (stride == 1)
      break;
    // Broadcasts aren't subject to slice checking
    if (stride == 0)
      continue;

    // Get the stride of the previous dimension, ignoring broadcast dims.
    size_t prevIdx = idx + 1;
    while (orderedStrides[prevIdx] == 0)
      prevIdx += 1;
    int64_t prevStride = orderedStrides[prevIdx];

    int64_t expectedStride = prevStride * orderedShape[prevIdx];
    if (stride < expectedStride) {
      emitError(
          UnknownLoc::get(getContext()),
          "!migraphx.shaped type can't be laid out in memory when the stride " +
              Twine(stride) + " at index " + Twine(idx) +
              " being smaller than the product of previous lengths " +
              Twine(expectedStride));
      return nullptr;
    }
    if (stride > expectedStride) {
      if (stride % prevStride != 0) {
        emitError(UnknownLoc::get(getContext()),
                  "!migraphx.shaped type can't be laid out in memory when the "
                  "stride " +
                      Twine(stride) + " at index " + Twine(idx) +
                      " does not evenly divide the previous stride " +
                      Twine(prevStride));
        return nullptr;
      }
      orderedShape[prevIdx] = stride / prevStride;
    }
  }
  return RankedTensorType::get(orderedShape, getElementType());
}

RankedTensorType MIXRShapedType::asFlatMemoryTensor() const {
  RankedTensorType memoryTensorType = asMemoryLayoutTensor();
  return memoryTensorType.clone(memoryTensorType.getNumElements());
}

void MIXRShapedType::getStridePermutation(SmallVectorImpl<int64_t> &ret) const {
  ArrayRef<int64_t> shape = getShape();
  ArrayRef<int64_t> strides = getStrides();
  size_t n = strides.size();
  ret.clear();
  ret.reserve(n);
  llvm::append_range(ret, llvm::iota_range<int64_t>(0, n, /*Inclusive=*/false));
  llvm::stable_sort(ret, [&](auto a, auto b) {
    return std::make_tuple(strides[a], shape[a]) >
           std::make_tuple(strides[b], shape[b]);
  });
  LLVM_DEBUG({
    llvm::dbgs() << "Found migraphx shaped type stride permutation: ";
    llvm::interleaveComma(ret, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

//===----------------------------------------------------------------------===//
// MIGraphXOps
//===----------------------------------------------------------------------===//

OpFoldResult LiteralOp::fold(FoldAdaptor adaptor) { return getValue(); }

OpFoldResult RecipOp::fold(FoldAdaptor operands) {
  // 1/(1/x) = x
  if (auto parentRecip = getInA().getDefiningOp<RecipOp>()) {
    return parentRecip.getInA();
  }
  return {};
}

LogicalResult LiteralOp::verify() {
  MIXRShapedType type = getResult().getType();
  ElementsAttr value = getValue();
  if (!value.isSplat()) {
    if (value.getType() != type.asTensor())
      return emitOpError("non-splat literals must have a value that matches "
                         "the literal's logical shape");
    int64_t expectedStride = 1;
    for (auto [len, stride] : llvm::zip(llvm::reverse(type.getShape()),
                                        llvm::reverse(type.getStrides()))) {
      if (stride != expectedStride)
        return emitOpError(
            "strides of non-splat literal are not in standard shape");
      expectedStride *= len;
    }
  }
  return success();
}
