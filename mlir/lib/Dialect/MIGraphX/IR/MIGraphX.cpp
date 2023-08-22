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
  if (!type.isa<MIXRShapedType>())
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

  SmallVector<int64_t> stridesToStandardPerm;
  getStridesToStandardShapePermutation(stridesToStandardPerm);
  SmallVector<int64_t, 4> orderedShape;
  SmallVector<int64_t, 4> orderedStrides;
  orderedShape.resize_for_overwrite(stridesToStandardPerm.size());
  orderedStrides.resize_for_overwrite(stridesToStandardPerm.size());
  for (auto [to, from] : llvm::enumerate(stridesToStandardPerm)) {
    orderedShape[to] = shape[from];
    orderedStrides[to] = strides[from];
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
  for (auto [idx, stride] :
       llvm::enumerate(ArrayRef<int64_t>(orderedStrides).drop_back())) {
    int64_t prevStride = orderedStrides[idx + 1];
    int64_t expectedStride = prevStride * orderedShape[idx + 1];
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
      orderedShape[idx + 1] = stride / expectedStride;
    }
  }
  return RankedTensorType::get(orderedShape, getElementType());
}

void MIXRShapedType::getStridesToStandardShapePermutation(
    SmallVectorImpl<int64_t> &ret) const {
  ArrayRef<int64_t> strides = getStrides();
  size_t n = strides.size();
  ret.resize_for_overwrite(n);
  // Vector that'll be sorted so that the dimension with the smallest stride
  // comes first. The second parameter is the reversed index of the dimension
  // in the original strides array, which ensures that sorting by < leads
  // to shapes like <1x2x1x3xf32, 6x3x3x1> are not permuted.
  SmallVector<std::pair<int64_t, size_t>, 4> indexedStrides;
  for (auto [idx, stride] : llvm::enumerate(strides)) {
    if (stride == 0) {
      // Broadcast dimensions stay put.
      ret[idx] = idx;
    } else {
      indexedStrides.push_back({stride, n - idx});
      ret[idx] = -1;
    }
  }
  llvm::sort(indexedStrides);
  // The strides are sorted backwards of how we want to deal with them.
  size_t nextStride = indexedStrides.size() - 1;
  for (int64_t &stride : ret) {
    if (stride != -1)
      // Broadcast dimension which is already in permutaiton.
      continue;
    // Undo the ordering hack above.
    stride = n - indexedStrides[nextStride--].second;
  }
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
