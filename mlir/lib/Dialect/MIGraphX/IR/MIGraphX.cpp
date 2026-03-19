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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
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
  return LiteralOp::create(builder, loc, type, elemsValue);
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

  return getChecked(
      [&]() -> InFlightDiagnostic {
        return parser.emitError(parser.getCurrentLocation());
      },
      shape, strides, elementType);
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
  Type elementType = getElementType();
  if (elementType.isInteger() && !elementType.isSignlessInteger()) {
    elementType =
        IntegerType::get(getContext(), elementType.getIntOrFloatBitWidth(),
                         IntegerType::SignednessSemantics::Signless);
  }
  return RankedTensorType::get(orderedShape, elementType);
}

RankedTensorType MIXRShapedType::asFlatMemoryTensor() const {
  RankedTensorType memoryTensorType = asMemoryLayoutTensor();
  if (!memoryTensorType)
    return nullptr;
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

LogicalResult ReshapeOp::verify() {
  MIXRShapedType inputType = getInput().getType();
  MIXRShapedType outType = getOutput().getType();
  ArrayAttr dimsAttr = getDims();

  // Dynamic shapes are not currently supported
  if (!inputType.hasStaticShape())
    return emitOpError("Dynamic shapes are not supported");

  if (static_cast<int64_t>(dimsAttr.size()) != outType.getRank())
    return emitOpError("number of dims (")
           << dimsAttr.size() << ") does not match result rank ("
           << outType.getRank() << ")";

  // Check that there is only a single -1 value
  int missingDims = llvm::count_if(
      dimsAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr a) { return a.getInt() == -1; });
  if (missingDims > 1)
    return emitOpError("expected at most one target dimension to be -1");

  // Check how many zero dimensions there are
  int numZeros = llvm::count_if(
      dimsAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr a) { return a.getInt() == 0; });

  if (missingDims > 0 && numZeros > 0)
    return emitOpError("Cannot mix missing dimensions with zero dimension");

  // Compare dimension values to output shape
  for (auto [dimVal, outDim] : llvm::zip(dimsAttr, outType.getShape())) {
    int64_t dimValue = cast<IntegerAttr>(dimVal).getInt();
    // We cannot handle negative dims values that aren't -1 
    if (dimValue < -1 ) {
      return emitOpError("Non -1 negative values are not supported");
    }

    // Output dimensions can't be negative
    if (outDim < 0)
      return emitOpError("Negative output dimensions are not supported");

    // Per-dimension consistency
    if (dimValue >= 0 && outDim != dimValue)
      return emitOpError("dimValue: ")
             << dimValue << " inconsistent with result dimension " << outDim;
  }

  // Check that the number of elements in the input and output types match
  int64_t inputElements = inputType.getNumElements();
  if (inputElements != outType.getNumElements())
    return emitOpError("input and output element counts do not match");

  return success();
}

LogicalResult UnpackOp::verify() {
  MIXRShapedType inType = getIn().getType();
  MIXRShapedType outType = getOut().getType();
  int64_t axis = getAxis();

  if (axis < 0 || axis > inType.getRank())
    return emitOpError("axis out of range of shape: ") << axis;
  // If we're not an int8 <-> int8 operator, we're in the middle of rewrites.
  if (inType.getElementType().isInteger(8) &&
      outType.getElementType().isInteger(8) &&
      inType.getDimSize(axis) * 2 != outType.getDimSize(axis))
    return emitOpError("expected length along input axis to be half the length "
                       "along output axis");
  return success();
}

static LogicalResult isValidDotOp(Operation *op, MIXRShapedType inAType,
                                  MIXRShapedType inBType,
                                  MIXRShapedType outputType) {
  ArrayRef<int64_t> shapeA = inAType.getShape();
  ArrayRef<int64_t> shapeB = inBType.getShape();
  ArrayRef<int64_t> shapeOut = outputType.getShape();
  int64_t outputRank = outputType.getRank();

  if (!llvm::all_of(
          ArrayRef<int64_t>{inAType.getRank(), inBType.getRank(), outputRank},
          [](int64_t rank) { return rank >= 2; })) {
    return op->emitOpError("expect operand to have rank greater or equal to 2");
  }

  // Batch dimensions (all dims except the last two) must be compatible.
  // Broadcasting is allowed when one operand's batch dims are all ones
  // or when one operand has no batch dims (rank 2). For example:
  //   A = {3, 2, 2, 2} and B = {1, 1, 2, 2} (batch B is all ones) - valid
  //   A = {3, 2, 2, 2} and B = {2, 2} (B has no batch dims) - valid
  //   A = {3, 2, 2, 2} and B = {2, 3, 2, 2} (batch dims differ) - invalid
  ArrayRef<int64_t> batchA = shapeA.drop_back(2);
  ArrayRef<int64_t> batchB = shapeB.drop_back(2);
  bool hasLeadingOnesB = llvm::all_of(batchB, [](int64_t d) { return d == 1; });
  if (!hasLeadingOnesB &&
      !std::equal(batchA.begin(), batchA.end(), batchB.begin(), batchB.end())) {
    return op->emitOpError("batch dimension mismatch: the first operand (")
           << inAType << ") and the second operand (" << inBType
           << ") have incompatible batch dimensions";
  }

  int64_t lastAShape = shapeA[shapeA.size() - 1];
  int64_t secondLastBShape = shapeB[shapeB.size() - 2];
  if (lastAShape != secondLastBShape) {
    return op->emitOpError(
               "contraction dimension mismatch: the first operand (")
           << inAType << ") and the second operand (" << inBType
           << ") have incompatible contraction dimensions";
  }

  // checking the output dimension, which must match the input
  if (!std::equal(shapeA.rbegin() + 2, shapeA.rend(), shapeOut.rbegin() + 2,
                  shapeOut.rend()) ||
      *std::prev(shapeOut.end()) != *std::prev(shapeB.end()) ||
      *std::prev(shapeOut.end(), 2) != *std::prev(shapeA.end(), 2)) {
    return op->emitOpError("result type is inconsistent with input shapes");
  }

  return success();
}

LogicalResult QuantDotOp::verify() {
  MIXRShapedType inAType = getInA().getType();
  MIXRShapedType inBType = getInB().getType();
  Type aElemType = inAType.getElementType();
  Type bElemType = inBType.getElementType();

  MIXRShapedType resultType = getResult().getType();

  bool hasScaleA = getScaleA() != nullptr;
  bool hasScaleB = getScaleB() != nullptr;
  if (hasScaleA ^ hasScaleB)
    return emitOpError("both scaleA and scaleB must be provided or neither");
  bool isScaledGemm = hasScaleA && hasScaleB;
  if (isScaledGemm) {
    ArrayRef<int64_t> scaleAShape = getScaleA().getType().getShape();
    ArrayRef<int64_t> inAShape = inAType.getShape();
    if (scaleAShape.size() != inAShape.size())
      return emitOpError("scaleA shape must have the same number of dimensions "
                         "as the input types");
    for (auto [scaleADim, inADim] : llvm::zip(scaleAShape, inAShape)) {
      if (scaleADim != inADim)
        return emitOpError(
            "scaleA shape must have the same dimensions as the input types");
    }
    ArrayRef<int64_t> scaleBShape = getScaleB().getType().getShape();
    ArrayRef<int64_t> inBShape = inBType.getShape();
    if (scaleBShape.size() != inBShape.size())
      return emitOpError("scaleB shape must have the same number of dimensions "
                         "as the input types");
    for (auto [scaleBDim, inBDim] : llvm::zip(scaleBShape, inBShape)) {
      if (scaleBDim != inBDim)
        return emitOpError(
            "scaleB shape must have the same dimensions as the input types");
    }
    if (aElemType != bElemType)
      return emitOpError("input types must have the same element type");
    if (!isa<Float4E2M1FNType>(aElemType) || !isa<Float4E2M1FNType>(bElemType))
      return emitOpError(
          "Scaled quant dot ops only support f4E2M1FN element type");
    if (!isa<Float32Type>(resultType.getElementType()))
      return emitOpError(
          "result type must be a float32 type for scaled quant dot ops");
  } else {
    if (isa<Float4E2M1FNType>(aElemType) || isa<Float4E2M1FNType>(bElemType))
      return emitOpError("Quant Dot ops requires scales to be provided to use "
                         "f4E2M1FN element type");
  }
  return isValidDotOp(getOperation(), inAType, inBType, getType());
}

LogicalResult DotOp::verify() {
  MIXRShapedType inAType = getInA().getType();
  MIXRShapedType inBType = getInB().getType();

  return isValidDotOp(getOperation(), inAType, inBType, getType());
}

LogicalResult SigmoidOp::verify() {
  if (!getInA().getType().getElementType().isFloat() ||
      !getResult().getType().getElementType().isFloat()) {
    return emitOpError("only support floating point");
  }

  return success();
}

LogicalResult WhereOp::verify() {
  MIXRShapedType condType = getCond().getType();
  MIXRShapedType inAType = getInA().getType();
  MIXRShapedType inBType = getInB().getType();
  MIXRShapedType resultType = getType();

  if (inAType.getElementType() != inBType.getElementType() ||
      inAType.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "input and output types must have the same element type");
  }

  if (condType.getShape() != inAType.getShape() ||
      condType.getShape() != inBType.getShape() ||
      condType.getShape() != resultType.getShape()) {
    return emitOpError("input and output types must have the same shape");
  }

  return success();
}
