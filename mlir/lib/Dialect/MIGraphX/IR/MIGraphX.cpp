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
#include "mlir/IR/ODSSupport.h"
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
#include "llvm/ADT/StringExtras.h"

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
  int missingDims =
      llvm::count_if(dimsAttr.getAsRange<IntegerAttr>(),
                     [](IntegerAttr a) { return a.getInt() == -1; });
  if (missingDims > 1)
    return emitOpError("expected at most one target dimension to be -1");

  // Check how many zero dimensions there are
  int numZeros = llvm::count_if(dimsAttr.getAsRange<IntegerAttr>(),
                                [](IntegerAttr a) { return a.getInt() == 0; });

  if (missingDims > 0 && numZeros > 0)
    return emitOpError("Cannot mix missing dimensions with zero dimension");

  // Compare dimension values to output shape
  for (auto [dimVal, outDim] : llvm::zip(dimsAttr, outType.getShape())) {
    int64_t dimValue = cast<IntegerAttr>(dimVal).getInt();
    // We cannot handle negative dims values that aren't -1
    if (dimValue < -1) {
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

LogicalResult SliceOp::verify() {
  auto convertSliceAttribute = [](ArrayAttr attr) -> SmallVector<int64_t, 4> {
    return llvm::map_to_vector(attr.getValue(), [](Attribute attr) {
      IntegerAttr integerAttr = dyn_cast<IntegerAttr>(attr);
      assert(integerAttr && "Tablegen asserts a I64 ArrayAttr");

      return integerAttr.getInt();
    });
  };

  SmallVector<int64_t, 4> axes = convertSliceAttribute(getAxes()),
                          starts = convertSliceAttribute(getStarts()),
                          ends = convertSliceAttribute(getEnds());

  if (axes.size() != starts.size() || axes.size() != ends.size()) {
    return emitOpError("axes, starts, and ends must have the same size");
  }
  ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  ArrayRef<int64_t> outputShape = getOutput().getType().getShape();
  if (inputShape.size() != outputShape.size()) {
    return emitOpError("input and output shapes must have the same rank");
  }

  if (llvm::any_of(axes, [](int64_t axis) { return axis < 0; }) ||
      llvm::any_of(starts, [](int64_t start) { return start < 0; }) ||
      llvm::any_of(ends, [](int64_t end) { return end < 0; })) {
    return emitOpError("all attribute must non non-negative");
  }

  int64_t inputRank = inputShape.size();
  if (llvm::any_of(axes, [&](int64_t axis) { return axis >= inputRank; })) {
    return emitOpError("axes is greater than input rank");
  }

  if (axes.size() != starts.size() || axes.size() != ends.size()) {
    return emitOpError(
        "axes, starts, and ends attribute must have the same size");
  }

  // end is greater than start
  if (llvm::any_of(llvm::zip(starts, ends), [&](auto value) {
        auto [start, end] = value;
        return start >= end;
      })) {
    return emitOpError("start is greater or equal to end");
  }

  if (llvm::any_of(llvm::zip_equal(axes, ends), [&](auto value) {
        auto [axis, end] = value;
        return end > inputShape[axis];
      })) {
    return emitOpError("end is greater than input shape");
  }

  SmallVector<int64_t, 4> inferredShape(inputShape);
  for (auto [axis, start, end] : llvm::zip(axes, starts, ends)) {
    inferredShape[axis] = end - start;
  }

  if (inferredShape != outputShape) {
    return emitOpError("input shape and attribute does not infer output shape");
  }

  return success();
}

/// Verifies that if `dependent` feature flag is set, the `required` feature
/// flag must also be set. Emits `msg` as an error if the dependency is
/// violated. Used to enforce constraints like "prefix_offset requires causal".
static LogicalResult verifyFeatureDependency(
    Operation *op, std::optional<AttentionFeatures> features,
    AttentionFeatures required, AttentionFeatures dependent, StringRef msg) {
  if (hasAttentionFeature(features, dependent) &&
      !hasAttentionFeature(features, required))
    return op->emitOpError(msg);
  return success();
}

/// Verifies that an operand is present when a feature flag is set.
/// Emits an error like "feature 'kvcache' requires 'currentSeqLen' operand"
/// if the flag is set but the operand is null.
static LogicalResult
verifyOperandRequiredByFeature(Operation *op, Value operand,
                               std::optional<AttentionFeatures> features,
                               AttentionFeatures flag, StringRef operandName) {
  if (hasAttentionFeature(features, flag) && !operand)
    return op->emitOpError("feature '")
           << stringifyAttentionFeatures(flag) << "' requires '" << operandName
           << "' operand";
  return success();
}

/// Verifies that an integer attribute is present when a feature flag is set.
/// Emits an error like
/// "feature 'sliding_window' requires 'slidingWindowSize' attribute"
/// if the flag is set but the attribute is absent.
static LogicalResult
verifyAttrRequiredByFeature(Operation *op, std::optional<int32_t> attr,
                            std::optional<AttentionFeatures> features,
                            AttentionFeatures flag, StringRef attrName) {
  if (hasAttentionFeature(features, flag) && !attr.has_value())
    return op->emitOpError("feature '")
           << stringifyAttentionFeatures(flag) << "' requires '" << attrName
           << "' attribute";
  return success();
}

/// Verifies that an operand is NOT present unless a feature flag is set.
/// Emits an error like "'currentSeqLen' operand requires feature 'kvcache'"
/// if the operand is present but the flag is not set. Prevents orphan
/// operands that have no effect without the corresponding feature.
static LogicalResult
verifyOrphanOperand(Operation *op, Value operand,
                    std::optional<AttentionFeatures> features,
                    AttentionFeatures flag, StringRef operandName) {
  if (operand && !hasAttentionFeature(features, flag))
    return op->emitOpError("'") << operandName << "' operand requires feature '"
                                << stringifyAttentionFeatures(flag) << "'";
  return success();
}

/// Verifies that an integer attribute is NOT present unless a feature flag
/// is set. Prevents orphan attributes like splitKV=2 without the splitkv
/// feature flag.
static LogicalResult verifyOrphanAttr(Operation *op,
                                      std::optional<int32_t> attr,
                                      std::optional<AttentionFeatures> features,
                                      AttentionFeatures flag,
                                      StringRef attrName) {
  if (attr && !hasAttentionFeature(features, flag))
    return op->emitOpError("'") << attrName << "' attribute requires feature '"
                                << stringifyAttentionFeatures(flag) << "'";
  return success();
}

/// Verifies that an attention operand parameterised by the per-head batch
/// dimensions of Q (e.g. currentSeqLen, prefixOffset) has been broadcast to
/// match Q's leading dims exactly. The shape must equal `qBatch` (e.g.
/// `[batch]` for 3D Q, `[batch, numHeads]` for 4D Q).
/// Producers with a per-batch sequence length must broadcast across heads
/// explicitly (e.g. via migraphx.multibroadcast) before constructing the
/// attention op. The flattened `[batch * numHeads]` layout is reserved for
/// the kernel-side `rock.attention` op and is materialised by the lowering.
static LogicalResult verifyAttentionLeadingDimsOperand(Operation *op,
                                                       Value operand,
                                                       ArrayRef<int64_t> qBatch,
                                                       StringRef name) {
  if (!operand)
    return success();
  auto shapedTy = cast<ShapedType>(operand.getType());
  ArrayRef<int64_t> shape = shapedTy.getShape();

  if (shape.size() == qBatch.size() &&
      std::equal(shape.begin(), shape.end(), qBatch.begin()))
    return success();

  return op->emitOpError("'")
         << name << "' shape must match Q leading dims [" << qBatch
         << "] (got [" << llvm::make_range(shape.begin(), shape.end())
         << "]); broadcast across heads explicitly via "
            "migraphx.multibroadcast if needed";
}

/// Verifies sliding window constraints: the window size must be positive,
/// currentSeqLen must be present, and the window size must not exceed the
/// maximum key sequence length. Mirrors rock::verifySlidingWindowConstraints.
static LogicalResult
verifySlidingWindowConstraints(Operation *op,
                               std::optional<int32_t> slidingWindowSize,
                               Value currentSeqLen, int64_t maxSeqLen) {
  if (!slidingWindowSize)
    return success();
  if (*slidingWindowSize <= 0)
    return op->emitOpError("slidingWindowSize must be positive");
  if (!currentSeqLen)
    return op->emitOpError(
        "slidingWindowSize requires currentSeqLen to be set");
  if (*slidingWindowSize > maxSeqLen)
    return op->emitOpError(
        "slidingWindowSize must not exceed max sequence length");
  return success();
}

LogicalResult AttentionOp::verify() {
  auto qType = cast<ShapedType>(getQueries().getType());
  auto kType = cast<ShapedType>(getKeys().getType());
  auto vType = cast<ShapedType>(getValues().getType());
  auto resultType = cast<ShapedType>(getResult().getType());

  int64_t qRank = qType.getRank();
  int64_t kRank = kType.getRank();
  int64_t vRank = vType.getRank();

  if (qRank < 2 || kRank < 2 || vRank < 2)
    return emitOpError("operands must have rank >= 2");

  ArrayRef<int64_t> qShape = qType.getShape();
  ArrayRef<int64_t> kShape = kType.getShape();
  ArrayRef<int64_t> vShape = vType.getShape();

  int64_t qHeadDim = qShape[qRank - 1];
  int64_t kHeadDim = kShape[kRank - 2];
  if (qHeadDim != kHeadDim)
    return emitOpError("head dimension mismatched for first gemm: "
                       "last dim of queries (")
           << qHeadDim << ") != second-to-last dim of keys (" << kHeadDim
           << ")";

  int64_t kSeqDim = kShape[kRank - 1];
  int64_t vSeqDim = vShape[vRank - 2];
  if (kSeqDim != vSeqDim)
    return emitOpError("sequence length dimension mismatch for second gemm: "
                       "last dim of keys (")
           << kSeqDim << ") != second-to-last dim of values (" << vSeqDim
           << ")";

  ArrayRef<int64_t> qBatch = qShape.drop_back(2);
  ArrayRef<int64_t> kBatch = kShape.drop_back(2);
  ArrayRef<int64_t> vBatch = vShape.drop_back(2);

  if (qBatch.size() != kBatch.size() || qBatch.size() != vBatch.size())
    return emitOpError("leading dimension mismatch: queries, keys, and values "
                       "must have the same number of leading dimensions");

  // K and V must have identical leading dims. Q's leading dims must either
  // equal K's or be divisible by K's (GQA: numHeadsQ is a multiple of
  // numHeadsKV).
  for (auto [i, dims] : llvm::enumerate(llvm::zip(qBatch, kBatch, vBatch))) {
    auto [qd, kd, vd] = dims;
    if (kd != vd)
      return emitOpError("leading dimension mismatch at dimension ")
             << i << ": keys=" << kd << " != values=" << vd;
    if (qd != kd && qd % kd != 0)
      return emitOpError("leading dimension mismatch at dimension ")
             << i << ": queries=" << qd
             << " is not equal to or divisible by keys=" << kd;
  }

  auto features = getFeatures();

  int64_t seqQ = qShape[qRank - 2];
  int64_t headV = vShape[vRank - 1];
  SmallVector<int64_t> expectedResultShape(qBatch.begin(), qBatch.end());
  bool inflateSplitKV =
      hasAttentionFeature(features, AttentionFeatures::splitkv) &&
      getSplitKVAttr() && getSplitKVAttr().getInt() > 1;
  if (inflateSplitKV) {
    expectedResultShape.push_back(getSplitKVAttr().getInt());
    expectedResultShape.push_back(seqQ);
    expectedResultShape.push_back(headV);
  } else {
    expectedResultShape.push_back(seqQ);
    expectedResultShape.push_back(headV);
  }

  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape.size() != expectedResultShape.size() ||
      !std::equal(resultShape.begin(), resultShape.end(),
                  expectedResultShape.begin()))
    return emitOpError("result shape is inconsistent with attention "
                       "dimensions: expected [")
           << llvm::make_range(expectedResultShape.begin(),
                               expectedResultShape.end())
           << "] but got ["
           << llvm::make_range(resultShape.begin(), resultShape.end()) << "]";

  if (auto lseVal = getLse()) {
    auto lseType = cast<ShapedType>(lseVal.getType());
    SmallVector<int64_t> expectedLseShape(qBatch.begin(), qBatch.end());
    if (inflateSplitKV)
      expectedLseShape.push_back(getSplitKVAttr().getInt());
    expectedLseShape.push_back(seqQ);
    ArrayRef<int64_t> lseShape = lseType.getShape();
    if (lseShape.size() != expectedLseShape.size() ||
        !std::equal(lseShape.begin(), lseShape.end(), expectedLseShape.begin()))
      return emitOpError("lse shape is inconsistent with attention "
                         "dimensions: expected [")
             << llvm::make_range(expectedLseShape.begin(),
                                 expectedLseShape.end())
             << "] but got ["
             << llvm::make_range(lseShape.begin(), lseShape.end()) << "]";
  }

  if (auto smType = getSoftmaxType()) {
    if (!isa<FloatType>(*smType))
      return emitOpError("softmaxType must be a float type, got ") << *smType;
  } else if (!isa<FloatType>(qType.getElementType())) {
    // When Q (and hence the QK output) is integer-typed, the producer must
    // explicitly pick a float softmax type. The body is expected to dequantize
    // the integer QK to that float type before softmax runs.
    return emitOpError(
        "softmaxType must be set explicitly when Q has a non-float element "
        "type; preSoftmaxBody must dequantize to that float type");
  }

  Region &body = getPreSoftmaxBody();
  bool hasPreSoftmaxInputs = !getPreSoftmaxElemWiseInputs().empty();
  bool hasNonTerminatorOps = false;
  // Allow ops in the body that either carry the Elementwise trait or are
  // explicitly accepted (e.g. dequantize/quantize that semantically act
  // elementwise but don't carry the trait yet). Keep this list narrow: every
  // entry here must have a corresponding scalar lowering in
  // MIGraphXAttentionToRock and downstream paths.
  auto isAllowedInPreSoftmaxBody = [](Operation &op) {
    Dialect *dialect = op.getDialect();
    if (!dialect || dialect->getNamespace() != "migraphx")
      return false;
    if (op.hasTrait<OpTrait::Elementwise>())
      return true;
    return isa<migraphx::DeQuantizeLinearOp>(op);
  };
  for (Block &block : body) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      hasNonTerminatorOps = true;
      if (!isAllowedInPreSoftmaxBody(op))
        return op.emitOpError(
                   "preSoftmaxBody must only contain elementwise migraphx ops "
                   "(or migraphx.dequantizelinear), but found '")
               << op.getName() << "'";
    }
  }

  if (hasPreSoftmaxInputs && !hasNonTerminatorOps)
    return emitOpError("preSoftmaxElemWiseInputs are provided but "
                       "preSoftmaxBody contains no operations");
  if (!hasPreSoftmaxInputs && hasNonTerminatorOps)
    return emitOpError("preSoftmaxBody contains operations but no "
                       "preSoftmaxElemWiseInputs are provided");

  if (hasPreSoftmaxInputs) {
    size_t expectedArgs = 1 + getPreSoftmaxElemWiseInputs().size();
    size_t actualArgs = body.front().getNumArguments();
    if (actualArgs != expectedArgs)
      return emitOpError("preSoftmaxBody block must have exactly ")
             << expectedArgs << " arguments (1 for QK result + "
             << getPreSoftmaxElemWiseInputs().size()
             << " preSoftmaxElemWiseInputs), got " << actualArgs;
  }

  // SingleBlockImplicitTerminator guarantees the block and yield exist.
  // When the body has ops, the yield must return the result value.
  // When the body is empty (no preSoftmaxInputs), the yield must be bare.
  assert(!body.empty() &&
         "SingleBlockImplicitTerminator should ensure a block");
  auto yieldOp = cast<migraphx::YieldOp>(body.front().getTerminator());
  if (hasNonTerminatorOps) {
    if (!yieldOp.getValue())
      return yieldOp.emitOpError(
          "must yield a value when preSoftmaxBody contains operations");
  } else if (yieldOp.getValue()) {
    return yieldOp.emitOpError(
        "must not yield a value when preSoftmaxBody is empty");
  }

  // When splitKV is enabled, preSoftmaxElemWiseInputs must have shapes
  // that include the split dimension (matching the split QK space), not
  // the original unsplit QK shape.
  int64_t effectiveSplitKV = 1;
  if (hasAttentionFeature(features, AttentionFeatures::splitkv) &&
      getSplitKVAttr())
    effectiveSplitKV = getSplitKVAttr().getInt();

  if (hasPreSoftmaxInputs && effectiveSplitKV > 1) {
    // Expected QK shape in split space: [B..., splitKV, seqQ, seqK/splitKV]
    int64_t seqK = kShape[kRank - 1];
    int64_t seqKPerSplit = seqK / effectiveSplitKV;
    SmallVector<int64_t> expectedQKShape(qBatch.begin(), qBatch.end());
    expectedQKShape.push_back(effectiveSplitKV);
    expectedQKShape.push_back(seqQ);
    expectedQKShape.push_back(seqKPerSplit);

    for (Value input : getPreSoftmaxElemWiseInputs()) {
      auto inputType = cast<ShapedType>(input.getType());
      ArrayRef<int64_t> inputShape = inputType.getShape();
      if (inputShape.size() != expectedQKShape.size())
        return emitOpError("preSoftmaxElemWiseInput shape rank (")
               << inputShape.size() << ") must match split QK shape rank ("
               << expectedQKShape.size() << ") when splitkv is enabled";
    }
  }

  // Feature flag validation (depends on features/splitKV already read above
  // for result/LSE shape checks).
  if (failed(verifyFeatureDependency(
          getOperation(), features, AttentionFeatures::kvcache,
          AttentionFeatures::sliding_window,
          "feature 'sliding_window' requires 'kvcache' to be set")))
    return failure();

  if (failed(verifyFeatureDependency(
          getOperation(), features, AttentionFeatures::causal,
          AttentionFeatures::prefix_offset,
          "feature 'prefix_offset' requires 'causal' to be set")))
    return failure();

  if (failed(verifyOperandRequiredByFeature(
          getOperation(), getCurrentSeqLen(), features,
          AttentionFeatures::kvcache, "currentSeqLen")))
    return failure();
  if (failed(verifyOperandRequiredByFeature(
          getOperation(), getPrefixOffset(), features,
          AttentionFeatures::prefix_offset, "prefixOffset")))
    return failure();

  if (failed(verifyAttrRequiredByFeature(
          getOperation(), getSlidingWindowSize(), features,
          AttentionFeatures::sliding_window, "slidingWindowSize")))
    return failure();

  if (hasAttentionFeature(features, AttentionFeatures::splitkv)) {
    if (!getLse())
      return emitOpError("feature '")
             << stringifyAttentionFeatures(AttentionFeatures::splitkv)
             << "' requires LSE result";
    if (!getSplitKVAttr() || getSplitKVAttr().getInt() <= 1)
      return emitOpError("feature '")
             << stringifyAttentionFeatures(AttentionFeatures::splitkv)
             << "' requires splitKV > 1";
    // The key sequence length must be evenly divisible by splitKV so that
    // K and V can be split into equal chunks.
    int64_t seqK = kShape[kRank - 1];
    int64_t splitKVVal = getSplitKVAttr().getInt();
    if (seqK % splitKVVal != 0)
      return emitOpError("key sequence length (")
             << seqK << ") must be divisible by splitKV (" << splitKVVal << ")";
  }

  if (failed(verifyOrphanOperand(getOperation(), getCurrentSeqLen(), features,
                                 AttentionFeatures::kvcache, "currentSeqLen")))
    return failure();
  if (failed(verifyOrphanOperand(getOperation(), getPrefixOffset(), features,
                                 AttentionFeatures::prefix_offset,
                                 "prefixOffset")))
    return failure();

  std::optional<int32_t> splitKVOrphan;
  if (getSplitKVAttr())
    splitKVOrphan = getSplitKVAttr().getInt();
  if (failed(verifyOrphanAttr(getOperation(), splitKVOrphan, features,
                              AttentionFeatures::splitkv, "splitKV")))
    return failure();
  if (failed(verifyOrphanAttr(getOperation(), getSlidingWindowSize(), features,
                              AttentionFeatures::sliding_window,
                              "slidingWindowSize")))
    return failure();

  if (getSplitKVAttr() && getSplitKVAttr().getInt() <= 0)
    return emitOpError("splitKV must be positive");

  if (failed(verifyAttentionLeadingDimsOperand(
          getOperation(), getCurrentSeqLen(), qBatch, "currentSeqLen")))
    return failure();
  if (failed(verifyAttentionLeadingDimsOperand(
          getOperation(), getPrefixOffset(), qBatch, "prefixOffset")))
    return failure();

  int64_t maxSeqLen = kShape[kRank - 1];
  if (failed(verifySlidingWindowConstraints(getOperation(),
                                            getSlidingWindowSize(),
                                            getCurrentSeqLen(), maxSeqLen)))
    return failure();

  return success();
}
