//===- MIOpenOps.cpp - MIOpen MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/MIOpen.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::miopen;

#include "mlir/Dialect/MIOpen/MIOpenOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MIOpenDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {} // namespace

namespace mlir {
namespace miopen {
Optional<TransformType> getTransformTypeForName(const StringRef name) {
  if (name == "PassThrough") {
    return TransformType::PassThrough;
  }
  if (name == "Pad") {
    return TransformType::Pad;
  }
  if (name == "Slice") {
    return TransformType::Slice;
  }
  if (name == "Embed") {
    return TransformType::Embed;
  }
  if (name == "Unmerge") {
    return TransformType::Unmerge;
  }
  if (name == "Merge") {
    return TransformType::Merge;
  }
  if (name == "Unfold") {
    return TransformType::Unfold;
  }
  return llvm::None;
}

const char *getNameForTransformType(const TransformType type) {
  switch (type) {
  case TransformType::PassThrough:
    return "PassThrough";
  case TransformType::Pad:
    return "Pad";
  case TransformType::Slice:
    return "Slice";
  case TransformType::Embed:
    return "Embed";
  case TransformType::Unmerge:
    return "Unmerge";
  case TransformType::Merge:
    return "Merge";
  case TransformType::Unfold:
    return "Unfold";
  }
  llvm_unreachable("Enum not one of the valid cases");
}

Optional<ConvOpType> getConvOpTypeForName(const StringRef name) {
  if (name == "conv2d") {
    return Conv2DOpType;
  }
  if (name == "conv2d_bwd_data") {
    return Conv2DBwdDataOpType;
  }
  if (name == "conv2d_bwd_weight") {
    return Conv2DBwdWeightOpType;
  }
  return llvm::None;
}

const char *getNameForConvOpType(const miopen::ConvOpType op) {
  switch (op) {
  case Conv2DOpType:
    return "conv2d";
  case Conv2DBwdDataOpType:
    return "conv2d_bwd_data";
  case Conv2DBwdWeightOpType:
    return "conv2d_bwd_weight";
  }
  llvm_unreachable("Invalid ConvOp type");
}

//===---------------------------------------------------------
// TransformAttr
//===---------------------------------------------------------
template <typename T>
static ParseResult
parseAndGather(mlir::AsmParser &parser, SmallVector<T> &ret,
               llvm::function_ref<ParseResult(T &)> getElement) {
  return parser.parseCommaSeparatedList([&]() -> ParseResult {
    T out;
    ParseResult res = getElement(out);
    if (res.succeeded()) {
      ret.push_back(out);
    }
    return res;
  });
}

mlir::Attribute TransformAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  llvm::SMLoc startLoc = parser.getCurrentLocation();
  if (parser.parseLess()) {
    return {};
  }

  std::string transformName;
  if (parser.parseKeywordOrString(&transformName)) {
    return {};
  }

  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  Optional<TransformType> transformType =
      getTransformTypeForName(transformName);
  if (!transformType.hasValue()) {
    parser.emitError(typeLoc, "expected a name of a known transform")
            .attachNote()
        << "The transforms are PassThrough, Pad, Slice, Embed, Unmerge, Merge, "
           "Unfold";
    return {};
  }

  llvm::SmallVector<int64_t> params;
  if (parser.parseOptionalLBrace().succeeded()) {
    if (parseAndGather<int64_t>(parser, params,
                                [&](int64_t &out) -> ParseResult {
                                  return parser.parseInteger(out);
                                }) ||
        parser.parseRBrace()) {
      return {};
    }
  }

  llvm::SmallVector<std::string> upperNamesStorage;
  llvm::SmallVector<unsigned> upperDims;
  if (parser.parseLSquare() ||
      parseAndGather<std::string>(parser, upperNamesStorage,
                                  [&](std::string &out) -> ParseResult {
                                    return parser.parseKeywordOrString(&out);
                                  }) ||
      parser.parseRSquare() || parser.parseKeyword("at") ||
      parser.parseLSquare() ||
      parseAndGather<unsigned>(parser, upperDims,
                               [&](unsigned &out) -> ParseResult {
                                 return parser.parseInteger(out);
                               }) ||
      parser.parseRSquare()) {
    return {};
  }

  if (parser.parseArrow()) {
    return {};
  }

  llvm::SmallVector<std::string> lowerNamesStorage;
  llvm::SmallVector<unsigned> lowerDims;
  if (parser.parseLSquare() ||
      parseAndGather<std::string>(parser, lowerNamesStorage,
                                  [&](std::string &out) -> ParseResult {
                                    return parser.parseKeywordOrString(&out);
                                  }) ||
      parser.parseRSquare() || parser.parseKeyword("at") ||
      parser.parseLSquare() ||
      parseAndGather<unsigned>(parser, lowerDims,
                               [&](unsigned &out) -> ParseResult {
                                 return parser.parseInteger(out);
                               }) ||
      parser.parseRSquare()) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  SmallVector<StringRef> upperNames;
  for (const std::string &name : upperNamesStorage) {
    upperNames.push_back(name);
  }
  SmallVector<StringRef> lowerNames;
  for (const std::string &name : lowerNamesStorage) {
    lowerNames.push_back(name);
  }

  return parser.getChecked<TransformAttr>(
      startLoc, parser.getContext(), transformType.getValue(), params,
      upperNames, upperDims, lowerNames, lowerDims);
}

void TransformAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  const char *name = getNameForTransformType(getType());
  printer.printKeywordOrString(name);
  ArrayRef<int64_t> params = getParams();
  if (params.size() > 0) {
    printer << "{";
    llvm::interleaveComma(params, printer);
    printer << "}";
  }
  printer << " [";
  llvm::interleaveComma(getUpperNames(), printer,
                        [&](StringRef s) { printer << "\"" << s << "\""; });
  printer << "] at [";
  llvm::interleaveComma(getUpperDims(), printer);
  printer << "] -> [";
  llvm::interleaveComma(getLowerNames(), printer,
                        [&](StringRef s) { printer << "\"" << s << "\""; });
  printer << "] at [";
  llvm::interleaveComma(getLowerDims(), printer);
  printer << "]>";
}

LogicalResult
TransformAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      TransformType type, llvm::ArrayRef<int64_t> params,
                      llvm::ArrayRef<llvm::StringRef> upperNames,
                      llvm::ArrayRef<unsigned> upperDims,
                      llvm::ArrayRef<llvm::StringRef> lowerNames,
                      llvm::ArrayRef<unsigned> lowerDims) {
  if (upperNames.size() != upperDims.size()) {
    return emitError() << "Have " << upperNames.size() << " names for "
                       << upperDims.size() << " dimensions";
  }
  if (lowerNames.size() != lowerDims.size()) {
    return emitError() << "Have " << lowerNames.size() << " names for "
                       << lowerDims.size() << " dimensions";
  }
  switch (type) {
  case TransformType::PassThrough: {
    if (upperDims.size() != lowerDims.size()) {
      return emitError()
             << "PassThrough must have the same number of inputs and outputs";
    }
    if (params.size() != 0) {
      return emitError() << "PassThrough has no parameters";
    }
    break;
  }
  case TransformType::Pad: // TODO, work out how this works
    break;
  case TransformType::Slice: // TODO, work out how this works
    break;
  case TransformType::Embed:
  case TransformType::Unmerge: {
    if (lowerDims.size() != 1) {
      return emitError()
             << "Embed and unmerge can only have one output argument";
    }
    if (params.size() != upperDims.size()) {
      return emitError() << "Embed and unmerge must specify one coefficient "
                            "per input dimension";
    }
    break;
  }
  case TransformType::Merge:
  case TransformType::Unfold: {
    if (upperDims.size() != 1) {
      return emitError()
             << "Merge and unfold can only have one input dimension";
    }
    if (params.size() != lowerDims.size()) {
      return emitError() << "Merge and unfold have one parameter per output "
                            "dimension (its size)";
    }
    break;
  }
  }
  return success();
}

TransformAttr getTransformAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MLIRContext *context, TransformType type, ArrayRef<int64_t> params,
    ArrayRef<StringRef> upperNames, ArrayRef<uint32_t> upperDims,
    ArrayRef<StringRef> lowerNames, ArrayRef<uint32_t> lowerDims) {
  return TransformAttr::getChecked(emitError, context, type, params, upperNames,
                                   upperDims, lowerNames, lowerDims);
}

//===---------------------------------------------------------
// TransformsAttr
//===---------------------------------------------------------

TransformsAttr getTransformsAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MLIRContext *context, ArrayRef<TransformAttr> ops, AffineMapAttr map,
    ArrayRef<int64_t> upperBounds, ArrayRef<int64_t> lowerBounds) {
  return TransformsAttr::getChecked(emitError, context, ops, map, upperBounds,
                                    lowerBounds);
}

LogicalResult
TransformsAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       ::llvm::ArrayRef<::mlir::miopen::TransformAttr> ops,
                       AffineMapAttr map, ArrayRef<int64_t> upperBounds,
                       ArrayRef<int64_t> lowerBounds) {
  AffineMap rawMap = map.getAffineMap();
  if (rawMap.getNumInputs() != upperBounds.size()) {
    return emitError() << "Affine map has " << rawMap.getNumInputs()
                       << " inputs but there are " << upperBounds.size()
                       << " input dimensions";
  }
  if (rawMap.getNumResults() != lowerBounds.size()) {
    return emitError() << "Affine map has " << rawMap.getNumResults()
                       << " outputs but there are " << lowerBounds.size()
                       << " outut dimensions";
  }
  return success();
}

} // namespace miopen
} // namespace mlir
//===----------------------------------------------------------------------===//
// MIOpenDialect
//===----------------------------------------------------------------------===//

void MIOpenDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/MIOpen/MIOpenAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MIOpen/MIOpenOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Convolution operations
//===----------------------------------------------------------------------===//
template <typename T> static LogicalResult verifyConvOp(T op) {
  auto isDisjointed = [&](llvm::StringRef tensor, llvm::StringRef dim1,
                          llvm::StringRef dim2) {
    auto layout = op->getAttr(tensor).template cast<ArrayAttr>().getValue();
    auto pos1 = -1, pos2 = -1;
    for (unsigned int i = 0; i < layout.size(); ++i) {
      if (layout[i].template cast<StringAttr>().getValue() == dim1)
        pos1 = i;
      if (layout[i].template cast<StringAttr>().getValue() == dim2)
        pos2 = i;
    }

    return (pos2 != pos1 + 1) && (pos1 != pos2 + 1);
  };

  if (isDisjointed("filter_layout", "y", "x") ||
      isDisjointed("input_layout", "hi", "wi"))
    return op.emitError("Disjointed yx or hw!");

  return success();
}

// Utility static member function of TransformOp to populate an ArrayAttr to
// track the bounds of a MemRefType.
ArrayAttr TransformOp::buildMemRefShapeAttr(OpBuilder &b,
                                            MemRefType memRefType) {
  auto shape = memRefType.getShape();
  SmallVector<Attribute> shapeAttr;
  for (auto s : shape)
    shapeAttr.push_back(b.getI32IntegerAttr(s));
  return b.getArrayAttr(shapeAttr);
}

//===----------------------------------------------------------------------===//
// ThreadwiseCopyOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ThreadwiseCopyOp op) {
  auto sourceCoord = op.sourceCoord();
  auto destCoord = op.destCoord();
  auto sourceType = op.source().getType().cast<MemRefType>();
  auto destType = op.dest().getType().cast<MemRefType>();
  auto sourceRank = sourceType.getRank();
  auto destRank = destType.getRank();
  auto sourceAffineMap = sourceType.getLayout().getAffineMap();
  auto destAffineMap = destType.getLayout().getAffineMap();

  unsigned expectedSourceCoords = sourceRank;
  unsigned expectedDestCoords = destRank;

  // check if memrefs have embedded affine maps.
  expectedSourceCoords = sourceAffineMap.getNumInputs();
  expectedDestCoords = destAffineMap.getNumInputs();

  // check if memrefs have externally defined affine maps.
  auto coordTransformAttrs = op->getAttr("coord_transforms");
  if (coordTransformAttrs) {
    for (auto coordTransformAttr :
         coordTransformAttrs.cast<ArrayAttr>().getValue()) {
      auto coordTransformDictAttr = coordTransformAttr.cast<DictionaryAttr>();
      auto operandIndex =
          coordTransformDictAttr.get("operand").cast<IntegerAttr>().getInt();
      auto affineMapsArrayAttr =
          coordTransformDictAttr.get("transforms").cast<ArrayAttr>().getValue();
      auto firstTransform =
          affineMapsArrayAttr[0].cast<AffineMapAttr>().getValue();
      auto lastTransform = affineMapsArrayAttr[affineMapsArrayAttr.size() - 1]
                               .cast<AffineMapAttr>()
                               .getValue();

      if (operandIndex == 0) {
        if (lastTransform.getNumResults() != sourceRank)
          return op.emitError(
              "Number of coordindates in externally defined affine map doesn't "
              "match the rank of the source memref");

        expectedSourceCoords = firstTransform.getNumInputs();
      } else if (operandIndex == 1) {
        if (lastTransform.getNumResults() != destRank)
          return op.emitError(
              "Number of coordindates in externally defined affine map doesn't "
              "match the rank of the destination memref");

        expectedDestCoords = firstTransform.getNumInputs();
      }
    }
  }

  if (sourceCoord.size() != expectedSourceCoords)
    return op.emitError(
        "Number of coordinates supplied doesn't match the rank, or affine maps "
        "of source memref");
  if (destCoord.size() != expectedSourceCoords)
    return op.emitError(
        "Number of coordinates supplied doesn't match the rank, or affine maps "
        "of destination memref");

  return success();
}

//===----------------------------------------------------------------------===//
// InWarpTransposeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(InWarpTransposeOp op) {
  constexpr size_t swizzleGroupSize = InWarpTransposeOp::swizzleGroupSize;
  if (!llvm::isPowerOf2_32(op.size())) {
    return op.emitOpError("transpose size " + Twine(op.size()) +
                          "must be a power of 2");
  }
  if (op.size() <= 0) {
    return op.emitOpError("transpose size must be strictly positive");
  }

  auto vectorLen = static_cast<size_t>(
      op.vector().getType().cast<VectorType>().getNumElements());
  if (vectorLen < swizzleGroupSize) {
    return op.emitOpError("Vector input must have at least" +
                          Twine(swizzleGroupSize) + "elements");
  }
  if (vectorLen < op.size()) {
    return op.emitError("Vector input can't be shorter than transpose size");
  }

  if (op.vector().getType().cast<VectorType>().getRank() != 1) {
    return op.emitError("Input vector must be 1-dimensional");
  }

  auto inGroupPerm = op.inGroupPerm();

  llvm::SmallSet<uint32_t, swizzleGroupSize> expected;
  llvm::SmallSet<uint32_t, swizzleGroupSize> found;

  for (uint32_t i = 0; i < swizzleGroupSize; i++) {
    expected.insert(i);
  }

  for (auto &i : inGroupPerm) {
    found.insert(i.cast<IntegerAttr>().getValue().getZExtValue());
  }

  if (found != expected) {
    return op.emitOpError("inGroupPerm is not a permutation on the output row");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenOps.cpp.inc"
