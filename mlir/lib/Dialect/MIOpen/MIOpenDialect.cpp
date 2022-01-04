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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::miopen;

#include "mlir/Dialect/MIOpen/MIOpenOpsDialect.cpp.inc"
#include "mlir/Dialect/MIOpen/MIOpenTypes.cpp.inc"
//===----------------------------------------------------------------------===//
// MIOpenDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
struct MIOpenOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (attr.isa<TransformsAttr>()) {
      os << "transforms";
      return AliasResult::OverridableAlias;
    }
    if (attr.isa<PaddingInfoAttr>()) {
      os << "gemm_padding_info";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

namespace mlir {
namespace miopen {
ArrayAttr noTransformsArray(Builder &b, size_t n) {
  llvm::SmallVector<Attribute, 4> ret;
  ret.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    ret.push_back(b.getArrayAttr({}));
  }
  return b.getArrayAttr(ret);
}

AsmPrinter &operator<<(AsmPrinter &printer, BwdPaddingKernelInfo v) {
  std::string toPrint = getBitsForBwdPaddingKernelInfo(v);
  return printer << "\"" << toPrint << "\"";
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
  StringRef name = getNameForTransformType(getType());
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
  if (type != TransformType::AddDim && lowerDims.size() == 0) {
    return emitError() << "The transformation must define outputs";
  }
  if (upperDims.size() == 0) {
    return emitError() << "The transformation must have at least one input";
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
  case TransformType::AddDim:
    if (upperDims.size() != 1) {
      return emitError() << "Can only add one dimension at a time";
    }
    if (params.size() != upperDims.size()) {
      return emitError() << "Must supply a size parameter for each dimension";
    }
    if (lowerDims.size() != 0) {
      return emitError() << "The added dimension cannot be mapped anywhere";
    }
    break;
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

  for (int64_t v : upperBounds) {
    if (v < 0) {
      return emitError() << "Upper bound/shape component less than 0";
    }
  }
  for (int64_t v : lowerBounds) {
    if (v < 0) {
      return emitError() << "Lower bound/shape component less than 0";
    }
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
  addInterfaces<MIOpenOpAsmDialectInterface>();
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
  for (auto outerPair :
       llvm::enumerate(op.transforms().getAsRange<ArrayAttr>())) {
    size_t index = outerPair.index();
    ArrayAttr transforms = outerPair.value();
    if (transforms.size() > 0) {
      auto firstTransform = transforms[0].cast<TransformsAttr>();
      auto lastTransform =
          transforms[transforms.size() - 1].cast<TransformsAttr>();
      AffineMap firstMap = firstTransform.getMap().getValue();
      AffineMap lastMap = lastTransform.getMap().getValue();
      if (index == 0) {
        if (lastMap.getNumResults() != sourceRank)
          return op.emitError(
              "Number of coordindates in externally defined affine map doesn't "
              "match the rank of the source memref");

        expectedSourceCoords = firstMap.getNumInputs();
      } else if (index == 1) {
        if (lastMap.getNumResults() != destRank)
          return op.emitError(
              "Number of coordindates in externally defined affine map doesn't "
              "match the rank of the destination memref");

        expectedDestCoords = firstMap.getNumInputs();
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
