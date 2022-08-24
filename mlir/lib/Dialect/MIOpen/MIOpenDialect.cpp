//===- MIOpenOps.cpp - MIOpen MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/MIOpen.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/MathExtras.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include <algorithm>
#include <iterator>
#include <limits>

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
    if (attr.isa<TransformMapAttr>()) {
      os << "transform_map";
      return AliasResult::OverridableAlias;
    }
    if (attr.isa<GeneralGemmParamsAttr>()) {
      os << "general_gemm_params";
      return AliasResult::OverridableAlias;
    }
    if (attr.isa<XdlopsGemmParamsAttr>()) {
      os << "xdlops_gemm_params";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

namespace mlir {
namespace miopen {

/// Constant Name for MIOpen Kernel Module
constexpr const ::llvm::StringLiteral MIOpenDialect::kKernelModuleName;

ArrayAttr noTransformsArray(Builder &b, size_t n) {
  llvm::SmallVector<Attribute, 4> ret;
  ret.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    ret.push_back(b.getArrayAttr({}));
  }
  return b.getArrayAttr(ret);
}

//===---------------------------------------------------------
// TransformAttr
//===---------------------------------------------------------
template <typename T>
static ParseResult
parseAndGather(mlir::AsmParser &parser, AsmParser::Delimiter delim,
               SmallVectorImpl<T> &ret,
               llvm::function_ref<ParseResult(T &)> getElement) {
  return parser.parseCommaSeparatedList(delim, [&]() -> ParseResult {
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
    if (parseAndGather<int64_t>(parser, AsmParser::Delimiter::None, params,
                                [&](int64_t &out) -> ParseResult {
                                  return parser.parseInteger(out);
                                }) ||
        parser.parseRBrace()) {
      return {};
    }
  }

  llvm::SmallVector<std::string> upperNamesStorage;
  llvm::SmallVector<unsigned> upperDims;
  if (parseAndGather<std::string>(parser, AsmParser::Delimiter::Square,
                                  upperNamesStorage,
                                  [&](std::string &out) -> ParseResult {
                                    return parser.parseKeywordOrString(&out);
                                  }) ||
      parser.parseKeyword("at") ||
      parseAndGather<unsigned>(parser, AsmParser::Delimiter::Square, upperDims,
                               [&](unsigned &out) -> ParseResult {
                                 return parser.parseInteger(out);
                               })) {
    return {};
  }

  if (parser.parseArrow()) {
    return {};
  }

  llvm::SmallVector<std::string> lowerNamesStorage;
  llvm::SmallVector<unsigned> lowerDims;
  if (parseAndGather<std::string>(parser, AsmParser::Delimiter::Square,
                                  lowerNamesStorage,
                                  [&](std::string &out) -> ParseResult {
                                    return parser.parseKeywordOrString(&out);
                                  }) ||
      parser.parseKeyword("at") ||
      parseAndGather<unsigned>(parser, AsmParser::Delimiter::Square, lowerDims,
                               [&](unsigned &out) -> ParseResult {
                                 return parser.parseInteger(out);
                               })) {
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
  case TransformType::Broadcast:
    if (upperDims.size() != lowerDims.size()) {
      return emitError() << "Broadcast must have same rank";
    }
    if (params.size() != lowerDims.size()) {
      return emitError()
             << "Broadcast must specify the output length for each dimension";
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
// TransformMapAttr
//===---------------------------------------------------------

TransformMapAttr getTransformMapAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MLIRContext *context, ArrayRef<TransformAttr> ops, AffineMapAttr map,
    ArrayRef<int64_t> upperBounds, ArrayRef<int64_t> lowerBounds) {
  return TransformMapAttr::getChecked(emitError, context, ops, map, upperBounds,
                                      lowerBounds);
}

LogicalResult TransformMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<::mlir::miopen::TransformAttr> ops, AffineMapAttr map,
    ArrayRef<int64_t> upperBounds, ArrayRef<int64_t> lowerBounds) {
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

LogicalResult Conv2DOp::verify() { return verifyConvOp(*this); }

LogicalResult Conv2DBwdDataOp::verify() { return verifyConvOp(*this); }

LogicalResult Conv2DBwdWeightOp::verify() { return verifyConvOp(*this); }

//===-----------------------------------------------------===//
// GemmOp
//===-----------------------------------------------------===//
LogicalResult GemmOp::verify() {
  auto typeA = a().getType().cast<MemRefType>(),
       typeB = b().getType().cast<MemRefType>(),
       typeC = c().getType().cast<MemRefType>();
  Type inElems = typeA.getElementType(), outElems = typeC.getElementType();
  if (inElems.isa<IntegerType>() && !outElems.isInteger(32))
    return emitOpError("integer-valued multiply must have i32 as its result");
  if (inElems.isa<FloatType>() && !outElems.isa<FloatType>())
    return emitOpError(
        "float-valued inputs must have a floating-point output type");

  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape(),
                    dimsC = typeC.getShape();
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1,
          offsetC = dimsC.size() == 2 ? 0 : 1;
  int64_t gA = offsetA ? dimsA[0] : 1, gB = offsetB ? dimsB[0] : 1,
          gC = offsetC ? dimsC[0] : 1;
  int64_t mA = dimsA[offsetA + (aTransposed() ? 1 : 0)],
          kA = dimsA[offsetA + (aTransposed() ? 0 : 1)],
          kB = dimsB[offsetB + (bTransposed() ? 1 : 0)],
          nB = dimsB[offsetB + (bTransposed() ? 0 : 1)],
          mC = dimsC[offsetC + (cTransposed() ? 1 : 0)],
          nC = dimsC[offsetC + (cTransposed() ? 0 : 1)];
  if (gA != gB || gA != gC)
    return emitOpError("group dimensions don't match")
           << " g_a = " << gA << " g_b = " << gB << " g_c = " << gC;
  if (mA != mC)
    return emitOpError("M dimensions don't match")
           << " m_a = " << mA << " m_c = " << mC;
  if (nB != nC)
    return emitOpError("N dimensions don't match")
           << " n_b = " << nB << " n_c = " << nC;
  if (kA != kB)
    return emitOpError("K dimensions don't match")
           << " k_a = " << kA << " k_b = " << kB;
  return success();
}

//===-----------------------------------------------------===//
// ExtractSliceOp
//===-----------------------------------------------------===//
LogicalResult ExtractSliceOp::canonicalize(ExtractSliceOp op,
                                           PatternRewriter &b) {
  // Extracting a vector of the same size as the source is a no-op, since it
  // has to happen from index 0 to ensure legality
  if (op.result().getType() == op.vector().getType()) {
    b.replaceOp(op, op.vector());
    return success();
  }
  return failure();
}

LogicalResult ExtractSliceOp::verify() {
  if (auto destType = result().getType().dyn_cast<VectorType>()) {
    size_t destSize = destType.getDimSize(0);
    size_t sourceSize = vector().getType().cast<VectorType>().getDimSize(0);
    if (destSize > sourceSize)
      return emitOpError("Output size " + Twine(destSize) +
                         " exceeds input size " + Twine(sourceSize));
  }
  return success();
}

//===-----------------------------------------------------===//
// InsertSliceOp
//===-----------------------------------------------------===//
LogicalResult InsertSliceOp::canonicalize(InsertSliceOp op,
                                          PatternRewriter &b) {
  // Per the in-bounds requirement, storing a slice of the same length as a
  // vector is just replacing dest with src, so drop the intermedation
  if (op.source().getType() == op.dest().getType()) {
    b.replaceOp(op, op.source());
    return success();
  }
  return failure();
}
LogicalResult InsertSliceOp::verify() {
  if (auto sourceType = source().getType().dyn_cast<VectorType>()) {
    size_t sourceSize = sourceType.getDimSize(0);
    size_t destSize = dest().getType().cast<VectorType>().getDimSize(0);
    if (sourceSize > destSize)
      return emitOpError(
          "Slice to store has length " + Twine(sourceSize) +
          " which is longer than destinanation's vector length " +
          Twine(destSize));
  }
  return success();
}

//===-----------------------------------------------------===//
// TransformingForOp
//===-----------------------------------------------------===//

static ArrayAttr maybeIndexArray(OpBuilder &b,
                                 Optional<ArrayRef<int64_t>> vals) {
  return vals.map([&b](ArrayRef<int64_t> v) { return b.getIndexArrayAttr(v); })
      .getValueOr(ArrayAttr{});
}

void TransformingForOp::build(OpBuilder &b, OperationState &state,
                              ArrayRef<ValueRange> inits,
                              ArrayRef<Attribute> transforms,
                              ArrayRef<int64_t> bounds,
                              Optional<ArrayRef<int64_t>> strides,
                              bool forceUnroll, bool useIndexDiffs,
                              ValueRange iterArgs) {
  build(b, state, inits, b.getArrayAttr(transforms),
        b.getIndexArrayAttr(bounds), maybeIndexArray(b, strides), forceUnroll,
        useIndexDiffs, iterArgs);
}

void TransformingForOp::build(OpBuilder &b, OperationState &state,
                              ArrayRef<ValueRange> inits,
                              ArrayRef<Attribute> transforms, ArrayAttr bounds,
                              ArrayAttr strides, bool forceUnroll,
                              bool useIndexDiffs, ValueRange iterArgs) {
  build(b, state, inits, b.getArrayAttr(transforms), bounds, strides,
        forceUnroll, useIndexDiffs, iterArgs);
}

void TransformingForOp::build(OpBuilder &b, OperationState &state,
                              ArrayRef<ValueRange> inits, ArrayAttr transforms,
                              ArrayRef<int64_t> bounds,
                              Optional<ArrayRef<int64_t>> strides,
                              bool forceUnroll, bool useIndexDiffs,
                              ValueRange iterArgs) {
  build(b, state, inits, transforms, b.getIndexArrayAttr(bounds),
        maybeIndexArray(b, strides), forceUnroll, useIndexDiffs, iterArgs);
}

void TransformingForOp::build(OpBuilder &b, OperationState &state,
                              ArrayRef<ValueRange> inits, ArrayAttr transforms,
                              ArrayAttr bounds, ArrayAttr strides,
                              bool forceUnroll, bool useIndexDiffs,
                              ValueRange iterArgs) {
  // Set up user-provided attributes
  state.addAttribute(boundsAttrName(state.name), bounds);
  if (!strides) {
    SmallVector<int64_t> strideVec(bounds.size(), 1LL);
    strides = b.getIndexArrayAttr(strideVec);
  }
  state.addAttribute(stridesAttrName(state.name), strides);
  state.addAttribute(transformsAttrName(state.name), transforms);
  if (forceUnroll)
    state.addAttribute(forceUnrollAttrName(state.name), b.getUnitAttr());
  if (useIndexDiffs)
    state.addAttribute(useIndexDiffsAttrName(state.name), b.getUnitAttr());

  int32_t upperLen = bounds.size();
  for (ValueRange upper : inits)
    state.addOperands(upper);
  state.addOperands(iterArgs);
  state.addTypes(iterArgs.getTypes());

  // Track sizes of variadic arguments to enable them to be looped up
  state.addAttribute(
      TransformingForOp::getOperandSegmentSizeAttr(),
      b.getI32VectorAttr({upperLen * static_cast<int32_t>(inits.size()),
                          static_cast<int32_t>(iterArgs.size())}));

  // Set up region and block
  Region *bodyRegion = state.addRegion();
  Block &bodyBlock = bodyRegion->emplaceBlock();

  // Track starting position of each domain's lower coordinates in the block
  // argument list so that we can give out references to appropriate slices
  // of that list
  SmallVector<int32_t> lowerStarts;
  int32_t nLower = 0;
  Type indexType = b.getIndexType();
  for (auto domain : transforms.getAsRange<ArrayAttr>()) {
    lowerStarts.push_back(nLower);
    int32_t len = 0;
    if (domain.empty()) // No transforms, copy upper coordinates
      len = upperLen;
    else
      len = domain[domain.size() - 1]
                .cast<TransformMapAttr>()
                .getLowerBounds()
                .size();
    for (int32_t i = 0; i < len; ++i)
      bodyBlock.addArgument(indexType, state.location);
    nLower += len;
  }
  lowerStarts.push_back(nLower);
  state.addAttribute(lowerStartsAttrName(state.name),
                     b.getI32VectorAttr(lowerStarts));

  for (Value v : iterArgs)
    bodyBlock.addArgument(v.getType(), v.getLoc());

  if (iterArgs.empty())
    ensureTerminator(*bodyRegion, b, state.location);
}

ParseResult TransformingForOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  using OperandType = OpAsmParser::UnresolvedOperand;
  using Delimiter = OpAsmParser::Delimiter;

  Builder &b = parser.getBuilder();
  Type indexTy = b.getIndexType();

  SmallVector<Attribute> transforms;
  SmallVector<int32_t> lowerStarts;
  SmallVector<OpAsmParser::Argument> lowerArgs;
  SmallVector<OperandType> upperInits;

  if (failed(parser.parseOptionalAttrDict(result.attributes)))
    return failure();

  // Parse iteration domains
  llvm::SMLoc transformedLoc = parser.getCurrentLocation();
  ParseResult loopItersParse = parser.parseCommaSeparatedList(
      Delimiter::None,
      [&]() -> ParseResult {
        llvm::SMLoc loopIterLoc = parser.getCurrentLocation();
        size_t oldNLower = lowerArgs.size();
        size_t oldNUpper = upperInits.size();
        lowerStarts.push_back(oldNLower);

        if (parser.parseArgumentList(lowerArgs, Delimiter::Paren) ||
            parser.parseEqual()) {
          return failure();
        }
        for (size_t i = 0; i < lowerArgs.size(); i++) {
          lowerArgs[i].type = indexTy;
        }
        ArrayAttr theseTransforms;
        if (parser.parseAttribute(theseTransforms)) {
          return failure();
        }
        if (parser.parseOperandList(upperInits, Delimiter::Paren)) {
          return failure();
        }
        if (theseTransforms.size() == 0) {
          if (upperInits.size() - oldNUpper != lowerArgs.size() - oldNLower) {
            return parser.emitError(loopIterLoc,
                                    "Expected same number of lower and upper "
                                    "arguments when transforms absent");
          }
        } else {
          for (Attribute a : theseTransforms) {
            if (!a.isa<TransformMapAttr>()) {
              return parser.emitError(loopIterLoc,
                                      "Expected transform map attributes");
            }
          }
          size_t nInputs = theseTransforms[0]
                               .cast<TransformMapAttr>()
                               .getMap()
                               .getAffineMap()
                               .getNumInputs();
          size_t nOutputs = theseTransforms[theseTransforms.size() - 1]
                                .cast<TransformMapAttr>()
                                .getMap()
                                .getAffineMap()
                                .getNumResults();
          if (upperInits.size() - oldNUpper != nInputs) {
            return parser.emitError(loopIterLoc,
                                    "Transformation sequence expected ")
                   << nInputs << " inputs";
          }
          if (lowerArgs.size() - oldNLower != nOutputs) {
            return parser.emitError(loopIterLoc,
                                    "Transformation sequence expected ")
                   << nOutputs << " outputs";
          }
        }
        transforms.push_back(theseTransforms);
        return success();
      },
      "for a loop iteration argument (lower coordinates, transforms, initial "
      "upper args)");
  if (loopItersParse) {
    return failure();
  }
  lowerStarts.push_back(lowerArgs.size());

  result.addAttribute(TransformingForOp::transformsAttrName(result.name),
                      b.getArrayAttr(transforms));
  result.addAttribute(TransformingForOp::lowerStartsAttrName(result.name),
                      b.getI32VectorAttr(lowerStarts));

  llvm::SMLoc iterArgsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::Argument> iterArgs;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> iterInits;
  llvm::SmallVector<mlir::Type> iterTypes;
  if (parser.parseOptionalKeyword("iter_args").succeeded()) {
    if (parser.parseAssignmentList(iterArgs, iterInits) ||
        parser.parseArrowTypeList(iterTypes)) {
      return failure();
    }
  }
  if (iterInits.size() != iterTypes.size())
    return parser.emitError(iterArgsLoc,
                            "Mismatch between number of iter_args and types");
  for (auto pair : llvm::zip(iterArgs, iterTypes)) {
    std::get<0>(pair).type = std::get<1>(pair);
  }

  if (parser.parseKeyword("bounds")) {
    return failure();
  }

  auto intListParser = [&](SmallVectorImpl<int64_t> &dest) -> ParseResult {
    int64_t res;
    if (parser.parseInteger(res)) {
      return failure();
    }
    dest.push_back(res);
    return success();
  };
  llvm::SmallVector<int64_t> bounds;
  ParseResult boundsRes = parser.parseCommaSeparatedList(
      Delimiter::Square, [&]() -> ParseResult { return intListParser(bounds); },
      "list of bounds");
  if (boundsRes) {
    return failure();
  }
  result.addAttribute(TransformingForOp::boundsAttrName(result.name),
                      b.getIndexArrayAttr(bounds));

  if (parser.parseKeyword("strides")) {
    return failure();
  }

  llvm::SmallVector<int64_t> strides;
  ParseResult stridesRes = parser.parseCommaSeparatedList(
      Delimiter::Square,
      [&]() -> ParseResult { return intListParser(strides); },
      "list of strides");
  if (stridesRes) {
    return failure();
  }
  result.addAttribute(TransformingForOp::stridesAttrName(result.name),
                      b.getIndexArrayAttr(strides));

  SmallVector<OpAsmParser::Argument> regionArgs = std::move(lowerArgs);
  regionArgs.append(iterArgs);
  result.addTypes(iterTypes);

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs)) {
    return failure();
  }
  TransformingForOp::ensureTerminator(*body, b, result.location);

  SmallVector<Type> upperInitTypes(upperInits.size(), indexTy);
  if (parser.resolveOperands(upperInits, upperInitTypes, transformedLoc,
                             result.operands) ||
      parser.resolveOperands(iterInits, iterTypes, iterArgsLoc,
                             result.operands)) {
    return failure();
  }

  result.addAttribute(
      TransformingForOp::getOperandSegmentSizeAttr(),
      b.getI32VectorAttr({static_cast<int32_t>(upperInits.size()),
                          static_cast<int32_t>(iterInits.size())}));
  return success();
}

void TransformingForOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(getOperation()->getAttrs(), /*elidedAttrs=*/{
                              TransformingForOp::getOperandSegmentSizeAttr(),
                              transformsAttrName(), lowerStartsAttrName(),
                              boundsAttrName(), stridesAttrName()});
  p << " ";
  for (uint32_t i = 0, e = domains(); i < e; ++i) {
    p << "(";
    p.printOperands(getLowerCoords(i));
    p << ") = ";
    p.printAttributeWithoutType(getTransforms(i));
    p << "(";
    p.printOperands(getUpperInits(i));
    p << ")";
    if (i != e - 1) {
      p << ", ";
    }
  }

  if (iterInits().size() > 0) {
    p << " iter_args (";
    llvm::interleaveComma(
        llvm::zip(getIterArgs(), iterInits()), p, [&](auto i) {
          Value init = std::get<1>(i);
          p << std::get<0>(i) << " = " << init;
        });
    p << ") -> (" << iterInits().getTypes() << ")";
  }
  p << " bounds [";
  llvm::interleaveComma(bounds().getAsValueRange<IntegerAttr>(), p,
                        [&](llvm::APInt bound) { p << bound; });
  p << "] ";
  p << "strides [";
  llvm::interleaveComma(strides().getAsValueRange<IntegerAttr>(), p,
                        [&](llvm::APInt stride) { p << stride; });
  p << "] ";
  p.printRegion(region(), /*printEntryBlockArgs=*/false);
}

LogicalResult TransformingForOp::verify() {
  if (bounds().empty())
    return emitOpError("Must have at least one iteration dimension");
  if (bounds().size() != strides().size())
    return emitOpError("Bounds list and strides list must have same length");

  for (size_t i = 0, e = bounds().size(); i < e; ++i) {
    int64_t bound = bounds()[i].cast<IntegerAttr>().getInt();
    int64_t stride = strides()[i].cast<IntegerAttr>().getInt();
    if (stride <= 0)
      return emitOpError("Negative and zero strides are not permitted");
    if (bound % stride != 0)
      return emitOpError(
          "Bound for dimension " + Twine(i) + " (" + Twine(bound) +
          ") does not evenly divide the stride in that dimension (" +
          Twine(stride));
  }

  if (getNumResults() != getIterArgs().size()) {
    return emitOpError(
        "Mismatch between number of yielded values and number of op results");
  }

  uint32_t lowerArgsCount = 0;
  if (lowerStarts().size() != domains() + 1) {
    return emitOpError(
        "Lower starts attribute doesn't have one entry per domain plus 1");
  }
  if (lowerStart(0) != 0) {
    return emitOpError("Region args don't start with lower coords");
  }

  for (uint32_t i = 0, e = domains(); i < e; ++i) {
    ArrayAttr transforms = getTransforms(i);
    auto lowerArgs = getLowerCoords(i);
    auto upperInits = getUpperInits(i);
    if (transforms.size() == 0) {
      if (upperInits.size() != lowerArgs.size()) {
        return emitOpError("Mismatch between number of lower and upper "
                           "coordinates without a transform in domain #" +
                           Twine(i));
      }
    } else {
      size_t nUpper = transforms[0]
                          .cast<TransformMapAttr>()
                          .getMap()
                          .getValue()
                          .getNumInputs();
      size_t nLower = transforms[transforms.size() - 1]
                          .cast<TransformMapAttr>()
                          .getMap()
                          .getValue()
                          .getNumResults();
      if (upperInits.size() != nUpper) {
        return emitOpError(
            "Mismatch between number of upper initial values "
            "and number of inputs to transform sequence in domain #" +
            Twine(i));
      }
      if (lowerArgs.size() != nLower) {
        return emitOpError(
            "Mismatch between number of lower arguments and "
            "number of outputs of transform sequence in domain #" +
            Twine(i));
      }
    }
    lowerArgsCount += lowerArgs.size();
    if (lowerStart(i + 1) != lowerArgsCount) {
      return emitOpError("Lower starts attribute not accurate after domain #" +
                         Twine(i));
    }
  }
  return success();
}

// Cribbed from AffineForOp
Region &TransformingForOp::getLoopBody() { return region(); }
bool TransformingForOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}
void TransformingForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
}

//===-----------------------------------------------------===//
// IndexDiffUpdateOp
//===-----------------------------------------------------===//
void IndexDiffUpdateOp::build(OpBuilder &b, OperationState &state,
                              TransformMapAttr transform, ValueRange upperDiff,
                              ValueRange lowerOrig) {
  llvm::SmallVector<Type> resultTypes(lowerOrig.size(), b.getIndexType());
  IndexDiffUpdateOp::build(b, state, resultTypes, resultTypes, transform,
                           upperDiff, lowerOrig);
}

LogicalResult IndexDiffUpdateOp::verify() {
  TransformMapAttr transform = map();
  size_t nLowerIn = lowerOrig().size();
  size_t nLowerOut = lowerIndices().size();

  if (nLowerIn != nLowerOut)
    return emitOpError("Got " + Twine(nLowerIn) + " lower inputs but " +
                       Twine(nLowerOut) + " lower outputs");

  size_t nUpper = upperDiffs().size();
  size_t nMapIn = transform.getUpperBounds().size();
  size_t nMapOut = transform.getLowerBounds().size();

  if (nUpper != nMapIn)
    return emitOpError("Expected " + Twine(nMapIn) + " upper diffs but got " +
                       Twine(nUpper));
  if (nMapOut != nLowerIn)
    return emitOpError("Expected " + Twine(nMapOut) +
                       " lower coordinates but got " + Twine(nLowerIn));
  return success();
}

//===-----------------------------------------------------===//
// BufferLoadOp
//===-----------------------------------------------------===//
LogicalResult BufferLoadOp::verify() {
  auto sourceType = source().getType().cast<MemRefType>();
  size_t nDims = sourceType.getRank();
  for (llvm::APInt dimVal : leftOobDims().getAsValueRange<IntegerAttr>()) {
    int32_t dim = dimVal.getSExtValue();
    if (dim < 0)
      return emitOpError("Left OOB dimensions must be non-negative, got " +
                         Twine(dim));
    if (static_cast<uint32_t>(dim) >= nDims)
      return emitOpError(
          "Left OOB dims must refer to one of the " + Twine(nDims) +
          " dimensions of the memref but got dimension " + Twine(dim));
  }
  for (llvm::APInt dimVal : rightOobDims().getAsValueRange<IntegerAttr>()) {
    int32_t dim = dimVal.getSExtValue();
    if (dim < 0)
      return emitOpError("Right OOB dimensions must be non-negative, got " +
                         Twine(dim));
    if (static_cast<uint32_t>(dim) >= nDims)
      return emitOpError(
          "Right OOB dims must refer to one of the " + Twine(nDims) +
          " dimensions of the memref but got dimension " + Twine(dim));
  }

  if (coords().size() != nDims)
    return emitOpError("Expected " + Twine(nDims) + " coordinates for load");
  if (sourceType.getMemorySpaceAsInt() != 0)
    return emitOpError("Source memref must live in global memory");
  if (mlir::getElementTypeOrSelf(result()) != sourceType.getElementType())
    return emitOpError(
        "Result element type must match source memref's element type");
  return success();
}

//===-----------------------------------------------------===//
// BufferStoreOp
//===-----------------------------------------------------===//
LogicalResult BufferStoreOp::verify() {
  auto destType = dest().getType().cast<MemRefType>();
  size_t nDims = destType.getRank();
  for (llvm::APInt dimVal : leftOobDims().getAsValueRange<IntegerAttr>()) {
    int32_t dim = dimVal.getSExtValue();
    if (dim < 0)
      return emitOpError("Left OOB dimensions must be non-negative, got " +
                         Twine(dim));
    if (static_cast<uint32_t>(dim) >= nDims)
      return emitOpError(
          "Left OOB dims must refer to one of the " + Twine(nDims) +
          " dimensions of the memref but got dimension " + Twine(dim));
  }
  for (llvm::APInt dimVal : rightOobDims().getAsValueRange<IntegerAttr>()) {
    int32_t dim = dimVal.getSExtValue();
    if (dim < 0)
      return emitOpError("Right OOB dimensions must be non-negative, got " +
                         Twine(dim));
    if (static_cast<uint32_t>(dim) >= nDims)
      return emitOpError(
          "Right OOB dims must refer to one of the " + Twine(nDims) +
          " dimensions of the memref but got dimension " + Twine(dim));
  }
  if (coords().size() != nDims)
    return emitOpError("Expected " + Twine(nDims) + " coordinates for store");
  if (destType.getMemorySpaceAsInt() != 0)
    return emitOpError("Destination memref must live in global memory");
  if (mlir::getElementTypeOrSelf(data()) != destType.getElementType())
    return emitOpError(
        "Element type of data must match element type of destination memref");
  return success();
}

//===-----------------------------------------------------===//
// InBoundsLoadOp
//===-----------------------------------------------------===//
LogicalResult InBoundsLoadOp::verify() {
  auto sourceType = source().getType().cast<MemRefType>();
  size_t nDims = sourceType.getRank();
  if (coords().size() != nDims)
    return emitOpError("Expected " + Twine(nDims) + " coordinates for load");
  Type resultType = result().getType();
  if (resultType.isa<ShapedType>() && !resultType.isa<VectorType>())
    return emitOpError(
        "Non-scalar loads must return vectors, not other shaped types");
  return success();
}

//===-----------------------------------------------------===//
// InBoundsLoadOp
//===-----------------------------------------------------===//
LogicalResult InBoundsStoreOp::verify() {
  auto destType = dest().getType().cast<MemRefType>();
  size_t nDims = destType.getRank();
  if (coords().size() != nDims)
    return emitOpError("Expected " + Twine(nDims) + " coordinates for store");
  Type dataType = data().getType();
  if (dataType.isa<ShapedType>() && !dataType.isa<VectorType>())
    return emitOpError(
        "Non-scalar data types must be vectors, not other shaped types");
  return success();
}

//===----------------------------------------------------------------------===//
// BlockwiseGemmOp
//===----------------------------------------------------------------------===//
LogicalResult BlockwiseGemmOp::verify() {
  auto blockAType = matrixA().getType().cast<MemRefType>();
  auto blockBType = matrixB().getType().cast<MemRefType>();
  auto bufferCType = matrixC().getType().cast<MemRefType>();

  int64_t k = blockAType.getShape()[0];
  int64_t m = blockAType.getShape()[1];
  int64_t n = blockBType.getShape()[1];
  int64_t kPack = blockAType.getShape()[2];

  if (k != blockBType.getShape()[0]) {
    return emitOpError("Mismatched k dimensions between A and B");
  }
  if (kPack != blockBType.getShape()[2]) {
    return emitOpError("Mismatched kPack between A and B");
  }

  // Obtain critical attributes.
  int64_t mC = bufferCType.getShape()[0];
  int64_t nC = bufferCType.getShape()[1];
  GeneralGemmParamsAttr params = paramsAttr();
  int64_t mPerThread = params.getMPerThread();
  int64_t nPerThread = params.getNPerThread();

  int64_t mThreadsPerCuwave = params.getMThreadsPerCuwave();
  int64_t nThreadsPerCuwave = params.getNThreadsPerCuwave();

  int64_t mCuwavesPerBlock = params.getMCuwavesPerBlock();
  int64_t nCuwavesPerBlock = params.getNCuwavesPerBlock();

  int64_t mRepeat = mC / mPerThread;
  int64_t nRepeat = nC / nPerThread;

  if (mRepeat * mCuwavesPerBlock * mThreadsPerCuwave * mPerThread != m)
    return emitOpError("The m turing attributes don't multiply to M_LDS");
  if (nRepeat * nCuwavesPerBlock * nThreadsPerCuwave * nPerThread != n)
    return emitOpError("The n turing parameters don't multiply to N_LDS");
  return success();
}

//===----------------------------------------------------------------------===//
// ThreadwiseGemmOp
//===----------------------------------------------------------------------===//
LogicalResult ThreadwiseGemmOp::verify() {
  ArrayRef<int64_t> aShape = matrixA().getType().cast<MemRefType>().getShape(),
                    bShape = matrixB().getType().cast<MemRefType>().getShape(),
                    cShape = matrixC().getType().cast<MemRefType>().getShape();

  if (aShape[0] != bShape[0])
    return emitOpError("K dimensions don't match");
  if (aShape[1] != cShape[0])
    return emitOpError("M dimensions don't match");
  if (bShape[1] != cShape[1])
    return emitOpError("N dimensions don't match");
  if (aShape[2] != bShape[2])
    return emitOpError("KPack dimensions don't match");
  return success();
}

//===----------------------------------------------------------------------===//
// InWarpTransposeOp
//===----------------------------------------------------------------------===//

LogicalResult InWarpTransposeOp::verify() {
  InWarpTransposeOp &op = *this;
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
// WorkgroupIdOp and WorkitemIdOp
//===----------------------------------------------------------------------===//
static ConstantIntRanges
getIdRange(StringRef idName, Operation *op,
           int64_t fallback = std::numeric_limits<int32_t>::max()) {
  uint32_t bitwidth =
      ConstantIntRanges::getStorageBitwidth(op->getResultTypes().front());
  APInt zero = APInt::getZero(bitwidth);
  APInt max(bitwidth, fallback);
  if (func::FuncOp container = op->getParentOfType<func::FuncOp>()) {
    if (IntegerAttr size =
            container->getAttr(idName).dyn_cast_or_null<IntegerAttr>()) {
      // Range inference uses ranges that're inclusive on both ends
      max = APInt(bitwidth, size.getValue().getSExtValue() - 1);
    }
  }
  return ConstantIntRanges::fromUnsigned(zero, max);
}

void WorkgroupIdOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                      SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), getIdRange("grid_size", getOperation()));
}

void WorkitemIdOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                     SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), getIdRange("block_size", getOperation()));
}
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenOps.cpp.inc"
