//===- RockOps.cpp - Rock MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/utility/math.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

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
using namespace mlir::rock;

#include "mlir/Dialect/Rock/IR/RockOpsDialect.cpp.inc"
#include "mlir/Dialect/Rock/IR/RockTypes.cpp.inc"
//===----------------------------------------------------------------------===//
// RockDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
struct RockOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<TransformMapAttr>(attr)) {
      os << "transform_map";
      return AliasResult::OverridableAlias;
    }
    if (isa<GeneralGemmParamsAttr>(attr)) {
      os << "general_gemm_params";
      return AliasResult::OverridableAlias;
    }
    if (isa<XdlopsGemmParamsAttr>(attr)) {
      os << "xldops_gemm_params";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

namespace mlir {
namespace rock {

/// Constant Name for Rock Kernel Module
constexpr const ::llvm::StringLiteral RockDialect::kKernelModuleName;

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
  std::optional<TransformType> transformType =
      getTransformTypeForName(transformName);
  if (!transformType.has_value()) {
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
      startLoc, parser.getContext(), transformType.value(), params, upperNames,
      upperDims, lowerNames, lowerDims);
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
  if (type != TransformType::AddDim && lowerDims.empty()) {
    return emitError() << "The transformation must define outputs";
  }
  if (type != TransformType::ConstDim && upperDims.empty()) {
    return emitError() << "The transformation must have at least one input";
  }

  switch (type) {
  case TransformType::PassThrough: {
    if (upperDims.size() != lowerDims.size()) {
      return emitError()
             << "PassThrough must have the same number of inputs and outputs";
    }
    if (!params.empty()) {
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
  case TransformType::Merge: {
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
    if (!lowerDims.empty()) {
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
  case TransformType::ConstDim:
    if (!upperDims.empty())
      return emitError() << "ConstDim must not take any inputs";
    if (params.size() != 2 * lowerDims.size())
      return emitError()
             << "ConstDim is parameterized by [value, length] pairs";
    for (size_t i = 0, e = params.size(); i < e; i += 2) {
      if (params[i] >= params[i + 1])
        return emitError() << "For constant dimension " << lowerDims[i / 2]
                           << " constant value " << params[i]
                           << " must be less than dimension "
                              "length "
                           << params[i + 1];
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
    DenseI64ArrayAttr upperBounds, DenseI64ArrayAttr lowerBounds) {
  return TransformMapAttr::getChecked(emitError, context, ops, map, upperBounds,
                                      lowerBounds);
}

LogicalResult TransformMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<::mlir::rock::TransformAttr> ops, AffineMapAttr map,
    DenseI64ArrayAttr upperBounds, DenseI64ArrayAttr lowerBounds) {
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

  for (int64_t v : upperBounds.asArrayRef()) {
    if (v < 0) {
      return emitError() << "Upper bound/shape component less than 0";
    }
  }
  for (int64_t v : lowerBounds.asArrayRef()) {
    if (v < 0) {
      return emitError() << "Lower bound/shape component less than 0";
    }
  }
  return success();
}

} // namespace rock
} // namespace mlir
//===----------------------------------------------------------------------===//
// RockDialect
//===----------------------------------------------------------------------===//

void RockDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Rock/IR/RockAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Rock/IR/RockOps.cpp.inc"
      >();
  addInterfaces<RockOpAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// Convolution operations
//===----------------------------------------------------------------------===//
ConvolutionDims ConvolutionDims::fromOp(Operation *op) {
  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  // Get shape of filter tensor.
  auto filterType = cast<MemRefType>(op->getOperand(0).getType());
  ArrayRef<int64_t> filterShape = filterType.getShape();

  // Get shape of input tensor.
  auto inputType = cast<MemRefType>(op->getOperand(1).getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Get shape of output tensor.
  auto outputType = cast<MemRefType>(op->getOperand(2).getType());
  ArrayRef<int64_t> outputShape = outputType.getShape();

  int64_t y, x, z, ho, wo, dout, hi, wi, di, k, c, n, g;
  y = x = z = ho = wo = dout = hi = wi = di = k = c = n = g = 0;

  for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr = cast<StringAttr>(filterLayoutAttr.getValue()[i]);
    auto inputAttr = cast<StringAttr>(inputLayoutAttr.getValue()[i]);
    auto outputAttr = cast<StringAttr>(outputLayoutAttr.getValue()[i]);

    if (filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "0") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x") {
      x = filterShape[i];
    } else if (filterAttr.getValue() == "1") {
      x = filterShape[i];
    } else if (filterAttr.getValue() == "2") {
      z = filterShape[i];
    } else if (filterAttr.getValue() == "k") {
      k = filterShape[i];
    } else if (filterAttr.getValue() == "c") {
      c = filterShape[i];
    } else if (filterAttr.getValue() == "g") {
      g = filterShape[i];
    }

    if (inputAttr.getValue() == "hi") {
      hi = inputShape[i];
    } else if (inputAttr.getValue() == "wi") {
      wi = inputShape[i];
    } else if (inputAttr.getValue() == "0i") {
      hi = inputShape[i];
    } else if (inputAttr.getValue() == "1i") {
      wi = inputShape[i];
    } else if (inputAttr.getValue() == "2i") {
      di = inputShape[i];
    } else if (inputAttr.getValue() == "ni") {
      n = inputShape[i];
    }

    if (outputAttr.getValue() == "ho") {
      ho = outputShape[i];
    } else if (outputAttr.getValue() == "wo") {
      wo = outputShape[i];
    } else if (outputAttr.getValue() == "0o") {
      ho = outputShape[i];
    } else if (outputAttr.getValue() == "1o") {
      wo = outputShape[i];
    } else if (outputAttr.getValue() == "2o") {
      dout = outputShape[i];
    }
  }

  SmallVector<int64_t> fil({y, x});
  if (z > 0)
    fil.push_back(z);
  SmallVector<int64_t> out({ho, wo});
  if (dout > 0)
    out.push_back(dout);
  SmallVector<int64_t> in({hi, wi});
  if (di > 0)
    in.push_back(di);
  return ConvolutionDims(fil, out, in, k, c, n, g);
}

ConvOpType mlir::rock::convOpTypeFromKernelType(KernelType kernelType) {
  switch (kernelType) {
  case KernelType::Conv:
    return ConvOpType::Fwd;
  case KernelType::ConvBwdData:
    return ConvOpType::BwdData;
  case KernelType::ConvBwdWeight:
    return ConvOpType::BwdWeight;
  case KernelType::Gemm:
    llvm_unreachable(
        "Gemm ops shouldn't be in convolution-specific lowering passes");
  case KernelType::Attention:
    llvm_unreachable(
        "Attention ops shouldn't be in convolution-specific lowering passes");
  }
  llvm_unreachable("Unsuppported KernelType");
}

KernelType mlir::rock::kernelTypeFromConvOpType(ConvOpType convOpType) {
  switch (convOpType) {
  case ConvOpType::Fwd:
    return KernelType::Conv;
  case ConvOpType::BwdData:
    return KernelType::ConvBwdData;
  case ConvOpType::BwdWeight:
    return KernelType::ConvBwdWeight;
  }
  llvm_unreachable("Unsupported ConvOpType");
}

GemmSize GemmSize::fromConvolution(ConvOpType type,
                                   const ConvolutionDims &sizes) {
  assert(type != ConvOpType::BwdData &&
         "Backward data convolutions cannot have their size computed without "
         "kernelId and other parameters. Use op.getGemmSize() instead");
  int64_t gemmGSize, gemmMSize, gemmKSize, gemmNSize;
  switch (type) {
  case ConvOpType::Fwd:
    gemmGSize = sizes.g;
    gemmMSize = sizes.k;
    // +++pf: should these accumulate sizes across all dimensions?
    gemmKSize = sizes.c * sizes.fil[0] * sizes.fil[1];
    gemmNSize = sizes.n * sizes.out[0] * sizes.out[1];
    break;
  case ConvOpType::BwdWeight:
    gemmGSize = sizes.g;
    gemmMSize = sizes.k;
    gemmKSize = sizes.n * sizes.out[0] * sizes.out[1];
    gemmNSize = sizes.c * sizes.fil[0] * sizes.fil[1];
    break;
  case ConvOpType::BwdData:
    llvm_unreachable("Should've been caught be an assert");
  }
  return GemmSize(gemmGSize, gemmMSize, gemmKSize, gemmNSize);
}

static bool isFloat8Type(Type type) {
  return isa<FloatType>(type) && type.getIntOrFloatBitWidth() == 8;
}

static LogicalResult verifyGemmTypes(Operation *op, GemmFeatures features,
                                     StringRef arch, Type elemTypeA,
                                     Type elemTypeB, Type elemTypeC) {
  bool isGfx11 = arch.contains("gfx11");
  if (bitEnumContainsAll(features, GemmFeatures::wmma)) {
    if (!(elemTypeA.isF16() || elemTypeA.isBF16() || elemTypeA.isInteger(8))) {
      if (isGfx11)
        return op->emitOpError(
            "Wmma gridwise supports only F16/BF16/int8 data types");
      if (!isFloat8Type(elemTypeA))
        return op->emitOpError(
            "Wmma gridwise supports only F16/BF16/int8/E4M3/E5M2 data types");
    }
    if (elemTypeA != elemTypeB)
      return op->emitOpError("Wmma gridwise does not support mixed types");
  }
  if (isa<FloatType>(elemTypeA) && !isa<FloatType>(elemTypeC)) {
    return op->emitOpError("floating-point input type ")
           << elemTypeA
           << " requires a floating-point output type, but the output type is "
           << elemTypeC;
  }
  if (isa<IntegerType>(elemTypeA) && !isa<IntegerType>(elemTypeC)) {
    return op->emitOpError("integer input type ")
           << elemTypeA
           << " requires an integer output type, but the output type is "
           << elemTypeC;
  }
  return success();
}

static LogicalResult verifyGemmTypes(RockGemmWrapperInterface gemmOp) {
  Type elemTypeA = gemmOp.getAType(), elemTypeB = gemmOp.getBType();
  Type elemTypeC = cast<ShapedType>(gemmOp.getOutArgument()->get().getType())
                       .getElementType();

  return verifyGemmTypes(gemmOp, gemmOp.getGemmFeatures(), gemmOp.getArch(),
                         elemTypeA, elemTypeB, elemTypeC);
}

static LogicalResult verifyConvOp(RockConvInterface convOp) {
  Operation *op = convOp.getOperation();
  auto isDisjointed = [&](llvm::StringRef tensor, llvm::StringRef dim1,
                          llvm::StringRef dim2) {
    auto layout = cast<ArrayAttr>(op->getAttr(tensor)).getValue();
    auto pos1 = -1, pos2 = -1;
    for (unsigned int i = 0; i < layout.size(); ++i) {
      if (cast<StringAttr>(layout[i]).getValue() == dim1)
        pos1 = i;
      if (cast<StringAttr>(layout[i]).getValue() == dim2)
        pos2 = i;
    }
    return (pos2 != pos1 + 1) && (pos1 != pos2 + 1);
  };

  if ((isDisjointed("filter_layout", "y", "x") &&
       isDisjointed("filter_layout", "0", "1")) ||
      (isDisjointed("input_layout", "hi", "wi") &&
       isDisjointed("input_layout", "0i", "1i") &&
       isDisjointed("input_layout", "0", "1")))
    return op->emitError("Disjointed yx or hw!");

  RockGemmWrapperInterface gemmOp = cast<RockGemmWrapperInterface>(*convOp);

  if (failed(verifyGemmTypes(gemmOp)))
    return failure();

  bool isAccel = bitEnumContainsAny(convOp.getFeatures(),
                                    GemmFeatures::mfma | GemmFeatures::wmma);
  if (gemmOp.getDerivedBlockSize().has_value() && !isAccel) {
    return op->emitOpError(
        "general kernels shouldn't have derived block size.");
  }

  return success();
}

LogicalResult ConvOp::verify() { return verifyConvOp(*this); }

LogicalResult ConvBwdDataOp::verify() { return verifyConvOp(*this); }

LogicalResult ConvBwdWeightOp::verify() { return verifyConvOp(*this); }

KernelType ConvOp::getKernelType() { return KernelType::Conv; }

KernelType ConvBwdDataOp::getKernelType() { return KernelType::ConvBwdData; }

KernelType ConvBwdWeightOp::getKernelType() {
  return KernelType::ConvBwdWeight;
}

Type ConvOp::getAType() { return getFilter().getType().getElementType(); }

Type ConvBwdDataOp::getAType() {
  return getFilter().getType().getElementType();
}

Type ConvBwdWeightOp::getAType() {
  return getOutput().getType().getElementType();
}

Type ConvOp::getBType() { return getInput().getType().getElementType(); }

Type ConvBwdDataOp::getBType() {
  return getOutput().getType().getElementType();
}

Type ConvBwdWeightOp::getBType() {
  return getInput().getType().getElementType();
}

Type ConvOp::getCType() { return getOutput().getType().getElementType(); }

Type ConvBwdDataOp::getCType() { return getInput().getType().getElementType(); }

Type ConvBwdWeightOp::getCType() {
  return getFilter().getType().getElementType();
}

OpOperand *ConvOp::getOutArgument() { return &(*this)->getOpOperand(2); }

OpOperand *ConvBwdDataOp::getOutArgument() { return &(*this)->getOpOperand(1); }

OpOperand *ConvBwdWeightOp::getOutArgument() {
  return &(*this)->getOpOperand(0);
}

GemmSize ConvOp::getGemmSize() {
  auto sizes = ConvolutionDims::fromOp(*this);
  return GemmSize::fromConvolution(ConvOpType::Fwd, sizes);
}

GemmSize ConvBwdDataOp::getGemmSize() {
  auto sizes = ConvolutionDims::fromOp(*this);
  auto padding = extractFromIntegerArrayAttr<int64_t>(this->getPadding());
  auto strides = extractFromIntegerArrayAttr<int64_t>(this->getStrides());
  auto dilations = extractFromIntegerArrayAttr<int64_t>(this->getDilations());
  int64_t kernelId = getKernelId().getSExtValue();

  SmallVector<int64_t, 5> gcdStrideDilations;
  assert(strides.size() == dilations.size());
  for (const auto &[stride, dilation] : zip(strides, dilations)) {
    gcdStrideDilations.push_back(math_util::gcd(stride, dilation));
  }

  SmallVector<int64_t, 5> filTilda;
  for (const auto &[stride, gcdSD] : zip(strides, gcdStrideDilations)) {
    filTilda.push_back(stride / gcdSD);
  }

  SmallVector<int64_t, 5> outTilda;
  for (const auto &[out, dilation, fil, stride] :
       zip(sizes.out, dilations, sizes.fil, strides)) {
    outTilda.push_back(
        out + math_util::integer_divide_ceil(dilation * (fil - 1), stride));
  }

  SmallVector<int64_t, 5> iTildaLeft;
  SmallVector<int64_t, 5> iTildaRight;
  for (const auto &[padindex, dilation, tilda, stride] :
       enumerate(dilations, filTilda, strides)) {
    iTildaLeft.push_back(math_util::integer_divide_floor(
        std::max((int64_t)0, padding[2 * padindex] - dilation * (tilda - 1)),
        stride));
  }
  for (const auto &[padindex, out, in, stride] :
       enumerate(outTilda, sizes.in, strides)) {
    iTildaRight.push_back(std::min(
        out,
        math_util::integer_divide_ceil(padding[2 * padindex] + in - 1, stride) +
            1));
  }

  SmallVector<int64_t, 5> tildaSlice;
  for (const auto &[right, left] : zip(iTildaRight, iTildaLeft))
    tildaSlice.push_back(right - left);

  SmallVector<int64_t, 3> iTilda;
  SmallVector<int64_t, 3> iDotSlice;
  int64_t product = 1;
  for (size_t i = 1; i < sizes.fil.size(); i++)
    product *= filTilda[i];
  int64_t divisor = 1;
  iTilda.resize(sizes.fil.size());
  switch (sizes.fil.size()) {
  default:
    llvm_unreachable("Only 2-D and 3-D have been implemented.");
    break;
  case 3:
    divisor = filTilda[2];
    iTilda[2] = kernelId % divisor;
    [[fallthrough]];
  case 2:
    iTilda[1] = (kernelId % product) / divisor;
    iTilda[0] = kernelId / product;
  }
  for (size_t i = 0; i < sizes.fil.size(); i++)
    iDotSlice.push_back(
        math_util::integer_divide_ceil(sizes.fil[i] - iTilda[i], filTilda[i]));

  int64_t g = sizes.g;
  int64_t m = sizes.c;
  int64_t k = sizes.k;
  for (auto ds : iDotSlice)
    k *= ds;
  int64_t n = sizes.n;
  for (auto ts : tildaSlice)
    n *= ts;

  return GemmSize(g, m, k, n);
}

GemmSize ConvBwdWeightOp::getGemmSize() {
  auto sizes = ConvolutionDims::fromOp(*this);
  return GemmSize::fromConvolution(ConvOpType::BwdWeight, sizes);
}

void ConvOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getOutputMutable(),
                       transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
                       transform::TransformMappingResource::get());

  effects.emplace_back(MemoryEffects::Read::get(), &getFilterMutable(),
                       transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
                       transform::TransformMappingResource::get());
}

void ConvBwdDataOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
                       transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getInputMutable(),
                       transform::TransformMappingResource::get());

  effects.emplace_back(MemoryEffects::Read::get(), &getFilterMutable(),
                       transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getOutputMutable(),
                       transform::TransformMappingResource::get());
}

void ConvBwdWeightOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  const bool hasWorkspace = getWorkspace() != nullptr;
  if (hasWorkspace) {
    OpOperand *wsm = &getWorkspaceMutable()[0];
    effects.emplace_back(MemoryEffects::Read::get(), wsm,
                         transform::TransformMappingResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), wsm,
                         transform::TransformMappingResource::get());
  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getFilterMutable(),
                         transform::TransformMappingResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &getFilterMutable(),
                         transform::TransformMappingResource::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
                       transform::TransformMappingResource::get());

  effects.emplace_back(MemoryEffects::Read::get(), &getOutputMutable(),
                       transform::TransformMappingResource::get());
}

//===-----------------------------------------------------===//
// GemmOp
//===-----------------------------------------------------===//

LogicalResult GemmOp::verify() {
  ShapedType typeA = getA().getType(), typeB = getB().getType(),
             typeC = getC().getType();
  Type inElems = typeA.getElementType(), outElems = typeC.getElementType();
  // The integer gemm will produce i32 and then truncate/extend to the requested
  // iN e.g. i8.
  if (isa<FloatType>(inElems) && !isa<FloatType>(outElems))
    return emitOpError(
        "float-valued inputs must have a floating-point output type");

  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape(),
                    dimsC = typeC.getShape();
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1,
          offsetC = dimsC.size() == 2 ? 0 : 1;
  int64_t gA = offsetA ? dimsA[0] : 1, gB = offsetB ? dimsB[0] : 1,
          gC = offsetC ? dimsC[0] : 1;
  int64_t mA = dimsA[offsetA + (getATransposed() ? 1 : 0)],
          kA = dimsA[offsetA + (getATransposed() ? 0 : 1)],
          kB = dimsB[offsetB + (getBTransposed() ? 1 : 0)],
          nB = dimsB[offsetB + (getBTransposed() ? 0 : 1)],
          mC = dimsC[offsetC + (getCTransposed() ? 1 : 0)],
          nC = dimsC[offsetC + (getCTransposed() ? 0 : 1)];
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

  bool isXdlops = bitEnumContainsAll(getFeatures(), GemmFeatures::mfma);
  bool isWmma = bitEnumContainsAll(getFeatures(), GemmFeatures::wmma);
  if (Attribute params = this->getParams().value_or(nullptr)) {
    if (isXdlops &&
        !isa<XdlopsGemmParamsAttr, XdlopsGemmDerivedParamsAttr>(params))
      return emitOpError("an xdlops GEMM has non-xdlops tuning parameters");
    if (getFeatures() == GemmFeatures::none &&
        !isa<GeneralGemmParamsAttr>(params))
      return emitOpError("an all-hardware gemm must used the general gemm "
                         "tuning parameters");
    if (getDerivedBlockSize().has_value() &&
        isa<GeneralGemmParamsAttr>(params)) {
      return emitOpError(
          "cannot have derivedBlockSize when gemm has generalGemmParams");
    }
  }

  if (getDerivedBlockSize().has_value() && !isXdlops && !isWmma) {
    return emitOpError(
        "general gemm kernels shouldn't have derived block size.");
  }

  RockGemmWrapperInterface gemmIfaceOp =
      cast<RockGemmWrapperInterface>(this->getOperation());
  if (failed(verifyGemmTypes(gemmIfaceOp)))
    return failure();
  return success();
}

KernelType GemmOp::getKernelType() { return KernelType::Gemm; }

Type GemmOp::getAType() { return getA().getType().getElementType(); }

Type GemmOp::getBType() { return getB().getType().getElementType(); }

Type GemmOp::getCType() { return getC().getType().getElementType(); }

OpOperand *GemmOp::getOutArgument() { return &(*this)->getOpOperand(2); }

GemmSize GemmOp::getGemmSize() {
  ShapedType typeA = getA().getType(), typeB = getB().getType();
  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape();
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1;
  int64_t g = offsetA ? dimsA[0] : 1,
          m = dimsA[offsetA + (getATransposed() ? 1 : 0)],
          k = dimsA[offsetA + (getATransposed() ? 0 : 1)],
          n = dimsB[offsetB + (getBTransposed() ? 0 : 1)];
  return GemmSize(g, m, k, n);
}

//===-----------------------------------------------------===//
// GridwiseGemmOp and GridwiseGemmAccel Op
//===-----------------------------------------------------===//
template <typename GridOp>
static LogicalResult verifyGridwiseGemm(GridOp op) {
  MemRefType aType = op.getA().getType(), bType = op.getB().getType(),
             cType = op.getC().getType();
  Type aElem = aType.getElementType(), bElem = bType.getElementType(),
       cElem = cType.getElementType();

  if (failed(
          verifyGemmTypes(op, op.getFeatures(), "gfx00", aElem, bElem, cElem)))
    return failure();
  if (aElem.isInteger(8) && !(cElem.isInteger(32) || cElem.isInteger(8)))
    return op.emitOpError("i8 input requires i32 or i8 output");
  if (isFloat8Type(aElem) && !cElem.isF32())
    return op.emitOpError("8-bit float input requires f32 output");

  ArrayRef<int64_t> aShape = aType.getShape(), bShape = bType.getShape(),
                    cShape = cType.getShape();
  int64_t g = aShape[0], k = aShape[1], m = aShape[2], n = bShape[2];
  if (bShape[0] != g || cShape[0] != g) {
    return op.emitOpError("Mismatched G dimensions in matrix multiply;")
           << " A[0] = " << g << " b[0] = " << bShape[0]
           << " C[0] = " << cShape[0];
  }
  if (cShape[1] != m)
    return op.emitOpError("Mismatched M dimensions in matrix multiply:")
           << " A[2] = " << m << " C[1] = " << cShape[1];
  if (bShape[1] != k)
    return op.emitOpError("Mismatched K dimensions in matrix multiply:")
           << " A[1] = " << k << " B[1] = " << bShape[1];
  if (cShape[2] != n)
    return op.emitOpError("Mismatched N dimensions in matrix multiply:")
           << " B[2] = " << n << " C[2] = " << cShape[2];

  constexpr int64_t intMax = std::numeric_limits<int32_t>::max();
  if (g > intMax)
    return op.emitOpError("G dimmension ")
           << g << " cannot be greater than int32_max " << intMax;
  if (m > intMax)
    return op.emitOpError("M dimmension ")
           << m << " cannot be greater than int32_max " << intMax;
  if (k > intMax)
    return op.emitOpError("K dimmension ")
           << k << " cannot be greater than int32_max " << intMax;
  if (n > intMax)
    return op.emitOpError("N dimmension ")
           << n << " cannot be greater than int32_max " << intMax;

  return success();
}

LogicalResult GridwiseGemmOp::verify() { return verifyGridwiseGemm(*this); }

LogicalResult GridwiseGemmAccelOp::verify() {
  return verifyGridwiseGemm(*this);
}

//===-----------------------------------------------------===//
// ExtractSliceOp
//===-----------------------------------------------------===//
LogicalResult ExtractSliceOp::canonicalize(ExtractSliceOp op,
                                           PatternRewriter &b) {
  // Extracting a vector of the same size as the source is a no-op, since it
  // has to happen from index 0 to ensure legality
  if (op.getResult().getType() == op.getVector().getType()) {
    b.replaceOp(op, op.getVector());
    return success();
  }
  return failure();
}

LogicalResult ExtractSliceOp::verify() {
  if (auto destType = dyn_cast<VectorType>(getResult().getType())) {
    size_t destSize = destType.getDimSize(0);
    size_t sourceSize = getVector().getType().getDimSize(0);
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
  if (op.getSource().getType() == op.getDest().getType()) {
    b.replaceOp(op, op.getSource());
    return success();
  }
  return failure();
}
LogicalResult InsertSliceOp::verify() {
  if (auto sourceType = dyn_cast<VectorType>(getSource().getType())) {
    size_t sourceSize = sourceType.getDimSize(0);
    size_t destSize = getDest().getType().getDimSize(0);
    if (sourceSize > destSize)
      return emitOpError(
          "Slice to store has length " + Twine(sourceSize) +
          " which is longer than destinanation's vector length " +
          Twine(destSize));
  }
  return success();
}

//===-----------------------------------------------------===//
// GpuAllocOp
//===-----------------------------------------------------===//

static int64_t getSize(MemRefType memref) {
  int64_t elementSize;
  Type type = memref.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type)) {
    elementSize =
        (vecType.getElementTypeBitWidth() * vecType.getNumElements()) / 8;
  } else {
    elementSize = type.getIntOrFloatBitWidth() / 8;
  }
  return memref.getNumElements() * elementSize;
}

LogicalResult GpuAllocOp::verify() {
  // Make sure the size is bigger than 0
  if (getSize(getOutput().getType()) > 0) {
    return success();
  }
  return emitError("The size of rock.alloc should be greather than zero.");
}

//===-----------------------------------------------------===//
// GpuDeallocOp
//===-----------------------------------------------------===//

LogicalResult GpuDeallocOp::verify() {
  // Make sure the input memref defining operation is a GpuAllocOp
  if (auto gpuAlloc = dyn_cast<GpuAllocOp>(getMemref().getDefiningOp())) {
    // Make sure the size is bigger than 0
    if (getSize(getMemref().getType()) > 0) {
      return success();
    }
    return emitError("The size of rock.dealloc should be greather than zero.");
  }
  return emitError("The operand of rock.dealloc must be the result of a "
                   "rock.alloc operation.");
}

//===-----------------------------------------------------===//
// ExtractMultiBufferOp
//===-----------------------------------------------------===//

LogicalResult ExtractMultiBufferOp::verify() {
  // Make sure the output buffer has the same type of
  // the buffers we are selecting from
  auto outputType = getOutput().getType();
  for (auto buffer : getBuffers())
    if (outputType != buffer.getType())
      return failure();
  return success();
}

//===-----------------------------------------------------===//
// TransformingForOp
//===-----------------------------------------------------===//

static ArrayAttr maybeIndexArray(OpBuilder &b,
                                 std::optional<ArrayRef<int64_t>> vals) {
  return llvm::transformOptional(
             vals, [&b](ArrayRef<int64_t> v) { return b.getIndexArrayAttr(v); })
      .value_or(ArrayAttr{});
}

void TransformingForOp::build(OpBuilder &b, OperationState &state,
                              ArrayRef<ValueRange> inits,
                              ArrayRef<Attribute> transforms,
                              ArrayRef<int64_t> bounds,
                              std::optional<ArrayRef<int64_t>> strides,
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
                              std::optional<ArrayRef<int64_t>> strides,
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
  state.addAttribute(getBoundsAttrName(state.name), bounds);
  if (!strides) {
    SmallVector<int64_t> strideVec(bounds.size(), 1LL);
    strides = b.getIndexArrayAttr(strideVec);
  }
  state.addAttribute(getStridesAttrName(state.name), strides);
  state.addAttribute(getTransformsAttrName(state.name), transforms);
  if (forceUnroll)
    state.addAttribute(getForceUnrollAttrName(state.name), b.getUnitAttr());
  if (useIndexDiffs)
    state.addAttribute(getUseIndexDiffsAttrName(state.name), b.getUnitAttr());

  int32_t upperLen = bounds.size();
  for (ValueRange upper : inits)
    state.addOperands(upper);
  state.addOperands(iterArgs);
  state.addTypes(iterArgs.getTypes());

  // Track sizes of variadic arguments to enable them to be looped up
  state.addAttribute(
      TransformingForOp::getOperandSegmentSizeAttr(),
      b.getDenseI32ArrayAttr({upperLen * static_cast<int32_t>(inits.size()),
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
      len = cast<TransformMapAttr>(domain[domain.size() - 1])
                .getLowerBounds()
                .size();
    for (int32_t i = 0; i < len; ++i)
      bodyBlock.addArgument(indexType, state.location);
    nLower += len;
  }
  // Validity arguments
  lowerStarts.push_back(nLower);
  int32_t nTransforms = transforms.size();
  for (int32_t i = 0; i < nTransforms; ++i) {
    bodyBlock.addArgument(b.getI1Type(), state.location);
  }
  nLower += nTransforms;
  // Iteration arguments
  lowerStarts.push_back(nLower);
  state.addAttribute(getLowerStartsAttrName(state.name),
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
        for (size_t i = oldNLower; i < lowerArgs.size(); ++i) {
          lowerArgs[i].type = indexTy;
        }
        ArrayAttr theseTransforms;
        if (parser.parseAttribute(theseTransforms)) {
          return failure();
        }
        if (parser.parseOperandList(upperInits, Delimiter::Paren)) {
          return failure();
        }
        if (theseTransforms.empty()) {
          if (upperInits.size() - oldNUpper != lowerArgs.size() - oldNLower) {
            return parser.emitError(loopIterLoc,
                                    "Expected same number of lower and upper "
                                    "arguments when transforms absent");
          }
        } else {
          for (Attribute a : theseTransforms) {
            if (!isa<TransformMapAttr>(a)) {
              return parser.emitError(loopIterLoc,
                                      "Expected transform map attributes");
            }
          }
          size_t nInputs = cast<TransformMapAttr>(theseTransforms[0])
                               .getMap()
                               .getAffineMap()
                               .getNumInputs();
          size_t nOutputs = cast<TransformMapAttr>(
                                theseTransforms[theseTransforms.size() - 1])
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
  size_t preValiditiesNLower = lowerArgs.size();
  // Validity arguments.
  llvm::SMLoc validitiesLoc = parser.getCurrentLocation();
  if (parser.parseArgumentList(lowerArgs, Delimiter::Paren) ||
      parser.parseEqual() || parser.parseKeyword("validity")) {
    return failure();
  }
  if (lowerArgs.size() - preValiditiesNLower != transforms.size())
    return parser.emitError(
        validitiesLoc, "Expected " + Twine(transforms.size()) +
                           " validity arguments, one per domain, but found " +
                           Twine(lowerArgs.size() - preValiditiesNLower));
  for (size_t i = preValiditiesNLower, e = lowerArgs.size(); i < e; ++i) {
    lowerArgs[i].type = b.getI1Type();
  }

  lowerStarts.push_back(lowerArgs.size());
  result.addAttribute(TransformingForOp::getTransformsAttrName(result.name),
                      b.getArrayAttr(transforms));
  result.addAttribute(TransformingForOp::getLowerStartsAttrName(result.name),
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
  result.addAttribute(TransformingForOp::getBoundsAttrName(result.name),
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
  result.addAttribute(TransformingForOp::getStridesAttrName(result.name),
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
      b.getDenseI32ArrayAttr({static_cast<int32_t>(upperInits.size()),
                              static_cast<int32_t>(iterInits.size())}));
  return success();
}

void TransformingForOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{TransformingForOp::getOperandSegmentSizeAttr(),
                       getTransformsAttrName(), getLowerStartsAttrName(),
                       getBoundsAttrName(), getStridesAttrName()});
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

  p << " (";
  p.printOperands(getValidities());
  p << ") = validity";

  if (!getIterInits().empty()) {
    p << " iter_args (";
    llvm::interleaveComma(llvm::zip(getIterArgs(), getIterInits()), p,
                          [&](auto i) {
                            Value init = std::get<1>(i);
                            p << std::get<0>(i) << " = " << init;
                          });
    p << ") -> (" << getIterInits().getTypes() << ")";
  }
  p << " bounds [";
  llvm::interleaveComma(getBounds().getAsValueRange<IntegerAttr>(), p,
                        [&](const llvm::APInt &bound) { p << bound; });
  p << "] ";
  p << "strides [";
  llvm::interleaveComma(getStrides().getAsValueRange<IntegerAttr>(), p,
                        [&](const llvm::APInt &stride) { p << stride; });
  p << "] ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
}

LogicalResult TransformingForOp::verify() {
  if (getBounds().empty())
    return emitOpError("Must have at least one iteration dimension");
  if (getBounds().size() != getStrides().size())
    return emitOpError("Bounds list and strides list must have same length");

  for (size_t i = 0, e = getBounds().size(); i < e; ++i) {
    int64_t bound = cast<IntegerAttr>(getBounds()[i]).getInt();
    int64_t stride = cast<IntegerAttr>(getStrides()[i]).getInt();
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
  if (getLowerStarts().size() != domains() + 2) {
    return emitOpError(
        "Lower starts attribute doesn't have one entry per domain plus 2");
  }
  if (getLowerStart(domains() + 1) - getLowerStart(domains()) != domains()) {
    return emitOpError("Validity domain doesn't contain one value per domain");
  }
  if (getLowerStart(0) != 0) {
    return emitOpError("Region args don't start with lower coords");
  }

  for (uint32_t i = 0, e = domains(); i < e; ++i) {
    ArrayAttr transforms = getTransforms(i);
    auto lowerArgs = getLowerCoords(i);
    auto upperInits = getUpperInits(i);
    if (transforms.empty()) {
      if (upperInits.size() != lowerArgs.size()) {
        return emitOpError("Mismatch between number of lower and upper "
                           "coordinates without a transform in domain #" +
                           Twine(i));
      }
    } else {
      size_t nUpper = cast<TransformMapAttr>(transforms[0])
                          .getMap()
                          .getValue()
                          .getNumInputs();
      size_t nLower = cast<TransformMapAttr>(transforms[transforms.size() - 1])
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
    if (getLowerStart(i + 1) != lowerArgsCount) {
      return emitOpError("Lower starts attribute not accurate after domain #" +
                         Twine(i));
    }
  }
  return success();
}

// Cribbed from AffineForOp
bool TransformingForOp::isDefinedOutsideOfLoop(Value value) {
  return !getRegion().isAncestor(value.getParentRegion());
}
void TransformingForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(*this);
}
SmallVector<Region *> TransformingForOp::getLoopRegions() {
  return {&getRegion()};
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
  TransformMapAttr transform = getMap();
  size_t nLowerIn = getLowerOrig().size();
  size_t nLowerOut = getLowerIndices().size();

  if (nLowerIn != nLowerOut)
    return emitOpError("Got " + Twine(nLowerIn) + " lower inputs but " +
                       Twine(nLowerOut) + " lower outputs");

  size_t nUpper = getUpperDiffs().size();
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
// GlobalLoadOp
//===-----------------------------------------------------===//
LogicalResult GlobalLoadOp::verify() {
  MemRefType sourceType = getSource().getType();
  size_t nDims = sourceType.getRank();

  if (getSourceCoord().size() != nDims)
    return emitOpError("Expected " + Twine(nDims) + " coordinates for load");
  if (getCanReadOffEnd() && nDims != 1)
    return emitOpError("can only have one dimension in canReadOffEnd loads");
  Attribute memSpaceAttr = sourceType.getMemorySpace();
  auto gpuMemSpaceAttr = dyn_cast_or_null<gpu::AddressSpaceAttr>(memSpaceAttr);
  if (memSpaceAttr && (!gpuMemSpaceAttr ||
                       gpuMemSpaceAttr.getValue() != gpu::AddressSpace::Global))
    return emitOpError("Source memref must live in global memory");
  return success();
}

//===-----------------------------------------------------===//
// GlobalStoreOp
//===-----------------------------------------------------===//
LogicalResult GlobalStoreOp::verify() {
  MemRefType destType = getDest().getType();
  size_t nDims = destType.getRank();
  if (getDestCoord().size() != nDims)
    return emitOpError("Expected " + Twine(nDims) + " coordinates for store");
  if (getCanStoreOffEnd() && nDims != 1)
    return emitOpError("can only have one dimension in a canStoreOffEnd write");
  Attribute memSpaceAttr = destType.getMemorySpace();
  auto gpuMemSpaceAttr = dyn_cast_or_null<gpu::AddressSpaceAttr>(memSpaceAttr);
  if (memSpaceAttr && (!gpuMemSpaceAttr ||
                       gpuMemSpaceAttr.getValue() != gpu::AddressSpace::Global))
    return emitOpError("Destination memref must live in global memory");
  if (getStoreMethod() == StoreMethod::AtomicMax &&
      isa<FloatType>(destType.getElementType()))
    if (!destType.getElementType().isF32())
      return emitOpError("atomic max for floats only supports f32");
  return success();
}

//===-----------------------------------------------------===//
// InBoundsLoadOp
//===-----------------------------------------------------===//
LogicalResult InBoundsLoadOp::verify() {
  MemRefType sourceType = getSource().getType();
  size_t nDims = sourceType.getRank();
  if (getCoords().size() != nDims)
    return emitOpError("Expected " + Twine(nDims) + " coordinates for load");
  Type resultType = getResult().getType();
  if (isa<ShapedType>(resultType) && !isa<VectorType>(resultType))
    return emitOpError(
        "Non-scalar loads must return vectors, not other shaped types");
  return success();
}

//===-----------------------------------------------------===//
// InBoundsLoadOp
//===-----------------------------------------------------===//
LogicalResult InBoundsStoreOp::verify() {
  MemRefType destType = getDest().getType();
  size_t nDims = destType.getRank();
  if (getCoords().size() != nDims)
    return emitOpError("Expected " + Twine(nDims) + " coordinates for store");
  Type dataType = getData().getType();
  if (isa<ShapedType>(dataType) && !isa<VectorType>(dataType))
    return emitOpError(
        "Non-scalar data types must be vectors, not other shaped types");
  return success();
}

//===-----------------------------------------------------===//
// ThreadwiseReadIntoOp
//===-----------------------------------------------------===//
SmallPtrSet<OpOperand *, 2> ThreadwiseReadIntoOp::getAcceptingViewOperands() {
  auto operands = getOperation()->getOpOperands();
  return {operands.begin()};
}

std::optional<OperandRange>
ThreadwiseReadIntoOp::getExtraIndices(OpOperand &operand) {
  if (!getAcceptingViewOperands().contains(&operand)) {
    return std::nullopt;
  }
  // Only one operand supports view
  return getExtraIndices();
}

Operation *
ThreadwiseReadIntoOp::cloneWithExtraIndices(OpBuilder &builder,
                                            OpOperand &operand, Value view,
                                            ArrayRef<Value> newExtraIndices) {
  if (!getAcceptingViewOperands().contains(&operand)) {
    return getOperation();
  }
  // Only one operand supports view
  auto newOp = builder.create<ThreadwiseReadIntoOp>(
      getLoc(), view, getDest(), getExtraViews(), newExtraIndices,
      getForceUnroll(), getUseIndexDiffs());
  return newOp.getOperation();
}

LogicalResult ThreadwiseReadIntoOp::verify() {
  MemRefType destType = getDest().getType();
  MemRefType srcType = getSource().getType();
  Attribute dstMemSpaceAttr = destType.getMemorySpace();
  Attribute srcMemSpaceAttr = srcType.getMemorySpace();
  auto gpuDstMemSpaceAttr =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(dstMemSpaceAttr);
  auto gpuSrcMemSpaceAttr =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(srcMemSpaceAttr);
  if (dstMemSpaceAttr &&
      (!gpuDstMemSpaceAttr ||
       gpuDstMemSpaceAttr.getValue() != gpu::AddressSpace::Private))
    return emitOpError("dest must be private registers");
  ArrayAttr extraViews = getExtraViews();
  ArrayRef<int64_t> inputShape;
  if (extraViews.empty())
    inputShape = getSource().getType().getShape();
  else
    inputShape = cast<TransformMapAttr>(extraViews[0]).getUpperBounds();

  size_t extraIdxCount = getExtraIndices().size();
  if (inputShape.empty()) {
    if (extraIdxCount != 0)
      return emitOpError("read from a scalar value cannot have coordinates");
  } else if (inputShape.size() != extraIdxCount + 1) {
    return emitOpError("source view must be extraIndices + 1");
  }

  // Add more constraints if we see vector buffers (e.g.,
  // memref<Kxvector<vxf16>>)
  VectorType srcVectorType = dyn_cast<VectorType>(srcType.getElementType());
  VectorType dstVectorType = dyn_cast<VectorType>(destType.getElementType());
  if ((srcVectorType || dstVectorType) &&
      gpuSrcMemSpaceAttr.getValue() != gpu::AddressSpace::Workgroup &&
      gpuSrcMemSpaceAttr.getValue() != gpu::AddressSpace::Private)
    return emitOpError(
        "Vector buffers are not allowed when we read from global memory");
  if (srcVectorType && dstVectorType) {
    int64_t srcVectorLen = srcVectorType.getNumElements();
    int64_t dstVectorLen = dstVectorType.getNumElements();
    if ((srcVectorLen > dstVectorLen && srcVectorLen % dstVectorLen != 0) ||
        (dstVectorLen > srcVectorLen && dstVectorLen % dstVectorLen != 0))
      return emitOpError(
          "Vector buffers vector's lengths need to be evenly divisible");
  }

  return success();
}

//===-----------------------------------------------------===//
// ThreadwiseWriteAllOp
//===-----------------------------------------------------===//

SmallPtrSet<OpOperand *, 2> ThreadwiseWriteAllOp::getAcceptingViewOperands() {
  auto operands = getOperation()->getOpOperands();
  return {operands.begin() + 1};
}

std::optional<OperandRange>
ThreadwiseWriteAllOp::getExtraIndices(OpOperand &operand) {
  if (!getAcceptingViewOperands().contains(&operand)) {
    return std::nullopt;
  }
  // Only one operand supports view
  return getExtraIndices();
}

Operation *
ThreadwiseWriteAllOp::cloneWithExtraIndices(OpBuilder &builder,
                                            OpOperand &operand, Value view,
                                            ArrayRef<Value> newExtraIndices) {
  if (!getAcceptingViewOperands().contains(&operand)) {
    return getOperation();
  }
  // Only one operand supports view
  auto newOp = builder.create<ThreadwiseWriteAllOp>(
      getLoc(), getSource(), view, getExtraViews(), newExtraIndices,
      getFeatures(), getStoreMethod(), getForceUnroll(), getUseIndexDiffs());
  return newOp.getOperation();
}

LogicalResult ThreadwiseWriteAllOp::verify() {
  MemRefType sourceType = getSource().getType();
  Attribute memSpaceAttr = sourceType.getMemorySpace();
  auto gpuMemSpaceAttr = dyn_cast_or_null<gpu::AddressSpaceAttr>(memSpaceAttr);
  if (memSpaceAttr && (!gpuMemSpaceAttr || gpuMemSpaceAttr.getValue() !=
                                               gpu::AddressSpace::Private))
    return emitOpError("source must be private registers");
  ArrayAttr extraViews = getExtraViews();
  ArrayRef<int64_t> outputShape;
  if (extraViews.empty())
    outputShape = getDest().getType().getShape();
  else
    outputShape = cast<TransformMapAttr>(extraViews[0]).getUpperBounds();

  size_t extraIdxCount = getExtraIndices().size();
  if (outputShape.empty()) {
    if (extraIdxCount != 0)
      return emitOpError("write to a scalar must have no coordinates");
  } else if (outputShape.size() != extraIdxCount + 1) {
    return emitOpError("dest view must be extraIndices + 1");
  }
  return success();
}

//===-----------------------------------------------------===//
// ThreadwiseCopyOp
//===-----------------------------------------------------===//
SmallPtrSet<OpOperand *, 2> ThreadwiseCopyOp::getAcceptingViewOperands() {
  auto operands = getOperation()->getOpOperands();
  int extraIndicesOpPos = (getExtraIndicesSource().empty() ? 0 : 1);
  return {operands.begin(), operands.begin() + extraIndicesOpPos + 1};
}

std::optional<OperandRange>
ThreadwiseCopyOp::getExtraIndices(OpOperand &operand) {
  if (!getAcceptingViewOperands().contains(&operand))
    return std::nullopt;
  return (operand.getOperandNumber() == 0 ? getExtraIndicesSource()
                                          : getExtraIndicesDest());
}

Operation *
ThreadwiseCopyOp::cloneWithExtraIndices(OpBuilder &builder, OpOperand &operand,
                                        Value view,
                                        ArrayRef<Value> newExtraIndices) {
  if (!getAcceptingViewOperands().contains(&operand))
    return getOperation();

  // Only one operand supports view
  ThreadwiseCopyOp newOp;
  if (operand.getOperandNumber() == 0) {
    newOp = builder.create<ThreadwiseCopyOp>(
        getLoc(), view, newExtraIndices, getDest(), getExtraIndicesDest(),
        getForceUnroll(), getUseIndexDiffs());
  } else {
    newOp = builder.create<ThreadwiseCopyOp>(
        getLoc(), getSource(), getExtraIndicesSource(), view, newExtraIndices,
        getForceUnroll(), getUseIndexDiffs());
  }
  return newOp.getOperation();
}

LogicalResult ThreadwiseCopyOp::verify() {
  auto srcShape = getSource().getType().getShape();
  auto dstShape = getDest().getType().getShape();
  // We can ignore the external indices, if there are any
  size_t extraIndicesDestSize = getExtraIndicesDest().size();
  size_t extraIndicesSourceSize = getExtraIndicesSource().size();
  SmallVector<int64_t> unextendedSrcShape(
      srcShape.begin() + extraIndicesSourceSize, srcShape.end());
  SmallVector<int64_t> unextendedDstShape(
      dstShape.begin() + extraIndicesDestSize, dstShape.end());
  if (unextendedDstShape != unextendedSrcShape)
    return emitOpError(
        "Un-extended source and dest buffers need to have the same shape.");

  return success();
}

//===----------------------------------------------------------------------===//
// BlockwiseGemmOp
//===----------------------------------------------------------------------===//

Value BlockwiseGemmOp::getDest() { return getMatrixC(); }

LogicalResult BlockwiseGemmOp::verify() {
  MemRefType blockAType = getMatrixA().getType(),
             blockBType = getMatrixB().getType();

  int64_t k = blockAType.getShape()[0];
  int64_t kPack = blockAType.getShape()[2];

  if (k != blockBType.getShape()[0]) {
    return emitOpError("Mismatched k dimensions between A and B");
  }
  if (kPack != blockBType.getShape()[2]) {
    return emitOpError("Mismatched kPack between A and B");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ThreadwiseGemmOp
//===----------------------------------------------------------------------===//
LogicalResult ThreadwiseGemmOp::verify() {
  ArrayRef<int64_t> aShape = getMatrixA().getType().getShape(),
                    bShape = getMatrixB().getType().getShape(),
                    cShape = getMatrixC().getType().getShape();

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
// ThreadwiseAccelGemmOp
//===----------------------------------------------------------------------===//
LogicalResult ThreadwiseAccelGemmOp::verify() {
  ArrayRef<int64_t> aShape = getMatrixA().getType().getShape(),
                    bShape = getMatrixB().getType().getShape(),
                    cShape = getMatrixC().getType().getShape();

  if (aShape.size() != 2)
    return emitOpError("A shape should be [M,K]");
  if (bShape.size() != 2)
    return emitOpError("B shape should be [N,K]");
  if (aShape.back() != bShape.back())
    return emitOpError("A and B K dimensions don't match");
  if (cShape.size() != 2)
    return emitOpError("C shape should be [M,N]");
  if (getComputeIndices().size() != 3)
    return emitOpError("ComputeIndices need to be a <i,j,k> tuple");

  return success();
}

//===----------------------------------------------------------------------===//
// GridwiseAttentionAccelOp
//===----------------------------------------------------------------------===//
LogicalResult GridwiseAttentionAccelOp::verify() {
  RockAccelTuningParamAttrInterface gemm0TuningParams = getParams0();
  int64_t gemm0kpack = gemm0TuningParams.getKpack();
  int64_t gemm0NPerBlock = gemm0TuningParams.getNPerBlock();
  if (gemm0NPerBlock % gemm0kpack != 0) {
    return emitError("NPerBlock should be divisble by kpack.");
  }

  int64_t linalgOpCount = 0;
  getPreSoftmaxBody().walk([&](linalg::GenericOp genOp) { linalgOpCount++; });
  if (linalgOpCount > 1) {
    return emitError(
        "More than 1 linalg generic op found in pre softmax fusion point.");
  }
  return success();
}

void GridwiseAttentionAccelOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();
  effects.emplace_back(read, &getOutMutable());
  effects.emplace_back(write, &getOutMutable());

  effects.emplace_back(read, &getQueriesMutable());
  effects.emplace_back(read, &getKeysMutable());
  effects.emplace_back(read, &getValuesMutable());
  for (auto &regionArg : getPreSoftmaxElemWiseInputsMutable())
    effects.emplace_back(read, &regionArg);
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
            dyn_cast_or_null<IntegerAttr>(container->getAttr(idName))) {
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

//===-----------------------------------------------------===//
// ReduceOp
//===-----------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  APInt axis = getAxis();
  ArrayRef<int64_t> inpShape = cast<ShapedType>(getIn().getType()).getShape();
  for (const auto &dimAndSize :
       llvm::enumerate(cast<ShapedType>(getOut().getType()).getShape())) {
    size_t dim = dimAndSize.index();
    int64_t dimSize = dimAndSize.value();
    if (dim == axis) {
      if (dimSize != 1) {
        return emitError("The size of the reduction dimension should be 1.");
      }
    } else {
      if (dimSize != inpShape[dim]) {
        return emitError("The size of the non-reduction dimension should "
                         "match the input.");
      }
    }
  }
  return success();
}

//===-----------------------------------------------------===//
// Blockwise_ReduceOp
//===-----------------------------------------------------===//

LogicalResult BlockwiseBroadcastReduceOp::verify() {
  ArrayAttr inputViewArrayAttr = getInputRegViewAttr();
  // This view should be {tid, iter} to {d0, ... , Dr , ... , dn};
  // where {d0, ... , Dr , ... , dn} represent a blockwise tile
  // of a larger tensor that is being reduced.
  size_t inputViewArrLen = inputViewArrayAttr.size();
  ArrayRef<int64_t> inputTensorShape =
      cast<TransformMapAttr>(inputViewArrayAttr[inputViewArrLen - 1])
          .getLowerBounds()
          .asArrayRef();
  ArrayAttr tidSubTileSliceView = getTidSubTileSliceView();
  int64_t axis = getAxis().getSExtValue();
  size_t tidSubTileSliceViewArrLen = tidSubTileSliceView.size();
  ArrayRef<int64_t> inputPartialReductionTensorShape =
      cast<TransformMapAttr>(tidSubTileSliceView[tidSubTileSliceViewArrLen - 1])
          .getLowerBounds()
          .asArrayRef();
  ArrayRef<int64_t> inputThreadView =
      cast<TransformMapAttr>(inputViewArrayAttr[0])
          .getUpperBounds()
          .asArrayRef();
  ArrayRef<int64_t> wsShape = getWorkspaceBuffer().getType().getShape();
  int64_t blockSize = getBlockSize();

  gpu::AddressSpaceAttr inMemSpaceAttr =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(
          getInput().getType().getMemorySpace());
  if (!inMemSpaceAttr) {
    return emitError("No gpu memspace attr found in input memref; the input "
                     "memref should be in regs");
  } else {
    if (inMemSpaceAttr.getValue() != gpu::AddressSpace::Private) {
      return emitError("input should be in regs.");
    }
  }

  gpu::AddressSpaceAttr outMemSpaceAttr =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(
          getOutput().getType().getMemorySpace());
  if (!outMemSpaceAttr) {
    return emitError("No gpu memspace attr found in output memref; the output "
                     "memref should be in regs");
  } else {
    if (outMemSpaceAttr.getValue() != gpu::AddressSpace::Private) {
      return emitError("output should be in regs.");
    }
  }

  gpu::AddressSpaceAttr wsMemSpaceAttr =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(
          getWorkspaceBuffer().getType().getMemorySpace());
  if (!wsMemSpaceAttr) {
    return emitError("No gpu memspace attr found in workspace memref; the "
                     "workspace memref should be in LDS");
  } else {
    if (wsMemSpaceAttr.getValue() != gpu::AddressSpace::Workgroup) {
      return emitError("workspace should be in LDS.");
    }
  }

  if (inputThreadView[0] != blockSize) {
    return emitError("first dimension of the input view should be equal to "
                     "the block size");
  }
  if (wsShape.size() != 1) {
    return emitError("workspace LDS buffer should be flat");
  }

  int64_t blockwiseInputPartialReductionTensorElements = 1;
  for (auto [dim, dimSize] : llvm::enumerate(inputTensorShape)) {
    if ((int64_t)dim == axis) {
      blockwiseInputPartialReductionTensorElements *=
          inputPartialReductionTensorShape[axis];
    } else {
      blockwiseInputPartialReductionTensorElements *= dimSize;
    }
  }
  if (blockwiseInputPartialReductionTensorElements > wsShape[0]) {
    return emitError(
        "workspace should be at least the size of elements per block");
  }
  return success();
}

//===-----------------------------------------------------===//
// BlockwiseFillOp
//===-----------------------------------------------------===//

LogicalResult BlockwiseFillOp::verify() {
  MemRefType memrefType = getMemref().getType();
  if (memrefType.getRank() != 1) {
    return emitError("Blockwise fill expects a flat memref");
  }
  if (gpu::AddressSpaceAttr memSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(
          memrefType.getMemorySpace())) {
    if (memSpace.getValue() != gpu::AddressSpace::Workgroup) {
      return emitError("Memory space is expected to be workgroup");
    }
  } else {
    return emitError("Memory space is expected to be workgroup");
  }
  int64_t numElements = getMemref().getType().getNumElements();
  if (VectorType vecType = dyn_cast<VectorType>(getValue().getType())) {
    if (numElements % vecType.getNumElements() != 0) {
      return emitError("The vector length is not a factor in memref size.");
    }
  }
  return success();
}

//===-----------------------------------------------------===//
// AttentionOp
//===-----------------------------------------------------===//

LogicalResult AttentionOp::verify() {
  ShapedType qType = getQueries().getType();
  int64_t qBatchDim = qType.getShape().size() == 3 ? qType.getShape()[0] : 1;
  ArrayRef<int64_t> qLastDims = qType.getShape().slice(qType.getRank() - 2);
  auto [queryM, queryK] = getQTransposed()
                              ? std::tuple{qLastDims[1], qLastDims[0]}
                              : std::tuple{qLastDims[0], qLastDims[1]};

  ShapedType kType = getKeys().getType();
  int64_t kBatchDim = kType.getShape().size() == 3 ? kType.getShape()[0] : 1;
  ArrayRef<int64_t> kLastDims = kType.getShape().slice(kType.getRank() - 2);
  auto [keyK, keyN] = getKTransposed() ? std::tuple{kLastDims[1], kLastDims[0]}
                                       : std::tuple{kLastDims[0], kLastDims[1]};

  ShapedType vType = getValues().getType();
  int64_t vBatchDim = vType.getShape().size() == 3 ? vType.getShape()[0] : 1;
  ArrayRef<int64_t> vLastDims = vType.getShape().slice(vType.getRank() - 2);
  auto [valueK, valueN] = getVTransposed()
                              ? std::tuple{vLastDims[1], vLastDims[0]}
                              : std::tuple{vLastDims[0], vLastDims[1]};

  if (qBatchDim != kBatchDim || kBatchDim != vBatchDim) {
    return emitError("Batch dimensions do not match");
  }
  if (queryK != keyK) {
    return emitError("reduction dimensions of first gemm do not match");
  }
  if (keyN != valueK) {
    return emitError("reduction dimensions of second gemm do not match");
  }
  return success();
}

void AttentionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();
  effects.emplace_back(read, &getOutMutable());
  effects.emplace_back(write, &getOutMutable());

  effects.emplace_back(read, &getQueriesMutable());
  effects.emplace_back(read, &getKeysMutable());
  effects.emplace_back(read, &getValuesMutable());
  for (auto &regionArg : getPreSoftmaxElemWiseInputsMutable())
    effects.emplace_back(read, &regionArg);
}

//===-----------------------------------------------------===//
// AttentionPerfConfig Attr
//===-----------------------------------------------------===//

AttnPerfConfigAttr AttnPerfConfigAttr::get(StringAttr perfConfigStrAttr) {
  // Here a conventional c++ string split is being
  // done because MLIR lacks parseSourceString() method
  // to parse Attributes and its only there for Ops.
  StringRef perfConfigStrRef = perfConfigStrAttr.strref();
  StringRef token;
  StringRef rest;
  std::tie(token, rest) = perfConfigStrRef.split(':');
  if (token != "attn") {
    return {};
  }
  std::tie(token, rest) = rest.split(':');
  if (token.substr(0, 1) != "v") {
    return {};
  }
  int version;
  if (!llvm::to_integer(token.slice(1, StringRef::npos), version)) {
    return {};
  }
  if (version != 1) {
    return {};
  }
  SmallVector<StringRef, 8> tokens;
  rest.split(tokens, ',');
  if (tokens.size() != 8) {
    return {};
  }
  SmallVector<int64_t, 8> params;
  llvm::transform(tokens, std::back_inserter(params), [](StringRef s) {
    int param;
    llvm::to_integer(s, param);
    return param;
  });
  return AttnPerfConfigAttr::get(perfConfigStrAttr.getContext(),
                                 /*mPerBlockG0=*/params[0],
                                 /*mPerBlockG1=*/params[1],
                                 /*nPerBlockG0=*/params[2],
                                 /*kpackPerBlock=*/params[3],
                                 /*mPerWave=*/params[4],
                                 /*mnPerXdl*/ params[5],
                                 /*kpack=*/params[6],
                                 /*forceUnroll=*/params[7] == 1);
}

//===-----------------------------------------------------===//
// StageOp
//===-----------------------------------------------------===//

void StageOp::print(OpAsmPrinter &p) {
  // p.printOptionalArrowTypeList(getResultTypes());

  p << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult StageOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Introduce the body region and parse it.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Rock/IR/RockAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Rock/IR/RockOps.cpp.inc"
