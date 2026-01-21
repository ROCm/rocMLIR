//===- RockOps.cpp - Rock MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Generator/ConvGenerator.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmFeaturesInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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
#include "mlir/IR/ValueRange.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>

using namespace mlir;
using namespace mlir::rock;

#include "mlir/Dialect/Rock/IR/RockOpsDialect.cpp.inc"
#include "mlir/Dialect/Rock/IR/RockTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

FailureOr<bool> mlir::rock::isWorkgroupMemorySpace(Attribute memorySpace) {
  if (!memorySpace)
    return failure();

  if (auto gpuMemSpace = dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuMemSpace.getValue() == gpu::AddressSpace::Workgroup;

  if (auto intMemSpace = dyn_cast<IntegerAttr>(memorySpace))
    return intMemSpace.getInt() ==
           static_cast<int64_t>(gpu::GPUDialect::getWorkgroupAddressSpace());

  return false;
}

static Type getElementTypeOrSelfRecursive(Type type) {
  while (auto shapedType = dyn_cast<ShapedType>(type)) {
    type = shapedType.getElementType();
  }
  return type;
}

template <int N>
struct rank : rank<N - 1> {};

template <>
struct rank<0> {};

template <typename OpType>
static auto
getGemmEffects(rank<1>, OpType &op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects)
    -> decltype(void(op.getScaleA()), void(op.getScaleB())) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();

  effects.emplace_back(read, &op.getCMutable());
  effects.emplace_back(write, &op.getCMutable());

  effects.emplace_back(read, &op.getAMutable());
  effects.emplace_back(read, &op.getBMutable());

  if (op.getScaleA()) {
    auto scaleARange = op.getScaleAMutable();
    if (!scaleARange.empty()) {
      effects.emplace_back(read, &scaleARange[0]);
    }
  }

  if (op.getScaleB()) {
    auto scaleBRange = op.getScaleBMutable();
    if (!scaleBRange.empty()) {
      effects.emplace_back(read, &scaleBRange[0]);
    }
  }
}

template <typename OpType>
static auto
getGemmEffects(rank<0>, OpType &op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects)
    -> void {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();

  effects.emplace_back(read, &op.getCMutable());
  effects.emplace_back(write, &op.getCMutable());

  effects.emplace_back(read, &op.getAMutable());
  effects.emplace_back(read, &op.getBMutable());
}

template <typename OpType>
static void
getGemmEffects(OpType &op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getGemmEffects(rank<1>{}, op, effects);
}

template <typename OpType>
static void
getGemmMatrixEffects(OpType &op,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();

  effects.emplace_back(read, &op.getMatrixCMutable());
  effects.emplace_back(write, &op.getMatrixCMutable());

  effects.emplace_back(read, &op.getMatrixAMutable());
  effects.emplace_back(read, &op.getMatrixBMutable());
}

template <typename OpType>
static void
getAttentionEffects(OpType &op,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();
  effects.emplace_back(read, &op.getOutMutable());
  effects.emplace_back(write, &op.getOutMutable());

  if (op.getLse()) {
    effects.emplace_back(read, &op.getLseMutable()[0]);
    effects.emplace_back(write, &op.getLseMutable()[0]);
  }
  if (op.getCurrentSeqLen()) {
    effects.emplace_back(read, &op.getCurrentSeqLenMutable()[0]);
  }

  effects.emplace_back(read, &op.getQueriesMutable());
  effects.emplace_back(read, &op.getKeysMutable());
  effects.emplace_back(read, &op.getValuesMutable());
  for (auto &regionArg : op.getPreSoftmaxElemWiseInputsMutable())
    effects.emplace_back(read, &regionArg);
}

template <typename OpType>
static void
getConvEffects(OpType &op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &op.getInputMutable(),
                       transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &op.getFilterMutable(),
                       transform::TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &op.getOutputMutable(),
                       transform::TransformMappingResource::get());
}

template <typename OpType>
static void
getCommonEffects(OpType &op,
                 SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();
  effects.emplace_back(read, &op.getSourceMutable());
  effects.emplace_back(write, &op.getDestMutable());
}

template <typename OpType>
static LogicalResult verifyScales(OpType op, Value matrix, Value scale,
                                  StringRef matrixName) {
  if (scale != nullptr) {
    ShapedType matrixType = cast<ShapedType>(matrix.getType());
    ShapedType scaleType = cast<ShapedType>(scale.getType());
    Type matrixElemType = getElementTypeOrSelfRecursive(matrixType);
    Type scaleElemType = getElementTypeOrSelfRecursive(scaleType);

    if (!isa<Float8E8M0FNUType>(scaleElemType)) {
      return op.emitError(
          llvm::formatv("Scale{0} must be of type Float8E8M0FNU.", matrixName));
    }
    if (!isa<Float4E2M1FNType>(matrixElemType)) {
      return op.emitError(
          llvm::formatv("For the scaled GEMMs, matrix{0} must be of "
                        "type Float4E2M1FNType.",
                        matrixName));
    }
    if (matrixType.getShape() != scaleType.getShape()) {
      return op.emitError(llvm::formatv(
          "Scale{0} shape must match matrix{0} shape.", matrixName));
    }
  }
  return success();
}

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
    if (isa<GemmParamsAttr>(attr)) {
      os << "accel_gemm_params";
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
ConvolutionDims ConvolutionDims::fromOp(Operation *op, bool enableOutput) {
  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  ArrayAttr outputLayoutAttr;
  if (enableOutput)
    outputLayoutAttr = op->template getAttrOfType<ArrayAttr>("output_layout");

  // Get shape of filter tensor.
  auto filterType = cast<ShapedType>(op->getOperand(0).getType());
  ArrayRef<int64_t> filterShape = filterType.getShape();

  // Get shape of input tensor.
  auto inputType = cast<ShapedType>(op->getOperand(1).getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Get shape of output tensor.
  auto outputType = cast<ShapedType>(op->getOperand(2).getType());
  ArrayRef<int64_t> outputShape = outputType.getShape();

  int64_t y, x, z, ho, wo, dout, hi, wi, di, k, c, n, g;
  y = x = z = ho = wo = dout = hi = wi = di = k = c = n = g = 0;

  for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr = cast<StringAttr>(filterLayoutAttr.getValue()[i]);
    auto inputAttr = cast<StringAttr>(inputLayoutAttr.getValue()[i]);
    StringAttr outputAttr;
    if (enableOutput)
      outputAttr = cast<StringAttr>(outputLayoutAttr.getValue()[i]);

    if (filterAttr.getValue() == "0" || filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x" || filterAttr.getValue() == "1") {
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

    if (inputAttr.getValue() == "hi" || inputAttr.getValue() == "0i") {
      hi = inputShape[i];
    } else if (inputAttr.getValue() == "wi" || inputAttr.getValue() == "1i") {
      wi = inputShape[i];
    } else if (inputAttr.getValue() == "2i") {
      di = inputShape[i];
    } else if (inputAttr.getValue() == "ni") {
      n = inputShape[i];
    }

    if (enableOutput) {
      if (outputAttr.getValue() == "ho" || outputAttr.getValue() == "0o") {
        ho = outputShape[i];
      } else if (outputAttr.getValue() == "wo" ||
                 outputAttr.getValue() == "1o") {
        wo = outputShape[i];
      } else if (outputAttr.getValue() == "2o") {
        dout = outputShape[i];
      }
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
  case KernelType::ConvElementwiseGemm:
    return ConvOpType::Fwd;
  case KernelType::ConvBwdData:
    return ConvOpType::BwdData;
  case KernelType::ConvBwdWeight:
    return ConvOpType::BwdWeight;
  case KernelType::Gemm:
    llvm_unreachable(
        "GEMM ops shouldn't be in convolution-specific lowering passes");
  case KernelType::Attention:
    llvm_unreachable(
        "Attention ops shouldn't be in convolution-specific lowering passes");
  case KernelType::GemmElementwiseGemm:
    llvm_unreachable(
        "gemm+gemm ops shouldn't be in convolution-specific lowering passes");
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
  bool isGfx1250 = arch.contains("gfx1250");
  if (isa<Float8E8M0FNUType>(elemTypeA) || isa<Float8E8M0FNUType>(elemTypeB)) {
    return op->emitOpError(
        "Matrix A or B is not allowed to have Float8E8M0FNU types");
  }
  if (bitEnumContainsAll(features, GemmFeatures::wmma)) {
    // Validate input data types based on architecture
    bool isValidTypeA = elemTypeA.isF16() || elemTypeA.isBF16() ||
                        elemTypeA.isInteger(8) || isFloat8Type(elemTypeA);

    // gfx1250 additionally supports F32
    if (isGfx1250)
      isValidTypeA = isValidTypeA || elemTypeA.isF32();

    // gfx11 doesn't support float8 types
    if (isGfx11 && isFloat8Type(elemTypeA))
      isValidTypeA = false;

    if (!isValidTypeA) {
      if (isGfx11)
        return op->emitOpError("Wmma supports only F16/BF16/int8 data types");
      if (isGfx1250)
        return op->emitOpError(
            "Wmma supports only F32/F16/BF16/int8/E4M3/E5M2 data types");
      return op->emitOpError(
          "Wmma supports only F16/BF16/int8/E4M3/E5M2 data types");
    }

    // Validate mixed types
    if (elemTypeA != elemTypeB) {
      // gfx1250 allows mixed precision for float8 types only
      bool allowMixed =
          isGfx1250 && isFloat8Type(elemTypeA) && isFloat8Type(elemTypeB);
      if (!allowMixed)
        return op->emitOpError(isGfx1250 ? "Wmma on gfx1250 supports mixed "
                                           "types only for FP8/BF8 combinations"
                                         : "Wmma does not support mixed types");
    }
  }
  if (bitEnumContainsAll(features, GemmFeatures::mfma)) {
    bool isGfx95 = arch.contains("gfx95");
    if (isGfx95 && (isa<Float8E4M3FNUZType, Float8E5M2FNUZType>(elemTypeA) ||
                    isa<Float8E4M3FNUZType, Float8E5M2FNUZType>(elemTypeB))) {
      return op->emitOpError(
          "Mfma does not support E4M3FNUZ/E5M2FNUZ data types");
    }
    if (!isGfx95 && arch.contains("gfx9") &&
        (isa<Float8E4M3FNType, Float8E5M2Type>(elemTypeA) ||
         isa<Float8E4M3FNType, Float8E5M2Type>(elemTypeB))) {
      return op->emitOpError("Mfma does not support E4M3/E5M2 data types ");
    }
    if (!isGfx95 && (isa<Float4E2M1FNType>(elemTypeA) ||
                     isa<Float4E2M1FNType>(elemTypeB))) {
      return op->emitOpError("Mfma does not support Float4E2M1FN data type ");
    }
  }
  if (elemTypeC) {
    if (isa<FloatType>(elemTypeA) && !isa<FloatType>(elemTypeC)) {
      return op->emitOpError("floating-point input type ")
             << elemTypeA
             << " requires a floating-point output type, but the output type "
                "is "
             << elemTypeC;
    }
    if (isa<IntegerType>(elemTypeA) && !isa<IntegerType>(elemTypeC)) {
      return op->emitOpError("integer input type ")
             << elemTypeA
             << " requires an integer output type, but the output type is "
             << elemTypeC;
    }
  }
  return success();
}

static LogicalResult verifyGemmTypes(RockGemmWrapperInterface gemmOp) {
  Type elemTypeA = gemmOp.getAType(), elemTypeB = gemmOp.getBType(),
       elemTypeC = gemmOp.getCType();

  StringAttr arch = rock::getArchValue(gemmOp);
  GemmFeatures features = rock::getFeatures(gemmOp);

  return verifyGemmTypes(gemmOp, features, arch, elemTypeA, elemTypeB,
                         elemTypeC);
}

static LogicalResult verifyConvOp(RockConvInterface convOp) {
  Operation *op = convOp.getOperation();
  RockGemmWrapperInterface gemmOp = cast<RockGemmWrapperInterface>(*convOp);

  if (failed(verifyGemmTypes(gemmOp)))
    return failure();

  auto features = rock::getFeatures(gemmOp);

  // Only perform this check for ops that have a feature attribute
  bool isAccel = rock::isAccel(features);
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

SmallVector<mlir::Type> GemmOp::getTypesForFeature() { return {getAType()}; }

SmallVector<mlir::Type> ConvOp::getTypesForFeature() { return {getAType()}; }

SmallVector<mlir::Type> ConvBwdDataOp::getTypesForFeature() {
  return {getAType()};
}

SmallVector<mlir::Type> ConvBwdWeightOp::getTypesForFeature() {
  return {getAType()};
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
  getConvEffects(*this, effects);
  effects.emplace_back(MemoryEffects::Write::get(), &getOutputMutable(),
                       transform::TransformMappingResource::get());
}

void ConvBwdDataOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getConvEffects(*this, effects);
  effects.emplace_back(MemoryEffects::Write::get(), &getInputMutable(),
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
  auto rankCheck = [&](ArrayRef<int64_t> dims,
                       StringRef name) -> LogicalResult {
    if (dims.size() != 2 && dims.size() != 3) {
      return emitOpError()
             << name
             << " must be a rank 2 or rank 3 tensor representing [G,] M, K";
    }
    return success();
  };
  if (failed(rankCheck(dimsA, "Matrix A")) ||
      failed(rankCheck(dimsB, "Matrix B")) ||
      failed(rankCheck(dimsC, "Matrix C"))) {
    return failure();
  }
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
  bool hasScaleA = getScaleA() != nullptr;
  bool hasScaleB = getScaleB() != nullptr;
  if (hasScaleA ^ hasScaleB) {
    return emitOpError("both scaleA and scaleB must be provided or neither");
  }
  // Unified verification for scaleA / scaleB.
  auto verifyScale = [&](Value scale, bool isA) -> LogicalResult {
    if (!scale)
      return success();
    ShapedType ty = cast<ShapedType>(scale.getType());
    ArrayRef<int64_t> dims = ty.getShape();
    StringRef scaleName = isA ? "scaleA" : "scaleB";
    if (failed(rankCheck(dims, scaleName)))
      return failure();
    Type elemType = ty.getElementType();
    if (!isa<Float8E8M0FNUType>(elemType) && !elemType.isF32())
      return emitOpError() << scaleName
                           << " must be of type Float8E8M0FNUType or f32";

    bool transposed = isA ? getAScaleTransposed() : getBScaleTransposed();
    int64_t offset = dims.size() == 2 ? 0 : 1;
    int64_t g = offset ? dims[0] : 1;
    int64_t first = dims[offset + (transposed ? 1 : 0)];
    int64_t second = dims[offset + (transposed ? 0 : 1)];

    int64_t expectedG = isA ? gA : gB;
    int64_t expectedFirst = isA ? mA : kB;  // scaleA: M; scaleB: K
    int64_t expectedSecond = isA ? kA : nB; // scaleA: K; scaleB: N

    StringRef firstName = isA ? "M" : "K";
    StringRef secondName = isA ? "K" : "N";

    if (second != expectedSecond)
      return emitOpError() << scaleName << "'s " << secondName
                           << " dimension must match matrix "
                           << (isA ? "A" : "B") << "'s " << secondName
                           << " dimension"
                           << " " << scaleName << "_" << secondName.lower()
                           << " = " << second << " " << (isA ? "k_a" : "n_b")
                           << " = " << expectedSecond;
    if (first != expectedFirst)
      return emitOpError() << scaleName << "'s " << firstName
                           << " dimension must match matrix "
                           << (isA ? "A" : "B") << "'s " << firstName
                           << " dimension"
                           << " " << scaleName << "_" << firstName.lower()
                           << " = " << first << " " << (isA ? "m_a" : "k_b")
                           << " = " << expectedFirst;
    if (g != expectedG)
      return emitOpError() << scaleName << "'s G dimension must match matrix "
                           << (isA ? "A" : "B") << "'s G dimension"
                           << " " << scaleName << "_g = " << g << " "
                           << (isA ? "g_a" : "g_b") << " = " << expectedG;
    return success();
  };

  if (failed(verifyScale(getScaleA(), /*isA=*/true)) ||
      failed(verifyScale(getScaleB(), /*isA=*/false)))
    return failure();
  if (hasScaleA && hasScaleB) {
    if (!isa<Float4E2M1FNType>(inElems)) {
      return emitOpError(
          "Scaled GEMMs are only supported for Float4E2M1FN input type");
    }
  }
  auto features = rock::getFeatures(this->getOperation());
  bool isMfma = bitEnumContainsAll(features, GemmFeatures::mfma);
  bool isWmma = bitEnumContainsAll(features, GemmFeatures::wmma);
  // if (Attribute params = this->getParams().value_or(nullptr)) {
  //   if (isMfma && !isa<GemmParamsAttr>(params))
  //     return emitOpError("a mfma GEMM has non-mfma tuning parameters");
  //   if (getFeatures() == GemmFeatures::none &&
  //       !isa<GeneralGemmParamsAttr>(params))
  //     return emitOpError("an all-hardware gemm must used the general gemm "
  //                        "tuning parameters");
  //   if (getDerivedBlockSize().has_value() &&
  //       isa<GeneralGemmParamsAttr>(params)) {
  //     return emitOpError(
  //         "cannot have derivedBlockSize when gemm has generalGemmParams");
  //   }
  // }

  if (getDerivedBlockSize().has_value() && !isMfma && !isWmma) {
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

void GemmOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getGemmEffects(*this, effects);
}

//===-----------------------------------------------------===//
//  GridwiseGemmAccel Op
//===-----------------------------------------------------===//
template <typename GridOp>
static LogicalResult verifyGridwiseGemm(GridOp op) {
  MemRefType aType = op.getA().getType(), bType = op.getB().getType(),
             cType = op.getC().getType();
  Type aElemType = getElementTypeOrSelfRecursive(aType);
  Type bElemType = getElementTypeOrSelfRecursive(bType);
  Type cElemType = getElementTypeOrSelfRecursive(cType);
  StringAttr archAttr =
      rock::getArch(op).value_or(StringAttr::get(op.getContext(), "gfx00"));
  if (failed(verifyGemmTypes(op, rock::getFeatures(op), archAttr, aElemType,
                             bElemType, cElemType)))
    return failure();
  if (aElemType.isInteger(8) &&
      !(cElemType.isInteger(32) || cElemType.isInteger(8)))
    return op.emitOpError("i8 input requires i32 or i8 output");
  if ((isFloat8Type(aElemType) || isa<Float4E2M1FNType>(aElemType)) &&
      !cElemType.isF32())
    return op.emitOpError("4-bit or 8-bit float input requires f32 output");

  ArrayRef<int64_t> aShape = aType.getShape(), bShape = bType.getShape(),
                    cShape = cType.getShape();
  int64_t g = aShape[0], k = aShape[2], m = aShape[1], n = bShape[2];
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

SmallVector<mlir::Type> GridwiseGemmAccelOp::getTypesForFeature() {
  return {getA().getType()};
}

LogicalResult GridwiseGemmAccelOp::verify() {
  Value scaleA = getScaleA();
  Value scaleB = getScaleB();
  bool hasScaleA = scaleA != nullptr;
  bool hasScaleB = scaleB != nullptr;
  if (hasScaleA ^ hasScaleB) {
    return emitOpError("both scaleA and scaleB must be provided or neither");
  }
  if (failed(verifyScales(*this, getA(), scaleA, "A")) ||
      failed(verifyScales(*this, getB(), scaleB, "B"))) {
    return failure();
  }
  return verifyGridwiseGemm(*this);
}

void GridwiseGemmAccelOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getGemmEffects(*this, effects);
}

//===-----------------------------------------------------===//
// GpuAllocOp
//===-----------------------------------------------------===//

static bool nonZero(MemRefType memref) {
  int64_t numElements = 1;
  Type type = memref.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type)) {
    numElements = vecType.getNumElements();
  }
  return memref.getNumElements() * numElements > 0;
}

LogicalResult GpuAllocOp::verify() {
  // Make sure the size is bigger than 0
  if (nonZero(getOutput().getType())) {
    return success();
  }
  return emitError("The size of rock.alloc should be greather than zero.");
}

//===----------------------------------------------------------------------===//
// BlockwiseLoadTileOp
//===----------------------------------------------------------------------===//

void BlockwiseLoadTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();

  effects.emplace_back(read, &getSourceMutable());
  effects.emplace_back(write, &getDestRegistersMutable());
}

SmallVector<mlir::Type> BlockwiseLoadTileOp::getTypesForFeature() {
  return {getSource().getType().getElementType()};
}

//===----------------------------------------------------------------------===//
// BlockwiseLoadTilePtrOp
//===----------------------------------------------------------------------===//

void BlockwiseLoadTilePtrOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();

  effects.emplace_back(read, &getPointerTensorMutable());
  effects.emplace_back(read, &getMaskTensorMutable());
  effects.emplace_back(write, &getDestRegistersMutable());
}

//===-----------------------------------------------------===//
// BlockwiseStoreTileOp
//===-----------------------------------------------------===//

SmallPtrSet<OpOperand *, 2> BlockwiseStoreTileOp::getAcceptingViewOperands() {
  auto operands = getOperation()->getOpOperands();
  return {operands.begin() + 1};
}

std::optional<OperandRange>
BlockwiseStoreTileOp::getExtraIndices(OpOperand &operand) {
  if (!getAcceptingViewOperands().contains(&operand)) {
    return std::nullopt;
  }
  // Only one operand supports view
  return getExtraIndices();
}

Operation *
BlockwiseStoreTileOp::cloneWithExtraIndices(OpBuilder &builder,
                                            OpOperand &operand, Value view,
                                            ArrayRef<Value> newExtraIndices) {
  if (!getAcceptingViewOperands().contains(&operand)) {
    return getOperation();
  }

  // Only one operand supports view
  auto newOp = BlockwiseStoreTileOp::create(
      builder, getLoc(), getSource(), view, newExtraIndices, getStoreMethod());
  return newOp.getOperation();
}

void BlockwiseStoreTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getCommonEffects(*this, effects);
}

LogicalResult BlockwiseStoreTileOp::verify() {
  MemRefType sourceType = getSource().getType();
  Attribute memSpaceAttr = sourceType.getMemorySpace();
  auto gpuMemSpaceAttr = dyn_cast_or_null<gpu::AddressSpaceAttr>(memSpaceAttr);
  if (memSpaceAttr && (!gpuMemSpaceAttr || gpuMemSpaceAttr.getValue() !=
                                               gpu::AddressSpace::Private))
    return emitOpError("source must be private registers");
  ArrayRef<int64_t> outputShape = getDest().getType().getShape();

  size_t extraIdxCount = getExtraIndices().size();
  if (outputShape.empty()) {
    if (extraIdxCount != 0)
      return emitOpError("write to a scalar must have no coordinates");
  }
  // TODO(roctriton): check that it matches the register allocation!

  // } else if (outputShape.size() != extraIdxCount + 1) {
  //   return emitOpError("dest view must be extraIndices + 1");
  // }
  return success();
}

//===-----------------------------------------------------===//
// BlockwiseStoreTilePtrOp
//===-----------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BlockwiseGemmAccelOp
//===----------------------------------------------------------------------===//

LogicalResult BlockwiseGemmAccelOp::verify() {
  bool hasScaleABuffer = getMatrixScaleA() != nullptr;
  bool hasScaleBBuffer = getMatrixScaleB() != nullptr;
  ShapedType aBufferType = cast<ShapedType>(getMatrixA().getType());
  ShapedType bBufferType = cast<ShapedType>(getMatrixB().getType());
  ShapedType cBufferType = cast<ShapedType>(getMatrixC().getType());
  Type aType = getElementTypeOrSelfRecursive(aBufferType);
  Type bType = getElementTypeOrSelfRecursive(bBufferType);
  Type cType = getElementTypeOrSelfRecursive(cBufferType);

  StringAttr archAttr = rock::getArch(*this).value_or(
      StringAttr::get(this->getContext(), "gfx00"));

  if (failed(verifyGemmTypes(*this, rock::getFeatures(*this), archAttr, aType,
                             bType, cType)))
    return failure();
  auto verifyMatrixAndScale = [&](Value bufferScale, ShapedType bufferType,
                                  const char *matrixName) -> LogicalResult {
    bool hasBufferScale = bufferScale != nullptr;

    if (hasBufferScale) {
      ShapedType bufferScaleType = cast<ShapedType>(bufferScale.getType());
      if (bufferType.getShape() != bufferScaleType.getShape()) {
        return emitOpError(llvm::formatv(
            "If scale{0} buffer is non-null, its shape must match "
            "buffer{0}'s shape.",
            matrixName));
      }
      Type bufferScaleElemType = getElementTypeOrSelfRecursive(bufferScaleType);
      if (!isa<Float8E8M0FNUType>(bufferScaleElemType)) {
        return emitOpError(llvm::formatv(
            "Scale{0} buffer must be of type Float8E8M0FNU.", matrixName));
      }
      Type bufferElemType = getElementTypeOrSelfRecursive(bufferType);
      if (!isa<Float4E2M1FNType>(bufferElemType)) {
        return emitOpError(
            llvm::formatv("For the scaled GEMMs, buffer{0} must be of "
                          "type Float4E2M1FNType.",
                          matrixName));
      }
    }

    return success();
  };

  // Verify matrix A and its scales
  if (failed(verifyMatrixAndScale(getMatrixScaleA(), aBufferType, "A")))
    return failure();

  // Verify matrix B and its scales
  if (failed(verifyMatrixAndScale(getMatrixScaleB(), bBufferType, "B")))
    return failure();

  if (hasScaleABuffer ^ hasScaleBBuffer)
    return emitOpError(
        "scaleA and scaleB buffers must both be present or both be null.");

  return success();
}

SmallVector<mlir::Type> BlockwiseGemmAccelOp::getTypesForFeature() {
  return {getMatrixA().getType().getElementType()};
}

void BlockwiseGemmAccelOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();

  effects.emplace_back(read, &getMatrixCMutable());
  effects.emplace_back(write, &getMatrixCMutable());

  effects.emplace_back(read, &getMatrixAMutable());
  effects.emplace_back(read, &getMatrixBMutable());
  if (getMatrixScaleA() && getMatrixScaleB()) {
    effects.emplace_back(read, &getMatrixScaleAMutable()[0]);
    effects.emplace_back(read, &getMatrixScaleBMutable()[0]);
  }
}

//===----------------------------------------------------------------------===//
// GridwiseAttentionAccelOp
//===----------------------------------------------------------------------===//
LogicalResult GridwiseAttentionAccelOp::verify() {
  if (getEnableSoftmax() && getStoreMethod() != StoreMethod::Set)
    return emitError("Only set store method is supported for attention.");

  RockAccelTuningParamAttrInterface gemm0TuningParams = getParams0();
  int64_t gemm0kpack = gemm0TuningParams.getKpack();
  int64_t gemm0NPerBlock = gemm0TuningParams.getNPerBlock();
  if (gemm0NPerBlock % gemm0kpack != 0) {
    return emitError("NPerBlock should be divisible by kpack.");
  }

  if (!getEnableSoftmax() && getLse())
    return emitError("LSE only works for attention.");

  if (!getEnableSoftmax() && getSplitKV() != 1)
    return emitError("split-kv is implemented for attention only.");

  if (!getEnableSoftmax() && getSoftmaxType()) {
    return emitError("Setting softmax type only works for attention.");
  }

  if (!getEnableSoftmax() && getCurrentSeqLen())
    return emitError("currentSeqLen only works for attention.");

  if (!getEnableSoftmax() && getPrefixOffset())
    return emitError("prefixOffset only works for attention.");

  if (!getEnableSoftmax() && getCausal())
    return emitError("causal only works for attention.");

  // Validate prefix offset constraints
  // prefixOffset requires causal to be enabled (prefix causal = causal +
  // prefixOffset)
  if (getPrefixOffset() && !getCausal())
    return emitError(
        "prefixOffset requires causal to be enabled. "
        "Prefix causal attention is causal masking with an offset.");

  return success();
}

SmallVector<mlir::Type> GridwiseAttentionAccelOp::getTypesForFeature() {
  return {getKeys().getType(), getValues().getType()};
}

void GridwiseAttentionAccelOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getAttentionEffects(*this, effects);
}

//===----------------------------------------------------------------------===//
// WorkgroupIdOp
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

//===-----------------------------------------------------===//
// ReduceOp
//===-----------------------------------------------------===//
void ReduceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();
  effects.emplace_back(read, &getInMutable());
  effects.emplace_back(read, &getOutMutable());
  effects.emplace_back(write, &getOutMutable());
}

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

  auto inElemType = getIn().getType().getElementType();
  auto outElemType = getOut().getType().getElementType();
  if (inElemType != outElemType)
    return emitError("element type of input and output is different");

  if (getReduceMethod() == ReduceMethod::Max && !outElemType.isF32())
    return emitError("reduce max only supports f32");

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
        "workspace should be at least the size of elements per block ");
  }
  return success();
}

//===-----------------------------------------------------===//
// GemmElementwiseGemmOp
//===-----------------------------------------------------===//

OpOperand *GemmElementwiseGemmOp::getOutArgument() {
  return &(*this)->getOpOperand(getNumOperands() - 1);
}

Type GemmElementwiseGemmOp::getOutType() { return getOut().getType(); }

Type GemmElementwiseGemmOp::getAType() { return getA().getType(); }

Type GemmElementwiseGemmOp::getBType() { return getB().getType(); }

Type GemmElementwiseGemmOp::getCType() { return getC().getType(); }

bool GemmElementwiseGemmOp::getTransposedA() { return getATransposed(); }

bool GemmElementwiseGemmOp::getTransposedB() { return getBTransposed(); }

bool GemmElementwiseGemmOp::getTransposedC() { return getCTransposed(); }

bool GemmElementwiseGemmOp::getTransposedOut() { return getOTransposed(); }

KernelType GemmElementwiseGemmOp::getKernelType() {
  return KernelType::GemmElementwiseGemm;
}

Region &GemmElementwiseGemmOp::getPreSecondGemmRegion() {
  return getPreSecondGemmBody();
}

SmallVector<mlir::Type> GemmElementwiseGemmOp::getTypesForFeature() {
  return {getAType(), getCType()};
}

GemmGemmSize GemmElementwiseGemmOp::getGemmGemmSize() {
  ShapedType typeA = getA().getType(), typeB = getB().getType(),
             typeC = getC().getType();
  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape(),
                    dimsC = typeC.getShape();
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1,
          offsetC = dimsC.size() == 2 ? 0 : 1;
  int64_t g = offsetA ? dimsA[0] : 1,
          m = dimsA[offsetA + (getATransposed() ? 1 : 0)],
          k = dimsA[offsetA + (getATransposed() ? 0 : 1)],
          n = dimsB[offsetB + (getBTransposed() ? 0 : 1)],
          o = dimsC[offsetC + (getCTransposed() ? 1 : 0)];
  return GemmGemmSize(g, m, k, n, o);
}

static LogicalResult verifyGemmPlusGemmLikeOp(RockGemmGemmWrapperInterface op,
                                              Value currentSeqLen, Value lse,
                                              int32_t numHeadsQ,
                                              int32_t numHeadsKV) {
  // number of heads for Q and K, V
  if (numHeadsQ <= 0) {
    return op.emitError("numHeadsQ must be positive");
  }
  if (numHeadsKV <= 0) {
    return op.emitError("numHeadsKV must be positive");
  }
  if (numHeadsQ % numHeadsKV != 0) {
    return op.emitError("numHeadsQ is not divisible by numHeadsKV");
  }
  int64_t factorGQA = numHeadsQ / numHeadsKV;

  ShapedType qType = cast<ShapedType>(op.getAType());
  int64_t qBatchDim = qType.getShape().size() == 3 ? qType.getShape()[0] : 1;
  ArrayRef<int64_t> qLastDims = qType.getShape().slice(qType.getRank() - 2);
  auto [queryM, queryK] = op.getTransposedA()
                              ? std::tuple{qLastDims[1], qLastDims[0]}
                              : std::tuple{qLastDims[0], qLastDims[1]};

  ShapedType kType = cast<ShapedType>(op.getBType());
  int64_t kBatchDim = kType.getShape().size() == 3 ? kType.getShape()[0] : 1;
  kBatchDim *= factorGQA;
  ArrayRef<int64_t> kLastDims = kType.getShape().slice(kType.getRank() - 2);
  auto [keyK, keyN] = op.getTransposedB()
                          ? std::tuple{kLastDims[1], kLastDims[0]}
                          : std::tuple{kLastDims[0], kLastDims[1]};

  ShapedType vType = cast<ShapedType>(op.getCType());
  int64_t vBatchDim = vType.getShape().size() == 3 ? vType.getShape()[0] : 1;
  vBatchDim *= factorGQA;
  ArrayRef<int64_t> vLastDims = vType.getShape().slice(vType.getRank() - 2);
  auto [valueK, valueN] = op.getTransposedC()
                              ? std::tuple{vLastDims[1], vLastDims[0]}
                              : std::tuple{vLastDims[0], vLastDims[1]};

  if (qBatchDim != kBatchDim || kBatchDim != vBatchDim) {
    return op.emitError("Batch dimensions do not match");
  }
  if (queryK != keyK) {
    return op.emitError("reduction dimensions of first gemm do not match");
  }
  if (keyN != valueK) {
    return op.emitError("reduction dimensions of second gemm do not match");
  }

  // check output type
  ShapedType oType = cast<ShapedType>(op.getOutType());
  int64_t oBatchDim = oType.getShape().size() == 3 ? oType.getShape()[0] : 1;
  int64_t oBatchDimOrig = oBatchDim;
  if (isa<AttentionOp>(op)) {
    int64_t splitKV = cast<AttentionOp>(op).getSplitKV();
    if (oBatchDim % splitKV != 0)
      return op.emitError("Batch size must be divisible by splitKV");

    oBatchDim = oBatchDim / splitKV;
  }

  ArrayRef<int64_t> oLastDims = oType.getShape().slice(oType.getRank() - 2);
  auto [outputSeqLen, outputHeadDim] =
      op.getTransposedOut() ? std::tuple{oLastDims[1], oLastDims[0]}
                            : std::tuple{oLastDims[0], oLastDims[1]};

  if (qType.getShape().size() != oType.getShape().size()) {
    return op.emitError("Number of dimensions do not match (Q and Output)");
  }
  if (qBatchDim != oBatchDim) {
    return op.emitError("Batch dimensions do not match (Q and Output)");
  }
  if (queryM != outputSeqLen) {
    return op.emitError("Sequence length does not match (Q and Output)");
  }
  if (valueN != outputHeadDim) {
    return op.emitError("Head dimensions do not match (V and Output)");
  }

  // check currentSeqLen (KV Cache)
  if (currentSeqLen) {
    ShapedType seqLenType = cast<ShapedType>(currentSeqLen.getType());
    if (seqLenType.getShape().size() != 1) {
      return op.emitError("Number of dimensions is not one (currentSeqLen)");
    }
    if (seqLenType.getShape()[0] != oBatchDim) {
      return op.emitError(
          "Batch dimensions do not match (currentSeqLen and Output)");
    }
  }

  // check LSE (log-sum-exp)
  if (lse) {
    ShapedType lseType = cast<ShapedType>(lse.getType());
    if (lseType.getShape().size() != 2) {
      return op.emitError("Number of dimensions is not two (LSE)");
    }
    if (lseType.getShape()[0] != oBatchDimOrig) {
      return op.emitError("Batch dimensions do not match (LSE and Output)");
    }
    if (lseType.getShape()[1] != queryM) {
      return op.emitError("SeqLenQ dimensions do not match (LSE and Q)");
    }
  }

  return success();
}

LogicalResult GemmElementwiseGemmOp::verify() {
  return verifyGemmPlusGemmLikeOp(*this, /*currentSeqLen=*/nullptr,
                                  /*lse=*/nullptr, /*numHeadsQ=*/1,
                                  /*numHeadsKV=*/1);
}

void GemmElementwiseGemmOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();
  effects.emplace_back(read, &getOutMutable());
  effects.emplace_back(write, &getOutMutable());

  effects.emplace_back(read, &getAMutable());
  effects.emplace_back(read, &getBMutable());
  effects.emplace_back(read, &getCMutable());
  for (auto &regionArg : getElemwiseInputsMutable())
    effects.emplace_back(read, &regionArg);
}

//===-----------------------------------------------------===//
// ConvElementwiseGemmOp
//===-----------------------------------------------------===//

OpOperand *ConvElementwiseGemmOp::getOutArgument() {
  return &(*this)->getOpOperand(getNumOperands() - 1);
}

Type ConvElementwiseGemmOp::getOutType() { return getOut().getType(); }

Type ConvElementwiseGemmOp::getAType() {
  auto size = getGemmGemmSize();
  auto elementType = getInput().getType().getElementType();
  int64_t dim1 = getTransposedA() ? size.k : size.n;
  int64_t dim2 = getTransposedA() ? size.n : size.k;
  return MemRefType::get({size.g, dim1, dim2}, elementType);
}

Type ConvElementwiseGemmOp::getBType() {
  auto size = getGemmGemmSize();
  auto elementType = getFilter().getType().getElementType();
  int64_t dim1 = getTransposedB() ? size.m : size.k;
  int64_t dim2 = getTransposedB() ? size.k : size.m;
  return MemRefType::get({size.g, dim1, dim2}, elementType);
}

Type ConvElementwiseGemmOp::getCType() { return getC().getType(); }

bool ConvElementwiseGemmOp::getTransposedA() {
  // see ConvToGemm pass
  return true;
}

bool ConvElementwiseGemmOp::getTransposedB() {
  // see ConvToGemm pass
  return false;
}

bool ConvElementwiseGemmOp::getTransposedC() { return getCTransposed(); }

bool ConvElementwiseGemmOp::getTransposedOut() { return getOTransposed(); }

KernelType ConvElementwiseGemmOp::getKernelType() {
  return KernelType::ConvElementwiseGemm;
}

Region &ConvElementwiseGemmOp::getPreSecondGemmRegion() {
  return getPreSecondGemmBody();
}

SmallVector<mlir::Type> ConvElementwiseGemmOp::getTypesForFeature() {
  return {getAType(), getCType()};
}

GemmGemmSize ConvElementwiseGemmOp::getGemmGemmSize() {
  auto strideVal = extractFromIntegerArrayAttr<int64_t>(getStrides());
  auto dilationVal = extractFromIntegerArrayAttr<int64_t>(getDilations());
  auto paddingVal = extractFromIntegerArrayAttr<int64_t>(getPadding());
  auto sizes = ConvolutionDims::fromOp(*this, false);

  // generate sizes.out with ConvGenerator
  sizes.out[0] = rock::ConvGenerator::outputDim(sizes.in[0], sizes.fil[0],
                                                paddingVal[0], paddingVal[1],
                                                strideVal[0], dilationVal[0]);
  sizes.out[1] = rock::ConvGenerator::outputDim(sizes.in[1], sizes.fil[1],
                                                paddingVal[2], paddingVal[3],
                                                strideVal[1], dilationVal[1]);

  rock::GemmSize gemmSize =
      rock::GemmSize::fromConvolution(rock::ConvOpType::Fwd, sizes);
  ArrayRef<int64_t> dimsC = getC().getType().getShape();
  int64_t offsetC = dimsC.size() == 2 ? 0 : 1;
  int64_t g = gemmSize.g, m = gemmSize.m, k = gemmSize.k, n = gemmSize.n,
          o = dimsC[offsetC + (getCTransposed() ? 0 : 1)];
  return GemmGemmSize(g, m, k, n, o);
}

LogicalResult ConvElementwiseGemmOp::verify() {
  return verifyGemmPlusGemmLikeOp(*this, /*currentSeqLen=*/nullptr,
                                  /*lse=*/nullptr, /*numHeadsQ=*/1,
                                  /*numHeadsKV=*/1);
}

void ConvElementwiseGemmOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();
  effects.emplace_back(read, &getOutMutable());
  effects.emplace_back(write, &getOutMutable());

  effects.emplace_back(read, &getFilterMutable());
  effects.emplace_back(read, &getInputMutable());
  effects.emplace_back(read, &getCMutable());
  for (auto &regionArg : getElemwiseInputsMutable())
    effects.emplace_back(read, &regionArg);
}

//===-----------------------------------------------------===//
// AttentionOp
//===-----------------------------------------------------===//

OpOperand *AttentionOp::getOutArgument() {
  // The output is the last operand unless LSE is used.
  // In that case, the output is the second to last operand.
  int64_t outIndex = getLse() ? 2 : 1;
  return &(*this)->getOpOperand(getNumOperands() - outIndex);
}

Type AttentionOp::getOutType() { return getOut().getType(); }

Type AttentionOp::getAType() { return getQueries().getType(); }

Type AttentionOp::getBType() { return getKeys().getType(); }

Type AttentionOp::getCType() { return getValues().getType(); }

bool AttentionOp::getTransposedA() { return getQTransposed(); }

bool AttentionOp::getTransposedB() { return getKTransposed(); }

bool AttentionOp::getTransposedC() { return getVTransposed(); }

bool AttentionOp::getTransposedOut() { return getOTransposed(); }

KernelType AttentionOp::getKernelType() { return KernelType::Attention; }

Region &AttentionOp::getPreSecondGemmRegion() { return getPreSoftmaxBody(); }

GemmGemmSize AttentionOp::getGemmGemmSize() {
  ShapedType typeA = getQueries().getType(), typeB = getKeys().getType(),
             typeC = getValues().getType();
  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape(),
                    dimsC = typeC.getShape();
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1,
          offsetC = dimsC.size() == 2 ? 0 : 1;
  int64_t g = offsetA ? dimsA[0] : 1,
          m = dimsA[offsetA + (getQTransposed() ? 1 : 0)],
          k = dimsA[offsetA + (getQTransposed() ? 0 : 1)],
          n = dimsB[offsetB + (getKTransposed() ? 0 : 1)],
          o = dimsC[offsetC + (getVTransposed() ? 1 : 0)];
  return GemmGemmSize(g, m, k, n, o);
}

SmallVector<mlir::Type> AttentionOp::getTypesForFeature() {
  return {getAType(), getCType()};
}

LogicalResult AttentionOp::verify() {
  if (getSplitKV() != 1 && !getLse())
    return emitError("Flash decoding needs LSE output");

  if (getSplitKV() <= 0)
    return emitError("Negative or zero split-kv does not make sense");

  if (getStoreMethod() != StoreMethod::Set)
    return emitError("Only set store method is supported for attention.");

  // Validate prefix offset constraints
  // prefixOffset requires causal to be enabled (prefix causal = causal +
  // prefixOffset)
  if (getPrefixOffset() && !getCausal())
    return emitError(
        "prefixOffset requires causal to be enabled. "
        "Prefix causal attention is causal masking with an offset.");

  return verifyGemmPlusGemmLikeOp(*this, getCurrentSeqLen(), getLse(),
                                  getNumHeadsQ(), getNumHeadsKV());
}

void AttentionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getAttentionEffects(*this, effects);
}

//===-----------------------------------------------------===//
// PerfConfigStr parsing
//===-----------------------------------------------------===//

namespace {

constexpr size_t SmallVectorInlineSize = 16;

struct PerfConfigParseResult {
  int version;
  SmallVector<int64_t, SmallVectorInlineSize> params;
};

std::optional<PerfConfigParseResult>
parsePerfConfigStr(StringRef configStr, StringRef expectedPrefix = "") {
  StringRef rest = configStr;

  // Handle optional prefix
  if (!expectedPrefix.empty()) {
    StringRef prefix;
    std::tie(prefix, rest) = rest.split(':');
    if (prefix != expectedPrefix)
      return std::nullopt;
  }

  // Parse "vN:"
  StringRef versionStr;
  std::tie(versionStr, rest) = rest.split(':');
  if (!versionStr.consume_front("v"))
    return std::nullopt;

  int version;
  if (!llvm::to_integer(versionStr, version))
    return std::nullopt;

  // Parse comma-separated parameters
  SmallVector<StringRef, SmallVectorInlineSize> tokens;
  rest.split(tokens, ',');

  SmallVector<int64_t, SmallVectorInlineSize> params;
  params.reserve(tokens.size());
  for (StringRef tok : tokens) {
    int64_t val;
    if (!llvm::to_integer(tok.trim(), val))
      return std::nullopt;
    params.push_back(val);
  }

  return PerfConfigParseResult{version, params};
}

std::tuple<int64_t, int64_t, int64_t> handleLegacyNPerWaveOrMnPerXdl(
    const SmallVectorImpl<int64_t> &params, int64_t &idx, int64_t mPerBlock,
    int64_t nPerBlock, int64_t mPerWave, bool isWmma) {
  int64_t nPerWave, mnPerXdl;
  if (isWmma) {
    mnPerXdl = 16; // default value 16 because older versions had no mnPerXdl
    nPerWave = params[idx++];
  } else {
    mnPerXdl = params[idx++];
    constexpr int64_t maxWavesPerWG = 4;
    int64_t mWaves = std::min(mPerBlock / mPerWave, maxWavesPerWG);
    int64_t nWaves = maxWavesPerWG / mWaves;
    mPerWave = mPerBlock / mWaves;
    nPerWave = std::max(nPerBlock / nWaves, mnPerXdl);
  }
  return {mPerWave, nPerWave, mnPerXdl};
}

} // namespace

//===-----------------------------------------------------===//
// GemmParamsAttr
//===-----------------------------------------------------===//

GemmParamsAttr GemmParamsAttr::get(StringAttr perfConfigStrAttr) {
  auto parsed = parsePerfConfigStr(perfConfigStrAttr.strref(), "gemm");
  if (!parsed) {
    return {};
  }

  int version = parsed->version;
  auto &params = parsed->params;

  size_t expectedCount = (version == 1)   ? 11
                                          : 0;
  if (expectedCount == 0 || params.size() != expectedCount) {
    return {};
  }

  int64_t idx = 0;
  int64_t mPerBlock = params[idx++];
  int64_t nPerBlock = params[idx++];
  int64_t kpackPerBlock = params[idx++];
  int64_t kpack = params[idx++];
  int64_t numCTAs = params[idx++];
  int64_t numWaves = params[idx++];
  int64_t matrixInstrNonkdim = params[idx++];
  int64_t splitKFactor = params[idx++];
  int64_t numStages = params[idx++];
  int64_t wavesPerEU = params[idx++];
  int64_t gridGroupSize = params[idx++];

  return GemmParamsAttr::get(
      perfConfigStrAttr.getContext(), mPerBlock, nPerBlock, kpackPerBlock,
      kpack, numCTAs, numWaves, matrixInstrNonkdim, splitKFactor, numStages,
      wavesPerEU, gridGroupSize);
}

//===-----------------------------------------------------===//
// GemmGemmParamsAttr
//===-----------------------------------------------------===//

GemmGemmParamsAttr GemmGemmParamsAttr::get(StringAttr perfConfigStrAttr) {
  auto parsed = parsePerfConfigStr(perfConfigStrAttr.strref(), "attn");
  if (!parsed) {
    return {};
  }

  int version = parsed->version;
  auto &params = parsed->params;

  size_t expectedCount = (version == 1)   ? 12
                                          : 0;
  if (expectedCount == 0 || params.size() != expectedCount) {
    return {};
  }

  int idx = 0;
  int64_t mPerBlockG0 = params[idx++];
  int64_t mPerBlockG1 = params[idx++];
  int64_t nPerBlockG0 = params[idx++];
  int64_t kpackPerBlock = params[idx++];
  int64_t kpack = params[idx++];
  int64_t numCTAs = params[idx++];
  int64_t numWaves = params[idx++];
  int64_t matrixInstrNonkdim = params[idx++];
  int64_t splitKFactor = params[idx++];
  int64_t numStages = params[idx++];
  int64_t wavesPerEU = params[idx++];
  int64_t gridGroupSize = params[idx++];

  return GemmGemmParamsAttr::get(
      perfConfigStrAttr.getContext(), mPerBlockG0, mPerBlockG1, nPerBlockG0,
      kpackPerBlock, kpack, numCTAs, numWaves, matrixInstrNonkdim, splitKFactor,
      numStages, wavesPerEU, gridGroupSize);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Rock/IR/RockAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Rock/IR/RockOps.cpp.inc"
