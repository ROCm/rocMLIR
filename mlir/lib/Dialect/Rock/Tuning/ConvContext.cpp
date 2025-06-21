#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "mlir/Dialect/Rock/Generator/ConvGenerator.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::rock;

static int getOptionalIntAttribute(Operation *op, StringRef attrName,
                                   int defaultValue) {
  if (op->hasAttrOfType<IntegerAttr>(attrName)) {
    return op->getAttrOfType<IntegerAttr>(attrName).getInt();
  }
  return defaultValue;
}

static void
populateDimIndexAndSize(const ArrayAttr &layoutAttr,
                        const ArrayRef<int64_t> &dim,
                        llvm::StringMap<DimIndexAndSize> &dimIndexAndSize) {
  assert(layoutAttr.size() == dim.size());
  size_t dimValSize = layoutAttr.size();
  for (size_t i = 0; i < dimValSize; ++i) {
    auto key = cast<StringAttr>(layoutAttr.getValue()[i]).getValue();

    // +++pf: update old keys.
    if (key == "y")
      key = "0";
    if (key == "x")
      key = "1";
    if (key[0] == 'h')
      key = StringAttr::get(layoutAttr.getContext(),
                            std::string("0") + key.drop_front());
    if (key[0] == 'w')
      key = StringAttr::get(layoutAttr.getContext(),
                            std::string("1") + key.drop_front());

    auto value = dim[i];
    dimIndexAndSize[key] = {i, value};
  }
}

ConvolutionDims ConvolutionContext::getConvDims() {
  llvm::SmallVector<int64_t, 4> fil;
  llvm::SmallVector<int64_t, 4> out;
  llvm::SmallVector<int64_t, 4> in;
  for (int i = 0;; i++) {
    std::string key = std::to_string(i);
    if (!dimIndexAndSize.contains(key))
      break;
    fil.push_back(dimIndexAndSize[key].size);
    out.push_back(dimIndexAndSize[key + "o"].size);
    in.push_back(dimIndexAndSize[key + "i"].size);
  }

  return ConvolutionDims(fil, out, in, dimIndexAndSize["k"].size,
                         dimIndexAndSize["c"].size, dimIndexAndSize["ni"].size,
                         dimIndexAndSize["g"].size);
}

ConvolutionContext mlir::rock::populateConvContext(Operation *op) {
  ConvOpType opType = convOpTypeFromKernelType(
      cast<RockGemmWrapperInterface>(op).getKernelType());

  assert(isa<RockConvInterface>(op) &&
         "The operation should be a conv-like operation");
  auto convOp = dyn_cast<RockConvInterface>(op);

  // XXX: Do we need these, especially since we're not actually serializing
  // anything to sqlite?
  if (opType == ConvOpType::BwdWeight) {
    assert(op->hasAttrOfType<IntegerAttr>("numCU"));
  }
  auto archVal = op->getAttrOfType<StringAttr>("arch").getValue();
  int numCu = getOptionalIntAttribute(op, "numCU",
                                      rock::lookupArchInfo(archVal).minNumCU);
  int gemmId = getOptionalIntAttribute(op, "gemmId", 0);

  llvm::StringMap<DimIndexAndSize> dimIndexAndSize;

  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr = op->getAttrOfType<ArrayAttr>("output_layout");

  auto strideVal = extractFromIntegerArrayAttr<int64_t>(convOp.getStrides());
  auto dilationVal =
      extractFromIntegerArrayAttr<int64_t>(convOp.getDilations());
  auto paddingVal = extractFromIntegerArrayAttr<int64_t>(convOp.getPadding());

  populateDimIndexAndSize(
      filterLayoutAttr,
      cast<MemRefType>(op->getOperand(0).getType()).getShape(),
      dimIndexAndSize);
  populateDimIndexAndSize(
      inputLayoutAttr, cast<MemRefType>(op->getOperand(1).getType()).getShape(),
      dimIndexAndSize);
  populateDimIndexAndSize(
      outputLayoutAttr,
      cast<MemRefType>(op->getOperand(2).getType()).getShape(),
      dimIndexAndSize);

  auto gemmIface = cast<RockGemmWrapperInterface>(op);
  Type dataTypeA = gemmIface.getAType(), dataTypeB = gemmIface.getBType();

  return {archVal,     numCu,      opType, dimIndexAndSize, strideVal,
          dilationVal, paddingVal, gemmId, dataTypeA,       dataTypeB};
}

ConvolutionContext
mlir::rock::populateConvContextFromConvGemm(ConvElementwiseGemmOp op) {
  auto archVal = op->getAttrOfType<StringAttr>("arch").getValue();
  int numCu = getOptionalIntAttribute(op, "numCU",
                                      rock::lookupArchInfo(archVal).minNumCU);
  int gemmId = getOptionalIntAttribute(op, "gemmId", 0);

  llvm::StringMap<DimIndexAndSize> dimIndexAndSize;

  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  SmallVector<StringAttr, 5> outLayoutSpec;

  auto strideVal = extractFromIntegerArrayAttr<int64_t>(op.getStrides());
  auto dilationVal = extractFromIntegerArrayAttr<int64_t>(op.getDilations());
  auto paddingVal = extractFromIntegerArrayAttr<int64_t>(op.getPadding());

  populateDimIndexAndSize(
      filterLayoutAttr,
      cast<MemRefType>(op->getOperand(0).getType()).getShape(),
      dimIndexAndSize);
  auto inputShape = cast<MemRefType>(op->getOperand(1).getType()).getShape();
  populateDimIndexAndSize(inputLayoutAttr, inputShape, dimIndexAndSize);

  // ["ni", "gi", "ci", "0i", "1i"] -> ["no", "go", "ko", "0o", "1o"]
  // input_layout = ["ni", "hi", "wi", "gi", "ci"], output_layout = ["no", "go",
  // "ko", "ho", "wo"]
  int64_t kernelSizeH = dimIndexAndSize["0"].size;
  int64_t kernelSizeW = dimIndexAndSize["1"].size;
  for (size_t i = 0; i < inputShape.size(); ++i) {
    auto key = cast<StringAttr>(inputLayoutAttr.getValue()[i]).getValue();
    auto inputSize = inputShape[i];
    if (key == "ni") {
      auto newKey =
          StringAttr::get(inputLayoutAttr.getContext(), std::string("no"));
      dimIndexAndSize[newKey] = {i, inputSize};
    } else if (key == "hi" || key == "0i") {
      int64_t ho = rock::ConvGenerator::outputDim(inputSize, kernelSizeH,
                                                  paddingVal[0], paddingVal[1],
                                                  strideVal[0], dilationVal[0]);
      auto newKey =
          StringAttr::get(inputLayoutAttr.getContext(), std::string("0o"));
      dimIndexAndSize[newKey] = {i, ho};
    } else if (key == "wi" || key == "1i") {
      int64_t wo = rock::ConvGenerator::outputDim(inputSize, kernelSizeW,
                                                  paddingVal[2], paddingVal[3],
                                                  strideVal[1], dilationVal[1]);
      auto newKey =
          StringAttr::get(inputLayoutAttr.getContext(), std::string("1o"));
      dimIndexAndSize[newKey] = {i, wo};
    } else if (key == "gi") {
      auto newKey =
          StringAttr::get(inputLayoutAttr.getContext(), std::string("go"));
      dimIndexAndSize[newKey] = {i, inputSize};
    } else if (key == "ci") {
      auto newKey =
          StringAttr::get(inputLayoutAttr.getContext(), std::string("ko"));
      dimIndexAndSize[newKey] = {i, dimIndexAndSize["k"].size};
    } else {
      llvm_unreachable("Invalid key");
    }
  }
  Type dataTypeA = op.getAType(), dataTypeB = op.getBType();

  return {archVal,     numCu,      ConvOpType::Fwd, dimIndexAndSize, strideVal,
          dilationVal, paddingVal, gemmId,          dataTypeA,       dataTypeB};
}
