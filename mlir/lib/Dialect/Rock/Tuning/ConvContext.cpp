#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

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
    auto key = layoutAttr.getValue()[i].cast<StringAttr>().getValue();

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
      op->getOperand(0).getType().template cast<MemRefType>().getShape(),
      dimIndexAndSize);
  populateDimIndexAndSize(
      inputLayoutAttr,
      op->getOperand(1).getType().template cast<MemRefType>().getShape(),
      dimIndexAndSize);
  populateDimIndexAndSize(
      outputLayoutAttr,
      op->getOperand(2).getType().template cast<MemRefType>().getShape(),
      dimIndexAndSize);

  auto gemmIface = cast<RockGemmWrapperInterface>(op);
  Type dataTypeA = gemmIface.getAType(), dataTypeB = gemmIface.getBType();

  return {archVal,     numCu,      opType, dimIndexAndSize, strideVal,
          dilationVal, paddingVal, gemmId, dataTypeA,       dataTypeB};
}
