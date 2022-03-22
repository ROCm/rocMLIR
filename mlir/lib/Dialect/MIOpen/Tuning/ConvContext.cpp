#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"

namespace mlir {

namespace {

void populateDimVal(const ArrayAttr &layoutAttr, const ArrayRef<int64_t> &dim,
                    llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal) {
  assert(layoutAttr.size() == dim.size());
  size_t dimValSize = layoutAttr.size();
  for (size_t i = 0; i < dimValSize; ++i) {
    auto key = layoutAttr.getValue()[i].cast<StringAttr>().getValue();
    auto value = dim[i];
    dimIndexVal[key] = std::make_pair(i, value);
  }
}

void populateSeqVal(const ArrayAttr &seqAttr,
                    llvm::SmallVector<int64_t, 0> &seqVal) {
  size_t seqValSize = seqAttr.size();
  for (size_t i = 0; i < seqValSize; ++i) {
    // Not nested array, push back the value and be done
    if (seqAttr.getValue()[i].dyn_cast<ArrayAttr>() == nullptr) {
      seqVal.push_back(seqAttr.getValue()[i].cast<IntegerAttr>().getInt());
      continue;
    }
    // There is nested values, continue to populate those
    for (size_t j = 0; j < seqAttr.getValue()[i].cast<ArrayAttr>().size();
         ++j) {
      seqVal.push_back(seqAttr.getValue()[i]
                           .cast<ArrayAttr>()
                           .getValue()[j]
                           .cast<IntegerAttr>()
                           .getInt());
    }
  }
}
} // namespace

ConvolutionContext populateConvContext(Operation *op) {
  miopen::ConvOpType opType = obtainConvDirection(op);

  auto archVal = op->template getAttrOfType<StringAttr>("arch").getValue();
  int numCuVal = op->template getAttrOfType<IntegerAttr>("num_cu").getInt();
  auto gemmIdAttr = op->template getAttrOfType<IntegerAttr>("gemm_id");
  int gemmId = 0;
  if (gemmIdAttr) {
    gemmId = gemmIdAttr.getInt();
  }

  llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;

  auto filterLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");

  auto strideAttr = op->template getAttrOfType<ArrayAttr>("strides");
  llvm::SmallVector<int64_t, 0> strideVal;
  populateSeqVal(strideAttr, strideVal);

  auto dilationAttr = op->template getAttrOfType<ArrayAttr>("dilations");
  llvm::SmallVector<int64_t, 0> dilationVal;
  populateSeqVal(dilationAttr, dilationVal);

  auto paddingAttr = op->template getAttrOfType<ArrayAttr>("padding");
  llvm::SmallVector<int64_t, 0> paddingVal;
  populateSeqVal(paddingAttr, paddingVal);

  populateDimVal(
      filterLayoutAttr,
      op->getOperand(0).getType().template cast<MemRefType>().getShape(),
      dimIndexVal);
  populateDimVal(
      inputLayoutAttr,
      op->getOperand(1).getType().template cast<MemRefType>().getShape(),
      dimIndexVal);
  populateDimVal(
      outputLayoutAttr,
      op->getOperand(2).getType().template cast<MemRefType>().getShape(),
      dimIndexVal);

  auto dataType = obtainConvDataType(op);

  return {archVal,     numCuVal,   opType, dimIndexVal, strideVal,
          dilationVal, paddingVal, gemmId, dataType};
}

} // namespace mlir
