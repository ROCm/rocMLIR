#include "mlir/Dialect/MIOpen/Tuning/ConvContext.h"

namespace mlir {

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

  if (isa<miopen::GridwiseGemmOp>(*op) || isa<miopen::GridwiseGemmV2Op>(*op)) {
    auto filterDimensionAttr =
        op->template getAttrOfType<ArrayAttr>("filter_dimension");
    auto inputDimensionAttr =
        op->template getAttrOfType<ArrayAttr>("input_dimension");
    auto outputDimensionAttr =
        op->template getAttrOfType<ArrayAttr>("output_dimension");
    populateDimVal(filterLayoutAttr, filterDimensionAttr, dimIndexVal);
    populateDimVal(inputLayoutAttr, inputDimensionAttr, dimIndexVal);
    populateDimVal(outputLayoutAttr, outputDimensionAttr, dimIndexVal);
  } else {
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
  }

  auto dataType = obtainConvDataType(op);

  return {archVal,     numCuVal,   opType, dimIndexVal, strideVal,
          dilationVal, paddingVal, gemmId, dataType};
}

} // namespace mlir
