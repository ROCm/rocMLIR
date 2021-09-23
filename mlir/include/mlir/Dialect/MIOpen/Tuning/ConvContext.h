//===--------- ConvContext.h - MLIR tuning parameter generation ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR convolution context for tuning
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_CONVCONTEXT_H
#define MLIR_DIALECT_MIOPEN_CONVCONTEXT_H

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Tuning/Serializable.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {

struct ConvolutionContext : SQLiteSerializable<ConvolutionContext> {
  llvm::SmallString<8> arch;
  int num_cu;
  miopen::ConvOpType opType;
  llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;
  llvm::SmallVector<int64_t, 0> strideVal;
  llvm::SmallVector<int64_t, 0> dilationVal;
  llvm::SmallVector<int64_t, 0> paddingVal;
  int gemmId;
  mlir::Type dataType;

  ConvolutionContext(const llvm::SmallString<8> &architecture, int numCu,
                     miopen::ConvOpType op,
                     llvm::StringMap<std::pair<size_t, int64_t>> dim,
                     llvm::SmallVector<int64_t, 0> stride,
                     llvm::SmallVector<int64_t, 0> dilation,
                     llvm::SmallVector<int64_t, 0> padding, int gemmid,
                     mlir::Type type)
      : arch(architecture), num_cu(numCu), opType(op), dimIndexVal(dim),
        strideVal(stride), dilationVal(dilation), paddingVal(padding),
        gemmId(gemmid), dataType(type) {}

  llvm::StringMap<std::pair<size_t, int64_t>> getDimIndexVal() const {
    return dimIndexVal;
  }
  llvm::SmallVector<int64_t, 0> getPaddingVal() const { return paddingVal; }
  llvm::SmallVector<int64_t, 0> getStrideVal() const { return strideVal; }
  llvm::SmallVector<int64_t, 0> getDilationVal() const { return dilationVal; }
  miopen::ConvOpType getOpType() const { return opType; }
  mlir::Type getDataType() const { return dataType; }

  static std::string tableName() { return "config"; }

  // here is xdlops part,non xdlops have different limitations
  bool isPaddingKernelSupport() {
    bool isSupport = false;
    if (opType == miopen::ConvOpType::Conv2DBwdDataOpType) {
      isSupport = true;
      if (getStrideVal()[0] > 1 && getStrideVal()[1] > 1) {
        // nhwc stride 2 only support gemmN padding , so disable it now
        if (getDimIndexVal()["ci"].first == 4)
          isSupport = false;
        // nchw stride2 and padh,padw>0 fail ,diable it now
        if (getPaddingVal()[0] >= 1 || getPaddingVal()[1] >= 1 ||
            getPaddingVal()[2] >= 1 || getPaddingVal()[3] >= 1) {
          isSupport = false;
        }
      }
    } else {
      isSupport = true;
    }
    return isSupport;
  }

  // Note: Keep it in sync with miopen/conv/problem_description
  template <class Self, class F> static void visit(Self &&self, F f) {
    // Input tensor dimensions
    f(std::to_string(self.getDimIndexVal()["ni"].second), "batchsize");
    f(std::to_string(self.getDimIndexVal()["ci"].second), "in_channels");
    f(std::to_string(self.getDimIndexVal()["hi"].second), "in_h");
    f(std::to_string(self.getDimIndexVal()["wi"].second), "in_w");
    // Filter tensor dimensions
    f(std::to_string(self.getDimIndexVal()["y"].second), "fil_h");
    f(std::to_string(self.getDimIndexVal()["x"].second), "fil_w");
    // Output tensor dimensions
    f(std::to_string(self.getDimIndexVal()["ko"].second), "out_channels");
    // Padding
    f(std::to_string(self.getPaddingVal()[0]), "pad_h_l");
    f(std::to_string(self.getPaddingVal()[1]), "pad_h_r");
    f(std::to_string(self.getPaddingVal()[2]), "pad_w_l");
    f(std::to_string(self.getPaddingVal()[3]), "pad_w_r");
    // Strides
    f(std::to_string(self.getStrideVal()[0]), "conv_stride_h");
    f(std::to_string(self.getStrideVal()[1]), "conv_stride_w");
    f(std::to_string(0), "conv_stride_d");
    f(std::to_string(self.getDilationVal()[0]), "dilation_h");
    f(std::to_string(self.getDilationVal()[1]), "dilation_w");
    f(std::to_string(0), "dilation_d");
    f(std::to_string(0), "bias");
    f(std::to_string(1), "group_count");
    // TODO use dimIndexVal to generate layout
    f("'" + std::string("NCHW") + "'", "layout");

    mlir::Type dataType = self.getDataType();
    if (dataType.isF32()) {
      f("'" + std::string("FP32") + "'", "data_type");
    } else if (dataType.isF16()) {
      f("'" + std::string("FP16") + "'", "data_type");
    } else if (dataType.isBF16()) {
      f("'" + std::string("BF16") + "'", "data_type");
    }

    switch (self.getOpType()) {
    case miopen::ConvOpType::Conv2DOpType:
      f("'F'", "direction");
      break;
    case miopen::ConvOpType::Conv2DBwdDataOpType:
      f("'B'", "direction");
      break;
    case miopen::ConvOpType::Conv2DBwdWeightOpType:
      f("'W'", "direction");
      break;
    }
  }
};

// T would be one of the following:
// - miopen::Conv2DOp
// - miopen::Conv2DBwdDataOp
// - miopen::Conv2DBwdWeightOp
template <typename T> static miopen::ConvOpType ObtainConvDirection(T &op) {
  miopen::ConvOpType opType = miopen::ConvOpType::Conv2DOpType;
  if (isa<miopen::Conv2DOp>(*op)) {
    opType = miopen::ConvOpType::Conv2DOpType;
  } else if (isa<miopen::Conv2DBwdDataOp>(*op)) {
    opType = miopen::ConvOpType::Conv2DBwdDataOpType;
  } else if (isa<miopen::Conv2DBwdWeightOp>(*op)) {
    opType = miopen::ConvOpType::Conv2DBwdWeightOpType;
  }
  return opType;
}

template <typename T> static mlir::Type obtainDataType(T &op) {
  return op.input().getType().template cast<MemRefType>().getElementType();
}

// TBD: Remove this function along with C++ code emitter.
static inline void
populateDimVal(const ArrayAttr &layoutAttr, const ArrayAttr &dimAttr,
               llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal) {
  assert(layoutAttr.size() == dimAttr.size());
  size_t dimValSize = layoutAttr.size();
  for (size_t i = 0; i < dimValSize; ++i) {
    auto key = layoutAttr.getValue()[i].cast<StringAttr>().getValue();
    auto value = dimAttr.getValue()[i].cast<IntegerAttr>().getInt();
    dimIndexVal[key] = std::make_pair(i, value);
  }
}

static inline void
populateDimVal(const ArrayAttr &layoutAttr, const ArrayRef<int64_t> &dim,
               llvm::StringMap<std::pair<size_t, int64_t>> &dimIndexVal) {
  assert(layoutAttr.size() == dim.size());
  size_t dimValSize = layoutAttr.size();
  for (size_t i = 0; i < dimValSize; ++i) {
    auto key = layoutAttr.getValue()[i].cast<StringAttr>().getValue();
    auto value = dim[i];
    dimIndexVal[key] = std::make_pair(i, value);
  }
}

static inline void populateSeqVal(const ArrayAttr &seqAttr,
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

// T would be one of the following:
// - miopen::Conv2DOp
// - miopen::Conv2DBwdDataOp
// - miopen::Conv2DBwdWeightOp
//
// TBD. Obsolete. Only used in C++ emitter path.
// Remove along with C++ emitter path.
// - miopen::GridwiseGemmOp
// - miopen::GridwiseGemmV2Op
template <typename T> static ConvolutionContext populateConvContext(T &op) {
  miopen::ConvOpType opType = ObtainConvDirection(op);

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
    populateDimVal(filterLayoutAttr,
                   op.filter().getType().template cast<MemRefType>().getShape(),
                   dimIndexVal);
    populateDimVal(inputLayoutAttr,
                   op.input().getType().template cast<MemRefType>().getShape(),
                   dimIndexVal);
    populateDimVal(outputLayoutAttr,
                   op.output().getType().template cast<MemRefType>().getShape(),
                   dimIndexVal);
  }

  auto dataType = obtainDataType(op);

  return {archVal,     numCuVal,   opType, dimIndexVal, strideVal,
          dilationVal, paddingVal, gemmId, dataType};
}

} // namespace mlir
#endif // MLIR_DIALECT_MIOPEN_CONVCONTEXT_H
