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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {

enum class DataType { f32, f16, bf16 };

struct ConvolutionContext : SQLiteSerializable<ConvolutionContext> {
  llvm::SmallString<8> arch;
  int num_cu;
  miopen::ConvOpType opType;
  llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;
  llvm::SmallVector<int64_t, 0> strideVal;
  llvm::SmallVector<int64_t, 0> dilationVal;
  llvm::SmallVector<int64_t, 0> paddingVal;
  int gemmId;
  DataType dataType;

  ConvolutionContext(const llvm::SmallString<8> &architecture, int numCu,
                     miopen::ConvOpType op,
                     llvm::StringMap<std::pair<size_t, int64_t>> dim,
                     llvm::SmallVector<int64_t, 0> stride,
                     llvm::SmallVector<int64_t, 0> dilation,
                     llvm::SmallVector<int64_t, 0> padding, int gemmid,
                     DataType type)
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
  DataType getDataType() const { return dataType; }

  static std::string tableName() { return "config"; }

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

    DataType dataType = self.getDataType();
    if (dataType == DataType::f32) {
      f("'" + std::string("FP32") + "'", "data_type");
    } else if (dataType == DataType::f16) {
      f("'" + std::string("FP16") + "'", "data_type");
    } else if (dataType == DataType::bf16) {
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

// T is either GriwiseGemmOp or GridwiseGemmV2Op
template <typename T> static miopen::ConvOpType ObtainConvDirection(T &op) {
  miopen::ConvOpType opType;
  auto kernel_algorithm =
      op->template getAttrOfType<StringAttr>("kernel_algorithm").getValue();
  if (kernel_algorithm.find(StringRef("backward_data")) != StringRef::npos) {
    opType = miopen::ConvOpType::Conv2DBwdDataOpType;
  } else if (kernel_algorithm.find(StringRef("backward_weight")) !=
             StringRef::npos) {
    opType = miopen::ConvOpType::Conv2DBwdWeightOpType;
  } else {
    opType = miopen::ConvOpType::Conv2DOpType;
  }
  return opType;
}

template <typename T> static DataType obtainDataType(T &op) {
  DataType ret = DataType::f32;
  Type type = op.input().getType().template cast<MemRefType>().getElementType();
  if (type.isF32()) {
    ret = DataType::f32;
  } else if (type.isF16()) {
    ret = DataType::f16;
  } else if (type.isBF16()) {
    ret = DataType::bf16;
  }
  return ret;
}

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

// T is either GriwiseGemmOp or GridwiseGemmV2Op
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
  auto filterDimensionAttr =
      op->template getAttrOfType<ArrayAttr>("filter_dimension");
  populateDimVal(filterLayoutAttr, filterDimensionAttr, dimIndexVal);
  auto inputLayoutAttr = op->template getAttrOfType<ArrayAttr>("input_layout");
  auto inputDimensionAttr =
      op->template getAttrOfType<ArrayAttr>("input_dimension");
  populateDimVal(inputLayoutAttr, inputDimensionAttr, dimIndexVal);
  auto outputLayoutAttr =
      op->template getAttrOfType<ArrayAttr>("output_layout");
  auto outputDimensionAttr =
      op->template getAttrOfType<ArrayAttr>("output_dimension");
  populateDimVal(outputLayoutAttr, outputDimensionAttr, dimIndexVal);

  auto strideAttr = op->template getAttrOfType<ArrayAttr>("strides");
  llvm::SmallVector<int64_t, 0> strideVal;
  populateSeqVal(strideAttr, strideVal);

  auto dilationAttr = op->template getAttrOfType<ArrayAttr>("dilations");
  llvm::SmallVector<int64_t, 0> dilationVal;
  populateSeqVal(dilationAttr, dilationVal);

  auto paddingAttr = op->template getAttrOfType<ArrayAttr>("padding");
  llvm::SmallVector<int64_t, 0> paddingVal;
  populateSeqVal(paddingAttr, paddingVal);

  auto dataType = obtainDataType(op);

  return {archVal,     numCuVal,   opType, dimIndexVal, strideVal,
          dilationVal, paddingVal, gemmId, dataType};
}

} // namespace mlir
#endif // MLIR_DIALECT_MIOPEN_CONVCONTEXT_H
