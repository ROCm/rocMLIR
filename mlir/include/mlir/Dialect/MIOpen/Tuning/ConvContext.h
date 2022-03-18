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

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Tuning/Serializable.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {

#if 0
struct ConvolutionConfig {
  // Users of ConvolutionContext:
  // - AffixTuningParameters.cpp : 
  //   - invoke tuning logic.
  // - GridwiseGemmParams.h :
  //   - tuning logic.

  // Fields from ConvolutionContext.
  llvm::SmallString<8> arch;

  int num_cu;

  miopen::ConvOpType opType;

  llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;
  llvm::SmallVector<int64_t, 0> strideVal;
  llvm::SmallVector<int64_t, 0> dilationVal;
  llvm::SmallVector<int64_t, 0> paddingVal;

  int gemmId;

  mlir::Type dataType;

  // Fields from Conv2dGenerator::Config.
  std::string chip;                             // arch
  std::string triple;                           // arch
  std::string features;                         // arch

  std::string perfConfig;                       // NOT found in ConvolutionContext
                                                // in op attributes

  int num_cu;                                   // num_cu

  bool xdlops;                                  // NOT found in ConvolutionContext
                                                // in op attributes

  llvm::Optional<miopen::ConvOpType> operation; // opType

  std::string dataTypeStr;                      // dataType

  int dilationHeight, dilationWidth;            // dilationVal
  int strideHeight, strideWidth;                // strideVal
  int paddingHeightLeft, paddingHeightRight;    // paddingVal
  int paddingWidthLeft, paddingWidthRight;      // paddingVal

  std::string filterLayout;                     // NOT found in ConvolutionContext
  std::string inputLayout;                      // NOT found in ConvolutionContext
  std::string outputLayout;                     // NOT found in ConvolutionContext
                                                // in op atttributes

  std::string kernelBaseName;                   // NOT found in ConvolutionContext
                                                // in FuncOp name

  int kernelId;                                 // gemmId

  SmallVector<int64_t, 5> filterDimension;      // NOT found in ConvolutionContext
  SmallVector<int64_t, 5> inputDimension;       // NOT found in ConvolutionContext
  SmallVector<int64_t, 5> outputDimension;      // NOT found in ConvolutionContext
                                                // in op operands

  int filterHeight;                             // NOT found in ConvolutionContext
  int filterWidth;                              // NOT found in ConvolutionContext
                                                // in op operands
};
#endif

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
    case miopen::ConvOpType::Fwd:
      f("'F'", "direction");
      break;
    case miopen::ConvOpType::BwdData:
      f("'B'", "direction");
      break;
    case miopen::ConvOpType::BwdWeight:
      f("'W'", "direction");
      break;
    }
  }
};

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

ConvolutionContext populateConvContext(Operation *op);

} // namespace mlir
#endif // MLIR_DIALECT_MIOPEN_CONVCONTEXT_H
