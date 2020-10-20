//===- ConvContext.h - MLIR tuning parameter generation --------*-===//
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

struct ConvolutionContext : SQLiteSerializable<ConvolutionContext> {
  llvm::SmallString<8> arch;
  int num_cu;
  miopen::ConvOpType opType;
  llvm::StringMap<std::pair<size_t, int64_t>> dimIndexVal;
  llvm::SmallVector<int64_t, 0> strideVal;
  llvm::SmallVector<int64_t, 0> dilationVal;
  llvm::SmallVector<int64_t, 0> paddingVal;

  ConvolutionContext(const llvm::SmallString<8> &architecture, int numCu,
                     miopen::ConvOpType op,
                     llvm::StringMap<std::pair<size_t, int64_t>> dim,
                     llvm::SmallVector<int64_t, 0> stride,
                     llvm::SmallVector<int64_t, 0> dilation,
                     llvm::SmallVector<int64_t, 0> padding)
      : arch(architecture), num_cu(numCu), opType(op), dimIndexVal(dim),
        strideVal(stride), dilationVal(dilation), paddingVal(padding) {}

  llvm::StringMap<std::pair<size_t, int64_t>> getDimIndexVal() const {
    return dimIndexVal;
  }
  llvm::SmallVector<int64_t, 0> getPaddingVal() const { return paddingVal; }
  llvm::SmallVector<int64_t, 0> getStrideVal() const { return strideVal; }
  llvm::SmallVector<int64_t, 0> getDilationVal() const { return dilationVal; }
  miopen::ConvOpType getOpType() const { return opType; }

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
    f(std::to_string(self.getPaddingVal()[0]), "pad_h");
    f(std::to_string(self.getPaddingVal()[1]), "pad_w");
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
    f("'" + std::string("FP32") + "'", "data_type");

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

} // namespace mlir
#endif // MLIR_DIALECT_MIOPEN_CONVCONTEXT_H
