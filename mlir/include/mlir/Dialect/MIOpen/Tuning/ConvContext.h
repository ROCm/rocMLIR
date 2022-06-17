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
#include <iterator>

namespace mlir {
namespace miopen {
struct DimIndexAndSize {
  size_t index;
  int64_t size;
};

struct ConvolutionDims {
  int64_t y;
  int64_t x;
  int64_t ho;
  int64_t wo;
  int64_t hi;
  int64_t wi;
  int64_t k;
  int64_t c;
  int64_t n;
  int64_t g;

  ConvolutionDims(int64_t y, int64_t x, int64_t ho, int64_t wo, int64_t hi,
                  int64_t wi, int64_t k, int64_t c, int64_t n, int64_t g)
      : y(y), x(x), ho(ho), wo(wo), hi(hi), wi(wi), k(k), c(c), n(n), g(g) {}
};

struct ConvolutionContext : SQLiteSerializable<ConvolutionContext> {
  llvm::SmallString<8> arch;
  int num_cu;
  ConvOpType opType;
  llvm::StringMap<DimIndexAndSize> dimIndexAndSize;
  llvm::SmallVector<int64_t, 2> strideVal;
  llvm::SmallVector<int64_t, 2> dilationVal;
  llvm::SmallVector<int64_t, 4> paddingVal;
  int gemmId;
  Type dataType;

  ConvolutionContext(const llvm::SmallString<8> &architecture, int numCu,
                     ConvOpType op, llvm::StringMap<DimIndexAndSize> dim,
                     ArrayRef<int64_t> stride, ArrayRef<int64_t> dilation,
                     ArrayRef<int64_t> padding, int gemmid, Type type)
      : arch(architecture), num_cu(numCu), opType(op), dimIndexAndSize(dim),
        strideVal(stride.begin(), stride.end()),
        dilationVal(dilation.begin(), dilation.end()),
        paddingVal(padding.begin(), padding.end()), gemmId(gemmid),
        dataType(type) {}

  llvm::StringMap<DimIndexAndSize> getDimIndexAndSize() const {
    return dimIndexAndSize;
  }
  ConvolutionDims getConvDims();

  ArrayRef<int64_t> getPaddingVal() const { return paddingVal; }
  ArrayRef<int64_t> getStrideVal() const { return strideVal; }
  ArrayRef<int64_t> getDilationVal() const { return dilationVal; }
  ConvOpType getOpType() const { return opType; }
  Type getDataType() const { return dataType; }

  static std::string tableName() { return "config"; }

  // Note: Keep it in sync with miopen/conv/problem_description
  template <class Self, class F> static void visit(Self &&self, F f) {
    // Input tensor dimensions
    f(std::to_string(self.getDimIndexAndSize()["ni"].size), "batchsize");
    f(std::to_string(self.getDimIndexAndSize()["ci"].size), "in_channels");
    f(std::to_string(self.getDimIndexAndSize()["hi"].size), "in_h");
    f(std::to_string(self.getDimIndexAndSize()["wi"].size), "in_w");
    // Filter tensor dimensions
    f(std::to_string(self.getDimIndexAndSize()["y"].size), "fil_h");
    f(std::to_string(self.getDimIndexAndSize()["x"].size), "fil_w");
    // Output tensor dimensions
    f(std::to_string(self.getDimIndexAndSize()["ko"].size), "out_channels");
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
    // TODO use dimIndexAndSize to generate layout
    f("'" + std::string("NCHW") + "'", "layout");

    Type dataType = self.getDataType();
    if (dataType.isF32()) {
      f("'" + std::string("FP32") + "'", "data_type");
    } else if (dataType.isF16()) {
      f("'" + std::string("FP16") + "'", "data_type");
    } else if (dataType.isBF16()) {
      f("'" + std::string("BF16") + "'", "data_type");
    }

    switch (self.getOpType()) {
    case ConvOpType::Fwd:
      f("'F'", "direction");
      break;
    case ConvOpType::BwdData:
      f("'B'", "direction");
      break;
    case ConvOpType::BwdWeight:
      f("'W'", "direction");
      break;
    }
  }
};

// Populate ConvContext from a given Convolution Op.
// TODO(whchung): adopt ConvolutionOp OpTrait check after supporting PR is in.
ConvolutionContext populateConvContext(Operation *op);

} // namespace miopen
} // namespace mlir
#endif // MLIR_DIALECT_MIOPEN_CONVCONTEXT_H
