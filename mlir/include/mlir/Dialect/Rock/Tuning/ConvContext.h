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

#ifndef MLIR_DIALECT_ROCK_CONVCONTEXT_H
#define MLIR_DIALECT_ROCK_CONVCONTEXT_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/Serializable.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include <iterator>

namespace mlir {
namespace rock {
struct DimIndexAndSize {
  size_t index;
  int64_t size;
};

struct ConvolutionContext {
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
};

// Populate ConvContext from a given Convolution Op.
// TODO(whchung): adopt ConvolutionOp OpTrait check after supporting PR is in.
ConvolutionContext populateConvContext(Operation *op);
} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_CONVCONTEXT_H
