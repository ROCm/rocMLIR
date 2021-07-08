//===- BackwardWeightV4r4Helper.h - Utility routines for AffineMap
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provide utility routines to check AffineMap instances.
//
//===----------------------------------------------------------------------===//

#ifndef BACKWARD_DATA_PADDING_KERNEL_HELPER_H
#define BACKWARD_DATA_PADDING_KERNEL_HELPER_H

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::miopen;

namespace mlir {
namespace miopen {
miopen::TransformOp padFilter(const int64_t gemmMExtra, const int64_t gemmNExtra, int64_t& gemmKExtra,
               miopen::TransformOp& gemmAPad, PatternRewriter &b,
               const mlir::Location& loc, llvm::DenseSet<int>& filterOobCheckDims,
               llvm::DenseMap<StringRef, int>& nameToDims, ArrayRef<int64_t> filterShape, const Type& filterElementType,
               const int64_t g, const int64_t k, const int64_t c, const int64_t yDotSlice, const int64_t xDotSlice);

miopen::TransformOp padInput(const int64_t gemmMExtra, const int64_t gemmNExtra, const int64_t gemmKExtra,
        miopen::TransformOp& gemmBPad, PatternRewriter &b,
        const Location& loc, llvm::DenseSet<int>& inputOobCheckDims, llvm::DenseMap<StringRef, int>& nameToDims,
		ArrayRef<int64_t> transformedShape, ArrayRef<int64_t> inputShape, const Type& inputElementType);

miopen::TransformOp padOutput(const int64_t gemmMExtra, const int64_t gemmNExtra, const int64_t gemmKExtra,
        miopen::TransformOp& gemmCPad, PatternRewriter &b,
        const Location& loc, llvm::DenseSet<int>& outputOobCheckDims, llvm::DenseMap<StringRef, int>& nameToDims,
		ArrayRef<int64_t> transformedShape, ArrayRef<int64_t> outputShape, const Type& outputElementType);

} // end namespace miopen
} // end namespace mlir
#endif
