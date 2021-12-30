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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MIOpen/AffineMapHelper.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::miopen;

namespace mlir {
namespace miopen {
inline LogicalResult isSupportedBackwardDataPaddingKernel(
    bool isXdlops, bool isStride2Pad1, int64_t gemmMExtra, int64_t gemmKExtra,
    int64_t gemmNExtra, mlir::miopen::Conv2DBwdDataOp &op) {
  if (gemmNExtra && gemmKExtra) {
    return op.emitOpError(
        "can't support backward data padding kernel when both pad "
        "gemmN and gemmK due to load issue\n");
  }

  if (isXdlops && (gemmMExtra || gemmNExtra)) {
    if (isStride2Pad1) {
      return op->emitOpError(
          "can't support backward data padding kernel when xdlops stride 2 "
          "pad_h,pad_w>0 and pad gemmM or gemmN due to store issue\n");
    }
  }
  return success();
}

} // end namespace miopen
} // end namespace mlir
#endif
