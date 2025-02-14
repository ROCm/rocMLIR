//===-- SortDimensionsMemoryLayout.h ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the passes for adding a transform that sorts dimensions by memory
// layout.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SORTDIMENSIONSMEMORYLAYOUT_SORTDIMENSIONSMEMORYLAYOUT_H
#define MLIR_CONVERSION_SORTDIMENSIONSMEMORYLAYOUT_SORTDIMENSIONSMEMORYLAYOUT_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_SORTDIMENSIONSMEMORYLAYOUTPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

void populateSortConvRewritePatterns(MLIRContext *context,
                                     RewritePatternSet &patterns);

void populateSortConvBwdDataRewritePatterns(MLIRContext *context,
                                            RewritePatternSet &patterns);

void populateSortConvBwdWeightRewritePatterns(MLIRContext *context,
                                              RewritePatternSet &patterns);

void populateSortGemmRewritePatterns(MLIRContext *context,
                                     RewritePatternSet &patterns);

void populateSortAttentionRewritePatterns(MLIRContext *context,
                                          RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_SORTDIMENSIONSMEMORYLAYOUT_SORTDIMENSIONSMEMORYLAYOUT_H
