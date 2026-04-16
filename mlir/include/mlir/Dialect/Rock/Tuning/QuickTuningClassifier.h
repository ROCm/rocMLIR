//===- QuickTuningClassifier.h - XGBoost-based perfconfig ranking ---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// This file declares the QuickTuningClassifier, which uses XGBoost models to
// rank quick-tune perfconfigs and select the top-N most likely performant ones
// for a given problem.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TUNING_QUICK_TUNING_CLASSIFIER_H
#define MLIR_DIALECT_ROCK_TUNING_QUICK_TUNING_CLASSIFIER_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace mlir {
namespace rock {

class QuickTuningClassifier {
public:
  /// Read ROCMLIR_QUICK_TUNE_TOP_N env var. Default 30, 0 disables classifier.
  static unsigned getTopN();

  /// Filter XDL/WMMA candidates down to the top-N using the classifier.
  /// Returns the full list if no model is found or top-N is 0.
  static std::vector<AccelGemmParamsAttr>
  filterTopN(const PopulateParamsInfo &info,
             llvm::ArrayRef<AccelGemmParamsAttr> candidates);

  /// Filter non-accel candidates down to the top-N.
  static std::vector<GeneralGemmParamsAttr>
  filterTopN(const PopulateParamsInfo &info,
             llvm::ArrayRef<GeneralGemmParamsAttr> candidates);

  /// Filter gemm-gemm (attention) candidates down to the top-N.
  static std::vector<GemmGemmParamsAttr>
  filterTopN(RockGemmGemmWrapperInterface op,
             llvm::ArrayRef<GemmGemmParamsAttr> candidates);
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TUNING_QUICK_TUNING_CLASSIFIER_H
