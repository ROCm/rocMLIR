//===- BackendUtils.h - MLIR to C++ option parsing ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares mlir rocm backend utils
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_ROCM_BACKENDUTILS_H_
#define MLIR_EXECUTIONENGINE_ROCM_BACKENDUTILS_H_

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

class BackendUtils {
public:
  BackendUtils(const std::string &triple = "", const std::string &chip = "",
               const std::string &feature = "");

  std::string getChip() { return chip; }
  std::string getFeatures() { return features; }
  std::string getTriple() { return triple; }

private:
  std::string triple;
  std::string chip;
  std::string features;

  void setupDefaults(std::string &chip, std::string &features,
                     std::string &triple);
  void configTarget(std::string &targetChip, std::string &features);
};
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_ROCM_BACKENDUTILS_H_
