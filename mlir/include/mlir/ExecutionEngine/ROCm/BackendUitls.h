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
#include "mlir/Target/ROCDLIR.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

class BackendUtils {
public:
  BackendUtils();
  BackendUtils(const std::string &triple, const std::string &chip,
               const std::string &feature);
  OwnedBlob compileISAToHsaco(const std::string isa, Location loc,
                              StringRef name);
  std::unique_ptr<llvm::Module>
  compileModuleToROCDLIR(Operation *m, llvm::LLVMContext &llvmContext,
                         llvm::StringRef name);
  std::string getChip() { return chip; }
  std::string getFeatures() { return features; }
  std::string getTriple() { return triple; }

private:
  std::string triple;
  std::string chip;
  std::string features;

  using Blob = SmallVector<char, 0>;
  void setupDefaults(std::string &chip, std::string &features,
                     std::string &triple);
  LogicalResult assembleIsa(const std::string isa, StringRef name,
                            Blob &result);
  LogicalResult assembleIsa(const std::string isa, StringRef name, Blob &result,
                            const std::string &tripleName,
                            const std::string &targetChip,
                            const std::string &features);
  LogicalResult createHsaco(const Blob &isaBlob, StringRef name,
                            Blob &hsacoBlob);
  void configTargetChip(std::string &targetChip);
  void configTargetFeatures(const std::string &chip, const std::string &triple,
                            std::string &features);
};
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_ROCM_BACKENDUTILS_H_
