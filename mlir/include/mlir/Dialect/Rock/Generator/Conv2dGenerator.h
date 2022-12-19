//===- Conv2dGenerator.h - MLIR to C++ option parsing ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares mlir Conv2dGenerator class
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_CONV2DGENERATOR_H_
#define MLIR_DIALECT_ROCK_CONV2DGENERATOR_H_

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace rock {
struct ConvolutionDims;

class Conv2dGenerator {
public:
  struct Config {
    std::string arch;
    // TODO: drop these
    std::string chip;
    std::string triple;
    std::string chipFeatures;
    std::string perfConfig;
    int num_cu;
    GemmFeatures features;
    llvm::Optional<rock::ConvOpType> operation;
    std::string dataTypeStr;
    int dilationHeight, dilationWidth;
    int strideHeight, strideWidth;
    int paddingHeightLeft, paddingHeightRight;
    int paddingWidthLeft, paddingWidthRight;
    std::string filterLayout;
    std::string inputLayout;
    std::string outputLayout;

    std::string kernelBaseName;

    int kernelId;

    SmallVector<int64_t, 5> filterDimension;
    SmallVector<int64_t, 5> inputDimension;
    SmallVector<int64_t, 5> outputDimension;

    int filterHeight;
    int filterWidth;
  };

  Conv2dGenerator(const std::string &arch = "", const std::string &chip = "",
                  const std::string &triple = "",
                  const std::string &chipFeatures = "",
                  const std::string &perfConfig = "", int num_cu = 0,
                  GemmFeatures features = GemmFeatures::none,
                  const Optional<rock::ConvOpType> operation = llvm::None,
                  const std::string &dataTypeStr = "f32",
                  int dilationHeight = 1, int dilationWidth = 1,
                  int strideHeight = 1, int strideWidth = 1,
                  int paddingHeightLeft = 0, int paddingHeightRight = 0,
                  int paddingWidthLeft = 0, int paddingWidthRight = 0,
                  const std::string &filterLayout = "kcyx",
                  const std::string &inputLayout = "nchw",
                  const std::string &outputLayout = "nkhw",
                  const std::string &kernelBaseName = "");

  Conv2dGenerator(const Config &_config);

  const Config &getConfig() const { return config; }
  void setKernelName(const std::string &newName);

  int getKernelCount(OpBuilder &builder) const;

  Type getDataType(OpBuilder &builder) const;

  void setDataType(std::string dataTypeStr);

  void flipXdlops();

  void setPerfConfig(StringRef perfConfig);

  ConvolutionDims getConvolutionDims() const;

  static inline constexpr int64_t outputDim(int64_t inputLen, int64_t filLen,
                                            int64_t leftPadLen,
                                            int64_t rightPadLen,
                                            int64_t strideLen, int64_t dilLen) {
    return (inputLen + leftPadLen + rightPadLen - (filLen - 1) * dilLen - 1) /
               strideLen +
           1;
  }

  LogicalResult parseConvConfig(const char *arguments);

  LogicalResult parseConvDims(int64_t batchSize, int64_t groupSize,
                              int64_t inputChannel, int64_t inputHeight,
                              int64_t inputWidth, int64_t outputChannel,
                              int64_t outputHeight, int64_t outputWidth,
                              int64_t filterHeight, int64_t filterWidth);

  LogicalResult genConvModule(ModuleOp &module, int kernel_id = -1,
                              bool is_verifier = false,
                              bool ignoreTuning = false);

  func::FuncOp getKernelFunc() const;

  template <typename Vector>
  std::string translateLayout(const Vector &src, const Vector &srcSpec,
                              const Vector &targetSpec) {
    auto permutation = layoutPermutation(src, srcSpec);

    std::string targetLayout;
    std::transform(permutation.begin(), permutation.end(),
                   std::back_inserter(targetLayout),
                   [&targetSpec](int64_t p) { return targetSpec[p]; });
    return targetLayout;
  }
  LogicalResult isApplicable(bool checkChip = true) const;

  // Utility function to query if a config requires additional workspace.
  bool hasWorkspace(OpBuilder &builder) const;

  // Utility function to fetch the size of workspace.
  int getWorkspaceSize(ModuleOp &module) const;

private:
  template <typename Vector>
  std::vector<int64_t> layoutPermutation(const Vector &src,
                                         const Vector &srcSpec) {
    std::vector<int64_t> permutation(srcSpec.size(), 0);
    std::transform(src.begin(), src.end(), permutation.begin(),
                   [&srcSpec](const char &s) {
                     auto it = std::find(srcSpec.begin(), srcSpec.end(), s);
                     return it - srcSpec.begin();
                   });
    return permutation;
  }
  int getBwdDataKernelCount() const;
  int getBwdWeightKernelCount(OpBuilder &builder) const;
  bool needExtraPadBwdWeight(OpBuilder &builder) const;
  LogicalResult hasValidDimension() const;
  LogicalResult hasValidChip() const;

  // Generator config
  Config config;

  // Generated Kernel Func
  func::FuncOp kernelFunc;
};

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_CONV2DGENERATOR_H_
