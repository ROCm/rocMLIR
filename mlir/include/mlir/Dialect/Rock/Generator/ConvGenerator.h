//===- ConvGenerator.h - MLIR to C++ option parsing ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares mlir ConvGenerator class
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_CONVGENERATOR_H_
#define MLIR_DIALECT_ROCK_CONVGENERATOR_H_

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace rock {
struct ConvolutionDims;

class ConvGenerator {
public:
  enum DIM { HEIGHT = 0, WIDTH = 1, DEPTH = 2 };

  struct Config {
    std::string arch;
    // TODO: drop these
    std::string chip;
    std::string triple;
    std::string chipFeatures;
    std::string perfConfig;
    std::optional<int> num_cu;
    bool reverseGrid;
    GemmFeatures features;
    std::optional<rock::ConvOpType> operation;
    std::string filterDataTypeStr;
    std::string inputDataTypeStr;
    std::string outputDataTypeStr;
    SmallVector<int64_t, 4> dilationDims;
    SmallVector<int64_t, 4> strideDims;
    SmallVector<int64_t, 4> paddingLeftDims;
    SmallVector<int64_t, 4> paddingRightDims;
    std::string filterLayout;
    std::string inputLayout;
    std::string outputLayout;

    std::string kernelBaseName;

    int kernelId;

    SmallVector<int64_t, 5> filterDimension;
    SmallVector<int64_t, 5> inputDimension;
    SmallVector<int64_t, 5> outputDimension;

    SmallVector<int64_t, 4> filterDims;
  };

  ConvGenerator(
      const std::string &arch = "", const std::string &chip = "",
      const std::string &triple = "", const std::string &chipFeatures = "",
      const std::string &perfConfig = "",
      std::optional<int> num_cu = std::nullopt, bool reverseGrid = false,
      GemmFeatures features = GemmFeatures::none,
      const std::optional<rock::ConvOpType> operation = std::nullopt,
      const std::string &filterDataTypeStr = "f32",
      const std::string &inputDataTypeStr = "f32",
      const std::string &outputDataTypeStr = "",
      ArrayRef<int> dilations = {1, 1}, ArrayRef<int> strides = {1, 1},
      ArrayRef<int> paddingLeft = {0, 0}, ArrayRef<int> paddingRight = {0, 0},
      const std::string &filterLayout = "kcyx",
      const std::string &inputLayout = "nchw",
      const std::string &outputLayout = "nkhw",
      const std::string &kernelBaseName = "");

  ConvGenerator(const Config &_config);

  const Config &getConfig() const { return config; }
  void setKernelName(const std::string &newName);

  LogicalResult getKernelCount(OpBuilder &builder, int &kernelCount) const;

  Type getFilterDataType(OpBuilder &builder) const;
  Type getInputDataType(OpBuilder &builder) const;
  Type getOutputDataType(OpBuilder &builder) const;

  void setDataTypes(const std::string &dataTypeStr);

  void flipAccel();

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

  LogicalResult parseConvConfig(OpBuilder &builder, const char *arguments);

  LogicalResult parseConvDims(int64_t batchSize, int64_t groupSize,
                              int64_t inputChannel, ArrayRef<int64_t> inputDims,
                              int64_t outputChannel,
                              ArrayRef<int64_t> outputDims,
                              ArrayRef<int64_t> filterDims);

  LogicalResult genConvModule(ModuleOp &module, int kernel_id = -1,
                              bool is_verifier = false,
                              bool ignoreTuning = false);

  func::FuncOp getKernelFunc() const;

  template <typename Vector>
  std::string translateLayout(const Vector &src, const Vector &srcSpec,
                              const Vector &targetSpec) {
#if 0
    auto permutation = layoutPermutation(src, srcSpec);

    std::string targetLayout;
    std::transform(permutation.begin(), permutation.end(),
                   std::back_inserter(targetLayout),
                   [&targetSpec](int64_t p) { return targetSpec[p]; });

    // +++pf:  update old key names.
    std::replace(targetLayout.begin(), targetLayout.end(), 'y', '0');
    std::replace(targetLayout.begin(), targetLayout.end(), 'x', '1');
    std::replace(targetLayout.begin(), targetLayout.end(), 'h', '0');
    std::replace(targetLayout.begin(), targetLayout.end(), 'w', '1');

    return targetLayout;
  }
  LogicalResult isApplicable(bool checkChip = true) const;

  // Utility function to query if a config requires additional workspace.
  LogicalResult hasWorkspace(OpBuilder &builder, bool &needWorkspace) const;

  // Utility function to fetch the size of workspace.
  LogicalResult getWorkspaceSize(ModuleOp &module, int &workspaceSize) const;

  // Utility function to get the number of CU for the specific GPU
  uint32_t getNumCU() const;

private:
  template <typename Vector>
  std::vector<int64_t> layoutPermutation(const Vector &src,
                                         const Vector &srcSpec) {
    std::vector<int64_t> permutation(src.size(), 0);
    std::transform(src.begin(), src.end(), permutation.begin(),
                   [&srcSpec](const char &s) {
                     auto it = std::find(srcSpec.begin(), srcSpec.end(), s);
                     return it - srcSpec.begin();
                   });
    return permutation;
  }
  int getBwdDataKernelCount() const;
  LogicalResult getBwdWeightKernelCount(OpBuilder &builder,
                                        int &kernelCount) const;
  LogicalResult needExtraPadBwdWeight(OpBuilder &builder,
                                      bool &needExtraPad) const;
  LogicalResult hasValidDimension() const;
  LogicalResult hasValidChip() const;

  // Generator config
  Config config;

  // Generated Kernel Func
  func::FuncOp kernelFunc;
};

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_CONVGENERATOR_H_
