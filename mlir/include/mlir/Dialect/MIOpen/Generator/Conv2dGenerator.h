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

#ifndef MLIR_DIALECT_MIOPEN_CONV2DGENERATOR_H_
#define MLIR_DIALECT_MIOPEN_CONV2DGENERATOR_H_

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Conv2dGenerator {
public:
  struct Config {
    std::string arch;
    int num_cu;
    bool xdlops;
    std::string operation;
    std::string dataTypeStr;
    int dilationHeight, dilationWidth;
    int strideHeight, strideWidth;
    int paddingHeightLeft, paddingHeightRight;
    int paddingWidthLeft, paddingWidthRight;
    std::string filterLayout;
    std::string inputLayout;
    std::string outputLayout;

    std::string kernelName;

    int kernelId;

    int outputSize;
    SmallVector<int64_t, 5> filterDimension;
    SmallVector<int64_t, 5> inputDimension;
    SmallVector<int64_t, 5> outputDimension;

    int filterHeight;
    int filterWidth;
  };

  Conv2dGenerator(const std::string &arch = "", int num_cu = 0,
                  bool xdlops = false, const std::string &operation = "conv2d",
                  const std::string &dataTypeStr = "f32",
                  int dilationHeight = 1, int dilationWidth = 1,
                  int strideHeight = 1, int strideWidth = 1,
                  int paddingHeightLeft = 0, int paddingHeightRight = 0,
                  int paddingWidthLeft = 0, int paddingWidthRight = 0,
                  const std::string &filterLayout = "kcyx",
                  const std::string &inputLayout = "nchw",
                  const std::string &outputLayout = "nkhw",
                  const std::string &kernelName = "");

  const Config &getConfig() const { return config; }
  void setKernelName(std::string newName);
  // return gemmIds, those numbers may not be consecutive
  llvm::SmallVector<int> getKernelCount() const;

  Type getDataType(OpBuilder &builder) const;

  LogicalResult parseConvConfig(const char *arguments);

  LogicalResult parseConvDims(int64_t batchSize, int64_t groupSize,
                              int64_t inputChannel, int64_t inputHeight,
                              int64_t inputWidth, int64_t outputChannel,
                              int64_t outputHeight, int64_t outputWidth,
                              int64_t filterHeight, int64_t filterWidth);

  LogicalResult genConvModule(ModuleOp &module, OpBuilder &builder,
                              int kernel_id = -1);

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
  llvm::SmallVector<int> getBwdDataKernelCount() const;

  // Generator config
  Config config;
};

} // namespace mlir
#endif // MLIR_DIALECT_MIOPEN_CONV2DGENERATOR_H_
