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
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Conv2dGenerator {
public:
  LogicalResult parseConvDims(std::string &inputLayout,
                              std::string &outputLayout,
                              std::string &filterLayout, int64_t batchSize,
                              int64_t inputChannel, int64_t inputHeight,
                              int64_t inputWidth, int64_t outputChannel,
                              int64_t outputHeight, int64_t outputWidth,
                              int64_t filterHeight, int64_t filterWidth,
                              SmallVector<int64_t, 4> &filterDimension,
                              SmallVector<int64_t, 4> &inputDimension,
                              SmallVector<int64_t, 4> &outputDimension);

  LogicalResult genConvModule(
      std::string &arch, int num_cu, std::string &operation,
      std::string &inputLayout, std::string &outputLayout,
      std::string &filterLayout, const SmallVector<int64_t, 4> &filterDimension,
      const SmallVector<int64_t, 4> &inputDimension,
      const SmallVector<int64_t, 4> &outputDimension, int dilationHeight,
      int dilationWidth, int strideHeight, int strideWidth, int paddingHeight,
      int paddingWidth, ModuleOp &module, OpBuilder &builder,
      std::string &kernelName, mlir::FloatType dataType, bool xdlops = false);

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
};

} // namespace mlir
#endif // MLIR_DIALECT_MIOPEN_CONV2DGENERATOR_H_
