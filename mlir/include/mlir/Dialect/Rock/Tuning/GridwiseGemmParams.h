//===- GridwiseGemmParams.h - MLIR tuning parameter generation --------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
#define MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H

#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/Serializable.h"

namespace llvm {
class raw_ostream;
} // end namespace llvm

namespace mlir {
class Type;
namespace rock {
enum class GemmDimension : uint32_t { G = 0, K = 1, MorN = 2 };
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, GemmDimension dim);

// Remove this enum after both xdlops and non-xdlops are converted to new
// gridwise gemm form. 0 : gemmG dimension. 1 : gemmK dimension. 2 : gemmM or
// gemmN dimension.
enum GemmDimensions { GemmG = 0, GemmK = 1, GemmMorN = 2 };

// Remove after old swizzle detector is removed.
constexpr int64_t gemmCDimM = 1;
constexpr int64_t gemmCDimN = 2;

struct InitParams {
  int64_t gemmMPerBlock;
  int64_t gemmNPerBlock;
  int64_t gemmKPerBlock;
  int64_t gemmKPack;
};

struct InitParamsNonXDL : InitParams, Serializable<InitParamsNonXDL> {
  constexpr InitParamsNonXDL(uint32_t bSize, int64_t mPerBlock,
                             int64_t nPerBlock, int64_t kPerBlock,
                             int64_t mPerThread, int64_t nPerThread)
      : InitParams{mPerBlock, nPerBlock, kPerBlock, int64_t(1)},
        gemmMPerThread(mPerThread), gemmNPerThread(nPerThread),
        blockSize(bSize) {}
  int64_t gemmMPerThread;
  int64_t gemmNPerThread;
  uint32_t blockSize;

  constexpr InitParamsNonXDL()
      : InitParamsNonXDL(0U, 0LL, 0LL, 0LL, 0LL, 0LL) {}

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.blockSize);
    f(self.gemmMPerBlock);
    f(self.gemmNPerBlock);
    f(self.gemmKPerBlock);
    f(self.gemmMPerThread);
    f(self.gemmNPerThread);
  }
};

struct InitParamsXDL : InitParams, Serializable<InitParamsXDL> {
  constexpr InitParamsXDL(int64_t mPerBlock, int64_t nPerBlock,
                          int64_t kPerBlock, int64_t mPerWave, int64_t nPerWave,
                          int64_t kPack, bool aThreadCopyMoreGemmK,
                          bool bThreadCopyMoreGemmKPack)
      : InitParams{mPerBlock, nPerBlock, kPerBlock, kPack},
        gemmMPerWave(mPerWave), gemmNPerWave(nPerWave),
        gemmAThreadCopyMoreGemmK(aThreadCopyMoreGemmK),
        gemmBThreadCopyMoreGemmKPack(bThreadCopyMoreGemmKPack) {}

  constexpr InitParamsXDL()
      : InitParamsXDL(0LL, 0LL, 0LL, 0LL, 0LL, 0LL, false, false) {}

  int64_t gemmMPerWave;
  int64_t gemmNPerWave;
  bool gemmAThreadCopyMoreGemmK;
  bool gemmBThreadCopyMoreGemmKPack;

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.gemmMPerBlock);
    f(self.gemmNPerBlock);
    f(self.gemmKPerBlock);
    f(self.gemmMPerWave);
    f(self.gemmNPerWave);
    f(self.gemmKPack);
    f(self.gemmAThreadCopyMoreGemmK);
    f(self.gemmBThreadCopyMoreGemmKPack);
  }
};

template <typename T> std::string genDebugForParams(T params) {
  std::ostringstream os;
  params.visit(params, [&os](auto &arg) { os << arg << ","; });
  os << "\n";
  return os.str();
}

template <typename InitParamType> class BasePopulateParams {
private:
  struct InitParamData {
    InitParamType paramSet;
    size_t original_pos;
    int64_t padding_amount;

    bool operator<(const InitParamData &rhs) {
      if (this->padding_amount < rhs.padding_amount) {
        return true;
      } else if (this->padding_amount == rhs.padding_amount) {
        if (this->original_pos < rhs.original_pos) {
          return true;
        }
        return false;
      }
      return false;
    }
  };

  std::vector<InitParamData>
  createParamData(const ArrayRef<InitParamType> &initParams,
                  const GemmSize &gemmSize) {
    std::vector<InitParamData> res;
    for (size_t pos = 0; pos < initParams.size(); pos++) {
      InitParamType paramSet = initParams[pos];
      InitParamData paramData;
      paramData.paramSet = paramSet;
      paramData.original_pos = pos;
      paramData.padding_amount = calculatePaddingAmount(paramSet, gemmSize);
      assert(paramData.original_pos >= 0);
      assert(paramData.padding_amount >= 0);
      res.push_back(paramData);
    }
    return res;
  }

protected:
  // Interface function to check whether the given GEMM is valid
  virtual LogicalResult isValidGemm(const InitParamType &param,
                                    const GemmSize &gemmSize) const = 0;

  // Interface function to calculate padding amount of a given param set
  virtual int64_t calculatePaddingAmount(const InitParamType &params,
                                         const GemmSize &gemmSize) const = 0;

  // This function will order the initParams used in a non-tuning flow to
  // prioritize params that does not require padding to come first while
  // otherwise maintaining the provided order.
  std::vector<InitParamType>
  orderInitParams(const ArrayRef<InitParamType> &initParams,
                  const GemmSize &gemmSize) {
    std::vector<InitParamData> initParamData =
        createParamData(initParams, gemmSize);
    std::sort(initParamData.begin(), initParamData.end());

    std::vector<InitParamType> orderedParams;
    orderedParams.resize(initParams.size());
    std::transform(initParamData.begin(), initParamData.end(),
                   orderedParams.begin(),
                   [](InitParamData paramData) -> InitParamType {
                     return paramData.paramSet;
                   });
    return orderedParams;
  }

public:
  virtual ~BasePopulateParams() {}
};

class PopulateParams : public BasePopulateParams<InitParamsNonXDL> {
private:
  static constexpr size_t nInitParameters = 21;
  static const InitParamsNonXDL initParameters[nInitParameters];
  // if can't select config from above , use this config to do
  // padding kernel for example , GemmK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk
  static const InitParams universalParameters;

  LogicalResult
  calculateBlockGemmPerformanceParameters(const InitParamsNonXDL &param,
                                          RockGemmWrapperInterface op);

  LogicalResult populateDerived(RockGemmWrapperInterface op,
                                const InitParamsNonXDL &validParams,
                                GemmSize &gemmSize, uint32_t &gridSize);

public:
  LogicalResult obtainTuningParameters(RockGemmWrapperInterface op,
                                       uint32_t blockSizeOverride,
                                       const std::string &perfConfig,
                                       InitParamsNonXDL &validParams,
                                       uint32_t &gridSize);

  std::vector<InitParamsNonXDL> getTuningParameters(KernelType opType,
                                                    Type dataType) const;

  const InitParams &getUniversalParameters() const;

  LogicalResult isValidGemm(const InitParamsNonXDL &param,
                            const GemmSize &gemmSize) const override;

  int64_t calculatePaddingAmount(const InitParamsNonXDL &params,
                                 const GemmSize &gemmSize) const override;
};

class PopulateParamsXDL : public BasePopulateParams<InitParamsXDL> {
private:
  static constexpr size_t nInitParameters = 9;
  // Initial tuning parameters for forward convolution and backward
  // convolution.
  static const InitParamsXDL initParameters[nInitParameters];

  static constexpr size_t nInitParametersForwardI8 = 12;
  // Tuning parameters for i8 convolutions.
  static const InitParamsXDL initParametersForwardI8[nInitParametersForwardI8];

  static constexpr int64_t waveSize = 64;

  // if can't select config from above , use this config to do
  // padding kernel for example , GEMMK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk.
  static const InitParams universalParameters;

  uint32_t obtainBlockSize(const InitParamsXDL &params, int64_t waveSize);

  LogicalResult getKBlocks(Conv2DBwdWeightOp op, const GemmSize &gemmSize,
                           const InitParamsXDL &params, int64_t &gemmKBlocks);

  LogicalResult isValidBlockwiseGemmXDLOPS(const InitParamsXDL &param,
                                           RockGemmWrapperInterface op,
                                           uint32_t blockSize);

  LogicalResult populateDerived(RockGemmWrapperInterface op,
                                const InitParamsXDL &validParams,
                                GemmSize &gemmSize, uint32_t &blockSize,
                                uint32_t &gridSize, int64_t &gemmKBlocks);

  LogicalResult populatePaddingKernelDerived(RockGemmWrapperInterface op,
                                             const InitParamsXDL &validParams,
                                             GemmSize &gemmSize,
                                             uint32_t &blockSize,
                                             uint32_t &gridSize);

public:
  LogicalResult obtainTuningParameters(RockGemmWrapperInterface op,
                                       uint32_t blockSizeOverride,
                                       const std::string &perfConfig,
                                       InitParamsXDL &validParams,
                                       uint32_t &blockSize, uint32_t &gridSize,
                                       int64_t &gemmKBlocks);

  std::vector<InitParamsXDL> getTuningParameters(KernelType opType,
                                                 Type dataType) const;
  const InitParams &getUniversalParameters() const;

  LogicalResult isValidGemm(const InitParamsXDL &param,
                            const GemmSize &gemmSize) const override;

  int64_t calculatePaddingAmount(const InitParamsXDL &params,
                                 const GemmSize &gemmSize) const override;
};

// This core function to calculate the required padding amount
// given a gemm size.
Optional<GemmSize> calculatePadding(int64_t kPerBlock, int64_t mPerBlock,
                                    int64_t nPerBlock, const GemmSize &gemmSize,
                                    int64_t kPack = 1);

/// Given a tuning parameter struct, determine how much padding the gemm with
/// a given gemm size requires. Returns None if no padding is needed. The
/// values in the returned gemm context represent the number of 0s that need to
/// be added to the given dimension.
Optional<GemmSize> requiredPadding(Attribute params, GemmSize gemmSize);

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
