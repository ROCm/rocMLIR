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

// Remove this enum after both accelerated and non-accelerated are converted to
// new gridwise gemm form. 0 : gemmG dimension. 1 : gemmK dimension. 2 : gemmM
// or gemmN dimension.
enum GemmDimensions { GemmG = 0, GemmK = 1, GemmMorN = 2 };

// Remove after old swizzle detector is removed.
constexpr int64_t gemmCDimM = 1;
constexpr int64_t gemmCDimN = 2;

struct InitParams {
  int64_t gemmMPerBlock;
  int64_t gemmNPerBlock;
  int64_t gemmKPerBlock;
};

// This core function to calculate the required padding amount
// given a gemm size.
std::optional<GemmSize> calculatePadding(int64_t kPerBlock, int64_t mPerBlock,
                                         int64_t nPerBlock,
                                         const GemmSize &gemmSize,
                                         int64_t kPack = 1);

/// Given a tuning parameter struct, determine how much padding the gemm with
/// a given gemm size requires. Returns None if no padding is needed. The
/// values in the returned gemm context represent the number of 0s that need to
/// be added to the given dimension.
std::optional<GemmSize> requiredPadding(Attribute params, GemmSize gemmSize);

/// Store information useful for populating perf configurations
struct PopulateParamsInfo {
  GemmSize gemmSize;
  SmallString<32> arch;
  GemmFeatures gemmFeatures;
  Type gemmAType;
  Type gemmBType;
  KernelType kernelType;
  int64_t batchSize;
  uint32_t numCu;

  PopulateParamsInfo(GemmSize gemmSize, StringRef arch,
                     GemmFeatures gemmFeatures, Type gemmAType, Type gemmBType,
                     KernelType kernelType)
      : gemmSize(gemmSize), arch(arch), gemmFeatures(gemmFeatures),
        gemmAType(gemmAType), gemmBType(gemmBType), kernelType(kernelType) {}

  PopulateParamsInfo(GemmSize gemmSize, StringRef arch,
                     GemmFeatures gemmFeatures, Type gemmAType, Type gemmBType,
                     KernelType kernelType, int64_t batchSize, uint32_t numCu)
      : gemmSize(gemmSize), arch(arch), gemmFeatures(gemmFeatures),
        gemmAType(gemmAType), gemmBType(gemmBType), kernelType(kernelType),
        batchSize(batchSize), numCu(numCu) {}

  /// Extract the relevant information from a RockGemmWrapperInterface operation
  static PopulateParamsInfo fromOp(RockGemmWrapperInterface op);
};

struct InitParamsNonAccel : InitParams, Serializable<InitParamsNonAccel> {
  constexpr InitParamsNonAccel(uint32_t bSize, int64_t mPerBlock,
                               int64_t nPerBlock, int64_t kPerBlock,
                               int64_t mPerThread, int64_t nPerThread)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerThread(mPerThread),
        gemmNPerThread(nPerThread), blockSize(bSize) {}
  int64_t gemmMPerThread;
  int64_t gemmNPerThread;
  uint32_t blockSize;

  constexpr InitParamsNonAccel()
      : InitParamsNonAccel(0U, 0LL, 0LL, 0LL, 0LL, 0LL) {}

  int64_t getKPack() { return 1; }

  template <class Self, class F> static void visit(Self &&self, F f) {
    f(self.blockSize);
    f(self.gemmMPerBlock);
    f(self.gemmNPerBlock);
    f(self.gemmKPerBlock);
    f(self.gemmMPerThread);
    f(self.gemmNPerThread);
  }
};

struct InitParamsAccel : InitParams, Serializable<InitParamsAccel> {
  constexpr InitParamsAccel(int64_t mPerBlock, int64_t nPerBlock,
                            int64_t kPerBlock, int64_t mPerWave,
                            int64_t nPerWave, int64_t kPack,
                            bool aThreadCopyMoreGemmK,
                            bool bThreadCopyMoreGemmKPack)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerWave(mPerWave),
        gemmNPerWave(nPerWave), gemmKPack(kPack),
        gemmAThreadCopyMoreGemmK(aThreadCopyMoreGemmK),
        gemmBThreadCopyMoreGemmKPack(bThreadCopyMoreGemmKPack) {}

  constexpr InitParamsAccel()
      : InitParamsAccel(0LL, 0LL, 0LL, 0LL, 0LL, 0LL, false, false) {}

  int64_t getKPack() { return gemmKPack; }

  int64_t gemmMPerWave;
  int64_t gemmNPerWave;
  int64_t gemmKPack;
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

  // Interface function to calculate padding amount of a given param set
  virtual int64_t calculatePaddingAmount(const InitParamType &params,
                                         const GemmSize &gemmSize) const = 0;

protected:
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

//
// Non acceleration parameter initialization interface
//
class PopulateParams : public BasePopulateParams<InitParamsNonAccel> {
private:
  static constexpr size_t nInitParameters = 21;
  static const InitParamsNonAccel initParameters[nInitParameters];
  // if can't select config from above , use this config to do
  // padding kernel for example , GemmK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk

  LogicalResult
  calculateBlockGemmPerformanceParameters(const InitParamsNonAccel &param);

  LogicalResult populateDerived(const InitParamsNonAccel &validParams,
                                GemmSize &gemmSize, uint32_t &gridSize);

public:
  LogicalResult obtainTuningParameters(RockGemmWrapperInterface op,
                                       uint32_t blockSizeOverride,
                                       const std::string &perfConfig,
                                       InitParamsNonAccel &validParams,
                                       uint32_t &gridSize);

  LogicalResult obtainTuningParameters(const PopulateParamsInfo &info,
                                       uint32_t blockSizeOverride,
                                       const std::string &perfConfig,
                                       InitParamsNonAccel &validParams,
                                       uint32_t &gridSize);

  std::vector<InitParamsNonAccel>
  getTuningParameters(KernelType opType, Type dataTypeA, Type dataTypeB) const;

  int64_t calculatePaddingAmount(const InitParamsNonAccel &params,
                                 const GemmSize &gemmSize) const override;
};

//
// Acceleration parameter initialization interface
//
class PopulateParamsAccel : public BasePopulateParams<InitParamsAccel> {
public:
  static std::unique_ptr<PopulateParamsAccel> select(GemmFeatures features);

  LogicalResult obtainTuningParameters(RockGemmWrapperInterface op,
                                       uint32_t blockSizeOverride,
                                       const std::string &perfConfig,
                                       InitParamsAccel &validParams,
                                       uint32_t &blockSize, uint32_t &gridSize,
                                       int64_t &gemmKBlocks);

  virtual LogicalResult obtainTuningParameters(
      const PopulateParamsInfo &info, uint32_t blockSizeOverride,
      const std::string &perfConfig, InitParamsAccel &validParams,
      uint32_t &blockSize, uint32_t &gridSize, int64_t &gemmKBlocks);

  int64_t calculatePaddingAmount(const InitParamsAccel &params,
                                 const GemmSize &gemmSize) const override;

  virtual std::vector<InitParamsAccel>
  getTuningParameters(KernelType opType, Type dataTypeA, Type dataTypeB,
                      StringRef arch) const = 0;
  virtual Attribute getGemmParamsAttr(OpBuilder builder,
                                      InitParamsAccel validParams) const = 0;

protected:
  virtual LogicalResult isValidBlockwiseGemm(const InitParamsAccel &param,
                                             Type dataTypeA, Type dataTypeB,
                                             StringRef arch,
                                             uint32_t blockSize) = 0;
  // if can't select config from above , use this config to do
  // padding kernel for example , GEMMK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk.
  uint32_t obtainBlockSize(const InitParamsAccel &params, int64_t waveSize);

  LogicalResult getKBlocks(int64_t batchSize, const GemmSize &gemmSize,
                           const InitParamsAccel &params, int64_t &gemmKBlocks,
                           uint32_t numCu);
  LogicalResult populateDerived(const InitParamsAccel &validParams,
                                const PopulateParamsInfo &info,
                                GemmSize &gemmSize, uint32_t &blockSize,
                                uint32_t &gridSize, int64_t &gemmKBlocks);
  LogicalResult populatePaddingKernelDerived(RockGemmWrapperInterface op,
                                             const InitParamsAccel &validParams,
                                             GemmSize &gemmSize,
                                             uint32_t &blockSize,
                                             uint32_t &gridSize);
};

//
// Xdlops interface
//
class PopulateParamsXDL : public PopulateParamsAccel {
  static constexpr size_t nInitParameters = 5;
  // Initial tuning parameters for forward convolution and backward
  // convolution.
  static const InitParamsAccel initParameters[nInitParameters];

  static constexpr size_t nInitParametersFp16 = 3;
  // Tuning parameters for fp16/bf16 convolutions.
  static const InitParamsAccel initParametersFp16[nInitParametersFp16];

  static constexpr size_t nInitParametersForward8Bit = 5;
  // Tuning parameters for i8 convolutions.
  static const InitParamsAccel
      initParametersForward8Bit[nInitParametersForward8Bit];

public:
  std::vector<InitParamsAccel>
  getTuningParameters(KernelType opType, Type dataTypeA, Type dataTypeB,
                      StringRef arch) const override;
  Attribute getGemmParamsAttr(OpBuilder builder,
                              InitParamsAccel validParams) const override;

protected:
  LogicalResult isValidBlockwiseGemm(const InitParamsAccel &param,
                                     Type dataTypeA, Type dataTypeB,
                                     StringRef arch,
                                     uint32_t blockSize) override;
};

//
// Wmma interface
//
class PopulateParamsWmma : public PopulateParamsAccel {
private:
  static constexpr size_t nInitParametersFp16 = 3;
  // Tuning parameters for fp16/bf16 convolutions.
  static const InitParamsAccel initParametersFp16[nInitParametersFp16];

  static constexpr size_t nInitParametersForward8Bit = 5;
  // Tuning parameters for i8 convolutions.
  static const InitParamsAccel
      initParametersForward8Bit[nInitParametersForward8Bit];

  std::vector<InitParamsAccel>
  getTuningParameters(KernelType opType, Type dataTypeA, Type dataTypeB,
                      StringRef arch) const override;

  Attribute getGemmParamsAttr(OpBuilder builder,
                              InitParamsAccel validParams) const override;

protected:
  LogicalResult isValidBlockwiseGemm(const InitParamsAccel &param,
                                     Type dataTypeA, Type dataTypeB,
                                     StringRef arch,
                                     uint32_t blockSize) override;
};

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
