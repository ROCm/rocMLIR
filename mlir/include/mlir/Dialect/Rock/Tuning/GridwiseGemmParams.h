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
#include <optional>

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

GemmSize calculatePaddedGemmSize(const InitParams &params, GemmSize gemmSize,
                                 int64_t kPack = 1);

/// Given a tuning parameter struct, determine how much padding the gemm with
/// a given gemm size requires. Returns None if no padding is needed. The
/// values in the returned gemm context represent the number of 0s that need to
/// be added to the given dimension.
std::optional<GemmSize> requiredPadding(Attribute params, GemmSize gemmSize);

int64_t obtainBlockSize(int64_t waveSize, int64_t mPerBlock, int64_t nPerBlock,
                        int64_t mPerWave, int64_t nPerWave);

int64_t obtainBlockSize(int64_t waveSize,
                        RockAccelTuningParamAttrInterface params);

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
  bool hasFusedReduction;

  PopulateParamsInfo(GemmSize gemmSize, StringRef arch,
                     GemmFeatures gemmFeatures, Type gemmAType, Type gemmBType,
                     KernelType kernelType)
      : gemmSize(gemmSize), arch(arch), gemmFeatures(gemmFeatures),
        gemmAType(gemmAType), gemmBType(gemmBType), kernelType(kernelType), hasFusedReduction(false) {}

  PopulateParamsInfo(GemmSize gemmSize, StringRef arch,
                     GemmFeatures gemmFeatures, Type gemmAType, Type gemmBType,
                     KernelType kernelType, int64_t batchSize, uint32_t numCu)
      : gemmSize(gemmSize), arch(arch), gemmFeatures(gemmFeatures),
        gemmAType(gemmAType), gemmBType(gemmBType), kernelType(kernelType),
        batchSize(batchSize), numCu(numCu), hasFusedReduction(false) {}

  /// Extract the relevant information from a RockGemmWrapperInterface operation
  static PopulateParamsInfo fromOp(RockGemmWrapperInterface op);
};

struct InitParamsNonAccel : InitParams, Serializable<InitParamsNonAccel> {
  int64_t gemmMPerThread;
  int64_t gemmNPerThread;
  uint32_t blockSize;
  int64_t splitKFactor;

  constexpr InitParamsNonAccel(uint32_t bSize, int64_t mPerBlock,
                               int64_t nPerBlock, int64_t kPerBlock,
                               int64_t mPerThread, int64_t nPerThread,
                               int64_t splitKFactor)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerThread(mPerThread),
        gemmNPerThread(nPerThread), blockSize(bSize),
        splitKFactor(splitKFactor) {}

  constexpr InitParamsNonAccel()
      : InitParamsNonAccel(0U, 0LL, 0LL, 0LL, 0LL, 0LL, 1LL) {}

  InitParamsNonAccel(GeneralGemmParamsAttr attr)
      : InitParams{attr.getMPerBlock(), attr.getNPerBlock(),
                   attr.getKPerBlock()},
        gemmMPerThread(attr.getMPerThread()),
        gemmNPerThread(attr.getNPerThread()), blockSize(attr.getBlockSize()),
        splitKFactor(attr.getSplitKFactor()){};

  int64_t getKPack() { return 1; }

  template <class Self, class F>
  static void visit(Self &&self, F f) {
    f(self.blockSize);
    f(self.gemmMPerBlock);
    f(self.gemmNPerBlock);
    f(self.gemmKPerBlock);
    f(self.gemmMPerThread);
    f(self.gemmNPerThread);
    if (self.version != Version::V1)
      f(self.splitKFactor);
  }
};

struct InitParamsAccel : InitParams, Serializable<InitParamsAccel> {
  constexpr InitParamsAccel(int64_t mPerBlock, int64_t nPerBlock,
                            int64_t kPerBlock, int64_t mPerWave,
                            int64_t nPerWaveOrMnPerXdl, int64_t kPack,
                            int64_t splitKFactor, bool aThreadCopyMoreGemmK,
                            bool bThreadCopyMoreGemmKPack)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerWave(mPerWave),
        gemmNPerWaveOrMnPerXdl(nPerWaveOrMnPerXdl), gemmKPack(kPack),
        splitKFactor(splitKFactor),
        gemmAThreadCopyMoreGemmK(aThreadCopyMoreGemmK),
        gemmBThreadCopyMoreGemmKPack(bThreadCopyMoreGemmKPack) {}

  constexpr InitParamsAccel()
      : InitParamsAccel(0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 1LL, false, false) {}

  InitParamsAccel(XdlopsGemmParamsAttr attr)
      : InitParams{attr.getMPerBlock(), attr.getNPerBlock(),
                   attr.getKpackPerBlock()},
        gemmMPerWave(attr.getMPerWave()),
        gemmNPerWaveOrMnPerXdl(attr.getMnPerXdl()), gemmKPack(attr.getKpack()),
        splitKFactor(attr.getSplitKFactor()),
        gemmAThreadCopyMoreGemmK(attr.getForceUnroll()),
        gemmBThreadCopyMoreGemmKPack(false){};

  InitParamsAccel(WmmaGemmParamsAttr attr)
      : InitParams{attr.getMPerBlock(), attr.getNPerBlock(),
                   attr.getKpackPerBlock()},
        gemmMPerWave(attr.getMPerWave()),
        gemmNPerWaveOrMnPerXdl(attr.getNPerWave()), gemmKPack(attr.getKpack()),
        splitKFactor(attr.getSplitKFactor()),
        gemmAThreadCopyMoreGemmK(attr.getForceUnroll()),
        gemmBThreadCopyMoreGemmKPack(false){};

  int64_t getKPack() { return gemmKPack; }

  int64_t gemmMPerWave;
  int64_t gemmNPerWaveOrMnPerXdl;
  int64_t gemmKPack;
  int64_t splitKFactor;
  bool gemmAThreadCopyMoreGemmK;
  bool gemmBThreadCopyMoreGemmKPack;

  template <class Self, class F>
  static void visit(Self &&self, F f) {
    f(self.gemmMPerBlock);
    f(self.gemmNPerBlock);
    f(self.gemmKPerBlock);
    f(self.gemmMPerWave);
    f(self.gemmNPerWaveOrMnPerXdl);
    f(self.gemmKPack);
    if (self.version != Version::V1) {
      f(self.splitKFactor);
    }
    f(self.gemmAThreadCopyMoreGemmK);
    f(self.gemmBThreadCopyMoreGemmKPack);
  }
};

template <typename T>
std::string genDebugForParams(T params) {
  std::ostringstream os;
  params.visit(params, [&os](auto &arg) { os << arg << ","; });
  os << "\n";
  return os.str();
}

template <typename InitParamType>
class BasePopulateParams {
private:
  struct InitParamData {
    InitParamType paramSet;
    size_t original_pos;
    int64_t padding_amount;

    bool operator<(const InitParamData &rhs) {
      if (this->padding_amount < rhs.padding_amount) {
        return true;
      } else if (this->padding_amount == rhs.padding_amount) {
        return (this->original_pos < rhs.original_pos);
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

public:
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

  // Succeed if the given `params` could be a valid set of tuning parameters for
  // `info`. This is not a guarantee that a given set of parameters will pass
  // applicability, but it should filter out inapplicable configs.
  virtual LogicalResult paramsProbablyValid(OpBuilder &b,
                                            const PopulateParamsInfo &info,
                                            const InitParamType &params) = 0;

  // Succced if `params` should be included in a "full" tuning space that
  // excludes those known to not yeild good performance on the problem described
  // in `info`. This function uses hardcoded heuristics.
  virtual LogicalResult couldBePerformant(const PopulateParamsInfo &info,
                                          const InitParamType &params) = 0;

  // Convert the provided InitParamType into an MLIR `Attribute`.
  virtual Attribute getGemmParamsAttr(OpBuilder &b,
                                      const InitParamType &params) const = 0;
  virtual ~BasePopulateParams() {}
};

//
// Non acceleration parameter initialization interface
//
class PopulateParams : public BasePopulateParams<InitParamsNonAccel> {
private:
  static constexpr size_t nInitParameters = 30;
  static const InitParamsNonAccel initParameters[nInitParameters];
  // if can't select config from above , use this config to do
  // padding kernel for example , GemmK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk

  LogicalResult
  calculateBlockGemmPerformanceParameters(const InitParamsNonAccel &param);

  LogicalResult populateDerived(const InitParamsNonAccel &validParams);

public:
  LogicalResult obtainTuningParameters(RockGemmWrapperInterface op,
                                       const StringRef perfConfig,
                                       InitParamsNonAccel &validParams);

  LogicalResult obtainTuningParameters(OpBuilder &b,
                                       const PopulateParamsInfo &info,
                                       const StringRef perfConfig,
                                       InitParamsNonAccel &validParams);

  // Return the vector of heuristic parameters for a given kernel type and dat
  // type.
  std::vector<InitParamsNonAccel>
  getTuningParameters(KernelType opType, Type dataTypeA, Type dataTypeB) const;

  Attribute getGemmParamsAttr(OpBuilder &b,
                              const InitParamsNonAccel &params) const override;

  LogicalResult paramsProbablyValid(OpBuilder &b,
                                    const PopulateParamsInfo &info,
                                    const InitParamsNonAccel &params) override;

  LogicalResult couldBePerformant(const PopulateParamsInfo &info,
                                  const InitParamsNonAccel &params) override;

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
                                       const StringRef perfConfig,
                                       InitParamsAccel &validParams);

  virtual LogicalResult obtainTuningParameters(OpBuilder &b,
                                               const PopulateParamsInfo &info,
                                               const StringRef perfConfig,
                                               InitParamsAccel &validParams);

  int64_t calculatePaddingAmount(const InitParamsAccel &params,
                                 const GemmSize &gemmSize) const override;

  // Return the set of heuristic tuning parameters for the given opType, data
  // types, and architecture.
  virtual std::vector<InitParamsAccel>
  getTuningParameters(KernelType opType, Type dataTypeA, Type dataTypeB,
                      StringRef arch) const = 0;
  Attribute
  getGemmParamsAttr(OpBuilder &builder,
                    const InitParamsAccel &validParams) const override = 0;

  // Note that this is a method on the general class because the distinguishing
  // of MFMA and WMMA paths is handled under the hood in populateDerived().
  LogicalResult paramsProbablyValid(OpBuilder &b,
                                    const PopulateParamsInfo &info,
                                    const InitParamsAccel &params) override;

  LogicalResult couldBePerformant(const PopulateParamsInfo &info,
                                  const InitParamsAccel &params) override;

  virtual LogicalResult
  isValidBlockwiseGemm(RockAccelTuningParamAttrInterface param, Type dataTypeA,
                       Type dataTypeB, StringRef arch,
                       bool enableBlockSizeUpperLimit = true,
                       bool enableDPerWaveFiltering = true) = 0;

protected:
  LogicalResult populatePaddingKernelDerived(RockGemmWrapperInterface op,
                                             const InitParamsAccel &validParams,
                                             GemmSize &gemmSize,
                                             uint32_t &blockSize,
                                             uint32_t &gridSize);

  /// The actual implementation of couldBePerformant(), which shouldn't exist
  /// once we merge gridwise_gemm and gridwise_gemm_accel and thus flatten
  /// out the class heirachy in this file.
  virtual LogicalResult specificCouldBePerformant(const InitParamsAccel &params,
                                                  Type dataTypeA,
                                                  Type dataTypeB) = 0;
};

//
// Xdlops interface
//
class PopulateParamsXDL : public PopulateParamsAccel {
  static constexpr size_t nInitParameters = 40;
  // Initial tuning parameters for forward convolution and backward
  // convolution.
  static const InitParamsAccel initParameters[nInitParameters];

  static constexpr size_t nInitParametersFp16 = 40;
  // Tuning parameters for fp16/bf16 convolutions.
  static const InitParamsAccel initParametersFp16[nInitParametersFp16];

  static constexpr size_t nInitParametersForward8Bit = 40;
  // Tuning parameters for i8 convolutions.
  static const InitParamsAccel
      initParametersForward8Bit[nInitParametersForward8Bit];

public:
  std::vector<InitParamsAccel>
  getTuningParameters(KernelType opType, Type dataTypeA, Type dataTypeB,
                      StringRef arch) const override;
  Attribute
  getGemmParamsAttr(OpBuilder &builder,
                    const InitParamsAccel &validParams) const override;
  LogicalResult
  isValidBlockwiseGemm(RockAccelTuningParamAttrInterface param, Type dataTypeA,
                       Type dataTypeB, StringRef arch,
                       bool enableBlockSizeUpperLimit = true,
                       bool enableDPerWaveFiltering = true) override;

protected:
  LogicalResult specificCouldBePerformant(const InitParamsAccel &params,
                                          Type dataTypeA,
                                          Type dataTypeB) override;
};

//
// Wmma interface
//
class PopulateParamsWmma : public PopulateParamsAccel {
private:
  static constexpr size_t nInitParametersFp16 = 30;
  // Tuning parameters for fp16/bf16 convolutions.
  static const InitParamsAccel initParametersFp16[nInitParametersFp16];

  static constexpr size_t nInitParametersForward8Bit = 30;
  // Tuning parameters for i8 convolutions.
  static const InitParamsAccel
      initParametersForward8Bit[nInitParametersForward8Bit];

public:
  std::vector<InitParamsAccel>
  getTuningParameters(KernelType opType, Type dataTypeA, Type dataTypeB,
                      StringRef arch) const override;

  Attribute
  getGemmParamsAttr(OpBuilder &builder,
                    const InitParamsAccel &validParams) const override;

  LogicalResult
  isValidBlockwiseGemm(RockAccelTuningParamAttrInterface param, Type dataTypeA,
                       Type dataTypeB, StringRef arch,
                       bool enableBlockSizeUpperLimit = true,
                       bool enableDPerWaveFiltering = true) override;

protected:
  LogicalResult specificCouldBePerformant(const InitParamsAccel &params,
                                          Type dataTypeA,
                                          Type dataTypeB) override;
};

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
