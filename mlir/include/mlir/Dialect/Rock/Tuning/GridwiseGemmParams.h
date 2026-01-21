//===- GridwiseGemmParams.h - MLIR tuning parameter generation ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation for single-gemm ops
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
#define MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H

#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include <optional>

namespace llvm {
class raw_ostream;
} // end namespace llvm

namespace mlir {
class Type;
namespace rock {
enum class GemmDimension : uint32_t { G = 0, K = 1, MorN = 2 };
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, GemmDimension dim);

// This core function to calculate the required padding amount
// given a gemm size.
std::optional<GemmSize> calculatePadding(int64_t kPerBlock, int64_t mPerBlock,
                                         int64_t nPerBlock,
                                         const GemmSize &gemmSize,
                                         int64_t kPack = 1);

GemmSize calculatePaddedGemmSize(int64_t kPerBlock, int64_t mPerBlock,
                                 int64_t nPerBlock, GemmSize gemmSize,
                                 int64_t kPack = 1);

/// Given a tuning parameter struct, determine how much padding the gemm with
/// a given gemm size requires. Returns None if no padding is needed. The
/// values in the returned gemm context represent the number of 0s that need to
/// be added to the given dimension. The mulBy* arguments multiply the
/// corresponding dimension of the attributes.
std::optional<GemmSize> requiredPadding(Attribute params, GemmSize gemmSize,
                                        int64_t mulByKPerBlock = 1,
                                        int64_t mulByMPerBlock = 1,
                                        int64_t mulByNPerBlock = 1);

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
        gemmAType(gemmAType), gemmBType(gemmBType), kernelType(kernelType),
        hasFusedReduction(false) {}

  PopulateParamsInfo(GemmSize gemmSize, StringRef arch,
                     GemmFeatures gemmFeatures, Type gemmAType, Type gemmBType,
                     KernelType kernelType, int64_t batchSize, uint32_t numCu)
      : gemmSize(gemmSize), arch(arch), gemmFeatures(gemmFeatures),
        gemmAType(gemmAType), gemmBType(gemmBType), kernelType(kernelType),
        batchSize(batchSize), numCu(numCu), hasFusedReduction(false) {}

  /// Extract the relevant information from a RockGemmWrapperInterface operation
  static PopulateParamsInfo fromOp(RockGemmWrapperInterface op);
};

template <typename ParamAttrType>
class BasePopulateParams {
private:
  struct ParamData {
    ParamAttrType paramAttr;
    size_t originalPos;
    int64_t paddingAmount;

    bool operator<(const ParamData &rhs) const {
      if (this->paddingAmount < rhs.paddingAmount) {
        return true;
      }
      if (this->paddingAmount == rhs.paddingAmount) {
        return (this->originalPos < rhs.originalPos);
      }
      return false;
    }
  };

  std::vector<ParamData> createParamData(ArrayRef<ParamAttrType> params,
                                         const GemmSize &gemmSize) {
    std::vector<ParamData> res;
    for (size_t pos = 0; pos < params.size(); pos++) {
      ParamAttrType paramAttr = params[pos];
      ParamData paramData;
      paramData.paramAttr = paramAttr;
      paramData.originalPos = pos;
      paramData.paddingAmount = calculatePaddingAmount(paramAttr, gemmSize);
      assert(paramData.originalPos >= 0);
      assert(paramData.paddingAmount >= 0);
      res.push_back(paramData);
    }
    return res;
  }

  // Interface function to calculate padding amount of a given param set
  virtual int64_t calculatePaddingAmount(ParamAttrType params,
                                         const GemmSize &gemmSize) const = 0;

public:
  // This function will order the params used in a non-tuning flow to
  // prioritize params that does not require padding to come first while
  // otherwise maintaining the provided order.
  std::vector<ParamAttrType> orderParams(ArrayRef<ParamAttrType> params,
                                         const GemmSize &gemmSize) {
    std::vector<ParamData> paramDataVec = createParamData(params, gemmSize);
    std::sort(paramDataVec.begin(), paramDataVec.end());

    std::vector<ParamAttrType> orderedParams;
    orderedParams.resize(params.size());
    std::transform(paramDataVec.begin(), paramDataVec.end(),
                   orderedParams.begin(),
                   [](ParamData paramData) -> ParamAttrType {
                     return paramData.paramAttr;
                   });
    return orderedParams;
  }

  // Succeed if the given `params` could be a valid set of tuning parameters for
  // `info`. This is not a guarantee that a given set of parameters will pass
  // applicability, but it should filter out inapplicable configs.
  virtual LogicalResult paramsProbablyValid(OpBuilder &b,
                                            const PopulateParamsInfo &info,
                                            ParamAttrType params) = 0;

  // Succeed if `params` should be included in a "full" tuning space that
  // excludes those known to not yield good performance on the problem described
  // in `info`. This function uses hardcoded heuristics.
  virtual LogicalResult couldBePerformant(const PopulateParamsInfo &info,
                                          ParamAttrType params) = 0;

  virtual ~BasePopulateParams() {}
};

//
// Acceleration parameter initialization interface
//
class PopulateParamsAccel : public BasePopulateParams<AccelGemmParamsAttr> {
public:
  static std::unique_ptr<PopulateParamsAccel> select(GemmFeatures features);

  LogicalResult obtainTuningParameters(OpBuilder &b,
                                       RockGemmWrapperInterface op,
                                       const StringRef perfConfig,
                                       AccelGemmParamsAttr &validParams);

  virtual LogicalResult
  obtainTuningParameters(OpBuilder &b, const PopulateParamsInfo &info,
                         const StringRef perfConfig,
                         AccelGemmParamsAttr &validParams);

  int64_t calculatePaddingAmount(AccelGemmParamsAttr params,
                                 const GemmSize &gemmSize) const override;

  // Return the set of heuristic tuning parameters for the given opType, data
  // types, and architecture.
  virtual std::vector<AccelGemmParamsAttr>
  getTuningParameters(OpBuilder &b, KernelType opType, Type dataTypeA,
                      Type dataTypeB, StringRef arch) const = 0;

  // Note that this is a method on the general class because the distinguishing
  // of MFMA and WMMA paths is handled under the hood in populateDerived().
  LogicalResult paramsProbablyValid(OpBuilder &b,
                                    const PopulateParamsInfo &info,
                                    AccelGemmParamsAttr params) override;

  LogicalResult couldBePerformant(const PopulateParamsInfo &info,
                                  AccelGemmParamsAttr params) override;

  virtual LogicalResult
  isValidBlockwiseGemm(RockAccelTuningParamAttrInterface param, Type dataTypeA,
                       Type dataTypeB, StringRef arch) = 0;

protected:
  LogicalResult populatePaddingKernelDerived(RockGemmWrapperInterface op,
                                             AccelGemmParamsAttr validParams,
                                             GemmSize &gemmSize,
                                             uint32_t &blockSize,
                                             uint32_t &gridSize);

  /// The actual implementation of couldBePerformant(), which shouldn't exist
  /// once we merge gridwise_gemm and gridwise_gemm_accel and thus flatten
  /// out the class hierarchy in this file.
  virtual LogicalResult specificCouldBePerformant(AccelGemmParamsAttr params,
                                                  Type dataTypeA,
                                                  Type dataTypeB) = 0;
};

//
// Xdlops interface
//
class PopulateParamsXDL : public PopulateParamsAccel {
#define XDL_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef XDL_DECLARATIONS_GEN

  friend class ParamLookupTable<AccelGemmParamsAttr>;

public:
  std::vector<AccelGemmParamsAttr>
  getTuningParameters(OpBuilder &b, KernelType opType, Type dataTypeA,
                      Type dataTypeB, StringRef arch) const override;

  LogicalResult isValidBlockwiseGemm(RockAccelTuningParamAttrInterface param,
                                     Type dataTypeA, Type dataTypeB,
                                     StringRef arch) override;

protected:
  LogicalResult specificCouldBePerformant(AccelGemmParamsAttr params,
                                          Type dataTypeA,
                                          Type dataTypeB) override;
};

//
// Wmma interface
//
class PopulateParamsWmma : public PopulateParamsAccel {
private:
#define Wmma_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Wmma_DECLARATIONS_GEN

  friend class ParamLookupTable<AccelGemmParamsAttr>;

public:
  std::vector<AccelGemmParamsAttr>
  getTuningParameters(OpBuilder &b, KernelType opType, Type dataTypeA,
                      Type dataTypeB, StringRef arch) const override;

  LogicalResult isValidBlockwiseGemm(RockAccelTuningParamAttrInterface param,
                                     Type dataTypeA, Type dataTypeB,
                                     StringRef arch) override;

protected:
  LogicalResult specificCouldBePerformant(AccelGemmParamsAttr params,
                                          Type dataTypeA,
                                          Type dataTypeB) override;
};

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
