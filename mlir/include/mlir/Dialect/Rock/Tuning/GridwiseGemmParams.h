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

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GemmContext.h"
#include "mlir/Dialect/Rock/Tuning/Serializable.h"

namespace llvm {
class raw_ostream;
} // end namespace llvm

namespace mlir {
class Type;
namespace rock {
struct ConvolutionContext;

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
};

struct GemmSize {
  int64_t gemmG;
  int64_t gemmM;
  int64_t gemmN;
  int64_t gemmK;
};

struct InitParamsNonXDL : InitParams, Serializable<InitParamsNonXDL> {
  constexpr InitParamsNonXDL(uint32_t bSize, int64_t mPerBlock,
                             int64_t nPerBlock, int64_t kPerBlock,
                             int64_t mPerThread, int64_t nPerThread)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerThread(mPerThread),
        gemmNPerThread(nPerThread), blockSize(bSize) {}
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
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerWave(mPerWave),
        gemmNPerWave(nPerWave), gemmKPack(kPack),
        gemmAThreadCopyMoreGemmK(aThreadCopyMoreGemmK),
        gemmBThreadCopyMoreGemmKPack(bThreadCopyMoreGemmKPack) {}

  constexpr InitParamsXDL()
      : InitParamsXDL(0LL, 0LL, 0LL, 0LL, 0LL, 0LL, false, false) {}

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
protected:
  // Interface function to check whether the given GEMM is valid
  virtual LogicalResult isValidGemm(const InitParamType &param,
                                    const GemmSize &gemmSize) const = 0;

  // This function will order the initParams used in a non-tuning flow to
  // prioritize params that does not require padding to come first while
  // otherwise maintaining the provided order.
  // TODO(@manupak) : improve this to order based on amount of padding added.
  std::vector<InitParamType>
  orderInitParams(const ArrayRef<InitParamType> &initParams,
                  const GemmSize &gemmSize) {
    std::vector<InitParamType> paddingRequiredParams;
    std::vector<InitParamType> paddingNotRequiredParams;
    for (const InitParamType &param : initParams) {
      if (isValidGemm(param, gemmSize).succeeded()) {
        paddingNotRequiredParams.emplace_back(param);
      } else {
        paddingRequiredParams.emplace_back(param);
      }
    }
    std::vector<InitParamType> orderedParams;
    std::move(paddingNotRequiredParams.begin(), paddingNotRequiredParams.end(),
              std::back_inserter(orderedParams));
    std::move(paddingRequiredParams.begin(), paddingRequiredParams.end(),
              std::back_inserter(orderedParams));
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
                                          const ConvolutionContext &ctx);

  LogicalResult populateDerived(ConvolutionContext &ctx,
                                const InitParamsNonXDL &validParams,
                                GemmSize &gemmSize, uint32_t &gridSize);

  LogicalResult
  populatePaddingKernelDerived(ConvolutionContext &ctx,
                               const InitParamsNonXDL &validParams,
                               GemmSize &gemmSize, uint32_t &gridSize);

public:
  LogicalResult obtainTuningParameters(Operation *op,
                                       uint32_t blockSizeOverride,
                                       const std::string &perfConfig,
                                       InitParamsNonXDL &validParams,
                                       uint32_t &gridSize);

  ArrayRef<InitParamsNonXDL> getTuningParameters(ConvOpType dir,
                                                 Type dataType) const;

  const InitParams &getUniversalParameters() const;

  LogicalResult isValidGemm(const InitParamsNonXDL &param,
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

  LogicalResult getKBlocks(ConvolutionContext &ctx, const GemmSize &gemmSize,
                           const InitParamsXDL &params, int64_t &gemmKBlocks);

  LogicalResult isValidBlockwiseGemmXDLOPS(const InitParamsXDL &param,
                                           ConvolutionContext &ctx,
                                           uint32_t blockSize);

  LogicalResult populateDerived(ConvolutionContext &ctx,
                                const InitParamsXDL &validParams,
                                GemmSize &gemmSize, uint32_t &blockSize,
                                uint32_t &gridSize, int64_t &gemmKBlocks);

  LogicalResult populatePaddingKernelDerived(ConvolutionContext &ctx,
                                             const InitParamsXDL &validParams,
                                             GemmSize &gemmSize,
                                             uint32_t &blockSize,
                                             uint32_t &gridSize);

  LogicalResult isValidGridGemmXdlops(GemmSize &gemmSize);

public:
  LogicalResult obtainTuningParameters(Operation *op,
                                       uint32_t blockSizeOverride,
                                       const std::string &perfConfig,
                                       InitParamsXDL &validParams,
                                       uint32_t &blockSize, uint32_t &gridSize,
                                       int64_t &gemmKBlocks);

  ArrayRef<InitParamsXDL> getTuningParameters(ConvOpType dir,
                                              Type dataType) const;
  const InitParams &getUniversalParameters() const;

  LogicalResult isValidGemm(const InitParamsXDL &param,
                            const GemmSize &gemmSize) const override;
};

// This core function to calculate the required padding amount
// given a gemm size.
Optional<GemmContext> calculatePadding(int64_t kPerBlock, int64_t mPerBlock,
                                       int64_t nPerBlock,
                                       const GemmContext &gemmSize,
                                       int64_t kPack = 1);

Optional<GemmContext> calculatePadding(int64_t kPerBlock, int64_t mPerBlock,
                                       int64_t nPerBlock,
                                       const GemmSize &gemmSize,
                                       int64_t kPack = 1);

// The function is used to compute extra padding sizes.
// For example, if gemmM size is 3 and gemmMPerBlock is 64,
// we set gemmMExtra be 64 so (gemmM+gemmMExtra)%gemmMPerBlock=0.
//
// If padding is needed, returns a GemmContext containing the number of elements
// needed to pad the M, N, and K dimensions (**not** the new gemm size).
// Otherwise, returns None
template <typename T>
Optional<GemmContext> calculatePaddingKernelSize(GemmContext gemmSize,
                                                 ConvOpType dir, Type dataType,
                                                 T populateParams) {
  bool needExtraPad = false;
  int64_t gemmMExtra, gemmNExtra, gemmKExtra;
  gemmMExtra = gemmNExtra = gemmKExtra = 0;

  auto configParams = populateParams.getTuningParameters(dir, dataType);
  size_t numOfFailedConfigs = 0;
  for (auto &params : configParams) {
    if (gemmSize.m % params.gemmMPerBlock == 0 &&
        gemmSize.k % params.gemmKPerBlock == 0 &&
        gemmSize.n % params.gemmNPerBlock == 0) {
      break;
    }
    numOfFailedConfigs++;
  }

  auto extraParams = populateParams.getUniversalParameters();
  if (numOfFailedConfigs == configParams.size()) {
    needExtraPad = true;
    int64_t gemmMRemain, gemmKRemain, gemmNRemain;

    gemmMRemain = gemmSize.m % extraParams.gemmMPerBlock;
    if (gemmMRemain != 0)
      gemmMExtra = extraParams.gemmMPerBlock - gemmMRemain;

    gemmNRemain = gemmSize.n % extraParams.gemmNPerBlock;
    if (gemmNRemain != 0)
      gemmNExtra = extraParams.gemmNPerBlock - gemmNRemain;

    gemmKRemain = gemmSize.k % extraParams.gemmKPerBlock;
    if (gemmKRemain != 0)
      gemmKExtra = extraParams.gemmKPerBlock - gemmKRemain;

    // llvm::errs() << "gemmMExtra: " << gemmMExtra << "gemmNExtra: " <<
    // gemmNExtra << "gemmKExtra: " << gemmKExtra << "\n";
  }

  if (needExtraPad)
    return GemmContext(gemmMExtra, gemmKExtra, gemmNExtra);
  return llvm::None;
}

/// Given a tuning parameter struct, determine how much padding the gemm with
/// a given gemm size requires. Returns None if no padding is needed. The
/// values in the returned gemm context represent the number of 0s that need to
/// be added to the given dimension.
Optional<GemmContext> requiredPadding(Attribute params, GemmContext gemmSize);

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
