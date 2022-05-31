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

#ifndef MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H
#define MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H

#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Tuning/Serializable.h"

namespace mlir {
class Type;
namespace miopen {
struct ConvolutionContext;
// 0 : gemmG dimension.
// 1 : gemmK dimension.
// 2 : gemmM or gemmN dimension.
enum GemmDimensions { GemmG = 0, GemmK = 1, GemmMorN = 2 };

constexpr int64_t gemmCDimG = 0;
constexpr int64_t gemmCDimM = 1;
constexpr int64_t gemmCDimN = 2;
constexpr int64_t gemmCSplitDimM2 = 3;
constexpr int64_t gemmCSplitDimN = 4;
constexpr int64_t gemmCSplitDimN2 = 5;

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

struct DerivedParams {
  int64_t srcVectorReadDim;
  int64_t srcDataPerRead;
  int64_t dstDataPerWrite;
  int64_t clusterLenGemmPos0; // G
  int64_t clusterLenGemmPos1; // K
  int64_t clusterLenGemmPos2; // M or N
  DerivedParams()
      : srcVectorReadDim(GemmG), srcDataPerRead(1), dstDataPerWrite(1),
        clusterLenGemmPos1(0), clusterLenGemmPos2(0) {}
};

struct DerivedOutParams {
  int64_t gemmVectorDim;
  int64_t destVectorDim;
  int64_t dataPerCopy;
  DerivedOutParams() : gemmVectorDim(-1), destVectorDim(-1), dataPerCopy(1) {}
};

struct InitParamsNonXDL : InitParams, Serializable<InitParamsNonXDL> {
  constexpr InitParamsNonXDL(int64_t bSize, int64_t mPerBlock,
                             int64_t nPerBlock, int64_t kPerBlock,
                             int64_t mPerThread, int64_t nPerThread)
      : InitParams{mPerBlock, nPerBlock, kPerBlock}, gemmMPerThread(mPerThread),
        gemmNPerThread(nPerThread), blockSize(bSize) {}
  int64_t gemmMPerThread;
  int64_t gemmNPerThread;
  int64_t blockSize;

  constexpr InitParamsNonXDL()
      : InitParamsNonXDL(0LL, 0LL, 0LL, 0LL, 0LL, 0LL) {}

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

// block gemm tuning params that sepcific the layout of thread-wise gemm in a
// workgroup
struct DerivedBlockGemmParams {
  int64_t gemmMLevel0Cluster;
  int64_t gemmNLevel0Cluster;
  int64_t gemmMLevel1Cluster;
  int64_t gemmNLevel1Cluster;
};

class PopulateParams {
private:
  static constexpr size_t nInitParameters = 21;
  static const InitParamsNonXDL initParameters[nInitParameters];
  // if can't select config from above , use this config to do
  // padding kernel for example , GemmK/block is 16 , if your gemmK is  13 , we
  // add more 3 gemmk
  static const InitParams universalParameters;

  LogicalResult
  calculateGemmABlockCopyPerformanceParameters(const InitParamsNonXDL &param,
                                               ConvolutionContext &ctx,
                                               DerivedParams &derived);

  LogicalResult
  calculateGemmBBlockCopyPerformanceParameters(const InitParamsNonXDL &param,
                                               ConvolutionContext &ctx,
                                               DerivedParams &derived);
  LogicalResult
  calculateGemmCBlockwiseCopyParams(const InitParamsNonXDL &params,
                                    ConvolutionContext &ctx,
                                    DerivedOutParams &out);
  LogicalResult
  calculateBlockGemmPerformanceParameters(const InitParamsNonXDL &param,
                                          const ConvolutionContext &ctx,
                                          DerivedBlockGemmParams &derived);

  LogicalResult
  populateDerived(ConvolutionContext &ctx, const InitParamsNonXDL &validParams,
                  GemmSize &gemmSize, DerivedParams &gemmADerivedParam,
                  DerivedParams &gemmBDerivedParam,
                  DerivedBlockGemmParams &blockGemmDerivedParam,
                  DerivedOutParams &gemmCDerivedParam, int64_t &gridSize);

  LogicalResult populatePaddingKernelDerived(
      ConvolutionContext &ctx, const InitParamsNonXDL &validParams,
      GemmSize &gemmSize, DerivedParams &gemmADerivedParam,
      DerivedParams &gemmBDerivedParam,
      DerivedBlockGemmParams &blockGemmDerivedParam,
      DerivedOutParams &gemmCDerivedParam, int64_t &gridSize);

public:
  LogicalResult obtainTuningParameters(
      Operation *op, int64_t blockSizeOverride, const std::string &perfConfig,
      InitParamsNonXDL &validParams, DerivedParams &gemmADerivedParam,
      DerivedParams &gemmBDerivedParam,
      DerivedBlockGemmParams &blockGemmDerivedParam,
      DerivedOutParams &gemmCDerivedParam, int64_t &gridSize);

  ArrayRef<InitParamsNonXDL> getTuningParameters(ConvOpType dir,
                                                 Type dataType) const;

  const InitParams &getUniversalParameters() const;

  LogicalResult isValidGemm(const InitParamsNonXDL &param, GemmSize &gemmSize);
};

class PopulateParamsXDL {
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

  int64_t obtainBlockSize(const InitParamsXDL &params, int64_t waveSize);

  LogicalResult getKBlocks(ConvolutionContext &ctx, const InitParamsXDL &params,
                           int64_t &gemmKBlocks);

  LogicalResult
  calculateGemmABlockCopyPerformanceParameters(const InitParamsXDL &param,
                                               ConvolutionContext &ctx,
                                               DerivedParams &derived);
  LogicalResult
  calculateGemmBBlockCopyPerformanceParameters(const InitParamsXDL &param,
                                               ConvolutionContext &ctx,
                                               DerivedParams &derived);

  LogicalResult calculateLdsNumberOfByte(const InitParamsXDL &param,
                                         const ConvolutionContext &ctx,
                                         DerivedParams gemmADerived,
                                         DerivedParams gemmBDerived,
                                         size_t &ldsSize);

  LogicalResult isValidBlockwiseGemmXDLOPS(const InitParamsXDL &param,
                                           ConvolutionContext &ctx,
                                           int64_t blockSize);

  LogicalResult
  populateDerived(ConvolutionContext &ctx, const InitParamsXDL &validParams,
                  GemmSize &gemmSize, DerivedParams &gemmADerivedParam,
                  DerivedParams &gemmBDerivedParam,
                  DerivedOutParams &gemmCDerivedParam, int64_t &blockSize,
                  int64_t &gridSize, int64_t &gemmKBlocks);

  LogicalResult populatePaddingKernelDerived(
      ConvolutionContext &ctx, const InitParamsXDL &validParams,
      GemmSize &gemmSize, DerivedParams &gemmADerivedParam,
      DerivedParams &gemmBDerivedParam, DerivedOutParams &gemmCDerivedParam,
      int64_t &blockSize, int64_t &gridSize);

  LogicalResult isValidGridGemmXdlops(GemmSize &gemmSize);

public:
  LogicalResult obtainTuningParameters(
      Operation *op, int64_t blockSizeOverride, const std::string &perfConfig,
      InitParamsXDL &validParams, DerivedParams &gemmADerivedParam,
      DerivedParams &gemmBDerivedParam, DerivedOutParams &gemmCDerivedParam,
      int64_t &blockSize, int64_t &gridSize, int64_t &gemmKBlocks);

  llvm::ArrayRef<InitParamsXDL> getTuningParameters(ConvOpType dir,
                                                    Type dataType) const;
  const InitParams &getUniversalParameters() const;

  LogicalResult isValidGemm(const InitParamsXDL &param,
                            GemmSize &gemmSize) const;
};

// The function is used to compute extra padding sizes.
// For example, if gemmM size is 3 and gemmMPerBlock is 64,
// we set gemmMExtra be 64 so (gemmM+gemmMExtra)%gemmMPerBlock=0.
//
// Returns:
// - isOriginalKernelSupport : a bool only used in backward data convolution.
// - needExtraPad : a bool to indicate whether padding kernel is needed.
// - gemmM/N/KExtra : additional padding required along Gemm M/N/K dimension.
//                    They would be all be 0 in case needExtraPad is false.
template <typename T>
std::tuple<bool, bool, int64_t, int64_t, int64_t>
calculatePaddingKernelSize(int64_t gemmMSize, int64_t gemmNSize,
                           int64_t gemmKSize, ConvOpType dir, Type dataType,
                           T populateParams) {
  bool isOriginalKernelSupport = true;
  bool needExtraPad = false;
  int64_t gemmMExtra, gemmNExtra, gemmKExtra;
  gemmMExtra = gemmNExtra = gemmKExtra = 0;

  auto configParams = populateParams.getTuningParameters(dir, dataType);
  size_t numOfFailedConfigs = 0;
  for (auto &params : configParams) {
    if (gemmMSize % params.gemmMPerBlock == 0 &&
        gemmKSize % params.gemmKPerBlock == 0 &&
        gemmNSize % params.gemmNPerBlock == 0) {
      isOriginalKernelSupport = true;
      break;
    }
    isOriginalKernelSupport = false;
    numOfFailedConfigs++;
  }

  auto extraParams = populateParams.getUniversalParameters();
  if (numOfFailedConfigs == configParams.size()) {
    needExtraPad = true;
    int64_t gemmMRemain, gemmKRemain, gemmNRemain;

    gemmMRemain = gemmMSize % extraParams.gemmMPerBlock;
    if (gemmMRemain != 0)
      gemmMExtra = extraParams.gemmMPerBlock - gemmMRemain;

    gemmNRemain = gemmNSize % extraParams.gemmNPerBlock;
    if (gemmNRemain != 0)
      gemmNExtra = extraParams.gemmNPerBlock - gemmNRemain;

    gemmKRemain = gemmKSize % extraParams.gemmKPerBlock;
    if (gemmKRemain != 0)
      gemmKExtra = extraParams.gemmKPerBlock - gemmKRemain;

    // llvm::errs() << "gemmMExtra: " << gemmMExtra << "gemmNExtra: " <<
    // gemmNExtra << "gemmKExtra: " << gemmKExtra << "\n";
  }

  return std::make_tuple(isOriginalKernelSupport, needExtraPad, gemmMExtra,
                         gemmNExtra, gemmKExtra);
}

} // namespace miopen
} // namespace mlir
#endif // MLIR_DIALECT_MIOPEN_GRIDWISE_GEMM_PARAMS_H
