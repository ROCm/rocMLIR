//===- RockTuningImpl.cpp - tuning API implementation ----*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file implements the tuning interfaces
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>

namespace mlir {
namespace rock {

// The full space is a brute-force search for attention kernels
template <typename Op>
static void createAttnTuningRangeBF(TuningParamSet *newSpace, Op attnOp,
                                    TuningParamSetKind kind) {
  static const std::vector<std::vector<uint32_t>> validRangeAttnParamsMFMA = {
      /*gemm0MPerBlock=*/{32, 64, 128, 256},
      /*gemm1MPerBlock=*/{32, 64, 128, 256},
      /*gemm0NPerBlock=*/{32, 64, 128, 256},
      /*kPackPerBlock=*/{8, 16, 32, 64},
      /*mPerWave=*/{32, 64, 128, 256},
      /*mnPerXdl=*/{4, 16, 32},
      /*kPack=*/{4, 8, 16}};
  static const std::vector<std::vector<uint32_t>> validRangeAttnParamsWMMA = {
      /*gemm0MPerBlock=*/{32, 64, 128},
      /*gemm1MPerBlock=*/{32, 64, 128},
      /*gemm0NPerBlock=*/{32, 64, 128, 256},
      /*kPackPerBlock=*/{8, 16, 32, 64},
      /*mPerWave=*/{32, 64},
      /*nPerWave=*/{32, 64},
      /*kPack=*/{4, 8, 16}};
  GemmFeatures features = attnOp.getGemmFeatures();
  int64_t numEUPerCU = rock::lookupArchInfo(attnOp.getArch()).numEUPerCU;
  std::vector<std::vector<uint32_t>> validRangeAttnParams;
  bool isWMMA = false;
  if (bitEnumContainsAny(features, GemmFeatures::mfma)) {
    validRangeAttnParams = validRangeAttnParamsMFMA;
  } else if (bitEnumContainsAny(features, GemmFeatures::wmma)) {
    isWMMA = true;
    validRangeAttnParams = validRangeAttnParamsWMMA;
  } else {
    // We only support GPUs with matrix accelerator extentions
    return;
  }
  OpBuilder b(attnOp.getContext());
  for (uint32_t gemm0MPerBlock : validRangeAttnParams[0]) {
    for (uint32_t gemm1MPerBlock : validRangeAttnParams[1]) {
      for (uint32_t gemm0NPerBlock : validRangeAttnParams[2]) {
        for (uint32_t gemmKPerBlock : validRangeAttnParams[3]) {
          for (uint32_t gemmMPerWave : validRangeAttnParams[4]) {
            for (uint32_t gemmMnPerXdlOrNPerWave : validRangeAttnParams[5]) {
              for (uint32_t gemmKPack : validRangeAttnParams[6]) {
                if (isWMMA) {
                  int64_t nPerWave = gemmMnPerXdlOrNPerWave;
                  int64_t rdnaWaves = (gemm0MPerBlock / gemmMPerWave) *
                                      (gemm0NPerBlock / nPerWave);
                  if (rdnaWaves < numEUPerCU) {
                    continue;
                  }
                }
                if (gemm0MPerBlock >= gemmMPerWave &&
                    gemm1MPerBlock >= gemmMPerWave &&
                    gemm1MPerBlock >= gemm0MPerBlock &&
                    gemm0NPerBlock >= gemmMnPerXdlOrNPerWave) {
                  auto params = AttnPerfConfigAttr::get(
                      attnOp.getContext(), gemm0MPerBlock, gemm1MPerBlock,
                      gemm0NPerBlock, gemmKPerBlock, gemmMPerWave,
                      gemmMnPerXdlOrNPerWave, gemmKPack, true);
                  newSpace->tuningRange.push_back(
                      cast<RockTuningParamAttrInterface>(params));
                }
              }
            }
          }
        }
      }
    }
  }
}

static double computeWorkImbalance(GemmSize origGemmSize, int32_t gemmMPerBlock,
                                   int32_t gemmNPerBlock, int32_t gemmKPerBlock,
                                   int32_t kPack, uint32_t numCUs,
                                   int32_t splitKFactor = 1) {
  const InitParams params{gemmMPerBlock, gemmNPerBlock, gemmKPerBlock};
  const GemmSize gemmSize =
      calculatePaddedGemmSize(params, origGemmSize, kPack);
  const auto numMTiles = (gemmSize.m + gemmMPerBlock - 1) / gemmMPerBlock;
  const auto numNTiles = (gemmSize.n + gemmNPerBlock - 1) / gemmNPerBlock;

  const double totalNumWorkGroups =
      gemmSize.g * numMTiles * numNTiles * splitKFactor;
  const double maxWorkGroupsPerCU = std::ceil(totalNumWorkGroups / numCUs);
  // imbalances = max. CU work / average work per CU
  return (maxWorkGroupsPerCU * numCUs) / totalNumWorkGroups;
}

static SmallVector<int64_t>
computeOptimalSplitKFactors(GemmSize origGemmSize, int32_t gemmMPerBlock,
                            int32_t gemmNPerBlock, int32_t gemmKPerBlock,
                            int32_t kPack, uint32_t numCUs) {
  SmallVector<int64_t> splitKValues = {1};

  const auto dataParallelGemmImbalance = computeWorkImbalance(
      origGemmSize, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock, kPack, numCUs);

  constexpr double imbalaceThreshold = 1.20;
  if (dataParallelGemmImbalance < imbalaceThreshold) {
    return splitKValues;
  }

  struct LocalData {
    int64_t splitKValue = 0;
    double workImbalance = 0.0;
  };
  SmallVector<LocalData> factors;
  constexpr double minGain = 1.30;
  // A large set of splitK values significantly increases tuning time,
  // after analysis, we've determined that using only splitK factors 3 and 4 is
  // sufficient.
  for (int64_t splitKFactor : {3, 4}) {
    const double imbalance =
        computeWorkImbalance(origGemmSize, gemmMPerBlock, gemmNPerBlock,
                             gemmKPerBlock, kPack, numCUs, splitKFactor);
    const auto gain = dataParallelGemmImbalance / imbalance;
    if (gain > minGain) {
      factors.emplace_back(LocalData{splitKFactor, imbalance});
    }
  }

  if (factors.empty()) {
    return splitKValues;
  }

  llvm::sort(factors.rbegin(), factors.rend(), [](LocalData &a, LocalData &b) {
    return a.workImbalance < b.workImbalance;
  });

  llvm::ArrayRef<LocalData> view(factors.data(), factors.size());
  llvm::for_each(view, [&](const LocalData &item) {
    splitKValues.push_back(item.splitKValue);
  });

  return splitKValues;
}

static SmallVector<int64_t>
computeOptimalSplitKFactors(RockGemmWrapperInterface gemmOp,
                            int32_t gemmMPerBlock, int32_t gemmNPerBlock,
                            int32_t gemmKPerBlock, int32_t kPack,
                            bool isSplitKFusible) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  SmallVector<int64_t> splitKValues = {1};

  if (!isSplitKFusible) {
    return splitKValues;
  }

  auto func = cast<func::FuncOp>(gemmOp->getParentOp());
  if (!func->hasAttr(rock::EnableSplitKForTuningAttr::getMnemonic())) {
    return splitKValues;
  }

  uint32_t numCUs = rock::lookupArchInfo(gemmOp.getArch()).minNumCU;
  if (gemmOp.getNumCU().has_value()) {
    numCUs = gemmOp.getNumCU().value();
  }

  return computeOptimalSplitKFactors(info.gemmSize, gemmMPerBlock,
                                     gemmNPerBlock, gemmKPerBlock, kPack,
                                     numCUs);
}

// The full space is a brute-force search starting with the configs that have
// the smallest parameters. This filters out perf configs that are
// known to be impossible during tthe AffixTuningParams check.
// If `kind` is Full, also filters out unlikely-to-be-good configurations.
static void createGemmTuningRangeBF(TuningParamSet *newSpace,
                                    RockGemmWrapperInterface gemmOp,
                                    bool isSplitKFusible,
                                    TuningParamSetKind kind) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);

  // blockSize M/block N/block K/block M/thread N/thread
  const std::vector<std::vector<uint32_t>> validRangeGeneralGemmParams = {
      {64, 128, 256}, {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};

  // only enable tuning over gemm schedules when doing exhaustive tuning
  auto getGemmSchedules = [](const TuningParamSetKind &tuningKind) {
    if (tuningKind == TuningParamSetKind::Exhaustive) {
      return std::vector<uint32_t>{1, 2};
    }
    return std::vector<uint32_t>{1};
  };

  // M/block N/block K/block M/wave N/wave kPack scheduleVersion
  // aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> validRangeAccelGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {4, 8, 16, 32, 64, 128},
      {4, 16, 32},
      {1, 4, 8},
      getGemmSchedules(kind),
      {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack scheduleVersion
  // aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>>
      validRangeAccelGemmParams8BitReduction = {{4, 8, 16, 32, 64, 128, 256},
                                                {16, 32, 64, 128, 256},
                                                {4, 8, 16, 32},
                                                {4, 8, 16, 32, 64, 128},
                                                {4, 8, 16, 32, 64, 128},
                                                {1, 4, 8, 16},
                                                getGemmSchedules(kind),
                                                {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack scheduleVersion
  // aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> validRangeWmmaGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16},
      getGemmSchedules(kind),
      {0, 1}};

  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    PopulateParamsXDL tuningInfo;
    // XDLOPS
    Type inTypeA = gemmOp.getAType();
    bool is8BitReduction =
        inTypeA.isInteger(8) ||
        (inTypeA.getIntOrFloatBitWidth() == 8 && isa<FloatType>(inTypeA));
    const std::vector<std::vector<uint32_t>> &xdlopsParams =
        is8BitReduction ? validRangeAccelGemmParams8BitReduction
                        : validRangeAccelGemmParams;
    for (uint32_t gemmMPerBlock : xdlopsParams[0]) {
      for (uint32_t gemmNPerBlock : xdlopsParams[1]) {
        for (uint32_t gemmKPerBlock : xdlopsParams[2]) {
          for (uint32_t gemmMPerWave : xdlopsParams[3]) {
            for (uint32_t gemmMnPerXdl : xdlopsParams[4]) {
              for (uint32_t gemmKPack : xdlopsParams[5]) {
                auto optimalSplitKFactors = computeOptimalSplitKFactors(
                    gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                    gemmKPack, isSplitKFusible);
                for (int64_t splitKFactor : optimalSplitKFactors) {
                  for (int64_t gemmSchedule : xdlopsParams[6]) {
                    for (uint32_t forceUnroll : xdlopsParams[7]) {
                      // hardcode outputSwizzle to heuristics = 2
                      InitParamsAccel gemmParams(
                          gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                          gemmMPerWave, gemmMnPerXdl, gemmKPack, splitKFactor,
                          gemmSchedule, 2, forceUnroll, true);
                      if (gemmMPerBlock >= gemmMPerWave &&
                          gemmNPerBlock >= gemmMnPerXdl) {
                        if (succeeded(tuningInfo.paramsProbablyValid(
                                b, info, gemmParams)) &&
                            (kind == TuningParamSetKind::Exhaustive ||
                             succeeded(tuningInfo.couldBePerformant(
                                 info, gemmParams))))
                          newSpace->tuningRange.push_back(
                              cast<RockTuningParamAttrInterface>(
                                  tuningInfo.getGemmParamsAttr(b, gemmParams)));
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    // Wmma
    const std::vector<std::vector<uint32_t>> &wmmaParams =
        validRangeWmmaGemmParams;
    PopulateParamsWmma tuningInfo;
    for (uint32_t gemmMPerBlock : wmmaParams[0]) {
      for (uint32_t gemmNPerBlock : wmmaParams[1]) {
        for (uint32_t gemmKPerBlock : wmmaParams[2]) {
          for (uint32_t gemmMPerWave : wmmaParams[3]) {
            for (uint32_t gemmNPerWave : wmmaParams[4]) {
              for (uint32_t gemmKPack : wmmaParams[5]) {
                auto optimalSplitKFactors = computeOptimalSplitKFactors(
                    gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                    gemmKPack, isSplitKFusible);
                for (auto splitKFactor : optimalSplitKFactors) {
                  for (uint32_t gemmSchedule : wmmaParams[6]) {
                    for (uint32_t forceUnroll : wmmaParams[7]) {
                      // hardcode outputSwizzle to heuristics = 2
                      InitParamsAccel gemmParams(
                          gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                          gemmMPerWave, gemmNPerWave, gemmKPack, splitKFactor,
                          gemmSchedule, 2, forceUnroll, true);
                      if (succeeded(tuningInfo.paramsProbablyValid(
                              b, info, gemmParams)) &&
                          (kind == TuningParamSetKind::Exhaustive ||
                           succeeded(
                               tuningInfo.couldBePerformant(info, gemmParams))))
                        newSpace->tuningRange.push_back(
                            cast<RockTuningParamAttrInterface>(
                                tuningInfo.getGemmParamsAttr(b, gemmParams)));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    // Non-XDLOPS
    PopulateParams tuningInfo;
    for (uint32_t blockSize : validRangeGeneralGemmParams[0]) {
      for (uint32_t gemmMPerBlock : validRangeGeneralGemmParams[1]) {
        for (uint32_t gemmNPerBlock : validRangeGeneralGemmParams[2]) {
          for (uint32_t gemmKPerBlock : validRangeGeneralGemmParams[3]) {
            for (uint32_t gemmMPerThread : validRangeGeneralGemmParams[4]) {
              auto optimalSplitKFactors = computeOptimalSplitKFactors(
                  gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock, 1,
                  isSplitKFusible);
              for (auto splitKFactor : optimalSplitKFactors) {
                for (uint32_t gemmNPerThread : validRangeGeneralGemmParams[5]) {
                  // hardcode schedule version to v1 and outputSwizzle to
                  // heuristics = 2
                  InitParamsNonAccel gemmParams(
                      blockSize, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                      gemmMPerThread, gemmNPerThread, splitKFactor, 1, 2);
                  if (succeeded(tuningInfo.paramsProbablyValid(b, info,
                                                               gemmParams)) &&
                      (kind == TuningParamSetKind::Exhaustive ||
                       succeeded(
                           tuningInfo.couldBePerformant(info, gemmParams))))
                    newSpace->tuningRange.push_back(
                        cast<RockTuningParamAttrInterface>(
                            tuningInfo.getGemmParamsAttr(b, gemmParams)));
                }
              }
            }
          }
        }
      }
    }
  }
}

static void createQuickTuningRange(TuningParamSet *newSpace,
                                   RockGemmWrapperInterface gemmOp) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    PopulateParamsXDL tuningInfo;

    for (InitParamsAccel param : tuningInfo.orderInitParams(
             tuningInfo.getTuningParameters(info.kernelType, info.gemmAType,
                                            info.gemmBType, info.arch),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
            tuningInfo.getGemmParamsAttr(b, param)));
    }
  } else if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    // Wmma
    PopulateParamsWmma tuningInfo;
    for (InitParamsAccel param : tuningInfo.orderInitParams(
             tuningInfo.getTuningParameters(info.kernelType, info.gemmAType,
                                            info.gemmBType, info.arch),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
            tuningInfo.getGemmParamsAttr(b, param)));
    }
  } else {
    // Non-XDLOPS
    PopulateParams tuningInfo;
    for (InitParamsNonAccel param : tuningInfo.orderInitParams(
             tuningInfo.getTuningParameters(info.kernelType, info.gemmAType,
                                            info.gemmBType),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
            tuningInfo.getGemmParamsAttr(b, param)));
    }
  }
}

// This is temporary workaround to make MIGraphX integration
// work until the tuning is setup for attention ops properly.
template <typename Op>
static void createAttnTuningRangeQuick(TuningParamSet *newSpace, Op attnOp,
                                       Type elemType) {
  OpBuilder b(attnOp.getContext());
  GemmFeatures currentFeatures = attnOp.getGemmFeatures();
  // g0Mpb, g1Mpb, g0Npb, Kpb, mPw, mnPxdl, kpack
  using PerfConfigVals =
      std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    const SmallVector<PerfConfigVals, 7> attnQuickTuningListMFMAF16{
        PerfConfigVals{32, 128, 128, 32, 32, 32, 4},
        PerfConfigVals{64, 64, 32, 16, 32, 16, 4},
        PerfConfigVals{32, 64, 64, 16, 32, 16, 4},
        PerfConfigVals{32, 64, 128, 16, 32, 16, 4},
        PerfConfigVals{64, 64, 64, 16, 32, 16, 4},
        PerfConfigVals{64, 64, 64, 16, 32, 32, 4}};
    const SmallVector<PerfConfigVals, 7> attnQuickTuningListMFMAF32{
        PerfConfigVals{32, 128, 64, 32, 32, 16, 4},
        PerfConfigVals{32, 64, 64, 32, 32, 16, 4},
        PerfConfigVals{32, 128, 128, 32, 32, 32, 4},
        PerfConfigVals{64, 64, 32, 16, 32, 16, 4},
        PerfConfigVals{32, 64, 64, 16, 32, 16, 4},
        PerfConfigVals{32, 64, 128, 16, 32, 32, 4},
        PerfConfigVals{64, 64, 64, 16, 32, 32, 4}};
    ArrayRef<PerfConfigVals> attnQuickTuningListMFMA =
        attnQuickTuningListMFMAF32;
    if (elemType.isF16()) {
      attnQuickTuningListMFMA = attnQuickTuningListMFMAF16;
    }
    for (auto [mPerBlockG0, mPerBlockG1, nPerBlockG0, kPackBerBlock, mPerWave,
               mnPerXdl, kPack] : attnQuickTuningListMFMA) {
      auto params = AttnPerfConfigAttr::get(
          attnOp.getContext(), mPerBlockG0, mPerBlockG1, nPerBlockG0,
          kPackBerBlock, mPerWave, mnPerXdl, kPack, true);
      newSpace->tuningRange.push_back(
          cast<RockTuningParamAttrInterface>(params));
    }
  } else if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    const SmallVector<PerfConfigVals, 7> attnQuickTuningListWMMA{
        PerfConfigVals{64, 128, 128, 8, 32, 32, 4},
        PerfConfigVals{64, 64, 256, 8, 64, 32, 8},
        PerfConfigVals{64, 64, 256, 16, 32, 32, 8},
        PerfConfigVals{64, 64, 32, 8, 32, 32, 4},
        PerfConfigVals{32, 64, 128, 8, 32, 32, 8},
        PerfConfigVals{64, 64, 128, 8, 64, 32, 8},
        PerfConfigVals{32, 32, 128, 8, 32, 32, 8},
        PerfConfigVals{128, 128, 128, 8, 32, 32, 8}};
    for (auto [mPerBlockG0, mPerBlockG1, nPerBlockG0, kPackBerBlock, mPerWave,
               mnPerXdl, kPack] : attnQuickTuningListWMMA) {
      auto params = AttnPerfConfigAttr::get(
          attnOp.getContext(), mPerBlockG0, mPerBlockG1, nPerBlockG0,
          kPackBerBlock, mPerWave, mnPerXdl, kPack, true);
      newSpace->tuningRange.push_back(
          cast<RockTuningParamAttrInterface>(params));
    }
  }
  // We only support GPUs with matrix accelerator extentions
}

TuningParamSet *createTunableParamSpace(ModuleOp mod, TuningParamSetKind kind) {
  struct TuningParamSet *newSpace;
  newSpace = new TuningParamSet();

  bool isSplitKFusible = succeeded(rock::testFusionLegalitySplitK(mod));

  // create range and heuristic
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createGemmTuningRangeBF(newSpace, op, isSplitKFusible, kind);
          break;
        case TuningParamSetKind::Quick:
          createQuickTuningRange(newSpace, op);
          break;
        }
        newSpace->primaryOpType = op.getKernelType();
        return WalkResult::interrupt();
      });
  WalkResult findGemmGemm =
      mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        Type elemType = cast<ShapedType>(op.getAType()).getElementType();
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createAttnTuningRangeBF(newSpace, op, kind);
          break;
        case TuningParamSetKind::Quick:
          createAttnTuningRangeQuick(newSpace, op, elemType);
        }
        return WalkResult::interrupt();
      });
  if (!findPrimary.wasInterrupted() && !findGemmGemm.wasInterrupted()) {
    llvm::report_fatal_error("Expected to find GEMM, convolution, attention, "
                             "gemm+gemm or conv+gemm op, and didn't.");
  }
  return newSpace;
}

bool tuningGetParam(TuningParamSet *tuningSpace, unsigned pos,
                    ParamEntry *paramEntry) {
  // out of bound check.
  if (pos > tuningSpace->tuningRange.size() - 1)
    return false;
  paramEntry->param = tuningSpace->tuningRange[pos];
  return true;
}

bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        SmallString<64> perfConfig;
        paramEntry->param.getPerfConfigStr(perfConfig);
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  WalkResult setGemmGemm =
      mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        SmallString<64> perfConfig;
        paramEntry->param.getPerfConfigStr(perfConfig);
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  return setPrimary.wasInterrupted() || setGemmGemm.wasInterrupted();
}

bool tuningSetStr(ModuleOp &mod, StringRef perfConfig) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  WalkResult setGemmGemm =
      mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  return setPrimary.wasInterrupted() || setGemmGemm.wasInterrupted();
}

TuningTable *tuningTableCreate() {
  struct TuningTable *newTable = new TuningTable();
  return newTable;
}

static LogicalResult
extractLayouts(Operation *op, llvm::StringMap<unsigned> &fLayoutMap,
               llvm::StringMap<unsigned> &iLayoutMap,
               llvm::StringMap<unsigned> &oLayoutMap, SmallString<6> &fLayout,
               SmallString<6> &iLayout, SmallString<6> &oLayout,
               bool computeOutput = true) {
  // Extract layout information
  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  ArrayAttr outputLayoutAttr;
  if (computeOutput)
    outputLayoutAttr = op->getAttrOfType<ArrayAttr>("output_layout");

  unsigned size = filterLayoutAttr.size();

  for (unsigned i = 0; i < size; ++i) {
    auto filterAttr = cast<StringAttr>(filterLayoutAttr.getValue()[i]);
    StringRef fKey = filterAttr.getValue();
    if (fKey == "y")
      fKey = "0";
    if (fKey == "x")
      fKey = "1";
    fLayoutMap[fKey] = i;
    auto inputAttr = cast<StringAttr>(inputLayoutAttr.getValue()[i]);
    StringRef iKey = inputAttr.getValue();
    if (iKey == "hi")
      iKey = "0i";
    if (iKey == "wi")
      iKey = "1i";
    iLayoutMap[iKey] = i;
    if (computeOutput) {
      auto outputAttr = cast<StringAttr>(outputLayoutAttr.getValue()[i]);
      StringRef oKey = outputAttr.getValue();
      if (oKey == "ho")
        oKey = "0o";
      if (oKey == "wo")
        oKey = "1o";
      oLayoutMap[oKey] = i;
    }
  }

  fLayout.assign(size, '#');
  iLayout.assign(size, '#');
  oLayout.assign(size, '#');

  // dimensions need to be mapped 1 to 1.
  fLayout[fLayoutMap["k"]] = 'N';
  fLayout[fLayoutMap["c"]] = 'C';
  fLayout[fLayoutMap["g"]] = 'G';
  iLayout[iLayoutMap["ni"]] = 'N';
  iLayout[iLayoutMap["ci"]] = 'C';
  iLayout[iLayoutMap["gi"]] = 'G';
  if (computeOutput) {
    oLayout[oLayoutMap["no"]] = 'N';
    oLayout[oLayoutMap["ko"]] = 'C';
    oLayout[oLayoutMap["go"]] = 'G';
  }

  for (unsigned i = 0; i < size - 3; i++) {
    std::string key = std::to_string(i);
    char val = '0' + i;
    fLayout[fLayoutMap[key]] = val;
    iLayout[iLayoutMap[key + "i"]] = val;
    if (computeOutput)
      oLayout[oLayoutMap[key + "o"]] = val;
  }

  if (computeOutput) {
    if (llvm::any_of(llvm::concat<const char>(fLayout, iLayout, oLayout),
                     [](const char c) { return c == '#'; }))
      return failure();
  } else {
    if (llvm::any_of(llvm::concat<const char>(fLayout, iLayout),
                     [](const char c) { return c == '#'; }))
      return failure();
  }
  return success();
}

static LogicalResult
getTuningProblemStr(RockGemmGemmWrapperInterface gemmGemmOp,
                    SmallVectorImpl<char> &out) {
  int32_t numCU = rock::lookupArchInfo(gemmGemmOp.getArch()).minNumCU;
  if (gemmGemmOp.getNumCU().has_value()) {
    numCU = gemmGemmOp.getNumCU().value();
  }
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  int64_t headDimQK;
  int64_t headDimV;
  int64_t seqLenQ;
  int64_t seqLenK;
  llvm::raw_svector_ostream problemOS(out);
  // ARCH string
  problemOS << gemmGemmOp.getArch() << tab;
  // Num of Compute Units
  problemOS << numCU << tab;

  ArrayRef<int64_t> qShape = cast<MemRefType>(gemmGemmOp.getAType()).getShape();
  ArrayRef<int64_t> kShape = cast<MemRefType>(gemmGemmOp.getBType()).getShape();
  ArrayRef<int64_t> vShape = cast<MemRefType>(gemmGemmOp.getCType()).getShape();

  bool isAttention = isa<AttentionOp>(gemmGemmOp);
  bool isConvGemm = isa<ConvElementwiseGemmOp>(gemmGemmOp);

  Type elemTypeQ = cast<MemRefType>(gemmGemmOp.getAType()).getElementType();
  problemOS << "-t ";
  if (elemTypeQ.isF32()) {
    problemOS << "f32" << sep;
  } else if (elemTypeQ.isF16()) {
    problemOS << "f16" << sep;
  } else if (elemTypeQ.isBF16()) {
    problemOS << "bf16" << sep;
  } else if (elemTypeQ.isInteger(8) && isAttention) {
    problemOS << "i8" << sep;
  } else {
    return gemmGemmOp.emitError("invalid type:") << elemTypeQ << "\n";
  }

  // Extract layout information
  llvm::StringMap<unsigned> fLayoutMap, iLayoutMap, oLayoutMap;
  SmallString<6> fLayout, iLayout, oLayout;

  if (isConvGemm) {
    if (failed(extractLayouts(gemmGemmOp, fLayoutMap, iLayoutMap, oLayoutMap,
                              fLayout, iLayout, oLayout, false)))
      return gemmGemmOp.emitError("layout can't be extracted");

    // filter layout
    problemOS << "-f " << fLayout << sep;
    // input layout
    problemOS << "-I " << iLayout << sep;
  } else {
    // TransQ
    if (isAttention)
      problemOS << "-transQ ";
    else
      problemOS << "-transA ";
    if (gemmGemmOp.getTransposedA()) {
      seqLenQ = qShape[2];
      headDimQK = qShape[1];
      problemOS << "true" << sep;
    } else {
      seqLenQ = qShape[1];
      headDimQK = qShape[2];
      problemOS << "false" << sep;
    }

    // TransK
    if (isAttention)
      problemOS << "-transK ";
    else
      problemOS << "-transB ";
    if (gemmGemmOp.getTransposedB()) {
      seqLenK = kShape[1];
      problemOS << "true" << sep;
    } else {
      seqLenK = kShape[2];
      problemOS << "false" << sep;
    }
  }

  // TransV
  if (isAttention)
    problemOS << "-transV ";
  else
    problemOS << "-transC ";
  if (gemmGemmOp.getTransposedC()) {
    headDimV = vShape[1];
    problemOS << "true" << sep;
  } else {
    headDimV = vShape[2];
    problemOS << "false" << sep;
  }

  // TransO
  problemOS << "-transO ";
  if (gemmGemmOp.getTransposedOut())
    problemOS << "true" << sep;
  else
    problemOS << "false" << sep;

  if (isAttention) {
    auto attentionOp = cast<AttentionOp>(gemmGemmOp);
    problemOS << "-causal ";
    if (attentionOp.getCausal())
      problemOS << "true" << sep;
    else
      problemOS << "false" << sep;

    problemOS << "-return_lse ";
    if (attentionOp.getLse())
      problemOS << "true" << sep;
    else
      problemOS << "false" << sep;
  }

  if (!isConvGemm)
    problemOS << "-g " << qShape[0] << sep;

  if (isAttention) {
    problemOS << "-seq_len_q " << seqLenQ << sep;
    problemOS << "-seq_len_k " << seqLenK << sep;
    problemOS << "-head_dim_qk " << headDimQK << sep;
    problemOS << "-head_dim_v " << headDimV;
  } else if (isConvGemm) {
    auto convGemmOp = cast<ConvElementwiseGemmOp>(gemmGemmOp);
    ArrayRef<int64_t> inShape = convGemmOp.getInput().getType().getShape();
    ArrayRef<int64_t> filShape = convGemmOp.getFilter().getType().getShape();

    // N
    problemOS << "-n " << inShape[iLayoutMap["ni"]] << sep;
    // C
    problemOS << "-c " << inShape[iLayoutMap["ci"]] * inShape[iLayoutMap["gi"]]
              << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["0i"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["1i"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] * filShape[fLayoutMap["g"]]
              << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["0"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["1"]] << sep;

    auto paddingVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getPadding());
    auto strideVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getStrides());
    auto dilationVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getDilations());

    // padding
    problemOS << "-p " << paddingVal[0] << " -q " << paddingVal[2] << sep;
    // stride
    problemOS << "-u " << strideVal[0] << " -v " << strideVal[1] << sep;
    // dilation
    problemOS << "-l " << dilationVal[0] << " -j " << dilationVal[1] << sep;
    // group
    problemOS << "-g " << inShape[iLayoutMap["gi"]] << sep;
    problemOS << "-gemmO " << headDimV;
  } else {
    problemOS << "-m " << seqLenQ << sep;
    problemOS << "-n " << seqLenK << sep;
    problemOS << "-k " << headDimQK << sep;
    problemOS << "-gemmO " << headDimV;
  }
  return success();
}

static LogicalResult getTuningProblemStr(rock::RockGemmWrapperInterface gemmIF,
                                         SmallVectorImpl<char> &out) {
  int32_t numCU = rock::lookupArchInfo(gemmIF.getArch()).minNumCU;
  if (gemmIF.getNumCU().has_value())
    numCU = gemmIF.getNumCU().value();
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  llvm::raw_svector_ostream problemOS(out);

  KernelType opType = gemmIF.getKernelType();
  Operation *gemmOp = gemmIF.getOperation();

  auto f8TypeStr = [](const Type &type) -> std::optional<StringLiteral> {
    if (isa<Float8E4M3FNUZType, Float8E4M3FNType>(type))
      return StringLiteral("fp8");
    if (isa<Float8E5M2FNUZType, Float8E5M2Type>(type))
      return StringLiteral("bf8");
    return std::nullopt;
  };

  // ARCH string
  problemOS << gemmIF.getArch() << tab;
  // Num of Compute Units
  problemOS << numCU << tab;

  if (opType == KernelType::Conv || opType == KernelType::ConvBwdData ||
      opType == KernelType::ConvBwdWeight) { // conv cases
    RockConvInterface convIF = dyn_cast<RockConvInterface>(gemmOp);

    ShapedType inType = convIF.getInput().getType();
    ArrayRef<int64_t> inShape = inType.getShape();
    ShapedType filType = convIF.getFilter().getType();
    ArrayRef<int64_t> filShape = filType.getShape();

    // Extract layout information
    llvm::StringMap<unsigned> fLayoutMap, iLayoutMap, oLayoutMap;
    SmallString<6> fLayout, iLayout, oLayout;
    if (failed(extractLayouts(gemmOp, fLayoutMap, iLayoutMap, oLayoutMap,
                              fLayout, iLayout, oLayout)))
      return convIF.emitError("layout can't be extracted");

    // Please keep these in sync with mlir/utils/performance/perfRunner.py

    // OP datatype
    Type inElemType = inType.getElementType();
    Type filElemType = filType.getElementType();
    if (inElemType.isF32()) {
      problemOS << "conv ";
    } else if (inElemType.isF16()) {
      problemOS << "convfp16 ";
    } else if (inElemType.isBF16()) {
      problemOS << "convbfp16 ";
    } else if (inElemType.isInteger(8)) {
      problemOS << "convint8 ";
    } else {
      auto inString = f8TypeStr(inElemType);
      auto filString = f8TypeStr(filElemType);
      if (inString && filString)
        problemOS << llvm::formatv("conv{0}_{1} ", *inString, *filString);
      else
        return failure();
    }

    // OP direction
    switch (opType) {
    case KernelType::Conv:
      problemOS << "-F 1" << sep;
      break;
    case KernelType::ConvBwdData:
      problemOS << "-F 2" << sep;
      break;
    case KernelType::ConvBwdWeight:
      problemOS << "-F 4" << sep;
      break;
    default:
      return failure();
    }

    // filter layout
    problemOS << "-f " << fLayout << sep;
    // input layout
    problemOS << "-I " << iLayout << sep;
    // output layout
    problemOS << "-O " << oLayout << sep;
    // N
    problemOS << "-n " << inShape[iLayoutMap["ni"]] << sep;
    // C
    problemOS << "-c " << inShape[iLayoutMap["ci"]] * inShape[iLayoutMap["gi"]]
              << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["0i"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["1i"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] * filShape[fLayoutMap["g"]]
              << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["0"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["1"]] << sep;

    auto paddingVal = extractFromIntegerArrayAttr<int64_t>(convIF.getPadding());
    auto strideVal = extractFromIntegerArrayAttr<int64_t>(convIF.getStrides());
    auto dilationVal =
        extractFromIntegerArrayAttr<int64_t>(convIF.getDilations());
    // padding
    problemOS << "-p " << paddingVal[0] << " -q " << paddingVal[2] << sep;
    // stride
    problemOS << "-u " << strideVal[0] << " -v " << strideVal[1] << sep;
    // dilation
    problemOS << "-l " << dilationVal[0] << " -j " << dilationVal[1] << sep;
    // group
    problemOS << "-g " << inShape[iLayoutMap["gi"]] << sep;

  } else if (opType == KernelType::Gemm) { // gemm case
    rock::GemmOp rGemmOp = dyn_cast<rock::GemmOp>(gemmOp);
    // Please keep these in sync with mlir/utils/performance/perfRunner.py
    // Data type
    problemOS << "-t ";
    Type elemTypeA = gemmIF.getAType(), elemTypeB = gemmIF.getBType();
    if (elemTypeA.isF32() && elemTypeB.isF32()) {
      problemOS << "f32";
    } else if (elemTypeA.isF16() && elemTypeB.isF16()) {
      problemOS << "f16";
    } else if (elemTypeA.isBF16() && elemTypeB.isBF16()) {
      problemOS << "bf16";
    } else if (elemTypeA.isInteger(8) && elemTypeB.isInteger(8)) {
      problemOS << "i8";
    } else {
      auto aString = f8TypeStr(elemTypeA);
      auto bString = f8TypeStr(elemTypeB);
      if (aString && bString)
        problemOS << llvm::formatv("{0}_{1}", *aString, *bString);
      else
        return failure();
    }

    // Output datatype
    auto outType = gemmIF.getOutArgument()->get().getType();
    auto elemTypeC = dyn_cast<mlir::MemRefType>(outType).getElementType();
    problemOS << " -out_datatype ";
    auto outStr = f8TypeStr(elemTypeC);
    if (outStr)
      problemOS << *outStr << sep;
    else
      problemOS << elemTypeC << sep;

    // TransA
    problemOS << "-transA ";
    if (rGemmOp.getATransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // TransB
    problemOS << "-transB ";
    if (rGemmOp.getBTransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // Gemmsize G/M/N/K
    problemOS << "-g " << gemmIF.getGemmSize().g << sep;
    problemOS << "-m " << gemmIF.getGemmSize().m << sep;
    problemOS << "-n " << gemmIF.getGemmSize().n << sep;
    problemOS << "-k " << gemmIF.getGemmSize().k << sep;
  } else {
    // Unknown op type, unreachable.
    return failure();
  }

  while (out.back() == sep) {
    // remove trailing whitespace
    out.pop_back();
  }

  return success();
}

// Suppose to return the structure of the given problem to tune, currently
// combines the string representation of the selected field of the primary
// operation. String format of the problem will not be required by the DB,
// since it can store each field separately.
// Currently serialize the problem in MIOpenDriver command friendly format
LogicalResult getTuningProblemStr(ModuleOp mod, SmallVectorImpl<char> &out) {
  {
    rock::RockGemmWrapperInterface gemmIF;
    WalkResult findPrimary =
        mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
          gemmIF = op;
          return WalkResult::interrupt();
        });
    if (findPrimary.wasInterrupted())
      return getTuningProblemStr(gemmIF, out);
  }
  {
    rock::RockGemmGemmWrapperInterface gemmGemmOp;
    WalkResult findGemmGemm =
        mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
          gemmGemmOp = op;
          return WalkResult::interrupt();
        });
    if (findGemmGemm.wasInterrupted())
      return getTuningProblemStr(gemmGemmOp, out);
  }
  return failure();
}

bool tuningTableUpdate(TuningTable *perfTable, StringRef problem,
                       StringRef perfConfig, float time) {
  if (problem.empty())
    return false;
  llvm::sys::SmartScopedWriter<true> guard(perfTable->lock);
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    if (entry.second <= time) {
      return false;
    }
  }
  perfTable->tuningMap[problem] = std::make_pair(perfConfig, time);
  return true;
}

LogicalResult tuningTableLookup(TuningTable *perfTable, ModuleOp &mod,
                                SmallVectorImpl<char> &out) {
  SmallString<2048> problem;
  if (failed(getTuningProblemStr(mod, problem)))
    return failure();
  llvm::sys::SmartScopedReader<true> guard(perfTable->lock);
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    out.assign(entry.first);
    return success();
  }
  return failure();
}

template <typename ParamType>
static int64_t retrieveSplitKValueImpl(StringRef perfConfig) {
  ParamType params;
  params.deserialize(perfConfig.str());
  return params.splitKFactor;
}

static int64_t retrieveSplitKValue(rock::GemmFeatures features,
                                   StringRef perfConfig) {
  if (isAccel(features)) {
    return retrieveSplitKValueImpl<rock::InitParamsAccel>(perfConfig);
  }
  return retrieveSplitKValueImpl<rock::InitParamsNonAccel>(perfConfig);
}

bool isSplitKRequested(rock::GemmFeatures features, StringRef perfConfig) {
  return retrieveSplitKValue(features, perfConfig) > 1;
}

bool isSplitKRequested(ModuleOp mod, StringRef perfConfig) {
  WalkResult gemmWalkResult =
      mod.walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        if (isSplitKRequested(op.getGemmFeatures(), perfConfig))
          return WalkResult::interrupt();

        return WalkResult::advance();
      });

  return gemmWalkResult.wasInterrupted();
}

RocmlirSplitKSelectionLikelihood isSplitKFaster(int64_t gDim, int64_t mDim,
                                                int64_t nDim, int64_t kDim,
                                                int64_t numCUs) {

  // Note, the following values are aggregated from `createGemmTuningRangeBF`,
  // see above.
  // M/block N/block K/block M/wave N/wave kPack
  const std::vector<std::vector<uint32_t>> rangeGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {1, 4, 8, 16}};

  rock::GemmSize gemmSize(gDim, mDim, kDim, nDim);
  llvm::SmallSetVector<int64_t, 10> splitKValues = {};
  double minWorkImbalance = std::numeric_limits<double>::max();
  for (uint32_t mPerBlock : rangeGemmParams[0]) {
    for (uint32_t nPerBlock : rangeGemmParams[1]) {
      for (uint32_t kPerBlock : rangeGemmParams[2]) {
        for (uint32_t kPack : rangeGemmParams[3]) {
          const double currWorkImbalance = computeWorkImbalance(
              gemmSize, mPerBlock, nPerBlock, kPerBlock, kPack, numCUs);
          minWorkImbalance = std::min(currWorkImbalance, minWorkImbalance);

          llvm::SmallVector<int64_t> currSplitKValues =
              computeOptimalSplitKFactors(gemmSize, mPerBlock, nPerBlock,
                                          kPerBlock, kPack, numCUs);
          llvm::for_each(currSplitKValues, [&splitKValues](int64_t value) {
            splitKValues.insert(value);
          });
        }
      }
    }
  }

  if (splitKValues.size() == 1) {
    return RocmlirSplitKSelectionLikelihood::never;
  }

  // TODO[split-K]: one needs to validate whether
  // 1.8 threshold is a resonable choice
  constexpr double workImbalanceThreshold{1.8};
  if (minWorkImbalance > workImbalanceThreshold) {
    return RocmlirSplitKSelectionLikelihood::always;
  }
  return RocmlirSplitKSelectionLikelihood::maybe;
}

bool isModuleFusible(ModuleOp module, StringRef perfConfig) {
  bool fusible = succeeded(rock::testFusionLegalityReduce(module));
  if (!rock::isSplitKRequested(module, perfConfig))
    return fusible;
  return fusible && succeeded(rock::testFusionLegalitySplitK(module));
}

} // namespace rock
} // namespace mlir
