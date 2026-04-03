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

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <random>

// Found experimentally, might need to change it if we add more params to the
// tuning space
#define NUM_RANDOM_PERFCONFIGS_PER_TILE_SIZE 50
#define RND_SEED 42

namespace mlir {
namespace rock {

static std::vector<uint32_t> computeDPerBlock(TuningParamSetKind tuningKind) {
  std::vector<uint32_t> dPerBlockList;

  if (tuningKind == TuningParamSetKind::Greedy) {
    for (uint32_t dPerBlock = 16; dPerBlock <= 256; dPerBlock += 16) {
      dPerBlockList.push_back(dPerBlock);
    }
  } else {
    for (uint32_t dPerBlock = 16; dPerBlock <= 256; dPerBlock *= 2) {
      dPerBlockList.push_back(dPerBlock);
    }
  }
  return dPerBlockList;
}

static SmallVector<uint32_t> compute1MPerBlock(TuningParamSetKind tuningKind,
                                               uint32_t gemm0MPerBlock) {
  SmallVector<uint32_t> mPerBlockList;
  if (tuningKind == TuningParamSetKind::Greedy) {
    for (uint32_t mPerBlock = gemm0MPerBlock; mPerBlock <= 256;
         mPerBlock += 16) {
      if (mPerBlock % gemm0MPerBlock == 0)
        mPerBlockList.push_back(mPerBlock);
    }
  } else {
    for (uint32_t mPerBlock = gemm0MPerBlock; mPerBlock <= 256;
         mPerBlock *= 2) {
      if (mPerBlock % gemm0MPerBlock == 0)
        mPerBlockList.push_back(mPerBlock);
    }
  }
  return mPerBlockList;
}

static SmallVector<uint32_t> computeDPerWave(TuningParamSetKind tuningKind,
                                             uint32_t dPerBlock,
                                             int64_t waveSize) {
  SmallVector<uint32_t> dPerWaveList;
  uint32_t maxDPerWave = (tuningKind != TuningParamSetKind::Full)
                             ? std::numeric_limits<uint32_t>::max()
                             : 128;

  uint32_t maxFactor = 16;
  if (tuningKind != TuningParamSetKind::Full) {
    maxFactor = maxHardwareWorkgroupSize / waveSize;
  }
  for (uint32_t factor = 1; factor <= maxFactor; factor *= 2) {
    if (dPerBlock % factor != 0)
      continue;

    uint32_t dPerWave = dPerBlock / factor;
    // mnPerXdl is 16 or higher (we do not use block != 1 mfmas)
    // and dPerWave >= mnPerXdl
    if (dPerWave >= 16 && dPerWave <= maxDPerWave)
      dPerWaveList.push_back(dPerWave);
  }
  assert(!dPerWaveList.empty() && "dPerWaveList can't be empty");
  return dPerWaveList;
}

static SmallVector<int64_t>
computeOptimalSplitKFactors(RockGemmGemmWrapperInterface gemmGemmOp,
                            int64_t gemm0NPerBlock) {
  SmallVector<int64_t> splitKValues = {1};

  auto func = cast<func::FuncOp>(gemmGemmOp->getParentOp());
  if (!func->hasAttr(rock::EnableSplitKForTuningAttr::getMnemonic())) {
    return splitKValues;
  }

  uint32_t numCUs =
      rock::lookupArchInfo(rock::getArchValue(gemmGemmOp)).minNumCU;
  auto opNumCUs = rock::getNumCU(gemmGemmOp);
  if (succeeded(opNumCUs))
    numCUs = opNumCUs.value();

  SmallVector<int64_t, 3> aShape =
      llvm::to_vector<3>(cast<MemRefType>(gemmGemmOp.getAType()).getShape());
  SmallVector<int64_t, 3> bShape =
      llvm::to_vector<3>(cast<MemRefType>(gemmGemmOp.getBType()).getShape());
  SmallVector<int64_t, 3> cShape =
      llvm::to_vector<3>(cast<MemRefType>(gemmGemmOp.getCType()).getShape());

  GemmSize gemm0Size(/*g=*/aShape[0], /*m=*/bShape[2],
                     /*k=*/aShape[1],
                     /*n=*/aShape[2]);
  int64_t gridSize = ((gemm0Size.n) / gemm0NPerBlock) * gemm0Size.g;

  // Simple heuristic, if gridSize >= numCUs, don't use splitK
  // TODO: improve this heuristic
  if (gridSize >= numCUs)
    return splitKValues;

  // Try splitK factors 3, 4 tend to help even when M is small
  // TODO: improve this heuristic
  return {1, 3, 4};
}

// only enable tuning over gemm schedules when doing exhaustive/greedy tuning
static std::vector<uint32_t>
getSchedules(Operation *op, const TuningParamSetKind &tuningKind) {
  auto features = rock::lookupArchInfo(rock::getArchValue(op)).defaultFeatures;
  bool directToLDS = isDirectToLDSSupported(features);

  std::vector<GemmLoadTileType> loadTypes{GemmLoadTileType::Default};
  if (tuningKind != TuningParamSetKind::Full) {
    loadTypes.push_back(GemmLoadTileType::DoubleBuffer);
    if (directToLDS) {
      loadTypes.push_back(GemmLoadTileType::DirectToLDSDefault);
      loadTypes.push_back(GemmLoadTileType::DirectToLDSDoubleBuffer);
    }
  }
  std::vector<uint32_t> schedules;
  schedules.reserve(loadTypes.size());

  for (auto loadType : loadTypes)
    schedules.push_back(static_cast<uint32_t>(loadType));

  return schedules;
}

static std::vector<std::vector<int64_t>>
getFinetuningParams(int64_t maxWavesPerEU) {
  std::vector<int64_t> wavesPerEUList;
  wavesPerEUList.push_back(0); // use heuristic
  for (int64_t wavesPerEU = 1; wavesPerEU <= maxWavesPerEU; wavesPerEU *= 2) {
    wavesPerEUList.push_back(wavesPerEU);
  }
  std::vector<std::vector<int64_t>> finetuningParams = {
      {0, 1},                 // outputSwizzle
      wavesPerEUList,         // wavesPerEU
      {0, 4, 8, 16, 32, 64}}; // gridGroupSize
  return finetuningParams;
}

static std::vector<std::vector<uint32_t>>
getAccelRangeGemm(RockGemmWrapperInterface gemmOp, TuningParamSetKind kind) {
  auto dPerBlock = computeDPerBlock(kind);
  std::vector<std::vector<uint32_t>> validRangeAccelGemmParams = {
      dPerBlock,                  // M/block
      dPerBlock,                  // N/block
      {1, 2, 4, 8},               // K/block
      {16, 32},                   // MnPerXdl
      {1, 4, 8, 16, 32},          // kPack
      getSchedules(gemmOp, kind), // scheduleVersion
      {0, 1}};                    // forceUnroll

  std::vector<std::vector<uint32_t>> validRangeAccelGemmParams8BitReduction = {
      dPerBlock,                  // M/block
      dPerBlock,                  // N/block
      {4, 8, 16, 32},             // K/block
      {16, 32},                   // MnPerXdl
      {1, 4, 8, 16},              // kPack
      getSchedules(gemmOp, kind), // scheduleVersion
      {0, 1}};                    // forceUnroll

  Type inTypeA = gemmOp.getAType();
  bool is8BitReduction =
      inTypeA.isInteger(8) ||
      (inTypeA.getIntOrFloatBitWidth() == 8 && isa<FloatType>(inTypeA));
  std::vector<std::vector<uint32_t>> &xdlopsParams =
      is8BitReduction ? validRangeAccelGemmParams8BitReduction
                      : validRangeAccelGemmParams;

  std::vector<std::vector<uint32_t>> validRangeWmmaGemmParams = {
      dPerBlock,                  // M/block
      dPerBlock,                  // N/block
      {1, 2, 4, 8},               // K/block
      {16},                       // MnPerXdl
      {4, 8, 16},                 // kPack
      getSchedules(gemmOp, kind), // scheduleVersion
      {0, 1}};                    // forceUnroll

  StringAttr arch = rock::getArchValue(gemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (archInfo.isMfma(gemmOp))
    return xdlopsParams;

  return validRangeWmmaGemmParams;
}

static std::vector<std::vector<uint32_t>>
getAccelRangeGemmGemm(RockGemmGemmWrapperInterface gemmGemmOp,
                      TuningParamSetKind kind) {
  auto dPerBlock = computeDPerBlock(kind);
  static const std::vector<std::vector<uint32_t>> validRangeGemmGemmParamsMFMA =
      {/*gemm0MPerBlock=*/dPerBlock,
       /*gemm0NPerBlock=*/dPerBlock,
       /*kPackPerBlock=*/{2, 4, 8, 16, 32, 64},
       /*mnPerXdl=*/{4, 16, 32},
       /*kPack=*/{4, 8, 16},
       getSchedules(gemmGemmOp, kind)};
  static const std::vector<std::vector<uint32_t>> validRangeGemmGemmParamsWMMA =
      {/*gemm0MPerBlock=*/dPerBlock,
       /*gemm0NPerBlock=*/dPerBlock,
       /*kPackPerBlock=*/{2, 4, 8, 16, 32, 64},
       /*mnPerXdl=*/{16},
       /*kPack=*/{4, 8, 16},
       getSchedules(gemmGemmOp, kind)};
  StringAttr arch = rock::getArchValue(gemmGemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);

  std::vector<std::vector<uint32_t>> validRangeGemmGemmParams;
  if (archInfo.isMfma(gemmGemmOp)) {
    validRangeGemmGemmParams = validRangeGemmGemmParamsMFMA;
  } else if (archInfo.isWmma(gemmGemmOp)) {
    validRangeGemmGemmParams = validRangeGemmGemmParamsWMMA;
  }
  return validRangeGemmGemmParams;
}

// Generate random configs for greedy first iteration (Phase 1)
static void createGemmGemmTuningRangeGreedyPhase1(
    TuningParamSet *newSpace, RockGemmGemmWrapperInterface gemmGemmOp,
    unsigned numRandomPerTileSize, unsigned int seed) {
  StringAttr arch = rock::getArchValue(gemmGemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (!archInfo.isAccel(gemmGemmOp)) {
    // We only support GPUs with matrix accelerator extensions
    return;
  }
  OpBuilder b(gemmGemmOp.getContext());
  const std::vector<std::vector<uint32_t>> params =
      getAccelRangeGemmGemm(gemmGemmOp, TuningParamSetKind::Greedy);
  std::mt19937 rng(seed);
  int64_t waveSize =
      rock::lookupArchInfo(rock::getArchValue(gemmGemmOp)).waveSize;
  bool isWmma = archInfo.isWmma(gemmGemmOp);
  int64_t numEUPerCU = archInfo.numEUPerCU;

  int64_t outputSwizzle{2}, wavesPerEU{0};
  for (uint32_t gemm0MPerBlock : params[0]) {
    SmallVector<uint32_t> mPerWaveRange =
        computeDPerWave(TuningParamSetKind::Greedy, gemm0MPerBlock, waveSize);
    SmallVector<uint32_t> mPerBlockGemm1 =
        compute1MPerBlock(TuningParamSetKind::Greedy, gemm0MPerBlock);
    for (uint32_t gemm1MPerBlock : mPerBlockGemm1) {
      for (uint32_t gemm0NPerBlock : params[1]) {
        SmallVector<uint32_t> nPerWaveRange = computeDPerWave(
            TuningParamSetKind::Greedy, gemm0NPerBlock, waveSize);
        auto optimalSplitKFactors =
            computeOptimalSplitKFactors(gemmGemmOp, gemm0NPerBlock);

        uint32_t totalIterations = params[2].size() * mPerWaveRange.size() *
                                   nPerWaveRange.size() * params[3].size() *
                                   params[4].size() * params[5].size() *
                                   optimalSplitKFactors.size();
        uint32_t numRandomIterations =
            std::min(numRandomPerTileSize, totalIterations);
        for (uint32_t randomIteration = 0, iterations = 0;
             (randomIteration < numRandomIterations) &&
             (iterations < 2 * numRandomIterations);
             iterations++) {
          uint32_t gemmKPerBlock = params[2][rng() % params[2].size()];
          uint32_t gemmMPerWave = mPerWaveRange[rng() % mPerWaveRange.size()];
          uint32_t gemmNPerWave = nPerWaveRange[rng() % nPerWaveRange.size()];
          uint32_t gemmMnPerXdl = params[3][rng() % params[3].size()];
          uint32_t gemmKPack = params[4][rng() % params[4].size()];
          uint32_t gemmSchedule = params[5][rng() % params[5].size()];
          uint32_t splitKFactor =
              optimalSplitKFactors[rng() % optimalSplitKFactors.size()];

          if (isWmma) {
            int64_t rdnaWaves = (gemm0MPerBlock / gemmMPerWave) *
                                (gemm0NPerBlock / gemmNPerWave);
            if (rdnaWaves < numEUPerCU)
              continue;
          }

          auto gemmGemmParams = GemmGemmParamsAttr::get(
              gemmGemmOp.getContext(), gemm0MPerBlock, gemm1MPerBlock,
              gemm0NPerBlock, gemmKPerBlock, gemmMPerWave, gemmNPerWave,
              gemmMnPerXdl, gemmKPack, splitKFactor, gemmSchedule,
              outputSwizzle, wavesPerEU, true);
          if (succeeded(PopulateParamsGemmGemm::paramsProbablyValid(
                  b, gemmGemmOp, gemmGemmParams))) {
            newSpace->tuningRange.insert(
                cast<RockTuningParamAttrInterface>(gemmGemmParams));
            randomIteration++;
          }
        }
      }
    }
  }
}

// Generate brute force configs for greedy second iteration (Phase 2)
static void
createGemmGemmTuningRangeGreedyPhase2(TuningParamSet *newSpace,
                                      RockGemmGemmWrapperInterface gemmGemmOp,
                                      StringRef winningConfig) {
  StringAttr arch = rock::getArchValue(gemmGemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (!archInfo.isAccel(gemmGemmOp)) {
    // We only support GPUs with matrix accelerator extensions
    return;
  }
  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParams =
      getAccelRangeGemmGemm(gemmGemmOp, TuningParamSetKind::Greedy);
  bool isWmma = archInfo.isWmma(gemmGemmOp);
  int64_t waveSize =
      rock::lookupArchInfo(rock::getArchValue(gemmGemmOp)).waveSize;
  int64_t numEUPerCU = archInfo.numEUPerCU;
  int64_t outputSwizzle{2}, wavesPerEU{0};
  OpBuilder b(gemmGemmOp.getContext());

  auto gemmGemmPerfConfig =
      GemmGemmParamsAttr::get(b.getStringAttr(winningConfig), isWmma);
  assert(gemmGemmPerfConfig &&
         "Tile sizes must be extracted from winning config");
  uint32_t gemm0MPerBlock = gemmGemmPerfConfig.getMPerBlockG0();
  uint32_t gemm1MPerBlock = gemmGemmPerfConfig.getMPerBlockG1();
  uint32_t gemm0NPerBlock = gemmGemmPerfConfig.getNPerBlockG0();

  SmallVector<uint32_t> mPerWaveRange =
      computeDPerWave(TuningParamSetKind::Greedy, gemm0MPerBlock, waveSize);
  SmallVector<uint32_t> nPerWaveRange =
      computeDPerWave(TuningParamSetKind::Greedy, gemm0NPerBlock, waveSize);
  auto optimalSplitKFactors =
      computeOptimalSplitKFactors(gemmGemmOp, gemm0NPerBlock);

  for (uint32_t gemmKPerBlock : validRangeGemmGemmParams[2]) {
    for (uint32_t gemmMPerWave : mPerWaveRange) {
      for (uint32_t gemmNPerWave : nPerWaveRange) {
        if (isWmma) {
          int64_t rdnaWaves =
              (gemm0MPerBlock / gemmMPerWave) * (gemm0NPerBlock / gemmNPerWave);
          if (rdnaWaves < numEUPerCU)
            continue;
        }
        for (uint32_t gemmMnPerXdl : validRangeGemmGemmParams[3]) {
          for (uint32_t gemmKPack : validRangeGemmGemmParams[4]) {
            for (int64_t splitKFactor : optimalSplitKFactors) {
              for (uint32_t gemmSchedule : validRangeGemmGemmParams[5]) {
                auto gemmGemmParams = GemmGemmParamsAttr::get(
                    gemmGemmOp.getContext(), gemm0MPerBlock, gemm1MPerBlock,
                    gemm0NPerBlock, gemmKPerBlock, gemmMPerWave, gemmNPerWave,
                    gemmMnPerXdl, gemmKPack, splitKFactor, gemmSchedule,
                    outputSwizzle, wavesPerEU, true);
                if (succeeded(PopulateParamsGemmGemm::paramsProbablyValid(
                        b, gemmGemmOp, gemmGemmParams))) {
                  newSpace->tuningRange.insert(
                      cast<RockTuningParamAttrInterface>(gemmGemmParams));
                }
              }
            }
          }
        }
      }
    }
  }
}

// With almost all tuning params already set, tune for fine-tuning parameters by
// brute force (greedy tuning, phase 3)
static void
createGemmGemmTuningRangeGreedyPhase3(TuningParamSet *newSpace,
                                      RockGemmGemmWrapperInterface gemmGemmOp,
                                      StringRef winningConfig) {
  StringAttr arch = rock::getArchValue(gemmGemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (!archInfo.isAccel(gemmGemmOp)) {
    // We only support GPUs with matrix accelerator extensions
    return;
  }
  bool isWmma = archInfo.isWmma(gemmGemmOp);
  OpBuilder b(gemmGemmOp.getContext());

  auto gemmGemmPerfConfig =
      GemmGemmParamsAttr::get(b.getStringAttr(winningConfig), isWmma);
  assert(gemmGemmPerfConfig &&
         "Tile sizes must be extracted from winning config");
  uint32_t gemm0MPerBlock = gemmGemmPerfConfig.getMPerBlockG0();
  uint32_t gemm1MPerBlock = gemmGemmPerfConfig.getMPerBlockG1();
  uint32_t gemm0NPerBlock = gemmGemmPerfConfig.getNPerBlockG0();
  uint32_t gemmKPerBlock = gemmGemmPerfConfig.getKpackPerBlock();
  uint32_t gemmMPerWave = gemmGemmPerfConfig.getMPerWave();
  uint32_t gemmNPerWave = gemmGemmPerfConfig.getNPerWave();
  uint32_t gemmMnPerXdl = gemmGemmPerfConfig.getMnPerXdl();
  uint32_t gemmKPack = gemmGemmPerfConfig.getKpack();
  uint32_t splitKFactor = gemmGemmPerfConfig.getSplitKFactor();
  uint32_t gemmSchedule = gemmGemmPerfConfig.getScheduleVersion();

  int64_t maxWavesPerEU =
      rock::lookupArchInfo(rock::getArchValue(gemmGemmOp)).maxWavesPerEU;
  auto finetuningParams = getFinetuningParams(maxWavesPerEU);

  for (int64_t outputSwizzle : finetuningParams[0]) {
    for (uint32_t wavesPerEU : finetuningParams[1]) {
      auto gemmGemmParams = GemmGemmParamsAttr::get(
          gemmGemmOp.getContext(), gemm0MPerBlock, gemm1MPerBlock,
          gemm0NPerBlock, gemmKPerBlock, gemmMPerWave, gemmNPerWave,
          gemmMnPerXdl, gemmKPack, splitKFactor, gemmSchedule, outputSwizzle,
          wavesPerEU, true);
      if (succeeded(PopulateParamsGemmGemm::paramsProbablyValid(
              b, gemmGemmOp, gemmGemmParams))) {
        newSpace->tuningRange.insert(
            cast<RockTuningParamAttrInterface>(gemmGemmParams));
      }
    }
  }
}

// Keep in sync with attentionSweeps.py
// The full space is a brute-force search for attention kernels
static void createGemmGemmTuningRangeBF(TuningParamSet *newSpace,
                                        RockGemmGemmWrapperInterface gemmGemmOp,
                                        TuningParamSetKind kind) {
  StringAttr arch = rock::getArchValue(gemmGemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  GemmFeatures features = archInfo.defaultFeatures;
  if (!archInfo.isAccel(gemmGemmOp)) {
    // We only support GPUs with matrix accelerator extensions
    return;
  }
  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParams =
      getAccelRangeGemmGemm(gemmGemmOp, kind);
  bool isWmma = bitEnumContainsAny(features, GemmFeatures::wmma);
  // auto archInfo = rock::lookupArchInfo(rock::getArchValue(gemmGemmOp));
  int64_t waveSize = archInfo.waveSize;
  int64_t numEUPerCU = archInfo.numEUPerCU;
  int64_t outputSwizzle{2}, wavesPerEU{0};
  OpBuilder b(gemmGemmOp.getContext());
  for (uint32_t gemm0MPerBlock : validRangeGemmGemmParams[0]) {
    SmallVector<uint32_t> mPerWaveRange =
        computeDPerWave(kind, gemm0MPerBlock, waveSize);
    SmallVector<uint32_t> mPerBlockGemm1 =
        compute1MPerBlock(kind, gemm0MPerBlock);
    for (uint32_t gemm1MPerBlock : mPerBlockGemm1) {
      for (uint32_t gemm0NPerBlock : validRangeGemmGemmParams[1]) {
        SmallVector<uint32_t> nPerWaveRange =
            computeDPerWave(kind, gemm0NPerBlock, waveSize);
        auto optimalSplitKFactors =
            computeOptimalSplitKFactors(gemmGemmOp, gemm0NPerBlock);

        for (uint32_t gemmKPerBlock : validRangeGemmGemmParams[2]) {
          for (uint32_t gemmMPerWave : mPerWaveRange) {
            for (uint32_t gemmNPerWave : nPerWaveRange) {
              for (uint32_t gemmMnPerXdl : validRangeGemmGemmParams[3]) {
                for (uint32_t gemmKPack : validRangeGemmGemmParams[4]) {
                  for (int64_t splitKFactor : optimalSplitKFactors) {
                    for (uint32_t gemmSchedule : validRangeGemmGemmParams[5]) {
                      if (isWmma) {
                        int64_t rdnaWaves = (gemm0MPerBlock / gemmMPerWave) *
                                            (gemm0NPerBlock / gemmNPerWave);
                        if (rdnaWaves < numEUPerCU) {
                          continue;
                        }
                      }
                      auto gemmGemmParams = GemmGemmParamsAttr::get(
                          gemmGemmOp.getContext(), gemm0MPerBlock,
                          gemm1MPerBlock, gemm0NPerBlock, gemmKPerBlock,
                          gemmMPerWave, gemmNPerWave, gemmMnPerXdl, gemmKPack,
                          splitKFactor, gemmSchedule, outputSwizzle, wavesPerEU,
                          true);
                      if (succeeded(PopulateParamsGemmGemm::paramsProbablyValid(
                              b, gemmGemmOp, gemmGemmParams))) {
                        newSpace->tuningRange.insert(
                            cast<RockTuningParamAttrInterface>(gemmGemmParams));
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
  }
}

static double computeWorkImbalance(GemmSize origGemmSize, int32_t gemmMPerBlock,
                                   int32_t gemmNPerBlock, int32_t gemmKPerBlock,
                                   int32_t kPack, uint32_t numCUs,
                                   int32_t splitKFactor = 1) {
  // Use calculatePaddedGemmSize with individual parameters
  const GemmSize gemmSize = calculatePaddedGemmSize(
      gemmKPerBlock, gemmMPerBlock, gemmNPerBlock, origGemmSize, kPack);
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
                            int32_t gemmKPerBlock, int32_t kPack) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  SmallVector<int64_t> splitKValues = {1};

  auto func = cast<func::FuncOp>(gemmOp->getParentOp());
  if (!func->hasAttr(rock::EnableSplitKForTuningAttr::getMnemonic())) {
    return splitKValues;
  }

  uint32_t numCUs = rock::lookupArchInfo(rock::getArchValue(gemmOp)).minNumCU;
  if (succeeded(rock::getNumCU(gemmOp))) {
    numCUs = rock::getNumCU(gemmOp).value();
  }

  return computeOptimalSplitKFactors(info.gemmSize, gemmMPerBlock,
                                     gemmNPerBlock, gemmKPerBlock, kPack,
                                     numCUs);
}

// Check if a RockGemmWrapperInterface op is a forward conv eligible for
// Winograd (3x3, stride=1, dilation=1, f32/f16, C*K >= 2048).
// Used to generate a small blockSize-only tuning space for Winograd.
static bool isWinogradEligible(RockGemmWrapperInterface op) {
  auto convOp = dyn_cast<ConvOp>(op.getOperation());
  if (!convOp)
    return false;
  auto strides = extractFromIntegerArrayAttr<int64_t>(convOp.getStrides());
  auto dilations = extractFromIntegerArrayAttr<int64_t>(convOp.getDilations());
  for (auto s : strides)
    if (s != 1)
      return false;
  for (auto d : dilations)
    if (d != 1)
      return false;
  auto filterType = cast<ShapedType>(convOp.getFilter().getType());
  Type elemType = filterType.getElementType();
  if (!elemType.isF32() && !elemType.isF16())
    return false;
  auto filterLayout = convOp->getAttrOfType<ArrayAttr>("filter_layout");
  if (!filterLayout)
    return false;
  int64_t filH = 0, filW = 0, C = 1, K = 1;
  for (auto [idx, attr] :
       llvm::enumerate(filterLayout.getAsRange<StringAttr>())) {
    StringRef name = attr.getValue();
    int64_t dim = filterType.getShape()[idx];
    if (name == "0")
      filH = dim;
    else if (name == "1")
      filW = dim;
    else if (name == "c")
      C = dim;
    else if (name == "k")
      K = dim;
  }
  if (filH != 3 || filW != 3)
    return false;
  if (C * K < 2048)
    return false;
  return true;
}

static void createWinogradTuningRange(TuningParamSet *newSpace,
                                      RockGemmWrapperInterface gemmOp) {
  OpBuilder b(gemmOp.getContext());
  for (uint32_t blockSize : {64u, 128u, 256u, 512u}) {
    auto params = GeneralGemmParamsAttr::get(
        b.getContext(), blockSize, /*kPerBlock=*/8, /*mPerBlock=*/64,
        /*nPerBlock=*/64, /*kPerThread=*/1, /*mPerThread=*/4,
        /*nPerThread=*/4, /*kpack=*/1, /*splitKFactor=*/1,
        /*scheduleVersion=*/1, /*outputSwizzle=*/2);
    newSpace->tuningRange.insert(
        cast<RockTuningParamAttrInterface>(params));
  }
}

// The full space is a brute-force search starting with the configs that have
// the smallest parameters. This filters out perf configs that are
// known to be impossible during tthe AffixTuningParams check.
// If `kind` is Full, also filters out unlikely-to-be-good configurations.
static void createGemmTuningRangeBF(TuningParamSet *newSpace,
                                    RockGemmWrapperInterface gemmOp,
                                    TuningParamSetKind kind) {
  if (isWinogradEligible(gemmOp)) {
    createWinogradTuningRange(newSpace, gemmOp);
    return;
  }
  auto info = PopulateParamsInfo::fromOp(gemmOp);

  // blockSize M/block N/block K/block M/thread N/thread
  const std::vector<std::vector<uint32_t>> validRangeGeneralGemmParams = {
      {64, 128, 256}, {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};

  const std::vector<std::vector<uint32_t>> accelParams =
      getAccelRangeGemm(gemmOp, kind);

  StringAttr arch = rock::getArchValue(gemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  std::unique_ptr<PopulateParamsAccel> tuningInfo;
  if (archInfo.isMfma(gemmOp))
    tuningInfo = std::make_unique<PopulateParamsXDL>();
  else
    tuningInfo = std::make_unique<PopulateParamsWmma>();
  int64_t waveSize = rock::lookupArchInfo(rock::getArchValue(gemmOp)).waveSize;

  // hardcode to use heuristics
  int64_t outputSwizzle{2}, wavesPerEU{0}, gridGroupSize{0};
  OpBuilder b(gemmOp.getContext());
  if (archInfo.isAccel(gemmOp)) {
    for (uint32_t gemmMPerBlock : accelParams[0]) {
      SmallVector<uint32_t> mPerWaveRange =
          computeDPerWave(kind, gemmMPerBlock, waveSize);
      for (uint32_t gemmNPerBlock : accelParams[1]) {
        SmallVector<uint32_t> nPerWaveRange =
            computeDPerWave(kind, gemmNPerBlock, waveSize);
        for (uint32_t gemmKPerBlock : accelParams[2]) {
          for (uint32_t gemmMPerWave : mPerWaveRange) {
            for (uint32_t gemmNPerWave : nPerWaveRange) {
              for (uint32_t gemmMnPerXdl : accelParams[3]) {
                for (uint32_t gemmKPack : accelParams[4]) {
                  auto optimalSplitKFactors = computeOptimalSplitKFactors(
                      gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                      gemmKPack);
                  for (int64_t splitKFactor : optimalSplitKFactors) {
                    for (int64_t gemmSchedule : accelParams[5]) {
                      for (uint32_t forceUnroll : accelParams[6]) {
                        auto gemmParams = AccelGemmParamsAttr::get(
                            b.getContext(), gemmKPerBlock, gemmMPerBlock,
                            gemmNPerBlock, gemmKPack, gemmMPerWave,
                            gemmNPerWave, gemmMnPerXdl, splitKFactor,
                            gemmSchedule, outputSwizzle, wavesPerEU,
                            gridGroupSize, forceUnroll);
                        if (gemmMPerBlock >= gemmMPerWave &&
                            gemmNPerBlock >= gemmNPerWave) {
                          if (succeeded(tuningInfo->paramsProbablyValid(
                                  b, info, gemmParams)) &&
                              (kind != TuningParamSetKind::Full ||
                               succeeded(tuningInfo->couldBePerformant(
                                   info, gemmParams))))
                            newSpace->tuningRange.insert(
                                cast<RockTuningParamAttrInterface>(gemmParams));
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
    }
  } else {
    // Non-accel
    PopulateParams tuningInfo;
    for (uint32_t blockSize : validRangeGeneralGemmParams[0]) {
      for (uint32_t gemmMPerBlock : validRangeGeneralGemmParams[1]) {
        for (uint32_t gemmNPerBlock : validRangeGeneralGemmParams[2]) {
          for (uint32_t gemmKPerBlock : validRangeGeneralGemmParams[3]) {
            for (uint32_t gemmMPerThread : validRangeGeneralGemmParams[4]) {
              auto optimalSplitKFactors = computeOptimalSplitKFactors(
                  gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock, 1);
              for (auto splitKFactor : optimalSplitKFactors) {
                for (uint32_t gemmNPerThread : validRangeGeneralGemmParams[5]) {
                  // hardcode schedule version to v1 and outputSwizzle to
                  // heuristics = 2
                  auto gemmParams = GeneralGemmParamsAttr::get(
                      b.getContext(), blockSize, gemmKPerBlock, gemmMPerBlock,
                      gemmNPerBlock, 1, gemmMPerThread, gemmNPerThread, 1,
                      splitKFactor, 1, 2);
                  if (succeeded(tuningInfo.paramsProbablyValid(b, info,
                                                               gemmParams)) &&
                      (kind != TuningParamSetKind::Full ||
                       succeeded(
                           tuningInfo.couldBePerformant(info, gemmParams))))
                    newSpace->tuningRange.insert(
                        cast<RockTuningParamAttrInterface>(gemmParams));
                }
              }
            }
          }
        }
      }
    }
  }
}

static void createGemmTuningRangeQuick(TuningParamSet *newSpace,
                                       RockGemmWrapperInterface gemmOp) {
  if (isWinogradEligible(gemmOp)) {
    createWinogradTuningRange(newSpace, gemmOp);
    return;
  }
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  OpBuilder b(gemmOp.getContext());
  StringAttr arch = rock::getArchValue(gemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  if (archInfo.isMfma(gemmOp)) {
    PopulateParamsXDL tuningInfo;

    for (AccelGemmParamsAttr param : tuningInfo.orderParams(
             tuningInfo.getTuningParameters(b, info.kernelType, info.gemmAType,
                                            info.gemmBType, info.arch),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.insert(cast<RockTuningParamAttrInterface>(param));
    }
  } else if (archInfo.isWmma(gemmOp)) {
    // Wmma
    PopulateParamsWmma tuningInfo;
    for (AccelGemmParamsAttr param : tuningInfo.orderParams(
             tuningInfo.getTuningParameters(b, info.kernelType, info.gemmAType,
                                            info.gemmBType, info.arch),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.insert(cast<RockTuningParamAttrInterface>(param));
    }
  } else {
    // Non-XDLOPS
    PopulateParams tuningInfo;
    for (GeneralGemmParamsAttr param : tuningInfo.orderParams(
             tuningInfo.getTuningParameters(b, info.kernelType, info.gemmAType,
                                            info.gemmBType, info.arch),
             info.gemmSize)) {
      if (succeeded(tuningInfo.paramsProbablyValid(b, info, param)) &&
          succeeded(tuningInfo.couldBePerformant(info, param)))
        newSpace->tuningRange.insert(cast<RockTuningParamAttrInterface>(param));
    }
  }
}

static void
createGemmGemmTuningRangeQuick(TuningParamSet *newSpace,
                               RockGemmGemmWrapperInterface gemmGemmOp) {
  OpBuilder b(gemmGemmOp.getContext());
  for (GemmGemmParamsAttr params :
       PopulateParamsGemmGemm::getTuningParameters(b, gemmGemmOp)) {
    if (succeeded(PopulateParamsGemmGemm::paramsProbablyValid(b, gemmGemmOp,
                                                              params))) {
      newSpace->tuningRange.insert(cast<RockTuningParamAttrInterface>(params));
    }
  }
}

bool needToUpdateBest(TuningParamSetKind kind) {
  switch (kind) {
  case TuningParamSetKind::Quick:
  case TuningParamSetKind::Full:
  case TuningParamSetKind::Exhaustive:
    return false;
  case TuningParamSetKind::Greedy:
    return true;
  }
  llvm_unreachable("invalid tuning kind");
}

unsigned getNumberOfIterations(TuningParamSetKind kind) {
  switch (kind) {
  case TuningParamSetKind::Quick:
  case TuningParamSetKind::Full:
  case TuningParamSetKind::Exhaustive:
    return 1;
  case TuningParamSetKind::Greedy:
    return 3;
  }
  llvm_unreachable("invalid tuning kind");
}

// Tune MperBlock and NPerBlock only, generating random configs for the rest of
// parameters (greedy tuning, phase 1)
static void createGemmTuningRangeGreedyPhase1(TuningParamSet *newSpace,
                                              RockGemmWrapperInterface gemmOp,
                                              unsigned numRandomPerTileSize,
                                              unsigned int seed) {
  StringAttr arch = rock::getArchValue(gemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  OpBuilder b(gemmOp.getContext());
  const std::vector<std::vector<uint32_t>> params =
      getAccelRangeGemm(gemmOp, TuningParamSetKind::Greedy);
  std::mt19937 rng(seed);
  std::unique_ptr<PopulateParamsAccel> tuningInfo;
  if (archInfo.isMfma(gemmOp))
    tuningInfo = std::make_unique<PopulateParamsXDL>();
  else
    tuningInfo = std::make_unique<PopulateParamsWmma>();
  int64_t waveSize = rock::lookupArchInfo(rock::getArchValue(gemmOp)).waveSize;

  // hardcode to use heuristics
  int64_t outputSwizzle{2}, wavesPerEU{0}, gridGroupSize{0};
  for (uint32_t gemmMPerBlock : params[0]) {
    SmallVector<uint32_t> mPerWaveRange =
        computeDPerWave(TuningParamSetKind::Greedy, gemmMPerBlock, waveSize);
    for (uint32_t gemmNPerBlock : params[1]) {
      SmallVector<uint32_t> nPerWaveRange =
          computeDPerWave(TuningParamSetKind::Greedy, gemmNPerBlock, waveSize);
      uint32_t totalIterations = params[2].size() * mPerWaveRange.size() *
                                 nPerWaveRange.size() * params[3].size() *
                                 params[4].size() * params[5].size() *
                                 params[6].size();
      uint32_t numRandomIterations =
          std::min(numRandomPerTileSize, totalIterations);
      for (uint32_t randomIteration = 0, iterations = 0;
           (randomIteration < numRandomIterations) &&
           (iterations < 2 * numRandomIterations);
           iterations++) {
        uint32_t gemmKPerBlock = params[2][rng() % params[2].size()];
        uint32_t gemmMPerWave = mPerWaveRange[rng() % mPerWaveRange.size()];
        uint32_t gemmNPerWave = nPerWaveRange[rng() % nPerWaveRange.size()];
        uint32_t gemmMnPerXdl = params[3][rng() % params[3].size()];
        uint32_t gemmKPack = params[4][rng() % params[4].size()];
        uint32_t gemmSchedule = params[5][rng() % params[5].size()];
        uint32_t forceUnroll = params[6][rng() % params[6].size()];
        auto optimalSplitKFactors = computeOptimalSplitKFactors(
            gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock, gemmKPack);
        uint32_t splitKFactor =
            optimalSplitKFactors[rng() % optimalSplitKFactors.size()];
        auto gemmParams = AccelGemmParamsAttr::get(
            b.getContext(), gemmKPerBlock, gemmMPerBlock, gemmNPerBlock,
            gemmKPack, gemmMPerWave, gemmNPerWave, gemmMnPerXdl, splitKFactor,
            gemmSchedule, outputSwizzle, wavesPerEU, gridGroupSize,
            forceUnroll);
        if (succeeded(tuningInfo->paramsProbablyValid(b, info, gemmParams))) {
          newSpace->tuningRange.insert(
              cast<RockTuningParamAttrInterface>(gemmParams));
          randomIteration++;
        }
      }
    }
  }
}

// With MperBlock and NPerBlock already set, tune for the rest of parameters by
// brute force (greedy tuning, phase 2)
static void createGemmTuningRangeGreedyPhase2(TuningParamSet *newSpace,
                                              RockGemmWrapperInterface gemmOp,
                                              StringRef winningConfig) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  OpBuilder b(gemmOp.getContext());
  StringAttr arch = rock::getArchValue(gemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  GemmFeatures currentFeatures = archInfo.defaultFeatures;

  uint32_t winningMPerBlock, winningNPerBlock;
  AccelGemmParamsAttr validParams;
  auto populateParamsAccelPtr = PopulateParamsAccel::select(currentFeatures);
  LogicalResult status = populateParamsAccelPtr->obtainTuningParameters(
      b, gemmOp, winningConfig, validParams);
  assert(llvm::succeeded(status) &&
         "Tile sizes must be extracted from winning config");
  winningMPerBlock = validParams.getMPerBlock();
  winningNPerBlock = validParams.getNPerBlock();

  const std::vector<std::vector<uint32_t>> params =
      getAccelRangeGemm(gemmOp, TuningParamSetKind::Greedy);
  std::unique_ptr<PopulateParamsAccel> tuningInfo;
  if (archInfo.isMfma(gemmOp))
    tuningInfo = std::make_unique<PopulateParamsXDL>();
  else
    tuningInfo = std::make_unique<PopulateParamsWmma>();
  int64_t waveSize = rock::lookupArchInfo(rock::getArchValue(gemmOp)).waveSize;
  SmallVector<uint32_t> mPerWaveRange =
      computeDPerWave(TuningParamSetKind::Greedy, winningMPerBlock, waveSize);
  SmallVector<uint32_t> nPerWaveRange =
      computeDPerWave(TuningParamSetKind::Greedy, winningNPerBlock, waveSize);

  // hardcode to use heuristics
  int64_t outputSwizzle{2}, wavesPerEU{0}, gridGroupSize{0};
  for (uint32_t gemmKPerBlock : params[2]) {
    for (uint32_t gemmMPerWave : mPerWaveRange) {
      for (uint32_t gemmNPerWave : nPerWaveRange) {
        for (uint32_t gemmMnPerXdl : params[3]) {
          for (uint32_t gemmKPack : params[4]) {
            auto optimalSplitKFactors = computeOptimalSplitKFactors(
                gemmOp, winningMPerBlock, winningNPerBlock, gemmKPerBlock,
                gemmKPack);
            for (int64_t splitKFactor : optimalSplitKFactors) {
              for (int64_t gemmSchedule : params[5]) {
                for (uint32_t forceUnroll : params[6]) {
                  auto gemmParams = AccelGemmParamsAttr::get(
                      b.getContext(), gemmKPerBlock, winningMPerBlock,
                      winningNPerBlock, gemmKPack, gemmMPerWave, gemmNPerWave,
                      gemmMnPerXdl, splitKFactor, gemmSchedule, outputSwizzle,
                      wavesPerEU, gridGroupSize, forceUnroll);
                  if (winningMPerBlock >= gemmMPerWave &&
                      winningNPerBlock >= gemmNPerWave) {
                    if (succeeded(tuningInfo->paramsProbablyValid(b, info,
                                                                  gemmParams)))
                      newSpace->tuningRange.insert(
                          cast<RockTuningParamAttrInterface>(gemmParams));
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

// With almost all tuning params already set, tune for fine-tuning parameters by
// brute force (greedy tuning, phase 3)
static void createGemmTuningRangeGreedyPhase3(TuningParamSet *newSpace,
                                              RockGemmWrapperInterface gemmOp,
                                              StringRef winningConfig) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  OpBuilder b(gemmOp.getContext());
  StringAttr arch = rock::getArchValue(gemmOp);
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  GemmFeatures currentFeatures = archInfo.defaultFeatures;

  AccelGemmParamsAttr validParams;
  auto populateParamsAccelPtr = PopulateParamsAccel::select(currentFeatures);
  LogicalResult status = populateParamsAccelPtr->obtainTuningParameters(
      b, gemmOp, winningConfig, validParams);
  assert(llvm::succeeded(status) &&
         "Tile sizes must be extracted from winning config");
  uint32_t mPerBlock = validParams.getMPerBlock();
  uint32_t nPerBlock = validParams.getNPerBlock();
  uint32_t kpackPerBlock = validParams.getKpackPerBlock();
  uint32_t mPerWave = validParams.getMPerWave();
  uint32_t nPerWave = validParams.getNPerWave();
  uint32_t mnPerXdl = validParams.getMnPerXdl();
  uint32_t kpack = validParams.getKpack();
  uint32_t splitKFactor = validParams.getSplitKFactor();
  uint32_t scheduleVersion = validParams.getScheduleVersion();
  uint32_t forceUnroll = validParams.getForceUnroll();

  // StringAttr arch = rock::getArchValue(gemmOp);
  // rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  std::unique_ptr<PopulateParamsAccel> tuningInfo;
  if (archInfo.isMfma(gemmOp))
    tuningInfo = std::make_unique<PopulateParamsXDL>();
  else
    tuningInfo = std::make_unique<PopulateParamsWmma>();

  int64_t maxWavesPerEU =
      rock::lookupArchInfo(rock::getArchValue(gemmOp)).maxWavesPerEU;
  auto finetuningParams = getFinetuningParams(maxWavesPerEU);

  for (int64_t outputSwizzle : finetuningParams[0]) {
    for (int64_t wavesPerEU : finetuningParams[1]) {
      for (int64_t gridGroupSize : finetuningParams[2]) {
        auto gemmParams = AccelGemmParamsAttr::get(
            b.getContext(), kpackPerBlock, mPerBlock, nPerBlock, kpack,
            mPerWave, nPerWave, mnPerXdl, splitKFactor, scheduleVersion,
            outputSwizzle, wavesPerEU, gridGroupSize, forceUnroll);
        if (succeeded(tuningInfo->paramsProbablyValid(b, info, gemmParams)))
          newSpace->tuningRange.insert(
              cast<RockTuningParamAttrInterface>(gemmParams));
      }
    }
  }
}

TuningParamSet *
createTunableParamSpace(ModuleOp mod, TuningParamSetKind kind,
                        rock::TuningParamSpaceSettings &settings) {
  struct TuningParamSet *newSpace;
  newSpace = new TuningParamSet();
  newSpace->effectiveKind = kind;

  // create range and heuristic
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        StringAttr arch = rock::getArchValue(op);
        rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
        // greedy is not implemented for non-accel
        if (!archInfo.isAccel(op) && kind == TuningParamSetKind::Greedy) {
          kind = TuningParamSetKind::Exhaustive;
          newSpace->effectiveKind = kind;
          llvm::errs() << "Greedy tuning not implemented for non-accel, using "
                          "Exhaustive instead\n";
        }
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createGemmTuningRangeBF(newSpace, op, kind);
          // tuning space for exhaustive and full should also include quick
          // tuning space
          [[fallthrough]];
        case TuningParamSetKind::Quick:
          createGemmTuningRangeQuick(newSpace, op);
          break;
        case TuningParamSetKind::Greedy:
          if (settings.iteration == 0) {
            // First iteration: random configs per tile size
            createGemmTuningRangeGreedyPhase1(
                newSpace, op, NUM_RANDOM_PERFCONFIGS_PER_TILE_SIZE, RND_SEED);
          } else if (settings.iteration == 1) {
            // Second iteration: brute force (except waves_per_eu,
            // output_swizzle and grid_group_size, which we hardcode to use the
            // heuristic) with winning tile sizes
            createGemmTuningRangeGreedyPhase2(newSpace, op,
                                              settings.winningConfig);
          } else {
            // Third iteration: brute force the remaining configs (waves_per_eu,
            // output_swizzle and grid_group_size)
            createGemmTuningRangeGreedyPhase3(newSpace, op,
                                              settings.winningConfig);
          }
          break;
        }
        newSpace->primaryOpType = op.getKernelType();
        return WalkResult::interrupt();
      });
  WalkResult findGemmGemm =
      mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createGemmGemmTuningRangeBF(newSpace, op, kind);
          // tuning space for exhaustive and full should also include quick
          // tuning space
          [[fallthrough]];
        case TuningParamSetKind::Quick:
          createGemmGemmTuningRangeQuick(newSpace, op);
          break;
        case TuningParamSetKind::Greedy:
          if (settings.iteration == 0) {
            // First iteration: random configs per tile size
            createGemmGemmTuningRangeGreedyPhase1(
                newSpace, op, NUM_RANDOM_PERFCONFIGS_PER_TILE_SIZE, RND_SEED);
          } else if (settings.iteration == 1) {
            // Second iteration: brute force (except waves_per_eu and
            // output_swizzle, which we hardcode to use the heuristic) with
            // winning tile sizes
            createGemmGemmTuningRangeGreedyPhase2(newSpace, op,
                                                  settings.winningConfig);
          } else {
            // Third iteration: brute force the remaining configs (waves_per_eu
            // and output_swizzle)
            createGemmGemmTuningRangeGreedyPhase3(newSpace, op,
                                                  settings.winningConfig);
          }
          break;
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
  int64_t numCU = rock::getNumCUValue(gemmGemmOp);
  int64_t numChiplets = rock::getNumChipletsValue(gemmGemmOp);
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  int64_t headDimQK;
  int64_t headDimV;
  int64_t seqLenQ;
  int64_t seqLenK;
  llvm::raw_svector_ostream problemOS(out);
  // ARCH string
  problemOS << StringRef(rock::getArchValue(gemmGemmOp)) << tab;
  // Number of Compute Units
  problemOS << numCU << tab;
  // Number of chiplets
  problemOS << numChiplets << tab;

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

    problemOS << "-split_kv " << attentionOp.getSplitKV() << sep;
    problemOS << "-num_heads_q " << attentionOp.getNumHeadsQ() << sep;
    problemOS << "-num_heads_kv " << attentionOp.getNumHeadsKV() << sep;
    problemOS << "-g " << qShape[0] / attentionOp.getNumHeadsQ() << sep;
  }

  if (!isConvGemm && !isAttention)
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
  int64_t numCU = rock::getNumCUValue(gemmIF);
  int64_t numChiplets = rock::getNumChipletsValue(gemmIF);
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
  problemOS << StringRef(rock::getArchValue(gemmIF)).trim("\"") << tab;
  // Number of Compute Units
  problemOS << numCU << tab;
  // Number of chiplets
  problemOS << numChiplets << tab;

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
    bool isScaledGemm =
        rGemmOp.getScaleA() != nullptr && rGemmOp.getScaleB() != nullptr;
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
    } else if (isa<Float4E2M1FNType>(elemTypeA) &&
               isa<Float4E2M1FNType>(elemTypeB)) {
      problemOS << "f4E2M1FN";
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

    if (isScaledGemm) {
      problemOS << "-scaledGemm" << sep;
      auto scaleA = rGemmOp.getScaleA();
      auto scaleB = rGemmOp.getScaleB();
      problemOS << "-scale_a_dtype ";
      auto scaleAElemType = scaleA.getType().getElementType();
      auto scaleBElemType = scaleB.getType().getElementType();
      if (scaleAElemType.isF32()) {
        problemOS << "f32";
      } else if (isa<Float8E8M0FNUType>(scaleAElemType)) {
        problemOS << "f8E8M0FNU";
      } else {
        llvm_unreachable("Unsupported scale A element type");
      }
      problemOS << sep;
      problemOS << "-scale_b_dtype ";
      if (scaleBElemType.isF32()) {
        problemOS << "f32";
      } else if (isa<Float8E8M0FNUType>(scaleBElemType)) {
        problemOS << "f8E8M0FNU";
      } else {
        llvm_unreachable("Unsupported scale B element type");
      }
      problemOS << sep;
      problemOS << "-transScaleA" << sep;
      if (rGemmOp.getAScaleTransposed()) {
        problemOS << "true" << sep;
      } else {
        problemOS << "false" << sep;
      }
      problemOS << "-transScaleB" << sep;
      if (rGemmOp.getBScaleTransposed()) {
        problemOS << "true" << sep;
      } else {
        problemOS << "false" << sep;
      }
    }

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

static int64_t retrieveSplitKValue(rock::GemmFeatures features,
                                   StringAttr perfConfig) {
  // TODO: This is a hack, we should not be checking features here.
  bool isWmma = bitEnumContainsAny(features, GemmFeatures::wmma);
  bool isAccel = rock::isAccel(features);
  auto gemmGemmPerfConfig = GemmGemmParamsAttr::get(perfConfig, isWmma);
  if (gemmGemmPerfConfig)
    return gemmGemmPerfConfig.getSplitKFactor();

  if (isAccel) {
    auto params = AccelGemmParamsAttr::get(perfConfig, isWmma);
    return params ? params.getSplitKFactor() : 1;
  }
  auto params = GeneralGemmParamsAttr::get(perfConfig);
  return params ? params.getSplitKFactor() : 1;
}

bool isSplitKRequested(rock::GemmFeatures features, StringAttr perfConfig) {
  return retrieveSplitKValue(features, perfConfig) > 1;
}

bool isSplitKRequested(ModuleOp mod, StringRef perfConfig) {
  auto perfConfigAttr = StringAttr::get(mod->getContext(), perfConfig);
  WalkResult walkResult = mod.walk([&](Operation *op) -> WalkResult {
    if (isa<RockGemmWrapperInterface, RockGemmGemmWrapperInterface>(op)) {
      StringAttr arch = rock::getArchValue(op);
      rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
      GemmFeatures features = archInfo.defaultFeatures;
      if (isSplitKRequested(features, perfConfigAttr))
        return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return walkResult.wasInterrupted();
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
  bool fusible = succeeded(rock::testFusionLegalityReduce(module)) &&
                 succeeded(rock::testFusionLegalityBwdDataConv(module)) &&
                 succeeded(rock::testFusionLegalityAttentionSplitKV(module));
  if (!rock::isSplitKRequested(module, perfConfig))
    return fusible;
  return fusible && succeeded(rock::testFusionLegalitySplitK(module));
}

} // namespace rock
} // namespace mlir