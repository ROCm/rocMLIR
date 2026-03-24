//===- QuickTuningClassifier.cpp - ML-based tuning config filter ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningClassifier.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "quick-tuning-classifier"

using namespace mlir;
using namespace mlir::rock;

//===----------------------------------------------------------------------===//
// Tree data structures and inference
//===----------------------------------------------------------------------===//
namespace {

struct TreeEnsembleTree {
  int16_t numNodes;
  const int16_t *featureIdx;
  const float *threshold;
  const int16_t *leftChild;
  const int16_t *rightChild;
  const float *leafValue;
};

struct TreeEnsemble {
  const TreeEnsembleTree *trees;
  int16_t numTrees;
  float baseScore;
  float decisionThreshold;
};

struct ClassifierEntry {
  StringLiteral key;
  const TreeEnsemble *classifier;
};

float predictTree(const TreeEnsembleTree &tree, ArrayRef<float> features) {
  int16_t node = 0;
  while (tree.leftChild[node] != -1) {
    node = (features[tree.featureIdx[node]] <= tree.threshold[node])
               ? tree.leftChild[node]
               : tree.rightChild[node];
  }
  return tree.leafValue[node];
}

/// Raw ensemble score (sum of tree outputs + base score).
/// Higher = more likely to be performant.
float ensembleScore(const TreeEnsemble &forest, ArrayRef<float> features) {
  float sum = forest.baseScore;
  for (int16_t i = 0; i < forest.numTrees; ++i)
    sum += predictTree(forest.trees[i], features);
  return sum;
}

//===----------------------------------------------------------------------===//
// Feature extraction -- must match ACCEL_FEATURE_NAMES in trainClassifier.py
//===----------------------------------------------------------------------===//

enum AccelFeatureIndex : int {
  kFeatGridSizePerCU = 0,
  kFeatMPadFrac,
  kFeatNPadFrac,
  kFeatKPadFrac,
  kFeatPaddingOverhead,
  kFeatNumWaves,
  kFeatKIters,
  kFeatLdsElements,
  kFeatKPerBlockPerMnPerXdl,
  kFeatOutputSwizzle,
  kFeatWavesPerEU,
  kFeatGridGroupSize,
  kFeatScheduleVersion,
  kFeatTileReuse,
  kFeatProblemAI,
  kNumAccelFeatures
};

SmallVector<float, kNumAccelFeatures>
extractAccelFeatures(const PopulateParamsInfo &info,
                     AccelGemmParamsAttr params) {
  float m = static_cast<float>(info.gemmSize.m);
  float n = static_cast<float>(info.gemmSize.n);
  float k = static_cast<float>(info.gemmSize.k);
  float mPB = static_cast<float>(params.getMPerBlock());
  float nPB = static_cast<float>(params.getNPerBlock());
  float kPB = static_cast<float>(params.getKpackPerBlock());
  float kPack = static_cast<float>(params.getKpack());
  float mPW = static_cast<float>(params.getMPerWave());
  float nPW = static_cast<float>(params.getNPerWave());
  float kEff = kPB * kPack;
  float splitK = static_cast<float>(params.getSplitKFactor());

  float mTiles = std::ceil(m / mPB);
  float nTiles = std::ceil(n / nPB);
  float kIters = std::ceil(k / kEff);
  float numWaves = (mPB / mPW) * (nPB / nPW);
  float gridSize = mTiles * nTiles * splitK;

  float numCU = static_cast<float>(info.numCu);

  float mPadded = mTiles * mPB;
  float nPadded = nTiles * nPB;
  float kPadded = kIters * kEff;
  float originalVolume = m * n * k;
  float paddedVolume = mPadded * nPadded * kPadded;

  SmallVector<float, kNumAccelFeatures> features(kNumAccelFeatures);
  features[kFeatGridSizePerCU] = gridSize / std::max(numCU, 1.0f);
  features[kFeatMPadFrac] =
      (mPB > 0) ? static_cast<float>(
                      (static_cast<int64_t>(mPB) -
                       math_util::mod_1_to_n(info.gemmSize.m, (int64_t)mPB)) %
                      static_cast<int64_t>(mPB)) /
                      mPB
                : 0.0f;
  features[kFeatNPadFrac] =
      (nPB > 0) ? static_cast<float>(
                      (static_cast<int64_t>(nPB) -
                       math_util::mod_1_to_n(info.gemmSize.n, (int64_t)nPB)) %
                      static_cast<int64_t>(nPB)) /
                      nPB
                : 0.0f;
  features[kFeatKPadFrac] =
      (kEff > 0) ? static_cast<float>(
                       (static_cast<int64_t>(kEff) -
                        math_util::mod_1_to_n(info.gemmSize.k, (int64_t)kEff)) %
                       static_cast<int64_t>(kEff)) /
                       kEff
                 : 0.0f;
  features[kFeatPaddingOverhead] =
      (originalVolume > 0) ? (paddedVolume / originalVolume) - 1.0f : 0.0f;
  features[kFeatNumWaves] = numWaves;
  features[kFeatKIters] = kIters;
  features[kFeatLdsElements] = (mPB + nPB) * kPB * kPack;
  float mnPerXdl = static_cast<float>(params.getMnPerXdl());
  features[kFeatKPerBlockPerMnPerXdl] = (mnPerXdl > 0) ? kEff / mnPerXdl : 0.0f;
  features[kFeatOutputSwizzle] = static_cast<float>(params.getOutputSwizzle());
  features[kFeatWavesPerEU] = static_cast<float>(params.getWavesPerEU());
  features[kFeatGridGroupSize] = static_cast<float>(params.getGridGroupSize());
  features[kFeatScheduleVersion] =
      static_cast<float>(params.getScheduleVersion());
  features[kFeatTileReuse] = (mPB + nPB > 0) ? (mPB * nPB) / (mPB + nPB) : 0.0f;
  float denom = m * k + k * n + m * n;
  features[kFeatProblemAI] = (denom > 0) ? (m * n * k) / denom : 0.0f;
  return features;
}

//===----------------------------------------------------------------------===//
// Generated classifier data and lookup table
//===----------------------------------------------------------------------===//

#define CLASSIFIER_DATA_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningClassifier.inc"
#undef CLASSIFIER_DATA_GEN

StringRef kernelTypeToOpStr(KernelType kt) {
  switch (kt) {
  case KernelType::Gemm:
    return "gemm";
  case KernelType::Conv:
  case KernelType::ConvBwdData:
  case KernelType::ConvBwdWeight:
    return "conv";
  case KernelType::Attention:
  case KernelType::GemmElementwiseGemm:
  case KernelType::ConvElementwiseGemm:
    return "attention";
  }
  return "gemm";
}

std::string getClassifierKey(KernelType kernelType, StringRef arch,
                             Type dataType) {
  std::string dtype;
  if (dataType.isBF16()) {
    dtype = "bf16";
  } else if (dataType.isFloat()) {
    unsigned bw = dataType.getIntOrFloatBitWidth();
    dtype = (bw <= 8 ? "fp" : "f") + std::to_string(bw);
  } else if (dataType.isInteger()) {
    dtype = "i" + std::to_string(dataType.getIntOrFloatBitWidth());
  } else {
    return "";
  }
  return (Twine(kernelTypeToOpStr(kernelType)) + "_" + rock::getChipName(arch) +
          "_" + dtype)
      .str();
}

/// Find all classifier entries that are "relatives" of the target key.
/// A relative has the same op prefix, same dtype suffix, and the same
/// first 4 characters of the arch portion (e.g. "gfx9" or "gfx1").
/// This mirrors ParamLookupTable::getRelatives().
SmallVector<std::pair<StringRef, const TreeEnsemble *>>
getRelatives(StringRef target) {
  // Key format: op_arch_dtype (e.g. "gemm_gfx1100_i8")
  auto firstSep = target.find('_');
  auto lastSep = target.rfind('_');
  if (firstSep == StringRef::npos || lastSep == firstSep)
    return {};
  StringRef suffix = target.substr(lastSep);           // e.g. "_i8"
  StringRef opPrefix = target.substr(0, firstSep + 1); // e.g. "gemm_"
  StringRef arch = target.substr(firstSep + 1, lastSep - firstSep - 1);
  constexpr size_t archPrefixLen = 4; // "gfx9", "gfx1", etc.
  StringRef archPrefix = arch.substr(0, std::min(archPrefixLen, arch.size()));

  SmallVector<std::pair<StringRef, const TreeEnsemble *>> relatives;
  for (const auto &entry : classifierTable) {
    StringRef candidate = entry.key;
    if (candidate.starts_with(opPrefix) && candidate.ends_with(suffix)) {
      auto candArchStart = candidate.find('_') + 1;
      auto candArchEnd = candidate.rfind('_');
      StringRef candArch =
          candidate.substr(candArchStart, candArchEnd - candArchStart);
      if (candArch.starts_with(archPrefix))
        relatives.push_back({candidate, entry.classifier});
    }
  }
  llvm::sort(relatives,
             [](const auto &a, const auto &b) { return a.first < b.first; });
  return relatives;
}

/// Look up a classifier by key, falling back to the lexicographically
/// closest relative if no exact match exists.
/// This mirrors ParamLookupTable::lookup() + findFallback().
const TreeEnsemble *lookupClassifier(StringRef key) {
  for (const auto &entry : classifierTable) {
    if (entry.key == key)
      return entry.classifier;
  }
  auto relatives = getRelatives(key);
  if (relatives.empty())
    return nullptr;

  auto it = std::lower_bound(
      relatives.begin(), relatives.end(), key,
      [](const auto &pair, StringRef val) { return pair.first < val; });
  if (it == relatives.end())
    return relatives.back().second;
  if (it == relatives.begin())
    return relatives.front().second;
  auto prev = std::prev(it);
  auto mismatchNext = std::mismatch(key.begin(), key.end(), it->first.begin());
  auto mismatchPrev =
      std::mismatch(key.begin(), key.end(), prev->first.begin());
  if (mismatchNext.first < mismatchPrev.first)
    return prev->second;
  return it->second;
}

/// Feature extraction for GemmGemm (attention) params.
/// Uses mPerBlockG0/nPerBlockG0 as the block sizes (matching the Python
/// training which parses the attention perfconfig's mPerBlock/nPerBlock).
SmallVector<float, kNumAccelFeatures>
extractGemmGemmFeatures(const GemmGemmSize &gemmSize, uint32_t numCu,
                        GemmGemmParamsAttr params) {
  float m = static_cast<float>(gemmSize.m);
  float n = static_cast<float>(gemmSize.n);
  float k = static_cast<float>(gemmSize.k);
  float mPB = static_cast<float>(params.getMPerBlockG0());
  float nPB = static_cast<float>(params.getNPerBlockG0());
  float kPB = static_cast<float>(params.getKpackPerBlock());
  float kPack = static_cast<float>(params.getKpack());
  float mPW = static_cast<float>(params.getMPerWave());
  float nPW = static_cast<float>(params.getNPerWave());
  float kEff = kPB * kPack;
  float splitK = static_cast<float>(params.getSplitKFactor());

  float mTiles = std::ceil(m / mPB);
  float nTiles = std::ceil(n / nPB);
  float kIters = std::ceil(k / kEff);
  float numWaves = (mPB / mPW) * (nPB / nPW);
  float gridSize = mTiles * nTiles * splitK;

  float numCUf = static_cast<float>(numCu);

  float mPadded = mTiles * mPB;
  float nPadded = nTiles * nPB;
  float kPadded = kIters * kEff;
  float originalVolume = m * n * k;
  float paddedVolume = mPadded * nPadded * kPadded;

  SmallVector<float, kNumAccelFeatures> features(kNumAccelFeatures);
  features[kFeatGridSizePerCU] = gridSize / std::max(numCUf, 1.0f);
  features[kFeatMPadFrac] =
      (mPB > 0) ? static_cast<float>(
                      (static_cast<int64_t>(mPB) -
                       math_util::mod_1_to_n(gemmSize.m, (int64_t)mPB)) %
                      static_cast<int64_t>(mPB)) /
                      mPB
                : 0.0f;
  features[kFeatNPadFrac] =
      (nPB > 0) ? static_cast<float>(
                      (static_cast<int64_t>(nPB) -
                       math_util::mod_1_to_n(gemmSize.n, (int64_t)nPB)) %
                      static_cast<int64_t>(nPB)) /
                      nPB
                : 0.0f;
  features[kFeatKPadFrac] =
      (kEff > 0) ? static_cast<float>(
                       (static_cast<int64_t>(kEff) -
                        math_util::mod_1_to_n(gemmSize.k, (int64_t)kEff)) %
                       static_cast<int64_t>(kEff)) /
                       kEff
                 : 0.0f;
  features[kFeatPaddingOverhead] =
      (originalVolume > 0) ? (paddedVolume / originalVolume) - 1.0f : 0.0f;
  features[kFeatNumWaves] = numWaves;
  features[kFeatKIters] = kIters;
  features[kFeatLdsElements] = (mPB + nPB) * kPB * kPack;
  float mnPerXdl = static_cast<float>(params.getMnPerXdl());
  features[kFeatKPerBlockPerMnPerXdl] = (mnPerXdl > 0) ? kEff / mnPerXdl : 0.0f;
  features[kFeatOutputSwizzle] = static_cast<float>(params.getOutputSwizzle());
  features[kFeatWavesPerEU] = static_cast<float>(params.getWavesPerEU());
  features[kFeatGridGroupSize] = 0.0f; // not in attention format
  features[kFeatScheduleVersion] =
      static_cast<float>(params.getScheduleVersion());
  features[kFeatTileReuse] = (mPB + nPB > 0) ? (mPB * nPB) / (mPB + nPB) : 0.0f;
  float denom = m * k + k * n + m * n;
  features[kFeatProblemAI] = (denom > 0) ? (m * n * k) / denom : 0.0f;
  return features;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::optional<float>
mlir::rock::classifierScoreConfig(const PopulateParamsInfo &info,
                                  AccelGemmParamsAttr params) {
  std::string key =
      getClassifierKey(info.kernelType, info.arch, info.gemmAType);
  const TreeEnsemble *clf = lookupClassifier(key);
  if (!clf)
    return std::nullopt;
  auto features = extractAccelFeatures(info, params);
  return ensembleScore(*clf, features);
}

std::optional<float>
mlir::rock::classifierScoreConfig(KernelType kernelType, StringRef arch,
                                  Type dataType, const GemmGemmSize &gemmSize,
                                  uint32_t numCu, GemmGemmParamsAttr params) {
  std::string key = getClassifierKey(kernelType, arch, dataType);
  const TreeEnsemble *clf = lookupClassifier(key);
  if (!clf)
    return std::nullopt;
  auto features = extractGemmGemmFeatures(gemmSize, numCu, params);
  return ensembleScore(*clf, features);
}
