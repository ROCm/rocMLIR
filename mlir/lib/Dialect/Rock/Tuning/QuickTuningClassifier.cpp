//===- QuickTuningClassifier.cpp - ML-based tuning config filter ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningClassifier.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cmath>

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

/// For XGBoost binary classification the leaf values are raw log-odds scores.
/// The prediction is: accept if (baseScore + sum_of_trees) >=
/// decisionThreshold. With baseScore=0 and decisionThreshold=0 this is
/// equivalent to sigmoid(sum) >= 0.5.
bool ensemblePredict(const TreeEnsemble &forest, ArrayRef<float> features) {
  float sum = forest.baseScore;
  for (int16_t i = 0; i < forest.numTrees; ++i)
    sum += predictTree(forest.trees[i], features);
  return sum >= forest.decisionThreshold;
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
  return (Twine(kernelTypeToOpStr(kernelType)) + "_" + arch + "_" + dtype)
      .str();
}

const TreeEnsemble *lookupClassifier(StringRef key) {
  for (const auto &entry : classifierTable) {
    if (entry.key == key)
      return entry.classifier;
  }
  return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

bool mlir::rock::classifierAcceptsConfig(const PopulateParamsInfo &info,
                                         AccelGemmParamsAttr params) {
  std::string key =
      getClassifierKey(info.kernelType, info.arch, info.gemmAType);
  const TreeEnsemble *clf = lookupClassifier(key);
  if (!clf)
    return true;
  auto features = extractAccelFeatures(info, params);
  return ensemblePredict(*clf, features);
}
