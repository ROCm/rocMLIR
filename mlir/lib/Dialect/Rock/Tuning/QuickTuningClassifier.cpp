//===- QuickTuningClassifier.cpp - XGBoost-based perfconfig ranking -------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/QuickTuningClassifier.h"
#include "mlir/Dialect/Rock/IR/GemmGemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <xgboost/c_api.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <string>

#define DEBUG_TYPE "rock-quick-tune-classifier"

using namespace mlir;
using namespace mlir::rock;

unsigned QuickTuningClassifier::getTopN() {
  if (const char *env = std::getenv("ROCMLIR_QUICK_TUNE_TOP_N")) {
    int val = std::atoi(env);
    if (val >= 0)
      return static_cast<unsigned>(val);
  }
  return 30;
}

// ---------------------------------------------------------------------------
// Model cache
// ---------------------------------------------------------------------------

namespace {

std::mutex &getModelCacheMutex() {
  static std::mutex mu;
  return mu;
}

std::map<std::string, BoosterHandle> &getModelCache() {
  static std::map<std::string, BoosterHandle> cache;
  return cache;
}

std::string getExecutableDir() {
  llvm::SmallString<256> realPath;
  auto ec = llvm::sys::fs::real_path("/proc/self/exe", realPath);
  if (ec)
    llvm::report_fatal_error(
        llvm::Twine("QuickTuningClassifier: cannot resolve /proc/self/exe: ") +
        ec.message());
  llvm::sys::path::remove_filename(realPath);
  return std::string(realPath);
}

BoosterHandle loadModel(const std::string &key) {
  std::lock_guard<std::mutex> lock(getModelCacheMutex());
  auto &cache = getModelCache();
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;

  std::string dir = getExecutableDir();
  std::string path = dir + "/models/" + key + ".ubj";

  if (!llvm::sys::fs::exists(path))
    llvm::report_fatal_error(
        llvm::Twine("QuickTuningClassifier: model file not found: ") + path);

  BoosterHandle booster = nullptr;
  if (XGBoosterCreate(nullptr, 0, &booster) != 0)
    llvm::report_fatal_error(
        llvm::Twine("QuickTuningClassifier: XGBoosterCreate failed for key '") +
        key + "'");
  if (XGBoosterLoadModel(booster, path.c_str()) != 0)
    llvm::report_fatal_error(
        llvm::Twine("QuickTuningClassifier: XGBoosterLoadModel failed for ") +
        path);

  LLVM_DEBUG(llvm::dbgs() << "QuickTuningClassifier: loaded model " << path
                          << "\n");
  cache[key] = booster;
  return booster;
}

// ---------------------------------------------------------------------------
// Model key construction (mirrors ParamLookupTable::makeKey)
// ---------------------------------------------------------------------------

std::string normalizeArch(StringRef arch) {
  auto gfxPos = arch.find("gfx");
  if (gfxPos == StringRef::npos)
    llvm::report_fatal_error(
        llvm::Twine("QuickTuningClassifier: invalid architecture string: ") +
        arch);
  auto remaining = arch.substr(gfxPos);
  auto endPos =
      remaining.find_if_not([](char c) { return llvm::isAlnum(c); }, 3);
  return std::string(remaining.substr(0, endPos));
}

std::string kernelTypeStr(KernelType kt) {
  switch (kt) {
  case KernelType::ConvBwdData:
  case KernelType::ConvBwdWeight:
    return stringifyEnum(KernelType::Conv).lower();
  default:
    return stringifyEnum(kt).lower();
  }
}

std::string dataTypeStr(Type dataType) {
  if (dataType.isBF16())
    return "bf16";
  if (dataType.isFloat()) {
    unsigned bw = dataType.getIntOrFloatBitWidth();
    switch (bw) {
    case 4:
    case 8:
      return "fp" + std::to_string(bw);
    case 16:
    case 32:
      return "f" + std::to_string(bw);
    default:
      llvm::report_fatal_error("QuickTuningClassifier: unsupported float "
                               "bitwidth: " +
                               llvm::Twine(bw));
    }
  }
  if (dataType.isInteger()) {
    unsigned bw = dataType.getIntOrFloatBitWidth();
    if (bw == 8)
      return "i8";
    llvm::report_fatal_error("QuickTuningClassifier: unsupported integer "
                             "bitwidth: " +
                             llvm::Twine(bw));
  }
  llvm::report_fatal_error("QuickTuningClassifier: unsupported data type");
}

std::string makeModelKey(StringRef arch, KernelType kt, Type dataTypeA) {
  return normalizeArch(arch) + "_" + kernelTypeStr(kt) + "_" +
         dataTypeStr(dataTypeA);
}

// ---------------------------------------------------------------------------
// Feature extraction: raw problem dims + raw perfconfig params.
// Each op type uses its own problem features.  Must match the Python
// feature builders in trainQuickTuneClassifier.py exactly.
// ---------------------------------------------------------------------------

/// Push raw conv problem dimensions (14 features) matching Python
/// _conv_problem_features: N,C,H,W,K,Y,X,padH,padW,strH,strW,dilH,dilW,dir.
void pushConvProblem(const PopulateParamsInfo &info,
                     llvm::SmallVectorImpl<float> &feats) {
  const auto &cm = *info.convMeta;
  feats.push_back(static_cast<float>(cm.batchN));
  feats.push_back(static_cast<float>(cm.cChannels));
  feats.push_back(static_cast<float>(cm.inH));
  feats.push_back(static_cast<float>(cm.inW));
  feats.push_back(static_cast<float>(cm.kChannels));
  feats.push_back(static_cast<float>(cm.filterH));
  feats.push_back(static_cast<float>(cm.filterW));
  feats.push_back(static_cast<float>(cm.padH));
  feats.push_back(static_cast<float>(cm.padW));
  feats.push_back(static_cast<float>(cm.strideH));
  feats.push_back(static_cast<float>(cm.strideW));
  feats.push_back(static_cast<float>(cm.dilH));
  feats.push_back(static_cast<float>(cm.dilW));
  float dir = 0.0f; // fwd
  if (info.kernelType == KernelType::ConvBwdData)
    dir = 1.0f;
  else if (info.kernelType == KernelType::ConvBwdWeight)
    dir = 2.0f;
  feats.push_back(dir);
}

/// Push raw GEMM problem dimensions (4 features): G, M, N, K.
void pushGemmProblem(const PopulateParamsInfo &info,
                     llvm::SmallVectorImpl<float> &feats) {
  const auto &gs = info.gemmSize;
  feats.push_back(static_cast<float>(gs.g));
  feats.push_back(static_cast<float>(gs.m));
  feats.push_back(static_cast<float>(gs.n));
  feats.push_back(static_cast<float>(gs.k));
}

void buildAccelFeatures(const PopulateParamsInfo &info, AccelGemmParamsAttr p,
                        llvm::SmallVectorImpl<float> &feats) {
  if (info.convMeta.has_value())
    pushConvProblem(info, feats);
  else
    pushGemmProblem(info, feats);

  feats.push_back(static_cast<float>(p.getMPerBlock()));
  feats.push_back(static_cast<float>(p.getNPerBlock()));
  feats.push_back(static_cast<float>(p.getKpackPerBlock()));
  feats.push_back(static_cast<float>(p.getMPerWave()));
  feats.push_back(static_cast<float>(p.getNPerWave()));
  feats.push_back(static_cast<float>(p.getMnPerXdl()));
  feats.push_back(static_cast<float>(p.getKpack()));
  feats.push_back(static_cast<float>(p.getForceUnroll()));
  feats.push_back(static_cast<float>(p.getScheduleVersion()));
  feats.push_back(static_cast<float>(p.getOutputSwizzle()));
  feats.push_back(static_cast<float>(p.getWavesPerEU()));
  feats.push_back(static_cast<float>(p.getGridGroupSize()));
  feats.push_back(static_cast<float>(p.getSplitKFactor()));
}

void buildGeneralFeatures(const PopulateParamsInfo &info,
                          GeneralGemmParamsAttr p,
                          llvm::SmallVectorImpl<float> &feats) {
  if (info.convMeta.has_value())
    pushConvProblem(info, feats);
  else
    pushGemmProblem(info, feats);

  feats.push_back(static_cast<float>(p.getBlockSize()));
  feats.push_back(static_cast<float>(p.getKPerBlock()));
  feats.push_back(static_cast<float>(p.getMPerBlock()));
  feats.push_back(static_cast<float>(p.getNPerBlock()));
  feats.push_back(static_cast<float>(p.getKPerThread()));
  feats.push_back(static_cast<float>(p.getMPerThread()));
  feats.push_back(static_cast<float>(p.getNPerThread()));
  feats.push_back(static_cast<float>(p.getKpack()));
  feats.push_back(static_cast<float>(p.getSplitKFactor()));
  feats.push_back(static_cast<float>(p.getScheduleVersion()));
  feats.push_back(static_cast<float>(p.getOutputSwizzle()));
}

void buildGemmGemmFeatures(GemmGemmSize gs, GemmGemmParamsAttr p,
                           llvm::SmallVectorImpl<float> &feats) {
  feats.push_back(static_cast<float>(gs.g));
  feats.push_back(static_cast<float>(gs.m));
  feats.push_back(static_cast<float>(gs.n));
  feats.push_back(static_cast<float>(gs.k));

  feats.push_back(static_cast<float>(p.getMPerBlockG0()));
  feats.push_back(static_cast<float>(p.getMPerBlockG1()));
  feats.push_back(static_cast<float>(p.getNPerBlockG0()));
  feats.push_back(static_cast<float>(p.getKpackPerBlock()));
  feats.push_back(static_cast<float>(p.getMPerWave()));
  feats.push_back(static_cast<float>(p.getNPerWave()));
  feats.push_back(static_cast<float>(p.getMnPerXdl()));
  feats.push_back(static_cast<float>(p.getKpack()));
  feats.push_back(static_cast<float>(p.getSplitKFactor()));
  feats.push_back(static_cast<float>(p.getScheduleVersion()));
  feats.push_back(static_cast<float>(p.getOutputSwizzle()));
  feats.push_back(static_cast<float>(p.getWavesPerEU()));
  feats.push_back(static_cast<float>(p.getForceUnroll()));
}

// ---------------------------------------------------------------------------
// Prediction + top-N selection
// ---------------------------------------------------------------------------

template <typename ParamAttr, typename FeatFn>
std::vector<ParamAttr>
predictAndSelectTopN(BoosterHandle booster, unsigned topN,
                     llvm::ArrayRef<ParamAttr> candidates,
                     FeatFn &&featFn) {
  size_t nCandidates = candidates.size();
  if (nCandidates == 0)
    return {};

  llvm::SmallVector<float> firstRow;
  featFn(candidates[0], firstRow);
  size_t nFeats = firstRow.size();

  std::vector<float> featureMatrix(nCandidates * nFeats);
  std::copy(firstRow.begin(), firstRow.end(), featureMatrix.begin());
  for (size_t i = 1; i < nCandidates; ++i) {
    llvm::SmallVector<float> row;
    featFn(candidates[i], row);
    assert(row.size() == nFeats);
    std::copy(row.begin(), row.end(), featureMatrix.begin() + i * nFeats);
  }

  DMatrixHandle dmat = nullptr;
  if (XGDMatrixCreateFromMat(
          featureMatrix.data(), nCandidates, nFeats,
          /*missing=*/std::numeric_limits<float>::quiet_NaN(), &dmat) != 0)
    llvm::report_fatal_error(
        llvm::Twine("QuickTuningClassifier: XGDMatrixCreateFromMat failed: ") +
        XGBGetLastError());

  bst_ulong outLen = 0;
  const float *outResult = nullptr;
  if (XGBoosterPredict(booster, dmat, 0, 0, 0, &outLen, &outResult) != 0) {
    std::string err = XGBGetLastError();
    XGDMatrixFree(dmat);
    llvm::report_fatal_error(
        llvm::Twine("QuickTuningClassifier: XGBoosterPredict failed (") +
        llvm::Twine(nFeats) + " features, " + llvm::Twine(nCandidates) +
        " candidates): " + err);
  }

  // Build indices sorted by predicted score descending.
  std::vector<size_t> indices(nCandidates);
  std::iota(indices.begin(), indices.end(), 0);
  size_t selectN = std::min(static_cast<size_t>(topN), nCandidates);
  std::partial_sort(
      indices.begin(), indices.begin() + selectN, indices.end(),
      [&](size_t a, size_t b) { return outResult[a] > outResult[b]; });

  std::vector<ParamAttr> result;
  result.reserve(selectN);
  for (size_t i = 0; i < selectN; ++i)
    result.push_back(candidates[indices[i]]);

  XGDMatrixFree(dmat);
  return result;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API -- Accel (XDL / WMMA)
// ---------------------------------------------------------------------------

std::vector<AccelGemmParamsAttr>
QuickTuningClassifier::filterTopN(const PopulateParamsInfo &info,
                                  llvm::ArrayRef<AccelGemmParamsAttr> cands) {
  unsigned topN = getTopN();
  if (topN == 0 || cands.size() <= topN) {
    LLVM_DEBUG(llvm::dbgs()
               << "QuickTuningClassifier[Accel]: topN=" << topN
               << ", candidates=" << cands.size() << ", skipping filter\n");
    return std::vector<AccelGemmParamsAttr>(cands.begin(), cands.end());
  }

  std::string key = makeModelKey(info.arch, info.kernelType, info.gemmAType);
  BoosterHandle booster = loadModel(key);

  LLVM_DEBUG(llvm::dbgs() << "QuickTuningClassifier[Accel]: filtering "
                          << cands.size() << " -> " << topN << " using model '"
                          << key << "'\n");
  return predictAndSelectTopN<AccelGemmParamsAttr>(
      booster, topN, cands,
      [&](AccelGemmParamsAttr p, llvm::SmallVectorImpl<float> &f) {
        buildAccelFeatures(info, p, f);
      });
}

// ---------------------------------------------------------------------------
// Public API -- NonAccel
// ---------------------------------------------------------------------------

std::vector<GeneralGemmParamsAttr>
QuickTuningClassifier::filterTopN(const PopulateParamsInfo &info,
                                  llvm::ArrayRef<GeneralGemmParamsAttr> cands) {
  unsigned topN = getTopN();
  if (topN == 0 || cands.size() <= topN) {
    LLVM_DEBUG(llvm::dbgs()
               << "QuickTuningClassifier[General]: topN=" << topN
               << ", candidates=" << cands.size() << ", skipping filter\n");
    return std::vector<GeneralGemmParamsAttr>(cands.begin(), cands.end());
  }

  std::string key = makeModelKey(info.arch, info.kernelType, info.gemmAType);
  BoosterHandle booster = loadModel(key);

  LLVM_DEBUG(llvm::dbgs() << "QuickTuningClassifier[General]: filtering "
                          << cands.size() << " -> " << topN << " using model '"
                          << key << "'\n");
  return predictAndSelectTopN<GeneralGemmParamsAttr>(
      booster, topN, cands,
      [&](GeneralGemmParamsAttr p, llvm::SmallVectorImpl<float> &f) {
        buildGeneralFeatures(info, p, f);
      });
}

// ---------------------------------------------------------------------------
// Public API -- GemmGemm (attention)
// ---------------------------------------------------------------------------

std::vector<GemmGemmParamsAttr>
QuickTuningClassifier::filterTopN(RockGemmGemmWrapperInterface op,
                                  llvm::ArrayRef<GemmGemmParamsAttr> cands) {
  unsigned topN = getTopN();
  if (topN == 0 || cands.size() <= topN) {
    LLVM_DEBUG(llvm::dbgs()
               << "QuickTuningClassifier[GemmGemm]: topN=" << topN
               << ", candidates=" << cands.size() << ", skipping filter\n");
    return std::vector<GemmGemmParamsAttr>(cands.begin(), cands.end());
  }

  StringAttr archAttr = rock::getArchValue(op);
  std::string key = makeModelKey(archAttr, op.getKernelType(), op.getAType());
  BoosterHandle booster = loadModel(key);

  LLVM_DEBUG(llvm::dbgs() << "QuickTuningClassifier[GemmGemm]: filtering "
                          << cands.size() << " -> " << topN << " using model '"
                          << key << "'\n");
  GemmGemmSize gs = op.getGemmGemmSize();
  return predictAndSelectTopN<GemmGemmParamsAttr>(
      booster, topN, cands,
      [&](GemmGemmParamsAttr p, llvm::SmallVectorImpl<float> &f) {
        buildGemmGemmFeatures(gs, p, f);
      });
}
