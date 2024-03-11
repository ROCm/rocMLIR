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

#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "llvm/ADT/SmallString.h"
#include <algorithm>
#include <vector>

namespace mlir {
namespace rock {

// The full space is a brute-force search for attention kernels
void createAttnTuningRangeBF(TuningParamSet *newSpace, AttentionOp attnOp,
                             TuningParamSetKind kind) {
  static const std::vector<std::vector<uint32_t>> validRangeAccelGemmParams = {
      {32, 64, 128, 256}, {32, 64, 128, 256}, {8, 16, 32, 64},
      {32, 64, 128, 256}, {32, 64, 128, 256}, {4, 8, 16}};
  constexpr uint64_t splitKFactor = 1;
  constexpr uint32_t forceUnroll = 1;
  OpBuilder b(attnOp.getContext());
  for (uint32_t gemmMPerBlock : validRangeAccelGemmParams[0]) {
    for (uint32_t gemmNPerBlock : validRangeAccelGemmParams[1]) {
      for (uint32_t gemmKPerBlock : validRangeAccelGemmParams[2]) {
        for (uint32_t gemmMPerWave : validRangeAccelGemmParams[3]) {
          for (uint32_t gemmNPerWave : validRangeAccelGemmParams[4]) {
            for (uint32_t gemmKPack : validRangeAccelGemmParams[5]) {
              if (gemmMPerBlock >= gemmMPerWave &&
                  gemmNPerBlock >= gemmNPerWave) {
                InitParamsAccel gemmParams(
                    gemmMPerBlock, gemmNPerBlock, gemmKPerBlock, gemmMPerWave,
                    gemmNPerWave, gemmKPack, splitKFactor, forceUnroll, true);
                GemmFeatures features = attnOp.getFeatures();
                auto populateParamsAccelPtr =
                    PopulateParamsAccel::select(features);
                Attribute params =
                    populateParamsAccelPtr->getGemmParamsAttr(b, gemmParams);
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

SmallVector<int64_t>
computeOptimalSplitKFactors(RockGemmWrapperInterface gemmOp,
                            int32_t gemmMPerBlock, int32_t gemmNPerBlock,
                            int32_t gemmKPerBlock, int32_t kPack) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  SmallVector<int64_t> splitKValues{1};
  if (!info.gemmAType.isF32()) {
    return splitKValues;
  }

  if (!gemmOp.getNumCU().has_value()) {
    return splitKValues;
  }
  const double numCUs = gemmOp.getNumCU().value();

  const InitParams params{gemmMPerBlock, gemmNPerBlock, gemmKPerBlock};
  const GemmSize gemmSize =
      calculatePaddedGemmSize(params, info.gemmSize, kPack);
  const double numMTiles = gemmSize.m / gemmMPerBlock;
  const double numNTiles = gemmSize.n / gemmNPerBlock;

  auto computeImbalance = [&](int32_t splitKFactor) -> double {
    const double totalNumWorkGroups = numMTiles * numNTiles * splitKFactor;
    const double maxWorkGroupsPerCU = std::ceil(totalNumWorkGroups / numCUs);
    // imbalances = max. CU work / average work per CU
    return (maxWorkGroupsPerCU * numCUs) / totalNumWorkGroups;
  };

  const auto dataPrallelGemmImbalabce = computeImbalance(1);
  constexpr double imbalaceThreshold{1.20};
  if (dataPrallelGemmImbalabce < imbalaceThreshold) {
    return splitKValues;
  }

  struct LocalData {
    int64_t splitKValue{};
    double workImbalance{};
  };
  SmallVector<LocalData> factors{};
  constexpr double minGain{1.30};
  // There are cases where perfect load balancing can be achieved with very
  // high splitK values. However, experiments show that performance
  // can considerably drop in such cases. Currently, we limit the `upperBound`
  // on purpose because the current heuristics does not consider the overheads
  // resulting from reducing partial solution along the split dimension.
  // This needs to be improved in the future.
  constexpr int32_t upperBound{32};
  for (int32_t splitKFactor = 2; splitKFactor < upperBound; ++splitKFactor) {
    const double imbalance = computeImbalance(splitKFactor);
    const auto gain = dataPrallelGemmImbalabce / imbalance;
    if (gain > minGain) {
      factors.emplace_back(LocalData{splitKFactor, imbalance});
    }
  }

  if (factors.empty()) {
    return splitKValues;
  }

  std::sort(factors.begin(), factors.end(), [](LocalData &a, LocalData &b) {
    return a.workImbalance < b.workImbalance;
  });

  size_t maxVariants{6};
  maxVariants = factors.size() > maxVariants ? maxVariants : factors.size();
  std::for_each(
      factors.begin(), factors.begin() + maxVariants,
      [&](LocalData &item) { splitKValues.push_back(item.splitKValue); });

  return splitKValues;
}

// The full space is a brute-force search starting with the configs that have
// the smallest parameters. This filters out perf configs that are
// known to be impossible during tthe AffixTuningParams check.
// If `kind` is Full, also filters out unlikely-to-be-good configurations.
void createGemmTuningRangeBF(TuningParamSet *newSpace,
                             RockGemmWrapperInterface gemmOp,
                             TuningParamSetKind kind) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);

  // blockSize M/block N/block K/block M/thread N/thread
  const std::vector<std::vector<uint32_t>> validRangeGeneralGemmParams = {
      {64, 128, 256}, {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> validRangeAccelGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16, 32, 64, 128},
      {1, 4, 8},
      {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>>
      validRangeAccelGemmParams8BitReduction = {{4, 8, 16, 32, 64, 128, 256},
                                                {16, 32, 64, 128, 256},
                                                {4, 8, 16, 32},
                                                {4, 8, 16, 32, 64, 128},
                                                {4, 8, 16, 32, 64, 128},
                                                {1, 4, 8, 16},
                                                {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> validRangeWmmaGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16},
      {0, 1}};

  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    PopulateParamsXDL tuningInfo;
    // XDLOPS
    Type inTypeA = gemmOp.getAType();
    bool is8BitReduction = inTypeA.isInteger(8) || inTypeA.isFloat8E5M2FNUZ() ||
                           inTypeA.isFloat8E4M3FNUZ();
    const std::vector<std::vector<uint32_t>> &xdlopsParams =
        is8BitReduction ? validRangeAccelGemmParams8BitReduction
                        : validRangeAccelGemmParams;
    for (uint32_t gemmMPerBlock : xdlopsParams[0]) {
      for (uint32_t gemmNPerBlock : xdlopsParams[1]) {
        for (uint32_t gemmKPerBlock : xdlopsParams[2]) {
          for (uint32_t gemmMPerWave : xdlopsParams[3]) {
            for (uint32_t gemmNPerWave : xdlopsParams[4]) {
              for (uint32_t gemmKPack : xdlopsParams[5]) {
                auto optimalSplitKFactors = computeOptimalSplitKFactors(
                    gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                    gemmKPack);
                for (int64_t splitKFactor : optimalSplitKFactors) {
                  for (uint32_t forceUnroll : xdlopsParams[6]) {
                    InitParamsAccel gemmParams(gemmMPerBlock, gemmNPerBlock,
                                               gemmKPerBlock, gemmMPerWave,
                                               gemmNPerWave, gemmKPack,
                                               splitKFactor, forceUnroll, true);
                    if (kind == TuningParamSetKind::Exhaustive ||
                        (succeeded(tuningInfo.paramsProbablyValid(
                             info, gemmParams)) &&
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
                    gemmKPack);
                for (auto splitKFactor : optimalSplitKFactors) {
                  for (uint32_t forceUnroll : wmmaParams[6]) {
                    InitParamsAccel gemmParams(gemmMPerBlock, gemmNPerBlock,
                                               gemmKPerBlock, gemmMPerWave,
                                               gemmNPerWave, gemmKPack,
                                               splitKFactor, forceUnroll, true);
                    if (succeeded(
                            tuningInfo.paramsProbablyValid(info, gemmParams)) &&
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
  } else {
    // Non-XDLOPS
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
                  InitParamsNonAccel gemmParams(
                      blockSize, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock,
                      gemmMPerThread, gemmNPerThread, splitKFactor);
                  if (succeeded(
                          tuningInfo.paramsProbablyValid(info, gemmParams)) &&
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

void createQuickTuningRange(TuningParamSet *newSpace,
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
      if (succeeded(tuningInfo.paramsProbablyValid(info, param)))
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
      if (succeeded(tuningInfo.paramsProbablyValid(info, param)))
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
      if (succeeded(tuningInfo.paramsProbablyValid(info, param)))
        newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
            tuningInfo.getGemmParamsAttr(b, param)));
    }
  }
}

// This is temporary workaround to make MIGraphX integration
// work until the tuning is setup for attention ops properly.
void createTuningRange(TuningParamSet *newSpace, AttentionOp attnOp) {
  OpBuilder b(attnOp.getContext());
  Type elemType = attnOp.getQueries().getType().getElementType();
  StringRef arch = attnOp.getArch();
  GemmFeatures currentFeatures = attnOp.getFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    PopulateParamsXDL tuningInfo;
    // This is hack to obtain the same quick tuning list as if it were a gemm
    // kernel. This should ideally be implemented as an interface fucntion of
    // a rock tunable op to retrieve this range.
    for (InitParamsAccel param : tuningInfo.getTuningParameters(
             rock::KernelType::Gemm, elemType, elemType, arch)) {
      newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
          tuningInfo.getGemmParamsAttr(b, param)));
    }
    // backup universal config that is known to fit in LDS
    newSpace->tuningRange.push_back(
        cast<RockTuningParamAttrInterface>(b.getAttr<XdlopsGemmParamsAttr>(
            /*kpackPerBlock=*/32, /*mPerBlock=*/32,
            /*nPerBlock=*/32, /*kpack=*/1,
            /*mPerWave=*/32, /*nPerWave=*/32, /*splitKFactor*/ 1,
            /*forceUnroll=*/true)));

    // add performant configs for tier1
    newSpace->tuningRange.push_back(
        cast<RockTuningParamAttrInterface>(b.getAttr<XdlopsGemmParamsAttr>(
            /*kpackPerBlock=*/8, /*mPerBlock=*/64,
            /*nPerBlock=*/128, /*kpack=*/8,
            /*mPerWave=*/32, /*nPerWave=*/32, /*splitKFactor*/ 1,
            /*forceUnroll=*/true)));
    newSpace->tuningRange.push_back(
        cast<RockTuningParamAttrInterface>(b.getAttr<XdlopsGemmParamsAttr>(
            /*kpackPerBlock=*/8, /*mPerBlock=*/64,
            /*nPerBlock=*/64, /*kpack=*/8,
            /*mPerWave=*/64, /*nPerWave=*/32, /*splitKFactor*/ 1,
            /*forceUnroll=*/true)));

    // add performant config for triton configs
    newSpace->tuningRange.push_back(
        cast<RockTuningParamAttrInterface>(b.getAttr<XdlopsGemmParamsAttr>(
            /*kpackPerBlock=*/16, /*mPerBlock=*/128,
            /*nPerBlock=*/128, /*kpack=*/8,
            /*mPerWave=*/64, /*nPerWave=*/32, /*splitKFactor*/ 1,
            /*forceUnroll=*/true)));
    newSpace->tuningRange.push_back(
        cast<RockTuningParamAttrInterface>(b.getAttr<XdlopsGemmParamsAttr>(
            /*kpackPerBlock=*/16, /*mPerBlock=*/128,
            /*nPerBlock=*/128, /*kpack=*/8,
            /*mPerWave=*/64, /*nPerWave=*/64, /*splitKFactor*/ 1,
            /*forceUnroll=*/true)));
    newSpace->tuningRange.push_back(
        cast<RockTuningParamAttrInterface>(b.getAttr<XdlopsGemmParamsAttr>(
            /*kpackPerBlock=*/32, /*mPerBlock=*/128,
            /*nPerBlock=*/256, /*kpack=*/4,
            /*mPerWave=*/128, /*nPerWave=*/32, /*splitKFactor*/ 1,
            /*forceUnroll=*/true)));
    newSpace->tuningRange.push_back(
        cast<RockTuningParamAttrInterface>(b.getAttr<XdlopsGemmParamsAttr>(
            /*kpackPerBlock=*/32, /*mPerBlock=*/64,
            /*nPerBlock=*/128, /*kpack=*/4,
            /*mPerWave=*/64, /*nPerWave=*/32, /*splitKFactor*/ 1,
            /*forceUnroll=*/true)));
  } else if (bitEnumContainsAll(currentFeatures, GemmFeatures::wmma)) {
    // Wmma
    PopulateParamsWmma tuningInfo;
    // This is hack to obtain the same quick tuning list as if it were a gemm
    // kernel. This should ideally be implemented as an interface fucntion of
    // a rock tunable op to retrieve this range.
    for (InitParamsAccel param : tuningInfo.getTuningParameters(
             rock::KernelType::Gemm, elemType, elemType, arch)) {
      newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(
          tuningInfo.getGemmParamsAttr(b, param)));
    }
    // backup universal config that is known to fit in LDS
    newSpace->tuningRange.push_back(
        cast<RockTuningParamAttrInterface>(b.getAttr<WmmaGemmParamsAttr>(
            /*kpackPerBlock=*/32, /*mPerBlock=*/32,
            /*nPerBlock=*/32, /*kpack=*/1,
            /*mPerWave=*/32, /*nPerWave=*/32, /*splitKFactor*/ 1,
            /*forceUnroll=*/true)));
  }
  // We only support GPUs with matrix accelerator extentions
}

TuningParamSet *createTunableParamSpace(ModuleOp &mod,
                                        TuningParamSetKind kind) {
  struct TuningParamSet *newSpace;
  newSpace = new TuningParamSet();

  // create range and heuristic
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createGemmTuningRangeBF(newSpace, op, kind);
          break;
        case TuningParamSetKind::Quick:
          createQuickTuningRange(newSpace, op);
        }
        newSpace->primaryOpType = op.getKernelType();
        return WalkResult::interrupt();
      });
  WalkResult findAttention = mod->walk([&](rock::AttentionOp op) -> WalkResult {
    // createTuningRange(newSpace, op);
    switch (kind) {
    case TuningParamSetKind::Full:
    case TuningParamSetKind::Exhaustive:
      createAttnTuningRangeBF(newSpace, op, kind);
      break;
    case TuningParamSetKind::Quick:
      createTuningRange(newSpace, op);
    }
    return WalkResult::interrupt();
  });
  if (!findPrimary.wasInterrupted() && !findAttention.wasInterrupted()) {
    delete newSpace;
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
  WalkResult setAttn = mod->walk([&](rock::AttentionOp op) -> WalkResult {
    auto *ctx = op.getContext();
    SmallString<64> perfConfig;
    paramEntry->param.getPerfConfigStr(perfConfig);
    StringAttr attr = StringAttr::get(ctx, perfConfig);
    op->setAttr("perf_config", attr);
    return WalkResult::interrupt();
  });
  return setPrimary.wasInterrupted() || setAttn.wasInterrupted();
}

bool tuningSetStr(ModuleOp &mod, StringRef perfConfig) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  WalkResult setAttn = mod->walk([&](rock::AttentionOp op) -> WalkResult {
    auto *ctx = op.getContext();
    StringAttr attr = StringAttr::get(ctx, perfConfig);
    op->setAttr("perf_config", attr);
    return WalkResult::interrupt();
  });
  return setPrimary.wasInterrupted() || setAttn.wasInterrupted();
}

TuningTable *tuningTableCreate() {
  struct TuningTable *newTable = new TuningTable();
  return newTable;
}

LogicalResult getTuningProblemStr(rock::AttentionOp attnOp,
                                  SmallVectorImpl<char> &out) {
  int32_t numCU = rock::lookupArchInfo(attnOp.getArch()).minNumCU;
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  int64_t headDimQK;
  int64_t headDimV;
  int64_t seqLenQ;
  int64_t seqLenK;
  llvm::raw_svector_ostream problemOS(out);
  // ARCH string
  problemOS << attnOp.getArch() << tab;
  // Num of Compute Units
  problemOS << numCU << tab;

  TypedValue<ShapedType> queries = attnOp.getQueries();
  TypedValue<ShapedType> keys = attnOp.getKeys();
  TypedValue<ShapedType> values = attnOp.getValues();
  ArrayRef<int64_t> qShape = queries.getType().getShape();
  ArrayRef<int64_t> kShape = keys.getType().getShape();
  ArrayRef<int64_t> vShape = values.getType().getShape();
  int64_t g = qShape[0];

  // TransQ
  problemOS << "-transQ ";
  if (attnOp.getQTransposed()) {
    seqLenQ = qShape[2];
    headDimQK = qShape[1];
    problemOS << "true" << sep;
  } else {
    seqLenQ = qShape[1];
    headDimQK = qShape[2];
    problemOS << "false" << sep;
  }

  // TransK
  problemOS << "-transK ";
  if (attnOp.getKTransposed()) {
    seqLenK = kShape[1];
    problemOS << "true" << sep;
  } else {
    seqLenK = kShape[2];
    problemOS << "false" << sep;
  }

  // TransV
  problemOS << "-transV ";
  if (attnOp.getVTransposed()) {
    headDimV = vShape[1];
    problemOS << "true" << sep;
  } else {
    headDimV = vShape[2];
    problemOS << "false" << sep;
  }

  // TransO
  problemOS << "-transO ";
  if (attnOp.getOTransposed())
    problemOS << "true" << sep;
  else
    problemOS << "false" << sep;

  problemOS << "-g " << g << sep;
  problemOS << "-seq_len_q " << seqLenQ << sep;
  problemOS << "-seq_len_k " << seqLenK << sep;
  problemOS << "-head_dim_qk " << headDimQK;
  problemOS << "-head_dim_v " << headDimV;
  return success();
}

LogicalResult getTuningProblemStr(rock::RockGemmWrapperInterface gemmIF,
                                  SmallVectorImpl<char> &out) {
  int32_t numCU = rock::lookupArchInfo(gemmIF.getArch()).minNumCU;
  if (gemmIF.getNumCU().has_value())
    numCU = gemmIF.getNumCU().value();
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  llvm::raw_svector_ostream problemOS(out);

  KernelType opType = gemmIF.getKernelType();
  Operation *gemmOp = gemmIF.getOperation();

  // ARCH string
  problemOS << gemmIF.getArch() << tab;
  // Num of Compute Units
  problemOS << numCU << tab;

  if (opType == KernelType::Conv2D || opType == KernelType::Conv2DBwdData ||
      opType == KernelType::Conv2DBwdWeight) { // conv cases
    RockConvInterface convIF = dyn_cast<RockConvInterface>(gemmOp);

    ShapedType inType = convIF.getInput().getType();
    ArrayRef<int64_t> inShape = inType.getShape();
    ShapedType filType = convIF.getFilter().getType();
    ArrayRef<int64_t> filShape = filType.getShape();

    // Extract layout information
    auto filterLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("output_layout");

    unsigned size = filterLayoutAttr.size();
    llvm::StringMap<unsigned> fLayoutMap;
    llvm::StringMap<unsigned> iLayoutMap;
    llvm::StringMap<unsigned> oLayoutMap;

    for (unsigned i = 0; i < size; ++i) {
      auto filterAttr =
          filterLayoutAttr.getValue()[i].template cast<StringAttr>();
      fLayoutMap[filterAttr.getValue()] = i;
    }
    for (unsigned i = 0; i < size; ++i) {
      auto inputAttr =
          inputLayoutAttr.getValue()[i].template cast<StringAttr>();
      iLayoutMap[inputAttr.getValue()] = i;
    }
    for (unsigned i = 0; i < size; ++i) {
      auto outputAttr =
          outputLayoutAttr.getValue()[i].template cast<StringAttr>();
      oLayoutMap[outputAttr.getValue()] = i;
    }

    SmallString<5> fLayout("#####");
    SmallString<5> iLayout("#####");
    SmallString<5> oLayout("#####");

    // dimensions need to be mapped 1 to 1.
    fLayout[fLayoutMap["k"]] = 'N';
    fLayout[fLayoutMap["c"]] = 'C';
    fLayout[fLayoutMap["y"]] = 'H';
    fLayout[fLayoutMap["x"]] = 'W';
    fLayout[fLayoutMap["g"]] = 'G';
    iLayout[iLayoutMap["ni"]] = 'N';
    iLayout[iLayoutMap["ci"]] = 'C';
    iLayout[iLayoutMap["hi"]] = 'H';
    iLayout[iLayoutMap["wi"]] = 'W';
    iLayout[iLayoutMap["gi"]] = 'G';
    oLayout[oLayoutMap["no"]] = 'N';
    oLayout[oLayoutMap["ko"]] = 'C';
    oLayout[oLayoutMap["ho"]] = 'H';
    oLayout[oLayoutMap["wo"]] = 'W';
    oLayout[oLayoutMap["go"]] = 'G';

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
    } else if (inElemType.isFloat8E4M3FNUZ() &&
               filElemType.isFloat8E4M3FNUZ()) {
      problemOS << "convfp8_fp8 ";
    } else if (inElemType.isFloat8E4M3FNUZ() &&
               filElemType.isFloat8E5M2FNUZ()) {
      problemOS << "convfp8_bf8 ";
    } else if (inElemType.isFloat8E5M2FNUZ() &&
               filElemType.isFloat8E4M3FNUZ()) {
      problemOS << "convbf8_fp8 ";
    } else if (inElemType.isFloat8E5M2FNUZ() &&
               filElemType.isFloat8E5M2FNUZ()) {
      problemOS << "convbf8_bf8 ";
    } else {
      return failure();
    }

    // OP direction
    switch (opType) {
    case KernelType::Conv2D:
      problemOS << "-F 1" << sep;
      break;
    case KernelType::Conv2DBwdData:
      problemOS << "-F 2" << sep;
      break;
    case KernelType::Conv2DBwdWeight:
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
    problemOS << "-c " << inShape[iLayoutMap["ci"]] << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["hi"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["wi"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["y"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["x"]] << sep;

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
    } else if (elemTypeA.isFloat8E4M3FNUZ() && elemTypeB.isFloat8E4M3FNUZ()) {
      problemOS << "fp8_fp8";
    } else if (elemTypeA.isFloat8E4M3FNUZ() && elemTypeB.isFloat8E5M2FNUZ()) {
      problemOS << "fp8_bf8";
    } else if (elemTypeA.isFloat8E5M2FNUZ() && elemTypeB.isFloat8E4M3FNUZ()) {
      problemOS << "bf8_fp8";
    } else if (elemTypeA.isFloat8E5M2FNUZ() && elemTypeB.isFloat8E5M2FNUZ()) {
      problemOS << "bf8_bf8";
    } else {
      // Unknown data type
      return failure();
    }

    // OUtput datatype
    auto outType = gemmIF.getOutArgument()->get().getType();
    auto elemTypeC = outType.dyn_cast<mlir::MemRefType>().getElementType();
    problemOS << " -out_datatype ";
    if (elemTypeC.isFloat8E4M3FNUZ()) {
      problemOS << "fp8" << sep;
    } else if (elemTypeC.isFloat8E5M2FNUZ()) {
      problemOS << "bf8" << sep;
    } else {
      problemOS << elemTypeC << sep;
    }

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
LogicalResult getTuningProblemStr(ModuleOp &mod, SmallVectorImpl<char> &out) {
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
    rock::AttentionOp attnOp;
    WalkResult findAttention =
        mod->walk([&](rock::AttentionOp op) -> WalkResult {
          attnOp = op;
          return WalkResult::interrupt();
        });
    if (findAttention.wasInterrupted())
      return getTuningProblemStr(attnOp, out);
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

} // namespace rock
} // namespace mlir
