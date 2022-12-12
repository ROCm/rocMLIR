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

#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "llvm/ADT/Hashing.h"

namespace mlir {
namespace rock {

// Brute-force search in incremental order
void createGemmTuningRangeBF(struct TunableParams *newSpace,
                             RockGemmWrapperInterface gemmOp) {

  // blockSize M/block N/block K/block M/thread N/thread
  const std::vector<std::vector<uint32_t>> ValidRangeGeneralGemmParams = {
      {64, 128, 256}, {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore bCopyMore
  const std::vector<std::vector<uint32_t>> ValidRangeXdlopsGemmParams = {
      {4, 8, 16, 32, 64, 128},
      {16, 32, 64, 128},
      {16, 32, 64, 128},
      {16, 32, 64},
      {16, 32, 64},
      {1, 4}};

  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    // XDLOPS
    for (uint32_t gemmMPerBlock : ValidRangeXdlopsGemmParams[0]) {
      for (uint32_t gemmNPerBlock : ValidRangeXdlopsGemmParams[1]) {
        for (uint32_t gemmKPerBlock : ValidRangeXdlopsGemmParams[2]) {
          for (uint32_t gemmMPerWave : ValidRangeXdlopsGemmParams[3]) {
            for (uint32_t gemmNPerWave : ValidRangeXdlopsGemmParams[4]) {
              for (uint32_t gemmKPack : ValidRangeXdlopsGemmParams[5]) {
                XdlopsGemmParamsAttr gemmParams =
                    b.getAttr<XdlopsGemmParamsAttr>(
                        gemmKPerBlock, gemmMPerBlock, gemmNPerBlock, gemmKPack,
                        gemmMPerWave, gemmNPerWave);
                newSpace->tuningRange.push_back(
                    gemmParams.cast<RockTuningParamAttrInterface>());
              }
            }
          }
        }
      }
    }
  } else {
    // Non-XDLOPS
    for (uint32_t blockSize : ValidRangeGeneralGemmParams[0]) {
      for (uint32_t gemmMPerBlock : ValidRangeGeneralGemmParams[1]) {
        for (uint32_t gemmNPerBlock : ValidRangeGeneralGemmParams[2]) {
          for (uint32_t gemmKPerBlock : ValidRangeGeneralGemmParams[3]) {
            for (uint32_t gemmMPerThread : ValidRangeGeneralGemmParams[4]) {
              for (uint32_t gemmNPerThread : ValidRangeGeneralGemmParams[5]) {

                GeneralGemmParamsAttr gemmParams =
                    b.getAttr<GeneralGemmParamsAttr>(
                        blockSize, gemmKPerBlock, gemmMPerBlock, gemmNPerBlock,
                        /*kPerThread=*/1, gemmMPerThread, gemmNPerThread,
                        /*kpack=*/1);
                newSpace->tuningRange.push_back(
                    gemmParams.cast<RockTuningParamAttrInterface>());
              }
            }
          }
        }
      }
    }
  }

  newSpace->numHeuristicQuick = 0;
}

TunableParams *createTunableParamSpace(ModuleOp &mod) {
  struct TunableParams *newSpace;
  newSpace = new TunableParams();

  // create range and heuristic
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        createGemmTuningRangeBF(newSpace, op);
        newSpace->primaryOpType = op.getKernelType();
        return WalkResult::interrupt();
      });
  if (!findPrimary.wasInterrupted()) {
    delete newSpace;
  }
  return newSpace;
}

bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto ctx = op.getContext();
        std::string perfConfig;
        paramEntry->param.getPerfConfigStr(perfConfig);
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  return setPrimary.wasInterrupted();
}

bool tuningSetStr(ModuleOp &mod, std::string perfConfig) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto ctx = op.getContext();
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  return setPrimary.wasInterrupted();
}

TuningTable *tuningTableCreate() {
  struct TuningTable *newTable = new TuningTable();
  return newTable;
}

// Suppose to return the structure of the given problem to tune, currently
// combines the string representation of the selected field of the primary
// operation. String format of the problem will not be required by the DB,
// since it can store each field separately.
//
// Current internal model of the tuning problem is assuming :
// tuning problem struct =
// primaryOp-{opType, arch}-+-conv_case-{inShapedType, filterShapedType,
//                   |          adding,Stride, Dilation, in/fil/out_layout}
//                   `---- gemm_case-{dataType, m, n, k}
// *Only take one conv&gemm operation into account
//
std::string getTuningProblemStr(ModuleOp &mod) {
  rock::RockGemmWrapperInterface gemmIF;
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        gemmIF = op;
        return WalkResult::interrupt();
      });
  if (!findPrimary.wasInterrupted())
    return std::string();
  std::string problemStr;
  char sep = '#';
  llvm::raw_string_ostream problemOS(problemStr);
  KernelType opType = gemmIF.getKernelType();
  Operation *gemmOp = gemmIF.getOperation();
  problemOS << gemmOp->getName().getStringRef() << sep;
  // conv case
  if (opType == KernelType::Conv2D) {
    RockConvInterface convIF = dyn_cast<RockConvInterface>(gemmOp);
    convIF.getFilter().getType().print(problemOS);
    problemOS << sep;
    convIF.getInput().getType().print(problemOS);
    problemOS << sep;
    problemOS << convIF.getPadding() << sep;
    problemOS << convIF.getStrides() << sep;
    problemOS << convIF.getDilations() << sep;

    // Layout information
    auto filterLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr =
        gemmOp->template getAttrOfType<ArrayAttr>("output_layout");

    unsigned size = filterLayoutAttr.size();
    for (unsigned i = 0; i < size; ++i) {
      auto filterAttr =
          filterLayoutAttr.getValue()[i].template cast<StringAttr>();
      problemOS << filterAttr.getValue();
    }
    for (unsigned i = 0; i < size; ++i) {
      auto inputAttr =
          inputLayoutAttr.getValue()[i].template cast<StringAttr>();
      problemOS << inputAttr.getValue();
    }
    for (unsigned i = 0; i < size; ++i) {
      auto outputAttr =
          outputLayoutAttr.getValue()[i].template cast<StringAttr>();
      problemOS << outputAttr.getValue();
    }
  }
  // gemm case
  else if (opType == KernelType::Gemm) {
    gemmIF.getInputType().print(problemOS);
    problemOS << sep;
    problemOS << gemmIF.getGemmSize().m << sep << gemmIF.getGemmSize().k << sep
              << gemmIF.getGemmSize().n << sep;
  }
  problemOS << sep << gemmIF.getArch() << sep;
  return problemStr;
}

bool tuningTableUpdate(TuningTable *perfTable, ModuleOp &mod,
                       std::string perfConfig, float time) {
  std::string problem = getTuningProblemStr(mod);
  if (problem.empty())
    return false;
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

std::string tuningTableLookup(TuningTable *perfTable, ModuleOp &mod) {
  std::string problem = getTuningProblemStr(mod);
  if (problem.empty())
    return std::string();
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    return entry.first;
  }
  return std::string();
}

} // namespace rock
} // namespace mlir
