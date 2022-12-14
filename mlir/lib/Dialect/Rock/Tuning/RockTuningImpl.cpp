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

namespace mlir {
namespace rock {

// Brute-force search in incremental order
void createGemmTuningRangeBF(struct TunableParams *newSpace,
                             RockGemmWrapperInterface gemmOp) {

  // blockSize M/block N/block K/block M/thread N/thread
  const std::vector<std::vector<uint32_t>> ValidRangeGeneralGemmParams = {
      {64, 128, 256}, {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> ValidRangeXdlopsGemmParams = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {1, 2, 4, 8},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16, 32, 64, 128},
      {1, 4, 8},
      {0, 1}};

  // M/block N/block K/block M/wave N/wave kPack aCopyMore/forceUnroll
  const std::vector<std::vector<uint32_t>> ValidRangeXdlopsGemmParamsI8 = {
      {4, 8, 16, 32, 64, 128, 256},
      {16, 32, 64, 128, 256},
      {8, 16, 32},
      {4, 8, 16, 32, 64, 128},
      {4, 8, 16, 32, 64, 128},
      {1, 4, 8},
      {0, 1}};

  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    // XDLOPS
    const std::vector<std::vector<uint32_t>> &xdlopsParams =
        gemmOp.getInputType().isInteger(8) ? ValidRangeXdlopsGemmParamsI8
                                           : ValidRangeXdlopsGemmParams;
    for (uint32_t gemmMPerBlock : xdlopsParams[0]) {
      for (uint32_t gemmNPerBlock : xdlopsParams[1]) {
        for (uint32_t gemmKPerBlock : xdlopsParams[2]) {
          for (uint32_t gemmMPerWave : xdlopsParams[3]) {
            for (uint32_t gemmNPerWave : xdlopsParams[4]) {
              for (uint32_t gemmKPack : xdlopsParams[5]) {
                for (uint32_t forceUnroll : xdlopsParams[6]) {
                  XdlopsGemmParamsAttr gemmParams =
                      b.getAttr<XdlopsGemmParamsAttr>(
                          gemmKPerBlock, gemmMPerBlock, gemmNPerBlock,
                          gemmKPack, gemmMPerWave, gemmNPerWave, forceUnroll);
                  newSpace->tuningRange.push_back(
                      gemmParams.cast<RockTuningParamAttrInterface>());
                }
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

} // namespace rock
} // namespace mlir
