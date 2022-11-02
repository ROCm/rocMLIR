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
        WalkResult::interrupt();
      });

  return newSpace;
}

bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto ctx = op.getContext();
        StringAttr attr =
            StringAttr::get(ctx, paramEntry->param..getPerfConfigStr());
        op->setAttr("perf_config", attr);
        WalkResult::interrupt();
      });
  return setPrimary.wasInterrupted();
}

} // namespace rock
} // namespace mlir
