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

  OpBuilder b(gemmOp.getContext());
  GemmFeatures currentFeatures = gemmOp.getGemmFeatures();
  if (bitEnumContainsAll(currentFeatures, GemmFeatures::mfma)) {
    // XDLOPS
    // M/block N/block K/block M/wave N/wave kPack aCopyMore bCopyMore
    const std::vector<std::vector<uint32_t>> tParams = {{4, 8, 16, 32, 64, 128},
                                                        {16, 32, 64, 128},
                                                        {16, 32, 64, 128},
                                                        {16, 32, 64},
                                                        {16, 32, 64},
                                                        {1, 4}};

    for (uint32_t gemmMPerBlock : tParams[0]) {
      for (uint32_t gemmNPerBlock : tParams[1]) {
        for (uint32_t gemmKPerBlock : tParams[2]) {
          for (uint32_t gemmMPerWave : tParams[3]) {
            for (uint32_t gemmNPerWave : tParams[4]) {
              for (uint32_t gemmKPack : tParams[5]) {
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
    // M/block N/block K/block M/thread N/thread
    const std::vector<std::vector<uint32_t>> tParams = {
        {32, 64, 128}, {32, 64, 128}, {4, 8, 16}, {2, 4}, {2, 4}};
    for (uint32_t gemmMPerBlock : tParams[0]) {
      for (uint32_t gemmNPerBlock : tParams[1]) {
        for (uint32_t gemmKPerBlock : tParams[2]) {
          for (uint32_t gemmMPerThread : tParams[3]) {
            for (uint32_t gemmNPerThread : tParams[4]) {
              GeneralGemmParamsAttr gemmParams =
                  b.getAttr<GeneralGemmParamsAttr>(
                      gemmKPerBlock, gemmMPerBlock, gemmNPerBlock,
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

  newSpace->numHeuristicQuick = 0;
}

TunableParams *createTunableParams(ModuleOp &mod) {
  struct TunableParams *newSpace;
  newSpace = new TunableParams();

  // create range and heuristic
  WalkResult findPrimary mod->walk([&](rock::RockGemmWrapperInterface op) {
    if (!WalkResult::interrupt()) {
      createGemmTuningRangeBF(newSpace, op);
      newSpace->primaryOpType = op.getKernelType();
    }
  });

  return findPrimary.wasInterrupted();
}

bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry) {
  WalkResult setPrimary mod->walk([&](rock::RockGemmWrapperInterface op) {
    if (!WalkResult::interrupt()) {
      op->setAttr("perf_config", paramEntry->perfString);
    }
  });
  return setPrimary.wasInterrupted();
}

} // namespace rock
} // namespace mlir
