//===--------- RockTuning.h - MLIR tuning parameter generation ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR base types for tuning
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
#define MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace rock {

// Parameter container holding a parameter and serialized string
struct ParamEntry {
  RockTuningParamAttrInterface param;
  KernelType primaryOpType;
};

// Total tuning space
struct TunableParams {
  std::vector<RockTuningParamAttrInterface> tuningRange;
  KernelType primaryOpType;
  int numHeuristicQuick;
};

TunableParams *createTunableParamSpace(ModuleOp &mod);

bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry);

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

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
