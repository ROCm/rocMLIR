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
  std::string configStr;
};

// Total tuning space
struct TunableParams {
  std::vector<RockTuningParamAttrInterface> tuningRange;
  KernelType primaryOpType;
  int numHeuristicQuick;
};

TunableParams *createTunableParamSpace(ModuleOp &mod);
bool tuningGetParam(TunableParams *tuningSpace, int pos,
                    ParamEntry *paramEntry);
bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry);
bool tuningSetStr(ModuleOp &mod, std::string perfConfig);

struct TuningTable {
  std::map<std::string, std::pair<std::string, float>> tuningMap;
  std::string problemBuf;
};

TuningTable *tuningTableCreate();
size_t getTuningHash(ModuleOp &mod);
std::string getTuningProblemStr(ModuleOp &mod);
bool tuningTableUpdate(TuningTable *perfTable, std::string problem,
                       std::string perfConfig, float time);
std::string tuningTableLookup(TuningTable *perfTable, ModuleOp &mod);

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
