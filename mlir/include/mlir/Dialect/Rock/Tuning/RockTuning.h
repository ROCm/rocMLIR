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
  std::vector<RockTuningParamAttrInterface> tuningRangeFull;
  std::vector<RockTuningParamAttrInterface> tuningRangeQuick;
  KernelType primaryOpType;
};

TunableParams *createTunableParamSpace(ModuleOp &mod);
bool tuningGetParamFull(TunableParams *tuningSpace, unsigned pos,
                        ParamEntry *paramEntry);
bool tuningGetParamQuick(TunableParams *tuningSpace, unsigned pos,
                         ParamEntry *paramEntry);
bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry);
bool tuningSetStr(ModuleOp &mod, StringRef perfConfig);

struct TuningTable {
  llvm::StringMap<std::pair<SmallString<64>, float>> tuningMap;
};

TuningTable *tuningTableCreate();
size_t getTuningHash(ModuleOp &mod);
LogicalResult getTuningProblemStr(ModuleOp &mod, SmallVectorImpl<char> &out);
bool tuningTableUpdate(TuningTable *perfTable, StringRef problem,
                       StringRef perfConfig, float time);
LogicalResult tuningTableLookup(TuningTable *perfTable, ModuleOp &mod,
                                SmallVectorImpl<char> &out);
LogicalResult tuningTableLookupByKey(TuningTable *perfTable,
                                     SmallVectorImpl<char> &out);

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
