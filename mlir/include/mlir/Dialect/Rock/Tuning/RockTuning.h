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

#include "mlir-c/Dialect/RockEnums.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/RWMutex.h"

namespace mlir {
namespace rock {

// The available sets of tuning parameters.
enum class TuningParamSetKind : uint32_t {
  // A short (around 10-15) list of tuning entries that should be tried to
  // quickly obtain reasonable performance on an unknown configuration.
  Quick = 0,
  // A full tuning space suitable for most offline tuning tasks which omits
  // configurations that have been shown not to yield good performance.
  // (Note: this filtering is currently unimplemented).
  Full = 1,
  // A tuning space consisting of all possible sets of tuning parameters,
  // excluding those that could not be applicable to the given problem.
  Exhaustive = 2,
};

// Parameter container holding a parameter and serialized string
struct ParamEntry {
  RockTuningParamAttrInterface param;
  KernelType primaryOpType;
};

// Total tuning space
struct TuningParamSet {
  std::vector<RockTuningParamAttrInterface> tuningRange;
  KernelType primaryOpType;
};

TuningParamSet *createTunableParamSpace(ModuleOp &mod, TuningParamSetKind kind);
// Get a parameters from the set of tunable parameters.
bool tuningGetParam(TuningParamSet *tuningSpace, unsigned pos,
                    ParamEntry *paramEntry);
bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry);
bool tuningSetStr(ModuleOp &mod, StringRef perfConfig);

// A tuning table for rocMLIR.
// Note that this table carries its own reader-writer lock so that it can be
// used from multiple client threads without requiring StringMap to be
// thread-safe.
struct TuningTable {
  llvm::sys::SmartRWMutex<true> lock;
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

bool isSplitKRequested(ModuleOp mod, StringRef perfConfig);

RocmlirSplitKSelectionLikelihood isSplitKFaster(int64_t gDim, int64_t mDim,
                                                int64_t nDim, int64_t kDim,
                                                int64_t numCUs);
} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
