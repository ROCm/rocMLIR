//===- MIGraphX.cpp - C Interface for MIGraphX dialect---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Rock.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"

#include <vector>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Rock, rock, mlir::rock::RockDialect)
DEFINE_C_API_PTR_METHODS(MlirRockTuningSpace, mlir::rock::TunableParams)
DEFINE_C_API_PTR_METHODS(MlirRockTuningParam, mlir::rock::ParamEntry)
DEFINE_C_API_PTR_METHODS(MlirRockTuningTable, mlir::rock::TuningTable)

using namespace mlir;

MLIR_CAPI_EXPORTED MlirRockTuningSpace
mlirRockTuningSpaceCreate(MlirModule module) {
  struct rock::TunableParams *newParams;
  auto mod = unwrap(module);
  newParams = rock::createTunableParamSpace(mod);
  return wrap(newParams);
}

MLIR_CAPI_EXPORTED int
mlirRockTuningGetNumParamsQuick(MlirRockTuningSpace params) {
  auto tuningSpace = unwrap(params);
  return tuningSpace->numHeuristicQuick;
}

MLIR_CAPI_EXPORTED
int mlirRockTuningGetNumParamsFull(MlirRockTuningSpace params) {
  auto tuningSpace = unwrap(params);
  return tuningSpace->tuningRange.size();
}

MLIR_CAPI_EXPORTED MlirRockTuningParam mlirRockTuningParamCreate() {
  rock::ParamEntry *param = new rock::ParamEntry();
  return wrap(param);
}

MLIR_CAPI_EXPORTED
void mlirRockTuningParamDestroy(MlirRockTuningParam param) {
  delete (unwrap(param));
}

MLIR_CAPI_EXPORTED
void mlirRockTuningSpaceDestroy(MlirRockTuningSpace params) {
  delete (unwrap(params));
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningParamGet(MlirRockTuningSpace params, int pos,
                            MlirRockTuningParam param) {
  auto tuningSpace = unwrap(params);
  auto paramEntry = unwrap(param);
  // out of bound check.
  if (pos < 0 || (unsigned int)pos > tuningSpace->tuningRange.size() - 1)
    return false;
  paramEntry->param = tuningSpace->tuningRange[pos];
  return true;
}

MLIR_CAPI_EXPORTED
const char *mlirRockTuningGetParamStr(MlirRockTuningParam param) {
  auto paramEntry = unwrap(param);
  paramEntry->param.getPerfConfigStr(paramEntry->configStr);
  return paramEntry->configStr.c_str();
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningSetParam(MlirModule module, MlirRockTuningParam param) {
  auto mod = unwrap(module);
  auto paramEntry = unwrap(param);
  return rock::tuningSetParam(mod, paramEntry);
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningSetFromStr(MlirModule module, char *perfCStr) {
  auto mod = unwrap(module);
  MlirStringRef perfStringRef = mlirStringRefCreateFromCString(perfCStr);
  std::string perfConfig = unwrap(perfStringRef).str();
  return rock::tuningSetStr(mod, perfConfig);
}

MLIR_CAPI_EXPORTED
MlirRockTuningTable mlirRockTuningTableCreate() {
  struct rock::TuningTable *newTable = rock::tuningTableCreate();
  return wrap(newTable);
}

MLIR_CAPI_EXPORTED
void mlirRockTuningTableDestroy(MlirRockTuningTable table) {
  delete unwrap(table);
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningUpdateTable(MlirRockTuningTable perfTable, char *probCStr,
                               char *perfCStr, float time) {
  MlirStringRef probStringRef = mlirStringRefCreateFromCString(probCStr);
  MlirStringRef perfStringRef = mlirStringRefCreateFromCString(perfCStr);
  std::string problem = unwrap(probStringRef).str();
  std::string perfConfig = unwrap(perfStringRef).str();
  auto pTable = unwrap(perfTable);
  return rock::tuningTableUpdate(pTable, problem, perfConfig, time);
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningSetFromTable(MlirRockTuningTable perfTable,
                                MlirModule module) {
  auto pTable = unwrap(perfTable);
  auto mod = unwrap(module);
  std::string perfConfig = rock::tuningTableLookup(pTable, mod);
  if (perfConfg.empty())
    return false;
  return rock::tuningSetStr(mod, perfConfig);
}

MLIR_CAPI_EXPORTED const char *
mlirRockTuningGetKey(MlirRockTuningTable perfTable, MlirModule module) {
  auto pTable = unwrap(perfTable);
  auto mod = unwrap(module);
  // Hold output string in the tuning table which has a buffer to store the
  // string formatted problem so the data is not lost in the border between C++
  // and C.
  pTable->problemBuf = rock::getTuningProblemStr(mod);
  return pTable->problemBuf.c_str();
}
