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

MLIR_CAPI_EXPORTED unsigned
mlirRockTuningGetNumParamsQuick(MlirRockTuningSpace params) {
  auto *tuningSpace = unwrap(params);
  return tuningSpace->tuningRangeQuick.size();
}

MLIR_CAPI_EXPORTED
unsigned mlirRockTuningGetNumParamsFull(MlirRockTuningSpace params) {
  auto *tuningSpace = unwrap(params);
  return tuningSpace->tuningRangeFull.size();
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
bool mlirRockTuningParamGetFull(MlirRockTuningSpace params, unsigned pos,
                                MlirRockTuningParam param) {
  auto *tuningSpace = unwrap(params);
  auto *paramEntry = unwrap(param);
  return rock::tuningGetParamFull(tuningSpace, pos, paramEntry);
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningParamGetQuick(MlirRockTuningSpace params, unsigned pos,
                                 MlirRockTuningParam param) {
  auto *tuningSpace = unwrap(params);
  auto *paramEntry = unwrap(param);
  return rock::tuningGetParamQuick(tuningSpace, pos, paramEntry);
}

MLIR_CAPI_EXPORTED
size_t mlirRockTuningParamToString(MlirRockTuningParam param, char *buf,
                                   size_t bufLen) {
  auto *paramEntry = unwrap(param);
  SmallString<ROCMLIR_TUNING_PARAM_STRING_BUFSZ> perfConfig;
  paramEntry->param.getPerfConfigStr(perfConfig);
  strncpy(buf, perfConfig.c_str(), bufLen);
  return perfConfig.size();
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningSetParam(MlirModule module, MlirRockTuningParam param) {
  auto mod = unwrap(module);
  auto *paramEntry = unwrap(param);
  return rock::tuningSetParam(mod, paramEntry);
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningSetFromStr(MlirModule module, const char *perfCStr) {
  auto mod = unwrap(module);
  StringRef perfConfig(perfCStr);
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
bool mlirRockTuningUpdateTable(MlirRockTuningTable perfTable,
                               const char *probCStr, const char *perfCStr,
                               float time) {
  StringRef problem = unwrap(mlirStringRefCreateFromCString(probCStr));
  StringRef perfConfig = unwrap(mlirStringRefCreateFromCString(perfCStr));
  auto *pTable = unwrap(perfTable);
  return rock::tuningTableUpdate(pTable, problem, perfConfig, time);
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningSetFromTable(MlirRockTuningTable perfTable,
                                MlirModule module) {
  auto *pTable = unwrap(perfTable);
  auto mod = unwrap(module);
  SmallString<ROCMLIR_TUNING_PARAM_STRING_BUFSZ> perfConfig;
  if (failed(rock::tuningTableLookup(pTable, mod, perfConfig)))
    return false;
  return rock::tuningSetStr(mod, perfConfig);
}

MLIR_CAPI_EXPORTED size_t mlirRockTuningGetKey(MlirModule module, char *buf,
                                               size_t bufLen) {
  auto mod = unwrap(module);
  SmallString<ROCMLIR_TUNING_KEY_BUFSZ> perfStr;
  if (failed(rock::getTuningProblemStr(mod, perfStr)))
    return (size_t)(-1);
  strncpy(buf, perfStr.c_str(), bufLen);
  return perfStr.size();
}
