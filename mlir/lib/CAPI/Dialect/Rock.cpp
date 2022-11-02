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
// DEFINE_C_API_METHOD(MlirRockTuningKey, const void);

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

MLIR_CAPI_EXPORTED MlirRockTuningParam
mlirRockTuningParamCreate(MlirRockTuningSpace params) {
  auto tuningSpace = unwrap(params);
  rock::ParamEntry *param = new rock::ParamEntry();
  param->param = tuningSpace->tuningRange[pos];
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
  if (pos > tuningSpace->tuningRange.size() - 1)
    return false;
  paramEntry->param = tuningSpace->tuningRange[pos];
  return true;
}

MLIR_CAPI_EXPORTED
char *mlirRockTuningGetParamStr(MlirRockTuningParam param) {
  auto paramEntry = unwrap(param);
  llvm::StringRef strRef = paramEntry.getPerfConfigStr();
  return strRef.data;
}

MLIR_CAPI_EXPORTED
bool mlirRockTuningSetParam(MlirModule module, MlirRockTuningParam param) {
  auto mod = unwrap(module);
  auto paramEntry = unwrap(param);
  return rock::tuningSetParam(mod, paramEntry);
}
