//===-- mlir-c/Dialect/Rock.h - C API for Rock dialect ---- C--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_ROCK_H
#define MLIR_C_DIALECT_ROCK_H

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Rock, rock);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirRockTuningSpace, void);
DEFINE_C_API_STRUCT(MlirRockTuningParam, void);
DEFINE_C_API_STRUCT(MlirRockTuningKey, const void);

MLIR_CAPI_EXPORTED MlirRockTuningSpace
mlirRockTuningSpaceCreate(MlirModule module);

MLIR_CAPI_EXPORTED int
mlirRockTuningGetNumParamsQuick(MlirRockTuningSpace params);

MLIR_CAPI_EXPORTED int
mlirRockTuningGetNumParamsFull(MlirRockTuningSpace params);

MLIR_CAPI_EXPORTED void
mlirRockTuningSpaceDestroy(MlirRockTuningSpace tuningSpace);

MLIR_CAPI_EXPORTED MlirRockTuningParam
mlirRockTuningCreateParamAt(MlirRockTuningSpace params, int pos);

MLIR_CAPI_EXPORTED
void mlirRockTuningParamDestroy(MlirRockTuningParam param);

MLIR_CAPI_EXPORTED
MlirStringRef mlirRockTuningGetParamStr(MlirRockTuningParam param);

MLIR_CAPI_EXPORTED bool mlirRockTuningSetParam(MlirModule module,
                                               MlirRockTuningParam param);

MLIR_CAPI_EXPORTED MlirRockTuningKey
mlirRockTuningGetKey(MlirRockTuningSpace params);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_ROCK_H
