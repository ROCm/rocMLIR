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

// Version 1: Add tuning API
// Version 2: expose quick tuning list separately, move to unsigned ints.
#define MLIR_ROCK_C_API_VERSION 2

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Rock, rock);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirRockTuningSpace, void);
DEFINE_C_API_STRUCT(MlirRockTuningParam, void);
DEFINE_C_API_STRUCT(MlirRockTuningTable, void);
// DEFINE_C_API_STRUCT(MlirRockGemmWrapperInterface, void);
DEFINE_C_API_STRUCT(MlirRockTuningKey, const void);

// Create full range of the tuning params space
MLIR_CAPI_EXPORTED MlirRockTuningSpace
mlirRockTuningSpaceCreate(MlirModule module);

// Returns the estimated number of tuning params that user can quickly find the
// optimal solution in the sorted array
MLIR_CAPI_EXPORTED unsigned
mlirRockTuningGetNumParamsQuick(MlirRockTuningSpace params);

// Returns total number of the tuning params in the array
MLIR_CAPI_EXPORTED unsigned
mlirRockTuningGetNumParamsFull(MlirRockTuningSpace params);

// Allocate memory for a single instance of the tuning params
MLIR_CAPI_EXPORTED MlirRockTuningParam mlirRockTuningParamCreate(void);

// Destroy given params allocation
MLIR_CAPI_EXPORTED
void mlirRockTuningParamDestroy(MlirRockTuningParam param);

// Destroy the tuning params space
MLIR_CAPI_EXPORTED
void mlirRockTuningSpaceDestroy(MlirRockTuningSpace params);

// Get tuning params at the given position in the full tuning table and return
// it into `dest`. Returns false on failure.
MLIR_CAPI_EXPORTED
bool mlirRockTuningParamGetFull(MlirRockTuningSpace params, unsigned pos,
                                MlirRockTuningParam param);

// Get tuning params at the given position in the quick tuning table and return
// it into `dest`. Returns false on failure.
MLIR_CAPI_EXPORTED
bool mlirRockTuningParamGetQuick(MlirRockTuningSpace params, unsigned pos,
                                 MlirRockTuningParam param);

// Returns cstring of the serialized perfconfig
MLIR_CAPI_EXPORTED
const char *mlirRockTuningGetParamStr(MlirRockTuningParam param);

// Set the tuning params of the given module using provided param
MLIR_CAPI_EXPORTED bool mlirRockTuningSetParam(MlirModule module,
                                               MlirRockTuningParam param);

// Set the tuning params of the given module using provided perf string
MLIR_CAPI_EXPORTED
bool mlirRockTuningSetFromStr(MlirModule module, char *perfCStr);

// Opaque pointer to tuning table storage. This could be used as an abstraction
// to access the database. Initially, it's pointing to a memory map for now.
MLIR_CAPI_EXPORTED
MlirRockTuningTable mlirRockTuningTableCreate(void);

// Destroy (close) the tuning table storage
MLIR_CAPI_EXPORTED
void mlirRockTuningTableDestroy(MlirRockTuningTable table);

// Update the table entry. This API tries to register/update the tuning result
// of a single problem into the tuning table. Current policy is only storing
// the best performing tuning parameter to simplify the underlying
// implementation, which can be revisited in the future. Returns true if the
// table was actually updated, and false if a better-performing entry exists.
MLIR_CAPI_EXPORTED
bool mlirRockTuningUpdateTable(MlirRockTuningTable perfTable,
                               const char *probCStr, const char *perfCStr,
                               float time);

// Search the tuning table and get the stored best value for the given problem.
// The definition of the tuning problem is internally described and opaque to
// the users.
MLIR_CAPI_EXPORTED
bool mlirRockTuningSetFromTable(MlirRockTuningTable perfTable,
                                MlirModule module);

// Returns the tuning problem in the C-string format for the inspection.
// The memory behind this pointer is owned by the tuning table.
MLIR_CAPI_EXPORTED const char *
mlirRockTuningGetKey(MlirRockTuningTable perfTable, MlirModule module);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_ROCK_H
