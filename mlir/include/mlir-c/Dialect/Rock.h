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

// See TuningParamSetKind in the C++ for descriptions of these flags.
enum RocmlirTuningParamSetKind {
  RocmlirTuningParamSetKindQuick = 0,
  RocmlirTuningParamSetKindFull = 1,
  RocmlirTuningParamSetKindExhaustive = 2
};
typedef enum RocmlirTuningParamSetKind RocmlirTuningParamSetKind;

// Create full range of the tuning params space
MLIR_CAPI_EXPORTED MlirRockTuningSpace
mlirRockTuningSpaceCreate(MlirModule module, RocmlirTuningParamSetKind kind);

// Returns the number of parameters in the given tuning space.
MLIR_CAPI_EXPORTED unsigned
mlirRockTuningGetNumParams(MlirRockTuningSpace params);

// Allocate memory for a single instance of the tuning params
MLIR_CAPI_EXPORTED MlirRockTuningParam mlirRockTuningParamCreate(void);

// Destroy given params allocation
MLIR_CAPI_EXPORTED
void mlirRockTuningParamDestroy(MlirRockTuningParam param);

// Destroy the tuning params space
MLIR_CAPI_EXPORTED
void mlirRockTuningSpaceDestroy(MlirRockTuningSpace params);

// Get tuning params at the given position in the tuning parameter set and
// return it into `dest`. Returns false on failure. This will not modify
// `params` and will copy into `param`, overwriting it.
MLIR_CAPI_EXPORTED
bool mlirRockTuningParamGet(MlirRockTuningSpace params, unsigned pos,
                            MlirRockTuningParam param);

// Get tuning params at the given position in the quick tuning table and return
// it into `dest`. Returns false on failure.This will not modify `params`
// and will copy into `param`, overwriting it.
MLIR_CAPI_EXPORTED
bool mlirRockTuningParamGetQuick(MlirRockTuningSpace params, unsigned pos,
                                 MlirRockTuningParam param);

// The recommended buffer size for a parameter string.
#define ROCMLIR_TUNING_PARAM_STRING_BUFSZ 64

// Generate the string representation of `param`. This representation will be
// copied into `buf`, which should point to `bufLen` bytes of memory. This
// function returns the true length of the parameter string, excluding the
// trailing null which is not guaranteed to be inserted - it is the caller's
// responsibility to ensure that the returned size is less than the size of the
// provided buffer and to handle the case where the buffer was too small (in
// which case, per strncpy(), no null terminator will be added to the buffer.)
MLIR_CAPI_EXPORTED
size_t mlirRockTuningParamToString(MlirRockTuningParam param, char *buf,
                                   size_t bufLen);

// Set the tuning params of the given module using provided param
MLIR_CAPI_EXPORTED bool mlirRockTuningSetParam(MlirModule module,
                                               MlirRockTuningParam param);

// Set the tuning params of the given module using provided perf string
MLIR_CAPI_EXPORTED
bool mlirRockTuningSetFromStr(MlirModule module, MlirStringRef perfStr);

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
                               MlirStringRef problemKey, MlirStringRef perfStr,
                               float time);

// Search the tuning table and get the stored best value for the given problem.
// The definition of the tuning problem is internally described and opaque to
// the users.
MLIR_CAPI_EXPORTED
bool mlirRockTuningSetFromTable(MlirRockTuningTable perfTable,
                                MlirModule module);

// The recommended buffer size for a .tuning key
#define ROCMLIR_TUNING_KEY_BUFSZ 2048

// Produces a string representation of the tuning table key for the problem
// found within `module`. The representation will be copied into `buf`, which
// should point to `bufLen` bytes of memory. This function returns the true
// length of the problem string, excluding the terminating null - it is the
// caller's responsibility to ensure that the returned size is less than the
// size of the provided buffer and to handle the case where the buffer was too
// small (in which case, per strncpy(), no null terminator will be added to the
// buffer).
//
// If the problem cannot be converted into a key for some reason (this shouldn't
// happen), returns (size_t)(-1).
MLIR_CAPI_EXPORTED size_t mlirRockTuningGetKey(MlirModule module, char *buf,
                                               size_t bufLen);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_ROCK_H
