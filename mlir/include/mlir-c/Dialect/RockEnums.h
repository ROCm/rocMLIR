//===-- mlir-c/Dialect/Rock.h - C API Enums for Rock dialect ---- C--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_ROCK_ENUMS_H
#define MLIR_C_DIALECT_ROCK_ENUMS_H

#ifdef __cplusplus
extern "C" {
#endif

// See TuningParamSetKind in the C++ for descriptions of these flags.
enum RocmlirTuningParamSetKind {
  RocmlirTuningParamSetKindQuick = 0,
  RocmlirTuningParamSetKindFull = 1,
  RocmlirTuningParamSetKindExhaustive = 2
};
typedef enum RocmlirTuningParamSetKind RocmlirTuningParamSetKind;

// TODO (ravil): document
enum RocmlirSplitKSelectionLikelihood { never = 0, maybe = 1, always = 2 };
typedef enum RocmlirSplitKSelectionLikelihood RocmlirSplitKSelectionLikelihood;

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_ROCK_ENUMS_H
