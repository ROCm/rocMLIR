//===-- mlir-c/Dialect/MIGraphX.h - C API for MIGraphX dialect --------*- C
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_MIGRAPHX_H
#define MLIR_C_DIALECT_MIGRAPHX_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

// Version 2: Use bare pointer ABI (kernels take just a pointer to the data
// buffer, not an entire memref struct). Also introduces this constant.
// Version 3: mlirMIGraphXAddBackendPipeline() to get full arch name instead of
// split strings of triple/chip/features
#define MLIR_MIGRAPHX_DIALECT_API_VERSION 3

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx);

// Phase 0 functions : Assuming the given module contains only one function

// Returns the required buffer size if called with null buffer
// and fill information in the passed ptr when provided.
MLIR_CAPI_EXPORTED void mlirGetKernelInfo(MlirModule module, int *size,
                                          void *data);

// Returns block_size and grid_size as uint32_t[2]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs);

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, int *size, char *bin);

// pipelines

MLIR_CAPI_EXPORTED void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm);

MLIR_CAPI_EXPORTED bool mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                                                       const char *arch);
#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_MIGRAPHX_H
