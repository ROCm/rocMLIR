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
// Version 4: The MLIR shaped type is added to better represent MIGRaphX's
// native type
// Version 5: Breaking changes:
//   - mlirMIGraphXAddBackendPipeline() now takes a MlirMIGraphXBackendOptions*
//     (arch, perfConfig, optLevel) instead of a separate arch arg.
//   - mlirGetKernelAttrs() returns uint32_t[3] {block_size, grid_size,
//     cluster_size} instead of uint32_t[2] {block_size, grid_size}.
//   - Removed: mlirGetKernelInfo(), mlirMIGraphXAddApplicabilityPipeline().
#define MLIR_MIGRAPHX_DIALECT_API_VERSION 5

typedef struct MlirMIGraphXBackendOptions {
  const char *arch;
  const char *perfConfig;
  int optLevel;
} MlirMIGraphXBackendOptions;

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx);

// Types

MLIR_CAPI_EXPORTED MlirTypeID rocmlirMIXRShapedTypeGetTypeId(void);

MLIR_CAPI_EXPORTED bool rocmlirIsAMIXRShapedType(MlirType type);

MLIR_CAPI_EXPORTED MlirType rocmlirMIXRShapedTypeGet(intptr_t rank,
                                                     const int64_t *shape,
                                                     const int64_t *strides,
                                                     MlirType elementType);

MLIR_CAPI_EXPORTED MlirType rocmlirMIXRShapedTypeAsTensor(MlirType type);

// Returns block_size, grid_size and cluster_size as uint32_t[3]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs);

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, size_t *size,
                                      char *bin);

// pipelines

/// Add the high-level pipeline that creates something that can be tuned.
/// Architecture, num_cu and num_chiplets information should be set on the
/// kernel function being compiled.
MLIR_CAPI_EXPORTED void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm);

/// Adds a full compile pipeline to the pass manager. This pipeline may either
/// receive the results of the high-level pipeline.
MLIR_CAPI_EXPORTED bool
mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                               const MlirMIGraphXBackendOptions *opts);
#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_MIGRAPHX_H
