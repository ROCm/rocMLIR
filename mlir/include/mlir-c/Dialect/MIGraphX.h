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
#define MLIR_MIGRAPHX_DIALECT_API_VERSION 4

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx);

// Types

MLIR_CAPI_EXPORTED MlirTypeID rocmlirMIXRShapedTypeGetTypeId(void);

MLIR_CAPI_EXPORTED bool rocmlirIsAMIXRShapedType(MlirType type);

MLIR_CAPI_EXPORTED MlirType rocmlirMIXRShapedTypeGet(intptr_t rank,
                                                     const int64_t *shape,
                                                     const int64_t *strides,
                                                     MlirType elementType);

// Phase 0 functions : Assuming the given module contains only one function

// Returns the required buffer size if called with null buffer
// and fill information in the passed ptr when provided.
MLIR_CAPI_EXPORTED void mlirGetKernelInfo(MlirModule module, int *size,
                                          void *data);

// Returns block_size and grid_size as uint32_t[2]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs);

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, size_t *size,
                                      char *bin);

// pipelines

/// Add the high-level pipeline that creates something that can be tuned.
/// Architecture and num_cu information should be set on the kernel function
/// being compiled.
MLIR_CAPI_EXPORTED void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm);

/// Adds the pipeline that checks if the kernel with a given tuning
/// configuration will actually compile to the pass manager. If this pipeline
/// fails, it's not a meaningful error. The input to this should have been run
/// through the high-level pipeline. This pipeline is only needed when tuning,
/// and should, ideally, be called on a clone of the results of teh highlevel
/// pipeline.
MLIR_CAPI_EXPORTED void
mlirMIGraphXAddApplicabilityPipeline(MlirPassManager pm);

/// Adds a full compile pipeline to the pass manager. This pipeline may either
/// receive the results of the high-level or applicability pipelines.
MLIR_CAPI_EXPORTED bool mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                                                       const char *arch);
#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_MIGRAPHX_H
