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

#include "mlir-c/Pass.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx);

// Phase 0 functions : Assuming the given module contains only one function

// Returns the number of operands in the FuncOp and fill information in the
// passed ptr.
MLIR_CAPI_EXPORTED void mlirGetKernelInfo(MlirModule module, void *data);
MLIR_CAPI_EXPORTED int mlirGetKernelInfoSize(MlirModule module);

// Returns block_size and grid_size as int[2]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, int *attrs);

// Returns the size of compiled binary
MLIR_CAPI_EXPORTED int mlirGetBinarySize(MlirModule module);

// Returns the compiled binary
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, char *bin);

// pipelines

MLIR_CAPI_EXPORTED void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm);

MLIR_CAPI_EXPORTED void mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                                                       const char *chip);
#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_MIGRAPHX_H
