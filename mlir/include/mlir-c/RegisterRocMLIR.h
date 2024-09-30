//===-- RegisterRocMLIR.h - C API for loading RocMLIR dialects -*- C-*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions.
//
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REGISTER_ROCMLIR_H
#define MLIR_C_REGISTER_ROCMLIR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Appends all rocMLIR dialects and extensions to the dialect registry.
MLIR_CAPI_EXPORTED void
mlirRegisterRocMLIRDialects(MlirDialectRegistry registry);

/// Register all compiler passes of rocMLIR.
MLIR_CAPI_EXPORTED void mlirRegisterRocMLIRPasses(void);

/// Register command-line options read from ROCMLIR_DEBUG_FLAGS.
MLIR_CAPI_EXPORTED void mlirRegisterRocMLIRLibCLOptions(void);

#ifdef __cplusplus
}
#endif
#endif // MLIR_R_REGISTER_ROCMLIR_H
