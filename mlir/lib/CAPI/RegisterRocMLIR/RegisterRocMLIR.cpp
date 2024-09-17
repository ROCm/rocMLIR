//===- RegisterRocMLIR.cpp - Register all rocMLIR entities -----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#include "mlir-c/RegisterRocMLIR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/InitRocMLIRCLOptions.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"

void mlirRegisterRocMLIRDialects(MlirDialectRegistry registry) {
  mlir::registerRocMLIRDialects(*unwrap(registry));
}

void mlirRegisterRocMLIRPasses() { mlir::registerRocMLIRPasses(); }

void mlirRegisterRocMLIROptions() { mlir::registerRocMLIROptions(); }
