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
#include "mlir/Pass/PassManager.h"
#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
#include "llvm/Support/CommandLine.h"

void mlirRegisterRocMLIRDialects(MlirDialectRegistry registry) {
  const char *fakeArgv[] = {"rocMLIR-invoked-as-library",
                            "--mlir-print-local-scope"};
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(sizeof(fakeArgv) / sizeof(const char*),
    fakeArgv, "Fake 'command line' for MIGraphX library debugging",
    nullptr, "ROCMLIR_DEBUG_FLAGS");
  mlir::registerRocMLIRDialects(*unwrap(registry));
}

void mlirRegisterRocMLIRPasses() { mlir::registerRocMLIRPasses(); }
