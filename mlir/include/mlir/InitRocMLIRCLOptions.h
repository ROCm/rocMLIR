//===- InitRocMLIRCLOptions.h - rocMLIR CL Options Registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file registers command-line options.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITROCMLIRCLOPTIONS_H_
#define MLIR_INITROCMLIRCLOPTIONS_H_

#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {

inline void registerMLIRCLOptions() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
}

} // namespace mlir

#endif // MLIR_INITROCMLIRCLOPTIONS_H_
