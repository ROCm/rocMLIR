//===- miopen-opt.cpp - MLIR Optimizer Driver with MIOpen
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitMIOpenDialects.h"
#include "mlir/InitMIOpenPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

// In test directory, no header file
namespace mlir {
namespace miopen {
void registerVectorizationInferenceTestPass();
} // end namespace miopen
} // end namespace mlir

#ifdef MLIR_INCLUDE_TESTS
void registerMIOpenTestPasses() {
  miopen::registerVectorizationInferenceTestPass();
}
#endif

int main(int argc, char **argv) {
  registerAllPasses();
  registerMIOpenPasses();

#ifdef MLIR_INCLUDE_TESTS
  registerMIOpenTestPasses();
#endif

  DialectRegistry registry;
  registerAllDialects(registry);
  registerMIOpenDialects(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR+MIOpen modular optimizer driver\n",
                        registry, /*preloadDialectsInContext=*/false));
}
