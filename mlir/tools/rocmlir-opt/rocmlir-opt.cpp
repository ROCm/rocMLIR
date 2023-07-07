//===- rocmlir-opt.cpp - MLIR Optimizer Driver with Rock
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

#include "mlir/InitRocMLIRDialects.h"
#include "mlir/InitRocMLIRPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

// In test directory, no header file
namespace mlir {
namespace rock {
void registerVectorizationInferenceTestPass();
} // end namespace rock
} // end namespace mlir

#ifdef MLIR_INCLUDE_TESTS
void registerRockTestPasses() {
  rock::registerVectorizationInferenceTestPass();
}
#endif

int main(int argc, char **argv) {
  registerRocMLIRPasses();

#ifdef MLIR_INCLUDE_TESTS
  registerRockTestPasses();
#endif

  DialectRegistry registry;
  registerRocMLIRDialects(registry);
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR+Rock modular optimizer driver\n", registry));
}
