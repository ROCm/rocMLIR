//===- mlir-translate.cpp - MLIR Translate Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RockPasses.h"
#include "mlir/Conversion/RockToGPU/RockToGPU.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/InitRockPasses.h"
#include "mlir/InitRockTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllTranslations();
  rock::registerRockTranslations();
  registerRockPasses();
  mlir::registerRockConversionPasses();
  rock::registerPasses();
  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
