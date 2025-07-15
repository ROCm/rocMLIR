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

#include "mlir/InitAllTranslations.h"
#include "mlir/InitRocMLIRTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllTranslations();
  rock::registerRocMLIRTranslations();
  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
