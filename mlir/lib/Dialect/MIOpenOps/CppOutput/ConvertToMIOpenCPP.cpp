//===- ConvertToMIOpenCPP.cpp - MLIR to MIOpen C++ conversion -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR MIOpen dialect and C++.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/MIOpenCPP.h"
#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

std::unique_ptr<llvm::StringRef> mlir::translateModuleToMIOpenCPP(ModuleOp m) {
  // Check constraints:
  //
  // The Module should only contain 1 function.
  // The Function should only contain exactly:
  // - 0 conv2d op.
  // - 5 transform ops (1 for filter, 3 for input, 1 for output).
  // - 1 gridwise gemm op.
  m.dump();

  return std::make_unique<llvm::StringRef>("Hello World");
}

static TranslateFromMLIRRegistration
    toCPP("mlir-to-miopencpp", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenCPP(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });
