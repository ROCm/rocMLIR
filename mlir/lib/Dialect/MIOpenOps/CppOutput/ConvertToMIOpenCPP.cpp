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
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> TunableParametersYAMLFile("tunable-parameters-yaml-file",
                                                      cl::desc("Tunable parameters YAML file"),
                                                      cl::value_desc("filename"),
                                                      cl::Hidden);

static cl::opt<bool> IsPopulateTunableParameters("populate-tunable-parameters-to-yaml-file",
                                                 cl::desc("Populate default tunable parameters to YAML file"),
                                                 cl::value_desc("bool"),
                                                 cl::Hidden);

// non-XDLOPS kernel generation.
static TranslateFromMLIRRegistration
    toCpp("mlir-to-miopen-cpp", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenCpp(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });

static TranslateFromMLIRRegistration
    toHeader("mlir-to-miopen-hpp", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenHeader(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });

static TranslateFromMLIRRegistration
    toCFlags("mlir-to-miopen-cflags", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenCFlags(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });

// XDLOPS kernel generation.
static TranslateFromMLIRRegistration
    toCppXDLOPS("mlir-to-miopen-cpp-xdlops", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenCppXDLOPS(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });

static TranslateFromMLIRRegistration
    toHeaderXDLOPS("mlir-to-miopen-hpp-xdlops", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenHeaderXDLOPS(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });

static TranslateFromMLIRRegistration
    toCFlagsXDLOPS("mlir-to-miopen-cflags-xdlops", [](ModuleOp module, llvm::raw_ostream &output) {
      auto sourceCode = mlir::translateModuleToMIOpenCFlagsXDLOPS(module);
      if (!sourceCode)
        return failure();

      output << *sourceCode;
      return success();
    });
