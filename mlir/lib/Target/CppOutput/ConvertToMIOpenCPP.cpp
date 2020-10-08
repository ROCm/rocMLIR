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

#include "mlir/Target/MIOpenCPP.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;

cl::opt<std::string> TunableParametersYAMLFile("tunable-parameters-yaml-file",
                                                      cl::desc("Tunable parameters YAML file"),
                                                      cl::value_desc("filename"),
                                                      cl::Hidden);

cl::opt<bool> IsPopulateTunableParameters("populate-tunable-parameters-to-yaml-file",
                                                 cl::desc("Populate default tunable parameters to YAML file"),
                                                 cl::value_desc("bool"),
                                                 cl::init(false));

namespace mlir {
void registerToMIOpenCPPTranslation() {
  // non-XDLOPS kernel generation.
  TranslateFromMLIRRegistration toCpp(
      "mlir-to-miopen-cpp", [](ModuleOp module, llvm::raw_ostream &output) {
        std::string source;
        mlir::translateModuleToMIOpenCpp(module, source);

        output << source;
        return success();
      });

  TranslateFromMLIRRegistration toHeader(
      "mlir-to-miopen-hpp", [](ModuleOp module, llvm::raw_ostream &output) {
        std::string header;
        mlir::translateModuleToMIOpenHeader(module, header);

        output << header;
        return success();
      });

  TranslateFromMLIRRegistration toCFlags(
      "mlir-to-miopen-cflags", [](ModuleOp module, llvm::raw_ostream &output) {
        std::string cflags;
        mlir::translateModuleToMIOpenCFlags(module, cflags);

        output << cflags;
        return success();
      });

  // XDLOPS kernel generation.
  TranslateFromMLIRRegistration
      toCppXDLOPS("mlir-to-miopen-cpp-xdlops", [](ModuleOp module, llvm::raw_ostream &output) {
        auto sourceCode = mlir::translateModuleToMIOpenCppXDLOPS(module);
        if (!sourceCode)
          return failure();
  
        output << *sourceCode;
        return success();
      });
  
  TranslateFromMLIRRegistration
      toHeaderXDLOPS("mlir-to-miopen-hpp-xdlops", [](ModuleOp module, llvm::raw_ostream &output) {
        auto sourceCode = mlir::translateModuleToMIOpenHeaderXDLOPS(module);
        if (!sourceCode)
          return failure();
  
        output << *sourceCode;
        return success();
      });
  
  TranslateFromMLIRRegistration
      toCFlagsXDLOPS("mlir-to-miopen-cflags-xdlops", [](ModuleOp module, llvm::raw_ostream &output) {
        auto sourceCode = mlir::translateModuleToMIOpenCFlagsXDLOPS(module);
        if (!sourceCode)
          return failure();
  
        output << *sourceCode;
        return success();
      });
}
} // namespace mlir
