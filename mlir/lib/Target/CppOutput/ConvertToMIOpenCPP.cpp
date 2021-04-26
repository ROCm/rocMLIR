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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/MIOpenCPP.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
void registerFromMIOpenToCPPTranslation() {
  auto dialectRegistrar = [](DialectRegistry &registry) {
    registry
        .insert<miopen::MIOpenDialect, LLVM::LLVMDialect, StandardOpsDialect>();
  };

  // non-XDLOPS kernel generation.
  TranslateFromMLIRRegistration toCpp(
      "mlir-to-miopen-cpp",
      [](ModuleOp module, llvm::raw_ostream &output) {
        std::string source;
        mlir::translateModuleFromMIOpenToCpp(module, source);

        output << source;
        return success();
      },
      dialectRegistrar);

  TranslateFromMLIRRegistration toHeader(
      "mlir-to-miopen-hpp",
      [](ModuleOp module, llvm::raw_ostream &output) {
        std::string header;
        mlir::translateModuleFromMIOpenToHeader(module, header);

        output << header;
        return success();
      },
      dialectRegistrar);

  TranslateFromMLIRRegistration toCFlags(
      "mlir-to-miopen-cflags",
      [](ModuleOp module, llvm::raw_ostream &output) {
        std::string cflags;
        mlir::translateModuleFromMIOpenToCFlags(module, cflags);

        output << cflags;
        return success();
      },
      dialectRegistrar);

  // XDLOPS kernel generation.
  TranslateFromMLIRRegistration toCppXDLOPS(
      "mlir-to-miopen-cpp-xdlops",
      [](ModuleOp module, llvm::raw_ostream &output) {
        auto sourceCode = mlir::translateModuleFromMIOpenToCppXDLOPS(module);
        if (!sourceCode)
          return failure();
  
        output << *sourceCode;
        return success();
      },
      dialectRegistrar);

  TranslateFromMLIRRegistration toHeaderXDLOPS(
      "mlir-to-miopen-hpp-xdlops",
      [](ModuleOp module, llvm::raw_ostream &output) {
        auto sourceCode = mlir::translateModuleFromMIOpenToHeaderXDLOPS(module);
        if (!sourceCode)
          return failure();
  
        output << *sourceCode;
        return success();
      },
      dialectRegistrar);

  TranslateFromMLIRRegistration toCFlagsXDLOPS(
      "mlir-to-miopen-cflags-xdlops",
      [](ModuleOp module, llvm::raw_ostream &output) {
        auto sourceCode = mlir::translateModuleFromMIOpenToCFlagsXDLOPS(module);
        if (!sourceCode)
          return failure();
  
        output << *sourceCode;
        return success();
      },
      dialectRegistrar);
}
} // namespace mlir
