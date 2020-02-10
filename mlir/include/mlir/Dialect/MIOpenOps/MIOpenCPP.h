//===- MIOpenCPP.h - MLIR to C++ for MIOpen conversion ----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the entry point for the MLIR to MIOpen C++ conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MIOPEN_CPP_H
#define MLIR_TARGET_MIOPEN_CPP_H

#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/YAMLTraits.h"

#include <memory>

LLVM_YAML_IS_STRING_MAP(int)

class TunableParametersBase {
public:
  TunableParametersBase(llvm::StringRef &&yamlFileName) : params(), configFileName(yamlFileName) {}
  TunableParametersBase() : TunableParametersBase("tunable.yaml") {}

  void init() {
    auto yaml = mlir::openInputFile(configFileName);
    if (!yaml) {
      customInit();
    } else {
      loadYAML(yaml->getBuffer());
    }
  }

  virtual void customInit() = 0;

  void print(llvm::raw_ostream &os) {
    for (auto kv : params) {
      os << " -D" << kv.first << "=" << kv.second;
    }
  }
  void printYAML(llvm::raw_ostream &os) {
    llvm::yaml::Output xout(os, nullptr, 0);
    xout << params;
    os.flush();
  }
  void loadYAML(llvm::StringRef yaml) {
    params.clear();
    llvm::yaml::Input yin(yaml);
    yin >> params;
  }
  int operator[](llvm::StringRef str) {
    if (params.find(str) != params.end()) {
      return params[str];
    }
    return 0;
  }
protected:
  std::map<std::string, int> params;
  llvm::StringRef configFileName;
};

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlir {

class OwningModuleRef;
class MLIRContext;
class ModuleOp;

/// Convert the given MLIR module into MIOpen C++ . In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::unique_ptr<llvm::StringRef> translateModuleToMIOpenCpp(ModuleOp m);

/// Convert the given MLIR module into MIOpen C++ Header. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::unique_ptr<llvm::StringRef> translateModuleToMIOpenHeader(ModuleOp m);

/// Convert the given MLIR module into MIOpen C++ compilation flags. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::unique_ptr<llvm::StringRef> translateModuleToMIOpenCFlags(ModuleOp m);

/// Convert the given MLIR module into MIOpen C++ . In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::unique_ptr<llvm::StringRef> translateModuleToMIOpenCppXDLOPS(ModuleOp m);

/// Convert the given MLIR module into MIOpen C++ Header. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::unique_ptr<llvm::StringRef> translateModuleToMIOpenHeaderXDLOPS(ModuleOp m);

/// Convert the given MLIR module into MIOpen C++ compilation flags. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::unique_ptr<llvm::StringRef> translateModuleToMIOpenCFlagsXDLOPS(ModuleOp m);

} // namespace mlir

#endif // MLIR_TARGET_MIOPEN_CPP_H
