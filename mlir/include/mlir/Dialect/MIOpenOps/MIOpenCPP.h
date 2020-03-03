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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

#include <memory>

extern llvm::cl::opt<std::string> TunableParametersYAMLFile;
extern llvm::cl::opt<bool> IsPopulateTunableParameters;

LLVM_YAML_IS_STRING_MAP(int)

// greatest common divisor, aka highest common factor
template <typename T>
T gcd(T x, T y)
{
    if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x - y, y);
    }
    else
    {
        return gcd(x, y - x);
    }
}

template <typename T, typename... Ys>
T gcd(T x, Ys... ys)
{
    return gcd(x, gcd(ys...));
}

// least common multiple
template <typename T>
T lcm(T x, T y)
{
    if(x == 0 || y == 0)
    {
        return 0;
    }
    else
    {
        return (x * y) / gcd(x, y);
    }
}

template <typename T, typename... Ys>
T lcm(T x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <typename T>
T integer_divide_ceil(T x, T y)
{
    return (x + y - 1) / y;
}

template <typename T>
T integer_least_multiple(T x, T y)
{
    return y * integer_divide_ceil(x, y);
}

struct ConvolutionContext {
    int64_t k, c, y, x;
    int64_t n, hi, wi;
    int64_t ho, wo;
    int64_t strideH, strideW;
    int64_t dilationH, dilationW;
    int64_t paddingHL, paddingHR, paddingWL, paddingWR;

    size_t dimKF, dimCF, dimYF, dimXF;
    size_t dimNO, dimKO, dimHO, dimWO;
    size_t dimNI, dimCI, dimHI, dimWI;
};

class TunableParametersBase {
public:
  TunableParametersBase(llvm::StringRef &&yamlFileName) : params(), configFileName(yamlFileName), ctx() {}
  TunableParametersBase() : TunableParametersBase("tunable.yaml") {}

  void init() {
    auto yaml = mlir::openInputFile(configFileName);
    if (!yaml) {
      customInit();
    } else {
      loadYAML(yaml->getBuffer());
    }
  }

  void initWithContext(ConvolutionContext &ctx) {
    this->ctx = ctx;
    init();
  }

  virtual void customInit() = 0;

  void print(llvm::raw_ostream &os) {
    for (auto kv : params) {
      os << " -D" << kv.first << "=" << kv.second;
    }
  }
  void dump() {
    auto outputYAMLFile = mlir::openOutputFile(configFileName);
    if (outputYAMLFile) {
      printYAML(outputYAMLFile->os());
      outputYAMLFile->keep();
    } else {
      llvm::errs() << "\nOpen output file failed: " << configFileName << "\n";
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
  void setValue(llvm::StringRef str, int value) {
    params[str] = value;
  }
protected:
  std::map<std::string, int> params;
  llvm::StringRef configFileName;
  ConvolutionContext ctx;
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
