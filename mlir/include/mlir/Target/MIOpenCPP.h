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

#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace mlir {

class OwningModuleRef;
class MLIRContext;
class ModuleOp;

/// Convert the given MLIR module into MIOpen C++ . In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
void translateModuleFromMIOpenToCpp(ModuleOp m, std::string &source);

/// Convert the given MLIR module into MIOpen C++ Header. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
void translateModuleFromMIOpenToHeader(ModuleOp m, std::string &header);

/// Convert the given MLIR module into MIOpen C++ compilation flags. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
void translateModuleFromMIOpenToCFlags(ModuleOp m, std::string &cflags);

/// Convert the given MLIR module into MIOpen C++ . In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::string translateModuleFromMIOpenToCppXDLOPS(ModuleOp m);

/// Convert the given MLIR module into MIOpen C++ Header. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::string translateModuleFromMIOpenToHeaderXDLOPS(ModuleOp m);

/// Convert the given MLIR module into MIOpen C++ compilation flags. In case of error, report it
/// to the error handler registered with the MLIR context, if any (obtained from
/// the MLIR module), and return `nullptr`.
std::string translateModuleFromMIOpenToCFlagsXDLOPS(ModuleOp m);

} // namespace mlir

#endif // MLIR_TARGET_MIOPEN_CPP_H
