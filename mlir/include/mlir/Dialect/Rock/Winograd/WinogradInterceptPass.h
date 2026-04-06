//===-- WinogradInterceptPass.h - Winograd pass decl ------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_INTERCEPT_PASS_H
#define MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_INTERCEPT_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace rock {

struct WinogradInterceptPassOptions {
  std::string triple;
  std::string chip;
  std::string features;
};

std::unique_ptr<OperationPass<ModuleOp>>
createWinogradInterceptPass(const WinogradInterceptPassOptions &opts = {});

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_WINOGRAD_WINOGRAD_INTERCEPT_PASS_H
