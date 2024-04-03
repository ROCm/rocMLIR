//===- TestFunctionFusibility.cpp - test fusion utils ----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2024 Advanced Micro Devices Inc.
//===-----------------------------------------------------===//

#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct FunctionFusibilityTestPass
    : public PassWrapper<FunctionFusibilityTestPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FunctionFusibilityTestPass)

  StringRef getArgument() const final {
    return "rock-function-fusibility-test";
  }
  StringRef getDescription() const final {
    return "Tests fusability of functions in Rock";
  }

  void runOnOperation() override;
};
} // end namespace

static LogicalResult analyse(func::FuncOp func) {
  OpBuilder builder(func.getContext());
  const bool testResult = rock::testFusibility(func);
  if (testResult) {
    func->setAttr("fusibile", builder.getStringAttr("yes"));
  } else {
    func->setAttr("fusibile", builder.getStringAttr("no"));
  }

  return success();
}

void FunctionFusibilityTestPass::runOnOperation() {
  func::FuncOp f = getOperation();
  if (failed(analyse(f))) {
    emitError(UnknownLoc::get(f.getContext()), "Pass failure");
    signalPassFailure();
  }
}

namespace mlir {
namespace rock {
void registerFusibilityTestPass() {
  PassRegistration<FunctionFusibilityTestPass>();
}
} // end namespace rock
} // end namespace mlir
