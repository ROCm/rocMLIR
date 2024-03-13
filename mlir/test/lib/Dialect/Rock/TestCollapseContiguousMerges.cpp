//===- TestCollapseContiguousMerges.cpp - test merge/unmerge folding ----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2024 Advanced Micro Devices Inc.
//===-----------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct CollapseContiguousMergesTestPass
    : public PassWrapper<CollapseContiguousMergesTestPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollapseContiguousMergesTestPass)

  static constexpr auto kTestOpName = "collapse_merges";
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RockDialect, func::FuncDialect>();
  }

  StringRef getArgument() const final {
    return "rock-collapse-contiguous-merges-test";
  }
  StringRef getDescription() const final {
    return "Tests rock::collapseContiguousMerges()";
  }

  void runOnOperation() override;
};
} // end namespace

static LogicalResult testVectorizationInference(func::FuncOp f) {
  OpBuilder b(f.getContext());
  WalkResult result = f.walk([&](Operation *op) -> WalkResult {
    if (op->getName().getIdentifier() !=
        CollapseContiguousMergesTestPass::kTestOpName)
      return WalkResult::advance();
    if (op->getNumOperands() != 1)
      return op->emitOpError("Expected one operand");
    if (op->getNumResults() != 0)
      return op->emitOpError("Expected no results");
    Value input = op->getOperand(0);
    if (!isa<ShapedType>(input.getType()))
      return op->emitOpError("Expected shaped type input");
    Value newInput = isolateTransforms(b, input);
    op->setOperand(0, newInput);
    collapseContiguousMerges(newInput);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void CollapseContiguousMergesTestPass::runOnOperation() {
  func::FuncOp f = getOperation();
  if (failed(testVectorizationInference(f))) {
    emitError(UnknownLoc::get(f.getContext()), "Pass failure");
    signalPassFailure();
  }
}

namespace mlir {
namespace rock {
void registerCollapseContiguousMergesTestPass() {
  PassRegistration<CollapseContiguousMergesTestPass>();
}
} // end namespace rock
} // end namespace mlir
