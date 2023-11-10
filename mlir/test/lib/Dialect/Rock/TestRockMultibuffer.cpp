//===- TestVectorizationInference.cpp - test max vector length code -----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Transforms/RockMultibuffer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct MultiBufferingTestPass
    : public PassWrapper<MultiBufferingTestPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MultiBufferingTestPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RockDialect>();
  }

  StringRef getArgument() const final { return "rock-multibuffer-test"; }
  StringRef getDescription() const final {
    return "Tests multibuffer code in Rock";
  }

  void runOnOperation() override;
};
} // end namespace

static const StringLiteral kMultiBuffer = "__multibuffer__";

static LogicalResult testMultiBuffering(func::FuncOp f) {
  SmallVector<std::pair<rock::GpuAllocOp, int64_t>> toMultibuffer;

  f.walk([&](rock::GpuAllocOp op) {
    if (op->hasAttr(kMultiBuffer)) {
      auto multiBufferAttr = op->getAttrOfType<IntegerAttr>(kMultiBuffer);
      toMultibuffer.push_back({op, multiBufferAttr.getInt()});
      op->removeAttr(kMultiBuffer);
    }
  });

  for (auto alloc : toMultibuffer)
    (void)rock::multiBuffer(alloc.first, alloc.second, true);

  return success();
}

void MultiBufferingTestPass::runOnOperation() {
  func::FuncOp f = getOperation();
  if (failed(testMultiBuffering(f))) {
    emitError(UnknownLoc::get(f.getContext()), "Pass failure");
    signalPassFailure();
  }
}

namespace mlir {
namespace rock {
void registerMultiBufferingTestPass() {
  PassRegistration<MultiBufferingTestPass>();
}
} // end namespace rock
} // end namespace mlir
