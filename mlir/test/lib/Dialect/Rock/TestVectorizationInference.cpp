//===- TestVectorizationInference.cpp - test max vector length code -----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

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
struct VectorizationInferenceTestPass
    : public PassWrapper<VectorizationInferenceTestPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizationInferenceTestPass)

  static constexpr auto kTestOpName = "get_length";
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RockDialect, func::FuncDialect>();
  }

  StringRef getArgument() const final {
    return "rock-vectorization-inference-test";
  }
  StringRef getDescription() const final {
    return "Tests vectorization inference code in Rock";
  }

  void runOnOperation() override;
};
} // end namespace

static LogicalResult testVectorizationInference(func::FuncOp f) {
  WalkResult result = f.walk([&](Operation *op) -> WalkResult {
    if (op->getName().getIdentifier() !=
        VectorizationInferenceTestPass::kTestOpName)
      return WalkResult::advance();
    if (op->getNumOperands() != 1)
      return op->emitOpError("Expected one operand");
    if (op->getNumResults() != 0)
      return op->emitOpError("Expected no results");
    Value input = op->getOperand(0);
    if (!isa<ShapedType>(input.getType()))
      return op->emitOpError("Expected shaped type input");
    auto inDim = op->getAttr("in_dim").dyn_cast_or_null<IntegerAttr>();
    if (!inDim)
      return op->emitOpError("Expected integer attribute `in_dim`");
    std::optional<int64_t> inDimLen = std::nullopt;
    auto maxLenOverride =
        op->getAttr("in_dim_len").dyn_cast_or_null<IntegerAttr>();
    if (maxLenOverride)
      inDimLen = maxLenOverride.getInt();
    Operation *operationRootForFusionTraversal = nullptr;
    if (op->hasAttr("traverseFusions")) {
      operationRootForFusionTraversal = op;
    }
    bool limitForDataType = op->hasAttrOfType<UnitAttr>("limitForDataType");
    VectorizationResult result = getMaxVectorization(
        input, inDim.getInt(), inDimLen, operationRootForFusionTraversal,
        /*ignoreDataType=*/!limitForDataType);
    MLIRContext *ctx = op->getContext();
    op->setAttr("result", IntegerAttr::get(IndexType::get(ctx), result.max));
    op->setAttr("bufferVectorSize",
                IntegerAttr::get(IndexType::get(ctx), result.bufferVectorSize));
    if (operationRootForFusionTraversal)
      op->setAttr("fusionTraversalStatus",
                  BoolAttr::get(ctx, result.fusionTraversalStatus.succeeded()));
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void VectorizationInferenceTestPass::runOnOperation() {
  func::FuncOp f = getOperation();
  if (failed(testVectorizationInference(f))) {
    emitError(UnknownLoc::get(f.getContext()), "Pass failure");
    signalPassFailure();
  }
}

namespace mlir {
namespace rock {
void registerVectorizationInferenceTestPass() {
  PassRegistration<VectorizationInferenceTestPass>();
}
} // end namespace rock
} // end namespace mlir
