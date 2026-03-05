//===- LinalgToRockPass.cpp - Lowering Linalg to Rock Dialect ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Linalg to the Rock dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/LinalgToRock/LinalgToRock.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_LINALGTOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

namespace {
struct LinalgToRockPass : public impl::LinalgToRockPassBase<LinalgToRockPass> {
  void runOnOperation() override;
};
} // namespace

static void populateLinalgToRockDialectConversion(ConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect,
                         rock::RockDialect, bufferization::BufferizationDialect,
                         math::MathDialect>();

  // tensor.insert_slice with operand of one tensor.empty op can be expanded
  // into rock.expand_strides
  target.addDynamicallyLegalOp<tensor::InsertSliceOp>(
      [](tensor::InsertSliceOp op) -> std::optional<bool> {
        auto emptyOp = op.getDest().getDefiningOp<tensor::EmptyOp>();
        if (!emptyOp){
          return true;
        }

        // Require statically known slice sizes that exactly match the
        // source tensor shape.
        auto srcType = dyn_cast<RankedTensorType>(op.getSource().getType());
        if (!srcType)
          return true;

        // into rock.expand_strides, but only in the exact expand-strides shape:
        // - dest is a tensor.empty with a single use
        // - all offsets are zero
        // - all strides are one
        // - all slice sizes are static and match the source tensor shape
        bool isExpandStride = llvm::all_of(op.getStaticOffsets(), [](int64_t offset) { return offset == 0; }) &&
          llvm::all_of(op.getStaticStrides(), [](int64_t stride) { return stride == 1; }) &&
          llvm::none_of(op.getStaticSizes(),
                  [](int64_t s) { return s == ShapedType::kDynamic; }) && op.getStaticSizes() == srcType.getShape();
        return !isExpandStride;
      });

  // We only allow Linalg operations that are elementwise. Fusion is supported
  // via linalg.generic when it is an elementwise operation. Elementwise
  // operations would be converted into linalg.generic in later passes
  target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
      [=](Operation *op) -> std::optional<bool> {
        auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
        if (!linalgOp) {
          return std::nullopt;
        }
        return linalg::isElementwise(linalgOp) || isa<linalg::GenericOp>(op) ||
               isa<linalg::YieldOp>(op);
      });
}

void LinalgToRockPass::runOnOperation() {
  MLIRContext &ctx = getContext();
  func::FuncOp func = getOperation();
  if (!func->hasAttr("rock.kernel")) {
    func->emitError("func op does not have the kernel attribute for "
                    "linalg-to-rock lowering");
    return signalPassFailure();
  }

  ConversionTarget bodyConversionTarget(ctx);
  TypeConverter converter;
  RewritePatternSet bodyPatterns(&ctx);
  populateLinalgToRockDialectConversion(bodyConversionTarget);
  rock::populateLinalgToRockConversionPattern(bodyPatterns, &ctx);
  if (failed(applyPartialConversion(func, bodyConversionTarget,
                                    std::move(bodyPatterns)))) {
    return signalPassFailure();
  }
}
