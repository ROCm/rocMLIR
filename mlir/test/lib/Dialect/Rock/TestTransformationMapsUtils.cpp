//===- TestVectorizationInference.cpp - test max vector length code -----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//
// test/lib/Dialect/Rock/testTransformationMapsUtilss.cpp

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include <tuple>

using namespace mlir;
using namespace mlir::rock;

namespace {
struct TransformMapsUtilsTestPass
    : public PassWrapper<TransformMapsUtilsTestPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformMapsUtilsTestPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RockDialect, func::FuncDialect>();
  }

  StringRef getArgument() const final {
    return "rock-transform-maps-utils-test";
  }
  StringRef getDescription() const final {
    return "Tests transformation map utils in Rock";
  }

  void runOnOperation() override;
};
} // end namespace

namespace {
struct ByIndices {
  static SetVector<int64_t> getRemovees(func::FuncOp f) {
    SetVector<int64_t> removeIndices;
    auto attrArray = f->getAttr("remove_dims_by_indices");
    for (auto &attr : attrArray.cast<ArrayAttr>()) {
      removeIndices.insert(attr.cast<IntegerAttr>().getInt());
    }
    return removeIndices;
  }
};

struct ByNames {
  static SetVector<StringRef> getRemovees(func::FuncOp f) {
    SetVector<StringRef> removeIndices;
    auto attrArray = f->getAttr("remove_dims_by_names");
    for (auto &attr : attrArray.cast<ArrayAttr>()) {
      removeIndices.insert(attr.cast<StringAttr>().getValue());
    }
    return removeIndices;
  }
};
} // namespace

template <typename RequestProcessor>
static LogicalResult testSubDimensions(func::FuncOp f) {
  Value returnValue;
  WalkResult walkResult = f.walk([&](func::ReturnOp op) -> WalkResult {
    if (op.getNumOperands() == 1) {
      returnValue = op->getOperand(0);
      return WalkResult::advance();
    }
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  OpBuilder builder(f.getContext());

  ArrayAttr transformAttrs;
  std::tie(std::ignore, transformAttrs, std::ignore) =
      rock::untransform(builder, returnValue);

  auto removees = RequestProcessor::getRemovees(f);
  rock::removeUpperDims(builder, transformAttrs, removees);
  return success();
}

void TransformMapsUtilsTestPass::runOnOperation() {
  func::FuncOp f = getOperation();
  if (f->getAttr("remove_dims_by_indices")) {
    if (failed(testSubDimensions<ByIndices>(f))) {
      emitError(UnknownLoc::get(f.getContext()), "Pass failure");
      signalPassFailure();
    }
  } else if (f->getAttr("remove_dims_by_names")) {
    if (failed(testSubDimensions<ByNames>(f))) {
      emitError(UnknownLoc::get(f.getContext()), "Pass failure");
      signalPassFailure();
    }
  }
}

namespace mlir {
namespace rock {
void registerTransformMapsUtilsTestPass() {
  PassRegistration<TransformMapsUtilsTestPass>();
}
} // end namespace rock
} // end namespace mlir
