//===- TestTransformationMapsUtils.cpp - test removeUpperDims utility -----===//
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
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
  static SetVector<int64_t> getTests(Operation *testHandleOp) {
    SetVector<int64_t> removeIndices;
    auto arrayAttr = cast<ArrayAttr>(testHandleOp->getAttr("indices_to_drop"));
    for (auto &attr : arrayAttr) {
      removeIndices.insert(cast<IntegerAttr>(attr).getInt());
    }
    return removeIndices;
  }
};

struct ByNames {
  static SetVector<StringRef> getTests(Operation *testHandleOp) {
    SetVector<StringRef> removeIndices;
    auto arrayAttr = cast<ArrayAttr>(testHandleOp->getAttr("names_to_drop"));
    for (auto &attr : arrayAttr) {
      removeIndices.insert(cast<StringAttr>(attr).getValue());
    }
    return removeIndices;
  }
};
} // namespace

struct RemoveDimTestPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  FailureOr<Operation *> getTestHandleOp(func::FuncOp func) const {
    Operation *testHandleOp = nullptr;
    WalkResult walkResult = func->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "remove_dims") {
        testHandleOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!walkResult.wasInterrupted()) {
      emitError(UnknownLoc::get(func.getContext()),
                "no `remove_dims` op found");
      return failure();
    }

    if (testHandleOp->getNumOperands() != 1) {
      emitError(UnknownLoc::get(func.getContext()),
                "`remove_dims` op must have a single operand");
      return failure();
    }
    return testHandleOp;
  }

  template <typename RequestProcessor>
  FailureOr<ArrayAttr> removeUpperDims(PatternRewriter &builder,
                                       Operation *testHandleOp) const {
    Value testValue = testHandleOp->getOperand(0);

    ArrayAttr transformAttrs;
    std::tie(std::ignore, transformAttrs, std::ignore) =
        rock::untransform(builder, testValue);

    auto removeDims = RequestProcessor::getTests(testHandleOp);
    FailureOr<ArrayAttr> maybeNewArrayAttr =
        rock::removeUpperDims(builder, transformAttrs, removeDims);

    return maybeNewArrayAttr;
  }

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &b) const final {

    FailureOr<Operation *> maybeTestHandleOp = getTestHandleOp(func);
    if (failed(maybeTestHandleOp)) {
      return failure();
    }
    Operation *testHandleOp = maybeTestHandleOp.value();

    // remove indices
    FailureOr<ArrayAttr> maybeNewTrMapAttrs;
    if (testHandleOp->getAttr("names_to_drop")) {
      maybeNewTrMapAttrs = removeUpperDims<ByNames>(b, testHandleOp);
    } else if (testHandleOp->getAttr("indices_to_drop")) {
      maybeNewTrMapAttrs = removeUpperDims<ByIndices>(b, testHandleOp);
    } else {
      emitError(UnknownLoc::get(func.getContext()),
                "`remove_dims` op does not have `drop` attr");
      return failure();
    }

    if (failed(maybeNewTrMapAttrs)) {
      return failure();
    }
    ArrayAttr newTrMapAttrs = maybeNewTrMapAttrs.value();

    // construct a new function signature and remove the old function body
    Attribute attr = *(std::prev(newTrMapAttrs.end()));
    auto firstTrMap = cast<rock::TransformMapAttr>(attr);
    DenseI64ArrayAttr firstLowerBounds = firstTrMap.getLowerBounds();

    Type inputType = func.getArgument(0).getType();
    Type inputElementType = cast<MemRefType>(inputType).getElementType();

    auto newInputType =
        MemRefType::get(firstLowerBounds.asArrayRef(), inputElementType);
    auto newFuncType =
        FunctionType::get(func->getContext(), {newInputType}, {});

    func.eraseBody();
    func.setFunctionType(newFuncType);

    Block *newEntryBlock = func.addEntryBlock();
    b.setInsertionPointToStart(newEntryBlock);

    // insert new transform ops
    Location loc = b.getUnknownLoc();
    Value input = newEntryBlock->getArgument(0);
    for (auto trMapAttr : llvm::reverse(newTrMapAttrs)) {
      auto trOp = b.create<rock::TransformOp>(
          loc, input, cast<TransformMapAttr>(trMapAttr));
      input = trOp.getOutput();
    }
    b.create<func::ReturnOp>(loc);

    return success();
  }
};

class TransformMapRewriter : public PatternRewriter {
public:
  TransformMapRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

void TransformMapsUtilsTestPass::runOnOperation() {
  MLIRContext *ctx = getOperation()->getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<RemoveDimTestPattern>(ctx);
  FrozenRewritePatternSet frozenPatternSet(std::move(patterns));

  PatternApplicator applicator(frozenPatternSet);
  applicator.applyDefaultCostModel();

  TransformMapRewriter rewriter(ctx);
  LogicalResult result = applicator.matchAndRewrite(getOperation(), rewriter);
  if (failed(result)) {
    signalPassFailure();
  }
}

namespace mlir {
namespace rock {
void registerTransformMapsUtilsTestPass() {
  PassRegistration<TransformMapsUtilsTestPass>();
}
} // end namespace rock
} // end namespace mlir
