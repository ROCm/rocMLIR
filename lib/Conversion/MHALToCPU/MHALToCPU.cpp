//===- MHALToCPU.cpp - Convert MHAL to CPU dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MHALToCPU/MHALToCPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "convert-mhal-to-cpu"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMHALTOCPUPASS
#include "mlir/Conversion/MHALPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mhal;

//===----------------------------------------------------------------------===//
// Convert MHAL dialect types to CPU types.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Convert mhal.launch ops with 'cpu' target to cpu.launch_func ops with
// required memory staging.
//===----------------------------------------------------------------------===//

namespace {
// Helper to pull out the called func
static Optional<func::FuncOp> getCalledFunc(mhal::LaunchOp op) {
  CallOpInterface callIf(op);
  if (auto *callable = callIf.resolveCallable()) {
    if (auto func = dyn_cast<func::FuncOp>(callable))
      return func;
  }

  return std::nullopt;
}

struct LaunchRewritePattern : public OpRewritePattern<mhal::LaunchOp> {
  using OpRewritePattern<mhal::LaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhal::LaunchOp op,
                                PatternRewriter &rw) const override {
    Location loc = op.getLoc();

    assert(op->getNumResults() == 1); // only 1 mhal.token

    if (auto func = getCalledFunc(op)) {
#if 0
      if (toCoroutine) {
        // Collect all outlined function inputs.
        SmallVector<mlir::Value> operands(launch->getOperands());

        // Collect types for the outlined function inputs and outputs.
        auto typesRange = llvm::map_range(
                                          operands, [](Value value) { return value.getType(); });
        SmallVector<Type, 4> inputTypes(typesRange.begin(), typesRange.end());
        auto outputTypes = launch.getResultTypes();

        auto funcType = FunctionType::get(ctx, inputTypes, outputTypes);
        func.setType(funcType);

        // insert tokens and awaits
        auto *block = &func.front();
        auto builder = ImplicitLocOpBuilder::atBlockBegin(loc, block);
        for (auto pair : llvm::enumerate(launch.getDependencies())) {
          auto dep = pair.value();
          auto barg =
            block->insertArgument(pair.index(), dep.getType(), dep.getLoc());
          builder.create<AwaitOp>(barg);
        }

        // replace return with async.yield
        for (Block &block : func.getBlocks()) {
          Operation *terminator = block.getTerminator();
          if (auto returnOp = dyn_cast<func::ReturnOp>(*terminator)) {
            ImplicitLocOpBuilder builder(loc, returnOp);
            builder.create<YieldOp>(returnOp.getOperands());
            returnOp.erase();
          }
        }


        // Adding entry/cleanup/suspend blocks.
        CoroMachinery coro = buildCoroutineLogic(launch, func, operands);
        return {func, coro};

      } else {
#endif

      // Replace the original `async.execute` with a call to outlined
      // function.
      rw.create<func::CallOp>(loc, *func, op.getArgOperands());

      // make dummy token
      // auto retToken =
      //   rw.create<RuntimeCreateOp>(TokenType::get(ctx)).getResult();
      // cb.create<RuntimeSetAvailableOp>(retToken);

      Value empty;
      op->replaceAllUsesWith(ValueRange(empty));
      op->erase();

      return success();
    }
    return rw.notifyMatchFailure(op, "func not found");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Remove all mhal.await ops
//===----------------------------------------------------------------------===//

namespace {
struct AwaitRewritePattern : public OpRewritePattern<mhal::AwaitOp> {
  using OpRewritePattern<mhal::AwaitOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhal::AwaitOp op,
                                PatternRewriter &rw) const override {
    rw.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//

namespace {
struct ConvertMHALToCPUPass
    : public impl::ConvertMHALToCPUPassBase<ConvertMHALToCPUPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertMHALToCPUPass::runOnOperation() {
  auto op = getOperation();
  MLIRContext *ctx = op->getContext();

  // Convert mhal.launch to func.call ops, remove all mhal.await ops
  RewritePatternSet patterns(ctx);
  patterns.add<LaunchRewritePattern>(ctx);
  patterns.add<AwaitRewritePattern>(ctx);

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}
