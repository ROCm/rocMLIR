//===- AsyncLaunch.cpp - Convert kernel func call ops to async.launch -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the async.launch pattern rewriter that converts kernel
// call ops to async.launch ops with inferred data-dependency converted to
// explicit async.token based dependence graph.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace {
class MIOpenAsyncLaunchPass
    : public MIOpenAsyncLaunchPassBase<MIOpenAsyncLaunchPass> {

  static bool isTerminator(Operation *op) {
    return op->mightHaveTrait<OpTrait::IsTerminator>();
  }
  static bool hasSideEffects(Operation *op) {
    return !MemoryEffectOpInterface::hasNoEffect(op);
  }

  llvm::SmallDenseMap<Value, Value> res2tokens;
  llvm::SmallDenseSet<Value> currentTokens;

  // If `op` implements the AsyncOpInterface, insert a `gpu.wait async` to
  // create a current token (unless it already exists), and 'thread' that token
  // through the `op` so that it executes asynchronously.
  //
  // If `op` is a terminator or an op with side-effects, insert a `gpu.wait` to
  // host-synchronize execution. A `!gpu.async.token` will therefore only be
  // used inside of its block and GPU execution will always synchronize with
  // the host at block boundaries.
  LogicalResult visit(Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      CallOpInterface callIf(call);
      if (auto *callable = callIf.resolveCallable()) {
        FuncOp func = dyn_cast<FuncOp>(callable);
        assert(func);
        if (func->hasAttr("kernel")) {
          // Replace call op with async version.
          return rewriteCallOp(call, func);
        }
      }
    }
    // Insert host sync before operation that reads an async::value
    for (auto operand : op->getOperands()) {
      if (auto itoken = res2tokens.lookup(operand)) {
        createWaitOp(op, itoken);
      }
    }
    // Insert host synchronization before terminator or op with side effects.
    if ((isTerminator(op) || hasSideEffects(op)) && currentTokens.size()) {
      for (auto token : currentTokens) {
        createWaitOp(op, token);
      }
      currentTokens.clear();
    }

    return success();
  }

  // Replaces asyncOp with a clone that returns a token.
  LogicalResult rewriteCallOp(func::CallOp op, FuncOp func) {
    OpBuilder builder(op);
    // builder.setInsertionPoint(op);
    // Find tokens related to inputs
    SmallVector<Value, 4> tokens;
    for (auto operand : op.getOperands()) {
      if (auto itoken = res2tokens.lookup(operand)) {
        tokens.push_back(itoken);

        // remove tokens that are consumed, the chain will satisfy the
        // final block await
        currentTokens.erase(itoken);
      }
    }

    // Clone the op to return a token in addition to the other results.
    auto alaunch = builder.create<async::LaunchOp>(op.getLoc(), func, tokens,
                                                   op->getOperands());

    // Replace the op with the async clone.
    auto results = alaunch->getResults();
    auto token = results.front();
    results = results.drop_front();

    // associate all results with the result token
    for (auto res : results)
      res2tokens.insert({res, token});

    currentTokens.insert(token);

    op->replaceAllUsesWith(results);
    op->erase();

    return success();
  }

  Value createWaitOp(Operation *op, Value token) {
    Value res;
    assert(token);
    OpBuilder builder(op);
    auto awaitOp = builder.create<async::AwaitOp>(op->getLoc(), token);
    auto results = awaitOp.getResults();
    if (results.size())
      res = results.front();
    return res;
  }

public:
  // Replaces synchronous GPU ops in the op's region with asynchronous ones and
  // inserts the necessary synchronization (as gpu.wait ops). Assumes sequential
  // execution semantics and that no GPU ops are asynchronous yet.
  void runOnOperation() override {
    auto walker = [this](Block *block) {
      for (Operation &op : make_early_inc_range(*block)) {
        if (failed(visit(&op)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    };

    if (getOperation()->walk(walker).wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::miopen::createMIOpenAsyncLaunchPass() {
  return std::make_unique<MIOpenAsyncLaunchPass>();
}
