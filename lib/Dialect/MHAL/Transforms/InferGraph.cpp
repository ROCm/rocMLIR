//===- InferGraph.cpp - Kernel func call ops to mhal.launch --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the mhal.launch pattern rewriter that converts kernel
// call ops to mhal.launch ops with inferred data-dependency converted to
// explicit mhal.token based dependence graph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace mhal {
#define GEN_PASS_DEF_MHALINFERGRAPHPASS
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"
} // namespace mhal
} // namespace mlir

#define DEBUG_TYPE "mhal-infer-graph"

using namespace mlir;
namespace {
class MHALInferGraphPass
    : public mhal::impl::MHALInferGraphPassBase<MHALInferGraphPass> {

  static bool isTerminator(Operation *op) {
    return op->mightHaveTrait<OpTrait::IsTerminator>();
  }
  static bool hasSideEffects(Operation *op) {
    if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      return !memInterface.hasNoEffect();
    }
    return true;
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
        func::FuncOp func = dyn_cast<func::FuncOp>(callable);
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
        if (currentTokens.contains(itoken)) {
          createWaitOp(op, itoken);
          currentTokens.erase(itoken);
        }
      }
    }
    // Insert host synchronization before terminator or op with side effects.
    if (isTerminator(op) || hasSideEffects(op)) {
      for (auto token : currentTokens) {
        createWaitOp(op, token);
      }
      currentTokens.clear();
    }

    return success();
  }

  // Replaces asyncOp with a clone that returns a token.
  LogicalResult rewriteCallOp(func::CallOp op, func::FuncOp func) {
    OpBuilder builder(op);
    // builder.setInsertionPoint(op);
    // Find tokens related to inputs
    SmallVector<Value, 4> tokens;
    for (auto operand : op.getOperands()) {
      // Disallow memrefs to avoid aliasing
      if (operand.getType().isa<MemRefType>())
        return op.emitOpError("unsupported MemRefTypes");

      if (auto itoken = res2tokens.lookup(operand)) {
        tokens.push_back(itoken);

        // remove tokens that are consumed, the chain will satisfy the
        // final block await
        currentTokens.erase(itoken);
      }
    }

    // Clone the op to return a token in addition to the other results.
    auto alaunch = builder.create<mhal::LaunchOp>(op.getLoc(), func, tokens,
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
    auto awaitOp = builder.create<mhal::AwaitOp>(op->getLoc(), token);
    auto results = awaitOp.getResults();
    if (results.size())
      res = results.front();
    return res;
  }

public:
  // Replaces synchronous call ops in the op's region with asynchronous ones and
  // inserts the necessary synchronization (as async.await ops). Assumes
  // sequential execution semantics and that no asynchronous ops yet.
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func->hasAttr("kernel")) {
      auto walker = [this](Block *block) {
        for (Operation &op : make_early_inc_range(*block)) {
          if (failed(visit(&op)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      };

      if (getOperation()->walk(walker).wasInterrupted())
        signalPassFailure();
    }
  }
};
} // namespace
