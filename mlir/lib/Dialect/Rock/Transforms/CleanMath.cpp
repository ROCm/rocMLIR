//===- CleanMath.cpp - Clean up math after lowering/unrolling loops  ---===//
//
// Copyright 2022 AMD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// Remove these includes after we're not working with a test pass upstream
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCLEANMATHPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-clean-math"

using namespace mlir;
// Remove after upstream integrates
using namespace mlir::dataflow;

namespace {
struct RockCleanMathPass
    : public rock::impl::RockCleanMathPassBase<RockCleanMathPass> {
  void runOnOperation() override;
};

/// This is a rewrite for a specific and irritating pattern that shows up in our
/// code a bunch on account of the whole index diff thing.
/// Specifically:
///    %prev = arith.remui %blah, %c1
///    %cur = arith.addi %prev, %c(M < N, usually 1)
///    %curModulus = arith.remui %cur, %cN
///    %checkWrap = arith.subi %curWrapped, %prev
///    %val = arith.addi %checkWrap, %cN
///
/// If %prev + 1 is still less than N, this is (%prev + 1) mod N - %prev + N
struct SimplifyCheckingSmallConstantOverflow
    : public OpRewritePattern<arith::RemUIOp> {
  using OpRewritePattern<arith::RemUIOp>::OpRewritePattern;
};
} // end namespace

/// This function hasn't come from anywhere and is relying on the overall
/// tests of the integer range inference implementation for its correctness.
static LogicalResult deleteTrivialRemainder(DataFlowSolver &solver,
                                            Operation &op) {
  if (!isa<arith::RemSIOp, arith::RemUIOp>(op))
    return failure();
  Value result = op.getResult(0);
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);
  auto rhsConstVal = rhs.getDefiningOp<arith::ConstantOp>();
  if (!rhsConstVal)
    return failure();
  APInt modulus = cast<IntegerAttr>(rhsConstVal.getValue()).getValue();
  if (!modulus.isStrictlyPositive())
    return failure();
  auto *maybeLhsRange = solver.lookupState<IntegerValueRangeLattice>(lhs);
  if (!maybeLhsRange || maybeLhsRange->getValue().isUninitialized())
    return failure();
  const ConstantIntRanges &lhsRange = maybeLhsRange->getValue().getValue();
  const APInt &min =
      llvm::isa<arith::RemUIOp>(op) ? lhsRange.umin() : lhsRange.smin();
  const APInt &max =
      llvm::isa<arith::RemUIOp>(op) ? lhsRange.umax() : lhsRange.smax();
  // The minima and maxima here are given as closed ranges, we must be strictly
  // less than the modulus.
  if (min.isNegative() || min.uge(modulus))
    return failure();
  if (max.isNegative() || max.uge(modulus))
    return failure();
  if (!min.ule(max))
    return failure();

  // With all those conditions out of the way, we know thas this invocation of
  // a remainder is a noop because the input is strictly within the range
  // [0, modulus), so get rid of it.
  result.replaceAllUsesWith(lhs);
  op.erase();
  return success();
}

/// From
/// external/llvm-project/mlir/test/lib/Transforms/TestIntRangeInference.cpp
/// This is all temporary until upstream integrates the inference framework

static LogicalResult replaceWithConstant(DataFlowSolver &solver, OpBuilder &b,
                                         OperationFolder &folder, Value value) {
  auto *maybeInferredRange =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
    return failure();
  const ConstantIntRanges &inferredRange =
      maybeInferredRange->getValue().getValue();
  std::optional<APInt> maybeConstValue = inferredRange.getConstantValue();
  if (!maybeConstValue.has_value())
    return failure();

  Operation *maybeDefiningOp = value.getDefiningOp();
  Dialect *valueDialect =
      maybeDefiningOp ? maybeDefiningOp->getDialect()
                      : value.getParentRegion()->getParentOp()->getDialect();
  Attribute constAttr = b.getIntegerAttr(value.getType(), *maybeConstValue);
  Value constant =
      folder.getOrCreateConstant(b.getInsertionBlock(), valueDialect, constAttr,
                                 value.getType(), value.getLoc());
  if (!constant)
    return failure();

  value.replaceAllUsesWith(constant);
  return success();
}

static void rewrite(DataFlowSolver &solver, MLIRContext *context,
                    MutableArrayRef<Region> initialRegions) {
  SmallVector<Block *> worklist;
  auto addToWorklist = [&](MutableArrayRef<Region> regions) {
    for (Region &region : regions)
      for (Block &block : llvm::reverse(region))
        worklist.push_back(&block);
  };

  OpBuilder builder(context);
  OperationFolder folder(context);

  addToWorklist(initialRegions);
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      builder.setInsertionPoint(&op);

      if (succeeded(deleteTrivialRemainder(solver, op)))
        continue;
      // Replace any result with constants.
      bool replacedAll = op.getNumResults() != 0;
      for (Value res : op.getResults())
        replacedAll &=
            succeeded(replaceWithConstant(solver, builder, folder, res));

      // If all of the results of the operation were replaced, try to erase
      // the operation completely.
      if (replacedAll && wouldOpBeTriviallyDead(&op)) {
        assert(op.use_empty() && "expected all uses to be replaced");
        op.erase();
        continue;
      }

      // Add any the regions of this operation to the worklist.
      addToWorklist(op.getRegions());
    }

    // Replace any block arguments with constants.
    builder.setInsertionPointToStart(block);
    for (BlockArgument arg : block->getArguments())
      (void)replaceWithConstant(solver, builder, folder, arg);
  }
}
// end removeable part

void RockCleanMathPass::runOnOperation() {
  func::FuncOp op = getOperation();

  // Run canonicalizers and CSE to clean up the code created by loop unrolling
  OpPassManager preAnalysisPipeline("func.func");
  preAnalysisPipeline.addPass(createCanonicalizerPass());
  preAnalysisPipeline.addPass(createCSEPass());
  (void)runPipeline(preAnalysisPipeline, op);

  // In the future, this'll just be another pass to add to the pipeline
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();
  rewrite(solver, op->getContext(), op->getRegions());
  // end removable part

  OpPassManager postAnalysisPipeline("func.func");
  postAnalysisPipeline.addPass(createCanonicalizerPass());
  postAnalysisPipeline.addPass(arith::createArithUnsignedWhenEquivalentPass());
  (void)runPipeline(postAnalysisPipeline, op);
}
