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

#include "PassDetail.h"

#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// Remove these includes after we're not working with a test pass upstream
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Transforms/FoldUtils.h"

#define DEBUG_TYPE "miopen-clean-math"

using namespace mlir;
// Remove after upstream integrates
using namespace mlir::dataflow;

namespace {
struct MIOpenCleanMathPass
    : public MIOpenCleanMathPassBase<MIOpenCleanMathPass> {
  void runOnOperation() override;
};
} // end namespace

/// From
/// external/llvm-project/mlir/test/lib/Transforms/TestIntRangeInference.cpp
/// This is all temporary until upstream integrates the inference framework

static LogicalResult replaceWithConstant(DataFlowSolver &solver, OpBuilder &b,
                                         OperationFolder &folder, Value value) {
  auto *maybeInferredRange =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!maybeInferredRange || maybeInferredRange->isUninitialized())
    return failure();
  const ConstantIntRanges &inferredRange =
      maybeInferredRange->getValue().getValue();
  Optional<APInt> maybeConstValue = inferredRange.getConstantValue();
  if (!maybeConstValue.hasValue())
    return failure();

  Operation *maybeDefiningOp = value.getDefiningOp();
  Dialect *valueDialect =
      maybeDefiningOp ? maybeDefiningOp->getDialect()
                      : value.getParentRegion()->getParentOp()->getDialect();
  Attribute constAttr = b.getIntegerAttr(value.getType(), *maybeConstValue);
  Value constant = folder.getOrCreateConstant(b, valueDialect, constAttr,
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

void MIOpenCleanMathPass::runOnOperation() {
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
  postAnalysisPipeline.addPass(
      arith::createArithmeticUnsignedWhenEquivalentPass());
  (void)runPipeline(postAnalysisPipeline, op);
}

std::unique_ptr<Pass> mlir::miopen::createMIOpenCleanMathPass() {
  return std::make_unique<MIOpenCleanMathPass>();
}
