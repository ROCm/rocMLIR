//===- InsertLoads.cpp - Insert rock.load ops for tensor arguments --------===//
//
// Copyright 2026 The MLIR Authors.
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
// =============================================================================
//
// This pass inserts rock.load operations for tensor operands that are
// consumed by non-viewlike operations. The rock.load op serves as a
// placeholder marking where loads from global memory need to happen.
//
// The load is inserted on the TRANSFORMED tensor (after transforms), not on
// the original function argument. This means the load captures the full
// view transformation chain.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKINSERTLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-insert-loads"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Check if a value traces back to a function argument through view-like ops.
static bool tracesToFunctionArg(Value val, func::FuncOp funcOp) {
  Value rootVal = rock::findBlockArgument(val).value_or(nullptr);
  auto blockArg = llvm::dyn_cast_or_null<BlockArgument>(rootVal);
  if (!blockArg)
    return false;
  return blockArg.getOwner() == &funcOp.getBody().front();
}

struct RockInsertLoadsPass
    : public rock::impl::RockInsertLoadsPassBase<RockInsertLoadsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockInsertLoadsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    return;
  }

  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Track which values we've already created loads for.
  // Maps transformed value -> rock.load result
  // This ensures deduplication: if the same transformed tensor is used by
  // multiple non-viewlike ops, we only create one load.
  llvm::DenseMap<Value, Value> valueToLoadResult;

  // Collect all non-viewlike operations that consume tensor values
  SmallVector<Operation *> nonViewLikeOps;
  funcOp.walk([&](Operation *op) {
    // Skip terminators and view-like operations
    if (op->hasTrait<OpTrait::IsTerminator>())
      return;
    if (isa<ViewLikeOpInterface>(op))
      return;
    // Skip rock.load and rock.store ops themselves
    if (isa<LoadOp, StoreOp>(op))
      return;

    nonViewLikeOps.push_back(op);
  });

  // For each non-viewlike op, check its operands and insert rock.load ops
  // for those that trace back to function arguments.
  for (Operation *op : nonViewLikeOps) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value val = operand.get();

      // Only handle tensor types
      if (!isa<RankedTensorType>(val.getType()))
        continue;

      // Check if this value traces back to a function argument
      if (!tracesToFunctionArg(val, funcOp))
        continue;

      // Check if we've already created a load for this exact value
      if (valueToLoadResult.count(val)) {
        // Replace the operand with the existing load result
        operand.set(valueToLoadResult[val]);
        continue;
      }

      // Create rock.load for this transformed tensor value
      // Insert the load right after the value's defining op (or at function
      // start if it's a block argument)
      if (Operation *defOp = val.getDefiningOp()) {
        builder.setInsertionPointAfter(defOp);
      } else {
        builder.setInsertionPointToStart(&funcOp.getBody().front());
      }

      auto loadOp =
          LoadOp::create(builder, funcOp.getLoc(), val.getType(), val);
      Value loadResult = loadOp.getResult();

      // Record this load for deduplication
      valueToLoadResult[val] = loadResult;

      // Replace the operand with the load result
      operand.set(loadResult);

      LLVM_DEBUG(llvm::dbgs()
                 << "Created rock.load for value: " << val << "\n");
    }
  }
}
