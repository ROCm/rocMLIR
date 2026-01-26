//===- LowerLoads.cpp - Lower rock.load ops for blockwise loads -----------===//
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
// This pass runs AFTER GridwiseGemmToBlockwise and processes rock.load markers.
//
// InsertLoads has already placed rock.load ops to mark where loads should happen.
// This pass traces through rock.load ops and fusion patterns to create the
// actual blockwise_load_tile ops with proper transform chains.
//
// Example:
//   Before (after InsertLoads + ToBlockwise):
//     %t1 = rock.transform %arg0 by <pre_transform>
//     %l1 = rock.load %t1
//     %post = rock.transform %l1 by <post_transform>
//     %loaded = rock.blockwise_load_tile %post[indices]
//
//   After:
//     %t1 = rock.transform %arg0 by <pre_transform>
//     %combined = rock.transform %t1 by <post_transform>
//     %loaded = rock.blockwise_load_tile %combined[indices]
//
// Fusion example:
//   Before:
//     %t1 = rock.transform %arg0
//     %l1 = rock.load %t1
//     %t2 = rock.transform %arg1
//     %l2 = rock.load %t2
//     %fused = arith.addf %l1, %l2
//     %post = rock.transform %fused by <post_transform>
//     %loaded = rock.blockwise_load_tile %post[indices]
//
//   After:
//     %t1 = rock.transform %arg0
//     %post1 = rock.transform %t1 by <post_transform>
//     %loaded1 = rock.blockwise_load_tile %post1[indices]
//     %t2 = rock.transform %arg1
//     %post2 = rock.transform %t2 by <post_transform>
//     %loaded2 = rock.blockwise_load_tile %post2[indices]
//     %fused = arith.addf %loaded1, %loaded2
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-loads"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Check if an operation is a fusion op (arith or math dialect).
static bool isFusionOp(Operation *op) {
  return isa<arith::ArithDialect, math::MathDialect>(op->getDialect());
}

/// Recursively reconstruct a value for blockwise loading.
/// Traces through transforms, rock.load ops, and fusion ops.
static Value reconstructForBlockwiseLoad(
    OpBuilder &builder, Location loc, Value originalVal,
    ArrayRef<TransformMapAttr> postTransforms, // Transforms accumulated (in reverse order)
    ValueRange blockIndices,                   // Indices for blockwise_load_tile
    Type tileType,                             // Result type of blockwise_load_tile
    IRMapping &valueMapping                    // Maps original values to loaded values
) {
  // Check if we've already processed this value
  if (valueMapping.contains(originalVal)) {
    return valueMapping.lookup(originalVal);
  }

  Operation *defOp = originalVal.getDefiningOp();
  if (!defOp) {
    // Block argument without rock.load - shouldn't happen in normal flow
    llvm_unreachable("Unexpected block argument without rock.load");
  }

  // Case 1: rock.load - trace through to get actual source and create load
  if (auto loadOp = dyn_cast<LoadOp>(defOp)) {
    Value loadSource = loadOp.getSource();
    
    // Collect transforms from load source to root (in reverse order)
    SmallVector<TransformMapAttr> preTransforms;
    auto [source, _] = rock::untransform(loadSource, preTransforms);
    
    // Combine: post-transforms (reversed) then pre-transforms (reversed)
    // rock::transform expects [last_to_apply, ..., first_to_apply] order
    SmallVector<Attribute> combinedTransforms;
    combinedTransforms.append(postTransforms.begin(), postTransforms.end());
    combinedTransforms.append(preTransforms.begin(), preTransforms.end());
    
    // Apply combined transforms using rock::transform utility
    if (!combinedTransforms.empty()) {
      ArrayAttr transformsAttr = builder.getArrayAttr(combinedTransforms);
      source = rock::transform(builder, source, transformsAttr);
    }
    
    // Create blockwise_load_tile
    auto loadTileOp = BlockwiseLoadTileOp::create(builder, loc, tileType,
                                                   source, blockIndices);
    valueMapping.map(originalVal, loadTileOp.getResult());
    return loadTileOp.getResult();
  }

  // Case 2: rock.transform - accumulate and recurse
  if (auto transformOp = dyn_cast<TransformOp>(defOp)) {
    SmallVector<TransformMapAttr> newPostTransforms;
    newPostTransforms.push_back(transformOp.getTransform());
    newPostTransforms.append(postTransforms.begin(), postTransforms.end());
    
    Value result = reconstructForBlockwiseLoad(
        builder, loc, transformOp.getInput(), newPostTransforms,
        blockIndices, tileType, valueMapping);
    valueMapping.map(originalVal, result);
    return result;
  }

  // Case 3: Fusion op (arith or math dialect)
  if (isFusionOp(defOp)) {
    // Reconstruct each operand
    IRMapping fusionMapping;
    for (Value operand : defOp->getOperands()) {
      Value reconstructed = reconstructForBlockwiseLoad(
          builder, loc, operand, postTransforms,
          blockIndices, tileType, valueMapping);
      fusionMapping.map(operand, reconstructed);
    }

    // Clone the fusion op with reconstructed operands
    Operation *cloned = builder.clone(*defOp, fusionMapping);
    cloned->getResult(0).setType(tileType);
    valueMapping.map(originalVal, cloned->getResult(0));
    return cloned->getResult(0);
  }

  // Unknown op - shouldn't happen for valid input
  llvm_unreachable("Unexpected op in load chain");
}

struct RockLowerLoadsPass
    : public rock::impl::RockLowerLoadsPassBase<RockLowerLoadsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockLowerLoadsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    return;
  }

  // Collect all blockwise_load_tile ops
  SmallVector<BlockwiseLoadTileOp> loadTilesToProcess;
  funcOp.walk([&](BlockwiseLoadTileOp loadTileOp) {
    loadTilesToProcess.push_back(loadTileOp);
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << loadTilesToProcess.size()
                          << " blockwise_load_tile ops to process\n");

  // Process each blockwise_load_tile
  for (BlockwiseLoadTileOp loadTileOp : loadTilesToProcess) {
    OpBuilder builder(loadTileOp);
    Location loc = loadTileOp.getLoc();

    Value source = loadTileOp.getSource();
    ValueRange indices = loadTileOp.getSourceIndices();
    Type resultType = loadTileOp.getResult().getType();

    LLVM_DEBUG(llvm::dbgs() << "Processing: " << loadTileOp << "\n");

    IRMapping valueMapping;
    SmallVector<TransformMapAttr> emptyTransforms;
    Value newResult = reconstructForBlockwiseLoad(
        builder, loc, source, emptyTransforms,
        indices, resultType, valueMapping);

    loadTileOp.getResult().replaceAllUsesWith(newResult);
    loadTileOp.erase();

    LLVM_DEBUG(llvm::dbgs() << "  Replaced with: " << newResult << "\n");
  }

  // Clean up dead ops
  bool changed = true;
  while (changed) {
    changed = false;

    funcOp.walk([&](LoadOp loadOp) {
      if (loadOp.getResult().use_empty()) {
        loadOp.erase();
        changed = true;
      }
    });

    funcOp.walk([&](TransformOp transformOp) {
      if (transformOp.getOutput().use_empty()) {
        transformOp.erase();
        changed = true;
      }
    });

    funcOp.walk([&](Operation *op) {
      if (isFusionOp(op) &&
          op->getNumResults() == 1 &&
          op->getResult(0).use_empty()) {
        op->erase();
        changed = true;
      }
    });
  }
}
