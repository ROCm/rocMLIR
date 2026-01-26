//===- InsertOutputFusionLoads.cpp - Insert BlockwiseLoadTileOp for output fusions ===//
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
// This pass runs AFTER GridwiseGemmToBlockwise and BEFORE RockLowerLoads.
// It creates rock.blockwise_load_tile for rock.load ops used in output fusions.
//
// Output fusion loads are rock.load ops that:
// 1. Are NOT reachable from any existing rock.blockwise_load_tile (input loads)
// 2. Feed into fusion ops (arith/math) that operate on the GEMM result
//
// Example:
//   Before (after GridwiseGemmToBlockwise):
//     %result = scf.for ... {
//       %loadedA = rock.blockwise_load_tile %a[indices]
//       ...
//     }
//     %fusionRoot = rock.untile %result
//     %bias_t = rock.transform %bias
//     %bias_loaded = rock.load %bias_t
//     %fused = arith.addf %fusionRoot, %bias_loaded
//     %out = rock.store %fused to %dest
//
//   After:
//     %result = scf.for ... { ... }
//     %fusionRoot = rock.untile %result
//     %bias_t = rock.transform %bias
//     %bias_loaded = rock.load %bias_t            // Still exists for LowerLoads to trace
//     %bias_wrapped = rock.transform %bias_loaded by <output_grid_subtile>
//     %bias_tile = rock.blockwise_load_tile %bias_wrapped[g_block, m_block, n_block]
//     %fused = arith.addf %fusionRoot, %bias_tile  // now operates on tiles
//     %out = rock.store %fused to %dest
//
//   The chain BlockwiseLoadTileOp -> transform -> rock.load allows LowerLoads
//   to trace back through rock.load and combine the pre-load transforms with
//   the output grid subtile transforms.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKINSERTOUTPUTFUSIONLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-insert-output-fusion-loads"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Check if an operation is a fusion op (arith or math dialect).
static bool isFusionOp(Operation *op) {
  return isa<arith::ArithDialect, math::MathDialect>(op->getDialect());
}

/// Collect all rock.load ops that are reachable from existing BlockwiseLoadTileOp.
/// These are input fusion loads and should NOT be processed by this pass.
static void collectInputFusionLoads(func::FuncOp funcOp,
                                    llvm::DenseSet<LoadOp> &inputLoads) {
  // For each BlockwiseLoadTileOp, trace back through its source to find all
  // rock.load ops that feed into it
  funcOp.walk([&](BlockwiseLoadTileOp loadTileOp) {
    SmallVector<Value> worklist;
    worklist.push_back(loadTileOp.getSource());
    
    while (!worklist.empty()) {
      Value val = worklist.pop_back_val();
      Operation *defOp = val.getDefiningOp();
      if (!defOp)
        continue;
      
      if (auto loadOp = dyn_cast<LoadOp>(defOp)) {
        inputLoads.insert(loadOp);
        // Don't trace further - we found the load marker
        continue;
      }
      
      if (auto viewLike = dyn_cast<ViewLikeOpInterface>(defOp)) {
        worklist.push_back(viewLike.getViewSource());
        continue;
      }
      
      // For fusion ops, trace through all operands
      if (isFusionOp(defOp)) {
        for (Value operand : defOp->getOperands()) {
          worklist.push_back(operand);
        }
      }
    }
  });
}

/// Find existing BlockwiseLoadTileOp to extract grid coordinates.
/// Returns the g_block, m_block, n_block indices.
static SmallVector<Value> findGridCoordinates(func::FuncOp funcOp) {
  SmallVector<Value> coords;
  
  funcOp.walk([&](BlockwiseLoadTileOp loadTileOp) {
    if (!coords.empty())
      return WalkResult::interrupt();
    
    // Input loads have indices [kIter, g_block, m_block, n_block]
    // We want the last 3 (g_block, m_block, n_block)
    ValueRange indices = loadTileOp.getSourceIndices();
    if (indices.size() >= 3) {
      // Take last 3 indices
      size_t offset = indices.size() - 3;
      coords.push_back(indices[offset]);     // g_block
      coords.push_back(indices[offset + 1]); // m_block
      coords.push_back(indices[offset + 2]); // n_block
    }
    return WalkResult::interrupt();
  });
  
  return coords;
}

/// Get output tile shape from UntileOp source type.
/// The source is the tile from the GEMM loop, which has shape [mPerBlock, nPerBlock].
static std::pair<int64_t, int64_t> getOutputTileShape(UntileOp fusionRoot) {
  auto sourceType = cast<RankedTensorType>(fusionRoot.getSource().getType());
  ArrayRef<int64_t> shape = sourceType.getShape();
  // Tile shape is [mPerBlock, nPerBlock]
  assert(shape.size() == 2 && "Expected 2D tile from GEMM loop");
  return {shape[0], shape[1]};
}

/// Get the output shape (G, M, N) and bidGridLengths from the output tensor.
static SmallVector<int64_t, 3>
getOutputInfo(UntileOp fusionRoot, int64_t mPerBlock, int64_t nPerBlock) {
  auto resultType = cast<RankedTensorType>(fusionRoot.getResult().getType());
  ArrayRef<int64_t> shape = resultType.getShape();
  // Output shape is [G, M, N]
  assert(shape.size() == 3 && "Expected 3D output shape [G, M, N]");
  
  int64_t G = shape[0];
  int64_t M = shape[1];
  int64_t N = shape[2];
  
  int64_t mBlocks = M / mPerBlock;
  int64_t nBlocks = N / nPerBlock;
  
  return {G, mBlocks, nBlocks};
}

struct RockInsertOutputFusionLoadsPass
    : public rock::impl::RockInsertOutputFusionLoadsPassBase<
          RockInsertOutputFusionLoadsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockInsertOutputFusionLoadsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    return;
  }

  // Step 1: Collect all input fusion loads (reachable from BlockwiseLoadTileOp)
  llvm::DenseSet<LoadOp> inputFusionLoads;
  collectInputFusionLoads(funcOp, inputFusionLoads);
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << inputFusionLoads.size()
                          << " input fusion loads\n");

  // Step 2: Find grid coordinates from existing BlockwiseLoadTileOp
  SmallVector<Value> gridCoords = findGridCoordinates(funcOp);
  if (gridCoords.size() != 3) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find grid coordinates\n");
    return;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found grid coordinates: g_block=" << gridCoords[0]
                          << ", m_block=" << gridCoords[1]
                          << ", n_block=" << gridCoords[2] << "\n");

  // Step 3: Find UntileOp to get tile shape
  UntileOp fusionRootOp = nullptr;
  funcOp.walk([&](UntileOp op) {
    fusionRootOp = op;
    return WalkResult::interrupt();
  });
  
  if (!fusionRootOp) {
    LLVM_DEBUG(llvm::dbgs() << "No UntileOp found\n");
    return;
  }
  
  auto [mPerBlock, nPerBlock] = getOutputTileShape(fusionRootOp);
  auto bidGridLengths = getOutputInfo(fusionRootOp, mPerBlock, nPerBlock);
  
  LLVM_DEBUG(llvm::dbgs() << "mPerBlock=" << mPerBlock
                          << ", nPerBlock=" << nPerBlock << "\n");

  // Step 4: Collect output fusion loads (rock.load ops NOT in inputFusionLoads)
  SmallVector<LoadOp> outputFusionLoads;
  funcOp.walk([&](LoadOp loadOp) {
    if (!inputFusionLoads.contains(loadOp)) {
      outputFusionLoads.push_back(loadOp);
    }
  });
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << outputFusionLoads.size()
                          << " output fusion loads\n");

  if (outputFusionLoads.empty()) {
    return;
  }

  // Step 5: Create BlockwiseLoadTileOp for each output fusion load
  OpBuilder builder(funcOp.getContext());
  
  for (LoadOp loadOp : outputFusionLoads) {
    builder.setInsertionPointAfter(loadOp);
    Location loc = loadOp.getLoc();
    
    // The source of BlockwiseLoadTileOp should be the rock.load RESULT
    // (wrapped with output transforms), not the input to rock.load.
    // This ensures LowerLoads can trace back through rock.load to find
    // the pre-transforms and create proper loads.
    Value loadResult = loadOp.getResult();
    
    // Collect existing uses before we create new ops
    SmallVector<OpOperand *> existingUses;
    for (OpOperand &use : loadResult.getUses()) {
      existingUses.push_back(&use);
    }
    
    // Compute output transforms for this load
    FailureOr<RegsAsMatrixSubTiles> maybeOutputViews = computeOutputTransforms(
        builder, loc, mPerBlock, nPerBlock, bidGridLengths);
    
    if (failed(maybeOutputViews)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to compute output transforms for: "
                              << loadOp << "\n");
      continue;
    }
    
    // Apply the grid subtile transform to the rock.load result
    // Chain: BlockwiseLoadTileOp.source -> transform -> rock.load result
    // LowerLoads will trace back through this and find rock.load
    Value wrappedSource = transform(builder, loadResult,
                                    maybeOutputViews->gridSubTile);
    
    // Determine the tile type (last 2 dimensions are the tile)
    auto sourceType = cast<RankedTensorType>(wrappedSource.getType());
    auto wrappedShape = sourceType.getShape();
    auto tileType = RankedTensorType::get(
        wrappedShape.take_back(2), sourceType.getElementType());
    
    // Create BlockwiseLoadTileOp with output indices [g_block, m_block, n_block]
    auto loadTileOp = BlockwiseLoadTileOp::create(
        builder, loc, tileType, wrappedSource, gridCoords);
    
    // Create UntileOp to convert tile type back to full tensor type.
    // This maintains type compatibility with the original rock.load result.
    // The UntileOp acts as a temporary bridge - we'll fix this later.
    auto untileOp = UntileOp::create(
        builder, loc, loadResult.getType(), loadTileOp.getResult());
    
    // Replace only the existing uses of rock.load result (not the new transform)
    // with the UntileOp result (which has the original full tensor type).
    // The transform op keeps using loadResult so LowerLoads can trace back.
    for (OpOperand *use : existingUses) {
      use->set(untileOp.getResult());
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Created BlockwiseLoadTileOp for output fusion load: "
                            << loadTileOp << "\n"
                            << "  with UntileOp: " << untileOp << "\n");
  }

  // Note: We don't clean up rock.load ops here because they are still used
  // by the transform ops we created. LowerLoads will clean them up after
  // processing the BlockwiseLoadTileOp.
}
