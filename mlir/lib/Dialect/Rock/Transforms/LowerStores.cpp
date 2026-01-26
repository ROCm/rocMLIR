//===- LowerStores.cpp - Lower rock.store ops to blockwise_store_tile -----===//
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
// This pass runs AFTER RockLowerLoads and converts rock.store to
// rock.blockwise_store_tile.
//
// At this point, the IR has:
// - UntileOp wrapping tile results (from GEMM and output fusion loads)
// - Fusion ops (arith.addf, etc.) operating on full tensor types
// - rock.store storing the fused result
//
// This pass:
// 1. Traces back from rock.store through fusion ops to find UntileOp
// 2. Gets the tile values from UntileOp.source()
// 3. Clones fusion ops to operate on tiles
// 4. Creates BlockwiseStoreTileOp with proper output transforms
// 5. Cleans up UntileOp and dead ops
//
// Example:
//   Before:
//     %gemm_tile = scf.for ... -> tensor<128x256xf32>
//     %gemm_full = rock.untile %gemm_tile : tile -> full
//     %bias_tile = rock.blockwise_load_tile %bias[indices] : ... -> tensor<128x256xf32>
//     %bias_full = rock.untile %bias_tile : tile -> full
//     %fused = arith.addf %gemm_full, %bias_full : tensor<full>
//     %out = rock.store %fused to %dest : tensor<full> -> tensor<flat>
//
//   After:
//     %gemm_tile = scf.for ... -> tensor<128x256xf32>
//     %bias_tile = rock.blockwise_load_tile %bias[indices] : ... -> tensor<128x256xf32>
//     %fused_tile = arith.addf %gemm_tile, %bias_tile : tensor<128x256xf32>
//     %out = rock.blockwise_store_tile %fused_tile -> %dest[indices] by set
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERSTORESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-stores"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Check if an operation is a fusion op (arith or math dialect).
static bool isFusionOp(Operation *op) {
  return isa<arith::ArithDialect, math::MathDialect>(op->getDialect());
}

/// Get the tile value from a UntileOp, or return the value if it's
/// already a tile (from BlockwiseLoadTileOp or similar).
static Value getTileValue(Value val, IRMapping &fullToTileMapping) {
  // Check if we've already mapped this value
  if (fullToTileMapping.contains(val))
    return fullToTileMapping.lookup(val);
  
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return val;
  
  // UntileOp: return its source (the tile)
  if (auto fusionRootOp = dyn_cast<UntileOp>(defOp)) {
    Value tile = fusionRootOp.getSource();
    fullToTileMapping.map(val, tile);
    return tile;
  }
  
  // BlockwiseLoadTileOp result is already a tile
  if (isa<BlockwiseLoadTileOp>(defOp)) {
    fullToTileMapping.map(val, val);
    return val;
  }
  
  return val;
}

/// Find grid coordinates from existing BlockwiseLoadTileOp in the function.
static SmallVector<Value> findGridCoordinates(func::FuncOp funcOp) {
  SmallVector<Value> coords;
  
  funcOp.walk([&](BlockwiseLoadTileOp loadTileOp) {
    if (!coords.empty())
      return WalkResult::interrupt();
    
    // Output loads have indices [g_block, m_block, n_block]
    // Input loads have indices [kIter, g_block, m_block, n_block]
    // We want the last 3 (g_block, m_block, n_block)
    ValueRange indices = loadTileOp.getSourceIndices();
    if (indices.size() >= 3) {
      size_t offset = indices.size() - 3;
      coords.push_back(indices[offset]);     // g_block
      coords.push_back(indices[offset + 1]); // m_block
      coords.push_back(indices[offset + 2]); // n_block
    }
    return WalkResult::interrupt();
  });
  
  return coords;
}

/// Get tile shape from a UntileOp source type.
static std::pair<int64_t, int64_t> getTileShape(UntileOp fusionRoot) {
  auto sourceType = cast<RankedTensorType>(fusionRoot.getSource().getType());
  ArrayRef<int64_t> shape = sourceType.getShape();
  assert(shape.size() == 2 && "Expected 2D tile");
  return {shape[0], shape[1]};
}

/// Get bidGridLengths from output shape and tile shape.
static SmallVector<int64_t, 3> getBidGridLengths(RankedTensorType outputType,
                                                  int64_t mPerBlock,
                                                  int64_t nPerBlock) {
  ArrayRef<int64_t> shape = outputType.getShape();
  assert(shape.size() == 3 && "Expected 3D output shape [G, M, N]");
  
  int64_t G = shape[0];
  int64_t M = shape[1];
  int64_t N = shape[2];
  
  int64_t mBlocks = M / mPerBlock;
  int64_t nBlocks = N / nPerBlock;
  
  return {G, mBlocks, nBlocks};
}

/// Recursively convert a full-tensor value to its tile equivalent.
/// Handles UntileOp and fusion ops.
static Value convertToTile(OpBuilder &builder, Location loc, Value fullVal,
                           Type tileType, IRMapping &fullToTileMapping) {
  // Check if already converted
  if (fullToTileMapping.contains(fullVal))
    return fullToTileMapping.lookup(fullVal);
  
  Operation *defOp = fullVal.getDefiningOp();
  if (!defOp) {
    // Block argument - shouldn't happen
    return fullVal;
  }
  
  // UntileOp: return its source (the tile)
  if (auto fusionRootOp = dyn_cast<UntileOp>(defOp)) {
    Value tile = fusionRootOp.getSource();
    fullToTileMapping.map(fullVal, tile);
    return tile;
  }
  
  // Fusion op: recursively convert operands and clone with tile types
  if (isFusionOp(defOp) && defOp->getNumResults() == 1) {
    IRMapping fusionMapping;
    for (Value operand : defOp->getOperands()) {
      Value tileOperand = convertToTile(builder, loc, operand, tileType,
                                        fullToTileMapping);
      fusionMapping.map(operand, tileOperand);
    }
    
    // Clone the fusion op with tile operands
    Operation *clonedOp = builder.clone(*defOp, fusionMapping);
    clonedOp->getResult(0).setType(tileType);
    
    fullToTileMapping.map(fullVal, clonedOp->getResult(0));
    return clonedOp->getResult(0);
  }
  
  // Other ops - return as-is (shouldn't happen for well-formed IR)
  return fullVal;
}

struct RockLowerStoresPass
    : public rock::impl::RockLowerStoresPassBase<RockLowerStoresPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockLowerStoresPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    return;
  }

  // Find grid coordinates from existing BlockwiseLoadTileOp
  SmallVector<Value> gridCoords = findGridCoordinates(funcOp);
  if (gridCoords.size() != 3) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find grid coordinates\n");
    return;
  }

  // Find a UntileOp to get tile shape
  UntileOp sampleFusionRoot = nullptr;
  funcOp.walk([&](UntileOp op) {
    sampleFusionRoot = op;
    return WalkResult::interrupt();
  });
  
  if (!sampleFusionRoot) {
    LLVM_DEBUG(llvm::dbgs() << "No UntileOp found\n");
    return;
  }
  
  auto [mPerBlock, nPerBlock] = getTileShape(sampleFusionRoot);
  
  LLVM_DEBUG(llvm::dbgs() << "Tile shape: mPerBlock=" << mPerBlock
                          << ", nPerBlock=" << nPerBlock << "\n");

  // Collect all rock.store ops
  SmallVector<StoreOp> storeOps;
  funcOp.walk([&](StoreOp storeOp) {
    storeOps.push_back(storeOp);
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << storeOps.size()
                          << " rock.store ops to process\n");

  for (StoreOp storeOp : storeOps) {
    OpBuilder builder(storeOp);
    Location loc = storeOp.getLoc();
    
    Value storeSource = storeOp.getSource();  // The fused result (full tensor)
    Value storeDest = storeOp.getDest();      // The destination (transformed arg)
    
    // Get the output type for computing transforms
    auto outputType = cast<RankedTensorType>(storeSource.getType());
    SmallVector<int64_t, 3> bidGridLengths = getBidGridLengths(outputType,
                                                               mPerBlock,
                                                               nPerBlock);
    
    // Compute output transforms
    FailureOr<RegsAsMatrixSubTiles> maybeOutputViews = computeOutputTransforms(
        builder, loc, mPerBlock, nPerBlock, bidGridLengths);
    
    if (failed(maybeOutputViews)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to compute output transforms\n");
      continue;
    }
    
    ArrayAttr idToMatrixCMaps = maybeOutputViews->gridSubTile;
    auto dest = rock::transform(builder, storeDest, idToMatrixCMaps);
    
    // Determine tile type
    auto tileType = RankedTensorType::get({mPerBlock, nPerBlock},
                                          outputType.getElementType());
    
    // Convert the store source from full tensor to tile
    IRMapping fullToTileMapping;
    Value fusedTile = convertToTile(builder, loc, storeSource, tileType,
                                    fullToTileMapping);
    
    LLVM_DEBUG(llvm::dbgs() << "Converted store source to tile: " << fusedTile
                            << "\n");
    
    // Create BlockwiseStoreTileOp
    // The dest should be the rock.store destination (already transformed)
    // extraViews contains the output grid subtile transforms
    auto bstOp = BlockwiseStoreTileOp::create(
        builder, loc, storeOp.getResult().getType(),
        fusedTile, dest, gridCoords,
        StoreMethod::Set);
    
    LLVM_DEBUG(llvm::dbgs() << "Created BlockwiseStoreTileOp: " << bstOp
                            << "\n");
    
    // Replace rock.store with BlockwiseStoreTileOp result
    storeOp.getResult().replaceAllUsesWith(bstOp.getResult());
    storeOp.erase();
  }

  // Clean up dead ops (UntileOp, fusion ops, etc.)
  bool changed = true;
  while (changed) {
    changed = false;

    funcOp.walk([&](UntileOp fusionRootOp) {
      if (fusionRootOp.getResult().use_empty()) {
        fusionRootOp.erase();
        changed = true;
      }
    });

    funcOp.walk([&](BlockwiseLoadTileOp loadTileOp) {
      if (loadTileOp.getResult().use_empty()) {
        loadTileOp.erase();
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
