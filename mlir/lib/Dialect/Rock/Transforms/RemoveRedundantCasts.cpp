//===--------------------- RemoveRedundantCasts.cpp -----------------------===//
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
//===----------------------------------------------------------------------===//
//
// This pass detects patterns where wider float values are truncated to a
// narrower float type, stored to a buffer, then loaded and extended back to the
// original wider input type. Replaces the extf uses with the original wide
// values, preserving precision.
//
// Note: The simpler truncf -> extf folding with no loads/stores is already
// handled by arith.truncf canonicalization patterns. This pass specifically
// deals with the more complex case where the values are stored to buffers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREMOVEREDUNDANTCASTSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-remove-redundant-casts"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct TruncfStoreInfo {
  arith::TruncFOp truncfOp;
  Value wideValue;
  Operation *storeOp;
  Value targetBuffer;
  SmallVector<Value> storeIndices;
};

struct LoadExtfInfo {
  Operation *loadOp;
  arith::ExtFOp extfOp;
  SmallVector<Value> loadIndices;
};

// A verified candidate for optimization - a (truncf->store, load->extf) pair
// that has passed all safety checks.
struct OptimizationCandidate {
  TruncfStoreInfo truncfStore;
  LoadExtfInfo loadExtf;
};

// Check if there are any other stores to the buffer that could interfere.
// Returns true if there are no intervening writes.
static bool hasNoInterveningWrites(Value buffer, Operation *ourStore) {
  // Conservative check: ensure our store is the ONLY store to this buffer.
  // This handles the common case where a buffer is written once and read
  // multiple times.
  for (Operation *user : buffer.getUsers()) {
    // Skip our own store
    if (user == ourStore)
      continue;

    // Check if this user is a store operation
    if (isa<InBoundsStoreOp, vector::StoreOp, memref::StoreOp,
            vector::TransferWriteOp>(user)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\t\tFound another store to buffer: " << *user << "\n");
      return false;
    }
  }
  return true;
}

// Find the ancestor of 'op' that is a direct child of 'block'.
static Operation *getAncestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

// Check if storeOp's enclosing operation dominates loadOp's enclosing operation.
// For ops in nested regions, finds their ancestors at a common nesting level
// and checks dominance between those ancestors.
static bool storeEnclosingOpDominatesLoad(Operation *storeOp, Operation *loadOp,
                                          DominanceInfo &domInfo) {
  // If they're in the same block, use direct dominance
  if (storeOp->getBlock() == loadOp->getBlock())
    return domInfo.properlyDominates(storeOp, loadOp);

  // Find a common ancestor block by walking up from the load
  for (Operation *loadWalk = loadOp; loadWalk;
       loadWalk = loadWalk->getParentOp()) {
    Block *block = loadWalk->getBlock();
    if (Operation *storeAncestor = getAncestorInBlock(storeOp, block)) {
      // Found common block - check if store's ancestor dominates load's
      return domInfo.properlyDominates(storeAncestor, loadWalk);
    }
  }

  return false;
}

// Verify that a (truncf->store, load->extf) pair is safe to optimize.
// Returns true if all safety conditions are met.
static bool verifySafety(const TruncfStoreInfo &truncfStore,
                         const LoadExtfInfo &loadExtf, DominanceInfo &domInfo) {
  // 4a. Store's enclosing op must dominate load's enclosing op
  if (!storeEnclosingOpDominatesLoad(truncfStore.storeOp, loadExtf.loadOp,
                                     domInfo)) {
    LLVM_DEBUG(llvm::dbgs() << "\t\tStore does not dominate load\n");
    return false;
  }

  // 4b. No intervening writes to the buffer
  // If our truncf store is the ONLY store to the buffer, then any value
  // read from the buffer must be a value we wrote - no need to match indices.
  if (!hasNoInterveningWrites(truncfStore.targetBuffer, truncfStore.storeOp)) {
    LLVM_DEBUG(llvm::dbgs() << "\t\tBuffer has other stores\n");
    return false;
  }
  // Note: We skip explicit index matching (4c) because hasNoInterveningWrites
  // guarantees our store is the only writer. Combined with the dominance check,
  // this means any loaded value must have come from our truncf store.

  // Note: We don't check if the wide value dominates the extf (4d) because
  // the wide value is typically defined inside a loop body and won't be
  // accessible at the extf location. Instead, Step 5 will create a shadow
  // buffer to store the wide value and redirect reads there.

  return true;
}

// Helper to check if a store operation directly stores a value.
// Returns the target buffer and indices if it's a supported store type.
static FailureOr<std::pair<Value, SmallVector<Value>>>
getStoreBufferAndIndices(Operation *op, Value storedValue) {
  if (auto inBoundsStore = dyn_cast<InBoundsStoreOp>(op)) {
    if (inBoundsStore.getData() == storedValue) {
      return std::pair<Value, SmallVector<Value>>(
          inBoundsStore.getDest(),
          SmallVector<Value>(inBoundsStore.getCoords()));
    }
  } else if (auto vectorStore = dyn_cast<vector::StoreOp>(op)) {
    if (vectorStore.getValueToStore() == storedValue) {
      return std::pair<Value, SmallVector<Value>>(
          vectorStore.getBase(),
          SmallVector<Value>(vectorStore.getIndices()));
    }
  } else if (auto memrefStore = dyn_cast<memref::StoreOp>(op)) {
    if (memrefStore.getValue() == storedValue) {
      return std::pair<Value, SmallVector<Value>>(
          memrefStore.getMemRef(),
          SmallVector<Value>(memrefStore.getIndices()));
    }
  }
  return failure();
}

// Helper to check if a load operation reads from a specific buffer.
// Returns the loaded value and indices if it's a supported load type.
static FailureOr<std::pair<Value, SmallVector<Value>>>
getLoadResultAndIndices(Operation *op, Value expectedBuffer) {
  if (auto inBoundsLoad = dyn_cast<InBoundsLoadOp>(op)) {
    if (inBoundsLoad.getSource() == expectedBuffer) {
      return std::pair<Value, SmallVector<Value>>(
          inBoundsLoad.getResult(),
          SmallVector<Value>(inBoundsLoad.getCoords()));
    }
  } else if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
    if (transferRead.getBase() == expectedBuffer) {
      return std::pair<Value, SmallVector<Value>>(
          transferRead.getResult(),
          SmallVector<Value>(transferRead.getIndices()));
    }
  } else if (auto memrefLoad = dyn_cast<memref::LoadOp>(op)) {
    if (memrefLoad.getMemRef() == expectedBuffer) {
      return std::pair<Value, SmallVector<Value>>(
          memrefLoad.getResult(),
          SmallVector<Value>(memrefLoad.getIndices()));
    }
  }
  return failure();
}

// Find all load -> extf patterns from a given buffer.
// A "direct extf" means the load result is used immediately by an extf
// operation with no intermediate operations modifying the value.
SmallVector<LoadExtfInfo> findDirectExtfReaders(Value narrowBuffer,
                                                Type wideType) {
  SmallVector<LoadExtfInfo> results;

  // Iterate over direct users of the buffer
  for (Operation *user : narrowBuffer.getUsers()) {
    // Check if this user is a load from our buffer
    FailureOr<std::pair<Value, SmallVector<Value>>> loadInfo =
        getLoadResultAndIndices(user, narrowBuffer);
    if (failed(loadInfo))
      continue;

    Value loadResult = loadInfo->first;

    // Check if the load result is used directly by an arith.extf
    for (Operation *loadUser : loadResult.getUsers()) {
      auto extfOp = dyn_cast<arith::ExtFOp>(loadUser);
      if (!extfOp)
        continue;

      // Verify the extf output type matches the expected wide type
      Type extfOutputType = getElementTypeOrSelf(extfOp.getOut().getType());
      if (extfOutputType != wideType)
        continue;

      LoadExtfInfo info;
      info.loadOp = user;
      info.extfOp = extfOp;
      info.loadIndices = std::move(loadInfo->second);
      results.push_back(info);
    }
  }

  return results;
}

// Find all arith.truncf operations that are directly stored to a buffer.
// A "direct store" means the truncf result is used immediately by a store
// operation with no intermediate operations modifying the value.
SmallVector<TruncfStoreInfo> findTruncfWithDirectStores(func::FuncOp funcOp) {
  SmallVector<TruncfStoreInfo> results;

  funcOp.walk([&](arith::TruncFOp truncfOp) -> WalkResult {
    Type inputType = getElementTypeOrSelf(truncfOp.getIn().getType());
    Type outputType = getElementTypeOrSelf(truncfOp.getOut().getType());

    // Check that this is a narrowing conversion (truncf)
    if (outputType.getIntOrFloatBitWidth() >= inputType.getIntOrFloatBitWidth())
      return WalkResult::advance();

    // Check for direct stores of the truncf result
    Value truncfResult = truncfOp.getOut();
    Value wideValue = truncfOp.getIn();

    for (Operation *user : truncfResult.getUsers()) {
      FailureOr<std::pair<Value, SmallVector<Value>>> storeInfo =
          getStoreBufferAndIndices(user, truncfResult);
      if (failed(storeInfo))
        continue;

      TruncfStoreInfo info;
      info.truncfOp = truncfOp;
      info.wideValue = wideValue;
      info.storeOp = user;
      info.targetBuffer = storeInfo->first;
      info.storeIndices = std::move(storeInfo->second);
      results.push_back(info);
    }

    return WalkResult::advance();
  });

  return results;
}

struct RockRemoveRedundantCastsPass
    : public rock::impl::RockRemoveRedundantCastsPassBase<
          RockRemoveRedundantCastsPass> {
  void runOnOperation() override;
};

} // end namespace

void RockRemoveRedundantCastsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  SmallVector<TruncfStoreInfo> truncfStores = findTruncfWithDirectStores(funcOp);

  if (truncfStores.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No truncf -> store patterns found, nothing to do.\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << truncfStores.size()
                          << " truncf -> store patterns to analyze.\n");

  // Collect verified optimization candidates
  SmallVector<OptimizationCandidate> candidates;

  // For each truncf -> store pair, find load -> extf readers and verify safety
  DominanceInfo domInfo(funcOp);
  for (const TruncfStoreInfo &truncfStore : truncfStores) {
    Type wideType = getElementTypeOrSelf(truncfStore.wideValue.getType());
    LLVM_DEBUG(llvm::dbgs() << "Analyzing buffer: " << truncfStore.targetBuffer
                            << "\n");
    SmallVector<LoadExtfInfo> extfReaders =
        findDirectExtfReaders(truncfStore.targetBuffer, wideType);

    if (extfReaders.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "\tNo load -> extf readers found.\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "\tFound " << extfReaders.size()
                            << " load -> extf readers.\n");

    // Verify safety of each load -> extf reader
    for (const LoadExtfInfo &loadExtf : extfReaders) {
      LLVM_DEBUG(llvm::dbgs() << "\tVerifying: load=" << *loadExtf.loadOp
                              << ", extf=" << loadExtf.extfOp << "\n");

      if (verifySafety(truncfStore, loadExtf, domInfo)) {
        LLVM_DEBUG(llvm::dbgs() << "\t\tSafety verified!\n");
        candidates.push_back({truncfStore, loadExtf});
      }
    }
  }

  if (candidates.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No safe optimization candidates found.\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << candidates.size()
                          << " safe truncation/extension candidates.\n");

  // TODO: Step 5: Apply the optimization
}
