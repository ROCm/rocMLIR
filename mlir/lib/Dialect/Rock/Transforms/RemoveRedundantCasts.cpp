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

// Find all arith.truncf operations that are directly stored to a buffer.
// A "direct store" means the truncf result is used immediately by a store
// operation with no intermediate operations modifying the value:
//   Valid:   truncf -> store
//   Invalid: truncf -> other_op -> store
SmallVector<TruncfStoreInfo> findTruncfWithDirectStores(func::FuncOp funcOp) {
  SmallVector<TruncfStoreInfo> results;

  funcOp.walk([&](arith::TruncFOp truncfOp) -> WalkResult {
    Type inputType = getElementTypeOrSelf(truncfOp.getIn().getType());
    Type outputType = getElementTypeOrSelf(truncfOp.getOut().getType());

    // Step 1: Check that this is a narrowing conversion (truncf)
    if (outputType.getIntOrFloatBitWidth() >= inputType.getIntOrFloatBitWidth())
      return WalkResult::advance();

    LLVM_DEBUG(llvm::dbgs() << "Found truncf: " << truncfOp << "\n");

    // Step 2: Check for direct stores of the truncf result
    Value truncfResult = truncfOp.getOut();
    Value wideValue = truncfOp.getIn();

    for (Operation *user : truncfResult.getUsers()) {
      FailureOr<std::pair<Value, SmallVector<Value>>> storeInfo =
          getStoreBufferAndIndices(user, truncfResult);
      if (failed(storeInfo))
        continue;

      LLVM_DEBUG(llvm::dbgs() << "  Found direct store: " << *user << "\n");
      LLVM_DEBUG(llvm::dbgs() << "  Target buffer: " << storeInfo->first
                              << "\n");

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

  LLVM_DEBUG(llvm::dbgs() << "Total truncf -> store pairs found: "
                          << results.size() << "\n");
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

  // TODO: Implement remaining steps of the algorithm:
  // Step 3: Find direct extf readers
  // Step 4: Verify safety
  // Step 5: Apply the optimization
}
