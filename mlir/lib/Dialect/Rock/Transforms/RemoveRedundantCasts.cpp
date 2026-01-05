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

// Collect all arith.truncf operations in the function that convert from
// a wider float type to a narrower float type.
SmallVector<arith::TruncFOp> findAllTruncfOps(func::FuncOp funcOp) {
  SmallVector<arith::TruncFOp> truncfOps;

  funcOp.walk([&](arith::TruncFOp truncfOp) -> WalkResult {
    Type inputType = getElementTypeOrSelf(truncfOp.getIn().getType());
    Type outputType = getElementTypeOrSelf(truncfOp.getOut().getType());

    // Check that this is a narrowing conversion (truncf)
    if (outputType.getIntOrFloatBitWidth() >= inputType.getIntOrFloatBitWidth())
      return WalkResult::advance();

    LLVM_DEBUG(llvm::dbgs() << "Found truncf: " << truncfOp << "\n");
    truncfOps.push_back(truncfOp);
    return WalkResult::advance();
  });

  LLVM_DEBUG(llvm::dbgs() << "Total truncf operations found: "
                          << truncfOps.size() << "\n");
  return truncfOps;
}

struct RockRemoveRedundantCastsPass
    : public rock::impl::RockRemoveRedundantCastsPassBase<
          RockRemoveRedundantCastsPass> {
  void runOnOperation() override;
};

} // end namespace

void RockRemoveRedundantCastsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Step 1: Find all truncf operations (f32 -> narrow float)
  SmallVector<arith::TruncFOp> truncfOps = findAllTruncfOps(funcOp);

  if (truncfOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No truncf operations found, nothing to do.\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << truncfOps.size()
                          << " truncf operations to analyze.\n");

  // TODO: Implement remaining steps of the algorithm:
  // Step 2: Check for direct stores
  // Step 3: Find direct extf readers
  // Step 4: Verify safety
  // Step 5: Apply the optimization
}
