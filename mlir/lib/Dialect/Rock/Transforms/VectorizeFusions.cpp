//===- VectorizeFusions.cpp - Clean up math after lowering/unrolling loops
//---===//
//
// Copyright 2024 AMD
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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKVECTORIZEFUSIONSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-vectorize-fusions"

using namespace mlir;
using mlir::gpu::AddressSpace;

template <typename MemrefTypedValue>
static AddressSpace getAddressSpace(MemrefTypedValue val) {
  if (val.getType().getMemorySpace()) {
    return cast<gpu::AddressSpaceAttr>(val.getType().getMemorySpace())
        .getValue();
  }
  return gpu::AddressSpace::Global;
}

namespace {
struct RockVectorizeFusionsPass
    : public rock::impl::RockVectorizeFusionsPassBase<
          RockVectorizeFusionsPass> {
  void runOnOperation() override;
};
} // end namespace

void RockVectorizeFusionsPass::runOnOperation() {
  func::FuncOp op = getOperation();
  IRRewriter b(op.getContext());

  op.walk([&](affine::AffineForOp loop) -> WalkResult {
    // Collect data types and information about the vectorization feasibility
    SmallVector<Type> loopTypes;
    bool canVectorize = true;
    for (auto &bodyOp : loop.getBody()->getOperations()) {
      if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(bodyOp)) {
        if (getAddressSpace(affineLoad.getMemref()) == AddressSpace::Private)
          loopTypes.push_back(affineLoad.getType());
      } else if (auto affineStore = dyn_cast<affine::AffineStoreOp>(bodyOp)) {
        if (getAddressSpace(affineStore.getMemref()) == AddressSpace::Private)
          loopTypes.push_back(affineStore.getMemRefType().getElementType());
      } else if (!isa<affine::AffineYieldOp>(bodyOp) &&
                 !isa<arith::ArithDialect>(bodyOp.getDialect()) &&
                 !isa<math::MathDialect>(bodyOp.getDialect())) {
        canVectorize = false;
      }
    }

    if (!canVectorize)
      return WalkResult::advance();

    if (loopTypes.empty())
      return WalkResult::advance();

    LLVM_DEBUG(llvm::dbgs() << "Try to vectorize: " << loop << "\n");
    const int64_t ub = loop.getConstantUpperBound();
    const int64_t lb = loop.getConstantLowerBound();
    const int64_t step = loop.getStep().getSExtValue();
    if (step > 1)
      return WalkResult::advance();
    const int64_t loopTripCount = (ub - lb);
    const int64_t maxVectorBitWidth = 64;

    // Look for the vectorization factor
    int64_t vectorizationFactor = loopTripCount;
    for (auto type : loopTypes)
      vectorizationFactor =
          math_util::gcd(maxVectorBitWidth / type.getIntOrFloatBitWidth(),
                         vectorizationFactor);

    if (vectorizationFactor == 1)
      return WalkResult::advance();

    LLVM_DEBUG(llvm::dbgs()
               << "With vector length = " << vectorizationFactor << "\n");
    DenseSet<Operation *> loops{loop};
    affine::vectorizeAffineLoops(op, loops,
                                 SmallVector<int64_t>{vectorizationFactor}, {});

    return WalkResult::advance();
  });

  op.walk([&](affine::AffineForOp loop) {
    // Make sure the transfer reads are in bounds
    for (auto trReadOp : loop.getRegion().getOps<vector::TransferReadOp>()) {
      VectorType readType = trReadOp.getVector().getType();
      SmallVector<bool> inBounds(readType.getRank(), true);
      trReadOp.setInBoundsAttr(b.getBoolArrayAttr(inBounds));
    }

    // Make sure the transfer writes are in bounds
    for (auto trWrtOp : loop.getRegion().getOps<vector::TransferWriteOp>()) {
      VectorType writeType = trWrtOp.getVector().getType();
      SmallVector<bool> inBounds(writeType.getRank(), true);
      trWrtOp.setInBoundsAttr(b.getBoolArrayAttr(inBounds));
    }
  });
}
