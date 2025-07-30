//===- RemoveRedundantCasts - MLIR Rock ops lowering passes -----===//
//
// Copyright 2025 The MLIR Authors.
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
// ============================================================
//
// This pass removes any redundant casts that exist in the IR.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREMOVEREDUNDANTCASTSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-remove-redundant-casts"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {

// Pattern to remove redundant f32 -> dtype -> f32 cast chains
struct RemoveRedundantTruncExtfPattern
    : public OpRewritePattern<arith::TruncFOp> {
  func::FuncOp func;

  RemoveRedundantTruncExtfPattern(MLIRContext *context, func::FuncOp func)
      : OpRewritePattern<arith::TruncFOp>(context), func(func) {}

  LogicalResult matchAndRewrite(arith::TruncFOp truncf,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Looking at truncf: " << truncf << "\n");

    // Ensure that the truncf operation cast down from f32
    if (!isF32ToSmallerType(truncf)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tNot a f32 to smaller type truncf, skipping\n");
      return failure();
    }

    // Find the corresponding extf that uses this truncf's result
    bool changed = false;
    for (auto &extfInfo : findCorrespondingExtfs(truncf)) {
      // Replace the user of the extf operation with the original f32 memref
      // value and clean up the extf operation
      LLVM_DEBUG(llvm::dbgs()
                 << "\tReplacing extf: " << extfInfo.extfOp
                 << " with source memref: " << extfInfo.srcMemref << "\n");
      extfInfo.extfOp.replaceAllUsesWith(extfInfo.srcMemref);
      extfInfo.extfOp.erase();
      changed |= true;
    }

    return changed ? success() : failure();
  }

private:
  struct ExtfInfo {
    arith::ExtFOp extfOp;
    Value memref; // Optional: if the pattern goes through memory
    Value srcMemref;
  };

  struct StoreInfo {
    Operation *storeOp;
    Value memref;
    Value srcMemref;
  };

  bool isF32ToSmallerType(arith::TruncFOp truncfOp) const {
    Type inputType = truncfOp.getIn().getType();
    Type elementType = inputType;

    // If it's a vector type, get the element type
    if (auto vectorType = dyn_cast<VectorType>(inputType)) {
      elementType = vectorType.getElementType();
    }

    return elementType.isF32();
  }

  SmallVector<ExtfInfo> findCorrespondingExtfs(arith::TruncFOp truncfOp) const {
    SmallVector<ExtfInfo> extfInfos;

    // Pattern 1: Direct use - truncf -> extf
    if (auto directExtf = findDirectExtfUse(truncfOp)) {
      extfInfos.push_back({directExtf, nullptr, truncfOp.getIn()});
    }

    // Pattern 2: Through memory
    // * truncf -> store -> load -> extf
    // * linalg.generic that yields the truncf result as a memref
    auto memoryExtfs = findStoredMemoryExtfUse(truncfOp);
    if (!memoryExtfs.empty())
      extfInfos.append(memoryExtfs.begin(), memoryExtfs.end());

    return extfInfos;
  }

  arith::ExtFOp findDirectExtfUse(arith::TruncFOp truncfOp) const {
    for (auto &use : truncfOp->getUses()) {
      Operation *userOp = use.getOwner();
      if (auto extfOp = dyn_cast<arith::ExtFOp>(userOp)) {
        if (extendsToF32(extfOp)) {
          LLVM_DEBUG(llvm::dbgs() << "\tDirect extf use back to f32 found: "
                                  << extfOp.getLoc() << "\n");
          return extfOp;
        }
      }
    }

    return nullptr;
  }

  SmallVector<ExtfInfo>
  findStoredMemoryExtfUse(arith::TruncFOp truncfOp) const {
    SmallVector<ExtfInfo> extfInfos;

    // Check if truncf is stored to memory
    auto storeInfo = findStoreOp(truncfOp);
    if (!storeInfo)
      return extfInfos;

    // Find loads from the same memory location
    auto loads = findLoadsFromMemref(storeInfo->memref, storeInfo->storeOp);

    // Check if any load is used by an extf that extends back to f32
    for (auto &loadOp : loads) {
      if (auto extfOp = findExtfUseOfLoad(loadOp, storeInfo->memref)) {
        if (extendsToF32(extfOp)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "\tFound redundant extf: " << extfOp.getLoc() << "\n");
          extfInfos.push_back(
              {extfOp, storeInfo->memref, storeInfo->srcMemref});
        }
      }
    }

    return extfInfos;
  }

  Value getSourceMemref(Value val) const {
    // If it's a block argument, it may be a memref directly
    if (isa<BlockArgument>(val)) {
      BlockArgument blockArg = cast<BlockArgument>(val);
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (auto genericOp = dyn_cast<linalg::GenericOp>(parentOp)) {
        unsigned idx = blockArg.getArgNumber();
        if (idx < genericOp.getInputs().size())
          return genericOp.getInputs()[idx];
      }
      return val;
    }

    // If it's the result of a memref.load, get the memref operand
    if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(val.getDefiningOp())) {
      return loadOp.getMemref();
    }

    // Otherwise, return null (or handle other cases as needed)
    return Value();
  }

  std::optional<StoreInfo> findStoreOp(arith::TruncFOp truncfOp) const {
    // We have already asserted that there is exactly one use of the truncfOp
    auto *userOp = truncfOp->getUses().begin()->getOwner();

    // Collect the input memref to the truncfOp
    auto srcMemref = getSourceMemref(truncfOp.getIn());

    // Check for different types of store operations
    if (auto storeOp = dyn_cast<rock::InBoundsStoreOp>(userOp)) {
      return StoreInfo{storeOp, getRootMemref(storeOp.getDest()), srcMemref};
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(userOp)) {
      return StoreInfo{storeOp, getRootMemref(storeOp.getMemref()), srcMemref};
    } else if (auto yieldOp = dyn_cast<linalg::YieldOp>(userOp)) {
      // The parent op of the yield should be a linalg::GenericOp
      Operation *parentOp = yieldOp->getParentOp();
      if (auto genericOp = dyn_cast<linalg::GenericOp>(parentOp)) {
        // Find which yield value is the result of truncfOp
        for (unsigned i = 0; i < yieldOp.getValues().size(); ++i) {
          if (yieldOp.getValues()[i] == truncfOp.getResult()) {
            // Map yield value index to output memref
            if (i < genericOp.getOutputs().size()) {
              return StoreInfo{genericOp,
                               getRootMemref(genericOp.getOutputs()[i]),
                               srcMemref};
            }
          }
        }
      }
    }

    return std::nullopt;
  }

  SmallVector<Operation *> findStoresFromMemref(Value memref) const {
    SmallVector<Operation *> allStores;
    llvm::SmallPtrSet<Value, 8> visited;
    SmallVector<Value> worklist{memref};

    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;
      for (auto &use : current.getUses()) {
        Operation *user = use.getOwner();
        if (auto transformOp = dyn_cast<rock::TransformOp>(user)) {
          for (Value result : transformOp->getResults())
            worklist.push_back(result);
          continue;
        }
        if (isStoreToMemref(user, current))
          allStores.push_back(user);
      }
    }
    return allStores;
  }

  SmallVector<Operation *> findLoadsFromMemref(Value memref,
                                               Operation *storeOp) const {
    SmallVector<Operation *> validLoads;
    DominanceInfo domInfo(func);

    // Gather all store ops to this memref
    auto allStores = findStoresFromMemref(memref);

    llvm::SmallPtrSet<Value, 8> visited;
    SmallVector<Value> worklist{memref};
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;
      for (auto &use : current.getUses()) {
        Operation *user = use.getOwner();
        if (auto transformOp = dyn_cast<rock::TransformOp>(user)) {
          for (Value result : transformOp->getResults())
            worklist.push_back(result);
          continue;
        }

        if (!(isLoadFromMemref(user, memref) ||
              isDirectMemrefUse(user, memref)))
          continue;

        // Dominance check: storeOp must dominate this use
        // Note: We can only perform dominance checks between two ops in the
        // same region. Since the store ops will likely be in linalg.generic or
        // rock::TransformingForOp we should also check to see if the storeOp
        // and user are in the same region, and if they aren't then we should
        // get the parent region of the storeOp and check if it dominates the
        // user.
        if (storeOp->getParentRegion() != user->getParentRegion()) {
          auto *storeParentOp = storeOp->getParentRegion()->getParentOp();
          if (!domInfo.dominates(storeParentOp, user))
            continue;
        } else {
          if (!domInfo.dominates(storeOp, user))
            continue;
        }

        // Check for intervening store
        bool hasInterveningStore = false;
        for (Operation *otherStore : allStores) {
          if (otherStore == storeOp)
            continue;

          // TODO: For correctness we want to handle all of the scenarios where
          // stores can be in different regions
          if (domInfo.dominates(storeOp, otherStore) &&
              domInfo.dominates(otherStore, user)) {
            hasInterveningStore = true;
            break;
          }
        }

        if (!hasInterveningStore)
          validLoads.push_back(user);
      }
    }

    return validLoads;
  }

  arith::ExtFOp findExtfUseOfLoad(Operation *loadOp, Value memref) const {
    // For memref::LoadOp
    if (auto load = dyn_cast<memref::LoadOp>(loadOp)) {
      if (load->hasOneUse()) {
        auto *userOp = load->getUses().begin()->getOwner();
        if (auto extfOp = dyn_cast<arith::ExtFOp>(userOp)) {
          // Check if extf is extending from the loaded value
          if (extfOp.getIn() == memref)
            return extfOp;
        }
      }
    }

    // For linalg::GenericOp that operates on memref
    if (auto genericOp = dyn_cast<linalg::GenericOp>(loadOp)) {
      auto &bodyBlock = genericOp.getRegion().front();
      // The block arguments correspond to the inputs/outputs of the generic
      for (auto &op : bodyBlock) {
        if (auto extfOp = dyn_cast<arith::ExtFOp>(&op)) {
          // Case 1: extf directly on block argument (memref input)
          for (auto arg : bodyBlock.getArguments()) {
            // Find the index of this block argument in the block argument list
            unsigned idx = arg.getArgNumber();
            // Check if this block argument corresponds to the memref input
            if (idx < genericOp.getInputs().size() &&
                genericOp.getInputs()[idx] == memref) {
              if (extfOp.getIn() == arg)
                return extfOp;
            }
          }
          // Case 2: extf on result of memref.load from the memref
          if (auto load =
                  dyn_cast<memref::LoadOp>(extfOp.getIn().getDefiningOp())) {
            if (load.getMemref() == memref)
              return extfOp;
          }
        }
      }
    }

    return nullptr;
  }

  bool extendsToF32(arith::ExtFOp extfOp) const {
    Type outputType = extfOp.getOut().getType();
    Type elementType = outputType;

    if (auto vectorType = dyn_cast<VectorType>(outputType)) {
      elementType = vectorType.getElementType();
    }

    return elementType.isF32();
  }

  bool isStoreToMemref(Operation *op, Value memref) const {
    if (auto storeOp = dyn_cast<rock::InBoundsStoreOp>(op))
      return storeOp.getDest() == memref;
    if (auto storeOp = dyn_cast<memref::StoreOp>(op))
      return storeOp.getMemref() == memref;
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
      return llvm::is_contained(genericOp.getOutputs(), memref);
    return false;
  }

  bool isLoadFromMemref(Operation *op, Value memref) const {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op))
      return loadOp.getMemref() == memref;
    return false;
  }

  bool isDirectMemrefUse(Operation *op, Value memref) const {
    // For ops that take memref as input. Right now we are only checking for
    // linalg.generic ops
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
      return llvm::is_contained(genericOp.getInputs(), memref);
    return false;
  }

  Value getRootMemref(Value memref) const {
    Value current = memref;
    while (auto transformOp =
               dyn_cast_or_null<rock::TransformOp>(current.getDefiningOp())) {
      current = transformOp.getInput();
    }
    return current;
  }
};

struct RockRemoveRedundantCastsPass
    : public rock::impl::RockRemoveRedundantCastsPassBase<
          RockRemoveRedundantCastsPass> {
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Running RemoveRedundantCasts\n");

    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveRedundantTruncExtfPattern>(&getContext(), func);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Finished RemoveRedundantCasts\n");
  }
};

} // end anonymous namespace