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
// This pass identifies and removes redundant floating-point cast chains in the
// IR, specifically patterns where a value is converted from f32 to a smaller
// type (e.g., f16) and then immediately extended back to f32. It also handles
// cases where such casts are stored and loaded through memory or passed through
// linalg.generic operations. By eliminating these unnecessary conversions, the
// pass simplifies the IR and can improve performance by reducing superfluous
// operations and memory traffic.
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

    // Find the corresponding extf(s) that uses this truncf's result
    bool changed = false;
    for (auto &extfInfo : findCorrespondingExtfs(truncf)) {
      // Try to replace the extf with the memref input to the original truncf
      // if possible. We do not touch the original truncf operation here, as we
      // allow for it to get cleaned up by DCE if applicable.
      auto res = replaceExtfWithMemref(extfInfo.extfOp, extfInfo.memref,
                                       extfInfo.srcMemref, rewriter);
      if (res) {
        LLVM_DEBUG(llvm::dbgs() << "\tReplacing extf: " << extfInfo.extfOp
                                << " with: " << extfInfo.srcMemref << "\n");
      }

      // TODO: If after the removal of the extfOps, the truncfOp (or its store)
      // no longer has any uses, we can also remove it. Note, DCE will not do
      // this removal for us

      changed |= res;
    }

    return changed ? success() : failure();
  }

private:
  struct ExtfInfo {
    arith::ExtFOp extfOp;
    // Will be set to nullptr if the pattern is a direct truncf -> extf use
    Value memref;
    Value srcMemref;
  };

  struct StoreInfo {
    Operation *storeOp;
    Value memref;
    Value srcMemref;
  };

  // This function will try to replace the extfOp with the srcMemref (the
  // orignal input to the truncfOp) if possible.
  bool replaceExtfWithMemref(arith::ExtFOp extfOp, Value memref,
                             Value srcMemref, PatternRewriter &rewriter) const {
    // If the optional memref value of the ExtfInfo struct is not set, then
    // this directly corresponds to the direct truncf -> extf case and we do
    // not need to do any additional handling work (e.g., for linalg.generic)
    if (!memref) {
      extfOp.replaceAllUsesWith(srcMemref);
      extfOp.erase();
      return true;
    }

    // If the extfOp is in a linalg.generic, then we need special handling
    if (auto genericOp = extfOp->getParentOfType<linalg::GenericOp>()) {
      // If the truncf op was in the same linalg.generic as the extfOp, then we
      // expect that it would have been handled by the direct replacement case
      // above. So we can safely assume that the extfOp is in a separate
      // linalg.generic.
      LLVM_DEBUG(llvm::dbgs() << "\tEXTFOP: " << extfOp << "\n");
      LLVM_DEBUG(llvm::dbgs() << "\tMEMREF: " << memref << "\n");
      LLVM_DEBUG(llvm::dbgs() << "\tSRC MEMREF: " << srcMemref << "\n");

      // Step 1: Create new rock.alloc op to hold the result from before the
      // truncf operation. This is only for the case where the input to the
      // truncf operation is not a memref<shapexf32>, but is instead something
      // like memref<vector<shapexf32>>

      // Step 2: Transform the rock.alloc if necessary to match the expected
      // output type

      // Step 3: Create a clone of the linalg.generic operation with the extf op
      // with the new (potentially transformed rock.alloc) as the input

      // Step 4: Remove the original linalg.generic operation

    }
    
    // We have reached an unsupported case, so we do not make any changes
    return false; 
  }

  // Helper function that checks if the truncf operation is converting from f32
  // to a smaller type (e.g., f16).
  bool isF32ToSmallerType(arith::TruncFOp truncfOp) const {
    Type inputType = truncfOp.getIn().getType();
    Type elementType = inputType;

    // If it's a vector type, get the element type
    if (auto vectorType = dyn_cast<VectorType>(inputType)) {
      elementType = vectorType.getElementType();
    }

    return elementType.isF32();
  }

  // This function will find all valid ExtfOps that correspond to uses of the 
  // orignal truncfOp.
  SmallVector<ExtfInfo> findCorrespondingExtfs(arith::TruncFOp truncfOp) const {
    SmallVector<ExtfInfo> extfInfos;

    // Pattern 1: Direct use - truncf -> extf
    if (auto directExtf = findDirectExtfUse(truncfOp)) {
      extfInfos.push_back({directExtf, nullptr, truncfOp.getIn()});
    }

    // Pattern 2: Through memory
    // truncf -> store -> linalg.generic -> extf
    auto memoryExtfs = findStoredMemoryExtfUse(truncfOp);
    if (!memoryExtfs.empty())
      extfInfos.append(memoryExtfs.begin(), memoryExtfs.end());

    return extfInfos;
  }

  // This function will find any direct ExtfOp uses of the original truncfOp
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

  // This function will find any ExtfOp uses of the original truncfOp that are
  // stored to memory and then loaded back through a linalg.generic operation.
  SmallVector<ExtfInfo>
  findStoredMemoryExtfUse(arith::TruncFOp truncfOp) const {
    SmallVector<ExtfInfo> extfInfos;

    // Check if truncf value is stored (this can be a rock.in_bounds_store
    // or a linalg.generic yield)
    auto storeInfo = findStoreOp(truncfOp);
    if (!storeInfo)
      return extfInfos;

    // Find valid linalg.generic uses from the same memory location
    auto loads = findLinalgGenericUsesFromMemref(storeInfo->memref,
                                                 storeInfo->storeOp);

    // Check if any load is used by an extf that extends back to f32
    for (auto &loadOp : loads) {
      if (auto extfOp = findExtfUseInGeneric(loadOp, storeInfo->memref)) {
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

  // Returns the source memref associated with the given value.
  // - If the value is a block argument, attempts to resolve it to the
  //   corresponding input of a parent linalg::GenericOp, if applicable;
  //   otherwise returns the block argument itself.
  // - If the value is the result of a memref::LoadOp, returns the memref
  //   operand of the load.
  // - For other cases, returns a null Value.
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

  // Finds the first store operation that writes the result of the given
  // truncfOp to memory. Handles different store patterns, including direct
  // memref stores, rock::InBoundsStoreOp, and linalg::GenericOp yields.
  // Returns a StoreInfo struct containing the store operation, the destination
  // memref, and the original source memref.
  std::optional<StoreInfo> findStoreOp(arith::TruncFOp truncfOp) const {
    // Collect the input memref to the truncfOp
    auto srcMemref = getSourceMemref(truncfOp.getIn());

    // Right now we assume that there is only going to be a single store of the
    // truncated value
    if (!llvm::hasSingleElement(truncfOp->getUses())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\ttruncf has multiple uses, skipping\n");
      return std::nullopt;
    }

    auto *userOp = truncfOp->getUses().begin()->getOwner();

    // Check for different types of store operations
    if (auto storeOp = dyn_cast<rock::InBoundsStoreOp>(userOp)) {
      return StoreInfo{storeOp, getRootMemref(storeOp.getDest()), srcMemref};
    } else if (auto yieldOp = dyn_cast<linalg::YieldOp>(userOp)) {
      // The parent op of the yield should be a linalg::GenericOp
      Operation *parentOp = yieldOp->getParentOp();
      if (!isa<linalg::GenericOp>(parentOp))
        return std::nullopt;

      // Find which yield value is the result of truncfOp
      auto genericOp = cast<linalg::GenericOp>(parentOp);
      for (unsigned i = 0; i < yieldOp.getValues().size(); ++i) {
        if (yieldOp.getValues()[i] == truncfOp.getResult()) {
          // Map yield value index to output memref
          return StoreInfo{genericOp,
                           getRootMemref(genericOp.getOutputs()[i]),
                           srcMemref};
        }
      }
    }

    return std::nullopt;
  }

  /// Finds all store operations that write to the given memref value.
  /// Traverses the use-def chain starting from the provided memref, following
  /// any rock::TransformOp results, and collects all operations that store to
  /// the memref (including memref::StoreOp, rock::InBoundsStoreOp, and
  /// linalg::GenericOp outputs).
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

  // This function finds all valid linalg.generic uses from a given memref. 
  // Valid uses are those that are dominated by a specific store operation
  // (the store of the truncated value) and do not have any intervening stores
  // to the same memref.
  SmallVector<Operation *> findLinalgGenericUsesFromMemref(Value memref,
                                                           Operation *storeOp) const {
    SmallVector<Operation *> validGenerics;
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

        if (!isDirectGenericMemrefUse(user, memref))
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
          // stores can be in different regions. If a store or load is in a
          // rock.transformingFor or linalg generic we should get the parent
          // region. We should make this change above as well
          if (domInfo.dominates(storeOp, otherStore) &&
              domInfo.dominates(otherStore, user)) {
            hasInterveningStore = true;
            break;
          }
        }

        if (!hasInterveningStore)
          validGenerics.push_back(user);
      }
    }

    return validGenerics;
  }

  // Finds an arith::ExtFOp that uses the result of a load operation from the
  // given memref. Returns the matching ExtFOp if found, otherwise returns
  // nullptr.
  // TODO: We might be able to simplify this logic and only handle the case
  // where the load is a linalg.generic op that directly uses the memref value
  arith::ExtFOp findExtfUseInGeneric(Operation *loadOp, Value memref) const {
    if (!isa<linalg::GenericOp>(loadOp))
      return nullptr; // Skip if the use is not in a linalg.generic

    auto genericOp = cast<linalg::GenericOp>(loadOp);
    Block &body = genericOp.getRegion().front();

    // Get the index of the block argument that corresponds to the memref
    Value inputArg = nullptr;
    unsigned idx = 0;
    for (auto ins : genericOp.getInputs()) {
      // Check if this input corresponds to the memref input
      if (getRootMemref(ins) == memref) {
        // The block argument index matches the input index
        inputArg = body.getArgument(idx);
        break;
      }
      idx++;
    }

    // Check that the first, and only, use of the inputArg is an ExtfOp
    if (!inputArg)
      return nullptr;

    if (!llvm::hasSingleElement(inputArg.getUses())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tInput argument has multiple uses, skipping\n");
      return nullptr;
    }

    Operation *userOp = inputArg.getUses().begin()->getOwner();
    if (auto extfOp = dyn_cast<arith::ExtFOp>(userOp)) {
      // Check if the ExtfOp extends to f32
      if (extendsToF32(extfOp)) {
        LLVM_DEBUG(llvm::dbgs() << "\tFound extf in generic: "
                                << extfOp.getLoc() << "\n");
        return extfOp;
      }
    }

    return nullptr;
  }

  // Helper function that checks if the ExtfOp is converting to f32 from a
  // smaller type (e.g., f16).
  bool extendsToF32(arith::ExtFOp extfOp) const {
    Type outputType = extfOp.getOut().getType();
    Type elementType = outputType;

    if (auto vectorType = dyn_cast<VectorType>(outputType)) {
      elementType = vectorType.getElementType();
    }

    return elementType.isF32();
  }

  bool isStoreToMemref(Operation *op, Value memref) const {
    if (auto storeOp = dyn_cast<rock::InBoundsStoreOp>(op)) {
      return storeOp.getDest() == memref;
    } else if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
      return llvm::is_contained(genericOp.getOutputs(), memref);
    return false;
  }


  bool isDirectGenericMemrefUse(Operation *op, Value memref) const {
    // For ops that take memref as input. Right now we are only checking for
    // linalg.generic ops
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
      return llvm::is_contained(genericOp.getInputs(), memref);
    return false;
  }

  Value getRootMemref(Value memref) const {
    Value current = memref;
    while (auto transformOp =
               dyn_cast<rock::TransformOp>(current.getDefiningOp())) {
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