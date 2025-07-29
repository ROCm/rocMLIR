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
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
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
struct RemoveRedundantTruncExtfPattern : public OpRewritePattern<rock::TransformingForOp> {
  using OpRewritePattern<rock::TransformingForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::TransformingForOp transformingForOp,
                                PatternRewriter &rewriter) const override {
    // Find all truncf operations that cast down from f32
    SmallVector<arith::TruncFOp> truncfs = findTruncfs(transformingForOp);
    if (truncfs.empty())
      return failure();
    
    // Process each truncf input to see if it matches our pattern
    bool anyPatternMatched = false;
    for (auto &truncfOp : truncfs) {
      LLVM_DEBUG(llvm::dbgs() << "Investigating potentially redundant "
                              << "truncf op: " << truncfOp << "\n");

      // Find the corresponding extf that uses this truncf's result
      if (auto extfInfo = findCorrespondingExtf(truncfOp)) {
        LLVM_DEBUG(llvm::dbgs() << "\tFound redundant cast pattern\n");
        
        // Replace the extf operation with the original f32 value
        // TODO: How we do this may need to be adjusted since we are working
        // with inputs to a linalg.generic
        // rewriter.replaceOp(extfInfo->extfOp, truncfOp.getIn());
        
        // If the truncf is no longer used, it will be cleaned up by DCE
        anyPatternMatched = true;
      }
    }
    
    return anyPatternMatched ? success() : failure();
  }

private:
  struct ExtfInfo {
    arith::ExtFOp extfOp;
    Value memref; // Optional: if the pattern goes through memory
  };

  SmallVector<arith::TruncFOp> findTruncfs(rock::TransformingForOp transformingForOp) const {
    SmallVector<arith::TruncFOp> truncfs;

    // Find all truncf operations that cast down from a f32
    transformingForOp.walk([&](arith::TruncFOp truncfOp) {
      Type inputType = truncfOp.getIn().getType();
      Type elementType = inputType;
  
      // If it's a vector type, get the element type
      if (auto vectorType = dyn_cast<VectorType>(inputType)) {
        elementType = vectorType.getElementType();
      }
        
      if (elementType.isF32()) {
        truncfs.push_back(truncfOp);
      }
    });
    
    return truncfs;
  }

  Optional<ExtfInfo> findCorrespondingExtf(arith::TruncFOp truncfOp) const {
    // Pattern 1: Direct use - truncf -> extf
    if (auto directExtf = findDirectExtfUse(truncfOp)) {
      return ExtfInfo{directExtf, nullptr};
    }

    // Pattern 2: Through memory - truncf -> store -> load -> extf
    if (auto memoryExtf = findMemoryExtfUse(truncfOp)) {
      return memoryExtf;
    }

    return None;
  }

  arith::ExtFOp findDirectExtfUse(arith::TruncFOp truncfOp) const {
    if (!truncfOp->hasOneUse()) {
      return nullptr;
    }

    auto *userOp = truncfOp->getUses().begin()->getOwner();
    if (auto extfOp = dyn_cast<arith::ExtFOp>(userOp)) {
      if (extendsToF32(extfOp)) {
        LLVM_DEBUG(llvm::dbgs() << "\tDirect extf use back to f32 found\n");
        return extfOp;
      }
    }

    return nullptr;
  }

  Optional<ExtfInfo> findMemoryExtfUse(arith::TruncFOp truncfOp) const {
    // Check if truncf is stored to memory
    auto storeInfo = findStoreOp(truncfOp);
    if (!storeInfo) {
      return None;
    }

    LLVM_DEBUG(llvm::dbgs() << "\tTruncf stored to memref\n");

    // Find loads from the same memory location
    auto loads = findLoadsFromMemref(storeInfo->memref, storeInfo->storeOp);
    
    // Check if any load is used by an extf that extends back to f32
    for (auto loadOp : loads) {
      if (auto extfOp = findExtfUseOfLoad(loadOp)) {
        if (extendsToF32(extfOp)) {
          LLVM_DEBUG(llvm::dbgs() << "\tFound extf use through memory\n");
          return ExtfInfo{extfOp, storeInfo->memref};
        }
      }
    }

    return None;
  }

  struct StoreInfo {
    Operation *storeOp;
    Value memref;
  };

  Optional<StoreInfo> findStoreOp(arith::TruncFOp truncfOp) const {
    if (!truncfOp->hasOneUse()) {
      return None;
    }

    auto *userOp = truncfOp->getUses().begin()->getOwner();
    
    // Check for different types of store operations
    if (auto storeOp = dyn_cast<rock::InBoundsStoreOp>(userOp)) {
      return StoreInfo{storeOp, storeOp.getDest()};
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(userOp)) {
      return StoreInfo{storeOp, storeOp.getMemref()};
    }
    if (auto genericOp = dyn_cast<linalg::GenericOp>(userOp)) {
      // Check if this is a linalg.generic that just stores the truncf result
      if (isSimpleStoreGeneric(genericOp, truncfOp.getResult())) {
        // Get the output memref from the generic op
        if (!genericOp.getOutputs().empty()) {
          return StoreInfo{genericOp, genericOp.getOutputs()[0]};
        }
      }
    }

    return None;
  }

  bool isSimpleStoreGeneric(linalg::GenericOp genericOp, Value truncfResult) const {
    // Check if this generic op just yields the truncf result without modification
    auto &bodyBlock = genericOp.getRegion().front();
    if (bodyBlock.getOperations().size() != 2) { // Should have one yield op + one other op max
      return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(bodyBlock.getTerminator());
    if (!yieldOp || yieldOp.getValues().size() != 1) {
      return false;
    }

    // The yielded value should be the block argument corresponding to truncfResult
    // or the truncfResult itself if it's from outside the generic
    Value yieldedValue = yieldOp.getValues()[0];
    
    // Simple heuristic: if the generic has the truncf as input and yields it directly
    return llvm::is_contained(genericOp.getInputs(), truncfResult) ||
           yieldedValue == truncfResult;
  }

  SmallVector<Operation*> findLoadsFromMemref(Value memref, Operation *storeOp) const {
    SmallVector<Operation*> loads;
    
    // Get the function containing the store operation
    auto func = storeOp->getParentOfType<func::FuncOp>();
    if (!func) {
      return loads;
    }

    // Walk through all operations after the store to find loads from the same memref
    bool foundStore = false;
    func.walk([&](Operation *op) {
      if (op == storeOp) {
        foundStore = true;
        return;
      }
      
      if (!foundStore) {
        return;
      }

      // Check for load operations
      if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        if (loadOp.getMemref() == memref) {
          loads.push_back(loadOp);
        }
      } else if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        // Check if this generic reads from our memref
        if (llvm::is_contained(genericOp.getInputs(), memref)) {
          loads.push_back(genericOp);
        }
      }
    });

    return loads;
  }

  arith::ExtFOp findExtfUseOfLoad(Operation *loadOp) const {
    // For memref::LoadOp
    if (auto load = dyn_cast<memref::LoadOp>(loadOp)) {
      if (load->hasOneUse()) {
        auto *userOp = load->getUses().begin()->getOwner();
        return dyn_cast<arith::ExtFOp>(userOp);
      }
    }
    
    // For linalg::GenericOp that loads and extends
    if (auto genericOp = dyn_cast<linalg::GenericOp>(loadOp)) {
      auto &bodyBlock = genericOp.getRegion().front();
      
      // Look for extf operations in the generic body
      for (auto &op : bodyBlock) {
        if (auto extfOp = dyn_cast<arith::ExtFOp>(&op)) {
          return extfOp;
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
  
};

struct RockRemoveRedundantCastsPass
    : public rock::impl::RockRemoveRedundantCastsPassBase<RockRemoveRedundantCastsPass> {
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Running RemoveRedundantCasts\n");
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveRedundantTruncExtfPattern>(&getContext());
    
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Finished RemoveRedundantCasts\n");
  }
};

} // end anonymous namespace