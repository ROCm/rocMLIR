//===- Unbufferize - Remove bufferization artifacts -----------------------===//
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
// This pass removes remaining bufferization artifacts after pointer conversion,
// converting the IR to pure tensor operations.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKUNBUFFERIZEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-unbufferize"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::triton;
using namespace mlir::arith;
using namespace mlir::bufferization;
using namespace mlir::memref;

namespace {

//===----------------------------------------------------------------------===//
// Pattern: Fold to_tensor(to_buffer(x)) -> x
//===----------------------------------------------------------------------===//
struct FoldToTensorOfToBuffer : public OpRewritePattern<ToTensorOp> {
  using OpRewritePattern<ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToTensorOp toTensorOp,
                                PatternRewriter &rewriter) const override {
    auto toBufferOp = toTensorOp.getBuffer().getDefiningOp<ToBufferOp>();
    if (!toBufferOp)
      return failure();

    Value sourceTensor = toBufferOp.getTensor();
    Type resultType = toTensorOp.getResult().getType();
    Type sourceType = sourceTensor.getType();

    // If types match, directly replace
    if (sourceType == resultType) {
      rewriter.replaceOp(toTensorOp, sourceTensor);
      return success();
    }

    // If types don't match, we might need a cast
    // For now, just replace and let subsequent passes handle type mismatches
    rewriter.replaceOp(toTensorOp, sourceTensor);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: Remove memref.copy where source/dest are to_buffer results
// that map to the same tensor
//===----------------------------------------------------------------------===//
struct RemoveRedundantCopy : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // Check if source is from a to_buffer
    auto srcToBuffer = copyOp.getSource().getDefiningOp<ToBufferOp>();
    if (!srcToBuffer)
      return failure();

    // Check if target is from a to_buffer or is a rock.alloc
    Value target = copyOp.getTarget();
    
    // If the target memref has no other uses besides this copy and
    // subsequent to_tensor ops, we can potentially eliminate this chain
    
    // For now, just erase the copy if both sides are from to_buffer
    auto dstToBuffer = target.getDefiningOp<ToBufferOp>();
    if (dstToBuffer) {
      // Both source and dest are to_buffer results - this is a no-op copy
      rewriter.eraseOp(copyOp);
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//
struct RockUnbufferizePass
    : public rock::impl::RockUnbufferizePassBase<RockUnbufferizePass> {
  void runOnOperation() override;
  
private:
  void collectDeadOps(func::FuncOp funcOp, SmallVectorImpl<Operation *> &opsToErase);
  void handleFillOps(func::FuncOp funcOp);
  void eraseDeadOps(SmallVectorImpl<Operation *> &opsToErase);
};

} // end anonymous namespace
void RockUnbufferizePass::collectDeadOps(func::FuncOp funcOp,
                                          SmallVectorImpl<Operation *> &opsToErase) {
  // Find dead to_buffer ops
  for (auto &block : funcOp.getBlocks()) {
    for (auto &op : block.getOperations()) {
      if (auto toBufferOp = dyn_cast<ToBufferOp>(&op)) {
        if (toBufferOp->use_empty()) {
          opsToErase.push_back(toBufferOp);
        }
      } else if (auto allocOp = dyn_cast<rock::GpuAllocOp>(&op)) {
        if (allocOp->use_empty()) {
          opsToErase.push_back(allocOp);
        }
      } else if (auto toTensorOp = dyn_cast<ToTensorOp>(&op)) {
        if (toTensorOp->use_empty()) {
          opsToErase.push_back(toTensorOp);
        }
      } else if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
        if (copyOp->use_empty()) {
          opsToErase.push_back(copyOp);
        }
      }
    }
  }
  
  // Also walk nested regions
  funcOp.walk([&](Operation *op) {
    if (op->getParentOp() == funcOp.getOperation())
      return WalkResult::skip();
      
    if (auto toBufferOp = dyn_cast<ToBufferOp>(op)) {
      if (toBufferOp->use_empty()) {
        opsToErase.push_back(toBufferOp);
      }
    } else if (auto allocOp = dyn_cast<rock::GpuAllocOp>(op)) {
      if (allocOp->use_empty()) {
        opsToErase.push_back(allocOp);
      }
    } else if (auto toTensorOp = dyn_cast<ToTensorOp>(op)) {
      if (toTensorOp->use_empty()) {
        opsToErase.push_back(toTensorOp);
      }
    } else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      if (copyOp->use_empty()) {
        opsToErase.push_back(copyOp);
      }
    }
    return WalkResult::advance();
  });
}

void RockUnbufferizePass::handleFillOps(func::FuncOp funcOp) {
  SmallVector<rock::FillOp> fillOps;
  funcOp.walk([&](rock::FillOp fillOp) {
    fillOps.push_back(fillOp);
    return WalkResult::advance();
  });
  
  for (auto fillOp : fillOps) {
    Value memref = fillOp.getInput();
    Value fillValue = fillOp.getValue();
    
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType)
      continue;

    // Check if this memref is only used by to_tensor ops after the fill
    bool allUsersAreToTensor = true;
    SmallVector<ToTensorOp> toTensorUsers;
    
    for (Operation *user : memref.getUsers()) {
      if (user == fillOp.getOperation())
        continue;
      
      if (auto toTensorOp = dyn_cast<ToTensorOp>(user)) {
        toTensorUsers.push_back(toTensorOp);
      } else {
        allUsersAreToTensor = false;
        break;
      }
    }

    if (!allUsersAreToTensor || toTensorUsers.empty())
      continue;

    // Create tensor type matching the memref shape
    auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                            memrefType.getElementType());

    // Create tt.splat at the fill location
    OpBuilder builder(fillOp);
    Location loc = fillOp.getLoc();
    Value splatTensor = triton::SplatOp::create(builder, loc, tensorType, fillValue);

    // Replace all to_tensor uses with the splat tensor
    for (auto toTensorOp : toTensorUsers) {
      toTensorOp.getResult().replaceAllUsesWith(splatTensor);
    }
  }
}

void RockUnbufferizePass::eraseDeadOps(SmallVectorImpl<Operation *> &opsToErase) {
  for (auto *op : opsToErase) {
    if (op->use_empty()) {
      op->erase();
    }
  }
  opsToErase.clear();
}

void RockUnbufferizePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *ctx = &getContext();

  // First, apply patterns to simplify the IR
  RewritePatternSet patterns(ctx);
  patterns.add<FoldToTensorOfToBuffer>(ctx);
  patterns.add<RemoveRedundantCopy>(ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Second pass: Remove dead operations
  SmallVector<Operation *, 16> opsToErase;
  collectDeadOps(funcOp, opsToErase);
  eraseDeadOps(opsToErase);

  // Third pass: Handle rock.fill operations
  handleFillOps(funcOp);

  // Fourth pass: Remove newly dead ops after fill conversion
  collectDeadOps(funcOp, opsToErase);
  eraseDeadOps(opsToErase);

  // Also remove dead fill ops
  SmallVector<rock::FillOp> fillOpsToRemove;
  funcOp.walk([&](rock::FillOp fillOp) {
    Value memref = fillOp.getInput();
    bool onlyUsedByFill = true;
    for (Operation *user : memref.getUsers()) {
      if (user != fillOp.getOperation()) {
        onlyUsedByFill = false;
        break;
      }
    }
    if (onlyUsedByFill) {
      fillOpsToRemove.push_back(fillOp);
    }
    return WalkResult::advance();
  });
  
  for (auto fillOp : fillOpsToRemove) {
    fillOp->erase();
  }

  // Fifth pass: Apply patterns again to clean up any remaining artifacts
  RewritePatternSet cleanupPatterns(ctx);
  cleanupPatterns.add<FoldToTensorOfToBuffer>(ctx);
  cleanupPatterns.add<RemoveRedundantCopy>(ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(cleanupPatterns)))) {
    signalPassFailure();
    return;
  }

  // Final cleanup of dead ops - iterate until no more changes
  bool changed = true;
  while (changed) {
    changed = false;
    opsToErase.clear();
    collectDeadOps(funcOp, opsToErase);
    
    if (!opsToErase.empty()) {
      changed = true;
      eraseDeadOps(opsToErase);
    }
  }
}

