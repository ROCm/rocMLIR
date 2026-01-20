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
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
// Pattern: Fold to_tensor(alloc) where alloc is written by copy(to_buffer(x))
// This handles the pattern:
//   %alloc = rock.alloc()
//   %buf = to_buffer(%tensor)
//   memref.copy(%buf, %alloc)
//   %result = to_tensor(%alloc)
// And replaces %result with %tensor
//===----------------------------------------------------------------------===//
struct FoldToTensorOfAllocWithCopy : public OpRewritePattern<ToTensorOp> {
  using OpRewritePattern<ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToTensorOp toTensorOp,
                                PatternRewriter &rewriter) const override {
    Value buffer = toTensorOp.getBuffer();

    // Check if the buffer is from a rock.alloc
    auto allocOp = buffer.getDefiningOp<rock::GpuAllocOp>();
    if (!allocOp)
      return failure();

    // Find the memref.copy that writes to this alloc
    memref::CopyOp copyOp = nullptr;
    for (Operation *user : allocOp.getResult().getUsers()) {
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() == allocOp.getResult()) {
          copyOp = copy;
          break;
        }
      }
    }

    if (!copyOp)
      return failure();

    // Check if the source of the copy is from a to_buffer
    auto srcToBuffer = copyOp.getSource().getDefiningOp<ToBufferOp>();
    if (!srcToBuffer)
      return failure();

    // Get the original tensor from the to_buffer
    Value sourceTensor = srcToBuffer.getTensor();
    Type resultType = toTensorOp.getResult().getType();
    Type sourceType = sourceTensor.getType();

    // Replace the to_tensor result with the original tensor
    if (sourceType == resultType) {
      rewriter.replaceOp(toTensorOp, sourceTensor);
    } else {
      // Types don't match exactly - this can happen with encoding differences
      // For now, just replace and let subsequent passes handle it
      rewriter.replaceOp(toTensorOp, sourceTensor);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//
struct RockUnbufferizePass
    : public rock::impl::RockUnbufferizePassBase<RockUnbufferizePass> {
  void runOnOperation() override;

private:
  void collectDeadOps(triton::FuncOp funcOp,
                      SmallVectorImpl<Operation *> &opsToErase);
  void eraseDeadOps(SmallVectorImpl<Operation *> &opsToErase);
};

} // end anonymous namespace
void RockUnbufferizePass::collectDeadOps(
    triton::FuncOp funcOp, SmallVectorImpl<Operation *> &opsToErase) {
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

void RockUnbufferizePass::eraseDeadOps(
    SmallVectorImpl<Operation *> &opsToErase) {
  for (auto *op : opsToErase) {
    if (op->use_empty()) {
      op->erase();
    }
  }
  opsToErase.clear();
}

void RockUnbufferizePass::runOnOperation() {
  triton::FuncOp funcOp = getOperation();
  MLIRContext *ctx = &getContext();

  // First, apply patterns to simplify the IR
  RewritePatternSet patterns(ctx);
  patterns.add<FoldToTensorOfToBuffer>(ctx);
  patterns.add<FoldToTensorOfAllocWithCopy>(ctx);
  patterns.add<RemoveRedundantCopy>(ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Second pass: Remove dead operations
  SmallVector<Operation *, 16> opsToErase;
  collectDeadOps(funcOp, opsToErase);
  eraseDeadOps(opsToErase);

  // Third pass: Apply patterns again to clean up any remaining artifacts
  RewritePatternSet cleanupPatterns(ctx);
  cleanupPatterns.add<FoldToTensorOfToBuffer>(ctx);
  cleanupPatterns.add<FoldToTensorOfAllocWithCopy>(ctx);
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
