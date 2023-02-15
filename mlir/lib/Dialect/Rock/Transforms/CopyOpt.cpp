//===- CopyOpt.cpp - Remove redundant memories
//------------------===//
//
// Copyright 2020 The MLIR Authors.
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
// This pass removes redundant local memories.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-copy-opt"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCOPYOPTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {
struct RockCopyOptPass
    : public rock::impl::RockCopyOptPassBase<RockCopyOptPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

//===- MICORewritePattern -------------------------------------------------===//
//===-  ------------------------------------------------===//
struct MICORewritePattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  // This function finds the destination-passing style
  // argument that the kernel will copy into it. It might
  // encounter view-like ops : CollapseShapeOp or ExpandShapeOp
  // where the reverse expression need to be built up for
  // the users of the found destination arg.
  LogicalResult findCopyDestArg(PatternRewriter &rewriter, Operation *op,
                                Value &copyDestArg,
                                memref::CopyOp &copyDest) const {
    if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      return findCopyDestArg(rewriter, copyOp, copyDestArg, copyDest);
    }
    if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(op)) {
      return findCopyDestArg(rewriter, collapseOp, copyDestArg, copyDest);
    }
    if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op)) {
      return findCopyDestArg(rewriter, expandOp, copyDestArg, copyDest);
    }
    return failure();
  }

  // Finds the destination arg when the copy op is found
  LogicalResult findCopyDestArg(PatternRewriter &rewriter,
                                memref::CopyOp &copyOp, Value &copyDestArg,
                                memref::CopyOp &copyDest) const {
    if (copyDestArg) {
      return failure();
    }
    if (copyOp.getTarget().getDefiningOp()) {
      // If the target is defined within the kernel it should
      // not be optimized out.
      return failure();
    }
    copyDestArg = copyOp.getTarget();
    copyDest = copyOp;
    return success();
  }

  // If there is a view-like op, the reverse expression needs to
  // be built up when the copy destination arg is found.
  template <typename ReassociativeReshapeOp>
  LogicalResult findCopyDestArg(PatternRewriter &rewriter,
                                ReassociativeReshapeOp &reassociativeReshapeOp,
                                Value &copyDestArg,
                                memref::CopyOp &copyDest) const {
    if (!reassociativeReshapeOp->hasOneUse()) {
      return failure();
    }
    if (findCopyDestArg(rewriter,
                        reassociativeReshapeOp->getUses().begin()->getOwner(),
                        copyDestArg, copyDest)
            .failed()) {
      return failure();
    }

    if (std::is_same<ReassociativeReshapeOp, memref::ExpandShapeOp>()) {
      copyDestArg = rewriter.create<memref::CollapseShapeOp>(
          reassociativeReshapeOp.getLoc(),
          reassociativeReshapeOp->getOperand(0).getType(), copyDestArg,
          reassociativeReshapeOp.getReassociation());
    } else if (std::is_same<ReassociativeReshapeOp,
                            memref::CollapseShapeOp>()) {
      copyDestArg = rewriter.create<memref::ExpandShapeOp>(
          reassociativeReshapeOp.getLoc(),
          reassociativeReshapeOp->getOperand(0).getType(), copyDestArg,
          reassociativeReshapeOp.getReassociation());
    } else {
      return failure();
    }

    return success();
  }

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &b) const override {
    LogicalResult fail = failure();

    // 0. Test compatibility
    // 0.0 Global Memory Space
    auto allocType = op.getType().template cast<MemRefType>();
    auto memSpace =
        allocType.getMemorySpace().dyn_cast_or_null<gpu::AddressSpaceAttr>();
    if ((memSpace) &&
        (memSpace.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace() ||
         memSpace.getValue() == gpu::GPUDialect::getPrivateAddressSpace()))
      return fail;

    Value allocaMem = op->getResult(0);

    // 1. Capture allocation->copy pattern
    Operation *writer = nullptr;
    memref::CopyOp copyDest;
    Value copyDestArg;
    for (auto &use : allocaMem.getUses()) {
      Operation *useOp = use.getOwner();
      if (auto laop = dyn_cast<linalg::GenericOp>(useOp)) {
        // 1.0 Output of linalg.generic
        if (writer)
          return fail;
        for (auto out : laop.getOutputs()) {
          if (out == allocaMem)
            writer = laop;
        }
        if (!writer)
          return fail;
      } else if (auto mrop = dyn_cast<rock::RockGemmWrapperInterface>(useOp)) {
        // 1.1 Direct output of a gemm-wrapping operation (mainly gemm itself,
        // which doesn't get transform()s after it)
        if (writer)
          return fail;
        if (mrop.getOutArgument()->get() != allocaMem) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "Allocation argument isn't in output position on gemm-like op"
              << mrop << "\n");
          return fail;
        }
        writer = mrop;
      } else if (auto mrop = dyn_cast<rock::TransformOp>(useOp)) {
        // 1.2 Input of rock.transform
        if (writer)
          return fail;
        rock::TransformOp originalMrop = mrop;
        Value mrval = mrop.getResult();
        // Dig through chains of transposes
        while (mrval.hasOneUse() && (mrop = dyn_cast_or_null<rock::TransformOp>(
                                         mrval.getUses().begin()->getOwner())))
          mrval = mrop.getResult();
        // 1.2.0 Confirm output of a gemm-like operation
        int cnt = 0;
        for (auto &mruse : mrval.getUses()) {
          if (auto gemmLike =
                  dyn_cast<rock::RockGemmWrapperInterface>(mruse.getOwner())) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Found gemm-like op " << gemmLike << "\n");
            if (gemmLike.getOutArgument()->get() != mrval)
              return fail;
          }
          cnt++;
        }
        if (cnt != 1)
          return fail;
        writer = originalMrop;
      } else if (auto callop = dyn_cast<CallOpInterface>(useOp)) {
        // 1.3 Assume call is the writer (fails for multiple calls)
        if (writer)
          return fail;
        writer = callop;
      } else {
        // The remaining uses of the allocate node should lead to
        // a copy node that copies the data to destination passing-style
        // argument, if not fail the pass.
        if (findCopyDestArg(b, useOp, copyDestArg, copyDest).failed()) {
          return fail;
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Found copy dest arg = " << copyDestArg
                            << " and writer " << writer << "\n");
    // 2. do it
    if (copyDestArg && writer) {
      if (copyDest)
        copyDest->erase();
      allocaMem.replaceAllUsesWith(copyDestArg);
      return success();
    }

    return fail;
  }
};

//===- Passes -------------------------------------------------------------===//
//===- RockCopyOptPass -  -----------------===//
//
void RockCopyOptPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MICORewritePattern>(ctx);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
