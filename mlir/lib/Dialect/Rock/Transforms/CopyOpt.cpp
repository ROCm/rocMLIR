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
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
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

  template <typename... TArgs>
  bool getForwardChain(Operation *op, SmallVector<Operation*> &chain) const {
    chain.push_back(op);
    if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
      // Source = chain.back()
      // Target = func arg
      return copyOp.getTarget().getDefiningOp() == nullptr;
    }
    if (isa<TArgs...>(op) && op->hasOneUse()) {
      // Follow def-use chain
      Operation *result = op->getUses().begin()->getOwner();
      return getForwardChain<TArgs...>(result, chain);
    }
    return false;
  }

  /////////////////////////////////////////////////////////////////////

  // %5 = memref.alloc
  // Uses:
  //   %6 = linalg.generic ... outs (%5)
  //   %7 = rock.transform %5
  //     %8 = memref.expand_shape %7
  //        memref.copy (%8, %arg3) ... must be source

  bool isWriter(linalg::GenericOp op, Value mem) const {
    // must be an output
    for (auto out : op.outputs()) {
      if (out == mem)
        return true;
    }
    return false;
  }

  bool isWriter(rock::RockGemmWrapperInterface op, Value mem) const {
    // 1.1 Direct output of a gemm-wrapping operation (mainly gemm itself,
    // which doesn't get transform()s after it)
    return op.getOutArgument()->get() == mem;
  }

  bool isWriter(rock::TransformOp op, Value mem) const {
    // 1.2 Input of rock.transform
    Value mrval = op.getResult();
    // Dig through chains of transposes
    // TODO: convert to forwardChain check
    while (mrval.hasOneUse() && (op = dyn_cast_or_null<rock::TransformOp>(mrval.getUses().begin()->getOwner())))
      mrval = op.getResult();
    // 1.2.0 Confirm output of a gemm-like operation
    int cnt = 0;
    for (auto &mruse : mrval.getUses()) {
      if (auto gemmLike =
          dyn_cast<rock::RockGemmWrapperInterface>(mruse.getOwner())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Found gemm-like op " << gemmLike << "\n");
        if (gemmLike.getOutArgument()->get() != mrval)
          return false;
      }
      cnt++;
    }
    return (cnt == 1);
  }

  bool isWriter(CallOpInterface op, Value mem) const {
    // 1.3 Assume call is the writer (fails for multiple calls)
    // TODO(sjw): check for writeonly!
    return true;
  }
  
  template <int T = 0>
  bool getWriter(Operation *op, Value mem) const {
    return false;
  }

  template <typename T, typename... TArgs>
  bool getWriter(Operation *op, Value mem) const {
    if (auto wop = dyn_cast<T>(op))
      return isWriter(wop, mem);
    return getWriter<TArgs...>(op, mem);
  }

  /////////////////////////////////////////////////////////////////////
  // If there is a view-like op, the inverse expression needs to
  // be built up when the copy destination arg is found.
  bool invertOp(PatternRewriter &b, memref::CollapseShapeOp &op,
                Value &chainVal) const {
    chainVal = b.create<memref::ExpandShapeOp>(
          op.getLoc(),
          op.getOperand().getType(), chainVal,
          op.getReassociation());
    return true;
  }
  bool invertOp(PatternRewriter &b, memref::ExpandShapeOp &op,
                Value &chainVal) const {
    chainVal = b.create<memref::CollapseShapeOp>(
          op.getLoc(),
          op.getOperand().getType(), chainVal,
          op.getReassociation());
    return true;
  }

  bool invertOp(PatternRewriter &b, rock::TransformOp &op,
                Value &chainVal) const {
    auto transformMap = rock::invertTransformMap(b, op.getTransform());
    if (!transformMap)
      return false;

    chainVal = b.create<rock::TransformOp>(op.getLoc(), chainVal, transformMap);
    return true;
  }

  template <int T = 0>
  bool invertChain(PatternRewriter &b, Operation *op, Value& chainVal) const {
    return false;
  }

  template <typename T, typename... TArgs>
  bool invertChain(PatternRewriter &b, Operation *op, Value& chainVal) const {
    if (auto wop = dyn_cast<T>(op))
      return invertOp(b, wop, chainVal);
    return invertChain<TArgs...>(b, op, chainVal);
  }

  /////////////////////////////////////////////////////////////////////
  // matchAndRewrite
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &b) const override {
    LogicalResult fail = failure();

    // 0. Test compatibility
    // 0.0 Global Memory Space
    auto allocType = op.getType().template cast<MemRefType>();
    auto memSpace = allocType.getMemorySpaceAsInt();
    if (memSpace == gpu::GPUDialect::getWorkgroupAddressSpace() ||
        memSpace == gpu::GPUDialect::getPrivateAddressSpace())
      return fail;

    Value allocaMem = op->getResult(0);

    // 1. Capture allocation->copy pattern
    Operation *writer = nullptr;
    SmallVector<Operation*> forwardChain;

    for (auto &use : allocaMem.getUses()) {
      Operation *useOp = use.getOwner();
      if (getWriter<linalg::GenericOp, rock::RockGemmWrapperInterface,
          rock::TransformOp, CallOpInterface>(useOp, allocaMem)) {
        if (writer)
          return fail;
        writer = useOp;
      } else {
        // The remaining uses of the allocate node should lead to
        // a copy node that copies the data to destination passing-style
        // argument, if not fail the pass.
        if (!forwardChain.empty())
          return fail;
        if (!getForwardChain<memref::ExpandShapeOp, memref::CollapseShapeOp,
            rock::TransformOp>(useOp, forwardChain))
          return fail;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Found copy chain = " << forwardChain.size()
                            << " and writer " << writer << "\n");
    // 2. do it
    if (writer) {

      // 2.0. reverse forwardChain
      auto copyOp = dyn_cast<memref::CopyOp>(forwardChain.back());
      Value chainVal = copyOp.getTarget();
      forwardChain.pop_back();
      
      for (auto op : llvm::reverse(forwardChain)) {
        if (!invertChain<memref::ExpandShapeOp, memref::CollapseShapeOp,
            rock::TransformOp>(b, op, chainVal))
          return fail;
      }
      
      // 2.1. replace mem with copy dest
      copyOp->erase();
      allocaMem.replaceAllUsesWith(chainVal);
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
