//===- Regularize.cpp - rewrites to allow Rock kernel fusion  ------===//
//
// Copyright 2022 Advanced Micro Devices.
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-regularize"

using namespace mlir;
using namespace mlir::rock;

namespace {

////////////////////////////////////////////////////////////////////////
////  Convert memref.collapse/expand_shape ops to rock.transform
////////////////////////////////////////////////////////////////////////
struct CollapseRewritePattern
    : public OpRewritePattern<memref::CollapseShapeOp> {
  using OpRewritePattern<memref::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CollapseShapeOp collapseOp,
                                PatternRewriter &rw) const final {
    Location loc = collapseOp.getLoc();
    ArrayRef<int64_t> inpShape = collapseOp.getSrcType().getShape();
    ArrayRef<int64_t> outShape = collapseOp.getResultType().getShape();
    SmallVector<ReassociationIndices, 4> reassocs =
        collapseOp.getReassociationIndices();

    rock::TransformMapAttr transform =
        rock::transformCollapseShape(rw, loc, inpShape, outShape, reassocs);
    if (!transform)
      return rw.notifyMatchFailure(
          loc, "could not translate memref collapse into rock transform");
    rw.replaceOpWithNewOp<rock::TransformOp>(collapseOp, collapseOp.getSrc(),
                                             transform);
    return success();
  }
};

struct ExpandRewritePattern : public OpRewritePattern<memref::ExpandShapeOp> {
  using OpRewritePattern<memref::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp expandOp,
                                PatternRewriter &rw) const final {
    Location loc = expandOp.getLoc();
    ArrayRef<int64_t> inpShape = expandOp.getSrcType().getShape();
    ArrayRef<int64_t> outShape = expandOp.getResultType().getShape();
    SmallVector<ReassociationIndices, 4> reassocs =
        expandOp.getReassociationIndices();

    rock::TransformMapAttr transform =
        rock::transformExpandShape(rw, loc, inpShape, outShape, reassocs);
    if (!transform)
      return rw.notifyMatchFailure(
          loc, "could not translate memref expansion into rock transform");
    rw.replaceOpWithNewOp<rock::TransformOp>(expandOp, expandOp.getSrc(),
                                             transform);
    return success();
  }
};

////////////////////////////////////////////////////////////////////////
////  Test linalg.generic for regularity
////////////////////////////////////////////////////////////////////////
static bool isRegularGeneric(linalg::GenericOp lgop) {
  // parallel
  for (utils::IteratorType iterType : lgop.getIteratorTypesArray()) {
    if (!linalg::isParallelIterator(iterType))
      return false; //"Only fully parallel supported"
  }

  // 1 output
  auto outs = lgop.getOutputs();
  if (outs.size() > 1)
    return false; //"Only 1 output supported"

  // all index maps must be identity
  auto idxMaps = lgop.getIndexingMapsArray();
  auto outIdxMap = idxMaps.back();
  if (!outIdxMap.isIdentity()) {
    return false; //"Only output identity map supported"
  }

  for (auto idxMap : idxMaps) {
    if (idxMap != outIdxMap)
      return false; //"Must be same index maps"
  }

  return true;
}

////////////////////////////////////////////////////////////////////////
////  Regularize linalg.generic inputs
////////////////////////////////////////////////////////////////////////
struct RegularizeGenericRewritePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp lgop,
                                PatternRewriter &rw) const override {
    LogicalResult lres = failure();

    // parallel
    for (utils::IteratorType iterType : lgop.getIteratorTypesArray()) {
      if (!linalg::isParallelIterator(iterType))
        return lgop.emitError("Only fully parallel supported");
    }

    // 1 output
    auto outs = lgop.getOutputs();
    if (outs.size() > 1)
      return lgop.emitError("Only 1 output supported");
    Value out = outs[0];
    auto outType = out.getType().cast<ShapedType>();

    // all index maps must be identity
    auto idxMaps = lgop.getIndexingMapsArray();
    auto outIdxMap = idxMaps.back();
    if (!outIdxMap.isIdentity()) {
      return lgop.emitError("Only output identity map supported");
    }

    // apply transforms to inputs
    SmallVector<Value> inps(lgop.getInputs());
    for (auto pair : llvm::zip(inps, idxMaps)) {
      if (auto inp = std::get<0>(pair)) {
        auto imap = std::get<1>(pair);
        if (imap != outIdxMap) {
          // inject a broadcast
          auto invertOutIdxMap = inversePermutation(outIdxMap);
          auto outToInpMap = imap.compose(invertOutIdxMap);
          Value regInp = rock::insertTransposeAndBroadcastTransforms(
              rw, outType.getShape(), inp, outToInpMap);
          lgop->replaceUsesOfWith(inp, regInp);
          lres = success();
        }
      }
    }

    // reset idxmaps
    rw.updateRootInPlace(lgop, [&]() {
      SmallVector<AffineMap, 5> newIdxMaps(idxMaps.size(), outIdxMap);
      lgop.setIndexingMapsAttr(rw.getAffineMapArrayAttr(newIdxMaps));
    });

    return lres;
  }
};

////////////////////////////////////////////////////////////////////////
////  Push Transforms Over alloc to writer
////////////////////////////////////////////////////////////////////////
struct PushTransformsUpRewritePattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  /////////////////////////////////////////////////////////////////////
  static bool collectChain(Value result, Operation *forwOp,
                           SmallVector<Operation *> &chain) {
    while (auto top = dyn_cast<rock::TransformOp>(forwOp)) {
      result = top.getResult();
      if (!result.hasOneUse()) {
        // currently restricted to 1 reader
        LLVM_DEBUG(llvm::dbgs() << "multiple readers on transform\n");
        return false; // TODO: fix when encountered
      }
      chain.push_back(forwOp);
      forwOp = (*result.getUses().begin()).getOwner();
    }
    chain.push_back(forwOp);
    if (auto lgop = dyn_cast<linalg::GenericOp>(forwOp)) {
      return llvm::is_contained(lgop.getOutputs(), result);
    } else if (auto mcop = dyn_cast<memref::CopyOp>(forwOp)) {
      // should never be output of memcpy?
      assert(mcop.getTarget() != result);
      return mcop.getTarget() == result;
    } else if (auto rgop = dyn_cast<rock::GridwiseGemmOp>(forwOp)) {
      return rgop.getC() == result;
    } else if (auto rgop = dyn_cast<rock::GridwiseGemmV2Op>(forwOp)) {
      return rgop.getC() == result;
    } else if (auto rgop = dyn_cast<rock::ThreadwiseWriteAllOp>(forwOp)) {
      return rgop.getDest() == result;
    }
    LLVM_DEBUG(llvm::dbgs() << "unsupported op\n" << *forwOp);
    return false;
  }

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rw) const override {
    LogicalResult lres = failure();
    Value buffer = alloc.getResult();

    // find writer and readChains (must be a transform)
    bool hasTransforms = false;
    Operation *writer = nullptr;
    SmallVector<SmallVector<Operation *>> readChains;
    for (auto &use : buffer.getUses()) {
      Operation *useOp = use.getOwner();
      SmallVector<Operation *> chain;
      bool isWriter = collectChain(buffer, useOp, chain);
      if (isWriter) {
        assert(writer == nullptr);
        writer = useOp;
      } else {
        hasTransforms |= chain.size() > 1;
        readChains.push_back(chain);
      }
    }
    if (hasTransforms) {
      PatternRewriter::InsertionGuard guard(rw);
      rw.setInsertionPoint(alloc);
      Location loc = alloc.getLoc();

      for (auto readChain : readChains) {
        if (readChain.size() > 1) {
          Operation *readOp = readChain.back();
          readChain.pop_back();
          assert(!isa<rock::TransformOp>(readOp));
          Operation *lastTOp = readChain.back();
          Value readInp = lastTOp->getResult(0);

          // create new buffer (substitue in reader)
          MemRefType nbufferType = readInp.getType().cast<MemRefType>();
          Value nbuffer =
              rw.create<memref::AllocOp>(loc, nbufferType).getResult();

          // update reader with new buffer input
          readOp->replaceUsesOfWith(readInp, nbuffer);

          // insert inverse transforms after new buffer to writer chain
          Value val = nbuffer;
          for (auto op : llvm::reverse(readChain)) {
            auto txOp = dyn_cast<rock::TransformOp>(op);
            auto itx = rock::invertTransformMap(rw, txOp.getTransform(), loc);
            if (!itx) {
              LLVM_DEBUG(llvm::dbgs() << "unable to invert transform\n");
              return failure();
            }
            auto top = rw.create<rock::TransformOp>(loc, val, itx);
            val = top.getResult();
          }
          writer->replaceUsesOfWith(alloc, val);

          lres = success();
        }
      }
    }
    return lres;
  }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct RockRegularizePass
    : public rock::impl::RockRegularizePassBase<RockRegularizePass> {
  void runOnOperation() override;
};
} // end namespace

void RockRegularizePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    // disable for non-kernels
    return;
  }

  {
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                           memref::MemRefDialect, linalg::LinalgDialect>();
    target.addIllegalOp<memref::ExpandShapeOp, memref::CollapseShapeOp>();
    target.addDynamicallyLegalOp<linalg::GenericOp>(
        [](linalg::GenericOp op) { return isRegularGeneric(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<CollapseRewritePattern, ExpandRewritePattern,
                 RegularizeGenericRewritePattern>(ctx);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<PushTransformsUpRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
}
