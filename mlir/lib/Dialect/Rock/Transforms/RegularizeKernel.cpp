//===- RegularizeKernel.cpp - rewrites to allow Rock kernel fusion  ------===//
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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEKERNELPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fold-transpose"

using namespace mlir;
using namespace mlir::rock;

namespace {

// This rewrite will rewrite the linalg IO that has view like-ops surrounding
// them to be consumed by the linalg operation itself adjusting the indexing
// maps to faithfully represent them.
struct CollapseRewritePattern
    : public OpRewritePattern<memref::CollapseShapeOp> {
  using OpRewritePattern<memref::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CollapseShapeOp collapse,
                                PatternRewriter &rw) const override {
    auto inpType = collapse.getOperand().getType().cast<ShapedType>();
    auto outType = collapse.getResultType().cast<ShapedType>();
    auto transform = rock::transformCollapseShape(rw, inpType.getShape(),
                                                  outType.getShape());
    rw.replaceOpWithNewOp<rock::TransformOp>(collapse, collapse.getOperand(),
                                             transform);
    return success();
  }
};

struct ExpandRewritePattern : public OpRewritePattern<memref::ExpandShapeOp> {
  using OpRewritePattern<memref::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp expand,
                                PatternRewriter &rw) const override {
    auto inpType = expand.getOperand().getType().cast<ShapedType>();
    auto outType = expand.getResultType().cast<ShapedType>();
    auto transform =
        rock::transformExpandShape(rw, inpType.getShape(), outType.getShape());
    rw.replaceOpWithNewOp<rock::TransformOp>(expand, expand.getOperand(),
                                             transform);
    return success();
  }
};

////////////////////////////////////////////////////////////////////////
////  Shuffle Transforms To Writers
////////////////////////////////////////////////////////////////////////
struct ShuffleTransformsUpRewritePattern
    : public OpRewritePattern<rock::GridwiseGemmOp> {
  using OpRewritePattern<rock::GridwiseGemmOp>::OpRewritePattern;

  static LogicalResult shuffleTransformsUp(PatternRewriter &b, Operation *start,
                                           Value result);
  static LogicalResult applyTransforms(PatternRewriter &b, Value alloc,
                                       Operation *forOp, Operation *backOp);

  LogicalResult matchAndRewrite(rock::GridwiseGemmOp ggemm,
                                PatternRewriter &rewriter) const override {
    // Input:
    // %a0 = alloc
    // %y  = tx0 (%a0)
    // %y  = gemm (%f0, %f1)
    // %z  = tx1 (%a0)
    // %a1 = alloc
    // %a1 = lag0 (%z, %f2)
    // %b  = tx2 (%a1)
    // %f3 = memcpy (%b)

    // Step 0:
    // %a0 = alloc
    // %y  = tx0 (%a0)
    // %y  = gemm (%f0, %f1)
    // %z  = tx1 (%a0)
    // %z' = tx2 (%z)
    // %f2' = tx2 (%f2)
    // %a1' = alloc
    // %a1' = lag0 (%z', %f2') -- all types changed to %a1'
    // %f3 = memcpy (%a1')

    // Step 1:
    // %a0' = alloc
    // %z2' = tx2-i (%a0') -- inverted
    // %z'  = tx1-i (%z2') -- inverted
    // %y  = tx0 (%z')
    // %y  = gemm (%f0, %f1)
    // %f2' = tx2 (%f2)
    // %a1' = alloc
    // %a1' = lag0 (%a0', %f2')
    // %f3 = memcpy (%a1')

    return shuffleTransformsUp(rewriter, ggemm, ggemm.getC());
  }
};

LogicalResult ShuffleTransformsUpRewritePattern::applyTransforms(
    PatternRewriter &b, Value alloc, Operation *forwOp, Operation *backOp) {
  LogicalResult lres = failure();
  Value result;
  SmallVector<TransformMapAttr> transforms;
  while (auto top = dyn_cast<rock::TransformOp>(forwOp)) {
    result = top.getResult();
    if (!result.hasOneUse()) {
      assert(0);        // TODO: fix when encountered
      return failure(); // currently restricted to 1 reader
    }
    auto tmap = rock::invertTransformMap(b, top.getTransform());
    if (!tmap)
      return failure(); // not invertible
    transforms.push_back(tmap);
    forwOp = (*result.getUses().begin()).getOwner();
  }
  if (transforms.size()) {
    // check forwOp is reader
    if (auto laop = dyn_cast<linalg::GenericOp>(forwOp)) {
      if (!llvm::is_contained(laop.getInputs(), result))
        return failure();
    } else if (auto mcop = dyn_cast<memref::CopyOp>(forwOp)) {
      if (mcop.getSource() != result)
        return failure();
    } else {
      assert(0); // TODO: check when encountered
      return failure();
    }

    // apply inverse transforms to reader aligned alloc
    PatternRewriter::InsertionGuard guard(b);
    b.setInsertionPoint(alloc.getDefiningOp());
    Value val = b.create<memref::AllocOp>(forwOp->getLoc(),
                                          result.getType().cast<MemRefType>())
                    .getResult();
    forwOp->replaceUsesOfWith(result, val);
    for (auto tx : llvm::reverse(transforms)) {
      auto top = b.create<rock::TransformOp>(forwOp->getLoc(), val, tx);
      val = top.getResult();
    }
    backOp->replaceUsesOfWith(alloc, val);
    lres = success();
  }
  // recurse to the next level
  if (auto laop = dyn_cast<linalg::GenericOp>(forwOp)) {
    if (laop.getNumOutputs() == 1) {
      if (succeeded(shuffleTransformsUp(b, forwOp, laop.getOutputs()[0])))
        lres = success();
    }
  }
  return lres;
}

LogicalResult ShuffleTransformsUpRewritePattern::shuffleTransformsUp(
    PatternRewriter &b, Operation *start, Value result) {
  Operation *prevOp = start;
  // trace back to alloc
  while (auto op = result.getDefiningOp()) {
    if (isa<memref::AllocOp>(op)) {
      // apply
      Operation *readPath = nullptr;
      for (auto &use : result.getUses()) {
        Operation *useOp = use.getOwner();
        if (useOp != prevOp) {
          if (readPath != nullptr)
            return failure(); // multi readers
          readPath = useOp;
        }
      }
      if (readPath == nullptr)
        return failure();
      return applyTransforms(b, result, readPath, prevOp);
    } else if (auto top = dyn_cast<rock::TransformOp>(op)) {
      prevOp = op;
      result = top.getOperand();
    } else {
      // unknown path
      assert(0);
    }
  }
  return failure();
}

struct PushTransformsUpRewritePattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  static bool collectChain(Value result, Operation *forwOp,
                           SmallVector<Operation *> &chain) {
    while (auto top = dyn_cast<rock::TransformOp>(forwOp)) {
      result = top.getResult();
      if (!result.hasOneUse()) {
        return false; // TODO: fix when encountered
        // return failure(); // currently restricted to 1 reader
      }
      chain.push_back(forwOp);
      forwOp = (*result.getUses().begin()).getOwner();
    }
    chain.push_back(forwOp);
    if (auto lgop = dyn_cast<linalg::GenericOp>(forwOp)) {
      return llvm::is_contained(lgop.getOutputs(), result);
    } else if (auto mcop = dyn_cast<memref::CopyOp>(forwOp)) {
      // should never be output of memcpy
      assert(mcop.getTarget() != result); //
      return mcop.getTarget() == result;
    } else if (auto rgop = dyn_cast<rock::GridwiseGemmOp>(forwOp)) {
      return rgop.getC() == result;
    }
    assert(0); // unknown op
    return false;
  }

  static LogicalResult updateWriter(PatternRewriter &rw,
                                    ArrayRef<Operation *> writer,
                                    ArrayRef<TransformMapAttr> transforms,
                                    Value nbuffer) {
    Operation *op = writer.back();
    if (auto lgop = dyn_cast<linalg::GenericOp>(op)) {
      // parallel
      for (StringRef iterType :
           lgop.iterator_types().getAsValueRange<StringAttr>())
        if (iterType != "parallel")
          return failure();

      // 1 output
      auto outs = lgop.getOutputs();
      if (outs.size() > 1)
        return failure();
      Value out = outs[0];

      // all index maps must be identity
      auto idxMaps = lgop.getIndexingMapsArray();
      auto idxMap = idxMaps[0];
      for (auto imap : idxMaps) {
        if (imap != idxMap || !imap.isIdentity())
          return failure();
      }

      // all input types == output type
      for (auto inp : lgop.getInputs()) {
        if (inp.getType() != out.getType())
          return failure();
      }

      Location loc = lgop.getLoc();

      // apply transforms to inputs
      SmallVector<Value> inps(lgop.getInputs());
      for (auto inp : inps) {
        Value val = inp;
        for (auto writeOp : llvm::reverse(writer)) {
          if (auto top = dyn_cast<rock::TransformOp>(writeOp)) {
            auto tx = rock::invertTransformMap(rw, top.getTransform());
            if (!tx)
              return failure();
            auto ntop = rw.create<rock::TransformOp>(loc, val, tx);
            val = ntop.getResult();
          }
        }
        for (auto tx : transforms) {
          auto top = rw.create<rock::TransformOp>(loc, val, tx);
          val = top.getResult();
        }
        lgop->replaceUsesOfWith(inp, val);
      }
      // update the output with the new buffer
      lgop->replaceUsesOfWith(out, nbuffer);

      // update index maps
      int64_t nrank = nbuffer.getType().cast<ShapedType>().getRank();
      SmallVector<AffineMap, 5> nIdxMaps;
      for (auto idxMap : idxMaps)
        nIdxMaps.push_back(
            AffineMap::getMultiDimIdentityMap(nrank, rw.getContext()));
      lgop.indexing_mapsAttr(rw.getAffineMapArrayAttr(nIdxMaps));

      // update parallel
      SmallVector<StringAttr, 5> nIterators(nrank,
                                            rw.getStringAttr("parallel"));
      lgop.iterator_typesAttr(rw.getArrayAttr(
          ArrayRef<Attribute>(nIterators.begin(), nIterators.end())));

      return success();
    } else if (auto rgop = dyn_cast<rock::GridwiseGemmOp>(op)) {
      // apply to output buffer
      Location loc = rgop.getLoc();
      Value val = nbuffer;
      for (auto tx : llvm::reverse(transforms)) {
        auto itx = rock::invertTransformMap(rw, tx);
        if (!itx) {
          assert(0);
          return failure();
        }
        auto top = rw.create<rock::TransformOp>(loc, val, itx);
        val = top.getResult();
      }
      if (writer.size() > 1) {
        auto top = dyn_cast<rock::TransformOp>(writer[0]);
        top->replaceUsesOfWith(top.getOperand(), val);
      } else {
        rgop->replaceUsesOfWith(rgop.getC(), val);
      }
      return success();
    }
    return failure();
  }

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rw) const override {
    LogicalResult lres = failure();
    Value buffer = alloc.getResult();

    // find writer and readers (must be a transform)
    bool hasTransforms = false;
    SmallVector<Operation *> writer;
    SmallVector<SmallVector<Operation *>> readers;
    for (auto &use : buffer.getUses()) {
      Operation *useOp = use.getOwner();
      SmallVector<Operation *> chain;
      bool isWriter = collectChain(buffer, useOp, chain);
      hasTransforms |= chain.size() > 1;
      if (isWriter) {
        assert(writer.empty());
        writer = chain;
      } else {
        readers.push_back(chain);
      }
    }
    if (hasTransforms) {
      for (auto reader : readers) {
        Operation *readOp = nullptr;
        Value readInp = buffer;
        SmallVector<TransformMapAttr> readTransforms;
        for (auto op : reader) {
          if (auto top = dyn_cast<rock::TransformOp>(op)) {
            readTransforms.push_back(top.getTransform());
            readInp = top.getResult();
          } else {
            readOp = op;
          }
        }
        if (readOp && readTransforms.size()) {
          // create new buffer (substitue in reader and writer)
          PatternRewriter::InsertionGuard guard(rw);
          rw.setInsertionPoint(alloc);
          Value nbuffer =
              rw.create<memref::AllocOp>(alloc.getLoc(),
                                         readInp.getType().cast<MemRefType>())
                  .getResult();
          // update writer inputs if possible
          if (succeeded(updateWriter(rw, writer, readTransforms, nbuffer))) {
            readOp->replaceUsesOfWith(readInp, nbuffer);
            lres = success();
          }
        }
      }
    }
    return lres;
  }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
struct RockRegularizeKernelPass
    : public rock::impl::RockRegularizeKernelPassBase<
          RockRegularizeKernelPass> {
  void runOnOperation() override;
};
} // end namespace

void RockRegularizeKernelPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    return;
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<CollapseRewritePattern, ExpandRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }

  if (0) {
    RewritePatternSet patterns(ctx);
    patterns.add<ShuffleTransformsUpRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<PushTransformsUpRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
}
