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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
////  Regularize linalg.generic inputs
////////////////////////////////////////////////////////////////////////
struct RegularizeGenericRewritePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp lgop,
                                PatternRewriter &rw) const override {
    LogicalResult lres = failure();
    Location loc = lgop.getLoc();

    // parallel
    for (StringRef iterType :
           lgop.iterator_types().getAsValueRange<StringAttr>())
      if (iterType != "parallel")
        return lgop.emitError("Only fully parallel supported");

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
        Value val = inp;
        if (imap != outIdxMap) {
          // inject a broadcast
          auto invertOutIdxMap = inversePermutation(outIdxMap);
          auto outToInpMap = imap.compose(invertOutIdxMap);
          val = rock::insertTransposeAndBroadcastTransforms(rw, outType.getShape(), val, outToInpMap);
          lgop->replaceUsesOfWith(inp, val);
          lres = success();
        }
      }
    }

    // reset idxmaps
    SmallVector<AffineMap, 5> newIdxMaps(idxMaps.size(), outIdxMap);
    lgop.indexing_mapsAttr(rw.getAffineMapArrayAttr(newIdxMaps));
    
    return lres;
  }
};

////////////////////////////////////////////////////////////////////////
////  Push Transforms Up To GEMM output
////////////////////////////////////////////////////////////////////////
#if 0
struct PushTransformsUpRewritePattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  /////////////////////////////////////////////////////////////////////
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
      // should never be output of memcpy?
      assert(mcop.getTarget() != result);
      return mcop.getTarget() == result;
    } else if (auto rgop = dyn_cast<rock::GridwiseGemmOp>(forwOp)) {
      return rgop.getC() == result;
    } else if (auto rgop = dyn_cast<rock::GridwiseGemmV2Op>(forwOp)) {
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
      // 1 output
      Value out = lgop.getOutputs()[0];
      auto outType = out.getType().cast<ShapedType>();

      // all index maps must be identity
      auto idxMaps = lgop.getIndexingMapsArray();
      auto outIdxMap = idxMaps.back();
      Location loc = lgop.getLoc();

      PatternRewriter::InsertionGuard guard(rw);
      rw.setInsertionPoint(lgop);
      // apply transforms to inputs
      SmallVector<Value> inps(lgop.getInputs());
      for (auto inp : inps) {
        Value val = inp;
        for (auto writeOp : llvm::reverse(writer)) {
          if (auto top = dyn_cast<rock::TransformOp>(writeOp)) {
            auto tx = rock::invertTransformMap(rw, top.getTransform());
            if (!tx)
              return lgop.emitError("Non-invertible transform");
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
    } else if (isa<rock::GridwiseGemmOp, rock::GridwiseGemmV2Op>(op)) {
      // apply to gemm output buffer
      Location loc = op->getLoc();
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
        // has transforms, replace head with new transforms
        auto top = dyn_cast<rock::TransformOp>(writer[0]);
        top->replaceUsesOfWith(top.getOperand(), val);
      } else {
        // replace gemm result with new transforms
        op->replaceUsesOfWith(op->getOperand(2), val);
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
#else
////////////////////////////////////////////////////////////////////////
////  Push Transforms Up To GEMM output
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
      // should never be output of memcpy?
      assert(mcop.getTarget() != result);
      return mcop.getTarget() == result;
    } else if (auto rgop = dyn_cast<rock::GridwiseGemmOp>(forwOp)) {
      return rgop.getC() == result;
    } else if (auto rgop = dyn_cast<rock::GridwiseGemmV2Op>(forwOp)) {
      return rgop.getC() == result;
    }
    assert(0); // unknown op
    return false;
  }

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rw) const override {
    LogicalResult lres = failure();
    Value buffer = alloc.getResult();

    // find writer and readers (must be a transform)
    bool hasTransforms = false;
    Operation *writer = nullptr;
    SmallVector<SmallVector<Operation *>> readers;
    for (auto &use : buffer.getUses()) {
      Operation *useOp = use.getOwner();
      SmallVector<Operation *> chain;
      bool isWriter = collectChain(buffer, useOp, chain);
      if (isWriter) {
        assert(writer == nullptr);
        writer = useOp;
      } else {
        hasTransforms |= chain.size() > 1;
        readers.push_back(chain);
      }
    }
    if (hasTransforms) {
      PatternRewriter::InsertionGuard guard(rw);
      rw.setInsertionPoint(alloc);
      Location loc = alloc.getLoc();
      
      for (auto reader : readers) {
        if (reader.size() > 1) {
          Operation *readOp = reader.back();
          reader.pop_back();
          assert(!isa<rock::TransformOp>(readOp));
          Operation *lastTOp = reader.back();
          Value readInp = lastTOp->getResult(0);
          
          // create new buffer (substitue in reader)
          MemRefType nbufType = readInp.getType().cast<MemRefType>();
          Value nbuffer = rw.create<memref::AllocOp>(loc, nbufType).getResult();

          // update reader with new buffer input
          readOp->replaceUsesOfWith(readInp, nbuffer);

          // insert inverse transforms after new buffer to writer chain
          Value val = nbuffer;
          for (auto op : llvm::reverse(reader)) {
            auto txOp = dyn_cast<rock::TransformOp>(op);
            auto itx = rock::invertTransformMap(rw, txOp.getTransform());
            if (!itx) {
              assert(0);
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
#endif  
////////////////////////////////////////////////////////////////////////////////
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
  if (!func->getAttrOfType<UnitAttr>("kernel")) {
    // disable for non-kernels (and all multi kernels)
    return;
  }

  {
#if 0
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithmeticDialect, rock::RockDialect,
                           memref::MemRefDialect, linalg::LinalgDialect>();
    target.addIllegalOp<memref::ExpandShapeOp, memref::CollapseShapeOp>();
    target.addDynamicallyLegalOp<linalg::GenericOp>([&](linalg::GenericOp lgop) {
      // parallel
      for (StringRef iterType :
             lgop.iterator_types().getAsValueRange<StringAttr>())
        if (iterType != "parallel")
          return false; //"Only fully parallel supported"

      // 1 output
      auto outs = lgop.getOutputs();
      if (outs.size() > 1)
        return false; //"Only 1 output supported"
      Value out = outs[0];
      auto outType = out.getType().cast<ShapedType>();

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
    });
    
    RewritePatternSet patterns(ctx);
    patterns.add<CollapseRewritePattern, ExpandRewritePattern,
                 RegularizeGenericRewritePattern>(ctx);
    if (failed(applyPartialConversion(func, target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
#else
    RewritePatternSet patterns(ctx);
    patterns.add<CollapseRewritePattern, ExpandRewritePattern,
                 RegularizeGenericRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
#endif
  }

  {
    RewritePatternSet patterns(ctx);
    patterns.add<PushTransformsUpRewritePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
}
