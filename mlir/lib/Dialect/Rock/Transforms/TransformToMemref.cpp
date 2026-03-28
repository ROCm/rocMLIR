//===- TransformToMemref - MLIR rock.transform conversion pass ---===//
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
// ============================================================
//
// This pass converts any remaining rock.transforms after rock
// lowering back to memref.expand/collapse_shape ops. Otherwise fails.
// This generally only happens for non-conv/gemm kernels such as init
// kernels.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTRANSFORMTOMEMREFPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-transform-to-memref"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

/// A TransformOp whose map consists of a single Slice lowers to memref.subview.
struct SliceTransformRewritePattern : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransformOp op,
                                PatternRewriter &b) const override {
    auto ops = llvm::to_vector(op.getTransform().getOps());
    if (ops.size() != 1)
      return failure();
    auto tattr = ops[0];
    if (tattr.getType() != rock::TransformType::Slice)
      return failure();

    auto src = cast<TypedValue<MemRefType>>(op.getOperand());
    auto res = cast<TypedValue<MemRefType>>(op.getResult());
    ArrayRef<int64_t> resShape = res.getType().getShape();

    // Slice{b0, e0, b1, e1, ..., bN, eN} — begin/end pairs per dim.
    ArrayRef<int64_t> params = tattr.getParams();
    int64_t rank = (int64_t)resShape.size();
    if ((int64_t)params.size() != rank * 2)
      return failure();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (int64_t i = 0; i < rank; ++i) {
      offsets.push_back(b.getIndexAttr(params[2 * i]));
      sizes.push_back(b.getIndexAttr(resShape[i]));
      strides.push_back(b.getIndexAttr(1));
    }

    // Use inferred result type so strides from a transposed source propagate.
    auto subview = b.create<memref::SubViewOp>(op.getLoc(), src, offsets,
                                               sizes, strides);
    b.replaceOp(op, subview.getResult());
    return success();
  }
};

/// A TransformOp that is a pure PassThrough dimension permutation (same rank,
/// no merges) lowers to memref.transpose.
struct TransposeTransformRewritePattern : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransformOp op,
                                PatternRewriter &b) const override {
    auto ops = llvm::to_vector(op.getTransform().getOps());
    auto src = cast<TypedValue<MemRefType>>(op.getOperand());
    auto res = cast<TypedValue<MemRefType>>(op.getResult());
    int64_t rank = (int64_t)res.getType().getRank();

    // Must be all PassThrough with same rank src and result.
    if (src.getType().getRank() != (size_t)rank)
      return failure();
    for (auto tattr : ops)
      if (tattr.getType() != rock::TransformType::PassThrough)
        return failure();

    // Build permutation: result dim i comes from source dim perm[i].
    SmallVector<int64_t> perm(rank, -1);
    for (auto tattr : ops) {
      ArrayRef<uint32_t> upper = tattr.getUpperDims();
      ArrayRef<uint32_t> lower = tattr.getLowerDims();
      for (auto [u, l] : llvm::zip(upper, lower))
        perm[u] = l;
    }
    for (int64_t p : perm)
      if (p < 0)
        return failure();

    // Identity permutation → just replace with the source.
    bool isIdentity = true;
    for (int64_t i = 0; i < rank; ++i)
      if (perm[i] != i) { isIdentity = false; break; }
    if (isIdentity) {
      b.replaceAllUsesWith(res, src);
      b.eraseOp(op);
      return success();
    }

    // memref.transpose takes an affine map attr (permutation map).
    SmallVector<unsigned> uperm(perm.begin(), perm.end());
    AffineMap permMap = AffineMap::getPermutationMap(uperm, op.getContext());
    b.replaceOpWithNewOp<memref::TransposeOp>(op, src,
                                              AffineMapAttr::get(permMap));
    return success();
  }
};

/// A TransformOp whose map contains AddDim (unit-dim insertion) possibly mixed
/// with PassThrough/Merge/Unmerge lowers to memref.reinterpret_cast.
/// This handles cases like (64x16x16) → (1x64x16x16) via AddDim+Unmerge.
struct AddDimTransformRewritePattern : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransformOp op,
                                PatternRewriter &b) const override {
    bool hasAddDim = false;
    for (auto tattr : op.getTransform().getOps()) {
      switch (tattr.getType()) {
      case rock::TransformType::AddDim:
        hasAddDim = true;
        break;
      case rock::TransformType::PassThrough:
      case rock::TransformType::Merge:
      case rock::TransformType::Unmerge:
        break;
      default:
        return failure(); // not handled here
      }
    }
    if (!hasAddDim)
      return failure();

    auto src = cast<TypedValue<MemRefType>>(op.getOperand());
    auto res = cast<TypedValue<MemRefType>>(op.getResult());
    ArrayRef<int64_t> resShape = res.getType().getShape();
    int64_t rank = (int64_t)resShape.size();

    // Compute row-major (C-contiguous) strides for the result shape.
    SmallVector<int64_t> strides(rank);
    strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * resShape[i + 1];

    SmallVector<OpFoldResult> strideVals, offsetVals, sizeVals;
    offsetVals.push_back(b.getIndexAttr(0));
    for (int64_t i = 0; i < rank; ++i) {
      sizeVals.push_back(b.getIndexAttr(resShape[i]));
      strideVals.push_back(b.getIndexAttr(strides[i]));
    }

    b.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, res.getType(), src, offsetVals[0], sizeVals, strideVals);
    return success();
  }
};

namespace {
struct RockTransformToMemrefPass
    : public rock::impl::RockTransformToMemrefPassBase<
          RockTransformToMemrefPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// TransformOp conversion to MemRef
//   This is needed for init kernels that don't fold rock.transform into
//   transforming_for ops.
//===----------------------------------------------------------------------===//
struct TransformRewritePattern : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransformOp op,
                                PatternRewriter &b) const override {
    auto src = cast<TypedValue<ShapedType>>(op.getOperand());
    auto srcShape = src.getType().getShape();
    auto res = cast<TypedValue<ShapedType>>(op.getResult());
    auto resShape = res.getType().getShape();

    bool expanded = resShape.size() > srcShape.size();
    SmallVector<ReassociationIndices> merges(expanded ? srcShape.size()
                                                      : resShape.size());

    // only converts simple expand/collapse form
    for (auto tattr : op.getTransform().getOps()) {
      ArrayRef<uint32_t> inDims =
          expanded ? tattr.getLowerDims() : tattr.getUpperDims();
      ArrayRef<uint32_t> outDims =
          expanded ? tattr.getUpperDims() : tattr.getLowerDims();
      switch (tattr.getType()) {
      case rock::TransformType::PassThrough:
        for (auto pair : llvm::zip(inDims, outDims)) {
          auto inDim = std::get<0>(pair);
          assert(merges[inDim].empty());
          merges[inDim].push_back(std::get<1>(pair));
        }
        break;
      case rock::TransformType::Pad:
      case rock::TransformType::Slice:
      case rock::TransformType::Embed:
      case rock::TransformType::Broadcast:
      case rock::TransformType::AddDim:
      case rock::TransformType::ConstDim:
        return failure(); // Unsupported
      case rock::TransformType::Unmerge:
      case rock::TransformType::Merge: {
        auto inDim = inDims[0];
        assert(merges[inDim].empty());
        for (auto outDim : outDims)
          merges[inDim].push_back(outDim);
        break;
      }
      }
    }

    if (srcShape == resShape) {
      b.replaceAllUsesWith(res, src);
      b.eraseOp(op);
    } else if (expanded) {
      b.replaceOpWithNewOp<memref::ExpandShapeOp>(op, res.getType(), src,
                                                  merges);
    } else {
      // memref.collapse_shape requires a contiguous (identity-layout) source.
      // If the source has a non-identity layout (e.g., from a preceding
      // memref.transpose), first materialize a contiguous copy.
      auto srcMemRefType = cast<MemRefType>(src.getType());
      if (!srcMemRefType.getLayout().isIdentity()) {
        MemRefType contType =
            MemRefType::get(srcShape, srcMemRefType.getElementType());
        auto tmp = b.create<memref::AllocOp>(op.getLoc(), contType);
        b.create<memref::CopyOp>(op.getLoc(), cast<Value>(src), tmp.getResult());
        b.replaceOpWithNewOp<memref::CollapseShapeOp>(op, tmp.getResult(),
                                                      merges);
      } else {
        b.replaceOpWithNewOp<memref::CollapseShapeOp>(op, src, merges);
      }
    }
    return success();
  }
};

void RockTransformToMemrefPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  // Apply the non-Rock transform patterns greedily first (transpose, slice,
  // AddDim), so that their strided/subview results propagate correctly before
  // we check that no TransformOp remains.
  RewritePatternSet greedyPatterns(ctx);
  greedyPatterns.add<TransposeTransformRewritePattern,
                     AddDimTransformRewritePattern,
                     SliceTransformRewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(greedyPatterns))))
    return signalPassFailure();

  // Now apply the reshape-based conversion for pure Merge/Unmerge/PassThrough
  // transforms, and verify no TransformOp remains.
  ConversionTarget target(*ctx);
  target.addIllegalOp<TransformOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         affine::AffineDialect, memref::MemRefDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<TransformRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
