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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

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

/// Pattern to convert pure Slice transforms to memref.subview
struct SliceTransformRewritePattern : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransformOp op,
                                PatternRewriter &b) const override {
    auto ops = op.getTransform().getOps();

    // Check if this is a pure Slice transform (single Slice op covering all
    // dims)
    if (ops.size() != 1)
      return failure();

    auto tattr = ops[0];
    if (tattr.getType() != rock::TransformType::Slice)
      return failure();

    auto src = cast<TypedValue<MemRefType>>(op.getOperand());
    auto srcType = src.getType();
    auto res = cast<TypedValue<MemRefType>>(op.getResult());
    auto resType = res.getType();

    // Slice params are pairs of (begin, end) for each dimension
    ArrayRef<int64_t> params = tattr.getParams();
    int64_t numDims = static_cast<int64_t>(params.size() / 2);

    if (numDims != srcType.getRank())
      return failure();

    // Build offsets, sizes, strides for subview
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (int64_t i = 0; i < numDims; ++i) {
      int64_t begin = params[i * 2];
      int64_t end = params[i * 2 + 1];
      int64_t size = end - begin;

      offsets.push_back(b.getIndexAttr(begin));
      sizes.push_back(b.getIndexAttr(size));
      strides.push_back(b.getIndexAttr(1));
    }

    // Create subview
    auto subviewType = memref::SubViewOp::inferRankReducedResultType(
        resType.getShape(), srcType, offsets, sizes, strides);

    auto subview = memref::SubViewOp::create(b, op.getLoc(), subviewType, src,
                                             offsets, sizes, strides);

    // If the result type matches exactly, we can replace directly
    if (subviewType == resType) {
      b.replaceOp(op, subview.getResult());
    } else {
      // Need a cast if the types differ (e.g., different strides in type)
      auto cast = memref::CastOp::create(b, op.getLoc(), resType, subview);
      b.replaceOp(op, cast.getResult());
    }
    return success();
  }
};

/// Pattern to convert transforms with PassThrough + Broadcast{1} to
/// memref.reinterpret_cast with zero strides for broadcast dimensions.
/// This handles broadcast from size-1 dimensions to larger dimensions.
struct BroadcastTransformRewritePattern : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransformOp op,
                                PatternRewriter &b) const override {
    auto ops = op.getTransform().getOps();
    auto src = cast<TypedValue<MemRefType>>(op.getOperand());
    auto srcType = src.getType();
    auto res = cast<TypedValue<MemRefType>>(op.getResult());
    auto resType = res.getType();

    // Check if ranks match (broadcast doesn't change rank)
    if (srcType.getRank() != resType.getRank())
      return failure();

    int64_t rank = srcType.getRank();
    SmallVector<bool> isBroadcast(rank, false);
    SmallVector<bool> isPassThrough(rank, false);

    // Analyze the transforms - must be only PassThrough and Broadcast{1}
    for (auto tattr : ops) {
      switch (tattr.getType()) {
      case rock::TransformType::PassThrough:
        for (auto dim : tattr.getUpperDims()) {
          if (dim >= static_cast<uint32_t>(rank))
            return failure();
          isPassThrough[dim] = true;
        }
        break;
      case rock::TransformType::Broadcast:
        // Broadcast must be from size 1 (params[i] == 1)
        for (size_t i = 0; i < tattr.getUpperDims().size(); ++i) {
          auto dim = tattr.getUpperDims()[i];
          if (dim >= static_cast<uint32_t>(rank))
            return failure();
          // Check that source dimension size is 1
          if (i >= tattr.getParams().size() || tattr.getParams()[i] != 1)
            return failure();
          if (srcType.getDimSize(dim) != 1)
            return failure();
          isBroadcast[dim] = true;
        }
        break;
      default:
        // Unsupported transform type in this pattern
        return failure();
      }
    }

    // All dimensions must be either PassThrough or Broadcast
    for (int64_t i = 0; i < rank; ++i) {
      if (!isPassThrough[i] && !isBroadcast[i])
        return failure();
    }

    // Build strides: 0 for broadcast dimensions, computed stride for others
    SmallVector<int64_t> strides(rank);
    SmallVector<int64_t> sizes = llvm::to_vector(resType.getShape());

    // Compute strides in row-major order, but set broadcast dims to 0
    int64_t stride = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
      if (isBroadcast[i]) {
        strides[i] = 0;
      } else {
        strides[i] = stride;
        stride *= srcType.getDimSize(i);
      }
    }

    // Create reinterpret_cast with the new strides
    auto staticOffsets = SmallVector<int64_t>(1, 0);
    auto newType =
        MemRefType::get(sizes, resType.getElementType(),
                        StridedLayoutAttr::get(b.getContext(), 0, strides));

    auto reinterpret = memref::ReinterpretCastOp::create(
        b, op.getLoc(), newType, src, /*offset=*/0, sizes, strides);

    // Cast to the expected result type if needed
    if (newType == resType) {
      b.replaceOp(op, reinterpret.getResult());
    } else {
      auto cast =
          memref::CastOp::create(b, op.getLoc(), resType, reinterpret);
      b.replaceOp(op, cast.getResult());
    }
    return success();
  }
};

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

    // Track which upper dimensions are from AddDim (unit dims not from source)
    SmallVector<int64_t> addDimUpperDims;

    // First pass: collect AddDim dimensions
    for (auto tattr : op.getTransform().getOps()) {
      if (tattr.getType() == rock::TransformType::AddDim) {
        // AddDim must have size 1 for us to handle it
        if (tattr.getParams().size() != 1 || tattr.getParams()[0] != 1)
          return failure();
        for (auto dim : tattr.getUpperDims())
          addDimUpperDims.push_back(dim);
      }
    }

    // Second pass: build merge groups
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
      case rock::TransformType::AddDim:
        // Handled separately - AddDim{1} dimensions will be attached to
        // adjacent groups after this loop
        break;
      case rock::TransformType::Pad:
      case rock::TransformType::Slice:
      case rock::TransformType::Embed:
      case rock::TransformType::Broadcast:
      case rock::TransformType::ConstDim:
        return failure(); // Unsupported in this pattern
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

    // For AddDim dimensions in expand case, attach them to the first non-empty
    // merge group (they represent unit dimensions added during expansion)
    if (expanded && !addDimUpperDims.empty()) {
      // Find the first non-empty merge group and add AddDim dims to it
      for (auto &merge : merges) {
        if (!merge.empty()) {
          for (auto addDim : addDimUpperDims) {
            merge.push_back(addDim);
          }
          // Sort to maintain dimension order
          llvm::sort(merge);
          break;
        }
      }
      // If no merge group was found, we can't handle this
      bool anyNonEmpty = llvm::any_of(merges, [](auto &m) { return !m.empty(); });
      if (!anyNonEmpty && !addDimUpperDims.empty())
        return failure();
    }

    if (srcShape == resShape) {
      b.replaceAllUsesWith(res, src);
      b.eraseOp(op);
    } else if (expanded) {
      b.replaceOpWithNewOp<memref::ExpandShapeOp>(op, res.getType(), src,
                                                  merges);
    } else {
      b.replaceOpWithNewOp<memref::CollapseShapeOp>(op, src, merges);
    }
    return success();
  }
};

void RockTransformToMemrefPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<TransformOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         affine::AffineDialect, memref::MemRefDialect>();

  RewritePatternSet patterns(ctx);
  // Add specialized patterns with higher benefit to try them first
  patterns.add<SliceTransformRewritePattern>(ctx, /*benefit=*/3);
  patterns.add<BroadcastTransformRewritePattern>(ctx, /*benefit=*/2);
  patterns.add<TransformRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
