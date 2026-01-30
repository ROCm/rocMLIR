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
struct TransformRewritePattern : public OpRewritePattern<TransformOp> {
  using OpRewritePattern<TransformOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransformOp op,
                                PatternRewriter &b) const override {
    auto src = cast<TypedValue<MemRefType>>(op.getOperand());
    MemRefType srcType = src.getType();
    ArrayRef<int64_t> srcShape = srcType.getShape();
    auto res = cast<TypedValue<MemRefType>>(op.getResult());
    MemRefType resType = res.getType();
    ArrayRef<int64_t> resShape = resType.getShape();

    // First, check if this is a Slice-only transform (possibly with
    // PassThrough). These need special handling via memref.subview.
    bool hasSlice = false;
    bool hasOtherTransforms = false;
    for (auto tattr : op.getTransform().getOps()) {
      switch (tattr.getType()) {
      case rock::TransformType::Slice:
        hasSlice = true;
        break;
      case rock::TransformType::PassThrough:
        // PassThrough is compatible with Slice
        break;
      default:
        hasOtherTransforms = true;
        break;
      }
    }

    // Handle Slice-only case (possibly with PassThrough)
    if (hasSlice && !hasOtherTransforms && srcShape.size() == resShape.size()) {
      // Build subview parameters
      SmallVector<OpFoldResult> offsets(srcShape.size());
      SmallVector<OpFoldResult> sizes(srcShape.size());
      SmallVector<OpFoldResult> strides(srcShape.size());

      // Initialize with identity (offset=0, size=result dim, stride=1)
      for (size_t i = 0; i < srcShape.size(); i++) {
        offsets[i] = b.getIndexAttr(0);
        sizes[i] = b.getIndexAttr(resShape[i]);
        strides[i] = b.getIndexAttr(1);
      }

      // Apply Slice parameters
      for (auto tattr : op.getTransform().getOps()) {
        if (tattr.getType() == rock::TransformType::Slice) {
          ArrayRef<int64_t> params = tattr.getParams();
          ArrayRef<uint32_t> upperDims = tattr.getUpperDims();
          // Slice params are [start0, end0, start1, end1, ...]
          for (size_t i = 0; i < upperDims.size(); i++) {
            int64_t start = params[i * 2];
            int64_t end = params[i * 2 + 1];
            uint32_t dim = upperDims[i];
            offsets[dim] = b.getIndexAttr(start);
            sizes[dim] = b.getIndexAttr(end - start);
          }
        }
      }

      auto subview = memref::SubViewOp::create(b, op.getLoc(), src, offsets,
                                               sizes, strides);

      // The subview result type may have a different layout than the expected
      // result type. Use memref.cast if needed.
      if (subview.getType() != resType) {
        b.replaceOpWithNewOp<memref::CastOp>(op, resType, subview);
      } else {
        b.replaceOp(op, subview.getResult());
      }
      return success();
    }

    // Handle expand/collapse shape transforms
    bool expanded = resShape.size() > srcShape.size();
    SmallVector<ReassociationIndices> merges(expanded ? srcShape.size()
                                                      : resShape.size());

    // Track AddDim result dimensions (for expanded case) to incorporate later
    SmallVector<uint32_t> addDimResultDims;

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
      case rock::TransformType::ConstDim:
        return failure(); // Unsupported
      case rock::TransformType::AddDim:
        // AddDim adds a dimension of size 1 that doesn't exist in source.
        // For expand_shape, we can handle this by including the AddDim
        // dimension in a source dimension's reassociation group.
        // Only supported when expanding (adding dimensions).
        if (!expanded)
          return failure();
        // Track the result dimension for later incorporation
        for (auto outDim : outDims)
          addDimResultDims.push_back(outDim);
        break;
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

    // If we have AddDim dimensions to incorporate (in expanded case)
    if (!addDimResultDims.empty()) {
      // For each AddDim result dimension, find an adjacent source dimension's
      // merge group to add it to. An AddDim at result index d should be grouped
      // with a source dimension whose merge group contains an adjacent result
      // index (d-1 or d+1).
      for (auto addDimIdx : addDimResultDims) {
        bool found = false;
        // First, try to find a merge group containing an adjacent result dim
        for (size_t srcDim = 0; srcDim < merges.size() && !found; srcDim++) {
          for (int32_t idx : merges[srcDim]) {
            if (idx == static_cast<int32_t>(addDimIdx) - 1 ||
                idx == static_cast<int32_t>(addDimIdx) + 1) {
              merges[srcDim].push_back(addDimIdx);
              found = true;
              break;
            }
          }
        }
        // Fall back to the last non-empty group
        if (!found) {
          for (int srcDim = merges.size() - 1; srcDim >= 0; srcDim--) {
            if (!merges[srcDim].empty()) {
              merges[srcDim].push_back(addDimIdx);
              found = true;
              break;
            }
          }
        }
        if (!found)
          return failure(); // No suitable group found
      }

      // Sort each reassociation group (required by expand_shape)
      for (auto &group : merges)
        llvm::sort(group);
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
  patterns.add<TransformRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
