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

    if (expanded)
      b.replaceOpWithNewOp<memref::ExpandShapeOp>(op, res.getType(), src,
                                                  merges);
    else
      b.replaceOpWithNewOp<memref::CollapseShapeOp>(op, src, merges);
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
