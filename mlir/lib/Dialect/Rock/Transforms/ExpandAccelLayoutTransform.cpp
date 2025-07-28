//===- ExpandAccelLayoutTransform - MLIR Rock ops lowering passes -----===//
//
// Copyright 2025 The MLIR Authors.
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
// This pass is needed in order to have a common MLIR representation for
// tuning when using accel layout for tensors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <numeric>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKEXPANDACCELLAYOUTTRANSFORMPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-expand-accel-layout-transform"

using namespace mlir;

namespace {
struct RockExpandAccelLayoutTransformPass
    : public rock::impl::RockExpandAccelLayoutTransformPassBase<
          RockExpandAccelLayoutTransformPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static Value accelLayoutToStandard(OpBuilder &b, ArrayRef<int64_t> shape,
                                   ArrayRef<StringRef> nameList,
                                   StringRef dimension, bool kBlockFirst,
                                   Value accelLayoutTensor) {
  assert(dimension == "m" || dimension == "n");
  Location loc = accelLayoutTensor.getLoc();
  auto logicalShapedTy = cast<ShapedType>(accelLayoutTensor.getType());

  SmallVector<uint32_t> upperDims(shape.size());
  std::iota(upperDims.begin(), upperDims.end(), 0);
  SmallVector<uint32_t> nonUnitUpperDim;
  SmallVector<int64_t> nonUnitUpperSize;
  SmallVector<StringRef> nonUnitUpperName;
  for (auto [upperDim, name, dimLen] : llvm::zip(upperDims, nameList, shape)) {
    if (dimLen != 1) {
      nonUnitUpperDim.push_back(upperDim);
      nonUnitUpperName.push_back(name);
      nonUnitUpperSize.push_back(dimLen);
    }
  }
  // there has to be at least one dimension that is unmerged
  if (nonUnitUpperDim.empty()) {
    nonUnitUpperDim.push_back(upperDims.back());
    nonUnitUpperName.push_back(nameList.back());
    nonUnitUpperSize.push_back(shape.back());
  }

  rock::BottomUpTMBuilder flattener(b, {"raw"},
                                    logicalShapedTy.getNumElements(), loc);
  flattener.unmerge(nonUnitUpperName, nonUnitUpperDim, "raw", nonUnitUpperSize);
  for (auto dim : upperDims) {
    if (!llvm::is_contained(nonUnitUpperDim, dim)) {
      flattener.addDim(nameList[dim], dim, shape[dim]);
    }
  }
  rock::TransformMapAttr flattenerAttr = flattener.get();

  auto transposer = rock::BottomUpTMBuilder::above(flattener, flattenerAttr);
  if (kBlockFirst)
    // B x d x k x kpackperblock x dperblock x kpack -> B x d x dperblock x k x
    // kpackperblock x kpack
    transposer.passThrough(ArrayRef<uint32_t>{0, 1, 2, 3, 4, 5},
                           ArrayRef<uint32_t>{0, 1, 4, 2, 3, 5});
  else
    // B x k x d x kpackperblock x dperblock x kpack -> B x d x dperblock x k x
    // kpackperblock x kpack
    transposer.passThrough(ArrayRef<uint32_t>{0, 1, 2, 3, 4, 5},
                           ArrayRef<uint32_t>{0, 2, 4, 1, 3, 5});
  rock::TransformMapAttr transposerAttr = transposer.get();

  // B x d x dperblock x k x kpackperblock x kpack -> B x D x K (or B x K x D if
  // transposed or B tensor)
  auto merger = rock::BottomUpTMBuilder::above(transposer, transposerAttr);
  // passThrough the batch dimension
  merger.passThrough(nameList[0]);
  uint32_t kOutDim = kBlockFirst ? 2 : 1;
  uint32_t dOutDim = kOutDim == 2 ? 1 : 2;
  merger.merge(dimension, dOutDim, {nameList[dOutDim], nameList[4]});
  merger.merge("k", kOutDim, {nameList[kOutDim], nameList[3], nameList[5]});
  rock::TransformMapAttr mergerAttr = merger.get();

  SmallVector<Attribute> transformAttrs{mergerAttr, transposerAttr,
                                        flattenerAttr};
  return rock::transform(b, accelLayoutTensor, b.getArrayAttr(transformAttrs));
}

struct ExpandAccelLayout
    : public OpRewritePattern<rock::AccelLayoutTransformOp> {
  using OpRewritePattern<rock::AccelLayoutTransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::AccelLayoutTransformOp op,
                                PatternRewriter &b) const override {
    StringRef dName = op.getIsA() ? "m" : "n";
    bool transposed = op.getTransposed();
    StringRef dPerBlockName = op.getIsA() ? "mPerBlock" : "nPerBlock";
    bool kBlockFirst =
        (dName == "m" && !transposed) || (dName == "n" && transposed);
    StringRef dim1Name = kBlockFirst ? dName : "k";
    StringRef dim2Name = kBlockFirst ? "k" : dName;
    SmallVector<StringRef> nameList = {
        "g", dim1Name, dim2Name, "kPackPerBlock", dPerBlockName, "kPack"};
    auto maybeParams = op.getParams();
    if (!maybeParams.has_value())
      return b.notifyMatchFailure(op, "missing tuning parameters");

    auto params = maybeParams.value();
    int64_t mPerBlock, nPerBlock, kPackPerBlock, kPack;
    kPack = params.getKpack();
    if (auto xdlopsParams =
            dyn_cast<rock::XdlopsGemmDerivedParamsAttr>(params)) {
      mPerBlock = xdlopsParams.getMPerBlock();
      nPerBlock = xdlopsParams.getNPerBlock();
      kPackPerBlock = xdlopsParams.getKpackPerBlock();
    } else if (auto wmmaParams = dyn_cast<rock::WmmaGemmParamsAttr>(params)) {
      mPerBlock = wmmaParams.getMPerBlock();
      nPerBlock = wmmaParams.getNPerBlock();
      kPackPerBlock = wmmaParams.getKpackPerBlock();
    } else if (auto generalParams =
                   dyn_cast<rock::GeneralGemmParamsAttr>(params)) {
      mPerBlock = generalParams.getMPerBlock();
      nPerBlock = generalParams.getNPerBlock();
      kPackPerBlock = generalParams.getKPerBlock();
      assert(kPack == 1);
    } else
      return b.notifyMatchFailure(op, "unsupported tuning parameters");

    int64_t dPerBlock = op.getIsA() ? mPerBlock : nPerBlock;
    ShapedType outputType = cast<ShapedType>(op.getType());
    if (outputType.getRank() != 3)
      return b.notifyMatchFailure(op, "wrong output type rank");

    int64_t g = outputType.getShape()[0];
    int64_t d =
        kBlockFirst ? outputType.getShape()[1] : outputType.getShape()[2];
    int64_t k =
        kBlockFirst ? outputType.getShape()[2] : outputType.getShape()[1];
    int64_t kPerBlock = kPackPerBlock * kPack;
    if (d % dPerBlock != 0 || k % kPerBlock != 0)
      return b.notifyMatchFailure(
          op, "output shape is not compatible with accel layout");

    int64_t dBlocks = d / dPerBlock;
    int64_t kBlocks = k / kPerBlock;
    int64_t dim1 = kBlockFirst ? dBlocks : kBlocks;
    int64_t dim2 = kBlockFirst ? kBlocks : dBlocks;

    SmallVector<int64_t> shapeList = {g,         dim1, dim2, kPackPerBlock,
                                      dPerBlock, kPack};

    Value result = accelLayoutToStandard(b, shapeList, nameList, dName,
                                         kBlockFirst, op.getInput());
    b.replaceOp(op, result);

    return success();
  }
};

void RockExpandAccelLayoutTransformPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addLegalDialect<rock::RockDialect>();
  target.addIllegalOp<rock::AccelLayoutTransformOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<ExpandAccelLayout>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
