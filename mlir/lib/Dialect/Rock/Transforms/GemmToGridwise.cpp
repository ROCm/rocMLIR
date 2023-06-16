//===- GemmToGridwise.cpp - Rock GEMM implementation ------------===//
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
//
// This pass converts rock.gemm into the appropriate rock.gridwise_gemm
// adding padding and group dimensions if needed.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <memory>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGEMMTOGRIDWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gemm-to-gridwise"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockGemmToGridwisePass
    : public rock::impl::RockGemmToGridwisePassBase<RockGemmToGridwisePass> {
  void runOnOperation() override;
};

struct GemmRewritePattern : public OpConversionPattern<GemmOp> {
  using OpConversionPattern<GemmOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(GemmOp op, GemmOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override;
};
} // end namespace

/// Normalize the argument into the form requested.
/// If a group dimension is not present, add one.
/// If doTranspose is true, meaning the user's transpose requests don't match
/// what the underlying gridwise gemm expects, transpose the matrix to match,
/// using firstDim as the name of the first dimension in the new value and
/// secondDim as the name of the second dimesion.
static Value normalizeMatrix(Value matrix, ConversionPatternRewriter &b,
                             Location loc, bool doTranspose, StringRef firstDim,
                             StringRef secondDim) {
  auto matrixType = matrix.getType().cast<MemRefType>();
  bool addGroup = matrixType.getShape().size() != 3;
  if (!addGroup && !doTranspose)
    return matrix;
  SmallVector<StringRef, 3> bottomNames;
  if (!addGroup)
    bottomNames.push_back("gemmG");
  if (doTranspose)
    bottomNames.append({secondDim, firstDim});
  else
    bottomNames.append({firstDim, secondDim});
  BottomUpTMBuilder normalizer(b, bottomNames, matrixType.getShape(), loc);

  if (addGroup)
    normalizer.addDim("gemmG", 0, 1);
  else
    normalizer.passThrough(normalizer.startName(0));

  normalizer.passThrough({firstDim, secondDim}, {1, 2}, {firstDim, secondDim});
  TransformMapAttr normalizeAttr = normalizer.get();
  return b.create<TransformOp>(loc, matrix, normalizeAttr);
}

/// Apply padding to a matrix in its `firstDim` and `secondDim` if applicable.
static Value padMatrix(Value matrix, ConversionPatternRewriter &b, Location loc,
                       StringRef firstDim, int64_t firstDimPad,
                       StringRef secondDim, int64_t secondDimPad) {
  if (firstDimPad == 0 && secondDimPad == 0)
    return matrix;
  ArrayRef<int64_t> shape = matrix.getType().cast<MemRefType>().getShape();
  BottomUpTMBuilder padder(b, {"gemmG", firstDim, secondDim}, shape, loc);
  padder.passThrough("gemmG");
  if (firstDimPad == 0) {
    padder.passThrough(firstDim);
  } else {
    SmallString<8> paddedName;
    (firstDim + Twine("Pad")).toVector(paddedName);
    padder.pad(paddedName, firstDim, 0, firstDimPad);
  }
  if (secondDimPad == 0) {
    padder.passThrough(secondDim);
  } else {
    SmallString<8> paddedName;
    (secondDim + Twine("Pad")).toVector(paddedName);
    padder.pad(paddedName, secondDim, 0, secondDimPad);
  }
  TransformMapAttr padAttr = padder.get();
  return b.create<TransformOp>(loc, matrix, padAttr);
}

LogicalResult
GemmRewritePattern::matchAndRewrite(GemmOp op, GemmOpAdaptor adaptor,
                                    ConversionPatternRewriter &rw) const {
  Location loc = op->getLoc();

  if (!adaptor.getA().getType().isa<MemRefType>())
    return op.emitOpError("Cannot lower unbufferized gemm to gridwise");

  Attribute params = op.getParams().value_or(nullptr);
  if (!params) {
    return op.emitOpError("cannot lower gemm without tuning parameters");
  }

  Value a = adaptor.getA(), b = adaptor.getB(), c = adaptor.getC();
  // Note: the gridwise ops take K x M and K x N, so A must be transposed if
  // it's in the natural M x K form
  a = normalizeMatrix(a, rw, loc, !op.getATransposed(), "gemmK", "gemmM");
  b = normalizeMatrix(b, rw, loc, op.getBTransposed(), "gemmK", "gemmN");
  c = normalizeMatrix(c, rw, loc, op.getCTransposed(), "gemmM", "gemmN");

  // Note, matrix dimension correctness is handled in the verifier
  ArrayRef<int64_t> aShape = a.getType().cast<MemRefType>().getShape();
  ArrayRef<int64_t> bShape = b.getType().cast<MemRefType>().getShape();
  GemmSize size(/*g=*/aShape[0], /*m=*/aShape[2], /*k=*/aShape[1],
                /*n=*/bShape[2]);
  GemmSize extraPad =
      requiredPadding(params, size).value_or(GemmSize{0, 0, 0, 0});

  a = padMatrix(a, rw, loc, "gemmK", extraPad.k, "gemmM", extraPad.m);
  b = padMatrix(b, rw, loc, "gemmK", extraPad.k, "gemmN", extraPad.n);
  c = padMatrix(c, rw, loc, "gemmM", extraPad.m, "gemmN", extraPad.n);

  IntegerAttr blockSize = op.getDerivedBlockSizeAttr();
  bool isAccel = rock::isAccel(op.getFeatures());

  if (isAccel && !blockSize)
    return op.emitOpError("block size must be set at lowering");
  IntegerAttr gridSize = op.getGridSizeAttr();
  if (!gridSize)
    return op.emitOpError("grid size must be set at lowering");

  if (isAccel) {
    rw.create<GridwiseGemmAccelOp>(
        loc, a, b, c, op.getArchAttr(), op.getFeaturesAttr(),
        op.getStoreMethodAttr(), blockSize, gridSize,
        params.cast<RockAccelTuningParamAttrInterface>());
    rw.eraseOp(op);
  } else {
    rw.create<GridwiseGemmOp>(loc, a, b, c, op.getFeaturesAttr(), gridSize,
                              params.cast<GeneralGemmParamsAttr>());
    rw.eraseOp(op);
  }
  return success();
}

void RockGemmToGridwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::GemmOp>();
  target.addLegalOp<rock::TransformOp, rock::GridwiseGemmOp,
                    rock::GridwiseGemmAccelOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<GemmRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
