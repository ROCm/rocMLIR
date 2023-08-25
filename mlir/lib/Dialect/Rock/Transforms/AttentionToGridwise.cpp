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
// This pass converts rock.attention into the appropriate
// rock.gridwise_attention adding padding and group dimensions if needed.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <memory>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKATTENTIONTOGRIDWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-attention-to-gridwise"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockAttentionToGridwisePass
    : public rock::impl::RockAttentionToGridwisePassBase<
          RockAttentionToGridwisePass> {
  void runOnOperation() override;
};

struct AttentionRewritePattern : public OpConversionPattern<AttentionOp> {
  using OpConversionPattern<AttentionOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AttentionOp op, AttentionOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override;
};
} // end namespace

static Attribute getTuningParams(ConversionPatternRewriter &rw,
                                 AttentionOp op) {
  Attribute params = op.getParams().value_or(nullptr);
  if (!params) {
    if (StringAttr perfConfigStrAttr =
            dyn_cast_or_null<StringAttr>(op->getAttr("perf_config"))) {
      InitParamsAccel accelParams;
      if (accelParams.deserialize(perfConfigStrAttr.str())) {
        GemmFeatures features = op.getFeatures();
        auto populateParamsAccelPtr = PopulateParamsAccel::select(features);
        params = populateParamsAccelPtr->getGemmParamsAttr(rw, accelParams);
        return params;
      }
    }
    // set a default one for now until the tuning flow is set up properly.
    params = rw.getAttr<XdlopsGemmParamsAttr>(
        /*kpackPerBlock=*/8, /*mPerBlock=*/32,
        /*nPerBlock=*/32, /*kpack=*/8,
        /*mPerWave=*/32, /*nPerWave=*/32, /*forceUnroll=*/true);
  }
  return params;
}

LogicalResult
AttentionRewritePattern::matchAndRewrite(AttentionOp op,
                                         AttentionOpAdaptor adaptor,
                                         ConversionPatternRewriter &rw) const {
  Location loc = op->getLoc();

  if (!adaptor.getQueries().getType().isa<MemRefType>())
    return op.emitOpError("Cannot lower unbufferized gemm to gridwise");

  bool isAccel = rock::isAccel(op.getFeatures());
  if (!isAccel) {
    return op.emitError("Currently, attention op is only supported on GPUs "
                        "with matrix accelerator extentions");
  }
  Attribute params = getTuningParams(rw, op);

  Value queries = adaptor.getQueries();
  Value keys = adaptor.getKeys();
  Value values = adaptor.getValues();
  Value out = adaptor.getOut();
  // Note: the gridwise ops take K x M and K x N, so A must be transposed if
  // it's in the natural M x K form
  queries = normalizeMatrix(queries, rw, loc, !op.getQTransposed(), "gemm0K",
                            "gemm0M");
  keys =
      normalizeMatrix(keys, rw, loc, op.getKTransposed(), "gemm0K", "gemm0N");
  values = normalizeMatrix(values, rw, loc, op.getVTransposed(), "gemm1K",
                           "gemm1N");
  out = normalizeMatrix(out, rw, loc, op.getOTransposed(), "gemm1M", "gemm1N");

  // Note, matrix dimension correctness is handled in the verifier
  ArrayRef<int64_t> queriesShape =
      queries.getType().cast<MemRefType>().getShape();
  ArrayRef<int64_t> keysShape = keys.getType().cast<MemRefType>().getShape();
  ArrayRef<int64_t> valuesShape =
      values.getType().cast<MemRefType>().getShape();
  GemmSize gemm0Size(/*g=*/queriesShape[0], /*m=*/queriesShape[2],
                     /*k=*/queriesShape[1],
                     /*n=*/keysShape[2]);
  GemmSize gemm0ExtraPad =
      requiredPadding(params, gemm0Size).value_or(GemmSize{0, 0, 0, 0});
  GemmSize gemm1Size(/*g=*/queriesShape[0], /*m=*/queriesShape[2],
                     /*k=*/valuesShape[1],
                     /*n=*/valuesShape[2]);
  GemmSize gemm1ExtraPad =
      requiredPadding(params, gemm1Size).value_or(GemmSize{0, 0, 0, 0});

  queries = padMatrix(queries, rw, loc, "gemm0K", gemm0ExtraPad.k, "gemm0M",
                      gemm0ExtraPad.m);
  keys = padMatrix(keys, rw, loc, "gemm0K", gemm0ExtraPad.k, "gemm0N",
                   gemm0ExtraPad.n);
  values = padMatrix(values, rw, loc, "gemm1K", gemm1ExtraPad.k, "gemm1N",
                     gemm1ExtraPad.n);
  out = padMatrix(out, rw, loc, "gemm1M", gemm1ExtraPad.m, "gemm1N",
                  gemm1ExtraPad.n);

  auto accelParams = params.cast<RockAccelTuningParamAttrInterface>();
  int64_t waveSize = rock::lookupArchInfo(op.getArchAttr()).waveSize;
  int64_t blockSize = waveSize * accelParams.getNPerBlock() *
                      accelParams.getMPerBlock() /
                      (accelParams.getMPerWave() * accelParams.getNPerWave());
  IntegerAttr blockSizeAttr = rw.getI32IntegerAttr(blockSize);
  int64_t gridSize = (gemm0Size.m / accelParams.getMPerBlock()) * gemm0Size.g;
  IntegerAttr gridSizeAttr = rw.getI32IntegerAttr(gridSize);

  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  func->setAttr("block_size", blockSizeAttr);
  func->setAttr("grid_size", gridSizeAttr);

  rw.create<GridwiseAttentionAccelOp>(
      loc, queries, keys, values,
      /*TODO(enable scale here once implemented)*/ nullptr, out,
      op.getArchAttr(), op.getFeaturesAttr(), blockSizeAttr, gridSizeAttr,
      accelParams);
  rw.eraseOp(op);
  return success();
}

void RockAttentionToGridwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::AttentionOp>();
  target.addLegalOp<rock::TransformOp, rock::GridwiseAttentionAccelOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<AttentionRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
