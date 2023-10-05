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
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
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

struct AttentionRewritePattern : public OpConversionPattern<AttentionOp> {
  using OpConversionPattern<AttentionOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AttentionOp op, AttentionOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override;
};

static Type getSmallestType(Type type1, Type type2) {
  return (type1.getIntOrFloatBitWidth() > type2.getIntOrFloatBitWidth())
             ? type2
             : type1;
}

static Type deduceAccumulatorElementType(Type elementTypeA, Type elementTypeB,
                                         Type elementTypeC,
                                         OpBuilder &builder) {
  // Determine the type used on VGPR to act as accumulator.
  // f32: f32.
  // f16, bf16: f32 to prevent overflow from happening.
  // i16 : i16.
  // fp8 (any combo) : f32.
  // i8: i32, since we have an i32 output
  auto type = getSmallestType(elementTypeA, elementTypeB);
  if (type.isa<FloatType>() && type.getIntOrFloatBitWidth() < 32) {
    return builder.getF32Type();
  } else if (type.isInteger(8)) {
    return builder.getI32Type();
  }
  return elementTypeC;
}

static Value getAccumulator(Value a, Value b, Value c, OpBuilder &builder,
                            Location loc) {
  auto aElementType = a.getType().cast<MemRefType>().getElementType();
  auto bElementType = b.getType().cast<MemRefType>().getElementType();
  auto cElementType = c.getType().cast<MemRefType>().getElementType();

  auto accumulatorElementType = deduceAccumulatorElementType(
      aElementType, bElementType, cElementType, builder);

  if (accumulatorElementType != cElementType) {
    auto accumulatorShape = c.getType().cast<MemRefType>().getShape();
    auto accumulatorType =
        MemRefType::get(accumulatorShape, accumulatorElementType);
    return builder.create<memref::AllocOp>(loc, accumulatorType);
  }
  return c;
}
} // end namespace

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
  IntegerAttr numCUAttr = op.getNumCUAttr();
  if (!numCUAttr) {
    int64_t minNumCU = rock::lookupArchInfo(op.getArchAttr()).minNumCU;
    numCUAttr = rw.getI32IntegerAttr(minNumCU);
  }

  bool isAccel = rock::isAccel(op.getFeatures());

  if (isAccel && !blockSize)
    return op.emitOpError("block size must be set at lowering");
  IntegerAttr gridSize = op.getGridSizeAttr();
  if (!gridSize)
    return op.emitOpError("grid size must be set at lowering");

  auto accumulator = getAccumulator(a, b, c, rw, loc);
  if (isAccel) {
    rw.create<GridwiseGemmAccelOp>(
        loc, a, b, accumulator, op.getArchAttr(), numCUAttr,
        op.getFeaturesAttr(), op.getStoreMethodAttr(), blockSize, gridSize,
        params.cast<RockAccelTuningParamAttrInterface>());
  } else {
    rw.create<GridwiseGemmOp>(loc, a, b, accumulator, op.getFeaturesAttr(),
                              numCUAttr, gridSize,
                              params.cast<GeneralGemmParamsAttr>());
  }

  if (accumulator != c) {
    auto map = rw.getMultiDimIdentityMap(3);
    rw.create<linalg::GenericOp>(
        loc, ValueRange{accumulator}, ValueRange{c},
        ArrayRef<AffineMap>{map, map},
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                      utils::IteratorType::parallel,
                                      utils::IteratorType::parallel},
        /*doc=*/"", /*library_call=*/"",
        [](OpBuilder &builder, Location loc, ValueRange elems) {
          Value accumulator = elems[0], c = elems[1];
          Type cType = c.getType();
          if (cType.isa<IntegerType>()) {
            Value cElement =
                builder.create<arith::TruncIOp>(loc, cType, accumulator);
            builder.create<linalg::YieldOp>(loc, cElement);
          } else {
            Value cElement =
                builder.create<arith::TruncFOp>(loc, cType, accumulator);
            builder.create<linalg::YieldOp>(loc, cElement);
          }
        });
  }
  rw.eraseOp(op);
  return success();
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
  RockAccelTuningParamAttrInterface params =
      op.getParamsAttr().cast<RockAccelTuningParamAttrInterface>();

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
  values =
      normalizeMatrix(values, rw, loc, op.getVTransposed(), "gemm1K", "gemm1N");
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

  Value scale = nullptr;
  if(Value scaleUnpadded = adaptor.getScale()){
    scale = padMatrix(scaleUnpadded, rw, loc, "gemm1M", gemm0ExtraPad.m, "gemm1N", gemm0ExtraPad.n);
  }
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  IntegerAttr blockSizeAttr = func->getAttr("block_size").cast<IntegerAttr>();
  IntegerAttr gridSizeAttr = func->getAttr("grid_size").cast<IntegerAttr>();
  IntegerAttr prePadG0MAttr;
  if (gemm0ExtraPad.m) {
    prePadG0MAttr = rw.getIndexAttr(gemm0Size.m);
  }
  IntegerAttr prePadG0NAttr;
  if (gemm0ExtraPad.n) {
    prePadG0NAttr = rw.getIndexAttr(gemm0Size.n);
  }
  rw.replaceOpWithNewOp<GridwiseAttentionAccelOp>(
      op, queries, keys, values,
      scale, out,
      op.getArchAttr(), op.getFeaturesAttr(), blockSizeAttr, gridSizeAttr,
      prePadG0MAttr, prePadG0NAttr, params);
  return success();
}

void RockGemmToGridwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::GemmOp, rock::AttentionOp>();
  target.addLegalOp<rock::TransformOp, rock::GridwiseGemmOp,
                    rock::GridwiseGemmAccelOp, rock::GridwiseAttentionAccelOp,
                    memref::AllocOp, linalg::GenericOp, arith::TruncIOp,
                    arith::TruncFOp>();

  target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GemmRewritePattern, AttentionRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
