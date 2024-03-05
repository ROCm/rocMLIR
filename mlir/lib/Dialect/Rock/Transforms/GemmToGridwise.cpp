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
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <memory>
#include <sstream>

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

  LogicalResult computeGridSize(ConversionPatternRewriter &rw, GemmOp op,
                                Value a, Value b) const;
};

struct AttentionRewritePattern : public OpConversionPattern<AttentionOp> {
  using OpConversionPattern<AttentionOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AttentionOp op, AttentionOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override;

  LogicalResult computeGridSize(ConversionPatternRewriter &rw, AttentionOp op,
                                Value queries, Value keys, Value values) const;
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

  MemRefType typeA = a.getType().cast<MemRefType>();
  MemRefType typeB = b.getType().cast<MemRefType>();
  Type elemTypeA = typeA.getElementType();
  Type elemTypeB = typeB.getElementType();
  ArrayRef<int64_t> aShape = typeA.getShape();
  ArrayRef<int64_t> bShape = typeB.getShape();

  // Extend input types to the highest-precision type among the inputs
  if (elemTypeA != elemTypeB &&
      !(elemTypeA.isFloat8E5M2FNUZ() && elemTypeB.isFloat8E4M3FNUZ()) &&
      !(elemTypeA.isFloat8E4M3FNUZ() && elemTypeB.isFloat8E5M2FNUZ())) {
    if (elemTypeA.getIntOrFloatBitWidth() > elemTypeB.getIntOrFloatBitWidth()) {
      MemRefType newBType = MemRefType::get(bShape, elemTypeA);
      memref::AllocOp newB = rw.create<memref::AllocOp>(loc, newBType);
      createTypeConversionLaGeneric(rw, loc, b, newB);
      b = newB;
    } else {
      MemRefType newAType = MemRefType::get(aShape, elemTypeB);
      memref::AllocOp newA = rw.create<memref::AllocOp>(loc, newAType);
      createTypeConversionLaGeneric(rw, loc, a, newA);
      a = newA;
    }
  }

  // Note: the gridwise ops take K x M and K x N, so A must be transposed if
  // it's in the natural M x K form
  a = normalizeMatrix(a, rw, loc, !op.getATransposed(), "gemmK", "gemmM");
  b = normalizeMatrix(b, rw, loc, op.getBTransposed(), "gemmK", "gemmN");
  c = normalizeMatrix(c, rw, loc, op.getCTransposed(), "gemmM", "gemmN");

  aShape = a.getType().cast<MemRefType>().getShape();
  bShape = b.getType().cast<MemRefType>().getShape();

  // Note, matrix dimension correctness is handled in the verifier
  GemmSize size(/*g=*/aShape[0], /*m=*/aShape[2], /*k=*/aShape[1],
                /*n=*/bShape[2]);
  GemmSize extraPad =
      requiredPadding(params, size).value_or(GemmSize{0, 0, 0, 0});

  a = padMatrix(a, rw, loc, "gemmK", extraPad.k, "gemmM", extraPad.m);
  b = padMatrix(b, rw, loc, "gemmK", extraPad.k, "gemmN", extraPad.n);
  c = padMatrix(c, rw, loc, "gemmM", extraPad.m, "gemmN", extraPad.n);

  if (failed(computeGridSize(rw, op, a, b))) {
    return op.emitError("failed to compute the grid size of `GemmOp`");
  }

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

LogicalResult GemmRewritePattern::computeGridSize(ConversionPatternRewriter &rw,
                                                  GemmOp op, Value a,
                                                  Value b) const {
  GemmFeatures features = op.getGemmFeatures();
  Attribute params = op.getParams().value();

  const auto aShape = a.getType().cast<MemRefType>().getShape();
  const auto bShape = b.getType().cast<MemRefType>().getShape();

  const int64_t G = aShape[0];
  const int64_t M = aShape[2];
  const int64_t N = bShape[2];

  auto mPerBlock{0};
  auto nPerBlock{0};

  if (isAccel(features)) {
    auto tuningParams = params.cast<RockAccelTuningParamAttrInterface>();
    mPerBlock = tuningParams.getMPerBlock();
    nPerBlock = tuningParams.getNPerBlock();
  } else {
    auto tuningParams = params.cast<GeneralGemmParamsAttr>();
    mPerBlock = tuningParams.getMPerBlock();
    nPerBlock = tuningParams.getNPerBlock();
  }
  const auto gridSize = (M / mPerBlock) * (N / nPerBlock) * G;

  op.setGridSizeAttr(rw.getI32IntegerAttr(gridSize));

  func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
  funcOp->setAttr("grid_size", rw.getI32IntegerAttr(gridSize));
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
  if (!op.getParams0().has_value()) {
    return op.emitError("gemm0 params is missing and it should've been "
                        "assigned by affix-tuing-params");
  }
  RockAccelTuningParamAttrInterface params0 =
      op.getParams0Attr().cast<RockAccelTuningParamAttrInterface>();
  if (!op.getParams1().has_value()) {
    return op.emitError("gemm1 params is missing and it should've been "
                        "assigned by affix-tuing-params");
  }
  RockAccelTuningParamAttrInterface params1 =
      op.getParams1Attr().cast<RockAccelTuningParamAttrInterface>();

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
  GemmSize gemm0Size(/*g=*/queriesShape[0], /*m=*/keysShape[2],
                     /*k=*/queriesShape[1],
                     /*n=*/queriesShape[2]);
  GemmSize gemm0ExtraPad =
      requiredPadding(params0, gemm0Size).value_or(GemmSize{0, 0, 0, 0});
  GemmSize gemm1Size(/*g=*/queriesShape[0], /*m=*/valuesShape[2],
                     /*k=*/valuesShape[1],
                     /*n=*/queriesShape[2]);
  GemmSize gemm1ExtraPad =
      requiredPadding(params1, gemm1Size).value_or(GemmSize{0, 0, 0, 0});

  queries = padMatrix(queries, rw, loc, "gemm0K", gemm0ExtraPad.k, "gemm0N",
                      gemm0ExtraPad.n);
  keys = padMatrix(keys, rw, loc, "gemm0K", gemm0ExtraPad.k, "gemm0M",
                   gemm0ExtraPad.m);
  values = padMatrix(values, rw, loc, "gemm1K", gemm1ExtraPad.k, "gemm1M",
                     gemm1ExtraPad.m);
  // In the transposed layout, from a tuning params point of view
  // the output dimensions are swapped. Though we will only be
  // swapping them inside gridwise lowering to keep the surrounding
  // fusions legit. So the extra pad needs to be swapped and applied.
  out = padMatrix(out, rw, loc, "gemm1N", gemm1ExtraPad.n, "gemm1M",
                  gemm1ExtraPad.m);

  if (failed(computeGridSize(rw, op, queries, keys, values))) {
    return op.emitError("failed to compute the grid size of `AttentionOp`");
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
  auto newOp = rw.create<GridwiseAttentionAccelOp>(
      loc, queries, keys, values, adaptor.getPreSoftmaxElemWiseInputs(), out,
      op.getArchAttr(), op.getFeaturesAttr(), blockSizeAttr, gridSizeAttr,
      /*disableQBypassLDS=*/nullptr, prePadG0MAttr, prePadG0NAttr, params0,
      params1);
  bool linalgOpFound = false;
  op.getPreSoftmaxBody().walk(
      [&](linalg::GenericOp genOp) { linalgOpFound = true; });
  if (linalgOpFound) {
    rw.inlineRegionBefore(op.getPreSoftmaxBody(), newOp.getPreSoftmaxBody(),
                          newOp.getPreSoftmaxBody().begin());
  }
  rw.replaceOp(op, newOp);
  return success();
}

LogicalResult
AttentionRewritePattern::computeGridSize(ConversionPatternRewriter &rw,
                                         AttentionOp op, Value queries,
                                         Value keys, Value values) const {

  RockAccelTuningParamAttrInterface accelParams0 =
      op.getParams0Attr().cast<RockAccelTuningParamAttrInterface>();

  RockAccelTuningParamAttrInterface accelParams1 =
      op.getParams1Attr().cast<RockAccelTuningParamAttrInterface>();

  SmallVector<int64_t, 3> queriesShape =
      llvm::to_vector<3>(queries.getType().cast<MemRefType>().getShape());

  SmallVector<int64_t, 3> keysShape =
      llvm::to_vector<3>(keys.getType().cast<MemRefType>().getShape());

  SmallVector<int64_t, 3> valuesShape =
      llvm::to_vector<3>(values.getType().cast<MemRefType>().getShape());

  GemmSize gemm0Size(/*g=*/queriesShape[0], /*m=*/keysShape[2],
                     /*k=*/queriesShape[1],
                     /*n=*/queriesShape[2]);
  GemmSize gemm1Size(/*g=*/queriesShape[0], /*m=*/valuesShape[2],
                     /*k=*/valuesShape[1],
                     /*n=*/queriesShape[2]);

  int64_t gridSize = ((gemm0Size.n) / accelParams0.getNPerBlock()) *
                     ((gemm1Size.m) / accelParams1.getMPerBlock()) *
                     gemm0Size.g;

  IntegerAttr gridSizeAttr = rw.getI32IntegerAttr(gridSize);
  func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
  funcOp->setAttr("grid_size", gridSizeAttr);
  return success();
}

void RockGemmToGridwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::GemmOp, rock::AttentionOp>();
  target.addLegalOp<rock::TransformOp, rock::GridwiseGemmOp,
                    rock::GridwiseGemmAccelOp, rock::GridwiseAttentionAccelOp,
                    memref::AllocOp, linalg::GenericOp, arith::TruncIOp,
                    arith::ExtFOp, arith::ExtSIOp, arith::TruncFOp>();

  target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GemmRewritePattern, AttentionRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
