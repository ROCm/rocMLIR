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
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
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

  std::tuple<Value, Value, Value>
  arrangeSplitKTransform(OpBuilder &builder, GemmOp op, Location loc,
                         int64_t splitKFactor, Value a, Value b, Value c) const;
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
  if (isa<FloatType>(type) && type.getIntOrFloatBitWidth() < 32) {
    return builder.getF32Type();
  } else if (type.isInteger(8)) {
    return builder.getI32Type();
  }
  return elementTypeC;
}

static Value getAccumulator(Value a, Value b, Value c, OpBuilder &builder,
                            Location loc) {
  auto aElementType = cast<MemRefType>(a.getType()).getElementType();
  auto bElementType = cast<MemRefType>(b.getType()).getElementType();
  auto cElementType = cast<MemRefType>(c.getType()).getElementType();

  auto accumulatorElementType = deduceAccumulatorElementType(
      aElementType, bElementType, cElementType, builder);

  if (accumulatorElementType != cElementType) {
    auto accumulatorShape = cast<MemRefType>(c.getType()).getShape();
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
  if (!isa<MemRefType>(adaptor.getA().getType()))
    return op.emitOpError("Cannot lower unbufferized gemm to gridwise");

  Attribute params = op.getParams().value_or(nullptr);
  if (!params) {
    return op.emitOpError("cannot lower gemm without tuning parameters");
  }

  Value a = adaptor.getA(), b = adaptor.getB(), c = adaptor.getC();

  MemRefType typeA = cast<MemRefType>(a.getType());
  MemRefType typeB = cast<MemRefType>(b.getType());
  MemRefType typeC = cast<MemRefType>(c.getType());
  Type elemTypeA = typeA.getElementType();
  Type elemTypeB = typeB.getElementType();
  Type elemTypeC = typeC.getElementType();
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

  const int64_t splitKFactor = op.getParams()->getSplitKFactor();
  if (splitKFactor > 1) {
    const auto isAllowedTypeC =
        elemTypeC == rw.getF32Type() || elemTypeC == rw.getF16Type();

    if (!bitEnumContainsAll(op.getFeatures(), GemmFeatures::atomic_add)) {
      return op.emitError(
          "Split-K `GemmOp` requires support of `atomic_add` hardware feature");
    }

    if (!isAllowedTypeC) {
      return op.emitError(
          "Split-K `GemmOp` currently supports only f32/f16 element types");
    }
    std::tie(a, b, c) =
        arrangeSplitKTransform(rw, op, loc, splitKFactor, a, b, c);
  }

  aShape = cast<MemRefType>(a.getType()).getShape();
  bShape = cast<MemRefType>(b.getType()).getShape();

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
        cast<RockAccelTuningParamAttrInterface>(params));
  } else {
    rw.create<GridwiseGemmOp>(loc, a, b, accumulator, op.getFeaturesAttr(),
                              op.getStoreMethodAttr(), numCUAttr, gridSize,
                              cast<GeneralGemmParamsAttr>(params));
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
          if (isa<IntegerType>(cType)) {
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

std::tuple<Value, Value, Value>
GemmRewritePattern::arrangeSplitKTransform(OpBuilder &builder, GemmOp op,
                                           Location loc, int64_t splitKFactor,
                                           Value a, Value b, Value c) const {
  // adjust the store method
  auto storeMethod =
      builder.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::AtomicAdd);
  op.setStoreMethodAttr(storeMethod);

  // set the prefill attribute
  auto func = llvm::cast<func::FuncOp>(op->getParentOp());
  auto attrName = rock::PrefillAttr::getMnemonic();
  auto elementType = cast<MemRefType>(c.getType()).getElementType();
  Attribute zero;
  if (llvm::isa<FloatType>(elementType)) {
    zero = builder.getFloatAttr(elementType, 0.0);
  } else {
    assert(llvm::isa<IntegerType>(elementType) &&
           "expecting `int` element type");
    zero = builder.getIntegerAttr(elementType, 0);
  }
  func.setArgAttrs(2, builder.getNamedAttr(attrName, zero));

  const int64_t origK = cast<MemRefType>(a.getType()).getShape()[1];
  const int64_t kPad =
      splitKFactor - math_util::mod_1_to_n(origK, splitKFactor);

  a = padMatrix(a, builder, loc, "gemmK", kPad, "gemmM", 0);
  b = padMatrix(b, builder, loc, "gemmK", kPad, "gemmN", 0);

  // perform coordinate transformations
  Value aNew{nullptr}, bNew{nullptr}, cNew{nullptr};
  ArrayRef<int64_t> aShape = cast<MemRefType>(a.getType()).getShape();
  ArrayRef<int64_t> bShape = cast<MemRefType>(b.getType()).getShape();
  ArrayRef<int64_t> cShape = cast<MemRefType>(c.getType()).getShape();

  const int64_t K = aShape[1];

  struct GemmOperandsData {
    Value &in;
    Value &out;
    SmallVector<StringRef> inputDimNames;
    ArrayRef<int64_t> inputShape;
  };

  llvm::SmallVector<GemmOperandsData, 2> gemmOperands{
      {a, aNew, {"gemmG", "gemmK", "gemmM"}, aShape},
      {b, bNew, {"gemmG", "gemmK", "gemmN"}, bShape}};
  for (auto &gemmOperand : gemmOperands) {
    // Prepare matrix A and B - i.e.,
    //    (gemmG, gemmK, gemmM) and (gemmG, gemmK, gemmN), respectively
    // Using bottom-up transformations
    // 1. unmerge (gemmK) -> (gemmKSplit, gemmK*)
    // 2. merge (gemmG, gemmKSplit) -> (gemmG*)

    StringRef preservedDimName;
    for (auto &dimName : gemmOperand.inputDimNames) {
      if ((dimName != "gemmK") && (dimName != "gemmG"))
        preservedDimName = dimName;
    }

    BottomUpTMBuilder unmergeTransform(builder, gemmOperand.inputDimNames,
                                       gemmOperand.inputShape, loc);

    unmergeTransform.passThrough({"gemmG", preservedDimName}, {0, 3},
                                 {"gemmG", preservedDimName});
    unmergeTransform.unmerge({"gemmKSplit", "gemmK"}, {1, 2}, "gemmK",
                             {splitKFactor, K / splitKFactor});

    auto unmergeTransformAttr = unmergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(unmergeTransformAttr);

    auto mergeTransform =
        BottomUpTMBuilder::above(unmergeTransform, unmergeTransformAttr);

    mergeTransform.merge("gemmG", 0, {"gemmG", "gemmKSplit"});
    mergeTransform.passThrough({"gemmK", preservedDimName}, {1, 2},
                               {"gemmK", preservedDimName});

    auto mergeTransformAttr = mergeTransform.get();
    transformAttrs.push_back(mergeTransformAttr);

    std::reverse(transformAttrs.begin(), transformAttrs.end());
    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    gemmOperand.out =
        mlir::rock::transform(builder, gemmOperand.in, arrayTransformAttrs);
  }

  {
    // Prepare matrix C - i.e., (gemmG, gemmM, gemmN)
    // Using top-down transformations
    // 1. merge (gemmG * gemmKSplit, gemmM, gemmN) -> (gemmG, gemmKSplit, gemmM,
    // gemmN)
    // 2. ignore (gemmG, gemmKSplit, gemmM, gemmN) -> (gemmG, gemmM, gemmN)

    const int64_t G = cShape[0];
    const int64_t M = cShape[1];
    const int64_t N = cShape[2];

    TopDownTMBuilder mergenTransform(builder, {"gemmG", "gemmM", "gemmN"},
                                     {G * splitKFactor, M, N});

    mergenTransform.merge({"gemmG", "gemmKSplit"}, {0, 1}, "gemmG",
                          {G, splitKFactor});
    mergenTransform.passThrough({"gemmM", "gemmN"}, {2, 3}, {"gemmM", "gemmN"});
    auto mergenTransformAttr = mergenTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(mergenTransformAttr);

    TopDownTMBuilder ignoreTransform =
        TopDownTMBuilder::below(mergenTransform, mergenTransformAttr);

    ignoreTransform.ignore("gemmKSplit");
    ignoreTransform.passThrough({"gemmG", "gemmM", "gemmN"}, {0, 1, 2},
                                {"gemmG", "gemmM", "gemmN"});

    TransformMapAttr ignoreTransformAttr = ignoreTransform.get();
    transformAttrs.push_back(ignoreTransformAttr);

    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    cNew = mlir::rock::transform(builder, c, arrayTransformAttrs);
  }
  return std::make_tuple(aNew, bNew, cNew);
}

LogicalResult GemmRewritePattern::computeGridSize(ConversionPatternRewriter &rw,
                                                  GemmOp op, Value a,
                                                  Value b) const {
  GemmFeatures features = op.getGemmFeatures();
  Attribute params = op.getParams().value();

  const auto aShape = cast<MemRefType>(a.getType()).getShape();
  const auto bShape = cast<MemRefType>(b.getType()).getShape();

  const int64_t G = aShape[0];
  const int64_t M = aShape[2];
  const int64_t N = bShape[2];

  auto mPerBlock{0};
  auto nPerBlock{0};

  if (isAccel(features)) {
    auto tuningParams = cast<RockAccelTuningParamAttrInterface>(params);
    mPerBlock = tuningParams.getMPerBlock();
    nPerBlock = tuningParams.getNPerBlock();
  } else {
    auto tuningParams = cast<GeneralGemmParamsAttr>(params);
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

  if (!isa<MemRefType>(adaptor.getQueries().getType()))
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
      cast<RockAccelTuningParamAttrInterface>(op.getParams0Attr());
  if (!op.getParams1().has_value()) {
    return op.emitError("gemm1 params is missing and it should've been "
                        "assigned by affix-tuing-params");
  }
  RockAccelTuningParamAttrInterface params1 =
      cast<RockAccelTuningParamAttrInterface>(op.getParams1Attr());

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
      cast<MemRefType>(queries.getType()).getShape();
  ArrayRef<int64_t> keysShape = cast<MemRefType>(keys.getType()).getShape();
  ArrayRef<int64_t> valuesShape = cast<MemRefType>(values.getType()).getShape();
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
  IntegerAttr blockSizeAttr = cast<IntegerAttr>(func->getAttr("block_size"));
  IntegerAttr gridSizeAttr = cast<IntegerAttr>(func->getAttr("grid_size"));
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
      cast<RockAccelTuningParamAttrInterface>(op.getParams0Attr());

  SmallVector<int64_t, 3> queriesShape =
      llvm::to_vector<3>(cast<MemRefType>(queries.getType()).getShape());

  SmallVector<int64_t, 3> keysShape =
      llvm::to_vector<3>(cast<MemRefType>(keys.getType()).getShape());

  SmallVector<int64_t, 3> valuesShape =
      llvm::to_vector<3>(cast<MemRefType>(values.getType()).getShape());

  GemmSize gemm0Size(/*g=*/queriesShape[0], /*m=*/keysShape[2],
                     /*k=*/queriesShape[1],
                     /*n=*/queriesShape[2]);
  GemmSize gemm1Size(/*g=*/queriesShape[0], /*m=*/valuesShape[2],
                     /*k=*/valuesShape[1],
                     /*n=*/queriesShape[2]);

  int64_t gridSize =
      ((gemm0Size.n) / accelParams0.getNPerBlock()) * gemm0Size.g;

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
