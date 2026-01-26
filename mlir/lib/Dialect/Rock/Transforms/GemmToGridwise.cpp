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
#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/LogicalResult.h"
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
  // Custom constructor taking an additional argument: bufferDeps
  GemmRewritePattern(MLIRContext *context,
                     const BufferDependencyAnalysis &bufferDeps)
      : OpConversionPattern<GemmOp>(context), bufferDeps(bufferDeps) {}

  struct SplitKTransformedOperands {
    Value a;
    Value b;
    Value c;
    Value scaleA;
    Value scaleB;
  };

  LogicalResult matchAndRewrite(GemmOp op, GemmOpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override;

  LogicalResult computeGridSize(ConversionPatternRewriter &rw, GemmOp op,
                                Value a, Value b) const;

  FailureOr<SplitKTransformedOperands>
  arrangeSplitKTransform(OpBuilder &builder, GemmOp op, Location loc,
                         int64_t splitKFactor, Value a, Value b, Value c,
                         Value scaleA, Value scaleB) const;

  const BufferDependencyAnalysis &bufferDeps;
};

struct GemmElementwiseGemmRewritePattern
    : public OpConversionPattern<GemmElementwiseGemmOp> {
  using OpConversionPattern<GemmElementwiseGemmOp>::OpConversionPattern;
  // Custom constructor taking an additional argument: bufferDeps
  GemmElementwiseGemmRewritePattern(MLIRContext *context,
                                    const BufferDependencyAnalysis &bufferDeps)
      : OpConversionPattern<GemmElementwiseGemmOp>(context),
        bufferDeps(bufferDeps) {}

  LogicalResult matchAndRewrite(GemmElementwiseGemmOp op,
                                GemmElementwiseGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override;
  const BufferDependencyAnalysis &bufferDeps;
};

struct AttentionRewritePattern : public OpConversionPattern<AttentionOp> {
  using OpConversionPattern<AttentionOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AttentionOp op, AttentionOpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override;
};

// Move num_heads dimension to sequence length dimension. This is useful for the
// decoding phase, when batch=1, seq_len_q = 1 and GQA (example: num_heads_q=64,
// num_heads_kv=8), we can move numRepeat=num_heads_q/num_heads_kv = 8, to the
// seq_len_q dimension and use the tile size better (otherwise seq_len_q=1 and
// it will get padded to 32). This reduces the amount of workgroups by
// numRepeat. However, typically decoding phase will use split_kv anyway to
// increase the number of workgroups.
static Value moveNumHeadsToSeqLenQ(OpBuilder builder, Location loc,
                                   Value inputTensor, int64_t numRepeats) {
  ArrayRef<int64_t> inpShape =
      cast<ShapedType>(inputTensor.getType()).getShape();

  assert(inpShape.size() == 3 && "input must be 3D");
  assert(inpShape[0] % numRepeats == 0 &&
         "gemmG must be divisible by numRepeats");

  int64_t newGemmG = inpShape[0] / numRepeats;
  SmallVector<StringRef> startNames = {"gemmG", "headDim", "seqLen"};

  // (gemmG, headDim, seqLen) -> (gemmG / numRepeats, headDim, seqLen,
  // numRepeats)
  rock::BottomUpTMBuilder unmerge(builder, startNames, inpShape);
  unmerge.unmerge({"gemmG", "numRepeats"}, {0, 3}, "gemmG",
                  {newGemmG, numRepeats});
  unmerge.passThrough({"seqLen", "headDim"}, {2, 1}, {"seqLen", "headDim"});
  auto unmergeAttr = unmerge.get();
  Value matrixUnmerge =
      rock::TransformOp::create(builder, loc, inputTensor, unmergeAttr);

  // (gemmG / numRepeats, headDim, seqLen, numRepeats) -> (gemmG / numRepeats,
  // headDim, seqLen * numRepeats)
  auto merger = rock::BottomUpTMBuilder::above(unmerge, unmergeAttr);
  merger.merge("seqLen", 2, {"seqLen", "numRepeats"});
  merger.passThrough(ArrayRef<uint32_t>{0, 1}, ArrayRef<uint32_t>{0, 1});
  auto mergerAttr = merger.get();
  return rock::TransformOp::create(builder, loc, matrixUnmerge, mergerAttr);
}

// Same as moveNumHeadsToSeqLenQ() but for currSeqLen tensor (KV-Cache)
static Value moveNumHeadsToSeqLenCurrSeqLen(OpBuilder builder, Location loc,
                                            Value inputTensor,
                                            int64_t numRepeats) {
  ArrayRef<int64_t> inpShape =
      cast<ShapedType>(inputTensor.getType()).getShape();

  assert(inpShape.size() == 1 && "input must be 1D");
  assert(inpShape[0] % numRepeats == 0 &&
         "gemmG must be divisible by numRepeats");

  int64_t newGemmG = inpShape[0] / numRepeats;
  SmallVector<StringRef> startNames = {"gemmG"};

  // (gemmG) -> (gemmG / numRepeats, numRepeats)
  rock::BottomUpTMBuilder unmerge(builder, startNames, inpShape);
  unmerge.unmerge({"gemmG", "numRepeats"}, {0, 1}, "gemmG",
                  {newGemmG, numRepeats});
  auto unmergeAttr = unmerge.get();
  Value matrixUnmerge =
      rock::TransformOp::create(builder, loc, inputTensor, unmergeAttr);

  // slice numRepeats to 1
  auto slicer = rock::BottomUpTMBuilder::above(unmerge, unmergeAttr);
  slicer.slice({"numRepeats"}, {"numRepeats"}, {0}, {1});
  slicer.passThrough(ArrayRef<uint32_t>{0}, ArrayRef<uint32_t>{0});
  auto slicerAttr = slicer.get();
  Value matrixSliced =
      rock::TransformOp::create(builder, loc, matrixUnmerge, slicerAttr);

  // (gemmG / numRepeats, headDim, seqLen, numRepeats) -> (gemmG / numRepeats,
  // headDim, seqLen * numRepeats)
  auto merger = rock::BottomUpTMBuilder::above(slicer, slicerAttr);
  merger.merge("seqLen", 0, {"gemmG", "numRepeats"});
  auto mergerAttr = merger.get();
  return rock::TransformOp::create(builder, loc, matrixSliced, mergerAttr);
}

// Same as moveNumHeadsToSeqLenQ() but for the output tensor
static Value moveNumHeadsToSeqLenOut(OpBuilder builder, Location loc,
                                     Value inputTensor, int64_t numRepeats,
                                     int64_t splitKV) {
  ArrayRef<int64_t> inpShape =
      cast<ShapedType>(inputTensor.getType()).getShape();

  assert((inpShape.size() == 2 || inpShape.size() == 3) &&
         "input must be 2D or 3D");
  assert(inpShape[0] % numRepeats == 0 &&
         "gemmG must be divisible by numRepeats");
  assert(inpShape[0] % splitKV == 0 && "gemmG must be divisible by numRepeats");

  int64_t newGemmG = inpShape[0] / (numRepeats * splitKV);
  bool isLSE = inpShape.size() == 2;

  SmallVector<StringRef> startNamesAll = {"gemmG", "seqLen", "headDim"};
  ArrayRef<StringRef> startNames =
      ArrayRef<StringRef>(startNamesAll).take_front(inpShape.size());

  // Note that for LSE, there are only two dimensions (gemmG, seqLen)
  // (gemmG, seqLen, headDim) -> (gemmG / (splitKV*numRepeats), splitKV, seqLen,
  // numRepeats, headDim)
  rock::BottomUpTMBuilder unmerge(builder, startNames, inpShape);
  unmerge.unmerge({"gemmG", "numRepeats", "splitKV"}, {0, 3, 1}, "gemmG",
                  {newGemmG, numRepeats, splitKV});
  if (isLSE)
    unmerge.passThrough({"seqLen"}, {2}, {"seqLen"});
  else
    unmerge.passThrough({"seqLen", "headDim"}, {2, 4}, {"seqLen", "headDim"});
  auto unmergeAttr = unmerge.get();
  Value matrixUnmerge =
      rock::TransformOp::create(builder, loc, inputTensor, unmergeAttr);

  // (gemmG / (splitKV*numRepeats), splitKV, seqLen, numRepeats, headDim) ->
  // (gemmG / numRepeats, seqLen * numRepeats, headDim)
  auto merger = rock::BottomUpTMBuilder::above(unmerge, unmergeAttr);
  merger.merge("seqLen", 1, {"seqLen", "numRepeats"});
  merger.merge("gemmG", 0, {"gemmG", "splitKV"});
  if (!isLSE)
    merger.passThrough({"headDim"}, {2}, {"headDim"});
  auto mergerAttr = merger.get();
  return rock::TransformOp::create(builder, loc, matrixUnmerge, mergerAttr);
}

// Result of GQA (Grouped-Query Attention) processing
struct GQAResult {
  IntegerAttr numRepeats;
  Value queries;
  Value keys;
  Value values;
  Value out;
  Value lse;
  Value currentSeqLen;
  Value prefixOffset;
};

// This function will implement GQA, moving numRepeat=num_heads_q/num_heads_kv
// to the seq_len_q dimension. See moveNumHeadsToSeqLenQ() comment for more
// details.
static GQAResult processGQA(ConversionPatternRewriter &rw, Location loc,
                            Value queries, Value keys, Value values, Value out,
                            Value lse, Value currentSeqLen, Value prefixOffset,
                            int64_t numHeadsQ, int64_t numHeadsKV,
                            int64_t splitKV) {
  assert(numHeadsQ % numHeadsKV == 0);
  IntegerAttr numRepeatsAttr = nullptr;

  if (numHeadsQ != numHeadsKV) {
    int64_t numRepeats = numHeadsQ / numHeadsKV;

    numRepeatsAttr = rw.getIndexAttr(numRepeats);
    queries = moveNumHeadsToSeqLenQ(rw, loc, queries, numRepeats);
    if (currentSeqLen)
      currentSeqLen =
          moveNumHeadsToSeqLenCurrSeqLen(rw, loc, currentSeqLen, numRepeats);
    if (prefixOffset)
      prefixOffset =
          moveNumHeadsToSeqLenCurrSeqLen(rw, loc, prefixOffset, numRepeats);
    out = moveNumHeadsToSeqLenOut(rw, loc, out, numRepeats, splitKV);
    if (lse)
      lse = moveNumHeadsToSeqLenOut(rw, loc, lse, numRepeats, splitKV);
  }

  return GQAResult{numRepeatsAttr, queries,     keys, values, out, lse,
                   currentSeqLen,  prefixOffset};
}

template <typename Op>
static LogicalResult
computeGridSizeAttentionGemmElmtGemm(ConversionPatternRewriter &rw, Op op,
                                     Value a, Value b, Value c,
                                     int64_t splitKV) {
  GemmParamsAttr accelParams0 =
      cast<GemmParamsAttr>(op.getGemm0Params().value());

  SmallVector<int64_t, 3> aShape =
      llvm::to_vector<3>(cast<MemRefType>(a.getType()).getShape());

  SmallVector<int64_t, 3> bShape =
      llvm::to_vector<3>(cast<MemRefType>(b.getType()).getShape());

  SmallVector<int64_t, 3> cShape =
      llvm::to_vector<3>(cast<MemRefType>(c.getType()).getShape());

  GemmSize gemm0Size(/*g=*/aShape[0], /*m=*/bShape[2],
                     /*k=*/aShape[1],
                     /*n=*/aShape[2]);

  int64_t gridSize =
      (gemm0Size.n / accelParams0.getNPerBlock()) * gemm0Size.g * splitKV;

  IntegerAttr gridSizeAttr = rw.getI32IntegerAttr(gridSize);
  func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
  funcOp->setAttr(rock::GridSizeAttr::getMnemonic(), gridSizeAttr);
  return success();
}

static FailureOr<std::tuple<Value, Value, Value, Value>>
arrangeGemmGemmSplitKTransform(OpBuilder &builder,
                               RockGemmGemmWrapperInterface op, Location loc,
                               const BufferDependencyAnalysis &bufferDeps,
                               int64_t splitNFactor, Value a, Value b, Value c,
                               Value out) {
  // adjust the store method
  auto storeMethod =
      builder.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::AtomicAdd);
  op.setStoreMethodAttr(storeMethod);

  // set the prefill attribute
  auto func = llvm::cast<func::FuncOp>(op->getParentOp());
  FailureOr<SmallVector<BlockArgument>> args =
      traceGemmOutputToArgs(out, func, bufferDeps);
  if (failed(args)) {
    return op->emitError("can't trace gemm output to output argument");
  }

  auto attrName = rock::PrefillAttr::getMnemonic();
  for (auto arg : args.value()) {
    // initialize to zeros
    auto elementType = cast<ShapedType>(arg.getType()).getElementType();
    Attribute zero;
    if (llvm::isa<FloatType>(elementType)) {
      zero = builder.getFloatAttr(elementType, 0.0f);
    } else if (llvm::isa<IntegerType>(elementType)) {
      zero = builder.getIntegerAttr(elementType, 0);
    } else {
      return op->emitError("expecting `float` or `int` element type");
    }
    func.setArgAttrs(arg.getArgNumber(), builder.getNamedAttr(attrName, zero));
  }

  const int64_t origN = cast<RankedTensorType>(b.getType()).getShape()[2];
  const int64_t nPad =
      splitNFactor - math_util::mod_1_to_n(origN, splitNFactor);

  b = padMatrix(b, builder, loc, "gemmK", 0, "gemmN", nPad);
  c = padMatrix(c, builder, loc, "gemmK", nPad, "gemmO", 0);

  // perform coordinate transformations
  Value aNew{nullptr}, bNew{nullptr}, cNew{nullptr}, outNew{nullptr};
  ArrayRef<int64_t> aShape = cast<RankedTensorType>(a.getType()).getShape();
  ArrayRef<int64_t> bShape = cast<RankedTensorType>(b.getType()).getShape();
  ArrayRef<int64_t> cShape = cast<RankedTensorType>(c.getType()).getShape();
  ArrayRef<int64_t> outShape = cast<RankedTensorType>(out.getType()).getShape();

  const int64_t N = bShape[2];

  struct GemmOperandsData {
    Value &in;
    Value &out;
    SmallVector<StringRef> inputDimNames;
    unsigned presevedDimIdx;
    unsigned splitDimIdx;
    ArrayRef<int64_t> inputShape;
  };

  llvm::SmallVector<GemmOperandsData, 2> gemmOperands{
      {b, bNew, {"gemmG", "gemmK", "gemmN"}, 1, 2, bShape},
      {c, cNew, {"gemmG", "gemmN", "gemmO"}, 2, 1, cShape}};
  for (auto &gemmOperand : gemmOperands) {
    // Prepare matrix B and C - i.e.,
    //    (gemmG, gemmK, gemmN) and (gemmG, gemmN, gemmO), respectively
    // Using bottom-up transformations
    // 1. unmerge (gemmN) -> (gemmNSplit, gemmN*)
    // 2. merge (gemmG, gemmNSplit) -> (gemmG*)

    StringRef preservedDimName =
        gemmOperand.inputDimNames[gemmOperand.presevedDimIdx];
    StringRef splitDimName = gemmOperand.inputDimNames[gemmOperand.splitDimIdx];
    assert(splitDimName == "gemmN");

    BottomUpTMBuilder unmergeTransform(builder, gemmOperand.inputDimNames,
                                       gemmOperand.inputShape, loc);

    unmergeTransform.passThrough({"gemmG", preservedDimName}, {0, 3},
                                 {"gemmG", preservedDimName});
    unmergeTransform.unmerge({"gemmNSplit", "gemmN"}, {1, 2}, "gemmN",
                             {splitNFactor, N / splitNFactor});
    auto unmergeTransformAttr = unmergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(unmergeTransformAttr);

    auto mergeTransform =
        BottomUpTMBuilder::above(unmergeTransform, unmergeTransformAttr);

    mergeTransform.merge("gemmG", 0, {"gemmG", "gemmNSplit"});
    mergeTransform.passThrough(
        {"gemmN", preservedDimName},
        {gemmOperand.splitDimIdx, gemmOperand.presevedDimIdx},
        {"gemmN", preservedDimName});

    auto mergeTransformAttr = mergeTransform.get();
    transformAttrs.push_back(mergeTransformAttr);

    std::reverse(transformAttrs.begin(), transformAttrs.end());
    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    gemmOperand.out =
        mlir::rock::transform(builder, gemmOperand.in, arrayTransformAttrs);
  }

  {
    // Prepare matrix A - i.e., (gemmG, gemmK, gemmM)
    // Using bottom-up transformations
    // 1. addDim (gemmNSplit)
    // 2. merge (gemmG, gemmNSplit) -> (gemmG*)
    BottomUpTMBuilder addDimTransform(builder, {"gemmG", "gemmK", "gemmM"},
                                      aShape, loc);

    addDimTransform.passThrough({"gemmG", "gemmK", "gemmM"});
    addDimTransform.addDim("gemmNSplit", 3, splitNFactor);
    auto addDimTransformAttr = addDimTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(addDimTransformAttr);

    auto mergeTransform =
        BottomUpTMBuilder::above(addDimTransform, addDimTransformAttr);

    mergeTransform.merge("gemmG", 0, {"gemmG", "gemmNSplit"});
    mergeTransform.passThrough({"gemmK", "gemmM"});

    auto mergeTransformAttr = mergeTransform.get();
    transformAttrs.push_back(mergeTransformAttr);

    std::reverse(transformAttrs.begin(), transformAttrs.end());
    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    aNew = mlir::rock::transform(builder, a, arrayTransformAttrs);
  }

  {
    // Prepare matrix out - i.e., (gemmG, gemmM, gemmO)
    // Using top-down transformations
    // 1. merge (gemmG * gemmNSplit, gemmM, gemmO) -> (gemmG, gemmNSplit, gemmM,
    // gemmO)
    // 2. ignore (gemmG, gemmNSplit, gemmM, gemmN) -> (gemmG, gemmM, gemmO)

    const int64_t G = outShape[0];
    const int64_t M = outShape[1];
    const int64_t O = outShape[2];

    TopDownTMBuilder mergeTransform(builder, {"gemmG", "gemmM", "gemmO"},
                                    {G * splitNFactor, M, O});

    mergeTransform.merge({"gemmG", "gemmNSplit"}, {0, 1}, "gemmG",
                         {G, splitNFactor});
    mergeTransform.passThrough({"gemmM", "gemmO"}, {2, 3}, {"gemmM", "gemmO"});
    auto mergeTransformAttr = mergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(mergeTransformAttr);

    TopDownTMBuilder ignoreTransform =
        TopDownTMBuilder::below(mergeTransform, mergeTransformAttr);

    ignoreTransform.ignore("gemmNSplit");
    ignoreTransform.passThrough({"gemmG", "gemmM", "gemmO"}, {0, 1, 2},
                                {"gemmG", "gemmM", "gemmO"});

    TransformMapAttr ignoreTransformAttr = ignoreTransform.get();
    transformAttrs.push_back(ignoreTransformAttr);

    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    outNew = mlir::rock::transform(builder, out, arrayTransformAttrs);
  }
  return std::make_tuple(aNew, bNew, cNew, outNew);
}

static LogicalResult commonAttentionGemmElmtGemm(
    ConversionPatternRewriter &rw, RockGemmGemmWrapperInterface op, Value a,
    Value b, Value c, Value out, Value lse, Value currentSeqLen,
    Value prefixOffset, UnitAttr causal, IntegerAttr splitKV,
    ValueRange elementwiseInputs, Region &preSecondOpRegion, bool enableSoftmax,
    TypeAttr softmaxType, int64_t numHeadsQ, int64_t numHeadsKV,
    std::optional<std::reference_wrapper<const BufferDependencyAnalysis>>
        bufferDeps,
    BoolAttr preSoftmaxHasSplitKVTransforms) {
  Location loc = op->getLoc();

  if (!isa<MemRefType>(op.getAType()))
    return op.emitOpError("Cannot lower unbufferized gemm to gridwise");

  bool isAccel = rock::isAccel(rock::getFeatures(op));
  if (!isAccel) {
    return op.emitError("Currently, op is only supported on GPUs "
                        "with matrix accelerator extensions");
  }
  if (!op.getGemm0Params().has_value()) {
    return op.emitError("gemm0 params is missing and it should've been "
                        "assigned by affix-tuning-params");
  }
  GemmParamsAttr params0 = cast<GemmParamsAttr>(op.getGemm0Params().value());
  if (!op.getGemm1Params().has_value()) {
    return op.emitError("gemm1 params is missing and it should've been "
                        "assigned by affix-tuning-params");
  }
  GemmParamsAttr params1 = cast<GemmParamsAttr>(op.getGemm1Params().value());

  // Note: the gridwise ops take K x M and K x N, so A must be transposed if
  // it's in the natural M x K form
  a = normalizeMatrix(a, rw, loc, !op.getTransposedA(), "gemm0K", "gemm0M");
  b = normalizeMatrix(b, rw, loc, op.getTransposedB(), "gemm0K", "gemm0N");
  c = normalizeMatrix(c, rw, loc, op.getTransposedC(), "gemm1K", "gemm1N");
  out =
      normalizeMatrix(out, rw, loc, op.getTransposedOut(), "gemm1M", "gemm1N");

  const int64_t splitKFactor = params1.getSplitKFactor();
  if (splitKFactor > 1) {
    if (enableSoftmax)
      return op.emitError("split-k is not supported for attention");

    assert(bufferDeps.has_value() &&
           "buffer dependency analysis is required for split-k");

    auto maybeSplitk = arrangeGemmGemmSplitKTransform(
        rw, op, loc, bufferDeps.value(), splitKFactor, a, b, c, out);
    if (failed(maybeSplitk))
      return maybeSplitk;

    std::tie(a, b, c, out) = maybeSplitk.value();
  }

  int64_t splitKVNum = splitKV.getInt();

  // Grouped-Query Attention (GQA)
  IntegerAttr numRepeatsGQA = nullptr;
  if (enableSoftmax) {
    GQAResult gqa =
        processGQA(rw, op.getLoc(), a, b, c, out, lse, currentSeqLen,
                   prefixOffset, numHeadsQ, numHeadsKV, splitKVNum);
    numRepeatsGQA = gqa.numRepeats;
    a = gqa.queries;
    b = gqa.keys;
    c = gqa.values;
    out = gqa.out;
    lse = gqa.lse;
    currentSeqLen = gqa.currentSeqLen;
    prefixOffset = gqa.prefixOffset;
  }

  // Note, matrix dimension correctness is handled in the verifier
  ArrayRef<int64_t> aShape = cast<MemRefType>(a.getType()).getShape();
  ArrayRef<int64_t> bShape = cast<MemRefType>(b.getType()).getShape();
  ArrayRef<int64_t> cShape = cast<MemRefType>(c.getType()).getShape();
  assert(cShape[1] == bShape[2]);
  GemmSize gemm0Size(/*g=*/aShape[0], /*m=*/bShape[2],
                     /*k=*/aShape[1],
                     /*n=*/aShape[2]);
  GemmSize gemm1Size(/*g=*/aShape[0], /*m=*/cShape[2],
                     /*k=*/cShape[1],
                     /*n=*/aShape[2]);
  GemmSize gemm0ExtraPad = requiredPadding(params0, gemm0Size, 1, splitKVNum)
                               .value_or(GemmSize{0, 0, 0, 0});
  GemmSize gemm1ExtraPad = requiredPadding(params1, gemm1Size, splitKVNum)
                               .value_or(GemmSize{0, 0, 0, 0});

  a = padMatrix(a, rw, loc, "gemm0K", gemm0ExtraPad.k, "gemm0N",
                gemm0ExtraPad.n);
  b = padMatrix(b, rw, loc, "gemm0K", gemm0ExtraPad.k, "gemm0M",
                gemm0ExtraPad.m);
  c = padMatrix(c, rw, loc, "gemm1K", gemm1ExtraPad.k, "gemm1M",
                gemm1ExtraPad.m);
  // In the transposed layout, from a tuning params point of view
  // the output dimensions are swapped. Though we will only be
  // swapping them inside gridwise lowering to keep the surrounding
  // fusions legit. So the extra pad needs to be swapped and applied.
  out = padMatrix(out, rw, loc, "gemm1N", gemm1ExtraPad.n, "gemm1M",
                  gemm1ExtraPad.m);
  if (lse)
    lse = padVector(lse, rw, loc, "gemm1N", gemm1ExtraPad.n);

  if (failed(
          computeGridSizeAttentionGemmElmtGemm(rw, op, a, b, c, splitKVNum))) {
    return op.emitError("failed to compute the grid size of "
                        "`GemmElementwiseGemmOp`/`AttentionOp`");
  }

  func::FuncOp func = op->template getParentOfType<func::FuncOp>();
  IntegerAttr prePadG0MAttr;
  if (gemm0ExtraPad.m) {
    prePadG0MAttr = rw.getIndexAttr(gemm0Size.m);
  }
  IntegerAttr prePadG0NAttr;
  if (gemm0ExtraPad.n) {
    prePadG0NAttr = rw.getIndexAttr(gemm0Size.n);
  }

  auto newOp = GridwiseAttentionOp::create(
      rw, loc, a, b, c, elementwiseInputs, currentSeqLen, prefixOffset, out,
      lse, causal, splitKV, op.getGemmFeaturesAttr(), op.getStoreMethodAttr(),
      /*disableQBypassLDS=*/nullptr, prePadG0MAttr, prePadG0NAttr,
      numRepeatsGQA, softmaxType, params0, params1,
      rw.getDenseI64ArrayAttr(op.getFirstGemmIndices()),
      rw.getBoolAttr(enableSoftmax), preSoftmaxHasSplitKVTransforms);
  bool linalgOpFound = false;
  preSecondOpRegion.walk(
      [&](linalg::GenericOp genOp) { linalgOpFound = true; });
  if (linalgOpFound) {
    rw.inlineRegionBefore(preSecondOpRegion, newOp.getPreSoftmaxBody(),
                          newOp.getPreSoftmaxBody().begin());
  }
  rw.replaceOp(op, newOp);
  return success();
}

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

// getAccumulator returns a tensor for the accumulator
static Value getAccumulator(Value a, Value b, Value c, OpBuilder &builder,
                            Location loc) {
  auto aElementType = cast<ShapedType>(a.getType()).getElementType();
  auto bElementType = cast<ShapedType>(b.getType()).getElementType();
  auto cElementType = cast<ShapedType>(c.getType()).getElementType();

  auto accumulatorElementType = deduceAccumulatorElementType(
      aElementType, bElementType, cElementType, builder);

  if (accumulatorElementType != cElementType) {
    auto accumulatorShape = cast<ShapedType>(c.getType()).getShape();
    return tensor::EmptyOp::create(builder, loc, accumulatorShape,
                                   accumulatorElementType);
  }
  return c;
}
} // end namespace

LogicalResult
GemmRewritePattern::matchAndRewrite(GemmOp op, GemmOpAdaptor adaptor,
                                    ConversionPatternRewriter &rw) const {
  Location loc = op->getLoc();

  Attribute params = op.getParams().value_or(nullptr);
  if (!params) {
    return op.emitOpError("cannot lower gemm without tuning parameters");
  }

  Value a = adaptor.getA(), b = adaptor.getB();
  // Find the StoreOp that uses this GemmOp's result to get the transformed C
  Value c;
  rock::StoreOp storeOp = nullptr;
  for (Operation *user : op.getResult().getUsers()) {
    if (auto sop = dyn_cast<rock::StoreOp>(user)) {
      storeOp = sop;
      c = storeOp.getDest();
      break;
    }
  }
  if (!c) {
    return op.emitOpError("GemmOp result must be used by a StoreOp to provide "
                          "the output tensor");
  }
  Value scaleA = adaptor.getScaleA(), scaleB = adaptor.getScaleB();

  ShapedType typeA = cast<ShapedType>(a.getType());
  ShapedType typeB = cast<ShapedType>(b.getType());
  Type elemTypeA = typeA.getElementType();
  Type elemTypeB = typeB.getElementType();
  ArrayRef<int64_t> aShape = typeA.getShape();
  ArrayRef<int64_t> bShape = typeB.getShape();

  auto elemAWidth = elemTypeA.getIntOrFloatBitWidth();
  auto elemBWidth = elemTypeB.getIntOrFloatBitWidth();
  // Extend input types to the highest-precision type among the inputs
  if (elemTypeA != elemTypeB &&
      (!isa<FloatType>(elemTypeA) || !isa<FloatType>(elemTypeB) ||
       elemAWidth != 8 || elemBWidth != 8)) {
    if (elemTypeA.getIntOrFloatBitWidth() > elemTypeB.getIntOrFloatBitWidth()) {
      auto emptyB = tensor::EmptyOp::create(rw, loc, bShape, elemTypeA);
      auto newBType = RankedTensorType::get(bShape, elemTypeA);
      b = createTypeConversionLaGenericTensor(rw, loc, b, emptyB, newBType);
    } else {
      auto emptyA = tensor::EmptyOp::create(rw, loc, aShape, elemTypeB);
      auto newAType = RankedTensorType::get(aShape, elemTypeB);
      a = createTypeConversionLaGenericTensor(rw, loc, a, emptyA, newAType);
    }
  }
  ArrayRef<int64_t> scaleAShape, scaleBShape;

  // Note: the gridwise ops take M x K and K x N
  a = normalizeMatrix(a, rw, loc, op.getATransposed(), "gemmM", "gemmK");
  b = normalizeMatrix(b, rw, loc, op.getBTransposed(), "gemmK", "gemmN");
  // Result is always in [G, M, N] order (not transposed)
  c = normalizeMatrix(c, rw, loc, /*transpose=*/false, "gemmM", "gemmN");
  aShape = cast<ShapedType>(a.getType()).getShape();
  bShape = cast<ShapedType>(b.getType()).getShape();
  if (scaleA && scaleB) {
    auto scaleAType = cast<ShapedType>(scaleA.getType());
    auto scaleBType = cast<ShapedType>(scaleB.getType());
    scaleAShape = scaleAType.getShape();
    scaleBShape = scaleBType.getShape();
    // keep both scales in the same layout as the matrices
    // do transpose as necessary to achieve this. This is required to align load
    // and store layouts with matrices.
    bool transposeScaleA = (aShape != scaleAShape);
    scaleA =
        normalizeMatrix(scaleA, rw, loc, transposeScaleA, "gemmM", "gemmK");
    bool transposeScaleB = (bShape != scaleBShape);
    scaleB =
        normalizeMatrix(scaleB, rw, loc, transposeScaleB, "gemmK", "gemmN");
  }

  const int64_t splitKFactor = op.getParams()->getSplitKFactor();
  if (splitKFactor > 1) {
    auto maybeSplitk = arrangeSplitKTransform(rw, op, loc, splitKFactor, a, b,
                                              c, scaleA, scaleB);
    if (failed(maybeSplitk))
      return maybeSplitk;

    auto &transformed = maybeSplitk.value();
    a = transformed.a;
    b = transformed.b;
    c = transformed.c;
    scaleA = transformed.scaleA;
    scaleB = transformed.scaleB;
    aShape = cast<ShapedType>(a.getType()).getShape();
    bShape = cast<ShapedType>(b.getType()).getShape();
    if (scaleA && scaleB) {
      scaleAShape = cast<ShapedType>(scaleA.getType()).getShape();
      scaleBShape = cast<ShapedType>(scaleB.getType()).getShape();
    }
  }

  // Note, matrix dimension correctness is handled in the verifier
  GemmSize size(/*g=*/aShape[0], /*m=*/aShape[1], /*k=*/aShape[2],
                /*n=*/bShape[2]);

  GemmSize extraPad =
      requiredPadding(params, size).value_or(GemmSize{0, 0, 0, 0});

  a = padMatrix(a, rw, loc, "gemmM", extraPad.m, "gemmK", extraPad.k);
  b = padMatrix(b, rw, loc, "gemmK", extraPad.k, "gemmN", extraPad.n);
  c = padMatrix(c, rw, loc, "gemmM", extraPad.m, "gemmN", extraPad.n);
  if (scaleA && scaleB) {
    scaleA =
        padMatrix(scaleA, rw, loc, "gemmM", extraPad.m, "gemmK", extraPad.k);
    scaleB =
        padMatrix(scaleB, rw, loc, "gemmK", extraPad.k, "gemmN", extraPad.n);
    auto scaleAType = cast<ShapedType>(scaleA.getType());
    auto scaleBType = cast<ShapedType>(scaleB.getType());
    scaleAShape = scaleAType.getShape();
    scaleBShape = scaleBType.getShape();
    Type f8e8m0Type = rw.getF8E8M0Type();
    if (scaleAType.getElementType() != f8e8m0Type) {
      auto emptyScaleA =
          tensor::EmptyOp::create(rw, loc, scaleAShape, f8e8m0Type);
      auto newScaleAType = RankedTensorType::get(scaleAShape, f8e8m0Type);
      scaleA = createTypeConversionLaGenericTensor(rw, loc, scaleA, emptyScaleA,
                                                   newScaleAType);
    }
    if (scaleBType.getElementType() != f8e8m0Type) {
      auto emptyScaleB =
          tensor::EmptyOp::create(rw, loc, scaleBShape, f8e8m0Type);
      auto newScaleBType = RankedTensorType::get(scaleBShape, f8e8m0Type);
      scaleB = createTypeConversionLaGenericTensor(rw, loc, scaleB, emptyScaleB,
                                                   newScaleBType);
    }
  }

  if (failed(computeGridSize(rw, op, a, b))) {
    return op.emitError("failed to compute the grid size of `GemmOp`");
  }

  auto accumulator = getAccumulator(a, b, c, rw, loc);
  auto accumulatorType = cast<RankedTensorType>(accumulator.getType());
  auto gridwiseOp = GridwiseGemmOp::create(
      rw, loc, accumulatorType, a, b, scaleA, scaleB,
      op.getFeaturesAttr(), op.getStoreMethodAttr(),
      cast<GemmParamsAttr>(params));
  Value gridwiseResult = gridwiseOp.getResult();

  // If accumulator type differs from output type, convert
  Value result = gridwiseResult;
  auto resultTensorType = cast<RankedTensorType>(op.getResult().getType());
  if (accumulatorType.getElementType() != resultTensorType.getElementType()) {
    auto map = rw.getMultiDimIdentityMap(3);
    auto emptyResult = tensor::EmptyOp::create(
        rw, loc, resultTensorType.getShape(), resultTensorType.getElementType());
    auto genericOp = linalg::GenericOp::create(
        rw, loc, TypeRange{resultTensorType}, ValueRange{gridwiseResult},
        ValueRange{emptyResult}, ArrayRef<AffineMap>{map, map},
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                      utils::IteratorType::parallel,
                                      utils::IteratorType::parallel},
        /*doc=*/"", /*libraryCall=*/"",
        [&resultTensorType](OpBuilder &builder, Location loc,
                            ValueRange elems) {
          Value accumulator = elems[0];
          Type cType = resultTensorType.getElementType();
          if (isa<IntegerType>(cType)) {
            Value cElement =
                arith::TruncIOp::create(builder, loc, cType, accumulator);
            linalg::YieldOp::create(builder, loc, cElement);
          } else {
            Value cElement =
                arith::TruncFOp::create(builder, loc, cType, accumulator);
            linalg::YieldOp::create(builder, loc, cElement);
          }
        });
    result = genericOp.getResult(0);
  }

  // Update the StoreOp to use the new result type (which may be different)
  // TODO(roctriton): Sometimes we need this sometimes we dont...this should be done in a cleaner way.
  if (storeOp) {
    rw.setInsertionPoint(storeOp);
    auto newStoreOp = rock::StoreOp::create(
        rw, storeOp.getLoc(), storeOp.getResult().getType(), result,
        storeOp.getDest());
    rw.replaceOp(storeOp, newStoreOp.getResult());
  }

  rw.replaceOp(op, result);
  return success();
}

FailureOr<GemmRewritePattern::SplitKTransformedOperands>
GemmRewritePattern::arrangeSplitKTransform(OpBuilder &builder, GemmOp op,
                                           Location loc, int64_t splitKFactor,
                                           Value a, Value b, Value c,
                                           Value scaleA, Value scaleB) const {
  // adjust the store method
  auto storeMethod =
      builder.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::AtomicAdd);
  op.setStoreMethodAttr(storeMethod);

  // set the prefill attribute
  // For backward data convolution with multiple V4R1 kernels, only the first
  // kernel (kernelId == 0) should set the prefill attribute. All kernels write
  // to disjoint regions of the same output buffer, so only one initialization
  // is needed.
  bool shouldSetPrefill = true;
  // Only Gemms that have been converted from conv_bwd_data ops will have the
  // kernelId attribute.
  if (auto kernelIdAttr = op->getAttrOfType<IntegerAttr>("kernelId")) {
    if (kernelIdAttr.getInt() > 0) {
      shouldSetPrefill = false;
    }
  }

  if (shouldSetPrefill) {
    // Use the result since GemmOp no longer has C as input
    Value matC = op.getResult();
    auto func = llvm::cast<func::FuncOp>(op->getParentOp());
    FailureOr<SmallVector<BlockArgument>> args =
        traceGemmOutputToArgs(matC, func, bufferDeps);
    if (failed(args)) {
      return op->emitError("can't trace gemm output to output argument");
    }

    auto attrName = rock::PrefillAttr::getMnemonic();
    for (auto arg : args.value()) {
      // initialize to zeros
      auto elementType = cast<ShapedType>(arg.getType()).getElementType();
      Attribute zero;
      if (llvm::isa<FloatType>(elementType)) {
        zero = builder.getFloatAttr(elementType, 0.0f);
      } else if (llvm::isa<IntegerType>(elementType)) {
        zero = builder.getIntegerAttr(elementType, 0);
      } else {
        return op->emitError("expecting `float` or `int` element type");
      }
      func.setArgAttrs(arg.getArgNumber(),
                       builder.getNamedAttr(attrName, zero));
    }
  }

  const int64_t origK = cast<RankedTensorType>(a.getType()).getShape()[2];
  int64_t kPad = 0;
  if (scaleA && scaleB) {
    // Hard code block size to 32 for now.
    // for the scaleGEMMs, split-K division needs to happen such that it doesn't
    // cut in the middle of the a block
    // TODO: Use AmdArchDbInfo to populate blockSize
    int64_t blockSize = 32;
    int64_t lcm = math_util::lcm(splitKFactor, blockSize);
    kPad = lcm - math_util::mod_1_to_n(origK, lcm);
  } else {
    kPad = splitKFactor - math_util::mod_1_to_n(origK, splitKFactor);
  }

  a = padMatrix(a, builder, loc, "gemmM", 0, "gemmK", kPad);
  b = padMatrix(b, builder, loc, "gemmK", kPad, "gemmN", 0);
  if (scaleA && scaleB) {
    scaleA = padMatrix(scaleA, builder, loc, "gemmM", 0, "gemmK", kPad);
    scaleB = padMatrix(scaleB, builder, loc, "gemmK", kPad, "gemmN", 0);
  }

  // perform coordinate transformations
  Value aNew{nullptr}, bNew{nullptr}, cNew{nullptr};
  Value scaleANew{nullptr}, scaleBNew{nullptr};
  ArrayRef<int64_t> aShape = cast<RankedTensorType>(a.getType()).getShape();
  ArrayRef<int64_t> bShape = cast<RankedTensorType>(b.getType()).getShape();
  ArrayRef<int64_t> cShape = cast<RankedTensorType>(c.getType()).getShape();

  const int64_t K = aShape[2];

  struct GemmOperandsData {
    Value &in;
    Value &out;
    SmallVector<StringRef> inputDimNames;
    ArrayRef<int64_t> inputShape;
    uint32_t nonKDim;
    uint32_t kDim;
    uint32_t newNonKDim;
  };

  llvm::SmallVector<GemmOperandsData, 4> gemmOperands{
      {a, aNew, {"gemmG", "gemmM", "gemmK"}, aShape, 1, 2, 1},
      {b, bNew, {"gemmG", "gemmK", "gemmN"}, bShape, 2, 1, 3}};
  if (scaleA && scaleB) {
    ArrayRef<int64_t> scaleAShape =
        cast<RankedTensorType>(scaleA.getType()).getShape();
    ArrayRef<int64_t> scaleBShape =
        cast<RankedTensorType>(scaleB.getType()).getShape();
    gemmOperands.push_back(
        {scaleA, scaleANew, {"gemmG", "gemmM", "gemmK"}, scaleAShape, 1, 2, 1});
    gemmOperands.push_back(
        {scaleB, scaleBNew, {"gemmG", "gemmK", "gemmN"}, scaleBShape, 2, 1, 3});
  }
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

    unmergeTransform.passThrough({"gemmG", preservedDimName},
                                 {0, gemmOperand.newNonKDim},
                                 {"gemmG", preservedDimName});
    unmergeTransform.unmerge({"gemmKSplit", "gemmK"},
                             {gemmOperand.kDim, gemmOperand.kDim + 1}, "gemmK",
                             {splitKFactor, K / splitKFactor});

    auto unmergeTransformAttr = unmergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(unmergeTransformAttr);

    auto mergeTransform =
        BottomUpTMBuilder::above(unmergeTransform, unmergeTransformAttr);

    mergeTransform.merge("gemmG", 0, {"gemmG", "gemmKSplit"});
    mergeTransform.passThrough({"gemmK", preservedDimName},
                               {gemmOperand.kDim, gemmOperand.nonKDim},
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

    TopDownTMBuilder mergeTransform(builder, {"gemmG", "gemmM", "gemmN"},
                                    {G * splitKFactor, M, N});

    mergeTransform.merge({"gemmG", "gemmKSplit"}, {0, 1}, "gemmG",
                         {G, splitKFactor});
    mergeTransform.passThrough({"gemmM", "gemmN"}, {2, 3}, {"gemmM", "gemmN"});
    auto mergeTransformAttr = mergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(mergeTransformAttr);

    TopDownTMBuilder ignoreTransform =
        TopDownTMBuilder::below(mergeTransform, mergeTransformAttr);

    ignoreTransform.ignore("gemmKSplit");
    ignoreTransform.passThrough({"gemmG", "gemmM", "gemmN"}, {0, 1, 2},
                                {"gemmG", "gemmM", "gemmN"});

    TransformMapAttr ignoreTransformAttr = ignoreTransform.get();
    transformAttrs.push_back(ignoreTransformAttr);

    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    cNew = mlir::rock::transform(builder, c, arrayTransformAttrs);
  }
  return SplitKTransformedOperands{aNew, bNew, cNew, scaleANew, scaleBNew};
}

LogicalResult GemmRewritePattern::computeGridSize(ConversionPatternRewriter &rw,
                                                  GemmOp op, Value a,
                                                  Value b) const {
  GemmFeatures features = rock::getFeatures(op);
  Attribute params = op.getParams().value();

  const auto aShape = cast<RankedTensorType>(a.getType()).getShape();
  const auto bShape = cast<RankedTensorType>(b.getType()).getShape();

  const int64_t G = aShape[0];
  const int64_t M = aShape[1];
  const int64_t N = bShape[2];

  auto mPerBlock{0};
  auto nPerBlock{0};

  if (isAccel(features)) {
    auto tuningParams = cast<GemmParamsAttr>(params);
    mPerBlock = tuningParams.getMPerBlock();
    nPerBlock = tuningParams.getNPerBlock();
  }
  const auto gridSize = (M / mPerBlock) * (N / nPerBlock) * G;
  assert(gridSize > 0);

  func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
  funcOp->setAttr(rock::GridSizeAttr::getMnemonic(),
                  rw.getI32IntegerAttr(gridSize));
  return success();
}

LogicalResult
AttentionRewritePattern::matchAndRewrite(AttentionOp op,
                                         AttentionOpAdaptor adaptor,
                                         ConversionPatternRewriter &rw) const {
  return commonAttentionGemmElmtGemm(
      rw, op, adaptor.getQueries(), adaptor.getKeys(), adaptor.getValues(),
      adaptor.getOut(), adaptor.getLse(), adaptor.getCurrentSeqLen(),
      adaptor.getPrefixOffset(), adaptor.getCausalAttr(),
      adaptor.getSplitKVAttr(), adaptor.getPreSoftmaxElemWiseInputs(),
      op.getPreSoftmaxBody(),
      /*enableSoftmax=*/true, op.getSoftmaxTypeAttr(), adaptor.getNumHeadsQ(),
      adaptor.getNumHeadsKV(),
      /*bufferDeps=*/std::nullopt,
      adaptor.getPreSoftmaxHasSplitKVTransformsAttr());
}

LogicalResult GemmElementwiseGemmRewritePattern::matchAndRewrite(
    GemmElementwiseGemmOp op, GemmElementwiseGemmOpAdaptor adaptor,
    ConversionPatternRewriter &rw) const {
  auto splitKV = rw.getI32IntegerAttr(1);
  return commonAttentionGemmElmtGemm(
      rw, op, adaptor.getA(), adaptor.getB(), adaptor.getC(), adaptor.getOut(),
      /*lse=*/nullptr,
      /*currentSeqLen=*/nullptr, /*prefixOffset=*/nullptr, /*causal=*/nullptr,
      splitKV, adaptor.getElemwiseInputs(), op.getPreSecondGemmBody(),
      /*enableSoftmax=*/false, /*softmaxType=*/nullptr, /*numHeadsQ=*/1,
      /*numHeadsKV=*/1, std::cref(bufferDeps),
      /*preSoftmaxHasSplitKVTransforms=*/rw.getBoolAttr(false));
}

void RockGemmToGridwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::GemmOp, rock::AttentionOp,
                      rock::GemmElementwiseGemmOp>();
  target.addLegalOp<rock::TransformOp, rock::GridwiseGemmOp,
                    rock::StoreOp,
                    rock::GridwiseAttentionOp, memref::AllocOp,
                    linalg::GenericOp, arith::TruncIOp, arith::ExtFOp,
                    arith::ExtSIOp, arith::TruncFOp>();

  target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect>();

  BufferDependencyAnalysis &bufferDeps =
      getAnalysis<BufferDependencyAnalysis>();

  RewritePatternSet patterns(ctx);
  patterns.add<GemmRewritePattern, GemmElementwiseGemmRewritePattern>(
      ctx, bufferDeps);
  patterns.add<AttentionRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
} // namespace
