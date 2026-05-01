//===- TosaOptionalDecompositions.cpp
//------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to apply the Tosa operations decompositions
// exposed as populate functions in
// include/mlir/Dialect/Tosa/Transforms/Passes.h
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXTRANSFORMPASS
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {

class QuantDotDecompose final : public OpRewritePattern<migraphx::QuantDotOp> {
public:
  using OpRewritePattern<migraphx::QuantDotOp>::OpRewritePattern;

private:
  // Helper function to convert a value to the target element type if needed
  Value convertToType(PatternRewriter &rewriter, Location loc, Value val,
                      Type targetElemType) const {
    auto shapedType = cast<MIXRShapedType>(val.getType());
    if (shapedType.getElementType() == targetElemType) {
      return val;
    }
    return migraphx::ConvertOp::create(
        rewriter, loc,
        MIXRShapedType::get(shapedType.getShape(), shapedType.getStrides(),
                            targetElemType),
        val);
  }

  // Helper function to apply scaling: converts input and scale to target type,
  // then multiplies them.
  // When the scale is not f8E8M0FNU, it is first roundtripped through
  // f8E8M0FNU (e.g. f32 -> f8E8M0FNU -> f32) so that the host decomposition
  // matches the kernel path, which casts scales to f8E8M0FNU block exponents
  // for tosa.matmul_t_block_scaled / rock.gemm.
  Value applyScale(PatternRewriter &rewriter, Location loc, Value input,
                   Value scale, Type targetElemType) const {
    Type scaleElemType = cast<MIXRShapedType>(scale.getType()).getElementType();
    Type f8E8M0Type = Float8E8M0FNUType::get(rewriter.getContext());
    if (scaleElemType != f8E8M0Type)
      scale = convertToType(rewriter, loc, scale, f8E8M0Type);
    Value convertedScale = convertToType(rewriter, loc, scale, targetElemType);
    Value convertedInput = convertToType(rewriter, loc, input, targetElemType);

    auto shapedType = cast<MIXRShapedType>(convertedInput.getType());
    return migraphx::MulOp::create(rewriter, loc,
                                   MIXRShapedType::get(shapedType.getShape(),
                                                       shapedType.getStrides(),
                                                       targetElemType),
                                   convertedInput, convertedScale);
  }

public:
  LogicalResult matchAndRewrite(migraphx::QuantDotOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op->getLoc();

    auto inA = op.getInA();
    auto inB = op.getInB();
    auto scaleA = op.getScaleA();
    auto scaleB = op.getScaleB();

    // Only decompose scaled GEMM operations (both scales must be present).
    // The verifier ensures both scales are provided together or neither.
    if (!scaleA && !scaleB) {
      return failure();
    }

    // Determine target output type
    auto resultType = op.getResult().getType();

    // For scaled operations, we always convert to F32 for proper computation
    Type computeElemType = rewriter.getF32Type();

    // Convert and scale inputs
    Value processedA = applyScale(rewriter, loc, inA, scaleA, computeElemType);
    Value processedB = applyScale(rewriter, loc, inB, scaleB, computeElemType);

    // Create the dot operation with processed inputs
    auto dotOp = migraphx::DotOp::create(rewriter, loc, resultType, processedA,
                                         processedB);
    if (auto attr = (*op).template getAttrOfType<StringAttr>("perf_config"))
      dotOp->setAttr("perf_config", attr);
    rewriter.replaceOp(op, dotOp->getResults()[0]);
    return success();
  }
};

static bool hasMIXRFeature(migraphx::AttentionOp op,
                           migraphx::AttentionFeatures flag) {
  return migraphx::hasAttentionFeature(op.getFeatures(), flag);
}

static MIXRShapedType makeContiguousType(ArrayRef<int64_t> shape,
                                         Type elemType) {
  SmallVector<int64_t, 4> strides(shape.size());
  int64_t s = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = s;
    s *= shape[i];
  }
  return MIXRShapedType::get(shape, strides, elemType);
}

/// Broadcast a 4D K or V tensor from [batch, numHeadsKV, D1, D2] to
/// [batch, numHeadsQ, D1, D2] for GQA by inserting a broadcast dimension
/// and reshaping: broadcast to [batch, numHeadsKV, repeat, D1, D2] then
/// reshape to [batch, numHeadsQ, D1, D2].
static Value broadcastForGQA(PatternRewriter &rewriter, Location loc, Value val,
                             int64_t numHeadsQ) {
  auto valType = cast<MIXRShapedType>(val.getType());
  ArrayRef<int64_t> shape = valType.getShape();
  int64_t numHeadsKV = shape[1];
  int64_t repeat = numHeadsQ / numHeadsKV;

  SmallVector<int64_t> bcShape = {shape[0], numHeadsKV, repeat, shape[2],
                                  shape[3]};
  SmallVector<int64_t> bcStrides = {
      valType.getStrides()[0], valType.getStrides()[1], 0,
      valType.getStrides()[2], valType.getStrides()[3]};
  auto bcType =
      MIXRShapedType::get(bcShape, bcStrides, valType.getElementType());
  Value bc = migraphx::MultiBroadcastOp::create(
      rewriter, loc, bcType, val, rewriter.getI64ArrayAttr(bcShape));

  SmallVector<int64_t> newShape = {shape[0], numHeadsQ, shape[2], shape[3]};
  return migraphx::ReshapeOp::create(
      rewriter, loc, makeContiguousType(newShape, valType.getElementType()), bc,
      rewriter.getI64ArrayAttr(newShape));
}

/// Returns the signed i32 integer type.
static IntegerType getSi32Type(MLIRContext *ctx) {
  return IntegerType::get(ctx, 32, IntegerType::Signed);
}

/// Creates a 1-D si32 literal with values [0, 1, ..., n-1].
/// Used to build row/column index tensors for causal and kvcache masks.
static Value createRangeIndices(PatternRewriter &rewriter, Location loc,
                                int64_t n) {
  SmallVector<int32_t> vals(n);
  std::iota(vals.begin(), vals.end(), 0);
  Type si32 = getSi32Type(rewriter.getContext());
  auto shapedTy = makeContiguousType({n}, si32);
  auto dense =
      DenseIntElementsAttr::get(RankedTensorType::get({n}, si32), vals);
  return migraphx::LiteralOp::create(rewriter, loc, shapedTy, dense);
}

/// Broadcasts a value to a target shape with specified strides.
/// Strides of 0 indicate broadcast dimensions.
static Value broadcastTo(PatternRewriter &rewriter, Location loc, Value val,
                         ArrayRef<int64_t> targetShape,
                         ArrayRef<int64_t> targetStrides) {
  auto vt = cast<MIXRShapedType>(val.getType());
  auto bt =
      MIXRShapedType::get(targetShape, targetStrides, vt.getElementType());
  return migraphx::MultiBroadcastOp::create(
      rewriter, loc, bt, val, rewriter.getI64ArrayAttr(targetShape));
}

/// Creates a scalar literal and broadcasts it to a target shape with all-zero
/// strides. Used to create broadcast -inf values for masking.
static Value createBroadcastScalar(PatternRewriter &rewriter, Location loc,
                                   ElementsAttr scalarAttr, Type elemTy,
                                   ArrayRef<int64_t> targetShape) {
  auto litTy = makeContiguousType({1}, elemTy);
  Value lit = migraphx::LiteralOp::create(rewriter, loc, litTy, scalarAttr);
  int64_t rank = targetShape.size();
  SmallVector<int64_t, 4> strides(rank, 0);
  auto bt = MIXRShapedType::get(targetShape, strides, elemTy);
  return migraphx::MultiBroadcastOp::create(
      rewriter, loc, bt, lit, rewriter.getI64ArrayAttr(targetShape));
}

/// Creates a -inf DenseElementsAttr for the given float element type.
static DenseElementsAttr getNegInfAttr(Type elemType) {
  auto floatTy = cast<FloatType>(elemType);
  return DenseElementsAttr::get(RankedTensorType::get({1}, elemType),
                                APFloat::getInf(floatTy.getFloatSemantics(),
                                                /*Negative=*/true));
}

/// Converts the element type of an MIXRShaped value if it differs from
/// newElemTy. Returns the value unchanged if types already match.
/// Used to convert i32 <-> si32 for index comparisons.
static Value convertMIXRElemType(PatternRewriter &rewriter, Location loc,
                                 Value val, Type newElemTy) {
  auto st = cast<MIXRShapedType>(val.getType());
  if (st.getElementType() == newElemTy)
    return val;
  auto dstTy = MIXRShapedType::get(st.getShape(), st.getStrides(), newElemTy);
  return migraphx::ConvertOp::create(rewriter, loc, dstTy, val);
}

/// Creates column index iota [0..seqK-1] broadcast to the QK shape.
static Value createBroadcastColIndices(PatternRewriter &rewriter, Location loc,
                                       ArrayRef<int64_t> qkShape) {
  int64_t rank = qkShape.size();
  Value colIota = createRangeIndices(rewriter, loc, qkShape[rank - 1]);
  SmallVector<int64_t, 8> bcStrides(rank, 0);
  bcStrides[rank - 1] = 1;
  return broadcastTo(rewriter, loc, colIota, qkShape, bcStrides);
}

/// Broadcasts an MIXRShaped operand (e.g. currentSeqLen, prefixOffset) to
/// the QK shape, preserving original strides in the leading dims.
static Value broadcastOperandToQKShape(PatternRewriter &rewriter, Location loc,
                                       Value operand,
                                       ArrayRef<int64_t> qkShape) {
  auto opType = cast<MIXRShapedType>(operand.getType());
  int64_t rank = qkShape.size();
  SmallVector<int64_t, 8> bcStrides(rank, 0);
  for (int64_t i = 0; i < opType.getRank(); ++i)
    bcStrides[i] = opType.getStrides()[i];
  return broadcastTo(rewriter, loc, operand, qkShape, bcStrides);
}

/// Applies a single mask to QK scores: computes greater(lhs, rhs), converts
/// to i8, and applies where(mask, -inf, qk). Non-zero mask values are
/// positions to be replaced with -inf (invalid positions).
static Value applyMask(PatternRewriter &rewriter, Location loc, Value qk,
                       Value lhs, Value rhs) {
  auto qkType = cast<MIXRShapedType>(qk.getType());
  ArrayRef<int64_t> qkShape = qkType.getShape();
  Type si32 = getSi32Type(rewriter.getContext());

  auto gtTy = makeContiguousType(qkShape, si32);
  Value gt = migraphx::Greater::create(rewriter, loc, gtTy, lhs, rhs);
  auto cvtI8Ty =
      MIXRShapedType::get(qkShape, gtTy.getStrides(), rewriter.getI8Type());
  Value mask = migraphx::ConvertOp::create(rewriter, loc, cvtI8Ty, gt);

  Type elemType = qkType.getElementType();
  Value bcNegInf = createBroadcastScalar(rewriter, loc, getNegInfAttr(elemType),
                                         elemType, qkShape);
  return migraphx::WhereOp::create(rewriter, loc, qkType, mask, bcNegInf, qk);
}

/// Causal mask: masks future positions where col > row (+ offset).
/// Computes greater(col, row + prefixOffset) and replaces those positions
/// with -inf. Matches the pattern from MIGraphX's expanded attention graph.
static Value applyCausalMask(PatternRewriter &rewriter, Location loc, Value qk,
                             Value prefixOffsetVal) {
  auto qkType = cast<MIXRShapedType>(qk.getType());
  ArrayRef<int64_t> qkShape = qkType.getShape();
  int64_t rank = qkType.getRank();
  int64_t seqQ = qkShape[rank - 2];
  Type si32 = getSi32Type(rewriter.getContext());

  Value bcCol = createBroadcastColIndices(rewriter, loc, qkShape);

  Value rowFlat = createRangeIndices(rewriter, loc, seqQ);
  auto row2Ty = makeContiguousType({seqQ, 1}, si32);
  Value rowIota2 = migraphx::ReshapeOp::create(
      rewriter, loc, row2Ty, rowFlat, rewriter.getI64ArrayAttr({seqQ, 1}));
  SmallVector<int64_t, 8> bcRowStrides(rank, 0);
  bcRowStrides[rank - 2] = 1;
  Value bcRow = broadcastTo(rewriter, loc, rowIota2, qkShape, bcRowStrides);

  if (prefixOffsetVal) {
    Value pref = convertMIXRElemType(rewriter, loc, prefixOffsetVal, si32);
    Value bcPref = broadcastOperandToQKShape(rewriter, loc, pref, qkShape);
    bcRow = migraphx::AddOp::create(
        rewriter, loc, makeContiguousType(qkShape, si32), bcRow, bcPref);
  }

  return applyMask(rewriter, loc, qk, bcCol, bcRow);
}

/// KV-cache mask: masks positions beyond currentSeqLen.
/// Computes greater(col, seqLen) and replaces those positions with -inf.
/// This matches the MIGraphX expanded graph pattern where positions with
/// col > seqLen are masked as invalid.
static Value applyKVCacheMask(PatternRewriter &rewriter, Location loc, Value qk,
                              Value currentSeqLen) {
  auto qkType = cast<MIXRShapedType>(qk.getType());
  ArrayRef<int64_t> qkShape = qkType.getShape();
  Type si32 = getSi32Type(rewriter.getContext());

  Value bcCol = createBroadcastColIndices(rewriter, loc, qkShape);
  Value bcSeqLen =
      broadcastOperandToQKShape(rewriter, loc, currentSeqLen, qkShape);
  bcSeqLen = convertMIXRElemType(rewriter, loc, bcSeqLen, si32);

  return applyMask(rewriter, loc, qk, bcCol, bcSeqLen);
}

/// Sliding window mask: masks positions outside the recent window.
/// Computes lowerBound = seqLen + (-windowSize), then masks positions
/// where greater(lowerBound, col), i.e. col < lowerBound (too old).
/// Matches the MIGraphX expanded graph pattern which uses
/// add(seqLen, -windowSize) as the lower bound.
static Value applySlidingWindowMask(PatternRewriter &rewriter, Location loc,
                                    Value qk, Value currentSeqLen,
                                    int32_t windowSize) {
  auto qkType = cast<MIXRShapedType>(qk.getType());
  ArrayRef<int64_t> qkShape = qkType.getShape();
  Type lenElemTy =
      cast<MIXRShapedType>(currentSeqLen.getType()).getElementType();
  Type si32 = getSi32Type(rewriter.getContext());

  Value bcSeqLen =
      broadcastOperandToQKShape(rewriter, loc, currentSeqLen, qkShape);

  auto intTy = cast<IntegerType>(lenElemTy);
  bool signedSemantics = intTy.isSigned() || intTy.isSignless();
  APInt negWindowAP(intTy.getWidth(), static_cast<uint64_t>(-windowSize),
                    signedSemantics);
  auto negWinDense = DenseElementsAttr::get(
      RankedTensorType::get({1}, lenElemTy), negWindowAP);
  Value bcNegWindow =
      createBroadcastScalar(rewriter, loc, negWinDense, lenElemTy, qkShape);

  Value lowerBound = migraphx::AddOp::create(
      rewriter, loc, makeContiguousType(qkShape, lenElemTy), bcSeqLen,
      bcNegWindow);
  Value lowerBoundI32 = convertMIXRElemType(rewriter, loc, lowerBound, si32);

  Value bcCol = createBroadcastColIndices(rewriter, loc, qkShape);
  return applyMask(rewriter, loc, qk, lowerBoundI32, bcCol);
}

/// Splits K's last dimension (seqK) into [splitKV, seqK/splitKV] and
/// transposes so the split dimension comes before hdQK:
/// K [B, hdQK, seqK] -> reshape [B, hdQK, S, seqK/S] -> transpose
/// [B, S, hdQK, seqK/S]
static Value splitKVReshapeK(PatternRewriter &rewriter, Location loc,
                             Value keys, MIXRShapedType kType,
                             int32_t splitKVVal) {
  ArrayRef<int64_t> kShape = kType.getShape();
  int64_t kRank = kType.getRank();
  int64_t seqK = kShape[kRank - 1];
  int64_t seqKPerSplit = seqK / splitKVVal;

  SmallVector<int64_t> kSplitShape(kShape.begin(), kShape.end() - 1);
  kSplitShape.push_back(splitKVVal);
  kSplitShape.push_back(seqKPerSplit);
  auto kSplitType = makeContiguousType(kSplitShape, kType.getElementType());
  Value kReshaped = migraphx::ReshapeOp::create(
      rewriter, loc, kSplitType, keys, rewriter.getI64ArrayAttr(kSplitShape));

  int64_t newKRank = kSplitShape.size();
  SmallVector<int64_t> kPerm;
  for (int64_t i = 0; i < newKRank - 3; ++i)
    kPerm.push_back(i);
  kPerm.push_back(newKRank - 2);
  kPerm.push_back(newKRank - 3);
  kPerm.push_back(newKRank - 1);

  SmallVector<int64_t> kTransShape(newKRank);
  auto kSplitMixr = cast<MIXRShapedType>(kSplitType);
  ArrayRef<int64_t> kSplitStrides = kSplitMixr.getStrides();
  SmallVector<int64_t> kTransStrides(newKRank);
  for (int64_t i = 0; i < newKRank; ++i)
    kTransShape[i] = kSplitShape[kPerm[i]];
  for (int64_t i = 0; i < newKRank; ++i)
    kTransStrides[i] = kSplitStrides[kPerm[i]];
  auto kTransType =
      MIXRShapedType::get(kTransShape, kTransStrides, kType.getElementType());
  return migraphx::TransposeOp::create(rewriter, loc, kTransType, kReshaped,
                                       rewriter.getI64ArrayAttr(kPerm));
}

/// Splits V's second-to-last dimension (seqK) into [splitKV, seqK/splitKV]:
/// V [B, seqK, hdV] -> reshape [B, S, seqK/S, hdV]
static Value splitKVReshapeV(PatternRewriter &rewriter, Location loc,
                             Value values, MIXRShapedType vType,
                             int32_t splitKVVal) {
  ArrayRef<int64_t> vShape = vType.getShape();
  int64_t vRank = vType.getRank();
  int64_t seqK = vShape[vRank - 2];
  int64_t seqKPerSplit = seqK / splitKVVal;
  int64_t headV = vShape[vRank - 1];

  SmallVector<int64_t> vSplitShape(vShape.begin(), vShape.end() - 2);
  vSplitShape.push_back(splitKVVal);
  vSplitShape.push_back(seqKPerSplit);
  vSplitShape.push_back(headV);
  auto vSplitType = makeContiguousType(vSplitShape, vType.getElementType());
  return migraphx::ReshapeOp::create(rewriter, loc, vSplitType, values,
                                     rewriter.getI64ArrayAttr(vSplitShape));
}

/// Broadcasts Q by inserting a split dimension of size 1 before the last two
/// dims, then broadcasting to splitKV:
/// Q [B, seqQ, hdQK] -> reshape [B, 1, seqQ, hdQK]
///                  -> broadcast [B, S, seqQ, hdQK]
static Value splitKVBroadcastQ(PatternRewriter &rewriter, Location loc,
                               Value queries, MIXRShapedType qType,
                               int32_t splitKVVal) {
  ArrayRef<int64_t> qShape = qType.getShape();
  int64_t qRank = qType.getRank();
  SmallVector<int64_t> qExpandShape(qShape.begin(), qShape.end() - 2);
  qExpandShape.push_back(1);
  qExpandShape.push_back(qShape[qRank - 2]);
  qExpandShape.push_back(qShape[qRank - 1]);
  auto qExpandType = makeContiguousType(qExpandShape, qType.getElementType());
  Value qExpanded =
      migraphx::ReshapeOp::create(rewriter, loc, qExpandType, queries,
                                  rewriter.getI64ArrayAttr(qExpandShape));

  SmallVector<int64_t> qBcShape(qExpandShape);
  qBcShape[qBcShape.size() - 3] = splitKVVal;
  SmallVector<int64_t> qBcStrides(qBcShape.size());
  ArrayRef<int64_t> expandStrides = qExpandType.getStrides();
  for (unsigned i = 0; i < qBcStrides.size(); ++i)
    qBcStrides[i] = expandStrides[i];
  qBcStrides[qBcShape.size() - 3] = 0;
  auto qBcType =
      MIXRShapedType::get(qBcShape, qBcStrides, qType.getElementType());
  return migraphx::MultiBroadcastOp::create(rewriter, loc, qBcType, qExpanded,
                                            rewriter.getI64ArrayAttr(qBcShape));
}

class AttentionDecompose final
    : public OpRewritePattern<migraphx::AttentionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(migraphx::AttentionOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    bool isSplitKV = hasMIXRFeature(op, migraphx::AttentionFeatures::splitkv);

    auto qType = cast<MIXRShapedType>(op.getQueries().getType());
    auto kType = cast<MIXRShapedType>(op.getKeys().getType());
    auto vType = cast<MIXRShapedType>(op.getValues().getType());

    int64_t qRank = qType.getRank();
    int64_t kRank = kType.getRank();

    // Handle GQA: if Q has more heads than K/V, broadcast K/V to match.
    Value queries = op.getQueries();
    Value keys = op.getKeys();
    Value values = op.getValues();

    if (qRank == 4 && kRank == 4 &&
        qType.getDimSize(1) != kType.getDimSize(1)) {
      int64_t numHeadsQ = qType.getDimSize(1);
      keys = broadcastForGQA(rewriter, loc, keys, numHeadsQ);
      values = broadcastForGQA(rewriter, loc, values, numHeadsQ);
      kType = cast<MIXRShapedType>(keys.getType());
      vType = cast<MIXRShapedType>(values.getType());
      kRank = kType.getRank();
    }

    // SplitKV: insert split dimension before the last two dims of Q; reshape
    // K's last dim and V's seq dim, then continue with Dot in this rewrite.
    // The verifier guarantees splitKV attr is present, > 1, LSE exists,
    // and seqK is divisible by splitKV.
    if (isSplitKV) {
      assert(op.getSplitKVAttr() && "verifier should ensure splitKV attr");
      assert(op.getSplitKVAttr().getInt() > 1 &&
             "verifier should ensure splitKV > 1");
      assert(op.getLse() && "verifier should ensure LSE for splitkv");
      int32_t splitKVVal = op.getSplitKVAttr().getInt();
      queries = splitKVBroadcastQ(rewriter, loc, queries, qType, splitKVVal);
      keys = splitKVReshapeK(rewriter, loc, keys, kType, splitKVVal);
      values = splitKVReshapeV(rewriter, loc, values, vType, splitKVVal);
      qType = cast<MIXRShapedType>(queries.getType());
      kType = cast<MIXRShapedType>(keys.getType());
      vType = cast<MIXRShapedType>(values.getType());
      qRank = qType.getRank();
      kRank = kType.getRank();
    }

    // Compute Q*K shape: batch dims from Q, seq_q from Q, seq_k from K. For
    // integer-typed Q/K the first GEMM is emitted as migraphx.quant_dot,
    // whose output is i32; the user-supplied preSoftmaxBody is then expected
    // to dequantize that i32 to the float softmax type. For float Q/K we
    // keep the original migraphx.dot path and the QK output matches Q's
    // element type.
    SmallVector<int64_t> qkShape(qType.getShape().begin(),
                                 qType.getShape().end());
    qkShape[qRank - 1] = kType.getShape()[kRank - 1];
    SmallVector<int64_t> qkStrides(qkShape.size());
    int64_t stride = 1;
    for (int64_t i = qkShape.size() - 1; i >= 0; --i) {
      qkStrides[i] = stride;
      stride *= qkShape[i];
    }
    Type elemType = qType.getElementType();
    bool isIntQK = !isa<FloatType>(elemType);
    Type qkElemType =
        isIntQK ? cast<Type>(IntegerType::get(rewriter.getContext(), 32))
                : elemType;
    auto qkType = MIXRShapedType::get(qkShape, qkStrides, qkElemType);

    // 1. First GEMM: Q * K
    Value qk;
    if (isIntQK) {
      qk = migraphx::QuantDotOp::create(rewriter, loc, qkType, queries, keys,
                                        /*scaleA=*/Value(),
                                        /*scaleB=*/Value());
    } else {
      qk = migraphx::DotOp::create(rewriter, loc, qkType, queries, keys);
    }

    // 2. Inline preSoftmaxBody elementwise ops.
    // The verifier guarantees that if preSoftmaxElemWiseInputs are present,
    // the body contains non-terminator ops, and the yield has a value.
    if (!op.getPreSoftmaxElemWiseInputs().empty()) {
      Block &block = op.getPreSoftmaxBody().front();
      IRMapping mapping;
      mapping.map(block.getArgument(0), qk);
      auto preSoftmaxInputs = op.getPreSoftmaxElemWiseInputs();
      for (unsigned i = 0; i < preSoftmaxInputs.size(); ++i)
        mapping.map(block.getArgument(i + 1), preSoftmaxInputs[i]);

      for (Operation &bodyOp : block) {
        if (bodyOp.hasTrait<OpTrait::IsTerminator>())
          continue;
        Operation *cloned = rewriter.clone(bodyOp, mapping);
        for (auto [oldRes, newRes] :
             llvm::zip(bodyOp.getResults(), cloned->getResults()))
          mapping.map(oldRes, newRes);
      }
      auto yieldOp = cast<migraphx::YieldOp>(block.getTerminator());
      qk = mapping.lookup(yieldOp.getValue());
    }

    // Apply feature-based masks sequentially. Each mask computes
    // greater(lhs, rhs) and applies where(mask, -inf, qk) independently.
    // This matches the MIGraphX expanded attention graph pattern where
    // each mask is a separate where op applied in order:
    //   1. Causal mask (if causal)
    //   2. Sliding window mask (if sliding_window)
    //   3. KV-cache mask (if kvcache)
    if (hasMIXRFeature(op, migraphx::AttentionFeatures::causal))
      qk = applyCausalMask(rewriter, loc, qk, op.getPrefixOffset());

    if (hasMIXRFeature(op, migraphx::AttentionFeatures::sliding_window)) {
      assert(op.getCurrentSeqLen() && "verifier should ensure currentSeqLen");
      assert(op.getSlidingWindowSizeAttr() &&
             "verifier should ensure slidingWindowSize");
      qk = applySlidingWindowMask(rewriter, loc, qk, op.getCurrentSeqLen(),
                                  op.getSlidingWindowSizeAttr().getInt());
    }

    if (hasMIXRFeature(op, migraphx::AttentionFeatures::kvcache)) {
      assert(op.getCurrentSeqLen() && "verifier should ensure currentSeqLen");
      qk = applyKVCacheMask(rewriter, loc, qk, op.getCurrentSeqLen());
    }

    // 3. Handle softmaxType: convert before softmax if needed. Mirror the
    // GPU side (rock::gridwise_attention_accel) by defaulting to V's
    // element type when softmaxType is unset; the verifier guarantees that
    // either softmaxType is explicitly set or the value entering softmax
    // already has V's element type, so this default is safe.
    Type qkCurrentElemType =
        cast<MIXRShapedType>(qk.getType()).getElementType();
    Type softmaxElemType = op.getSoftmaxType().value_or(vType.getElementType());

    if (softmaxElemType != qkCurrentElemType) {
      auto qkShaped = cast<MIXRShapedType>(qk.getType());
      auto convertedType = MIXRShapedType::get(
          qkShaped.getShape(), qkShaped.getStrides(), softmaxElemType);
      qk = migraphx::ConvertOp::create(rewriter, loc, convertedType, qk);
    }

    auto qkSoftmaxType = cast<MIXRShapedType>(qk.getType());
    int64_t softmaxAxis = qkSoftmaxType.getRank() - 1;

    // Compute reduced shape (last dim becomes 1) for reduce ops
    auto computeReducedType = [&](MIXRShapedType fullType) {
      SmallVector<int64_t> rShape(fullType.getShape());
      rShape[softmaxAxis] = 1;
      SmallVector<int64_t> rStrides(rShape.size());
      int64_t s = 1;
      for (int64_t i = rShape.size() - 1; i >= 0; --i) {
        rStrides[i] = s;
        s *= rShape[i];
      }
      return MIXRShapedType::get(rShape, rStrides, fullType.getElementType());
    };

    Value softmaxResult;
    Value lseValue;

    bool needLse = op.getLse() != nullptr;

    if (needLse) {
      // Decompose softmax manually to extract LSE intermediates:
      //   max = reduce_max(qk, axis=-1)
      //   norm = qk - max
      //   exp_val = exp(norm)
      //   sum_exp = reduce_sum(exp_val, axis=-1)
      //   recip = recip(sum_exp)
      //   softmax_result = exp_val * recip
      //   lse = log(sum_exp) + max
      auto axisAttr = rewriter.getI64ArrayAttr({softmaxAxis});
      auto reducedType = computeReducedType(qkSoftmaxType);

      Value maxVal = migraphx::ReduceMaxOp::create(rewriter, loc, reducedType,
                                                   qk, axisAttr);
      Value norm =
          migraphx::SubOp::create(rewriter, loc, qkSoftmaxType, qk, maxVal);
      Value expVal =
          migraphx::ExpOp::create(rewriter, loc, qkSoftmaxType, norm);
      Value sumExp = migraphx::ReduceSumOp::create(rewriter, loc, reducedType,
                                                   expVal, axisAttr);
      Value recip =
          migraphx::RecipOp::create(rewriter, loc, reducedType, sumExp);
      softmaxResult =
          migraphx::MulOp::create(rewriter, loc, qkSoftmaxType, expVal, recip);

      // LSE = log(sum_exp) + max
      Value logSumExp =
          migraphx::LogOp::create(rewriter, loc, reducedType, sumExp);
      lseValue = migraphx::AddOp::create(rewriter, loc, reducedType, logSumExp,
                                         maxVal);
    } else {
      // 4. Use migraphx.softmax when LSE is not needed
      softmaxResult =
          migraphx::SoftmaxOp::create(rewriter, loc, qkSoftmaxType, qk,
                                      rewriter.getI64IntegerAttr(softmaxAxis));
    }

    // 5. Convert back if softmaxType differs from values element type
    if (softmaxElemType != vType.getElementType()) {
      auto smShaped = cast<MIXRShapedType>(softmaxResult.getType());
      auto convertedBack = MIXRShapedType::get(
          smShaped.getShape(), smShaped.getStrides(), vType.getElementType());
      softmaxResult = migraphx::ConvertOp::create(rewriter, loc, convertedBack,
                                                  softmaxResult);
    }

    // 6. Second GEMM: softmax(QK) * V.
    // TODO: The CPU-side migraphx.dot accumulates in the operand element
    // type rather than promoting to f32 the way the GPU mfma path does
    // (rock::gridwise_attention_accel keeps gemm1's accumulator at
    // softmaxType / f32). For long sequences this can produce slightly
    // less accurate CPU reference results than the GPU; widen the dot's
    // internal accumulator (or split into f32 partial sums + downcast)
    // to match the GPU's mfma precision.
    auto resultType = cast<MIXRShapedType>(op.getResult().getType());

    Value result = migraphx::DotOp::create(rewriter, loc, resultType,
                                           softmaxResult, values);

    SmallVector<Value> results;
    results.push_back(result);

    // 7. Add LSE output if requested
    if (needLse) {
      auto lseOutputType = cast<MIXRShapedType>(op.getLse().getType());

      // The reduce ops keep the reduced axis as size 1 (e.g., [2, 64, 1])
      // but the verifier enforces LSE shape without the trailing 1
      // (e.g., [2, 64]). Reshape to drop it.
      SmallVector<int64_t> lseShape(lseOutputType.getShape());
      SmallVector<int64_t> lseStrides(lseShape.size());
      int64_t ls = 1;
      for (int64_t i = lseShape.size() - 1; i >= 0; --i) {
        lseStrides[i] = ls;
        ls *= lseShape[i];
      }
      auto reshapedLseType = MIXRShapedType::get(
          lseShape, lseStrides,
          cast<MIXRShapedType>(lseValue.getType()).getElementType());
      lseValue =
          migraphx::ReshapeOp::create(rewriter, loc, reshapedLseType, lseValue,
                                      rewriter.getI64ArrayAttr(lseShape));

      // Convert LSE element type if needed (e.g., f16 -> f32)
      if (reshapedLseType.getElementType() != lseOutputType.getElementType()) {
        lseValue =
            migraphx::ConvertOp::create(rewriter, loc, lseOutputType, lseValue);
      }

      results.push_back(lseValue);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

class SqrtDecompose final : public OpConversionPattern<migraphx::SqrtOp> {
public:
  using OpConversionPattern<migraphx::SqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    auto inA = op->getOperand(0);
    auto outputTy = cast<ShapedType>(op->getResults()[0].getType());
    auto rSop = migraphx::RsqrtOp::create(rewriter, loc, outputTy, inA);
    auto rCop = migraphx::RecipOp::create(rewriter, loc, outputTy, rSop);

    rewriter.replaceOp(op, rCop->getResults()[0]);
    return success();
  }
};

void populateMIGraphXSqrt(MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<SqrtDecompose>(context);
}

struct MIGraphXTransforms
    : public migraphx::impl::MIGraphXTransformPassBase<MIGraphXTransforms> {
  void runOnOperation() override {
    auto &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<migraphx::MIGraphXDialect, func::FuncDialect,
                           tosa::TosaDialect, mhal::MHALDialect>();
    target.addIllegalOp<migraphx::SqrtOp>();
    auto func = getOperation();

    populateMIGraphXSqrt(&ctx, patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
    // Only run with non-kernel functions:
    // - QuantDotDecompose: tosa.matmul_t_block_scaled doesn't have conversion
    //   to linalg in the upstream passes. For the kernel side, it is converted
    //   to rock.gemm with scales in TosaToRock.cpp.
    //   TODO: Remove once upstream TOSA adds this conversion.
    // - AttentionDecompose: migraphx.attention is decomposed into primitive
    //   migraphx ops (dot, softmax, etc.) for the host/CPU path. For the
    //   kernel side, migraphx.attention is preserved and lowered directly
    //   to rock.attention via the MIGraphXAttentionToRock pass.
    if (!func->hasAttr("rock.kernel")) {
      RewritePatternSet hostPatterns(&ctx);
      hostPatterns.add<QuantDotDecompose>(&ctx);
      hostPatterns.add<AttentionDecompose>(&ctx);
      if (failed(applyPatternsGreedily(func, std::move(hostPatterns))))
        signalPassFailure();
    }
  }
};

} // namespace
