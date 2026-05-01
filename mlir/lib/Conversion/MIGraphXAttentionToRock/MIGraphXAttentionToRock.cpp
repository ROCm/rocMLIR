//===- MIGraphXAttentionToRock.cpp - Lower migraphx.attention to rock
//------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MIGraphX/IR/AttentionUtils.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "migraphx-attention-to-rock"

namespace mlir {
#define GEN_PASS_DEF_MIGRAPHXATTENTIONTOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {

/// Determine the number of attention heads from an MIXRShaped value.
/// For 4D shapes [batch, numHeads, seq, head_dim], returns shape[1].
/// For other ranks (typically 3D, where the user has already collapsed
/// [batch, numHeads] into a single batch dim), returns 1; rock.attention
/// then treats the entire leading dim as the batch with one head per
/// batch element. The verifier guarantees that GQA (numHeadsQ !=
/// numHeadsKV) is only legal when Q is at least rank 4, so the rank<4
/// fallback here can only fire when numHeadsQ == numHeadsKV (in which
/// case the choice between "1 head, big batch" and "real heads, smaller
/// batch" is numerically irrelevant for the attention math).
static int32_t getNumHeads(Value val) {
  auto shapedTy = cast<ShapedType>(val.getType());
  if (shapedTy.getRank() == 4)
    return shapedTy.getDimSize(1);
  return 1;
}

/// Build reassociation indices for collapsing/expanding all leading dims
/// (dims 0..rank-3) into a single dim, keeping the last two dims separate.
static SmallVector<SmallVector<int64_t, 2>> getLeadingDimReassoc(int64_t rank) {
  SmallVector<SmallVector<int64_t, 2>> reassoc;
  SmallVector<int64_t, 2> firstGroup;
  for (int64_t i = 0; i < rank - 2; ++i)
    firstGroup.push_back(i);
  reassoc.push_back(firstGroup);
  reassoc.push_back(SmallVector<int64_t, 2>{rank - 2});
  reassoc.push_back(SmallVector<int64_t, 2>{rank - 1});
  return reassoc;
}

/// Get a 3D tensor type by collapsing all leading dims into one.
/// Returns the original type if already 3D or lower.
static RankedTensorType getCollapsed3DType(RankedTensorType ty) {
  if (ty.getRank() <= 3)
    return ty;
  ArrayRef<int64_t> shape = ty.getShape();
  int64_t collapsed = 1;
  for (int64_t i = 0; i < ty.getRank() - 2; ++i)
    collapsed *= shape[i];
  return RankedTensorType::get(
      {collapsed, shape[ty.getRank() - 2], shape[ty.getRank() - 1]},
      ty.getElementType());
}

/// Collapse a tensor value from 4D+ to 3D for rock.attention.
/// Returns the value unchanged if already 3D or lower.
static Value collapseTo3D(PatternRewriter &rewriter, Location loc, Value val) {
  auto shapedTy = cast<RankedTensorType>(val.getType());
  auto newType = getCollapsed3DType(shapedTy);
  if (newType == shapedTy)
    return val;
  return tensor::CollapseShapeOp::create(
      rewriter, loc, newType, val, getLeadingDimReassoc(shapedTy.getRank()));
}

static Value collapseTo1D(PatternRewriter &rewriter, Location loc, Value val) {
  auto shapedTy = cast<RankedTensorType>(val.getType());
  int64_t rank = shapedTy.getRank();
  if (rank <= 1)
    return val;
  SmallVector<ReassociationIndices> reassoc;
  ReassociationIndices allDims;
  for (int64_t i = 0; i < rank; ++i)
    allDims.push_back(i);
  reassoc.push_back(allDims);
  return tensor::CollapseShapeOp::create(rewriter, loc, val, reassoc);
}

static Value prepareOptionalOperand(PatternRewriter &rewriter, Location loc,
                                    Value mixrVal) {
  if (!mixrVal)
    return Value();
  Value tensor = migraphx::AsLogicalShapeOp::create(rewriter, loc, mixrVal);
  return collapseTo1D(rewriter, loc, tensor);
}

/// True when the given (possibly absent) MIXRShaped element type is an
/// unsigned integer. Mirrors the rule used by MIGraphXToTosa::createCastOp
/// and MIGraphXToLinalg::castTensor: signless and signed integers go
/// through the regular cast path; only explicitly-unsigned integers need
/// the unsigned conversion.
static bool isMixrUnsignedInt(Type mixrElemTy) {
  auto intTy = dyn_cast<IntegerType>(mixrElemTy);
  return intTy && intTy.isUnsigned();
}

/// Lower a single migraphx body op to its scalar arith / math equivalent.
/// Returns nullptr if the op is not supported. The set of supported ops
/// must be kept in sync with the verifier's allowlist in
/// AttentionOp::verify so the verifier never accepts a body the lowering
/// can't handle. The mapping mirrors MIGraphXToLinalg's
/// ElementwiseConverter / GenericElementwiseOpConverter coverage but
/// emits scalar arith/math ops directly for use inside a linalg.generic
/// body, similar in shape to upstream's
/// TosaToLinalg::createLinalgBodyCalculationForElementwiseOp.
static Value lowerMIGraphXElementwiseToScalar(Operation &bodyOp,
                                              ArrayRef<Value> operands,
                                              Type resultElemTy,
                                              PatternRewriter &rewriter,
                                              Location loc) {
  // Binary float arithmetic.
  if (isa<migraphx::MulOp>(bodyOp))
    return arith::MulFOp::create(rewriter, loc, operands[0], operands[1]);
  if (isa<migraphx::AddOp>(bodyOp))
    return arith::AddFOp::create(rewriter, loc, operands[0], operands[1]);
  if (isa<migraphx::SubOp>(bodyOp))
    return arith::SubFOp::create(rewriter, loc, operands[0], operands[1]);
  if (isa<migraphx::DivOp>(bodyOp))
    return arith::DivFOp::create(rewriter, loc, operands[0], operands[1]);
  if (isa<migraphx::PowOp>(bodyOp))
    return math::PowFOp::create(rewriter, loc, operands[0], operands[1]);

  // Unary float arithmetic.
  if (isa<migraphx::NegOp>(bodyOp))
    return arith::NegFOp::create(rewriter, loc, operands[0]);
  if (isa<migraphx::AbsOp>(bodyOp))
    return math::AbsFOp::create(rewriter, loc, operands[0]);
  if (isa<migraphx::CeilOp>(bodyOp))
    return math::CeilOp::create(rewriter, loc, operands[0]);
  if (isa<migraphx::FloorOp>(bodyOp))
    return math::FloorOp::create(rewriter, loc, operands[0]);
  if (isa<migraphx::ExpOp>(bodyOp))
    return math::ExpOp::create(rewriter, loc, operands[0]);
  if (isa<migraphx::LogOp>(bodyOp))
    return math::LogOp::create(rewriter, loc, operands[0]);
  if (isa<migraphx::SqrtOp>(bodyOp))
    return math::SqrtOp::create(rewriter, loc, operands[0]);
  if (isa<migraphx::TanhOp>(bodyOp))
    return math::TanhOp::create(rewriter, loc, operands[0]);
  if (isa<migraphx::ErfOp>(bodyOp))
    return math::ErfOp::create(rewriter, loc, operands[0]);

  // recip(x) = 1 / x
  if (isa<migraphx::RecipOp>(bodyOp)) {
    Value one = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(resultElemTy, 1.0));
    return arith::DivFOp::create(rewriter, loc, one, operands[0]);
  }

  // relu(x) = max(0, x)
  if (isa<migraphx::ReluOp>(bodyOp)) {
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(resultElemTy));
    return arith::MaximumFOp::create(rewriter, loc, zero, operands[0]);
  }

  // sigmoid(x) = 1 / (1 + exp(-x))
  if (isa<migraphx::SigmoidOp>(bodyOp)) {
    Value one = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(resultElemTy, 1.0));
    Value negX = arith::NegFOp::create(rewriter, loc, operands[0]);
    Value expNegX = math::ExpOp::create(rewriter, loc, negX);
    Value denom = arith::AddFOp::create(rewriter, loc, one, expNegX);
    return arith::DivFOp::create(rewriter, loc, one, denom);
  }

  // where(cond_i8, a, b) = arith.select(cond_i1, a, b). MIGraphX represents
  // booleans as i8 (see migraphx.where's cond constraint), so cast the i8
  // condition to i1 before the select.
  if (isa<migraphx::WhereOp>(bodyOp)) {
    Value cond =
        convertScalarToDtype(rewriter, loc, operands[0], rewriter.getI1Type(),
                             /*isUnsignedCast=*/false);
    return arith::SelectOp::create(rewriter, loc, cond, operands[1],
                                   operands[2]);
  }

  // convert: rely on mlir::convertScalarToDtype (the upstream helper used by
  // MIGraphXToLinalg::castTensor) to pick sitofp/uitofp/extf/truncf based on
  // the source/destination types and the signedness flag derived from the
  // original MIGraphX-side element type (signedness is dropped by
  // MIXRShapedType::asTensor, so we have to read it from the MIXR op).
  if (auto convert = dyn_cast<migraphx::ConvertOp>(&bodyOp)) {
    Type inputMixrElemTy = getElementTypeOrSelf(convert.getInA().getType());
    return convertScalarToDtype(rewriter, loc, operands[0], resultElemTy,
                                isMixrUnsignedInt(inputMixrElemTy));
  }

  // dequantizelinear: out = (cast<float>(input) - cast<float>(bias)) * scale.
  // operands[0] = input (any int/float), operands[1] = scale (float, same
  // type as result), operands[2] = bias (optional, same type as input).
  if (auto dq = dyn_cast<migraphx::DeQuantizeLinearOp>(&bodyOp)) {
    Type inputMixrElemTy = getElementTypeOrSelf(dq.getInput().getType());
    Value casted =
        convertScalarToDtype(rewriter, loc, operands[0], resultElemTy,
                             isMixrUnsignedInt(inputMixrElemTy));
    Value shifted = casted;
    if (operands.size() == 3) {
      Type biasMixrElemTy = getElementTypeOrSelf(dq.getBias().getType());
      Value biasCasted =
          convertScalarToDtype(rewriter, loc, operands[2], resultElemTy,
                               isMixrUnsignedInt(biasMixrElemTy));
      shifted = arith::SubFOp::create(rewriter, loc, casted, biasCasted);
    }
    return arith::MulFOp::create(rewriter, loc, shifted, operands[1]);
  }
  return nullptr;
}

struct AttentionToRockPattern : public OpRewritePattern<migraphx::AttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(migraphx::AttentionOp op,
                                PatternRewriter &rewriter) const override {
    auto features = op.getFeatures();

    Location loc = op.getLoc();

    auto mixrResultType = cast<MIXRShapedType>(op.getResult().getType());
    RankedTensorType resultType = mixrResultType.asTensor();

    // Convert MIXRShaped inputs to tensors via migraphx.mlir.as.logical.shape
    Value queries =
        migraphx::AsLogicalShapeOp::create(rewriter, loc, op.getQueries());
    Value keys =
        migraphx::AsLogicalShapeOp::create(rewriter, loc, op.getKeys());
    Value values =
        migraphx::AsLogicalShapeOp::create(rewriter, loc, op.getValues());

    SmallVector<Value> preSoftmaxInputs;
    for (Value input : op.getPreSoftmaxElemWiseInputs()) {
      Value tensor = migraphx::AsLogicalShapeOp::create(rewriter, loc, input);
      preSoftmaxInputs.push_back(collapseTo3D(rewriter, loc, tensor));
    }

    // Collapse output type to 3D if needed
    RankedTensorType rockResultType = getCollapsed3DType(resultType);

    // Allocate output tensor
    Value output = bufferization::AllocTensorOp::create(
        rewriter, loc, rockResultType, ValueRange{});

    // Allocate LSE output if needed. Rock expects LSE as 2D [batch, seqQ]
    // where batch includes splitKV. Collapse leading dims to get there.
    Value lseOut;
    RankedTensorType lseType;
    RankedTensorType origLseType;
    if (op.getLse()) {
      origLseType = cast<MIXRShapedType>(op.getLse().getType()).asTensor();
      // Collapse to 2D: all leading dims (including splitKV) into batch
      if (origLseType.getRank() > 2) {
        ArrayRef<int64_t> lseShape = origLseType.getShape();
        int64_t collapsedBatch = 1;
        for (int64_t i = 0; i < origLseType.getRank() - 1; ++i)
          collapsedBatch *= lseShape[i];
        lseType = RankedTensorType::get(
            {collapsedBatch, lseShape[origLseType.getRank() - 1]},
            origLseType.getElementType());
      } else {
        lseType = origLseType;
      }
      lseOut = bufferization::AllocTensorOp::create(rewriter, loc, lseType,
                                                    ValueRange{});
    }

    int32_t numHeadsQ = getNumHeads(op.getQueries());
    int32_t numHeadsKV = getNumHeads(op.getKeys());

    queries = collapseTo3D(rewriter, loc, queries);
    keys = collapseTo3D(rewriter, loc, keys);
    values = collapseTo3D(rewriter, loc, values);

    // softmaxType attribute
    TypeAttr softmaxTypeAttr;
    if (op.getSoftmaxType())
      softmaxTypeAttr = TypeAttr::get(*op.getSoftmaxType());

    Value currentSeqLen =
        prepareOptionalOperand(rewriter, loc, op.getCurrentSeqLen());
    Value prefixOffset =
        prepareOptionalOperand(rewriter, loc, op.getPrefixOffset());

    UnitAttr causalAttr;
    if (hasAttentionFeature(features, AttentionFeatures::causal))
      causalAttr = rewriter.getUnitAttr();

    // Forward splitKV
    int32_t splitKVVal = 1;
    if (op.getSplitKV())
      splitKVVal = op.getSplitKVAttr().getInt();

    // Forward slidingWindowSize
    IntegerAttr slidingWindowSizeAttr;
    if (op.getSlidingWindowSize())
      slidingWindowSizeAttr = op.getSlidingWindowSizeAttr();

    // Build the firstGemmIndices: index 0 for the QK result in the
    // preSoftmaxBody block args
    int64_t firstGemmBlockIndex = 0;

    // Create rock.attention op
    auto rockAttn = rock::AttentionOp::create(
        rewriter, loc,
        /*result=*/rockResultType,
        /*lseOut=*/lseType,
        /*queries=*/queries,
        /*keys=*/keys,
        /*values=*/values,
        /*preSoftmaxElemWiseInputs=*/preSoftmaxInputs,
        /*currentSeqLen=*/currentSeqLen,
        /*prefixOffset=*/prefixOffset,
        /*out=*/output,
        /*lse=*/lseOut,
        /*numHeadsQ=*/rewriter.getI32IntegerAttr(numHeadsQ),
        /*numHeadsKV=*/rewriter.getI32IntegerAttr(numHeadsKV),
        /*qTransposed=*/nullptr,
        /*kTransposed=*/nullptr,
        /*vTransposed=*/nullptr,
        /*oTransposed=*/nullptr,
        /*causal=*/causalAttr,
        /*splitKV=*/rewriter.getI32IntegerAttr(splitKVVal),
        /*slidingWindowSize=*/slidingWindowSizeAttr,
        /*features=*/nullptr,
        rewriter.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
        softmaxTypeAttr,
        /*params0=*/nullptr,
        /*params1=*/nullptr,
        /*firstGemmIndices=*/
        rewriter.getDenseI64ArrayAttr(firstGemmBlockIndex),
        /*preSoftmaxHasSplitKVTransforms=*/rewriter.getBoolAttr(false));

    // Forward perf_config if present on the source op
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      rockAttn->setAttr("perf_config", attr);

    // Build the preSoftmaxBody region for rock.attention. Convert migraphx
    // elementwise ops to memref-based linalg.generic ops with the
    // memref.alloc + memref.copy pattern that rock's downstream pipeline
    // expects.
    {
      Region &srcRegion = op.getPreSoftmaxBody();
      Region &dstRegion = rockAttn.getPreSoftmaxBody();
      PatternRewriter::InsertionGuard guard(rewriter);

      if (!preSoftmaxInputs.empty() && !srcRegion.empty()) {
        Block &srcBlock = srcRegion.front();
        Block *dstBlock = &dstRegion.emplaceBlock();
        IRMapping mapping;

        // Add block args as memref types (converted from MIXRShaped),
        // using collapsed 3D shapes to prevent vectorization crash.
        for (BlockArgument srcArg : srcBlock.getArguments()) {
          auto mixrTy = cast<MIXRShapedType>(srcArg.getType());
          auto collapsedTy = getCollapsed3DType(mixrTy.asTensor());
          auto memrefTy = MemRefType::get(collapsedTy.getShape(),
                                          collapsedTy.getElementType());
          auto dstArg = dstBlock->addArgument(memrefTy, loc);
          mapping.map(srcArg, dstArg);
        }

        rewriter.setInsertionPointToStart(dstBlock);

        // Build a single fused linalg.generic containing all the scalar
        // elementwise ops from the preSoftmaxBody. All block args become
        // inputs; one output is allocated for the result.
        SmallVector<Value> genericInputs;
        for (BlockArgument srcArg : srcBlock.getArguments())
          genericInputs.push_back(mapping.lookup(srcArg));

        // Determine output type from the yield operand, using collapsed 3D
        // shape.
        auto yieldOp = cast<migraphx::YieldOp>(srcBlock.getTerminator());
        auto outputMixrTy = cast<MIXRShapedType>(yieldOp.getValue().getType());
        auto collapsedOutTy = getCollapsed3DType(outputMixrTy.asTensor());
        auto outputMemrefTy = MemRefType::get(collapsedOutTy.getShape(),
                                              collapsedOutTy.getElementType());
        Value alloc = memref::AllocOp::create(rewriter, loc, outputMemrefTy);

        int64_t rank = collapsedOutTy.getRank();
        SmallVector<AffineMap> indexingMaps(
            genericInputs.size() + 1, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType> iterTypes(
            rank, utils::IteratorType::parallel);

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{}, genericInputs, ValueRange{alloc},
            indexingMaps, iterTypes);

        // Build the fused body: map each block arg to a generic block arg,
        // then chain all scalar ops. Each generic block arg uses the element
        // type of the corresponding linalg.generic input (so e.g. an i32 QK
        // input from i8 GEMM stays i32 in the body until a dequantize op
        // upcasts it).
        Block *genBlock = rewriter.createBlock(&genericOp.getRegion(),
                                               genericOp.getRegion().end());
        for (Value input : genericInputs)
          genBlock->addArgument(getElementTypeOrSelf(input.getType()), loc);
        // Output (last) block arg matches the output memref's element type.
        genBlock->addArgument(outputMemrefTy.getElementType(), loc);

        // Map preSoftmaxBody block args to generic block args (scalars)
        IRMapping scalarMapping;
        for (auto [i, srcArg] : llvm::enumerate(srcBlock.getArguments()))
          scalarMapping.map(srcArg, genBlock->getArgument(i));

        rewriter.setInsertionPointToStart(genBlock);
        for (Operation &bodyOp : srcBlock) {
          if (bodyOp.hasTrait<OpTrait::IsTerminator>())
            continue;

          SmallVector<Value> scalarOperands;
          for (Value operand : bodyOp.getOperands())
            scalarOperands.push_back(scalarMapping.lookup(operand));

          Type bodyResultElemTy =
              getElementTypeOrSelf(bodyOp.getResult(0).getType());
          Value scalarResult = lowerMIGraphXElementwiseToScalar(
              bodyOp, scalarOperands, bodyResultElemTy, rewriter, loc);
          // Lock-step contract with isAllowedInPreSoftmaxBody: the verifier
          // only accepts body ops in the allowlist, so anything that reaches
          // here and the dispatcher fails to handle is a coupling bug
          // between the two lists. Trip the assertion loudly in debug
          // builds; release builds still get the structured error below.
          assert(
              (scalarResult || !migraphx::isAllowedInPreSoftmaxBody(bodyOp)) &&
              "isAllowedInPreSoftmaxBody and lowerMIGraphXElementwiseToScalar "
              "have drifted: verifier-approved op was not lowered. Update "
              "both AttentionUtils.h::isAllowedInPreSoftmaxBody and this "
              "dispatch table together.");
          if (!scalarResult)
            return bodyOp.emitError(
                       "unsupported migraphx op in preSoftmaxBody: ")
                   << bodyOp.getName();
          scalarMapping.map(bodyOp.getResult(0), scalarResult);
        }
        Value yieldedScalar = scalarMapping.lookup(yieldOp.getValue());
        linalg::YieldOp::create(rewriter, loc, yieldedScalar);

        // Copy result to output memref arg + yield
        rewriter.setInsertionPointAfter(genericOp);
        Value outMemref = dstBlock->addArgument(outputMemrefTy, loc);
        memref::CopyOp::create(rewriter, loc, alloc, outMemref);
        rock::YieldOp::create(rewriter, loc);
      }

      // Ensure at least one block exists
      if (dstRegion.empty()) {
        Block *emptyBlock = &dstRegion.emplaceBlock();
        rewriter.setInsertionPointToStart(emptyBlock);
        rock::YieldOp::create(rewriter, loc);
      }
    }

    // Replace the migraphx.attention results, expanding back to original
    // shape if we collapsed to 3D, then wrapping tensor -> MIXRShaped
    SmallVector<Value> results;
    Value attnResult = rockAttn.getResult();
    if (rockResultType != resultType) {
      attnResult = tensor::ExpandShapeOp::create(
          rewriter, loc, resultType, attnResult,
          getLeadingDimReassoc(resultType.getRank()));
    }
    results.push_back(migraphx::AsUnderlyingShapeOp::create(
        rewriter, loc, mixrResultType, attnResult));

    if (op.getLse()) {
      auto mixrLseType = cast<MIXRShapedType>(op.getLse().getType());
      Value lseResult = rockAttn.getLseOut();
      // Expand LSE back from 2D to original shape if it was collapsed
      if (origLseType && lseType != origLseType) {
        SmallVector<SmallVector<int64_t, 2>> lseReassoc;
        SmallVector<int64_t, 2> leadingDims;
        for (int64_t i = 0; i < origLseType.getRank() - 1; ++i)
          leadingDims.push_back(i);
        lseReassoc.push_back(leadingDims);
        lseReassoc.push_back(
            SmallVector<int64_t, 2>{origLseType.getRank() - 1});
        lseResult = tensor::ExpandShapeOp::create(rewriter, loc, origLseType,
                                                  lseResult, lseReassoc);
      }
      results.push_back(migraphx::AsUnderlyingShapeOp::create(
          rewriter, loc, mixrLseType, lseResult));
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct MIGraphXAttentionToRockPass
    : public impl::MIGraphXAttentionToRockPassBase<
          MIGraphXAttentionToRockPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();

    op->walk([&](func::FuncOp func) {
      bool isKernelFunc = func->hasAttr("rock.kernel");
      if (!isKernelFunc)
        return;
      RewritePatternSet patterns(ctx);
      patterns.add<AttentionToRockPattern>(ctx);
      if (failed(applyPatternsGreedily(func, std::move(patterns))))
        signalPassFailure();
    });
  }
};

} // namespace
