//===- MIGraphXAttentionToRock.cpp - Lower migraphx.attention to rock ------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MIGRAPHXATTENTIONTOROCKPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {

/// Determine the number of attention heads from an MIXRShaped value.
/// For 4D shapes [batch, numHeads, seq, head_dim], returns shape[1].
/// For 3D shapes, looks through migraphx.reshape to find the original
/// 4D shape. Returns 1 if the number of heads cannot be determined.
static int32_t getNumHeads(Value val) {
  auto shapedTy = cast<ShapedType>(val.getType());
  if (shapedTy.getRank() == 4)
    return shapedTy.getDimSize(1);
  if (auto reshape = val.getDefiningOp<migraphx::ReshapeOp>()) {
    auto inputType = cast<ShapedType>(reshape.getInput().getType());
    if (inputType.getRank() == 4)
      return inputType.getDimSize(1);
  }
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
  return tensor::CollapseShapeOp::create(rewriter, loc, newType, val,
                                          getLeadingDimReassoc(shapedTy.getRank()));
}

struct AttentionToRockPattern
    : public OpRewritePattern<migraphx::AttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(migraphx::AttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto mixrResultType = cast<MIXRShapedType>(op.getResult().getType());
    RankedTensorType resultType = mixrResultType.asTensor();

    // Convert MIXRShaped inputs to tensors via migraphx.mlir.as.logical.shape
    Value queries = migraphx::AsLogicalShapeOp::create(
        rewriter, loc, op.getQueries());
    Value keys = migraphx::AsLogicalShapeOp::create(
        rewriter, loc, op.getKeys());
    Value values = migraphx::AsLogicalShapeOp::create(
        rewriter, loc, op.getValues());

    SmallVector<Value> preSoftmaxInputs;
    for (Value input : op.getPreSoftmaxElemWiseInputs())
      preSoftmaxInputs.push_back(
          migraphx::AsLogicalShapeOp::create(rewriter, loc, input));

    // Collapse output type to 3D if needed
    RankedTensorType rockResultType = getCollapsed3DType(resultType);

    // Allocate output tensor
    Value output = bufferization::AllocTensorOp::create(
        rewriter, loc, rockResultType, ValueRange{});

    // Allocate LSE output if needed
    Value lseOut;
    RankedTensorType lseType;
    if (op.getLse()) {
      lseType = cast<MIXRShapedType>(op.getLse().getType()).asTensor();
      lseOut = bufferization::AllocTensorOp::create(
          rewriter, loc, lseType, ValueRange{});
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
        /*currentSeqLen=*/Value(),
        /*prefixOffset=*/Value(),
        /*out=*/output,
        /*lse=*/lseOut,
        /*numHeadsQ=*/rewriter.getI32IntegerAttr(numHeadsQ),
        /*numHeadsKV=*/rewriter.getI32IntegerAttr(numHeadsKV),
        /*qTransposed=*/nullptr,
        /*kTransposed=*/nullptr,
        /*vTransposed=*/nullptr,
        /*oTransposed=*/nullptr,
        /*causal=*/nullptr,
        /*splitKV=*/rewriter.getI32IntegerAttr(1),
        /*slidingWindowSize=*/nullptr,
        /*features=*/nullptr,
        rewriter.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
        softmaxTypeAttr,
        /*params0=*/nullptr,
        /*params1=*/nullptr,
        /*firstGemmIndices=*/
        rewriter.getDenseI64ArrayAttr(firstGemmBlockIndex));

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

        // Add block args as memref types (converted from MIXRShaped)
        for (BlockArgument srcArg : srcBlock.getArguments()) {
          auto mixrTy = cast<MIXRShapedType>(srcArg.getType());
          auto memrefTy =
              MemRefType::get(mixrTy.getShape(), mixrTy.getElementType());
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

        // Determine output type from the last non-terminator op's result
        MIXRShapedType outputMixrTy;
        for (Operation &bodyOp : llvm::reverse(srcBlock))
          if (!bodyOp.hasTrait<OpTrait::IsTerminator>()) {
            outputMixrTy = cast<MIXRShapedType>(bodyOp.getResult(0).getType());
            break;
          }
        auto outputMemrefTy = MemRefType::get(outputMixrTy.getShape(),
                                              outputMixrTy.getElementType());
        Value alloc = memref::AllocOp::create(rewriter, loc, outputMemrefTy);

        int64_t rank = outputMixrTy.getRank();
        SmallVector<AffineMap> indexingMaps(
            genericInputs.size() + 1, rewriter.getMultiDimIdentityMap(rank));
        SmallVector<utils::IteratorType> iterTypes(
            rank, utils::IteratorType::parallel);

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{}, genericInputs, ValueRange{alloc},
            indexingMaps, iterTypes);

        // Build the fused body: map each block arg to a generic block arg,
        // then chain all scalar ops
        Block *genBlock = rewriter.createBlock(
            &genericOp.getRegion(), genericOp.getRegion().end());
        Type elemTy = outputMixrTy.getElementType();
        for (size_t i = 0; i < genericInputs.size() + 1; ++i)
          genBlock->addArgument(elemTy, loc);

        // Map preSoftmaxBody block args to generic block args (scalars)
        IRMapping scalarMapping;
        for (auto [i, srcArg] : llvm::enumerate(srcBlock.getArguments()))
          scalarMapping.map(srcArg, genBlock->getArgument(i));

        rewriter.setInsertionPointToStart(genBlock);
        Value lastScalar;
        for (Operation &bodyOp : srcBlock) {
          if (bodyOp.hasTrait<OpTrait::IsTerminator>())
            continue;

          SmallVector<Value> scalarOperands;
          for (Value operand : bodyOp.getOperands())
            scalarOperands.push_back(scalarMapping.lookup(operand));

          Value scalarResult;
          if (isa<migraphx::MulOp>(bodyOp))
            scalarResult = arith::MulFOp::create(
                rewriter, loc, scalarOperands[0], scalarOperands[1]);
          else if (isa<migraphx::AddOp>(bodyOp))
            scalarResult = arith::AddFOp::create(
                rewriter, loc, scalarOperands[0], scalarOperands[1]);
          else if (isa<migraphx::SubOp>(bodyOp))
            scalarResult = arith::SubFOp::create(
                rewriter, loc, scalarOperands[0], scalarOperands[1]);
          else
            return bodyOp.emitError(
                "unsupported migraphx op in preSoftmaxBody: ")
                   << bodyOp.getName();
          scalarMapping.map(bodyOp.getResult(0), scalarResult);
          lastScalar = scalarResult;
        }
        linalg::YieldOp::create(rewriter, loc, lastScalar);

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
      results.push_back(migraphx::AsUnderlyingShapeOp::create(
          rewriter, loc, mixrLseType, rockAttn.getLseOut()));
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
      if (!func->hasAttr("kernel"))
        return;
      RewritePatternSet patterns(ctx);
      patterns.add<AttentionToRockPattern>(ctx);
      if (failed(applyPatternsGreedily(func, std::move(patterns))))
        signalPassFailure();
    });
  }
};

} // namespace
