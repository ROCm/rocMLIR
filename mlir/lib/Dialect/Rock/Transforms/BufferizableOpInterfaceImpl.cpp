//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-bufferizable-op-interface-impl"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::rock;

namespace mlir {
namespace rock {
namespace {

/// Bufferization of attention-like ops (AttentionOp, PagedAttentionOp).
template <typename AttentionOpTy>
struct AttentionLikeInterface
    : public BufferizableOpInterface::ExternalModel<
          AttentionLikeInterface<AttentionOpTy>, AttentionOpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto attnOp = mlir::cast<AttentionOpTy>(op);
    return (&opOperand != attnOp.getOutArgument() &&
            &opOperand != attnOp.getOutLseArgument());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto attnOp = mlir::cast<AttentionOpTy>(op);
    return (&opOperand == attnOp.getOutArgument() ||
            &opOperand == attnOp.getOutLseArgument());
  }

  // The buffer corresponding to the destination must equal the buffer
  // corresponding to the returned tensor
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    auto attnOp = mlir::cast<AttentionOpTy>(op);
    return (&opOperand == attnOp.getOutArgument() ||
            &opOperand == attnOp.getOutLseArgument());
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto attnOp = mlir::cast<AttentionOpTy>(op);
    AliasingValueList result;

    if (&opOperand == attnOp.getOutArgument()) {
      // First output argument aliases with first result
      result.addAlias({op->getResult(0), BufferRelation::Equivalent});
    } else if (&opOperand == attnOp.getOutLseArgument()) {
      // Second output argument aliases with second result
      result.addAlias({op->getResult(1), BufferRelation::Equivalent});
    }

    return result;
  }

  // The output argument is equal to the returned value
  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto attnOp = mlir::cast<AttentionOpTy>(op);

    // If the op already has 0 results, it's the bufferized clone
    if (op->getNumResults() == 0)
      return success();

    SmallVector<Value> bufferArgs;
    Value outBuffer, outLseBuffer;

    for (OpOperand &operand : op->getOpOperands()) {
      Value val = operand.get();
      Value buffer;

      // If already a memref, use it directly. Otherwise, get the buffer.
      if (isa<MemRefType>(val.getType())) {
        buffer = val;
      } else {
        FailureOr<Value> bufferResult =
            getBuffer(rewriter, val, options, state);
        if (failed(bufferResult)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Failed to bufferize value " << val << "\n");
          return failure();
        }
        buffer = *bufferResult;
      }

      bufferArgs.push_back(buffer);
      if (&operand == attnOp.getOutArgument())
        outBuffer = buffer;
      else if (&operand == attnOp.getOutLseArgument())
        outLseBuffer = buffer;
    }
    if (!outBuffer) {
      return op->emitOpError("Couldn't find output argument");
    }
    // no need to check outLseBuffer, because it is optional
    Operation *clonedOp =
        clone(rewriter, op, /*newResultTypes=*/TypeRange{}, bufferArgs);
    clonedOp->setAttr("resultSegmentSizes",
                      rewriter.getDenseI32ArrayAttr({0, 0}));
    SmallVector<Value> replacements = {outBuffer};
    if (outLseBuffer)
      replacements.push_back(outLseBuffer);
    replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

/// Bufferization of gemm-like ops, which rewrite to themselves with memref
/// arguments.
template <typename Concrete>
struct GemmLikeInterface
    : public BufferizableOpInterface::ExternalModel<GemmLikeInterface<Concrete>,
                                                    Concrete> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto cop = mlir::cast<Concrete>(op);
    return (&opOperand != cop.getOutArgument());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto cop = mlir::cast<Concrete>(op);
    return (&opOperand == cop.getOutArgument());
  }

  // The buffer corresponding to the destination must equal the buffer
  // corresponding to the returned tensor
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    auto cop = mlir::cast<Concrete>(op);
    return (&opOperand == cop.getOutArgument());
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto cop = mlir::cast<Concrete>(op);
    if (&opOperand == cop.getOutArgument()) {
      SmallVector<AliasingValue, 4> opResults;
      AliasingValueList result;
      for (auto opResult : op->getOpResults())
        result.addAlias({opResult, BufferRelation::Equivalent});
      return result;
    }
    return {};
  }

  // The output argument is equal to the returned value
  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto cop = mlir::cast<Concrete>(op);
    SmallVector<Value> bufferArgs;
    Value outBuffer;

    for (OpOperand &operand : op->getOpOperands()) {
      FailureOr<Value> buffer =
          getBuffer(rewriter, operand.get(), options, state);
      if (failed(buffer)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to bufferize value " << operand.get() << "\n");
        return failure();
      }
      bufferArgs.push_back(*buffer);
      if (&operand == cop.getOutArgument())
        outBuffer = *buffer;
    }
    if (!outBuffer) {
      return op->emitOpError("Couldn't find output argument");
    }
    clone(rewriter, op, /*newResultTypes=*/TypeRange{}, bufferArgs);
    replaceOpWithBufferizedValues(rewriter, op, outBuffer);
    return success();
  }
};

/// Bufferization of rock.transform, which bufferizes to itself but with memrefs
/// Based of of bufferization for tensor.expand_shape
struct TransformOpInterface
    : public BufferizableOpInterface::ExternalModel<TransformOpInterface,
                                                    rock::TransformOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    AliasingValueList result;
    for (auto opResult : op->getOpResults())
      result.addAlias({opResult, BufferRelation::Equivalent});
    return result;
  }

  // The output argument is equal to the returned value
  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto transformOp = mlir::cast<rock::TransformOp>(op);
    FailureOr<Value> input =
        getBuffer(rewriter, transformOp.getInput(), options, state);
    if (failed(input))
      return failure();

    replaceOpWithNewBufferizedOp<rock::TransformOp>(rewriter, op, *input,
                                                    transformOp.getTransform());
    return success();
  }
};

/// Bufferization of rock.tensor_untransform_cast, which bufferizes to the
/// buffer cerrosponding to the transformed argument (but untransformed)
struct TensorUntransformCastOpInterface
    : public BufferizableOpInterface::ExternalModel<
          TensorUntransformCastOpInterface, rock::TensorUntransformCastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    auto castOp = mlir::cast<rock::TensorUntransformCastOp>(op);
    Value operand = opOperand.get();
    return (operand == castOp.getTransformedResult());
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto castOp = mlir::cast<rock::TensorUntransformCastOp>(op);
    Value operand = opOperand.get();
    if (operand == castOp.getTransformedResult()) {
      AliasingValueList result;
      for (auto opResult : op->getOpResults())
        result.addAlias({opResult, BufferRelation::Equivalent});
      return result;
    }
    return {};
  }

  // The output argument is equal to the returned value
  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto castOp = mlir::cast<rock::TensorUntransformCastOp>(op);
    FailureOr<Value> transformedArg =
        getBuffer(rewriter, castOp.getTransformedArg(), options, state);
    if (failed(transformedArg))
      return failure();
    FailureOr<Value> transformedResult =
        getBuffer(rewriter, castOp.getTransformedResult(), options, state);
    if (failed(transformedResult))
      return failure();
    if (*transformedArg != *transformedResult)
      return op->emitOpError(
          "transformed argument and result map to different results");

    Value buffer = std::get<0>(rock::untransform(rewriter, *transformedArg));
    ArrayRef<int64_t> bufferShape =
        cast<ShapedType>(buffer.getType()).getShape();
    ArrayRef<int64_t> resultShape =
        cast<ShapedType>(castOp.getUntransformed().getType()).getShape();
    if (bufferShape != resultShape)
      return op->emitOpError("buffer shape not equal to result shape");
    replaceOpWithBufferizedValues(rewriter, op, buffer);
    return success();
  }
};

/// Bufferization of rock.paged_deref op.
/// This op reads page pointers and an optional mask, and produces a new tensor
/// containing the dereferenced data. Similar to rock.transform, it supports
/// both tensor and memref types.
struct PagedDerefOpInterface
    : public BufferizableOpInterface::ExternalModel<PagedDerefOpInterface,
                                                    PagedDerefOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Both pagePointers and addressMask are read
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // This op doesn't write to any of its inputs - it produces a new result
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    // No in-place requirement since we produce a new buffer
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // The inputs don't alias with the output - we produce a new result
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    // Result is a new value, not equivalent to any input
    return BufferRelation::Unknown;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto derefOp = mlir::cast<PagedDerefOp>(op);

    // Get buffers for input operands
    FailureOr<Value> pagePointersBuffer =
        getBuffer(rewriter, derefOp.getPagePointers(), options, state);
    if (failed(pagePointersBuffer))
      return failure();

    Value addressMaskBuffer;
    if (derefOp.getAddressMask()) {
      FailureOr<Value> maskBuffer =
          getBuffer(rewriter, derefOp.getAddressMask(), options, state);
      if (failed(maskBuffer))
        return failure();
      addressMaskBuffer = *maskBuffer;
    }

    // Determine the result memref type from the tensor type
    auto resultType = derefOp.getOutput().getType();
    MemRefType resultMemRefType;
    if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
      resultMemRefType = MemRefType::get(tensorType.getShape(),
                                         tensorType.getElementType());
    } else if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
      // Already a memref (shouldn't happen during bufferization, but handle it)
      resultMemRefType = memrefType;
    } else {
      return op->emitOpError("expected tensor or memref result type");
    }

    // Create new op with memref types
    replaceOpWithNewBufferizedOp<PagedDerefOp>(
        rewriter, op, resultMemRefType, *pagePointersBuffer, addressMaskBuffer);
    return success();
  }
};

} // namespace
} // namespace rock
} // namespace mlir

void mlir::rock::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, rock::RockDialect *dialect) {
    ConvOp::attachInterface<GemmLikeInterface<ConvOp>>(*ctx);
    ConvBwdDataOp::attachInterface<GemmLikeInterface<ConvBwdDataOp>>(*ctx);
    ConvBwdWeightOp::attachInterface<GemmLikeInterface<ConvBwdWeightOp>>(*ctx);
    GemmOp::attachInterface<GemmLikeInterface<GemmOp>>(*ctx);
    ReduceOp::attachInterface<GemmLikeInterface<ReduceOp>>(*ctx);

    // While these utility kernels aren't gemm wrappers, strictly, they still
    // bufferize like them
    ConvertingCopyKernelOp::attachInterface<
        GemmLikeInterface<ConvertingCopyKernelOp>>(*ctx);
    GemmElementwiseGemmOp::attachInterface<
        GemmLikeInterface<GemmElementwiseGemmOp>>(*ctx);
    ConvElementwiseGemmOp::attachInterface<
        GemmLikeInterface<ConvElementwiseGemmOp>>(*ctx);

    // Attention-like ops (regular and paged)
    AttentionOp::attachInterface<AttentionLikeInterface<AttentionOp>>(*ctx);
    PagedAttentionOp::attachInterface<
        AttentionLikeInterface<PagedAttentionOp>>(*ctx);

    // Paged deref op
    PagedDerefOp::attachInterface<PagedDerefOpInterface>(*ctx);

    TransformOp::attachInterface<TransformOpInterface>(*ctx);
    TensorUntransformCastOp::attachInterface<TensorUntransformCastOpInterface>(
        *ctx);
  });
}
