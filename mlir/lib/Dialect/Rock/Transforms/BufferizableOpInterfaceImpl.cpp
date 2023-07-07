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

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-bufferizable-op-interface-impl"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::rock;

namespace mlir {
namespace rock {
namespace {

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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto cop = mlir::cast<Concrete>(op);
    if (&opOperand == cop.getOutArgument()) {
      SmallVector<AliasingOpResult, 4> opResults;
      AliasingOpResultList result;
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
                          const BufferizationOptions &options) const {
    auto cop = mlir::cast<Concrete>(op);
    SmallVector<Value> bufferArgs;
    Value outBuffer;

    for (OpOperand &operand : op->getOpOperands()) {
      FailureOr<Value> buffer = getBuffer(rewriter, operand.get(), options);
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
      return op->emitOpError("Couldn't find output argument\n");
    }

    rewriter.create<Concrete>(op->getLoc(), TypeRange{}, bufferArgs,
                              op->getAttrs());
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    AliasingOpResultList result;
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
                          const BufferizationOptions &options) const {
    auto transformOp = mlir::cast<rock::TransformOp>(op);
    FailureOr<Value> input =
        getBuffer(rewriter, transformOp.getInput(), options);
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto castOp = mlir::cast<rock::TensorUntransformCastOp>(op);
    Value operand = opOperand.get();
    if (operand == castOp.getTransformedResult()) {
      AliasingOpResultList result;
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
                          const BufferizationOptions &options) const {
    auto castOp = mlir::cast<rock::TensorUntransformCastOp>(op);
    FailureOr<Value> transformedArg =
        getBuffer(rewriter, castOp.getTransformedArg(), options);
    if (failed(transformedArg))
      return failure();
    FailureOr<Value> transformedResult =
        getBuffer(rewriter, castOp.getTransformedResult(), options);
    if (failed(transformedResult))
      return failure();
    if (*transformedArg != *transformedResult)
      return op->emitOpError(
          "transformed argument and result map to different results");

    Value buffer = std::get<0>(rock::untransform(rewriter, *transformedArg));
    ArrayRef<int64_t> bufferShape =
        buffer.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> resultShape =
        castOp.getUntransformed().getType().cast<ShapedType>().getShape();
    if (bufferShape != resultShape)
      return op->emitOpError("buffer shape not equal to result shape");
    replaceOpWithBufferizedValues(rewriter, op, buffer);
    return success();
  }
};

} // namespace
} // namespace rock
} // namespace mlir

void mlir::rock::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, rock::RockDialect *dialect) {
    Conv2DOp::attachInterface<GemmLikeInterface<Conv2DOp>>(*ctx);
    Conv2DBwdDataOp::attachInterface<GemmLikeInterface<Conv2DBwdDataOp>>(*ctx);
    Conv2DBwdWeightOp::attachInterface<GemmLikeInterface<Conv2DBwdWeightOp>>(
        *ctx);
    GemmOp::attachInterface<GemmLikeInterface<GemmOp>>(*ctx);
    ReduceOp::attachInterface<GemmLikeInterface<ReduceOp>>(*ctx);

    // While these utility kernels aren't gemm wrappers, strictly, they still
    // bufferize like them
    ZeroInitKernelOp::attachInterface<GemmLikeInterface<ZeroInitKernelOp>>(
        *ctx);
    ConvertingCopyKernelOp::attachInterface<
        GemmLikeInterface<ConvertingCopyKernelOp>>(*ctx);

    TransformOp::attachInterface<TransformOpInterface>(*ctx);
    TensorUntransformCastOp::attachInterface<TensorUntransformCastOpInterface>(
        *ctx);
  });
}
