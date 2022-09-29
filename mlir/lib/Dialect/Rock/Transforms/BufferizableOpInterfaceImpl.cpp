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

/// Bufferization of rock.tensor_conv2d, which rewrites to a rock.conv2d
/// that operates on memrefs.
struct TensorConv2DOpInterface
    : public BufferizableOpInterface::ExternalModel<TensorConv2DOpInterface,
                                                    rock::TensorConv2DOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto cop = mlir::cast<rock::TensorConv2DOp>(op);
    Value operand = opOperand.get();
    return (operand == cop.getFilter() || operand == cop.getInput());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto cop = mlir::cast<rock::TensorConv2DOp>(op);
    Value operand = opOperand.get();
    return (operand == cop.getOutput());
  }

  // The buffer corresponding to the destination must equal the buffer
  // corresponding to the returned tensor
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    auto cop = mlir::cast<rock::TensorConv2DOp>(op);
    Value operand = opOperand.get();
    return (operand == cop.getOutput());
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto cop = mlir::cast<rock::TensorConv2DOp>(op);
    Value operand = opOperand.get();
    if (operand == cop.getOutput())
      return op->getOpResults();
    return {};
  }

  // The output argument is equal to the returned value
  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto cop = mlir::cast<rock::TensorConv2DOp>(op);
    FailureOr<Value> filter = getBuffer(rewriter, cop.getFilter(), options);
    if (failed(filter))
      return failure();
    FailureOr<Value> input = getBuffer(rewriter, cop.getInput(), options);
    if (failed(input))
      return failure();
    FailureOr<Value> output = getBuffer(rewriter, cop.getOutput(), options);
    if (failed(output))
      return failure();

    SmallVector<Value> args = {*filter, *input, *output};
    rewriter.create<rock::Conv2DOp>(op->getLoc(), TypeRange{}, args,
                                    op->getAttrs());
    replaceOpWithBufferizedValues(rewriter, op, {*output});
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

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return op->getOpResults();
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

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto castOp = mlir::cast<rock::TensorUntransformCastOp>(op);
    Value operand = opOperand.get();
    if (operand == castOp.getTransformedResult())
      return op->getOpResults();
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
    TensorConv2DOp::attachInterface<TensorConv2DOpInterface>(*ctx);
    TransformOp::attachInterface<TransformOpInterface>(*ctx);
    TensorUntransformCastOp::attachInterface<TensorUntransformCastOpInterface>(
        *ctx);
  });
}
