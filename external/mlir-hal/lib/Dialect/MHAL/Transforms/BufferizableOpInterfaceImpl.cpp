//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MHAL/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::bufferization::func_ext;
using namespace mlir::mhal;
using namespace mlir::func;

namespace mlir {
namespace mhal {
namespace {

/// Return the index of the bbArg in the given FuncOp that is equivalent to the
/// specified return value (if any).
static std::optional<int64_t>
getEquivalentFuncArgIdx(FuncOp funcOp, const FuncAnalysisState &state,
                        int64_t returnValIdx) {
  auto funcOpIt = state.equivalentFuncArgs.find(funcOp);
  if (funcOpIt == state.equivalentFuncArgs.end())
    // No equivalence info stores for funcOp.
    return std::nullopt;

  auto retValIt = funcOpIt->getSecond().find(returnValIdx);
  if (retValIt == funcOpIt->getSecond().end())
    // Return value has no equivalent bbArg.
    return std::nullopt;

  return retValIt->getSecond();
}

/// Return the FuncOp called by `callOp`.
static FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Get FuncAnalysisState.
static const FuncAnalysisState &
getFuncAnalysisState(const AnalysisState &state) {
  assert(isa<OneShotAnalysisState>(state) && "expected OneShotAnalysisState");
  auto *result = static_cast<const OneShotAnalysisState &>(state)
                     .getExtension<FuncAnalysisState>();
  assert(result && "FuncAnalysisState does not exist");
  return *result;
}

/// Return the state (phase) of analysis of the FuncOp.
static FuncOpAnalysisState getFuncOpAnalysisState(const AnalysisState &state,
                                                  FuncOp funcOp) {
  if (!isa<OneShotAnalysisState>(state))
    return FuncOpAnalysisState::NotAnalyzed;
  auto *funcState = static_cast<const OneShotAnalysisState &>(state)
                        .getExtension<FuncAnalysisState>();
  if (!funcState)
    return FuncOpAnalysisState::NotAnalyzed;
  const auto &analyzedFuncOps = funcState->analyzedFuncOps;
  auto it = analyzedFuncOps.find(funcOp);
  if (it == analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}

/// Bufferization of mhal.launch.
struct LaunchOpInterface
    : public BufferizableOpInterface::ExternalModel<LaunchOpInterface,
                                                    mhal::LaunchOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    mlir::CallOpInterface callOp(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is read.
      return true;

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    return funcState.readBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    mlir::CallOpInterface callOp(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is written.
      return true;

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    return funcState.writtenBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    mlir::CallOpInterface callOp(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    if (getFuncOpAnalysisState(state, funcOp) !=
        FuncOpAnalysisState::Analyzed) {
      // FuncOp not analyzed yet. Any OpResult may be aliasing.
      return bufferization::detail::unknownGetAliasingOpResults(opOperand);
    }

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    auto aliasingReturnVals =
        funcState.aliasingReturnVals.lookup(funcOp).lookup(
            opOperand.getOperandNumber());

    // Check if the aliasing OpResult is equivalent to the OpOperand.
    std::optional<int64_t> equivalent = {};
    if (aliasingReturnVals.size() == 1) {
      equivalent = getEquivalentFuncArgIdx(funcOp, funcState,
                                           aliasingReturnVals.front());
      assert((!equivalent.has_value() ||
              *equivalent == opOperand.getOperandNumber()) &&
             "inconsistent analysis state");
    }
    AliasingOpResultList result;
    for (int64_t resultIdx : aliasingReturnVals)
      result.addAlias({callOp->getOpResult(resultIdx),
                       equivalent.has_value() ? BufferRelation::Equivalent
                                              : BufferRelation::Unknown,
                       /*isDefinite=*/equivalent.has_value()});
    return result;
  }

  /// All function arguments are writable. It is the responsibility of the
  /// CallOp to insert buffer copies where necessary.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    mlir::CallOpInterface callOp(op);
    auto callOperands = callOp.getArgOperands();
    auto callResultTypes = callOp.getCallResultTypes();
    unsigned numOperands = callOp->getNumOperands();
    unsigned numResults = callOp->getNumResults();
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    FunctionType funcType = funcOp.getFunctionType();

    // Result types of the bufferized CallOp.
    SmallVector<Type> resultTypes;
    // Replacement values for the existing CallOp. These are usually the results
    // of the bufferized CallOp, unless a tensor result folds onto an operand.
    SmallVector<Value> replacementValues(numResults, Value());
    // For non-tensor results: A mapping from return val indices of the old
    // CallOp to return val indices of the bufferized CallOp.
    SmallVector<std::optional<unsigned>> retValMapping(numResults,
                                                       std::nullopt);
    // Operands of the bufferized CallOp.
    SmallVector<Value> newOperands(numOperands, Value());

    // 1. Compute the result types of the new CallOp.
    unsigned funcResultIdx = 0;
    for (const auto &it : llvm::enumerate(callOp->getResultTypes())) {
      unsigned returnValIdx = it.index();
      Type returnType = it.value();
      if (!returnType.isa<TensorType>()) {
        // Non-tensor values are returned.
        retValMapping[returnValIdx] = resultTypes.size();
        resultTypes.push_back(returnType);
        if (returnType == callResultTypes[funcResultIdx])
          funcResultIdx++;
        continue;
      }
      assert(returnType == callResultTypes[funcResultIdx]);

      retValMapping[returnValIdx] = resultTypes.size();
      resultTypes.push_back(funcType.getResult(funcResultIdx));
      funcResultIdx++;
    }

    // 2. Rewrite tensor operands as memrefs based on `bufferizedFuncType`.
    unsigned funcOperandIdx = 0;
    for (OpOperand &opOperand : callOp->getOpOperands()) {
      unsigned idx = opOperand.getOperandNumber();
      Value tensorOperand = opOperand.get();
      // Non-tensor operands are just copied.
      if (tensorOperand.getType().isa<mhal::TokenType>()) {
        newOperands[idx] = tensorOperand;
        continue;
      }
      if (!tensorOperand.getType().isa<TensorType>()) {
        newOperands[idx] = tensorOperand;
        if (tensorOperand == callOperands[funcOperandIdx])
          funcOperandIdx++;
        continue;
      }

      // Retrieve buffers for tensor operands.
      Value buffer = newOperands[idx];
      if (!buffer) {
        FailureOr<Value> maybeBuffer =
            getBuffer(rewriter, opOperand.get(), options);
        if (failed(maybeBuffer))
          return failure();
        buffer = *maybeBuffer;
      }

      // Caller / callee type mismatch is handled with a CastOp.
      // auto memRefType = funcType.getInput(idx);
      auto memRefType = funcType.getInput(funcOperandIdx);
      // Since we don't yet have a clear layout story, to_memref may
      // conservatively turn tensors into more dynamic memref than necessary.
      // If the memref type of the callee fails, introduce an extra memref.cast
      // that will either canonicalize away or fail compilation until we can do
      // something better.
      if (buffer.getType() != memRefType) {
        assert(
            memref::CastOp::areCastCompatible(buffer.getType(), memRefType) &&
            "CallOp::bufferize: cast incompatible");
        Value castBuffer = rewriter.create<memref::CastOp>(callOp.getLoc(),
                                                           memRefType, buffer);
        buffer = castBuffer;
      }
      newOperands[idx] = buffer;
      funcOperandIdx++;
    }

    // 3. Create the new CallOp.
    Operation *newCallOp =
        callOp.clone(rewriter, callOp.getLoc(), resultTypes, newOperands);

    // Get replacement values.
    for (unsigned i = 0; i < replacementValues.size(); ++i) {
      if (replacementValues[i])
        continue;
      replacementValues[i] = newCallOp->getResult(*retValMapping[i]);
    }

    // 4. Replace the old op with the new op.
    replaceOpWithBufferizedValues(rewriter, callOp, replacementValues);

    return success();
  }
};

} // namespace
} // namespace mhal
} // namespace mlir

void mlir::mhal::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, mhal::MHALDialect *dialect) {
    mhal::LaunchOp::attachInterface<LaunchOpInterface>(*ctx);
  });
}
