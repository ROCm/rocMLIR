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
  SymbolRefAttr sym = dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
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
    mhal::LaunchOp launchOp = cast<mhal::LaunchOp>(op);
    auto opOperandIdx =
        opOperand.getOperandNumber() - launchOp.getDependencies().size();
    mlir::CallOpInterface callOp(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is read.
      return true;

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    return funcState.readBbArgs.lookup(funcOp).contains(opOperandIdx);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // operands are always inputs
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // results never alias with operands
    AliasingValueList result;
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
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    // Result types of the bufferized CallOp.
    SmallVector<Type> resultTypes;

    // Operands of the bufferized CallOp.
    SmallVector<Value> newOperands(numOperands, Value());

    // 1. Compute the result types of the new CallOp.
    unsigned funcResultIdx = 0;
    for (const auto &it : llvm::enumerate(callOp->getResults())) {
      auto returnVal = it.value();
      Type returnType = returnVal.getType();
      if (isa<TensorType>(returnType)) {
        assert(returnType == callResultTypes[funcResultIdx++]);
        FailureOr<BaseMemRefType> memrefType =
            bufferization::getBufferType(returnVal, options);
        if (failed(memrefType))
          return failure();
        resultTypes.push_back(*memrefType);
      } else {
        // Non-tensor values are returned.
        resultTypes.push_back(returnType);
        if (returnType == callResultTypes[funcResultIdx])
          funcResultIdx++;
      }
    }

    // 2. Rewrite tensor operands as memrefs based on `bufferizedFuncType`.
    unsigned funcOperandIdx = 0;
    for (OpOperand &opOperand : callOp->getOpOperands()) {
      unsigned idx = opOperand.getOperandNumber();
      Value tensorOperand = opOperand.get();
      // Non-tensor operands are just copied.
      if (isa<mhal::TokenType>(tensorOperand.getType())) {
        newOperands[idx] = tensorOperand;
        continue;
      }
      if (!isa<TensorType>(tensorOperand.getType())) {
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
      newOperands[idx] = buffer;
      funcOperandIdx++;
    }

    // 3. Create the new CallOp.
    Operation *newCallOp =
        callOp.clone(rewriter, callOp.getLoc(), resultTypes, newOperands);

    // 4. Replace the old op with the new op.
    replaceOpWithBufferizedValues(rewriter, callOp, newCallOp->getResults());

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
