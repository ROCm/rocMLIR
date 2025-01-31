//===- fusionUtils.cpp - Rock utility for fusion -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::rock;
using namespace arith;

bool validOperationGemmOut(Operation &op) {
  return isa<MulFOp, DivFOp, AddFOp, SubFOp, SIToFPOp, UIToFPOp, NegFOp,
             ExtUIOp, ExtSIOp, ExtFOp, TruncFOp, TruncIOp>(op);
}

static LogicalResult validOutputAtomicAdd(Type outType, GemmFeatures features) {
  // Split-K currently supports only f32/f16/bf16 element types
  if (!isa<Float32Type, Float16Type, BFloat16Type>(outType))
    return failure();

  if (isa<Float32Type>(outType) &&
      !bitEnumContainsAll(features, GemmFeatures::atomic_add))
    return failure();

  if (isa<Float16Type>(outType) &&
      !bitEnumContainsAll(features, GemmFeatures::atomic_add_f16))
    return failure();

  if (isa<BFloat16Type>(outType) &&
      !bitEnumContainsAll(features, GemmFeatures::atomic_add_bf16))
    return failure();
    
  return success();
}

LogicalResult mlir::rock::checkValidOutputFusion(
    linalg::GenericOp genericOp, Value gemmResult, GemmFeatures features,
    SmallVector<std::tuple<Operation *, int>> &adds) {
  /* We can only fuse:
  - add/sub gemmResult, otherTensor (which will be converted to add gemmResult,
  otherTensor/splitKFactor)
  - add/sub gemmResult, gemmResult
  - mul/div gemmResult, otherTensor
  - neg
  - type conversion functions
  Where gemmResult != otherTensor for all cases.
  */
  auto outputs = genericOp.getOutputs();
  assert(outputs.size() == 1);

  // find tensor index
  int tensorIndex = -1;
  for (int i = 0; i < static_cast<int>(genericOp->getNumOperands()); ++i) {
    auto genericOpInputAlloc = findMemrefAlloc(genericOp->getOperand(i));
    if (llvm::succeeded(genericOpInputAlloc) &&
        genericOpInputAlloc->getMemref() == gemmResult)
      tensorIndex = i;
  }
  if (tensorIndex == -1)
    return failure();

  llvm::DenseSet<Value> derivedGemmResult;
  Block &body = genericOp.getRegion().front();
  derivedGemmResult.insert(body.getArgument(tensorIndex));

  for (Operation &nestedOp : body.without_terminator()) {
    // check if any operand is derived from the GEMM result
    int numGemmResults = 0;
    for (Value operand : nestedOp.getOperands()) {
      if (derivedGemmResult.contains(operand))
        ++numGemmResults;
    }
    if (numGemmResults > 0) {
      // check it's a valid operation
      if (!validOperationGemmOut(nestedOp)) {
        return failure();
      }

      if (isa<MulFOp, DivFOp>(nestedOp) && numGemmResults > 1) {
        // gemmOut^2 is not allowed
        return failure();
      }

      // save add and sub ops to modify them: divide by splitKFactor
      // if both operands come from gemmOut, no need to modify anything
      if (isa<AddFOp, SubFOp>(nestedOp) && numGemmResults == 1) {
        int index = derivedGemmResult.contains(nestedOp.getOperand(0)) ? 0 : 1;
        adds.push_back(std::make_tuple(&nestedOp, index));
      }

      // add the op results to our tracked set since they're derived from the
      // GEMM result
      for (Value result : nestedOp.getResults())
        derivedGemmResult.insert(result);
    }
  }
  return success();
}

LogicalResult mlir::rock::testFusionLegalitySplitK(func::FuncOp func) {
  auto analysis = BufferDependencyAnalysis(func.getOperation());
  const auto &readersTable = analysis.getReadersTable();
  const auto &writersTable = analysis.getWritersTable();

  // can't fuse reduce_max with split-k
  WalkResult reduceMaxRes = func.walk([](ReduceOp reduceOp) -> WalkResult {
    if (reduceOp.getReduceMethod() == ReduceMethod::Max)
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (reduceMaxRes.wasInterrupted())
    return failure();

  WalkResult walkResult =
      func.walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
        auto gemmResult = gemmOp.getOutArgument()->get();

        auto maybeBlockArgs = traceGemmOutputToArgs(gemmResult, func, analysis);
        if (failed(maybeBlockArgs))
          return WalkResult::interrupt();

        // Verify hardware compatibility (split-k) for kernel output.
        // Checks if atomic_add operations are supported by the target hardware.
        auto blockArgs = maybeBlockArgs.value();
        for (auto blockArg : blockArgs) {
          auto outElementType =
              cast<ShapedType>(blockArg.getType()).getElementType();
          if (failed(validOutputAtomicAdd(outElementType,
                                          gemmOp.getGemmFeatures())))
            return WalkResult::interrupt();
        }

        auto maybeAlloc = findMemrefAlloc(gemmResult);
        if (failed(maybeAlloc)) {
          return WalkResult::advance();
        }

        // save all `linalg::GenericOp` that read from a gemm output
        SmallVector<linalg::GenericOp> genericOps;
        if (readersTable.contains(*maybeAlloc)) {
          for (OpOperand *op : readersTable.at(*maybeAlloc)) {
            if (isa<linalg::GenericOp>(op->getOwner())) {
              genericOps.push_back(
                  llvm::dyn_cast<linalg::GenericOp>(op->getOwner()));
            }
          }
        }

        // make sure that no `linalg::GenericOp` writes to a gemm output
        if (writersTable.contains(maybeAlloc.value())) {
          for (OpOperand *op : writersTable.at(*maybeAlloc)) {
            if (isa<linalg::GenericOp>(op->getOwner())) {
              return WalkResult::interrupt();
            }
          }
        }

        // check if generic ops are valid fusions
        for (auto genericOp : genericOps) {
          SmallVector<std::tuple<Operation *, int>> adds;
          if (failed(checkValidOutputFusion(genericOp, maybeAlloc.value(),
                                            gemmOp.getGemmFeatures(), adds)))
            return WalkResult::interrupt();
        }

        return WalkResult::advance();
      });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalitySplitK(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  assert(std::distance(funcs.begin(), funcs.end()) &&
         "expected ModuleOp containing a single func::FuncOp");
  func::FuncOp func = *(funcs.begin());
  return testFusionLegalitySplitK(func);
}

LogicalResult mlir::rock::testFusionLegalityReduce(func::FuncOp func) {
  WalkResult walkResult = func.walk([&](rock::ReduceOp reduceOp) -> WalkResult {
    auto outElemType = reduceOp.getOut().getType().getElementType();
    if (reduceOp.getReduceMethod() == ReduceMethod::Max) {
      if (!isa<Float32Type>(outElemType))
        return WalkResult::interrupt();

      if (!bitEnumContainsAll(reduceOp.getFeatures(),
                              GemmFeatures::atomic_fmax_f32))
        return WalkResult::interrupt();
    } else {
      if (failed(validOutputAtomicAdd(outElemType, reduceOp.getFeatures())))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalityReduce(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  assert(std::distance(funcs.begin(), funcs.end()) &&
         "expected ModuleOp containing a single func::FuncOp");
  func::FuncOp func = *(funcs.begin());
  return testFusionLegalityReduce(func);
}
