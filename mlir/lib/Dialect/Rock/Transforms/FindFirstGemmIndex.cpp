//===- FindFirstGemmIndex.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fusions can generate multiple linalg.generic ops and firstGemmIdx might not
// match between block inputs and linalg.generic op after linalg.generic ops
// have been fused. This pass traces the fused linalg.generic op back to the
// firstGemmIdx of the block inputs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFINDFIRSTGEMMINDEXPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-find-first-gemm-index"

using namespace mlir;

namespace {
struct RockFindFirstGemmIndexPass
    : public rock::impl::RockFindFirstGemmIndexPassBase<
          RockFindFirstGemmIndexPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static bool canTraceToFirstGemmArg(Value input, Value firstGemmArg) {
  // trace back to block arguments (through ViewLike ops)
  FailureOr<Value> maybeBlockArg = rock::findBlockArgument(input);
  if (failed(maybeBlockArg))
    return false;

  // If the input is a block argument, check if it matches the first gemm
  // argument.
  return maybeBlockArg.value() == firstGemmArg;
}

static LogicalResult
reassignFirstGemmIndex(func::FuncOp &func,
                       rock::RockGemmGemmWrapperInterface gemmGemmOp) {
  // Get the first gemm indices from the gemmGemmOp.
  ArrayRef<int64_t> firstGemmIndices = gemmGemmOp.getFirstGemmIndices();
  // initially firstGemmIndices should refer to blockArgument index for the
  // first gemm in preSecondGemmRegion and therefore it's size should be 1
  if (firstGemmIndices.size() != 1) {
    return gemmGemmOp.emitError(
        "Expected exactly one first gemm index, found: " +
        std::to_string(firstGemmIndices.size()));
  }

  // get linalg.generic ops in the preSecondGemmRegion
  SmallVector<linalg::GenericOp> genOpsList;
  gemmGemmOp.getPreSecondGemmRegion().walk(
      [&genOpsList](linalg::GenericOp genOp) { genOpsList.push_back(genOp); });

  // no fusion, nothing to do
  if (genOpsList.empty())
    return success();
  llvm::MapVector<linalg::GenericOp, int64_t> genOpToIndexMap;
  if (firstGemmIndices[0] >=
      gemmGemmOp.getPreSecondGemmRegion().getNumArguments()) {
    return gemmGemmOp.emitError(
        "First gemm index out of bounds for preSecondGemmRegion");
  }
  Value firstGemmOutArg =
      gemmGemmOp.getPreSecondGemmRegion().getArgument(firstGemmIndices[0]);

  BufferDependencyAnalysis analysis(func.getOperation());
  /*
  genOpsList is list of linalg.generic ops in the preSecondGemmRegion in the
  order they appear after the first gemm
  */
  for (auto genOp : genOpsList) {
    SmallVector<int64_t> newFirstGemmIndices;
    for (auto [index, input] : llvm::enumerate(genOp.getInputs())) {
      if (canTraceToFirstGemmArg(input, firstGemmOutArg)) {
        // If the input can be traced back to the first gemm argument, we found
        // the new index.
        newFirstGemmIndices.push_back(index);
      } else {
        // trace input to any of the previously visited linalg.generic ops
        SmallVector<rock::TransformOp> transformOps;
        Value untransformed;
        std::tie(untransformed, std::ignore) =
            rock::untransform(input, transformOps);
        if (memref::AllocOp allocOp =
                untransformed.getDefiningOp<memref::AllocOp>()) {
          std::optional<llvm::SmallVector<OpOperand *>> writers =
              analysis.getWriters(allocOp);
          if (writers == std::nullopt) {
            // If the allocOp has no writers, we cannot trace it back to a
            // previous linalg.generic op.
            continue;
          }
          // else
          // If the allocOp has writers, check if any of them is a
          // linalg.generic op that we have already visited.
          for (OpOperand *writer : writers.value()) {
            if (auto writerOp =
                    dyn_cast<linalg::GenericOp>(writer->getOwner())) {
              // If the writer is a linalg.generic op, check if it is in the
              // genOpsList and if it has been processed already.
              if (genOpToIndexMap.contains(writerOp))
                newFirstGemmIndices.push_back(index);
            }
          }
        }
      }
    }
    if (newFirstGemmIndices.empty()) {
      // If no inputs can be traced back to the first gemm argument, we cannot
      // reassign the index for this generic op.
      LLVM_DEBUG(llvm::dbgs() << genOp << "\n");
      return gemmGemmOp.emitError(
          "Cannot trace first gemm index for linalg.generic op");
    }
    if (newFirstGemmIndices.size() > 1) {
      // If multiple inputs can be traced back, we cannot determine a single
      // index.
      LLVM_DEBUG(llvm::dbgs() << genOp << "\n");
      return gemmGemmOp.emitError(
          "Multiple inputs trace back to first gemm argument");
    }
    genOpToIndexMap[genOp] = newFirstGemmIndices[0];
  }
  auto indicesRange = llvm::map_range(
      genOpToIndexMap, [](const auto &pair) { return pair.second; });
  SmallVector<int64_t> newfirstGemmIndices(indicesRange.begin(),
                                           indicesRange.end());
  // Set the new first gemm index in the gemmGemmOp.
  gemmGemmOp.setFirstGemmIndices(newfirstGemmIndices);
  return success();
}

void RockFindFirstGemmIndexPass::runOnOperation() {
  auto func = getOperation();
  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  // find gemm+gemm like operations with fusion
  SmallVector<rock::RockGemmGemmWrapperInterface> gemmGemmOps;
  func.walk([&gemmGemmOps](rock::RockGemmGemmWrapperInterface gemmGemmOp) {
    gemmGemmOps.push_back(gemmGemmOp);
  });

  // no gemm+gemm like operations found, nothing to do
  if (gemmGemmOps.empty())
    return;

  if (gemmGemmOps.size() != 1) {
    func.emitError(
        "More than one gemm+gemm like operation found, expected only one.");
    return signalPassFailure();
  }

  if (failed(reassignFirstGemmIndex(func, gemmGemmOps[0]))) {
    return signalPassFailure();
  }
}
