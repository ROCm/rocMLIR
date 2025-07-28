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
  // Get the first gemm index from the gemmGemmOp.
  uint32_t firstGemmIndex = gemmGemmOp.getFirstGemmIndex();

  // get linalg.generic ops in the preSecondGemmRegion
  SmallVector<linalg::GenericOp> genOps;
  gemmGemmOp.getPreSecondGemmRegion().walk(
      [&genOps](linalg::GenericOp genOp) { genOps.push_back(genOp); });

  // no fusion, nothing to do
  if (genOps.empty())
    return success();

  if (genOps.size() != 1)
    return gemmGemmOp.emitError(
        "More than one linalg.generic operation found, expected only one.");

  linalg::GenericOp genOp = genOps[0];
  assert(firstGemmIndex <
         gemmGemmOp.getPreSecondGemmRegion().getNumArguments());
  Value firstGemmArg =
      gemmGemmOp.getPreSecondGemmRegion().getArgument(firstGemmIndex);

  // try to trace args of linalg.generic op back to the first gemm argument
  int64_t newFirstGemmIndex = -1;
  for (auto [index, input] : llvm::enumerate(genOp.getInputs())) {
    if (canTraceToFirstGemmArg(input, firstGemmArg)) {
      // If the input can be traced back to the first gemm argument, we found
      // the new index.
      newFirstGemmIndex = index;
      break;
    }
  }
  if (newFirstGemmIndex == -1) {
    return gemmGemmOp.emitError(
        "Could not find a matching input for the first gemm index.");
  }

  // Set the new first gemm index in the gemmGemmOp.
  gemmGemmOp.setFirstGemmIndex(static_cast<uint32_t>(newFirstGemmIndex));
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
