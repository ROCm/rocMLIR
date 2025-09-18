//===----------------------- RemoveOutputAlloc.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eliminates redundant temporary output buffers produced by backward data
// convolution ops. Here is the general algorithm that this follows:
//   - The pass looks for a memref.alloc that captures the convolution output,
//     followed by a chain of Rock transform ops whose final
//     value is copied into a function argument.
//   - It reconstructs the inverse transform sequence directly on the function
//     argument
//   - Creates a new transform op targeting that argument, rewrites uses to the
//     new value, and erases the original alloc and memref.copy.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
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
#define GEN_PASS_DEF_ROCKREMOVEOUTPUTALLOCPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-remove-output-alloc"

using namespace mlir;

namespace {
struct RockRemoveOutputAllocPass
    : public rock::impl::RockRemoveOutputAllocPassBase<
          RockRemoveOutputAllocPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

FailureOr<memref::CopyOp> findCopyOpToFuncArg(OpBuilder b, Value allocOp) {
  // Traverse the users of the memref::Alloc to find the memref::Copy
  // that is used to write to a function argument
  DenseSet<Value> visited;
  SmallVector<Value> worklist;
  worklist.push_back(allocOp);

  while (!worklist.empty()) {
    Value curVal = worklist.pop_back_val();
    if (visited.contains(curVal))
      continue;
    
    visited.insert(curVal);

    for (auto *user : curVal.getUsers()) {
      if (auto memrefCpy = dyn_cast<memref::CopyOp>(user)) {
        // If the destination of this copy is a function argument, then
        // we have found our candidate for removal
        auto untransformTuple = rock::untransform(b, memrefCpy.getTarget());
        auto &rawVal = std::get<0>(untransformTuple);
        if (isa<BlockArgument>(rawVal))
          return memrefCpy;
      } else if (auto transformOp = dyn_cast<rock::TransformOp>(user)) {
        // If we have already visited this value, then skip it
        Value res = transformOp.getResult();
        worklist.push_back(res);
      }
    }
  }

  return failure();
}

void RockRemoveOutputAllocPass::runOnOperation() {
  auto func = getOperation();
  SmallVector<Operation *> opsToErase;

  // For the time being, we only want to perform this workaround for
  // memref.allocs that contain the output result of a backwards data
  // convolution op
  func.walk([&](rock::ConvBwdDataOp bwdData) {
    // Find the alloc that this op writes to
    // Note: The input of the bwdData op is the tensor that gets written to,
    // not the output.
    OpBuilder b{bwdData};
    auto untransformTuple = rock::untransform(b, bwdData.getInput());
    auto &allocOp = std::get<0>(untransformTuple);
    b.setInsertionPointAfter(allocOp.getDefiningOp());

    // If allocOp is not a memref::Alloc, then we can exit early
    if (!isa<memref::AllocOp>(allocOp.getDefiningOp()))
      return;

    auto copyOp = findCopyOpToFuncArg(b, allocOp);
    if (failed(copyOp))
      return;

    auto copyUntransformTuple = rock::untransform(b, copyOp->getSource());
    ArrayAttr views = std::get<1>(copyUntransformTuple);
    auto result = rock::invertTransforms(b, copyOp->getSource().getLoc(), views);

    // There are some transforms that are not invertible. If we hit this case,
    // then there is nothing further we can do here.
    if (!result)
      return;

    // Create a new rock::Transform op that applies the inverse transforms
    // to the output arg of the bwdData op
    auto newTransformOp =
        rock::transform(b, copyOp->getTarget(), result);
    allocOp.replaceAllUsesWith(newTransformOp);

    // We are safe to add the allocOp to the list of ops to delete since
    // we disable fusions for bwd_data convs and therefore this will be the
    // only op that uses this.
    opsToErase.push_back(*copyOp);
    opsToErase.push_back(allocOp.getDefiningOp());
  });

  // Iterate over the ops to erase and remove them from the IR
  for (auto &op : opsToErase)
    op->erase();
}
