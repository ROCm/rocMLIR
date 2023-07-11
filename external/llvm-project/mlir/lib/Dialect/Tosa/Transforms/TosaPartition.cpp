//===- TosaPartition.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace conv2d followed by elementwise op with call to function containing
// them.  Generalised, outline any anchor op, all its trailing elementwise ops,
// and all its leading elementwise ops.  (Where "elementwise" itself is
// generalised to include transpose and reshape ops.)
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OutlinerUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <deque>
#include <iostream>

using llvm::SmallVector;

// TODO(kdrewnia): Make it so list options can have defaults, and then get rid
// of needing to set defaults here
namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAPARTITION
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

// Tosa ops can broadcast values along axes, which allows for
// element-wise operations without fully-matching dimensions.  The
// Elementwise trait is strict about matching dimensions, but
// broadcastable ops are also element-wise, and we know that an
// additional set of ops are also element-wise.
bool isElementwiseOp(Operation *op) {
  return op->hasTrait<OpTrait::Elementwise>() ||
         op->hasTrait<OpTrait::ResultsBroadcastableShape>() ||
         // clang-format off
    isa<tosa::CastOp,
        tosa::ClampOp,
        tosa::ErfOp,
        tosa::SigmoidOp,
        tosa::TanhOp,
// ResultsBroadcastableShape
//         tosa::AddOp,
//         tosa::ArithmeticRightShiftOp,
//         tosa::BitwiseAndOp,
//         tosa::BitwiseOrOp,
//         tosa::BitwiseXorOp,
//         tosa::DivOp,
//         tosa::LogicalAndOp,
//         tosa::LogicalLeftShiftOp,
//         tosa::LogicalRightShiftOp,
//         tosa::LogicalOrOp,
//         tosa::LogicalXorOp,
//         tosa::MaximumOp,
//         tosa::MinimumOp,
//         tosa::MulOp,
//         tosa::PowOp,
//         tosa::SubOp,
        tosa::AbsOp,
//         tosa::BitwiseNotOp,
        tosa::CeilOp,
        tosa::ClzOp,
        tosa::ExpOp,
        tosa::FloorOp,
        tosa::LogOp,
        tosa::LogicalNotOp,
        tosa::NegateOp,
        tosa::ReciprocalOp,
        tosa::RsqrtOp,
        tosa::SelectOp,
        tosa::EqualOp,
        tosa::GreaterOp,
        tosa::GreaterEqualOp
       >(op);
  // clang-format on
}

bool isFuseableOp(Operation *op) { return isElementwiseOp(op); }

bool isAnchorOp(Operation *op, Pass::ListOption<std::string> &anchorOps) {
  if (anchorOps.empty()) // ListOption doesn't have a default value.
    anchorOps = {"tosa.conv2d", "tosa.matmul", "tosa.depthwise_conv2d",
                 "tosa.fully_connected"};
  return llvm::is_contained(anchorOps, op->getName().getIdentifier().str());
}

bool isTransposeOp(Operation *op) {
  return isa<tosa::TransposeOp, tosa::ReshapeOp>(op);
}

bool isSliceOp(Operation *op) { return isa<tosa::SliceOp>(op); }

bool isConstSplatOp(Operation *op) {
  if (tosa::ConstOp constOp = dyn_cast<tosa::ConstOp>(op)) {
    if (constOp.getValue().isSplat()) {
      return true;
    }
  }
  return false;
}

bool isTransposeConfigConstant(Operation *op) {
  return op->hasTrait<OpTrait::ConstantLike>() &&
         llvm::any_of(op->getUsers(), [&](Operation *u) {
           return isa<TransposeOp>(u) && u->getOperand(1) == op->getResult(0);
         });
}

bool isAlwaysLeadingOp(Operation *op) {
  return isConstantZero(op) || isTransposeOp(op) || isSliceOp(op) ||
         isConstSplatOp(op) || isTransposeConfigConstant(op);
}

bool isLeadingOp(Operation *op, bool trailingOnly) {
  return isAlwaysLeadingOp(op) || (!trailingOnly && isFuseableOp(op));
}

bool isTrailingOp(Operation *op) {
  return isTransposeOp(op) || isFuseableOp(op);
}

class TosaPartitionPass
    : public tosa::impl::TosaPartitionBase<TosaPartitionPass> {
public:
  using tosa::impl::TosaPartitionBase<TosaPartitionPass>::TosaPartitionBase;

  void runOnOperation() override;
};

void TosaPartitionPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto anchorPred = [&](Operation *op) { return isAnchorOp(op, anchorOps); };
  auto leadingPred = [&](Operation *op) {
    return isLeadingOp(op, trailingOnly);
  };
  Outliner p(anchorPred, leadingPred, isTrailingOp, partitionTagOpt);
  p.outline(module);
}

} // namespace
