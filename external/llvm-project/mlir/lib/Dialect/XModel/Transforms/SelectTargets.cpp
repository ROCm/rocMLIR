//===- SelectTargets.cpp - Select IP target for execution -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the async.launch pattern rewriter that converts kernel
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/XModel/IR/XModel.h"
#include "mlir/Dialect/XModel/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace xmodel {
#define GEN_PASS_DEF_XMODELSELECTTARGETSPASS
#include "mlir/Dialect/XModel/Transforms/Passes.h.inc"
} // namespace xmodel
} // namespace mlir

#define DEBUG_TYPE "xmodel-select-targets"

using namespace mlir;
namespace {
struct XModelSelectTargetsPass
    : public xmodel::impl::XModelSelectTargetsPassBase<
          XModelSelectTargetsPass> {
  using xmodel::impl::XModelSelectTargetsPassBase<
      XModelSelectTargetsPass>::XModelSelectTargetsPassBase;

  bool testType(xmodel::TargetType type) const {
    auto typeStr = xmodel::getNameForTargetType(type);
    for (auto targetType : targetTypes) {
      if (targetType == typeStr)
        return true;
    }
    return false;
  }

  bool testChip(StringRef chip) const {
    for (auto targetChip : targetChips) {
      if (targetChip == chip)
        return true;
    }
    return false;
  }

  // Replaces synchronous call ops in the op's region with asynchronous ones and
  // inserts the necessary synchronization (as async.await ops). Assumes
  // sequential execution semantics and that no asynchronous ops yet.
  void runOnOperation() override {
    const char *targets_tag = "xmodel.targets";
    func::FuncOp func = getOperation();
    OpBuilder b(func);
    if (auto targets = func->getAttrOfType<ArrayAttr>(targets_tag)) {
      xmodel::KernelPackageAttr targetKrn;
      for (auto targetAttr : targets.getValue()) {
        auto pkgAttr = targetAttr.cast<xmodel::KernelPackageAttr>();
        auto type = pkgAttr.getType();
        auto chip = pkgAttr.getTarget();
        if (testType(type) && testChip(chip)) {
          // TODO(sjw): test perf
          targetKrn = pkgAttr;
        }
      }
      if (targetKrn)
        func->setAttr(targets_tag, b.getArrayAttr(targetKrn));
      else {
        func->removeAttr(targets_tag);
      }
    }
  }
};
} // namespace
