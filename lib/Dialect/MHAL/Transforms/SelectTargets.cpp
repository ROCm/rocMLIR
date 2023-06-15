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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/SystemDevice.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace mhal {
#define GEN_PASS_DEF_MHALSELECTTARGETSPASS
#include "mlir/Dialect/MHAL/Transforms/Passes.h.inc"
} // namespace mhal
} // namespace mlir

#define DEBUG_TYPE "mhal-select-targets"

using namespace mlir;
namespace {
struct MHALSelectTargetsPass
    : public mhal::impl::MHALSelectTargetsPassBase<MHALSelectTargetsPass> {
  using mhal::impl::MHALSelectTargetsPassBase<
      MHALSelectTargetsPass>::MHALSelectTargetsPassBase;

  bool testType(mhal::TargetType type) const {
    if (targetTypes.empty())
      return true;
    auto typeStr = mhal::getNameForTargetType(type);
    for (auto targetType : targetTypes) {
      if (targetType == typeStr)
        return true;
    }
    return false;
  }

  bool testArch(mhal::TargetType type, StringRef arch) const {
    if (!testType(type))
      return false;
    mhal::SystemDevice testDev{mhal::SystemDevice::Type::EGPU};
    if (failed(testDev.parse(arch)))
      return false;

    for (auto targetArch : targetArchs) {
      mhal::SystemDevice targetDev{mhal::SystemDevice::Type::EGPU};
      if (succeeded(targetDev.parse(targetArch))) {
        if (targetDev.isCompatible(testDev))
          return true;
      }
    }
    return false;
  }

  // Replaces synchronous call ops in the op's region with asynchronous ones and
  // inserts the necessary synchronization (as async.await ops). Assumes
  // sequential execution semantics and that no asynchronous ops yet.
  void runOnOperation() override {
    constexpr llvm::StringLiteral targetsTag = "mhal.targets";
    func::FuncOp func = getOperation();
    OpBuilder b(func);
    if (auto targets = func->getAttrOfType<ArrayAttr>(targetsTag)) {
      mhal::KernelPackageAttr targetKrn;
      for (auto targetAttr : targets.getValue()) {
        auto pkgAttr = targetAttr.cast<mhal::KernelPackageAttr>();
        auto type = pkgAttr.getType();
        auto arch = pkgAttr.getTarget();
        if (testArch(type, arch)) {
          // TODO(sjw): test perf
          targetKrn = pkgAttr;
        }
      }
      if (targetKrn)
        func->setAttr(targetsTag, b.getArrayAttr(targetKrn));
      else {
        if (targetArchs.size() && !testType(mhal::TargetType::CPU)) {
          func.emitError("target object not found");
          signalPassFailure();
        }
        func->removeAttr(targetsTag);
      }
    }
  }
};
} // namespace
