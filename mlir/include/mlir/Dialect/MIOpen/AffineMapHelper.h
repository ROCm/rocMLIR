//===- AffineMapHelper.h - Utility routines for AffineMap ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provide utility routines to check AffineMap instances.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_AFFINEMAP_HELPER_H
#define MLIR_DIALECT_MIOPEN_AFFINEMAP_HELPER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace miopen {

//===----------------------------------------------------------------------===//
// Check if an AffineMap has division or remainder inside.
//===----------------------------------------------------------------------===//
inline bool hasDivisionOrRemainder(AffineMap map) {
  bool ret = false;
  if (!map) return false;
  map.walkExprs([&ret](AffineExpr expr) {
    if (expr.getKind() == AffineExprKind::Mod ||
        expr.getKind() == AffineExprKind::FloorDiv ||
	expr.getKind() == AffineExprKind::CeilDiv)
      ret = true;
  });

  // XXX. hack. always return false for now for performance reason.
  // May need more sophisticated checks to determine if we would truly go OOB.
  //return ret;
  return false;
}

//===----------------------------------------------------------------------===//
// Check if an AffineMap has padding, which is represented as a minus expression
// with a constant operand.
//===----------------------------------------------------------------------===//
inline bool hasPadding(AffineMap map) {
  bool ret = false;
  if (!map) return false;
  map.walkExprs([&ret](AffineExpr expr) {
    auto hasMinusConstant = [](AffineExpr expr) -> bool {
      if (expr.getKind() == AffineExprKind::Constant) {
        auto constantExpr = expr.template dyn_cast<AffineConstantExpr>();
        if (constantExpr.getValue() < 0)
          return true;
      }
      return false;
    };
    auto binaryExpr = expr.template dyn_cast<AffineBinaryOpExpr>();
    if (binaryExpr) {
      ret |= hasMinusConstant(binaryExpr.getLHS());
      if (ret)
        return;
      ret |= hasMinusConstant(binaryExpr.getRHS());
    }
  });
  return ret;
}

} // end namespace miopen
} // end namespace mlir
#endif // MLIR_DIALECT_MIOPEN_AFFINEMAP_HELPER_H
