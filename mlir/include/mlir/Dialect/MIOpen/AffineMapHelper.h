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

using namespace mlir;

namespace mlir {
namespace miopen {

//===----------------------------------------------------------------------===//
// Utility function to compose affine maps.
//===----------------------------------------------------------------------===//
inline AffineMap composeTransforms(ArrayRef<AffineMap> affineMaps) {
  int64_t iter = affineMaps.size() - 1;
  AffineMap transform = affineMaps[iter];
  --iter;
  while (iter >= 0) {
    transform = transform.compose(affineMaps[iter]);
    --iter;
  }
  return transform;
}

inline AffineMap composeTransforms(ArrayAttr affineMaps) {
  int64_t iter = affineMaps.size() - 1;
  AffineMap transform =
      affineMaps[iter].template cast<AffineMapAttr>().getValue();
  --iter;
  while (iter >= 0) {
    transform = transform.compose(
        affineMaps[iter].template cast<AffineMapAttr>().getValue());
    --iter;
  }
  return transform;
}

//===----------------------------------------------------------------------===//
// Check if an AffineMap has division or remainder inside.
//===----------------------------------------------------------------------===//
inline bool hasDivisionOrRemainder(AffineMap map) {
  bool ret = false;
  if (!map)
    return false;
  map.walkExprs([&ret](AffineExpr expr) {
    if (expr.getKind() == AffineExprKind::Mod ||
        expr.getKind() == AffineExprKind::FloorDiv ||
        expr.getKind() == AffineExprKind::CeilDiv)
      ret = true;
  });

  // XXX. hack. always return false for now for performance reason.
  // May need more sophisticated checks to determine if we would truly go OOB.
  // return ret;
  return false;
}

//===----------------------------------------------------------------------===//
// Check if an AffineExpr has padding, which is represented as a minus
// expression with a constant operand.
//===----------------------------------------------------------------------===//
inline bool hasPadding(AffineExpr expr) {
  bool ret = false;
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
    AffineExpr tmp = binaryExpr.getLHS();

    tmp.walk([&ret](AffineExpr expr_sub) {
      if (expr_sub.getKind() == AffineExprKind::Constant) {
        auto constantSubExpr = expr_sub.template dyn_cast<AffineConstantExpr>();
        if (constantSubExpr.getValue() < 0)
          ret = true;
      }
    });

    if (ret) {
      return ret;
    }
    // RHS always has only one expression like :
    // d1 * 4  or  1024 , skip walk through it
    // just use hasMinusConstant
    ret |= hasMinusConstant(binaryExpr.getRHS());
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// Check if an AffineMap has padding, which is represented as a minus expression
// with a constant operand.
//===----------------------------------------------------------------------===//
inline bool hasPadding(AffineMap map) {
  bool ret = false;
  if (!map)
    return false;
  map.walkExprs([&ret](AffineExpr expr) { ret |= hasPadding(expr); });
  return ret;
}

} // end namespace miopen
} // end namespace mlir
#endif // MLIR_DIALECT_MIOPEN_AFFINEMAP_HELPER_H
