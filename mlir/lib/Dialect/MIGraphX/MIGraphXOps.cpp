//===- MIGraphXOps.cpp - MIGraphX MLIR Operations
//-----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"

#include "mlir/Dialect/MIGraphX/MIGraphXOpsDialect.cpp.inc"
#include "mlir/Dialect/MIGraphX/MIGraphXTypes.cpp.inc"

using namespace mlir;
// using namespace mlir::migraphx;
using namespace migraphx;

//===----------------------------------------------------------------------===//
// MIGraphXDialect
//===----------------------------------------------------------------------===//

void MIGraphXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MIGraphX/MIGraphXOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/MIGraphX/MIGraphXOps.cpp.inc"

OpFoldResult RecipOp::fold(FoldAdaptor operands) {
  // 1/(1/x) = x
  if (auto parentRecip = getInA().getDefiningOp<RecipOp>()) {
    return parentRecip.getInA();
  }
  return {};
}
