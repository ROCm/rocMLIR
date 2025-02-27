//===- OutlinerUtils.h ------------------------------------------===//
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
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <deque>
#include <iostream>

using llvm::SmallVector;

namespace mlir {

bool isConstantZero(Operation *op);

////////////////////////////////////////////////////////////////////////////////

class Outliner {
public:
  Outliner(function_ref<bool(Operation *)> _isAnchorOp,
           function_ref<bool(Operation *)> _isLeadingOp,
           function_ref<bool(Operation *)> _isTrailingOp,
           function_ref<bool(Operation *)> _isTerminatingOp,
           StringRef _outlineTag)
      : isAnchorOp(_isAnchorOp), isLeadingOp(_isLeadingOp),
        isTrailingOp(_isTrailingOp), isTerminatingOp(_isTerminatingOp),
        outlineTag(_outlineTag) {}

  void outline(ModuleOp module);

  function_ref<bool(Operation *)> isAnchorOp;
  function_ref<bool(Operation *)> isLeadingOp;
  function_ref<bool(Operation *)> isTrailingOp;
  function_ref<bool(Operation *)> isTerminatingOp;
  StringRef outlineTag;
};

} // namespace mlir
